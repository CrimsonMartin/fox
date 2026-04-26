use std::sync::Arc;

use anyhow::Result;

use crate::scheduler::StopReason;

use super::model::{
    InferenceRequestForModel, Logits, VisionDecodeParams, VisionPrefillParams,
    VisionPreprocessParams,
};
use super::InferenceEngine;

impl InferenceEngine {
    /// Main inference loop.
    ///
    /// Vision requests: CLIP preprocessing runs in parallel via the mtmd pool,
    /// then all preprocessed results are decoded into KV in a single batch.
    /// Text prefill and token decode are batched as usual.
    pub async fn run_loop(self: Arc<Self>) -> Result<()> {
        let engine = self.clone();
        let mut last_prefix_hits: u64 = 0;
        let mut last_prefix_misses: u64 = 0;

        loop {
            let batch = engine.scheduler.schedule_step();

            // Refresh gauges and propagate counter deltas every scheduling step.
            if let Some(m) = &engine.metrics {
                use std::sync::atomic::Ordering;
                m.kv_cache_usage_ratio
                    .set(engine.kv_cache.memory_usage() as f64);
                m.queue_depth.set(engine.scheduler.queue_depth() as i64);
                m.active_requests
                    .set(engine.scheduler.active_requests() as i64);

                let cur_hits = engine.scheduler.prefix_hits.load(Ordering::Relaxed);
                let cur_misses = engine.scheduler.prefix_misses.load(Ordering::Relaxed);
                let dh = cur_hits.saturating_sub(last_prefix_hits);
                let dm = cur_misses.saturating_sub(last_prefix_misses);
                if dh > 0 {
                    m.prefix_cache_hits_total.inc_by(dh);
                }
                if dm > 0 {
                    m.prefix_cache_misses_total.inc_by(dm);
                }
                last_prefix_hits = cur_hits;
                last_prefix_misses = cur_misses;
            }

            for seq_id in &batch.preempted_seq_ids {
                engine.model.clear_sequence(*seq_id);
            }

            if batch.is_empty() {
                engine.scheduler.wait_for_work().await;
                continue;
            }

            // Partition prefill into vision and text-only requests.
            let mut vision_params: Vec<(u64, i32, Option<i32>, VisionPrefillParams)> = Vec::new();
            let mut text_prefill_ids = Vec::new();
            if !batch.prefill.is_empty() {
                let prefill_requests = engine.scheduler.get_running(&batch.prefill);
                for req in &prefill_requests {
                    if let (Some(vision_image), Some(vision_prompt)) =
                        (&req.vision_image, &req.vision_prompt)
                    {
                        vision_params.push((
                            req.id,
                            req.kv_seq_id,
                            req.prefix_seq_id,
                            VisionPrefillParams {
                                seq_id: req.kv_seq_id,
                                text_prompt: vision_prompt.clone(),
                                image_bytes: Arc::clone(vision_image),
                                temperature: req.sampling.temperature,
                                top_p: req.sampling.top_p,
                                top_k: req.sampling.top_k,
                                repetition_penalty: req.sampling.repetition_penalty,
                                seed: req.sampling.seed,
                            },
                        ));
                    } else {
                        text_prefill_ids.push(req.id);
                    }
                }
            }

            let mut decode_ids = batch.decode.clone();

            // 1. Text prefill (batched, one mutex acquisition).
            //    Chunked prefill: only completed requests get logits/sampling;
            //    partial requests stay in Prefilling state for the next iteration.
            if !text_prefill_ids.is_empty() {
                match engine.run_prefill(&text_prefill_ids).await {
                    Ok((completed_results, partial_ids)) => {
                        if !partial_ids.is_empty() {
                            tracing::debug!(
                                partial = partial_ids.len(),
                                completed = completed_results.len(),
                                "chunked prefill: {} request(s) need more chunks",
                                partial_ids.len()
                            );
                        }
                        if !completed_results.is_empty() {
                            engine.handle_logits(&completed_results, true).await?;
                            // Requests that just completed prefill are now in Decoding
                            // state with their first token generated. Include them in
                            // this iteration's decode batch so the GPU processes all
                            // active sequences together instead of waiting for the next
                            // schedule_step.
                            for (req_id, _) in &completed_results {
                                decode_ids.push(*req_id);
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            "prefill failed (KV cache full?): {} — stopping {} request(s) with Length",
                            e,
                            text_prefill_ids.len()
                        );
                        let failed_reqs = engine.scheduler.get_running(&text_prefill_ids);
                        for req in &failed_reqs {
                            if req.kv_seq_id >= 0 {
                                engine.model.clear_sequence(req.kv_seq_id);
                            }
                        }
                        for req_id in &text_prefill_ids {
                            engine.scheduler.mark_finished(*req_id, StopReason::Length);
                        }
                    }
                }
            }

            // 2. Vision KV decode — CLIP was pre-computed at submission time.
            if !vision_params.is_empty() {
                for (_, kv_seq_id, prefix_seq_id, _) in &vision_params {
                    if let Some(psid) = *prefix_seq_id {
                        engine.model.clear_sequence(psid);
                        engine.scheduler.return_prefix_seq_id(psid);
                    }
                    engine.model.clear_sequence(*kv_seq_id);
                }

                let t0 = std::time::Instant::now();
                let mut decode_params: Vec<VisionDecodeParams> = Vec::new();
                let mut decode_req_ids: Vec<u64> = Vec::new();
                let mut fallback_handles = Vec::new();

                for (req_id, kv_seq_id, _, params) in vision_params {
                    if let Some((_, preprocessed)) =
                        engine.vision_preprocess_results.remove(&req_id)
                    {
                        decode_req_ids.push(req_id);
                        decode_params.push(VisionDecodeParams {
                            seq_id: kv_seq_id,
                            preprocessed,
                            temperature: params.temperature,
                            top_p: params.top_p,
                            top_k: params.top_k,
                            repetition_penalty: params.repetition_penalty,
                            seed: params.seed,
                        });
                    } else {
                        let model = engine.model.clone();
                        let pp = VisionPreprocessParams {
                            text_prompt: params.text_prompt.clone(),
                            image_bytes: Arc::clone(&params.image_bytes),
                        };
                        let handle =
                            tokio::task::spawn_blocking(move || model.vision_preprocess_sync(&pp));
                        fallback_handles.push((req_id, kv_seq_id, params, handle));
                    }
                }

                for (req_id, kv_seq_id, params, handle) in fallback_handles {
                    match handle.await {
                        Ok(Ok(preprocessed)) => {
                            decode_req_ids.push(req_id);
                            decode_params.push(VisionDecodeParams {
                                seq_id: kv_seq_id,
                                preprocessed,
                                temperature: params.temperature,
                                top_p: params.top_p,
                                top_k: params.top_k,
                                repetition_penalty: params.repetition_penalty,
                                seed: params.seed,
                            });
                        }
                        Ok(Err(e)) => {
                            tracing::warn!(
                                request_id = req_id,
                                "vision preprocess failed: {} — stopping with Length",
                                e
                            );
                            engine.scheduler.mark_finished(req_id, StopReason::Length);
                        }
                        Err(e) => {
                            tracing::warn!(
                                request_id = req_id,
                                "vision preprocess spawn failed: {} — stopping with Length",
                                e
                            );
                            engine.scheduler.mark_finished(req_id, StopReason::Length);
                        }
                    }
                }
                let preprocess_ms = t0.elapsed().as_millis() as u64;

                if !decode_params.is_empty() {
                    let n_vision = decode_req_ids.len();
                    let model = engine.model.clone();
                    let decode_result = tokio::task::spawn_blocking(move || {
                        model.vision_decode_prefill_batch_sync(decode_params)
                    })
                    .await;
                    let total_ms = t0.elapsed().as_millis() as u64;
                    tracing::info!(
                        n = n_vision,
                        preprocess_ms,
                        decode_ms = total_ms - preprocess_ms,
                        total_ms,
                        "vision batch prefill complete"
                    );

                    match decode_result {
                        Ok(Ok(results)) => {
                            for (i, (n_past, logits)) in results.into_iter().enumerate() {
                                let req_id = decode_req_ids[i];
                                engine.scheduler.set_prefilled_tokens(req_id, n_past);
                                engine.handle_logits(&[(req_id, logits)], true).await?;
                            }
                        }
                        Ok(Err(e)) => {
                            tracing::warn!("vision batch decode failed: {}", e);
                            for req_id in &decode_req_ids {
                                engine.scheduler.mark_finished(*req_id, StopReason::Length);
                            }
                        }
                        Err(e) => {
                            tracing::warn!("vision batch decode spawn failed: {}", e);
                            for req_id in &decode_req_ids {
                                engine.scheduler.mark_finished(*req_id, StopReason::Length);
                            }
                        }
                    }
                }
            }

            // 3. Batched decode for all Decoding requests.
            if !decode_ids.is_empty() {
                let t_decode = std::time::Instant::now();
                match engine.run_decode(&decode_ids).await {
                    Ok(decode_results) => {
                        let gpu_ms = t_decode.elapsed().as_millis() as u64;
                        engine.handle_logits(&decode_results, false).await?;
                        let total_ms = t_decode.elapsed().as_millis() as u64;
                        if total_ms > 10 {
                            tracing::info!(
                                n = decode_ids.len(),
                                gpu_ms,
                                total_ms,
                                "decode step slow"
                            );
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            "decode failed (KV cache full?): {} — stopping {} request(s) with Length",
                            e,
                            decode_ids.len()
                        );
                        for req_id in &decode_ids {
                            engine.model.cleanup_request(*req_id);
                            engine.scheduler.mark_finished(*req_id, StopReason::Length);
                        }
                    }
                }
            }
        }
    }

    /// Text-only prefill. Returns `(completed_results, partial_req_ids)`.
    /// `completed_results` contains logits for requests that finished their final chunk.
    /// `partial_req_ids` contains IDs of requests that still have tokens remaining (chunked prefill).
    pub(super) async fn run_prefill(
        &self,
        req_ids: &[u64],
    ) -> Result<(Vec<(u64, Logits)>, Vec<u64>)> {
        let requests = self.scheduler.get_running(req_ids);
        let chunk_limit = self.chunked_prefill_tokens;

        let model_requests: Vec<InferenceRequestForModel> = requests
            .iter()
            .map(|r| InferenceRequestForModel {
                id: r.id,
                prompt_tokens: Arc::clone(&r.prompt_tokens),
                last_token: r.last_token,
                generated_tokens: r.generated_tokens,
                max_new_tokens: r.max_new_tokens,
                context_len: r.context_len(),
                kv_seq_id: r.kv_seq_id,
                temperature: r.sampling.temperature,
                top_p: r.sampling.top_p,
                top_k: r.sampling.top_k,
                min_p: r.sampling.min_p,
                repetition_penalty: r.sampling.repetition_penalty,
                frequency_penalty: r.sampling.frequency_penalty,
                presence_penalty: r.sampling.presence_penalty,
                seed: r.sampling.seed,
                generated_token_ids: r.generated_token_ids.clone(),
                skip_prefix_tokens: r.skip_prefix_tokens,
                prefix_seq_id: r.prefix_seq_id,
                prefill_chunk_progress: r.prefill_chunk_progress,
                prefill_chunk_limit: chunk_limit,
                grammar: r.sampling.grammar.clone(),
            })
            .collect();

        let prefix_cleanup: Vec<i32> = model_requests
            .iter()
            .filter_map(|r| r.prefix_seq_id)
            .collect();

        let model = self.model.clone();
        let ids = req_ids.to_vec();
        let raw = tokio::task::spawn_blocking(move || model.prefill_sync(&ids, &model_requests))
            .await
            .map_err(|e| anyhow::anyhow!("prefill spawn_blocking: {}", e))??;

        for prefix_seq_id in prefix_cleanup {
            self.model.clear_sequence(prefix_seq_id);
            self.scheduler.return_prefix_seq_id(prefix_seq_id);
        }

        let mut completed = Vec::new();
        let mut partial_ids = Vec::new();
        for (id, logits, tokens_in_kv) in raw {
            if tokens_in_kv > 0 {
                self.scheduler.set_prefilled_tokens(id, tokens_in_kv);
            }
            // Check if this request has more tokens to prefill.
            if let Some(req) = requests.iter().find(|r| r.id == id) {
                let effective_skip = req
                    .skip_prefix_tokens
                    .saturating_sub(1)
                    .min(req.prompt_tokens.len());
                let total_to_submit = req.prompt_tokens.len() - effective_skip;
                let progress = req.prefill_chunk_progress;
                let submitted_this_chunk = if chunk_limit > 0 {
                    (total_to_submit - progress).min(chunk_limit)
                } else {
                    total_to_submit - progress
                };
                let new_progress = progress + submitted_this_chunk;

                if chunk_limit > 0 && new_progress < total_to_submit {
                    self.scheduler
                        .advance_prefill_chunk_progress(id, submitted_this_chunk);
                    partial_ids.push(id);
                } else {
                    completed.push((id, logits));
                }
            } else {
                completed.push((id, logits));
            }
        }

        Ok((completed, partial_ids))
    }

    pub(super) async fn run_decode(&self, req_ids: &[u64]) -> Result<Vec<(u64, Logits)>> {
        let requests = self.scheduler.get_running(req_ids);

        // Copy-on-write: if any block in a decoding request is shared (ref_count > 1),
        // allocate a new exclusive copy before llama.cpp writes to it.
        for req in &requests {
            for (logical_idx, &block_id) in req.page_table.entries.iter().enumerate() {
                if self.kv_cache.is_shared(block_id) {
                    if let Some(new_block_id) = self.kv_cache.copy_on_write(block_id) {
                        self.scheduler
                            .cow_update_page_table(req.id, logical_idx, new_block_id);
                        tracing::debug!(
                            request_id = req.id,
                            logical_idx,
                            old_block = block_id,
                            new_block = new_block_id,
                            "CoW: privatised shared KV block before decode"
                        );
                    }
                }
            }
        }

        let model_requests: Vec<InferenceRequestForModel> = requests
            .iter()
            .map(|r| InferenceRequestForModel {
                id: r.id,
                prompt_tokens: Arc::clone(&r.prompt_tokens),
                last_token: r.last_token,
                generated_tokens: r.generated_tokens,
                max_new_tokens: r.max_new_tokens,
                context_len: r.context_len(),
                kv_seq_id: r.kv_seq_id,
                temperature: r.sampling.temperature,
                top_p: r.sampling.top_p,
                top_k: r.sampling.top_k,
                min_p: r.sampling.min_p,
                repetition_penalty: r.sampling.repetition_penalty,
                frequency_penalty: r.sampling.frequency_penalty,
                presence_penalty: r.sampling.presence_penalty,
                seed: r.sampling.seed,
                generated_token_ids: r.generated_token_ids.clone(),
                skip_prefix_tokens: 0,
                prefix_seq_id: None,
                prefill_chunk_progress: 0,
                prefill_chunk_limit: 0,
                grammar: r.sampling.grammar.clone(),
            })
            .collect();
        let model = self.model.clone();
        let req_ids_vec = req_ids.to_vec();
        tokio::task::spawn_blocking(move || model.decode_sync(&req_ids_vec, &model_requests))
            .await
            .map_err(|e| anyhow::anyhow!("decode spawn_blocking: {}", e))?
    }
}
