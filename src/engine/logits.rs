use anyhow::Result;
use tracing::debug;

use crate::scheduler::{StopReason, Token};

use super::model::Logits;
use super::output_filter::{
    apply_output_filter, check_stop_sequences, drain_valid_utf8, PerRequestState,
};
use super::{InferenceEngine, SPM_SPACE};

impl InferenceEngine {
    pub(super) async fn handle_logits(
        &self,
        results: &[(u64, Logits)],
        from_prefill: bool,
    ) -> Result<()> {
        let req_ids: Vec<u64> = results.iter().map(|(id, _)| *id).collect();
        let running = self.scheduler.get_running(&req_ids);
        let eos_token_id = self.model.eos_token_id();

        let mut token_updates: Vec<(u64, i32)> = Vec::new();

        for (req_id, logits) in results {
            let req = running.iter().find(|r| r.id == *req_id);
            let Some(req) = req else {
                continue;
            };

            let token_id = logits.sampled_token;
            let is_eos = self.model.is_eog_token(token_id);
            let _ = eos_token_id;
            let reached_max = req.generated_tokens + 1 >= req.max_new_tokens;

            let token_bytes: Vec<u8> = if is_eos {
                vec![]
            } else {
                self.model.token_to_piece_bytes(token_id)
            };

            let (text, is_stop_hit) = {
                let mut state =
                    self.per_request_state
                        .entry(*req_id)
                        .or_insert_with(|| PerRequestState {
                            show_thinking: req.sampling.show_thinking,
                            in_thinking: req.sampling.initial_in_thinking,
                            emit_think_open_tag: req.sampling.initial_in_thinking,
                            model_control_patterns: self.model_stop_tokens.clone(),
                            max_thinking_chars: req.sampling.max_thinking_chars,
                            ..Default::default()
                        });

                state.utf8_buf.extend_from_slice(&token_bytes);
                let raw_text = drain_valid_utf8(&mut state.utf8_buf).replace(SPM_SPACE, " ");
                let (filtered, control_stop) = apply_output_filter(&mut state, &raw_text);
                let (text, user_stop) =
                    check_stop_sequences(&mut state, filtered, &req.sampling.stop);

                (text, control_stop || user_stop)
            };

            let is_done = is_eos || reached_max || is_stop_hit;
            let stop_reason: Option<StopReason> = if is_done {
                Some(if is_stop_hit {
                    StopReason::StopSequence
                } else if is_eos {
                    StopReason::Eos
                } else {
                    StopReason::Length
                })
            } else {
                None
            };

            let send_ok = req
                .response_tx
                .send(Token {
                    id: *req_id,
                    token_id,
                    text,
                    is_eos,
                    stop_reason: stop_reason.clone(),
                })
                .is_ok();

            debug!(
                request_id = req_id,
                token_id, is_stop_hit, "token generated"
            );

            if !send_ok {
                if req.kv_seq_id >= 0 {
                    self.model.clear_sequence(req.kv_seq_id);
                }
                self.model.cleanup_request(*req_id);
                self.scheduler.mark_finished(*req_id, StopReason::Preempt);
                self.per_request_state.remove(req_id);
                continue;
            }

            if let Some(m) = &self.metrics {
                m.tokens_generated_total.inc();
                if req.generated_tokens == 0 {
                    let ttft = req.submitted_at.elapsed().as_secs_f64();
                    m.ttft_seconds
                        .with_label_values(&[&self.model_name])
                        .observe(ttft);
                }
            }

            if is_done {
                if let Some(m) = &self.metrics {
                    let reason_label = match &stop_reason {
                        Some(StopReason::Eos) => "stop",
                        Some(StopReason::Length) => "length",
                        Some(StopReason::StopSequence) => "stop",
                        Some(StopReason::Preempt) => "preempt",
                        None => "unknown",
                    };
                    m.requests_total.with_label_values(&[reason_label]).inc();
                    let elapsed = req.submitted_at.elapsed().as_secs_f64();
                    m.request_latency_seconds.observe(elapsed);
                }

                let should_clear = if self.supports_prefix_cache
                    && matches!(
                        stop_reason,
                        Some(StopReason::Eos)
                            | Some(StopReason::Length)
                            | Some(StopReason::StopSequence)
                    ) {
                    !self.scheduler.try_insert_prefix(*req_id)
                } else {
                    true
                };

                if should_clear && req.kv_seq_id >= 0 {
                    self.model.clear_sequence(req.kv_seq_id);
                }

                self.model.cleanup_request(*req_id);
                self.scheduler.mark_finished(*req_id, stop_reason.unwrap());

                self.per_request_state.remove(req_id);
            } else {
                token_updates.push((*req_id, token_id));
            }
        }

        if !token_updates.is_empty() {
            self.scheduler
                .batch_update_after_tokens(&token_updates, from_prefill);
        }

        Ok(())
    }
}
