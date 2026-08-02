use std::sync::atomic::Ordering;

use tracing::{debug, info};

use crate::kv_cache::PageTable;

use super::batch;
use super::batch::{ScheduledBatch, StopReason};
use super::Scheduler;

impl Scheduler {
    /// Number of blocks needed for a request (prompt + max_new_tokens, in blocks).
    pub(super) fn blocks_needed(&self, req: &batch::InferenceRequest) -> usize {
        let total_tokens = req.n_positions() + req.max_new_tokens;
        let block_size = self.kv_cache.block_size();
        total_tokens.div_ceil(block_size)
    }

    /// One scheduling step. Returns prefill and decode batches.
    ///
    /// 1. Retire Finished requests: park each one's sequence as a reusable idle slot
    ///    (KV kept, blocks kept), or release it outright when its KV must not be
    ///    reused.
    /// 2. Admit from waiting_queue, choosing each request's slot by longest-common-
    ///    prefix affinity so it inherits as much resident KV as possible.
    ///    Admission NEVER preempts a *running* request: blocks are fully reserved at
    ///    admission (prompt + max_new_tokens), so running requests never grow —
    ///    evicting an older running request for a newer waiting one is both unfair
    ///    and livelock-prone (the pair can evict each other forever). A request that
    ///    doesn't fit waits (FIFO). Reclaiming an *idle* slot is not preemption: its
    ///    request already finished and the client already has its output.
    /// 3. Return prefill and decode id lists, plus the KV trims/clears the engine
    ///    must apply before the next prefill.
    pub fn schedule_step(&self) -> ScheduledBatch {
        // Lock ordering (must be consistent across ALL callers to avoid deadlock):
        //   running_batch → waiting_queue → slots
        let mut running = match self.running_batch.lock() {
            Ok(g) => g,
            Err(e) => {
                tracing::error!("running_batch lock poisoned: {}", e);
                return ScheduledBatch::default();
            }
        };
        let mut waiting = match self.waiting_queue.lock() {
            Ok(g) => g,
            Err(e) => {
                tracing::error!("waiting_queue lock poisoned: {}", e);
                return ScheduledBatch::default();
            }
        };
        let mut slots = match self.slots.lock() {
            Ok(g) => g,
            Err(e) => {
                tracing::error!("slots lock poisoned: {}", e);
                return ScheduledBatch::default();
            }
        };

        let mut kv_trims: Vec<(i32, usize)> = Vec::new();
        let mut kv_clears: Vec<i32> = Vec::new();

        // 1. Retire Finished requests.
        //
        // `park_finished` (called from the engine on the completion path) has already
        // decided each one's fate and recorded it in `park_state`; here we only apply
        // it. A parked request handed its blocks to its slot and left `kv_seq_id = -1`,
        // so it is skipped below and its blocks are NOT freed — that is the whole
        // point: the sequence stays resident as a cache entry.
        let (finished, still_running): (Vec<_>, Vec<_>) = std::mem::take(&mut *running)
            .into_iter()
            .partition(|r| r.is_finished());

        for req in &finished {
            if req.kv_seq_id < 0 {
                continue; // already parked into its slot
            }
            let mut blocks = slots.release(req.kv_seq_id);
            blocks.extend_from_slice(req.page_table.block_ids());
            if !blocks.is_empty() {
                self.kv_cache.free_blocks(&blocks);
                debug!(
                    request_id = req.id,
                    blocks = blocks.len(),
                    "freed KV blocks for finished request"
                );
            }
            kv_clears.push(req.kv_seq_id);
        }
        *running = still_running;

        // 2. Admit from waiting_queue
        let mut prefill = Vec::new();
        let mut decode = Vec::new();
        // Always empty since admission stopped preempting; retained so future
        // preemption sources (priority, growth) can reuse the engine-side clearing.
        let preempted_seq_ids = Vec::new();

        'admit: while let Some(mut req) = waiting.pop_front() {
            // A request that could never fit, even into an empty pool, is rejected
            // synchronously by `Scheduler::submit()` before it ever reaches this queue
            // (0.16) — it's a static check (prompt + max_new_tokens vs. total pool size)
            // that doesn't need a scheduler turn. No corresponding check needed here.

            // Pick the slot whose resident KV this prompt can reuse most of.
            //
            // `skip_prefix_cache` (LoRA requests): KV computed under one adapter's
            // weights is invalid input for a different adapter (or none) at the same
            // positions, so such a request must start from a clean slot — see
            // docs/design/lora-support.md.
            let allow_reuse = self.kv_reuse && !req.skip_prefix_cache;
            let Some(choice) =
                slots.select(&req.prompt_tokens, self.slot_prompt_similarity, allow_reuse)
            else {
                // Every slot is Busy — wait for one to retire.
                waiting.push_front(req);
                break 'admit;
            };

            // Token-exact reuse, then llama-server's [TAG_PROMPT_LOGITS] guard
            // (server-context.cpp:3356-3361): if the prompt is *entirely* resident
            // there is nothing left to decode and no logits would be produced, so
            // step one token back and recompute the final position.
            let mut n_past = choice.lcp.min(req.n_positions());
            if n_past > 0 && n_past == req.n_positions() {
                n_past -= 1;
            }

            // Blocks are a budget, not addresses (see slots.rs): the request inherits
            // whatever the slot already holds and tops up only the difference.
            let needed = self.blocks_needed(&req);
            let have = slots.blocks_at(choice.index);
            let top_up = needed.saturating_sub(have);

            // Make room by reclaiming idle slots — LRU first, never the slot we just
            // chose, and never a Busy one. Not preemption; see SlotTable::reclaim_lru.
            while top_up > 0 && !self.kv_cache.can_allocate(top_up) {
                let Some((victim_seq, victim_blocks)) = slots.reclaim_lru(choice.index) else {
                    break;
                };
                self.kv_cache.free_blocks(&victim_blocks);
                kv_clears.push(victim_seq);
                debug!(
                    seq_id = victim_seq,
                    blocks = victim_blocks.len(),
                    "reclaimed idle slot to make room"
                );
            }

            if top_up > 0 && !self.kv_cache.can_allocate(top_up) {
                // Still short — wait for capacity (FIFO head-of-line). Running
                // requests keep their reservations.
                waiting.push_front(req);
                break 'admit;
            }

            let new_ids = if top_up > 0 {
                match self.kv_cache.allocate(top_up) {
                    Ok(ids) => ids,
                    Err(_) => {
                        waiting.push_front(req);
                        break 'admit;
                    }
                }
            } else {
                Vec::new()
            };

            let id = req.id;
            let (seq_id, mut blocks) = slots.claim(choice.index, id);
            // Give back any surplus the previous occupant held beyond this request's
            // reservation, so a short prompt after a long one doesn't pin the pool.
            if blocks.len() > needed {
                let surplus = blocks.split_off(needed);
                self.kv_cache.free_blocks(&surplus);
            }
            blocks.extend(new_ids);

            req.page_table = PageTable::new(blocks);
            req.kv_seq_id = seq_id;
            req.skip_prefix_tokens = n_past;
            req.prefill_pos = n_past;
            req.prefix_seq_id = None;
            req.stop_reason = None;
            req.state = batch::RequestState::Prefilling;

            // Drop everything the slot holds past the divergence point before the
            // next prefill writes there (server-context.cpp:3392-3399). Stale cells
            // beyond n_past would collide with this request's own positions.
            kv_trims.push((seq_id, n_past));

            if n_past > 0 {
                self.prefix_hits.fetch_add(1, Ordering::Relaxed);
                info!(
                    request_id = id,
                    seq_id,
                    cached_tokens = n_past,
                    prompt_tokens = req.n_positions(),
                    "slot prefix hit — skipping prefill of resident tokens"
                );
            } else {
                self.prefix_misses.fetch_add(1, Ordering::Relaxed);
                info!(request_id = id, seq_id, "request admitted to batch");
            }
            running.push(req);
        }

        // 4. Build the prefill and decode lists from the running batch. A request stays
        //    `Prefilling` across steps until its prompt is fully chunked into the KV, so
        //    it is re-emitted to `prefill` each step (both freshly admitted and
        //    still-in-progress); `Decoding` requests generate one token per step.
        for req in running.iter() {
            match req.state {
                batch::RequestState::Prefilling => prefill.push(req.id),
                batch::RequestState::Decoding => decode.push(req.id),
                _ => {}
            }
        }

        ScheduledBatch {
            prefill,
            decode,
            preempted_seq_ids,
            kv_trims,
            kv_clears,
        }
    }

    /// Replace the physical block at `logical_idx` in the request's page table with `new_block_id`.
    ///
    /// Called by the engine's CoW path after `KVCacheManager::copy_on_write` has allocated a
    /// new exclusive block for a request that was sharing a block with the prefix cache.
    pub fn cow_update_page_table(&self, req_id: u64, logical_idx: usize, new_block_id: usize) {
        if let Ok(mut running) = self.running_batch.lock() {
            for req in running.iter_mut() {
                if req.id == req_id {
                    if let Some(entry) = req.page_table.entries.get_mut(logical_idx) {
                        *entry = new_block_id;
                    }
                    break;
                }
            }
        }
    }

    /// Advance a request's prefill cursor after a chunk was submitted to the model.
    /// Called every prefill step; the request stays `Prefilling` (and is re-emitted
    /// to the prefill batch) until its final chunk is sampled by `handle_logits`.
    pub fn advance_prefill(&self, req_id: u64, new_prefill_pos: usize) {
        if let Ok(mut running) = self.running_batch.lock() {
            for req in running.iter_mut() {
                if req.id == req_id {
                    req.prefill_pos = new_prefill_pos;
                    break;
                }
            }
        }
    }

    /// Record that context rolling discarded `n_discard` of a request's oldest KV
    /// tokens. Reduces its logical context length (via `rolled_tokens`) so the next
    /// decode position matches the shifted KV cache.
    pub fn record_context_roll(&self, req_id: u64, n_discard: usize) {
        if let Ok(mut running) = self.running_batch.lock() {
            for req in running.iter_mut() {
                if req.id == req_id {
                    req.rolled_tokens += n_discard;
                    break;
                }
            }
        }
    }

    /// Record how many tokens were actually submitted to llama.cpp during prefill.
    /// Must be called once per request immediately after `run_prefill` returns.
    pub fn set_prefilled_tokens(&self, req_id: u64, count: usize) {
        if let Ok(mut running) = self.running_batch.lock() {
            for req in running.iter_mut() {
                if req.id == req_id {
                    req.prefilled_tokens = count;
                    break;
                }
            }
        }
    }

    /// Update request state after a generated token.
    pub fn update_after_token(&self, req_id: u64, token_id: i32, from_prefill: bool) {
        let mut running = match self.running_batch.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        for req in running.iter_mut() {
            if req.id == req_id {
                req.last_token = Some(token_id);
                req.generated_tokens += 1;
                req.generated_token_ids.push(token_id);
                if from_prefill && req.state == batch::RequestState::Prefilling {
                    req.state = batch::RequestState::Decoding;
                }
                break;
            }
        }
    }

    /// Mark request as Finished with the given stop reason.
    pub fn mark_finished(&self, req_id: u64, stop_reason: StopReason) {
        let mut running = match self.running_batch.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        for req in running.iter_mut() {
            if req.id == req_id {
                req.state = batch::RequestState::Finished;
                req.stop_reason = Some(stop_reason);
                break;
            }
        }
    }

    /// Park a finished request's sequence so its KV stays resident and reusable.
    ///
    /// This is the counterpart to llama-server keeping `slot.prompt.tokens` after a
    /// task completes (`server-context.cpp:489`). The whole logical sequence —
    /// **prompt *and* generated tokens** — becomes the slot's resident token list, so
    /// the next turn of a conversation (whose prompt contains the previous reply)
    /// matches well past where the previous prompt ended. The old block-hash cache
    /// discarded the generated tail, which is why multi-turn chat never hit it.
    ///
    /// The request keeps no blocks: they transfer to the slot, and `kv_seq_id` is set
    /// to `-1` so `schedule_step`'s retire pass knows not to free them.
    ///
    /// Returns `true` if the sequence was parked. `false` means the caller must clear
    /// the llama.cpp sequence itself — the KV is not safe to reuse:
    ///
    /// * **`--kv-reuse false`** — reuse disabled outright.
    /// * **context-rolled** (`rolled_tokens > 0`) — rolling discards the oldest KV
    ///   window and shifts the rest, so resident positions no longer line up with the
    ///   token list; a later LCP match would read the wrong cells.
    /// * **LoRA** (`skip_prefix_cache`) — KV computed under one adapter's weights is
    ///   invalid input for another (docs/design/lora-support.md).
    /// * **multimodal** — `prompt_tokens` is empty for these (positions come from
    ///   image chunks), so the token list can't describe what's resident.
    ///
    /// Lock ordering: running_batch → slots (matches schedule_step).
    pub fn park_finished(&self, req_id: u64) -> bool {
        if !self.kv_reuse {
            return false;
        }
        let mut running = match self.running_batch.lock() {
            Ok(g) => g,
            Err(_) => return false,
        };
        let mut slots = match self.slots.lock() {
            Ok(g) => g,
            Err(_) => return false,
        };

        let Some(req) = running.iter_mut().find(|r| r.id == req_id) else {
            return false;
        };
        if req.kv_seq_id < 0
            || req.rolled_tokens > 0
            || req.skip_prefix_cache
            || req.multimodal.is_some()
        {
            return false;
        }

        // What llama.cpp actually holds for this sequence: the prompt it prefilled
        // followed by every token it generated.
        let mut resident = req.prompt_tokens.clone();
        resident.extend_from_slice(&req.generated_token_ids);

        let seq_id = req.kv_seq_id;
        let blocks = std::mem::take(&mut req.page_table).entries;
        if !slots.park(seq_id, resident, blocks.clone()) {
            // Unknown seq_id (defensive) — hand the blocks back rather than leaking
            // them. `page_table` was already emptied above, so free the copy we hold.
            self.kv_cache.free_blocks(&blocks);
            return false;
        }

        // Zero out seq ownership so schedule_step's retire pass won't double-free.
        req.kv_seq_id = -1;
        debug!(
            request_id = req_id,
            seq_id,
            resident_tokens = req.prompt_tokens.len() + req.generated_token_ids.len(),
            "parked sequence as reusable idle slot"
        );
        true
    }

    /// Get running requests by IDs.
    pub fn get_running(&self, ids: &[u64]) -> Vec<batch::InferenceRequest> {
        let running = match self.running_batch.lock() {
            Ok(g) => g,
            Err(_) => return vec![],
        };
        let id_set: ahash::AHashSet<_> = ids.iter().copied().collect();
        running
            .iter()
            .filter(|r| id_set.contains(&r.id))
            .cloned()
            .collect()
    }

    /// Swap a decoding request out of the GPU KV cache into the `Swapped` state.
    ///
    /// The `page_table` is retained (the blocks remain allocated but are
    /// logically "on CPU" after the caller copies the raw KV tensors to a CPU
    /// buffer).  The `kv_seq_id` is kept so the engine can clear the llama.cpp
    /// sequence slot immediately after the caller copies the data out.
    ///
    /// Returns `true` if the request was found in `Decoding` state and
    /// transitioned to `Swapped`; `false` otherwise.
    ///
    /// # Implementation note
    /// The actual byte-level KV transfer (GPU → CPU memcpy) must be performed
    /// by the *caller* **before** calling this method, since the scheduler has
    /// no access to the model's tensor buffers.  See [`RequestState::Swapped`]
    /// for the current limitations.
    pub fn swap_out(&self, req_id: u64) -> bool {
        if let Ok(mut running) = self.running_batch.lock() {
            for req in running.iter_mut() {
                if req.id == req_id && req.state == batch::RequestState::Decoding {
                    req.state = batch::RequestState::Swapped;
                    tracing::debug!(request_id = req_id, "request swapped out to CPU");
                    return true;
                }
            }
        }
        false
    }

    /// Swap a previously swapped-out request back in to the GPU KV cache.
    ///
    /// Transitions the request from `Swapped` to `Decoding`.  The caller must
    /// have already copied the KV data from the CPU buffer back to the GPU
    /// **before** calling this method.
    ///
    /// Returns `true` if the request was found in `Swapped` state and
    /// transitioned to `Decoding`; `false` otherwise.
    pub fn swap_in(&self, req_id: u64) -> bool {
        if let Ok(mut running) = self.running_batch.lock() {
            for req in running.iter_mut() {
                if req.id == req_id && req.state == batch::RequestState::Swapped {
                    req.state = batch::RequestState::Decoding;
                    tracing::debug!(request_id = req_id, "request swapped back in to GPU");
                    return true;
                }
            }
        }
        false
    }

    pub fn queue_depth(&self) -> usize {
        self.waiting_queue.lock().map(|q| q.len()).unwrap_or(0)
    }

    pub fn active_requests(&self) -> usize {
        self.running_batch.lock().map(|r| r.len()).unwrap_or(0)
    }

    /// Number of slots currently holding reusable KV (the analogue of the old
    /// prefix cache's entry count).
    pub fn prefix_cache_size(&self) -> usize {
        self.slots
            .lock()
            .map(|s| s.count(|slot| slot.state == super::slots::SlotState::Idle))
            .unwrap_or(0)
    }
}
