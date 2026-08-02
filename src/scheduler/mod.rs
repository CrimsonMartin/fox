// Continuous batching scheduler with prefix caching.
// Flow: Waiting -> Prefilling -> Decoding -> Finished
// Blocks are fully reserved at admission, so admission never preempts: a request
// that doesn't fit waits in the queue (FIFO) until running requests finish.

mod batch;
mod schedule;
mod slots;

#[allow(unused_imports)]
pub use batch::{
    InferenceRequest, LoraSelection, RequestState, SamplingParams, ScheduledBatch, StopReason,
    Token, TokenLogprob, TopLogprob,
};

use std::collections::VecDeque;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use tracing::info;

use crate::kv_cache::KVCacheManager;

pub use slots::SlotSnapshot;
use slots::SlotTable;

/// Default LCP-similarity floor below which an idle slot's KV is not worth
/// inheriting. Mirrors llama-server's `-sps/--slot-prompt-similarity` default.
pub const DEFAULT_SLOT_PROMPT_SIMILARITY: f32 = 0.1;

/// Scheduler managing waiting queue and running batch.
pub struct Scheduler {
    pub(super) waiting_queue: std::sync::Mutex<VecDeque<InferenceRequest>>,
    pub(super) running_batch: std::sync::Mutex<Vec<InferenceRequest>>,
    pub(super) kv_cache: Arc<KVCacheManager>,
    /// Notified when a new request is submitted, waking the engine loop.
    work_notify: tokio::sync::Notify,
    /// The llama.cpp sequence slots, one per concurrent request, each remembering
    /// the tokens resident in its KV. This *is* fox's prefix cache: it replaces the
    /// old `seq_id_pool` + block-hash `LruCache` pair (see `slots.rs` for why).
    ///
    /// Slot index == `seq_id`, matching `llama-server`'s `slot.id = i`. IDs are no
    /// longer required to be handed out densely: `split_equal`'s
    /// consecutive-ascending constraint only applies to a non-unified cache
    /// (`n_stream > 1`), and fox sets `kv_unified = true` since `1c36faf`, so
    /// `llama-kv-cache.cpp:725` picks `split_simple`, which has no ordering
    /// requirement. See `docs/design/llama-server-gap-analysis.md` §0.1. (The batch
    /// is still *emitted* in ascending seq_id order — see `do_decode_batch`.)
    ///
    /// Lock ordering: always acquire `running_batch` → `waiting_queue` → `slots`.
    pub(crate) slots: std::sync::Mutex<SlotTable>,
    /// Minimum fraction of an incoming prompt that must already be resident in an
    /// idle slot before that slot's KV is inherited instead of starting fresh.
    pub(super) slot_prompt_similarity: f32,
    /// Master switch for KV reuse (`--kv-reuse`). When false, finished sequences are
    /// always cleared and every prompt is prefilled from token 0 — the pre-0.19
    /// behaviour, kept as an escape hatch and as the A/B baseline arm.
    pub(super) kv_reuse: bool,
    /// Lifetime hit counter (for metrics / logging).
    pub prefix_hits: AtomicU64,
    /// Lifetime miss counter (for metrics / logging).
    pub prefix_misses: AtomicU64,
    /// Maximum requests allowed to wait in `waiting_queue` before `submit()` rejects
    /// new ones with `SubmitError::QueueFull`. 0 = unbounded.
    max_queue_depth: usize,
}

/// Why `Scheduler::submit()` refused a request. Both variants are synchronous,
/// pre-queue rejections — the request never touches `waiting_queue` or gets a
/// response channel that could otherwise hang or silently close.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmitError {
    /// `waiting_queue` is already at `max_queue_depth`.
    QueueFull { depth: usize, max: usize },
    /// The request needs more blocks than the KV pool will ever have, even empty.
    TooLarge {
        needed_blocks: usize,
        total_blocks: usize,
    },
}

impl std::fmt::Display for SubmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SubmitError::QueueFull { depth, max } => {
                write!(f, "queue full ({depth}/{max} requests waiting)")
            }
            SubmitError::TooLarge {
                needed_blocks,
                total_blocks,
            } => write!(
                f,
                "request needs {needed_blocks} KV blocks but the pool only has {total_blocks}"
            ),
        }
    }
}

impl Scheduler {
    pub fn new(kv_cache: Arc<KVCacheManager>, max_batch_size: usize) -> Self {
        Self::with_max_queue_depth(kv_cache, max_batch_size, 0)
    }

    /// Like `new`, with an explicit queue-depth cap (0 = unbounded).
    pub fn with_max_queue_depth(
        kv_cache: Arc<KVCacheManager>,
        max_batch_size: usize,
        max_queue_depth: usize,
    ) -> Self {
        Self {
            waiting_queue: std::sync::Mutex::new(VecDeque::new()),
            running_batch: std::sync::Mutex::new(Vec::new()),
            kv_cache,
            work_notify: tokio::sync::Notify::new(),
            slots: std::sync::Mutex::new(SlotTable::new(max_batch_size)),
            slot_prompt_similarity: DEFAULT_SLOT_PROMPT_SIMILARITY,
            kv_reuse: true,
            prefix_hits: AtomicU64::new(0),
            prefix_misses: AtomicU64::new(0),
            max_queue_depth,
        }
    }

    /// Override the KV-reuse policy (`--kv-reuse` / `--slot-prompt-similarity`).
    /// Chained onto a constructor so the many test and bench call sites that only
    /// care about the defaults stay untouched.
    pub fn with_kv_reuse(mut self, kv_reuse: bool, slot_prompt_similarity: f32) -> Self {
        self.kv_reuse = kv_reuse;
        self.slot_prompt_similarity = slot_prompt_similarity.clamp(0.0, 1.0);
        self
    }

    /// Submit a request to the waiting queue. Rejects synchronously (before the
    /// request ever enters the queue or gets a response channel) when the queue is
    /// full or the request could never fit in the KV pool even when empty — both
    /// checks are statically knowable at submission time, so there is no need to
    /// wait for a scheduler turn to reject them.
    pub fn submit(&self, req: InferenceRequest) -> Result<(), SubmitError> {
        let needed = self.blocks_needed(&req);
        let total = self.kv_cache.total_blocks();
        if needed > total {
            return Err(SubmitError::TooLarge {
                needed_blocks: needed,
                total_blocks: total,
            });
        }

        let mut q = match self.waiting_queue.lock() {
            Ok(g) => g,
            Err(e) => {
                tracing::error!("waiting_queue lock poisoned on submit: {}", e);
                e.into_inner()
            }
        };
        if self.max_queue_depth > 0 && q.len() >= self.max_queue_depth {
            return Err(SubmitError::QueueFull {
                depth: q.len(),
                max: self.max_queue_depth,
            });
        }
        info!(request_id = req.id, "request admitted to waiting queue");
        q.push_back(req);
        drop(q);
        self.work_notify.notify_one();
        Ok(())
    }

    /// Per-slot state for `GET /slots`. Empty on a poisoned lock rather than
    /// panicking — introspection must never take the server down.
    pub fn slots_snapshot(&self) -> Vec<SlotSnapshot> {
        self.slots.lock().map(|s| s.snapshot()).unwrap_or_default()
    }

    /// Wait until at least one request is available to schedule.
    pub async fn wait_for_work(&self) {
        self.work_notify.notified().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::model::ModelConfig;
    use crate::kv_cache::KVCacheManager;
    use std::sync::atomic::Ordering;

    #[test]
    fn test_scheduler_submit_and_schedule() {
        let config = ModelConfig {
            num_layers: 32,
            num_heads: 32,
            num_heads_kv: 32,
            head_dim: 128,
            n_embd: 4096,
            vocab_size: 32000,
        };
        let kv = Arc::new(KVCacheManager::new(&config, 1_000_000_000, 0.5, 16, 1, 1));
        let sched = Scheduler::new(kv, 8);

        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let req = InferenceRequest::new(1, vec![1, 2, 3], 10, SamplingParams::default(), tx);
        sched.submit(req).unwrap();

        assert_eq!(sched.queue_depth(), 1);
        let batch = sched.schedule_step();
        assert_eq!(batch.prefill, vec![1]);
        assert_eq!(sched.queue_depth(), 0);
    }

    fn test_kv(block_size: usize) -> Arc<KVCacheManager> {
        let config = ModelConfig {
            num_layers: 2,
            num_heads: 2,
            num_heads_kv: 2,
            head_dim: 64,
            n_embd: 128,
            vocab_size: 1000,
        };
        Arc::new(KVCacheManager::new(
            &config,
            500_000_000,
            0.5,
            block_size,
            1,
            1,
        ))
    }

    /// Run a request to completion and park its sequence, the way the engine does:
    /// admit → generate `gen` tokens → finish → park.
    fn run_and_park(sched: &Scheduler, id: u64, prompt: Vec<i32>, gen: &[i32]) {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        sched
            .submit(InferenceRequest::new(
                id,
                prompt,
                8,
                SamplingParams::default(),
                tx,
            ))
            .unwrap();
        sched.schedule_step();
        for (i, &t) in gen.iter().enumerate() {
            sched.update_after_token(id, t, i == 0);
        }
        sched.mark_finished(id, StopReason::Eos);
        assert!(sched.park_finished(id), "request {id} should have parked");
    }

    /// A parked sequence is matched token-exactly, not rounded down to a block
    /// boundary — and the generated tail is part of what's reusable.
    #[test]
    fn parked_sequence_is_reused_token_exactly() {
        let kv = test_kv(16);
        let sched = Scheduler::new(kv, 8);

        // 18 tokens: deliberately NOT a multiple of block_size (16). The old
        // block-hash cache would have reused only 16 of them.
        let tokens: Vec<i32> = (1..=18).collect();
        run_and_park(&sched, 42, tokens.clone(), &[777]);
        assert_eq!(sched.prefix_cache_size(), 1, "one slot holds reusable KV");
        sched.schedule_step(); // retire the finished request

        let (tx2, _rx2) = tokio::sync::mpsc::unbounded_channel();
        sched
            .submit(InferenceRequest::new(
                99,
                tokens,
                5,
                SamplingParams::default(),
                tx2,
            ))
            .unwrap();
        let batch = sched.schedule_step();

        assert!(batch.prefill.contains(&99));
        assert_eq!(sched.prefix_hits.load(Ordering::Relaxed), 1);

        let running = sched.running_batch.lock().unwrap();
        let req = running.iter().find(|r| r.id == 99).expect("99 running");
        // All 18 prompt tokens are resident, but [TAG_PROMPT_LOGITS] steps one back
        // so there is always a token left to decode and produce logits from.
        assert_eq!(req.skip_prefix_tokens, 17);
        assert_eq!(req.prefill_pos, 17, "prefill resumes at the skip boundary");
        assert_eq!(req.kv_seq_id, 0, "inherits the parked sequence in place");
        assert!(
            batch.kv_trims.contains(&(0, 17)),
            "everything past the divergence point must be trimmed before prefill"
        );
    }

    /// Two prompts sharing a prefix: the newcomer reuses exactly the shared span.
    #[test]
    fn partial_prefix_match_reuses_only_the_shared_span() {
        let kv = test_kv(16);
        let sched = Scheduler::new(kv, 8);

        let shared: Vec<i32> = (1..=16).collect();
        run_and_park(&sched, 1, shared.clone(), &[777]);
        sched.schedule_step();

        // Diverges right after the shared span (777 != 100).
        let mut tokens_b = shared.clone();
        tokens_b.extend([100i32, 101, 102, 103]);

        let (tx2, _rx2) = tokio::sync::mpsc::unbounded_channel();
        sched
            .submit(InferenceRequest::new(
                2,
                tokens_b,
                5,
                SamplingParams::default(),
                tx2,
            ))
            .unwrap();
        let batch = sched.schedule_step();

        assert!(batch.prefill.contains(&2));
        assert_eq!(sched.prefix_hits.load(Ordering::Relaxed), 1);

        let running = sched.running_batch.lock().unwrap();
        let req_b = running.iter().find(|r| r.id == 2).expect("req B running");
        assert_eq!(
            req_b.skip_prefix_tokens, 16,
            "reuse stops exactly where the prompts diverge"
        );
    }

    /// The generated reply is reusable too — this is the multi-turn chat case the
    /// old block-hash cache could never serve, because it discarded generation.
    #[test]
    fn generated_tokens_are_reusable_by_the_next_turn() {
        let kv = test_kv(16);
        let sched = Scheduler::new(kv, 8);

        let turn1: Vec<i32> = (1..=20).collect();
        let reply = [900, 901, 902];
        run_and_park(&sched, 1, turn1.clone(), &reply);
        sched.schedule_step();

        // Turn 2's prompt = turn 1 + the assistant's reply + the new user message.
        let mut turn2 = turn1.clone();
        turn2.extend_from_slice(&reply);
        turn2.extend([500i32, 501]);

        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        sched
            .submit(InferenceRequest::new(
                2,
                turn2,
                5,
                SamplingParams::default(),
                tx,
            ))
            .unwrap();
        sched.schedule_step();

        let running = sched.running_batch.lock().unwrap();
        let req = running.iter().find(|r| r.id == 2).expect("turn 2 running");
        assert_eq!(
            req.skip_prefix_tokens,
            turn1.len() + reply.len(),
            "the reply must be reused, not just the previous prompt"
        );
    }

    /// A context-rolled request's resident positions no longer line up with its token
    /// list, so its KV must never become a reuse candidate.
    #[test]
    fn rolled_and_lora_requests_are_never_parked() {
        let kv = test_kv(16);
        let sched = Scheduler::new(kv, 8);

        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        sched
            .submit(InferenceRequest::new(
                1,
                (1..=20).collect(),
                8,
                SamplingParams::default(),
                tx,
            ))
            .unwrap();
        sched.schedule_step();
        sched.record_context_roll(1, 4);
        sched.mark_finished(1, StopReason::Eos);
        assert!(!sched.park_finished(1), "rolled request must not park");

        let (tx2, _rx2) = tokio::sync::mpsc::unbounded_channel();
        let lora_req =
            InferenceRequest::new(2, (1..=20).collect(), 8, SamplingParams::default(), tx2)
                .with_lora(LoraSelection {
                    name: "adapter".into(),
                    scale: 1.0,
                });
        sched.submit(lora_req).unwrap();
        sched.schedule_step();
        sched.mark_finished(2, StopReason::Eos);
        assert!(!sched.park_finished(2), "LoRA request must not park");
    }

    /// Under block pressure, a parked (idle) slot is reclaimed to admit a new request
    /// — and a *running* one never is. Parking KV instead of freeing it is only safe
    /// because of this: without reclaim, warm slots would pin the pool forever.
    #[test]
    fn idle_slot_is_reclaimed_under_block_pressure() {
        use slots::SlotState;

        // 6 blocks total; each request below needs 2 (20 prompt + 8 new = 28 tokens).
        let kv = Arc::new(KVCacheManager::from_kv_tokens(16 * 6, 16));
        let sched = Scheduler::new(kv.clone(), 4);
        assert_eq!(kv.total_blocks(), 6);

        let prompt = |seed: i32| -> Vec<i32> { (0..20).map(|i| seed * 1000 + i).collect() };

        // Park one request, then fill the pool with two live ones.
        run_and_park(&sched, 1, prompt(1), &[777]);
        sched.schedule_step();
        assert_eq!(sched.prefix_cache_size(), 1, "slot 0 parked");

        for id in [2u64, 3] {
            let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
            sched
                .submit(InferenceRequest::new(
                    id,
                    prompt(id as i32),
                    8,
                    SamplingParams::default(),
                    tx,
                ))
                .unwrap();
            sched.schedule_step();
        }
        assert_eq!(kv.allocated_blocks(), 6, "pool is now full");

        // A fourth, unrelated request can only fit if the idle slot gives up its blocks.
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        sched
            .submit(InferenceRequest::new(
                4,
                prompt(4),
                8,
                SamplingParams::default(),
                tx,
            ))
            .unwrap();
        let batch = sched.schedule_step();

        assert!(
            batch.prefill.contains(&4),
            "request 4 should have been admitted by reclaiming the idle slot"
        );
        assert!(
            batch.kv_clears.contains(&0),
            "the reclaimed sequence's KV must be wiped before reuse"
        );
        assert_eq!(
            sched.prefix_cache_size(),
            0,
            "the parked slot was reclaimed"
        );

        // The two live requests were never touched — reclaim is not preemption.
        let running = sched.running_batch.lock().unwrap();
        for id in [2u64, 3] {
            assert!(
                running.iter().any(|r| r.id == id && !r.is_finished()),
                "running request {id} must survive reclamation"
            );
        }
        let slots = sched.slots.lock().unwrap();
        assert_eq!(
            slots.count(|s| matches!(s.state, SlotState::Busy(_))),
            3,
            "requests 2, 3 and 4 hold slots"
        );
    }

    /// `--kv-reuse false` restores the pre-0.19 behaviour end to end.
    #[test]
    fn kv_reuse_disabled_never_parks_or_hits() {
        let kv = test_kv(16);
        let sched = Scheduler::new(kv, 8).with_kv_reuse(false, DEFAULT_SLOT_PROMPT_SIMILARITY);

        let tokens: Vec<i32> = (1..=18).collect();
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        sched
            .submit(InferenceRequest::new(
                1,
                tokens.clone(),
                8,
                SamplingParams::default(),
                tx,
            ))
            .unwrap();
        sched.schedule_step();
        sched.mark_finished(1, StopReason::Eos);
        assert!(!sched.park_finished(1));
        sched.schedule_step();

        let (tx2, _rx2) = tokio::sync::mpsc::unbounded_channel();
        sched
            .submit(InferenceRequest::new(
                2,
                tokens,
                5,
                SamplingParams::default(),
                tx2,
            ))
            .unwrap();
        sched.schedule_step();

        assert_eq!(sched.prefix_hits.load(Ordering::Relaxed), 0);
        let running = sched.running_batch.lock().unwrap();
        let req = running.iter().find(|r| r.id == 2).unwrap();
        assert_eq!(req.skip_prefix_tokens, 0, "must prefill from token 0");
    }

    /// Prefix-cache eviction stress test — settles design-doc §7's open question
    /// ("does the prefix cache leak blocks/seq-ids under churn?") empirically.
    ///
    /// It drives real `schedule_step` + `try_insert_prefix` over hundreds of
    /// admit / finish / cache / hit / refuse-when-full cycles and asserts strict
    /// conservation after every step, under a **move** ownership model: each seq_id
    /// and each KV block is owned by exactly one of {pool, a running request, a
    /// prefix-cache entry} — never dropped, never duplicated. (The engine layers a
    /// KV-copy optimization on top via `return_prefix_seq_id`; the leak question is
    /// about the scheduler's bookkeeping, which the move model exercises exactly.)
    ///
    /// A leak would show up as either a seq_id that can't be accounted for
    /// (`seen.len() != TOTAL_SEQ`) or an allocated KV block that nothing references
    /// (`allocated_blocks() != reachable`), or a non-zero allocation after draining.
    #[test]
    fn stress_slot_reuse_no_leak() {
        use slots::SlotState;
        use std::collections::HashSet;

        const TOTAL_SEQ: usize = 8; // = max_batch_size → slots {0..8}
                                    // Plenty of blocks: this test targets slot churn, not block starvation, so
                                    // keep the loop live and deterministic.
        let kv = test_kv(16);
        let sched = Scheduler::new(kv.clone(), TOTAL_SEQ);

        // Deterministic prompt of `blocks` full 16-token blocks (+3 leftover tokens),
        // content keyed by `seed` so distinct seeds have distinct prefixes.
        let prompt = |seed: i32, blocks: usize| -> Vec<i32> {
            (0..(blocks * 16 + 3) as i32)
                .map(|i| seed * 1000 + i)
                .collect()
        };

        // Assert full conservation of sequences and blocks against the live state.
        let check_conservation = |label: &str| {
            let running = sched.running_batch.lock().unwrap();
            let slots = sched.slots.lock().unwrap();

            // Every seq_id appears exactly once, in exactly one state, and all
            // TOTAL_SEQ are accounted for.
            let mut seen: HashSet<i32> = HashSet::new();
            for slot in slots.iter() {
                assert!(
                    seen.insert(slot.seq_id),
                    "{label}: seq {} duplicated in the slot table",
                    slot.seq_id
                );
            }
            assert_eq!(
                seen.len(),
                TOTAL_SEQ,
                "{label}: seq_ids not conserved (found {}, expected {TOTAL_SEQ})",
                seen.len()
            );

            // A Busy slot must correspond to exactly one live request holding that
            // seq_id, and no two running requests may share one.
            let mut running_seqs: HashSet<i32> = HashSet::new();
            for r in running.iter() {
                if r.kv_seq_id >= 0 {
                    assert!(
                        running_seqs.insert(r.kv_seq_id),
                        "{label}: seq {} claimed by two running requests",
                        r.kv_seq_id
                    );
                }
            }
            for slot in slots.iter() {
                if let SlotState::Busy(req_id) = slot.state {
                    assert!(
                        running
                            .iter()
                            .any(|r| r.id == req_id && r.kv_seq_id == slot.seq_id),
                        "{label}: slot {} is Busy({req_id}) with no matching running request",
                        slot.seq_id
                    );
                } else {
                    assert!(
                        !running_seqs.contains(&slot.seq_id),
                        "{label}: slot {} is not Busy but a running request holds it",
                        slot.seq_id
                    );
                }
            }

            // Every allocated block is reachable from a running request or a slot —
            // nothing dropped on the floor, nothing counted twice.
            let running_blocks: usize = running.iter().map(|r| r.page_table.len()).sum();
            let slot_blocks = slots.charged_blocks();
            assert_eq!(
                kv.allocated_blocks(),
                running_blocks + slot_blocks,
                "{label}: KV block leak — {} allocated but only {} reachable",
                kv.allocated_blocks(),
                running_blocks + slot_blocks
            );

            // Reusable KV can never exceed the number of sequences that exist.
            assert!(
                slots.count(|s| s.state == SlotState::Idle) <= slots.len(),
                "{label}: more idle slots than the table holds"
            );
        };

        let mut observed_hit = false;

        for iter in 0..400usize {
            // 1. Submit one request. 2/3 reuse a small shared-prompt set (so their
            //    prefixes hit once parked); 1/3 are unique (misses → fresh slots).
            let seed = if iter % 3 == 0 {
                100_000 + iter as i32 // unique → miss
            } else {
                (iter % 3) as i32 // {1,2} shared → hit candidate
            };
            let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
            let id = iter as u64 + 1;
            let req = InferenceRequest::new(id, prompt(seed, 2), 8, SamplingParams::default(), tx);
            sched.submit(req).unwrap();

            // 2. Schedule: retires the previous iteration's finishes and admits.
            let hits_pre = sched.prefix_hits.load(Ordering::Relaxed);
            let _batch = sched.schedule_step();
            if sched.prefix_hits.load(Ordering::Relaxed) > hits_pre {
                observed_hit = true;
            }

            // 3. Conservation must hold after every step.
            check_conservation("mid-churn");

            // 4. Finish the oldest running request once a few are in flight, parking
            //    it so its KV becomes reusable — the churn this test exists to drive.
            let finish_id = {
                let mut running = sched.running_batch.lock().unwrap();
                if running.len() >= 3 {
                    let r = &mut running[0];
                    r.state = RequestState::Finished;
                    Some(r.id)
                } else {
                    None
                }
            };
            if let Some(id) = finish_id {
                sched.park_finished(id);
            }
            check_conservation("post-park");
        }

        // The churn must have actually exercised the interesting paths, or the test
        // would be conserving trivially.
        // (Block starvation and the reclaim path are deliberately NOT exercised here —
        // this pool is oversized on purpose so the loop stays live and deterministic.
        // `idle_slot_is_reclaimed_under_block_pressure` covers reclaim directly.)
        assert!(observed_hit, "stress loop never produced a prefix hit");
        assert!(sched.prefix_hits.load(Ordering::Relaxed) > 0);

        // 5. Drain everything and prove nothing leaked: finish all running requests,
        //    let schedule_step retire them, then release every parked slot. Allocation
        //    must fall back to zero with every slot Free.
        {
            let mut running = sched.running_batch.lock().unwrap();
            for r in running.iter_mut() {
                r.state = RequestState::Finished;
            }
        }
        sched.schedule_step();
        {
            let mut slots = sched.slots.lock().unwrap();
            let seq_ids: Vec<i32> = slots.iter().map(|s| s.seq_id).collect();
            for seq_id in seq_ids {
                let blocks = slots.release(seq_id);
                kv.free_blocks(&blocks);
            }
        }
        assert_eq!(
            kv.allocated_blocks(),
            0,
            "KV blocks leaked after full churn + drain"
        );
        let slots = sched.slots.lock().unwrap();
        assert_eq!(slots.len(), TOTAL_SEQ, "slot table lost sequences");
        assert!(
            slots.iter().all(|s| s.state == SlotState::Free),
            "not every slot returned to Free after drain"
        );
    }

    /// Chunked prefill state machine (S1): a request whose prompt is prefilled over
    /// several steps must stay `Prefilling` — and be re-emitted to the prefill batch —
    /// until its cursor reaches the prompt end and the first token is sampled, at which
    /// point it moves to the decode batch. The model (do_prefill) advances the cursor
    /// via `advance_prefill`; here we drive those transitions directly (no FFI).
    #[test]
    fn chunked_prefill_stays_prefilling_until_complete() {
        let config = ModelConfig {
            num_layers: 2,
            num_heads: 2,
            num_heads_kv: 2,
            head_dim: 64,
            n_embd: 128,
            vocab_size: 1000,
        };
        let kv = Arc::new(KVCacheManager::new(&config, 500_000_000, 0.5, 16, 1, 1));
        let sched = Scheduler::new(kv, 8);

        // 48-token prompt = 3 full blocks; chunked prefill would span several steps.
        let prompt: Vec<i32> = (0..48).collect();
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        sched
            .submit(InferenceRequest::new(
                1,
                prompt,
                8,
                SamplingParams::default(),
                tx,
            ))
            .unwrap();

        // Step 1: admitted → emitted to prefill, not decode.
        let b = sched.schedule_step();
        assert_eq!(b.prefill, vec![1], "admitted request must prefill");
        assert!(b.decode.is_empty());

        // Model submitted a non-final chunk (16 of 48 tokens): still Prefilling.
        sched.advance_prefill(1, 16);

        // Step 2: incomplete prefill must be RE-EMITTED to prefill (the whole point —
        // it no longer completes in a single step) and never to decode.
        let b = sched.schedule_step();
        assert_eq!(b.prefill, vec![1], "incomplete prefill must be re-emitted");
        assert!(
            b.decode.is_empty(),
            "must not decode before prefill completes"
        );

        // Final chunk reaches the prompt end and the first token is sampled →
        // update_after_token(from_prefill=true) transitions the request to Decoding.
        sched.advance_prefill(1, 48);
        sched.update_after_token(1, 42, true);

        // Step 3: now Decoding → emitted to decode, no longer to prefill.
        let b = sched.schedule_step();
        assert!(
            b.prefill.is_empty(),
            "completed prefill must not re-emit to prefill"
        );
        assert_eq!(b.decode, vec![1], "completed request must decode");
    }

    #[test]
    fn context_roll_reduces_logical_context_len() {
        // context_len() = prefilled + generated - rolled. A roll must shift the next
        // decode position down by exactly the discarded amount, and rolls accumulate.
        let config = ModelConfig {
            num_layers: 2,
            num_heads: 2,
            num_heads_kv: 2,
            head_dim: 64,
            n_embd: 128,
            vocab_size: 1000,
        };
        let kv = Arc::new(KVCacheManager::new(&config, 500_000_000, 0.5, 16, 1, 1));
        let sched = Scheduler::new(kv, 8);

        // Put a request in the running batch that has "filled" 100 tokens of context.
        {
            let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
            let mut req =
                InferenceRequest::new(7, (0..40).collect(), 200, SamplingParams::default(), tx);
            req.kv_seq_id = 0;
            req.prefilled_tokens = 40;
            req.generated_tokens = 60; // context_len = 100
            req.state = RequestState::Decoding;
            sched.running_batch.lock().unwrap().push(req);
        }

        let ctx_len = |s: &Scheduler| s.get_running(&[7])[0].context_len();
        assert_eq!(ctx_len(&sched), 100);

        // Roll away 30 oldest tokens → live length drops to 70 (next decode pos = 69).
        sched.record_context_roll(7, 30);
        assert_eq!(
            ctx_len(&sched),
            70,
            "rolled tokens subtract from context_len"
        );

        // Generation keeps advancing on top of the reduced length.
        sched.update_after_token(7, 123, false);
        assert_eq!(
            ctx_len(&sched),
            71,
            "generated token adds above the rolled window"
        );

        // A second roll accumulates with the first.
        sched.record_context_roll(7, 20);
        assert_eq!(ctx_len(&sched), 51, "rolls accumulate");
    }
    #[test]
    fn admission_never_preempts_running_requests() {
        // Blocks are fully reserved at admission, so a newcomer that doesn't fit must
        // WAIT — evicting the older running request would discard a generation whose
        // text the client already received (and the pair can evict each other forever:
        // the livelock this test originally exposed).
        let config = ModelConfig {
            num_layers: 2,
            num_heads: 2,
            num_heads_kv: 2,
            head_dim: 64,
            n_embd: 128,
            vocab_size: 1000,
        };
        // A deliberately tiny pool: both requests cannot fit at once.
        let kv = Arc::new(KVCacheManager::new(&config, 200_000, 0.5, 16, 1, 1));
        let total = kv.total_blocks();
        assert!(total >= 4, "pool too small to stage the scenario: {total}");
        let sched = Scheduler::new(kv, 4);

        // Request 1 reserves most of the pool and starts generating.
        let prompt1: Vec<i32> = (0..16).collect();
        let max_new1 = (total - 2) * 16 - prompt1.len();
        let (tx1, _rx1) = tokio::sync::mpsc::unbounded_channel();
        sched
            .submit(InferenceRequest::new(
                1,
                prompt1.clone(),
                max_new1,
                SamplingParams::default(),
                tx1,
            ))
            .unwrap();
        let b = sched.schedule_step();
        assert_eq!(b.prefill, vec![1], "request 1 admitted");
        sched.set_prefilled_tokens(1, prompt1.len());
        for tok in [101, 102, 103] {
            sched.update_after_token(1, tok, tok == 101); // first token completes prefill
        }

        // Request 2 needs more blocks than remain. It must WAIT, not evict request 1.
        let (tx2, _rx2) = tokio::sync::mpsc::unbounded_channel();
        sched
            .submit(InferenceRequest::new(
                2,
                (0..16).collect(),
                32,
                SamplingParams::default(),
                tx2,
            ))
            .unwrap();
        let b = sched.schedule_step();
        assert!(
            b.preempted_seq_ids.is_empty(),
            "admission must never preempt a running request"
        );
        assert_eq!(
            b.decode,
            vec![1],
            "request 1 keeps running while request 2 waits"
        );
        assert_eq!(sched.queue_depth(), 1, "request 2 queued");
        {
            let running = sched.running_batch.lock().unwrap();
            let r1 = running.iter().find(|r| r.id == 1).expect("req 1 running");
            assert_eq!(r1.generated_tokens, 3, "generation state untouched");
        }

        // Once request 1 finishes, request 2 gets its turn.
        sched.mark_finished(1, StopReason::Eos);
        let b = sched.schedule_step();
        assert_eq!(
            b.prefill,
            vec![2],
            "request 2 admitted after request 1 finishes"
        );
    }

    #[test]
    fn oversized_request_is_rejected_synchronously_by_submit() {
        // A request that could never fit even into an EMPTY pool is rejected by
        // `submit()` itself — before it ever enters the queue — so it can never
        // block the queue head. This used to be a schedule_step()-time check; 0.16
        // moved it to submit() since it's statically knowable at submission time.
        let config = ModelConfig {
            num_layers: 2,
            num_heads: 2,
            num_heads_kv: 2,
            head_dim: 64,
            n_embd: 128,
            vocab_size: 1000,
        };
        let kv = Arc::new(KVCacheManager::new(&config, 200_000, 0.5, 16, 1, 1));
        let total = kv.total_blocks();
        let sched = Scheduler::new(kv, 4);

        // Oversized: needs more blocks than the entire pool.
        let (tx1, _rx1) = tokio::sync::mpsc::unbounded_channel();
        let err = sched
            .submit(InferenceRequest::new(
                1,
                (0..16).collect(),
                (total + 2) * 16,
                SamplingParams::default(),
                tx1,
            ))
            .expect_err("oversized request must be rejected by submit(), not queued");
        assert!(matches!(err, SubmitError::TooLarge { .. }));
        assert_eq!(
            sched.queue_depth(),
            0,
            "rejected request never entered the queue"
        );

        // A normal request submits and schedules fine.
        let (tx2, _rx2) = tokio::sync::mpsc::unbounded_channel();
        sched
            .submit(InferenceRequest::new(
                2,
                (0..16).collect(),
                16,
                SamplingParams::default(),
                tx2,
            ))
            .unwrap();

        let b = sched.schedule_step();
        assert_eq!(b.prefill, vec![2], "the normal request is admitted");
        assert_eq!(sched.queue_depth(), 0, "nothing left waiting");
        assert_eq!(sched.active_requests(), 1, "only the normal request runs");
    }

    #[test]
    fn submit_rejects_when_queue_full() {
        let config = ModelConfig {
            num_layers: 2,
            num_heads: 2,
            num_heads_kv: 2,
            head_dim: 64,
            n_embd: 128,
            vocab_size: 1000,
        };
        let kv = Arc::new(KVCacheManager::new(&config, 500_000_000, 0.5, 16, 1, 1));
        // max_batch_size=1 keeps the seq_id pool tiny so nothing gets admitted off
        // the queue between the two submits; max_queue_depth=1 caps the queue itself.
        let sched = Scheduler::with_max_queue_depth(kv, 1, 1);

        let (tx1, _rx1) = tokio::sync::mpsc::unbounded_channel();
        sched
            .submit(InferenceRequest::new(
                1,
                (0..16).collect(),
                8,
                SamplingParams::default(),
                tx1,
            ))
            .expect("first submit fills the queue to its cap, should succeed");
        assert_eq!(sched.queue_depth(), 1);

        let (tx2, _rx2) = tokio::sync::mpsc::unbounded_channel();
        let err = sched
            .submit(InferenceRequest::new(
                2,
                (0..16).collect(),
                8,
                SamplingParams::default(),
                tx2,
            ))
            .expect_err("second submit must be rejected — queue is at max_queue_depth");
        assert!(matches!(err, SubmitError::QueueFull { depth: 1, max: 1 }));
        assert_eq!(
            sched.queue_depth(),
            1,
            "rejected request never entered the queue"
        );
    }

    #[test]
    fn submit_unbounded_by_default() {
        // max_queue_depth=0 (the default via `Scheduler::new`) means unbounded — this
        // guards against accidentally changing the default and silently capping every
        // existing deployment's queue.
        let config = ModelConfig {
            num_layers: 2,
            num_heads: 2,
            num_heads_kv: 2,
            head_dim: 64,
            n_embd: 128,
            vocab_size: 1000,
        };
        let kv = Arc::new(KVCacheManager::new(&config, 500_000_000, 0.5, 16, 1, 1));
        let sched = Scheduler::new(kv, 1);
        for id in 1..=50u64 {
            let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
            sched
                .submit(InferenceRequest::new(
                    id,
                    (0..16).collect(),
                    8,
                    SamplingParams::default(),
                    tx,
                ))
                .expect("unbounded queue must accept many more requests than the batch size");
        }
        assert_eq!(sched.queue_depth(), 50);
    }
}
