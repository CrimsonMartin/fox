# fox vs Ollama on the Radeon 890M — ROCm benchmarking (2026-08-01)

Status: **root cause found and fixed — fox now matches/beats Ollama on this
benchmark.** Two independent bugs in fox's own code (not llama.cpp, not the
GPU) combined to explain nearly the whole gap: an expensive default sampling
path, and — the bigger one — a `seq_id` allocation/ordering bug that
silently defeated llama.cpp's own multi-sequence batching. Fixed throughput:
**~46-52 t/s → ~122-146 t/s** on the standard benchmark, against Ollama's
~110-148 t/s (run-to-run) and vanilla `llama-server`'s 173 t/s. An earlier
pass through this investigation wrongly concluded the gap was a structural
llama.cpp/ggml-cuda kernel limit; that conclusion is **retracted** below,
with the evidence that overturned it, since it was briefly committed to
`vllm-gap-analysis.md`/`STATUS.md` before being corrected. Related:
[`engine-capabilities-checklist.md`](engine-capabilities-checklist.md)'s
target-machine section, [`vllm-gap-analysis.md`](vllm-gap-analysis.md) §1.
A follow-up caveat — that the fix held only for low-prefix-cache-reuse
traffic, and degraded back toward baseline under a shared system prompt —
was **also fixed since** (2026-08-01, third pass), by switching to a
unified KV cache: **median 158.2 t/s, range [155.0, 158.9]**, above
Ollama's 144-155 and at ~91% of `llama-server`. See "Known limitation" (now
resolved), "Attempted fix" (the ruled-out repair), and "The fix that
actually closed it: `kv_unified`" below.

## Why this exists

The stated goal for this machine (AMD Ryzen + Radeon 890M) is for fox to beat
both Ollama and vLLM on it, not just be API-compatible with them. vLLM doesn't
run here at all (no NVIDIA GPU). Ollama does, so it's the real competitor —
this doc is a head-to-head throughput investigation against it, same
hardware, same GGUF weights, same underlying llama.cpp lineage.

## Setup

Both engines run in Docker (host stayed unmodified — no ROCm/Vulkan dev
packages installed outside containers):

- **fox:vulkan** — `Dockerfile.vulkan` (already existed), Ubuntu 24.04 +
  `glslc`/Vulkan dev headers at build time.
- **fox:rocm** — new `Dockerfile.rocm`, `rocm/dev-ubuntu-24.04` base so
  `hipcc`/clang are present at build time. Pinned to ROCm 7.2 (see Results).
- **ollama:rocm** — official `ollama/ollama:rocm` image.
- **llamaserver-bench** — a throwaway image (never committed, built from a
  scratch Dockerfile) compiling vanilla `llama.cpp`'s own `llama-server`
  binary from the **exact same vendored commit fox uses**, with ROCm/HIP —
  used to isolate "is this fox's code or llama.cpp's" (see below). Not part
  of the repo.

The 890M is `gfx1150` (RDNA 3.5), not in ROCm's supported-device list for
any of these engines. All three need `HSA_OVERRIDE_GFX_VERSION=11.0.0` to
make the driver present it as `gfx1100` (a supported RDNA3 desktop chip);
Ollama additionally needs `OLLAMA_IGPU_ENABLE=1` (it has its own iGPU
drop-guard). Fox's ROCm build targets `gfx1100` at compile time to match
(`AMDGPU_TARGETS=gfx1100`).

All engines serve the **exact same GGUF** (`llama-3.2-1b-instruct-q8_0`,
imported into Ollama via `docker cp` + a minimal `Modelfile` rather than
letting it pull its own copy) — isolates serving-engine differences from
model/quant differences.

`fox-bench` (`--concurrency 4 --requests 40 --max-tokens 256` for the
sustained-load numbers below, smaller for early exploratory runs) drove
every comparison.

## build.rs changes (kept)

1. **`AMDGPU_TARGETS` passthrough** — `build.rs` never forwarded a GPU
   architecture to CMake for the HIP backend. CMake's HIP language
   auto-detects the target by probing a visible GPU at *build* time, which
   doesn't exist in a build container. Added: if the `AMDGPU_TARGETS` env var
   is set, forward it as a CMake define. Without this, the ROCm build fails
   to configure at all inside Docker.
2. No other build.rs changes. `GGML_HIP_MMQ_MFMA` was considered
   (Ollama-parity flag) but **ruled out by reading the vendor source**: it's
   gated behind `defined(CDNA)` (`ggml-cuda/common.cuh`) — RDNA3/gfx11 never
   takes that branch, so it's a no-op for this GPU regardless of what fox
   sets.

`Dockerfile.rocm` (new) mirrors `Dockerfile.vulkan`'s two-stage structure;
its header comment has the exact build/run commands, including the
`--device /dev/kfd --device /dev/dri --group-add video --group-add <render-gid>`
passthrough (get the render GID via `getent group render` — passing the
group by *name* fails inside minimal containers that lack that `/etc/group`
entry). It also needed `hipblas-dev`/`rocblas-dev` explicitly (not just the
runtime `hipblas`/`rocblas` packages) — without them, `libggml-hip.so`
silently fails to dlopen and the server falls back to CPU with no error.

## Results — theories tested and ruled out

| Comparison | fox | Other | Notes |
|---|---|---|---|
| fox-ROCm vs fox-Vulkan (conc=4) | **50.7 t/s** | 43.0 t/s (Vulkan) | ROCm backend is a real, if modest, win over Vulkan for fox itself |
| fox-ROCm vs Ollama-ROCm (conc=4, `OLLAMA_NUM_PARALLEL=4`) | 48-52 t/s | **93.8-136.8 t/s** | the gap that mattered (run-to-run variance in this range; not yet root-caused at this point) |
| fox-ROCm vs Ollama-ROCm (conc=1, solo) | 44.7 t/s | 48.6 t/s | **parity** — same llama.cpp core, same per-token speed alone |
| fox-ROCm, `--max-context-len 4096` vs default (131072, model's trained ctx) | 44.3 t/s | — | **no improvement** — ruled out KV-cache-size/memory-bandwidth theory |
| fox-ROCm pinned to **ROCm 7.2** (matches Ollama's bundled `libamdhip64.so.7.2`) vs `rocm/dev-ubuntu-24.04:latest` (ROCm 7.14, bleeding-edge) | 48.8 t/s | — | **no improvement** — ruled out ROCm library version |
| Vanilla **llama-server** (same llama.cpp commit as fox, no fox/Ollama) | — | **173.0 t/s** | **the decisive test — see below** |
| fox-ROCm, after the sampling fix (this doc's actual finding) | **58-66 t/s** | 122-137 t/s (Ollama), 173 t/s (llama-server) | real +25-40% improvement, gap not fully closed |

Theories tested and ruled out, in the order investigated:

- **Backend choice (Vulkan vs ROCm)** — ROCm is ~15% faster than Vulkan for
  fox alone, but that's a fraction of the ~2x gap vs Ollama. Not the main
  cause.
- **KV-cache footprint** (fox defaults to the model's full trained context —
  131072 tokens — when `--max-context-len` isn't passed) — forcing
  `--max-context-len 4096` didn't move throughput. Not the cause.
- **ROCm library version** — pinning to Ollama's exact ROCm 7.2 gave a
  statistically identical result to the original ROCm 7.14 build. Not the
  cause. (`Dockerfile.rocm` stays pinned to 7.2 anyway, for a cleaner
  apples-to-apples comparison going forward.)
- **A newer llama.cpp version/patches than fox's vendored commit** — Ollama
  pins `b10091` (2026-07-22), ~3-4 weeks ahead of fox's vendored `6f4f53f2b`
  (2026-06-29), and applies no patches to the batching/KV-cache files.
  Diffed the relevant files directly between the two commits: no
  behavior-relevant difference. Not the cause.
- **Scheduler admission/pipelining** — does fox eagerly re-fill a freed
  decode slot, or does it lose a step? Measured directly via a temporary
  per-tick log (`prefill_ids.len()`/`decode_ids.len()`, reverted after use):
  realized decode batch width averages **~3.85 of a possible 4**, identical
  on CPU and ROCm. The scheduler hands the model a near-full batch on
  almost every step. Not the cause.
- **llama.cpp's ubatch-splitter selection** (`split_equal` vs
  `split_simple`) — confirmed via a temporary one-line `fprintf` in the
  vendored `llama-kv-cache.cpp` (reverted, never committed — this repo's
  submodule convention is to never commit vendor changes) that the correct,
  fully-batching `split_equal` path runs, not `split_simple`. An earlier
  `LLAMA_BATCH_DEBUG=1` capture that seemed to show `split_simple` running
  was a misread of a different, unrelated debug line (a `graph_reserve`
  warm-up ubatch, not the real decode ubatch). Not the cause.
- **GPU kernel dispatch width** (`rocprofv3` profiling) — a short benchmark
  (8 requests, 64 tokens) showed 94.5% of GPU time in single-token
  (`ncols_dst=1`) matmul kernels. A longer, more realistic capture (40
  requests, 256 tokens) found a *mix* instead — `ncols_dst=1: 47.5%,
  2: 37.5%, 3: 8.4%, 4: ~1.1%`, weighted average **~1.64 of 4** — which at
  the time was read as "not a wall of 1s, so probably just natural
  completion-timing variance, not a real bug." **That reading was itself
  wrong — this measurement was the earliest direct evidence of the actual
  seq_id-ordering bug** (see below), just not recognized as such until
  later. `ncols_dst<4` most of the time is exactly what a `seq_id`-ordering
  failure in llama.cpp's `split_equal` looks like from the outside: some
  ticks luck into an ascending-enough ID assignment to batch several
  sequences together (2-3), others don't and collapse to `1`, depending on
  the essentially-random order a LIFO pool happened to hand out IDs in that
  moment — not the "one prefill step of staggering" explanation this doc
  originally gave it.

## The wrong turn: "closed as structural" (retracted)

At this point in the investigation, with scheduling, splitting, and batch
width all ruled out or explained, the ~1.64-vs-~3.85 mismatch between what
the scheduler hands the model and what the GPU kernel actually executes was
concluded to be an llama.cpp/ggml-cuda kernel limitation — "the same 'fox
rides llama.cpp's kernels' ceiling as PagedAttention/FlashAttention" — and
was briefly committed as **closed, structural, not fox backlog** in
`vllm-gap-analysis.md` and `STATUS.md`.

**This was wrong, and a single test proved it**: compiling and running
vanilla `llama-server` — the same llama.cpp source, same vendored commit
fox uses, same GPU, no fox code and no Ollama Go layer at all — on the
identical benchmark. Result: **173.0 t/s**, not only far above fox's ~50
t/s but *also* above Ollama's own ~122 t/s. If the ggml-cuda kernel itself
couldn't sustain wide batching, `llama-server` built from the *exact same
kernel source* couldn't have hit 173 t/s either. The kernel is fine; the
bottleneck was always going to be found in code fox actually owns. The
"closed as structural" edits were reverted in both files.

## The real root cause: fox's own sampling defaults

Timing instrumentation added directly inside `do_decode_batch`
(`src/engine/model/llama_cpp/batch.rs`, all temporary and since removed)
isolated where the ~76-88ms average per-tick cost (4-wide batch) actually
went:

| Stage | Cost |
|---|---|
| `ffi::llama_decode` call itself | ~44-50ms |
| Per-request `sample_constrained`/`sample_token` | ~3.6-4.6ms **each**, ×4 |
| Everything else (post-processing, streaming) | ~0.4ms |

The per-request sampling cost was the first surprise — `sample_token`
(`src/engine/model/sampling.rs`) should be a cheap top-k/softmax operation,
not several milliseconds. Further breakdown pinpointed it: `k` (the
resolved `top_k`) was **`0`** on every single sample. From
`src/api/shared/sampling_defaults.rs`:

```rust
/// Disabled — OpenAI exposes no `top_k`. `0` means "off" in the sampler.
pub const TOP_K: u32 = 0;
```

This is a **deliberate** fox design choice (matching real OpenAI's API,
which has no `top_k` parameter at all) — not a bug in the sense of "wrong
value," but one with a severe, unanticipated performance cost: with `top_k`
disabled, `sample_token` computed `exp()` over the **entire ~128,256-token
vocabulary twice** (once for the normalizing sum, once building the
probability vector) and then **fully sorted** all 128,256 entries just to
apply `top_p`/`min_p` truncation — every single generated token, every
request. Ollama and `llama-server` never hit this path: they default
`top_k = 40` **regardless of API surface** (confirmed via `llama-server`'s
own `/props` endpoint), so real OpenAI-API-shaped requests against them
still get the cheap top-40 path fox's OpenAI surface deliberately opts out
of.

## The fix

Two changes in `src/engine/model/sampling.rs` and its call sites, both
behavior-preserving (same output distribution, no default changed):

1. **Adaptive candidate selection** (`sample_token`) — when `top_k` is
   disabled, instead of sorting/exponentiating the full vocab, adaptively
   grow a by-logit candidate pool (64 → 256 → 1024 → … via
   `select_nth_unstable_by`, an O(n)-average partition) until it provably
   contains enough of the distribution to make `top_p`/`min_p` truncation
   give the *exact same result* as sorting everything — falling back to the
   full vocab only if a request's parameters genuinely need it (e.g.
   `top_p` at/near `1.0`). `max_l`/`exp_sum` are still computed with one
   full linear pass each (unavoidable — that's the real normalizer), but the
   expensive full sort and the *second* full `exp()` pass are gone in the
   common case. When `top_k > 0`, the same partition primitive already
   finds the top-k set directly in O(n) average instead of a full sort.
2. **Skip the full-vocab logits copy when nothing reads it** — a new
   `needs_logits: bool` on `InferenceRequestForModel`
   (`src/engine/model/mod.rs`), set from `r.sampling.logprobs.is_some()`
   (`src/engine/run.rs`, both `run_prefill`/`run_decode`), gates the
   `logits_slice.to_vec()` copy in `do_decode_batch`/`do_prefill_batch` —
   only OpenAI `logprobs` ever reads `Logits.values`, so the ~513KB copy
   is now skipped whenever a request doesn't ask for it (the common case).

Correctness verified: all existing `sampling::` unit tests pass unchanged
(including `sample_token_top_p_restricts_candidates` and
`min_p_keeps_only_dominant_token`, which directly exercise the new adaptive
path), all 12 golden tests (real model) pass, the full stub integration
suite (40 tests, including `test_v1_chat_logprobs_present_when_requested`
and `test_v1_chat_logprobs_absent_by_default`) passes, and a live spot-check
with `logprobs: true` against the running ROCm build returned a coherent
completion with populated `logprobs`/`top_logprobs`.

## The second, bigger root cause: `seq_id` allocation order

The sampling fix alone (~58-66 t/s) still left fox at roughly half of
Ollama's throughput. Further instrumentation inside `do_decode_batch`
found `ffi::llama_decode` itself still costing ~44-50ms per 4-wide call
even after the sampling fix — and reading `llama-context.cpp` explained why
that number is misleading: `llama_decode` launches GPU compute
**asynchronously** (`ggml_backend_sched_graph_compute_async`) and returns
before the GPU finishes; the actual completion wait happens later, inside
`llama_get_logits_ith` (`ctx->synchronize()`). Timing both call sites
directly showed the *first* `llama_get_logits_ith` call per tick
consistently took ~17-18ms (the real GPU wait — plausibly close to what
`llama-server`'s 173 t/s implies is needed here), leaving **~40ms of
synchronous CPU-side cost inside `llama_decode` itself** unexplained.
Capping `n_batch`/`n_ubatch`/`n_seq_max` to match `llama-server`'s exact
launch config ruled out context/batch sizing as the cause.

**The real explanation was upstream of all of this**: this doc's own
earlier `rocprofv3` profiling (see "Results" above) had already found that
GPU kernel dispatch averaged only `ncols_dst≈1.64` of a possible 4 — and at
the time, that was (wrongly) chalked up to normal batch-width variance from
requests finishing at different times. It wasn't. The real cause: fox's
`seq_id_pool` (`src/scheduler/mod.rs`) was a plain `Vec`-backed LIFO stack,
handing out whatever ID was pushed back most recently — and
`do_decode_batch` (`src/engine/model/llama_cpp/batch.rs`) emitted the
`llama_batch` in **scheduler-admission order**, not seq_id order. But
llama.cpp's `split_equal` (the splitter used whenever the KV cache is
non-unified — i.e. always, here, since `n_stream = n_seq_max > 1`) is
called with `sequential=true`, and only keeps growing the *same* ubatch
while walking the batch as long as
`batch.seq_id[i][0] == last_seq_id + 1` (`llama-batch.cpp`) — **strictly
consecutive, strictly increasing, in emission order.** Four genuinely
concurrent requests with scattered or out-of-order seq_ids (entirely
possible with a LIFO pool and admission-order emission) silently fail this
check and get split into four separate 1-token ubatches — real GEMV-level
serialization, invisible from fox's own scheduler-level metrics (which only
see "4 requests decoding this step," never how llama.cpp's splitter grouped
them). This is the mechanism the whole `ncols_dst` investigation earlier in
this doc was circling without landing on.

Fixed in two places, both required together:

1. `src/scheduler/mod.rs`: `seq_id_pool` is now a `BinaryHeap<Reverse<i32>>`
   min-heap instead of a `Vec` stack — always hands out the *lowest* free
   ID, so N concurrent requests occupy IDs `0..N-1` densely (mirroring how
   `llama-server` assigns `slot.id = i`).
2. `src/engine/model/llama_cpp/batch.rs` (`do_decode_batch`): the
   `llama_batch` is now emitted in **ascending `kv_seq_id` order**, not
   scheduler-admission order — a dense seq_id pool alone isn't sufficient if
   the batch itself isn't walked in that order. Logits are read back via an
   inverse index (`slot_of`) so the returned per-request order is
   unaffected by the emission reordering.

## Measured result

| Metric (concurrency=4, 40 requests, 256 max tokens) | Baseline | +sampling fix | +seq_id fix |
|---|---|---|---|
| Throughput | ~46-52 t/s | ~58-66 t/s | **~122-146 t/s** |
| vs Ollama (~110-148 t/s, run-to-run) | ~half | ~half | **on par, sometimes ahead** |
| vs `llama-server` native (173 t/s) | ~30% | ~35% | **~70-85%** |

Both fixes were needed — the sampling fix alone left fox at roughly half of
Ollama; the seq_id fix on top of it took fox from "roughly half of Ollama"
to "matching or beating Ollama" on this exact benchmark, run twice to guard
against a fluke (122.0 t/s and 146.3 t/s, both far above every pre-fix
number recorded in this doc, both 40/40 requests with zero errors on fox's
side). The remaining gap to `llama-server`'s 173 t/s is unexplored but far
smaller than what this investigation started with, and no longer suggests
an upstream/structural ceiling — fox's own request-lifecycle overhead
(scheduling, HTTP/streaming layers) is the more likely remaining source,
not a fundamental limitation.

## Known limitation (RESOLVED — see "The fix that actually closed it" below): the seq_id fix degrades under heavy prefix-cache reuse

The ad-hoc single-shot numbers above (fresh container, one comparison, then
torn down) turned out to be an optimistic case. Built `scripts/repeat_bench.sh`
(committed — see below) to properly benchmark with warmup, multiple
repetitions, alternating engine order, and automatic discarding of runs with
request errors, specifically because single ad-hoc runs on this hardware
showed too much run-to-run variance to trust. Running it for 5 sustained
repetitions against the **same long-lived fox container** (rather than a
fresh one per comparison) surfaced a real, reproducible degradation the
one-shot numbers missed entirely: throughput settled at a rock-stable
**~52.7 t/s** (range 52.6-52.9 across all 5 repetitions) — back down to the
pre-seq_id-fix baseline, while Ollama stayed at 144-155 t/s the whole time.

Root cause, confirmed via `docker logs | grep seq_id` and `/metrics`: **not**
a resource leak (`ferrumox_kv_cache_usage_ratio` sits at ~0.4% at idle —
blocks are freed correctly). The seq_id min-heap and ascending-emission-order
fix above guarantees dense, consecutive IDs only for requests that get a
**fresh pool pop** (a genuine prefix-cache miss). But `try_insert_prefix`'s
block-level prefix cache donates a finished request's **existing** seq_id to
the cache entry, and a future cache **hit** (`schedule.rs`'s admission path,
`req.kv_seq_id = hit.seq_id`) inherits that donated ID as-is — not a fresh
pop from the ascending pool. Every chat request shares the same first ~16
tokens (BOS + role-header boilerplate from the chat template) regardless of
user content, so in practice almost every request hits this shared-header
cache entry and inherits whatever seq_id was last donated to it — which
drifts to arbitrary, non-consecutive values (observed: IDs cycling at 29 and
31, with nothing below 29 admitted for extended stretches) as the cache
churns. llama.cpp's `split_equal` requires **strictly** consecutive
`+1`-incrementing seq_ids to merge sequences into one ubatch — sorting
ascending (which `do_decode_batch` already does) cannot repair a gap; only a
genuinely dense set of IDs can satisfy it. A set like `{0, 1, 29, 31}` still
splits into multiple ubatches even sorted.

**This means the fix's real-world benefit is workload-dependent**: it fully
holds for traffic with low prefix-cache reuse (varied prompts/no shared
system prompt), which is what the earlier one-shot benchmarks happened to
exercise (short runs, cache still mostly cold). It degrades toward the
pre-fix baseline under heavy reuse — which, notably, is not just a synthetic
benchmark artifact: **any real deployment where multiple concurrent
conversations share a common system prompt hits this same pattern**, since
that shared prefix is exactly the kind of content the block-level cache is
designed to reuse.

**Not fixed here** — see the next section: the obvious fix (migrate via
`llama_memory_seq_cp`) was attempted and **does not work** — it crashes the
server. A real fix needs a different mechanism.

## Attempted fix: migrate cache-hit requests to a fresh seq_id via `llama_memory_seq_cp` — crashes, reverted

The natural fix for the limitation above is to give a prefix-cache-hit
request a **fresh**, densely-allocated `seq_id` (via `Scheduler::
try_pop_fresh_seq_id`) instead of the donated one, copying the cached
prefix's KV data across with `llama_memory_seq_cp` before this step's
prefill/decode runs — mirroring how `Model::roll_context` already does FFI
work at the engine layer. This was fully implemented (new
`batch::PrefixHitMigration`, `Scheduler::try_pop_fresh_seq_id`/
`finalize_seq_migration`, a migration loop in `run_loop` calling the
already-existing `Model::copy_sequence_range`) and passed `cargo fmt`/
`clippy`/the full stub test suite, including a new unit test exercising the
migration end-to-end at the scheduler level.

**It crashed the server on the very first prefix-cache hit** when validated
against a real ROCm build under `scripts/repeat_bench.sh`'s sustained-load
test (the same test that surfaced the original limitation):

```
/app/vendor/llama.cpp/src/llama-kv-cache.cpp:518: GGML_ASSERT(is_full && "seq_cp() is only supported for full KV buffers") failed
```

Root cause, confirmed by reading `vendor/llama.cpp/src/llama-kv-cache.cpp`
directly (`llama_kv_cache::seq_cp`, ~line 463): fox's KV cache is
**non-unified** (`n_stream = n_seq_max > 1`, one stream per seq_id — this is
the same setting the seq_id-ordering fix above depends on for
`split_equal` to batch concurrent sequences at all). When the source and
destination seq_ids live in different streams (`s0 != s1` — true for any
migration to a genuinely different seq_id, by construction), `seq_cp`
takes the "cross-stream" path, which only supports copying the **entire**
KV buffer (`p0`/`p1` must span `[0, get_size())`, i.e. the full `n_ctx`,
not just the `cached_tokens` prefix) — `GGML_ASSERT(is_full)` rejects
anything narrower, including the exact "copy just the cached prefix"
partial range this fix needs. The same-stream fast path (cheap metadata-only
remap, no assert) exists but is unreachable here: two different seq_ids in
a non-unified cache are never in the same stream by construction, so a
migration to a fresh id always takes the cross-stream path. This is not a
fox bug or a version-specific quirk — it's how `seq_cp` is documented to
behave for split/non-unified caches in fox's vendored llama.cpp commit, and
the comment `// TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]` right above it
suggests upstream is aware partial cross-stream copies aren't supported yet
either. Confirmed via the "verify against upstream before calling something
structural" lesson from this doc's own earlier retraction — this time the
upstream check confirmed the limitation is real, not a fox misuse.

**Reverted in full** (`src/scheduler/batch.rs`, `src/scheduler/mod.rs`,
`src/scheduler/schedule.rs`, `src/engine/run.rs`) — the crash makes this
strictly worse than the known limitation it tried to fix, so nothing from
this attempt should ship. `Model::copy_sequence_range`/`supports_seq_copy`
themselves are untouched and still exist as a capability probe only (as
before this attempt); they should not be invoked as a real per-request KV
migration mechanism against fox's current non-unified KV cache without a
different approach to the copy itself.

**What a real fix would need instead** (not attempted yet): since
`llama_memory_seq_cp` cannot do a partial cross-stream copy, options are
(a) avoid the copy entirely — treat a prefix-cache hit whose donated
`seq_id` falls outside the pool's current dense/low range as a miss instead
(recompute the ~16-token prefix rather than reuse it), trading a small,
bounded amount of recompute for keeping seq_ids dense; (b) a full-buffer
`seq_cp` (accepting the cost of copying the entire per-stream KV buffer,
not just the cached prefix) if that cost turns out to be acceptable at this
model's context size — unmeasured, and likely not acceptable at large
`n_ctx`; (c) switch to a unified KV cache (`n_stream = 1`) for the
non-batched dimension, if `split_equal`'s consecutive-seq_id requirement
turns out not to actually need `n_stream > 1` — unconfirmed, needs reading
`llama-batch.cpp`/`llama-kv-cache.cpp` more closely than this session did.
None of these were evaluated in depth; this is a genuine open design
problem, not a known-good approach waiting to be typed in.

## The fix that actually closed it: `kv_unified` (2026-08-01, third pass)

Alternative (c) above turned out to be correct, and its stated uncertainty
("if `split_equal`'s consecutive-seq_id requirement turns out not to
actually need `n_stream > 1`") resolves cleanly by reading the two files it
names. The requirement isn't a property of `split_equal` that fox must
satisfy — it's a property of *which splitter runs at all*:

```cpp
// llama-kv-cache.cpp, llama_kv_cache::init_batch
auto ubatch = n_stream == 1 ? balloc.split_simple(n_ubatch)
                            : balloc.split_equal(n_ubatch, true);
```

`n_stream` is `unified ? 1 : n_seq_max`. So a unified KV cache doesn't make
fox *satisfy* the consecutive-ID rule — it takes the code path where the
rule does not exist. `split_simple` has no equivalent of `split_equal`'s
`batch.seq_id[i][0] == last_seq_id + 1` guard, so it folds the whole decode
batch into one ubatch regardless of which IDs the scheduler happens to
hold. Prefix-cache hits can donate whatever `seq_id` they like.

**The change** is two lines: `ctx_params.kv_unified = true` in both
`LlamaCppModel::load()` and `::new_context()`
(`src/engine/model/llama_cpp/mod.rs`).

### Measuring it without patching the vendor

Verifying this by throughput alone is hopeless on this hardware — the
run-to-run spread swamps the effect (one config measured [72.3, 154.6] t/s
across 5 repetitions). So this pass measured the *mechanism* instead: the
actual width of each decode ubatch, which is deterministic and needs one
short run.

llama.cpp already traces this under `LLAMA_BATCH_DEBUG=1`, but fox installs
a `noop_log` callback that drops llama.cpp's log entirely — which is why
earlier passes resorted to patching the vendored source and rebuilding.
Added instead: **`FOX_LLAMA_LOG=1` forwards llama.cpp's log to stderr**
(same file, next to `noop_log`). Combined:

```bash
LLAMA_BATCH_DEBUG=1 FOX_LLAMA_LOG=1 fox serve --model-path <model.gguf>
# then, from the log:  grep -A4 'equal_seqs   = 0' | grep n_tokens
```

No submodule edits, nothing to remember to revert.

### Result

Same protocol both times: one warmup round, then 3 sustained rounds of
`fox-bench --concurrency 4 --requests 40 --max-tokens 128` against one
long-lived server (the sustained-load shape that surfaced the limitation in
the first place), counting decode ubatch widths.

| Decode ubatch width | Before (non-unified) | After (`kv_unified`) |
|---|---|---|
| 1 token | 5498 | 46 |
| 2 | 1257 | 116 |
| 3 | 379 | 410 |
| 4 (full) | 1444 | 7070 |
| **weighted average** | **1.74 / 4** | **3.90 / 4** |

The before-column's 1.74 independently reproduces the ~1.64 that
`rocprofv3` measured at the kernel level in the first pass — same
fragmentation, measured two unrelated ways. After the change, **zero**
ubatches take the `split_equal` path (`equal_seqs = 1` count: 0),
confirming the splitter switch rather than inferring it.

ROCm throughput (`scripts/repeat_bench.sh`, 5 repetitions, same container
lifetime), for the target machine:

| | median | range |
|---|---|---|
| Before | 110.9 t/s | [72.3, 154.6] |
| After | **158.2 t/s** | **[155.0, 158.9]** |

The collapsing range matters as much as the median: the wild run-to-run
variance previously attributed to thermal/iGPU noise was substantially this
fragmentation, appearing or not depending on which seq_ids the prefix cache
happened to be recycling. fox is now above Ollama's 144-155 t/s on this
benchmark and at ~91% of vanilla `llama-server`'s 173 t/s.

### Cost

**No extra memory.** The KV allocation is identical either way — only its
shape changes (`llama_kv_cache: size` line, same model, `--max-context-len 4096`):

| | total | cells | seqs/streams |
|---|---|---|---|
| Before | 4224.00 MiB | 4096 (per stream) | 33/33 |
| After | 4224.00 MiB | 135168 (shared) | 33/1 |

The shared pool is in fact strictly more flexible: a single long
conversation can exceed the per-stream ceiling when other slots are idle,
which the non-unified layout cannot do.

Verified with `make e2e` (22 passed, 0 failed — including 4-way concurrency,
context rolling past `n_ctx`, embeddings, and mid-stream disconnect) plus
the full stub suite and `cargo test --release --lib` (347 tests).

**Side note**: this also retires the crashing `llama_memory_seq_cp`
migration from the previous section as unnecessary — and would have
unblocked it anyway, since the `GGML_ASSERT(is_full)` that killed it only
guards the *cross-stream* path; with `n_stream = 1` every `seq_cp` is
same-stream, where partial ranges are supported.

**Not verified**: whether `kv_unified` interacts badly with models fox
hasn't exercised here (SWA/sliding-window architectures, recurrent/hybrid
models). The e2e suite and golden tests only cover Llama-family GGUFs on
this machine.

## `n_batch`/`n_ubatch` experiment (tried, reverted, unrelated finding)

Separately from the residual above, capping `n_batch`/`n_ubatch` at 2048
**alone** (without `--max-batch-size`) was tried earlier as its own
experiment and gave a real, if modest, **+7-13%** (54.4 t/s vs the
then-baseline 48-51 t/s) — reverted anyway, because it has a correctness
risk: `do_get_embeddings` (`batch.rs`) submits an embedding request's
**entire** input as one unchunked `llama_decode` call, and for a
non-causal (encoder-style, BERT-family) embedding model,
`llama-context.cpp` asserts `n_ubatch >= n_tokens` — an embedding input
longer than 2048 tokens on such a model would crash where today (uncapped
`n_batch`) it doesn't. Gating this safely needs either new FFI plumbing
(llama.cpp exposes no way to query causal/non-causal before creating a
context) or a GGUF-metadata heuristic — real work, not obviously justified
by 7-13% alone. If someone wants this independently of the residual above,
the code to restore is: cap `n_batch`/`n_ubatch` at
`effective_max_ctx.min(2048).max(max_batch_size)` in both
`LlamaCppModel::load()` and `::new_context()`
(`src/engine/model/llama_cpp/mod.rs`), gated on the model being causal.

## What's next

1. ~~**Highest priority**: fix the prefix-cache/seq_id interaction.~~
   **Done** — alternative (c), the unified KV cache, closed it; see "The fix
   that actually closed it: `kv_unified`" above. The other two alternatives
   (skip-cache-on-stale-id, full-buffer copy) are moot and were never
   implemented.
2. Close the remaining gap to `llama-server`'s 173 t/s (fox is at ~91% of it
   now, up from ~30-35% before these fixes, and now *under sustained
   prefix-cache reuse*, not just the easy case) — no longer suggests an
   upstream ceiling, so this is exploratory rather than a known lead: fox's
   own request-lifecycle overhead (HTTP/streaming layers, scheduler tick
   overhead) is the more likely place to look next, not llama.cpp internals.
   Note the decode ubatch width is now 3.90/4, not 4.00 — the residual is
   the expected prefill-step gap when a finished request is replaced, so
   there is little left to win at the batching layer specifically.
3. Profile Ollama/`llama-server` itself with `rocprofv3` (its minimal image
   ships no profiling tools — would need building or copying in a
   profiling-capable image) to directly confirm it's hitting `ncols_dst=4`
   consistently, as a sanity check against fox's now-fixed dispatch pattern.
4. Re-run a `rocprofv3` capture on fox's own fixed build to directly
   confirm `ncols_dst=4` now dominates (this doc's fix is verified via
   aggregate throughput; a kernel-level reconfirmation would be the last
   piece of direct evidence, blocked this session by the ROCm runtime image
   missing `libdw.so.1`, a `rocprofv3` dependency not installed in
   `Dockerfile.rocm`'s minimal runtime stage).
5. The independent 7-13% `n_batch`/`n_ubatch` win in the section above, if
   someone builds the causal/non-causal model detection it needs.

## Benchmarking methodology: use `scripts/repeat_bench.sh`, not one-off runs

Single ad-hoc `fox-bench` invocations (fresh container, one comparison) are
what produced every number in this doc until the prefix-cache finding above
— and they're exactly what missed that degradation, since a short one-shot
run never gives the prefix cache time to saturate with donated seq_ids the
way a real, sustained server process does. `scripts/repeat_bench.sh` (new,
committed) runs N repetitions against **already-running** servers, with a
discarded warmup request per engine, alternating which engine goes first
each round (cancels thermal/cache ordering bias), and drops (with a loud
warning, retried once first) any repetition that comes back with request
errors instead of silently averaging in a result computed on a smaller
sample. It reports median + [min, max], not a single number — use it for
any future fox-vs-X comparison on this hardware; a single run here isn't
trustworthy enough to draw conclusions from, as this whole investigation
kept demonstrating.

## Where to look

| Concern | File |
|---|---|
| `AMDGPU_TARGETS` passthrough | `build.rs` (ROCm/HIP auto-detection block) |
| ROCm Docker build | `Dockerfile.rocm` |
| Repeated/statistically-sound benchmarking | `scripts/repeat_bench.sh` |
| **The main fix** — dense/ascending `seq_id` allocation | `src/scheduler/mod.rs` (`seq_id_pool`, now a min-heap) |
| **The main fix** — batch emitted in ascending `seq_id` order | `src/engine/model/llama_cpp/batch.rs` (`do_decode_batch`) |
| **The closing fix** — unified KV cache, so `split_simple` runs instead of `split_equal` | `src/engine/model/llama_cpp/mod.rs` (`ctx_params.kv_unified` in `load()` and `new_context()`) |
| **Measuring ubatch widths** — forward llama.cpp's own log (pairs with `LLAMA_BATCH_DEBUG=1`) | `src/engine/model/llama_cpp/mod.rs` (`FOX_LLAMA_LOG`, next to `noop_log`) |
| **Resolved limitation** — prefix-cache hits inherit a stale seq_id (now harmless) | `src/scheduler/schedule.rs` (prefix-hit admission path, `req.kv_seq_id = hit.seq_id`), `src/scheduler/mod.rs` (`try_insert_prefix`) |
| **Ruled-out fix** — `seq_cp` can't do a partial cross-stream KV copy | `vendor/llama.cpp/src/llama-kv-cache.cpp:463` (`llama_kv_cache::seq_cp`, `is_full` assert at line 518) |
| **The sampling fix** — adaptive candidate selection | `src/engine/model/sampling.rs` (`sample_token`) |
| **The sampling fix** — skip logits copy when unneeded | `src/engine/model/mod.rs` (`needs_logits`), `src/engine/run.rs`, `src/engine/model/llama_cpp/batch.rs` |
| OpenAI vs Ollama sampling defaults (`top_k=0` vs `40`) | `src/api/shared/sampling_defaults.rs` |
| Scheduler admission order (verified correct — ~3.85/4 batch width) | `src/scheduler/schedule.rs`, `src/engine/run.rs` (`run_loop`) |
| llama.cpp's split selection and its `sequential`/ascending-seq_id requirement | `vendor/llama.cpp/src/llama-kv-cache.cpp:725`, `vendor/llama.cpp/src/llama-batch.cpp` (`split_equal`) |
| llama.cpp's async decode + deferred sync | `vendor/llama.cpp/src/llama-context.cpp` (`decode()`, `graph_compute()`, `llama_get_logits_ith()`) |
