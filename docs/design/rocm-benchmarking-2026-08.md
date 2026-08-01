# fox vs Ollama on the Radeon 890M — ROCm benchmarking (2026-08-01)

Status: **closed as structural (2026-08-01)** — root cause located and
confirmed to sit inside llama.cpp's ggml-cuda/HIP backend, not in any layer
fox owns. Tracked as a structural (not backlog) gap in
[`vllm-gap-analysis.md`](vllm-gap-analysis.md) §1, alongside PagedAttention
and FlashAttention — the same "fox rides llama.cpp's kernels" ceiling. This
doc records the full evidence chain for whoever revisits it if a future
llama.cpp release changes the underlying kernel's batching behavior. Related:
[`engine-capabilities-checklist.md`](engine-capabilities-checklist.md)'s
target-machine section.

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
  `hipcc`/clang are present at build time. Required two `build.rs` fixes
  (both kept, see below) plus one **temporary, since-reverted** debug patch
  to `mod.rs`'s log callback used only to capture `LLAMA_BATCH_DEBUG=1`
  output — not part of the shipped code.
- **ollama:rocm** — official `ollama/ollama:rocm` image.

The 890M is `gfx1150` (RDNA 3.5), not in ROCm's supported-device list for
either engine. Both need `HSA_OVERRIDE_GFX_VERSION=11.0.0` to make the driver
present it as `gfx1100` (a supported RDNA3 desktop chip); Ollama additionally
needs `OLLAMA_IGPU_ENABLE=1` (it has its own iGPU drop-guard). Fox's ROCm
build targets `gfx1100` at compile time to match (`AMDGPU_TARGETS=gfx1100`).

Both engines serve the **exact same GGUF** (`llama-3.2-1b-instruct-q8_0`,
imported into Ollama via `docker cp` + a minimal `Modelfile` rather than
letting it pull its own copy) — isolates serving-engine differences from
model/quant differences.

`fox-bench` (`--concurrency 4 --requests 20 --max-tokens 128`, default
prompt) drove every comparison unless noted.

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
   sets. `GGML_HIP_ROCWMMA_FATTN` was also considered but not tried — see
   why in the results below (flash-attention isn't the bottleneck here).

`Dockerfile.rocm` (new) mirrors `Dockerfile.vulkan`'s two-stage structure;
its header comment has the exact build/run commands, including the
`--device /dev/kfd --device /dev/dri --group-add video --group-add <render-gid>`
passthrough (get the render GID via `getent group render` — passing the
group by *name* fails inside minimal containers that lack that `/etc/group`
entry).

## Results

| Comparison | fox | Ollama | Notes |
|---|---|---|---|
| fox-ROCm vs fox-Vulkan (conc=4) | **50.7 t/s** | 43.0 t/s (Vulkan) | ROCm backend is a real, if modest, win over Vulkan for fox itself |
| fox-ROCm vs Ollama-ROCm (conc=4, `OLLAMA_NUM_PARALLEL=4`) | 48.0 t/s | **93.8–120.6 t/s** | the gap that mattered |
| fox-ROCm vs Ollama-ROCm (conc=1, solo) | 44.7 t/s | 48.6 t/s | **parity** — same llama.cpp core, same per-token speed alone |
| fox-ROCm, `--max-context-len 4096` vs default (131072, model's trained ctx) | 44.3 t/s | — | **no improvement** — ruled out KV-cache-size/memory-bandwidth theory |
| fox-ROCm pinned to **ROCm 7.2** (matches Ollama's bundled `libamdhip64.so.7.2`) vs the original build on `rocm/dev-ubuntu-24.04:latest` (ROCm 7.14, bleeding-edge) | 48.8 t/s | — | **no improvement** (48-51 t/s either way) — ruled out ROCm library version |

Three theories were tested and **ruled out**:

- **Backend choice (Vulkan vs ROCm)** — ROCm is ~15% faster than Vulkan for
  fox alone, but that's a fraction of the ~2x gap vs Ollama. Not the main
  cause.
- **KV-cache footprint** (fox defaults to the model's full trained context —
  131072 tokens — when `--max-context-len` isn't passed, vs a smaller
  footprint elsewhere) — forcing `--max-context-len 4096` didn't move
  throughput. Not the cause. (Ollama's own `llama-server` also runs
  `n_ctx_slot=131072` per slot here, so this was never an apples-to-oranges
  setting to begin with.)
- **ROCm library version** — Ollama bundles ROCm 7.2 (`libamdhip64.so.7.2`);
  the original `Dockerfile.rocm` built against `rocm/dev-ubuntu-24.04:latest`,
  which turned out to be ROCm 7.14 (much newer, plausibly less mature for
  RDNA3). Rebuilt pinned to `rocm/dev-ubuntu-24.04:7.2` (had to additionally
  install `rocblas`/`hipblas` — not in that base image by default, and their
  absence makes `libggml-hip.so` silently fail to dlopen, falling back to
  CPU with no error) and re-ran the same benchmark: **48.8 t/s, statistically
  identical to the 7.14 result.** Not the cause. `Dockerfile.rocm` stays
  pinned to 7.2 anyway, for a cleaner apples-to-apples comparison going
  forward, but it isn't why fox trails Ollama.
- **A newer llama.cpp version than fox's vendored commit** — Ollama pins
  `b10091` (2026-07-22) via `LLAMA_CPP_VERSION` in its own repo, only ~3-4
  weeks ahead of fox's vendored `6f4f53f2b` (2026-06-29), and applies no
  patches to the batching/KV-cache files (`llama/compat/*.patch` doesn't
  touch them). Diffed `llama-batch.cpp`/`llama-kv-cache.cpp`/`llama-context.cpp`
  between the two commits directly: the `n_stream == 1 ? split_simple :
  split_equal` selection is byte-identical in both; the only related change
  is an unrelated new `n_keep_tail` parameter added to `split_equal` for a
  different feature (avoiding ubatch-boundary splits). Ollama is not running
  meaningfully different split-selection logic — whatever's happening is
  either present in Ollama too (and hidden by something else), or is
  upstream of where this investigation stopped.

**Scheduling/admission was checked directly from server logs and is
correct**: fox's scheduler does admit and run ~4 concurrent decode streams,
requests finish in near-lockstep as true continuous batching would predict.
The bug (if it is one) is not "fox secretly serializes requests."

## Root cause, as far as it was traced

Per-request decode speed **collapses under concurrency far more for fox than
for Ollama**: solo ~45 tok/s → ~12 tok/s per stream at 4-way concurrent for
fox (-73%), vs Ollama's ~48 → ~25-30 tok/s (-40%). Aggregate throughput barely
moves for fox with more concurrent requests; it scales close to linearly for
Ollama.

**First profiling pass (methodologically flawed, corrected below)**: profiled
a *short* 4-way-concurrent run (8 requests, 64 max tokens) with
`rocprofv3 --kernel-trace --stats` and found 94.5% of GPU kernel time in
`mul_mat_vec_q` instantiated with `ncols_dst=1` (llama.cpp's single-column
GEMV kernel — one token per call, not a multi-token batch). This was
initially read as "fox never gets true 4-wide batching." **That reading does
not survive a longer, more realistic capture** — see below.

**Second profiling pass (sustained load, the real picture)**: re-ran with 40
requests / 256 max tokens (concurrency still 4) specifically to guarantee
genuine 4-way overlap for most of the run, rather than a short burst where
request-completion staggering could dominate the sample. Result — a real
*mix* of batch widths, not a wall of `ncols_dst=1`:

| `ncols_dst` (tokens in that matmul call) | % of GPU time |
|---|---|
| 1 | 47.5% |
| 2 | 37.5% |
| 3 | 8.4% |
| 4 (all 4 concurrent sequences in one call) | ~1.1% |

Weighted-average batch width: **~1.64 of a possible 4** — real batching is
happening (the earlier 94.5%-at-1 number was substantially a short-benchmark
artifact), but full 4-wide overlap is rare. This is architecturally
*expected*, not necessarily a bug: as soon as one of the 4 concurrent
sequences finishes (a shorter response, or hits a stop token before
`max_tokens`), the next queued request needs its own **prefill** step before
it can join a decode ubatch — for at least one scheduler step, the freed
slot sits idle rather than immediately backfilled, which is exactly the kind
of natural-completion variance that produces a 1-4 mix rather than a flat 4.
Confirmed via `llama-graph.cpp` reading, too: `llm_graph_context::build_ffn`
and the QKV projection reshapes (`llama-graph.cpp:1542-1544`) use the flat
`ubatch.n_tokens` as the batch dimension throughout — there's no
per-sequence-group tensor reshape that would artificially force
`ncols_dst=1` regardless of real concurrency, ruling out the "ggml-cuda
lacks a fused 3D-batched GEMV kernel" hypothesis from the previous write-up
of this doc.

**What this means for the throughput gap**: it's **not** "fox fails to
batch." Two threads followed from the ~1.64/4 number above — one is now
resolved, definitively ruling out fox's own code:

1. **Keeping the batch fuller — resolved, fox is not the cause.** Does fox
   eagerly prefill a queued request the moment a decode slot frees up (so it
   can rejoin the very next decode step), or does it lose a step? Verified
   directly: added a temporary per-scheduler-tick log (`src/engine/run.rs`'s
   `run_loop`, logging `prefill_ids.len()`/`decode_ids.len()` each
   iteration; reverted after use, never committed) and measured real
   sustained-load traffic (40 requests, 256 max tokens, concurrency 4) on
   **both** the native CPU build and the ROCm build:

   | Backend | Avg. decode batch width (of a possible 4) |
   |---|---|
   | CPU (native) | 3.81–3.83 |
   | ROCm | 3.85–3.88 |

   Essentially identical regardless of backend speed. **fox's scheduler
   hands the model a near-full batch on almost every decode step** — the
   handful of degraded ticks (batch width 1–3) are rare (~10% combined) and
   match exactly what's expected when a shorter response finishes early and
   its replacement needs one prefill step before rejoining. This is not
   where the throughput gap lives.
2. **Per-call GPU kernel width — confirmed as the actual gap.** The
   scheduler hands the model layer ~3.85 of 4 possible sequences per step,
   which `do_decode_batch` correctly folds into one `llama_batch` /
   `llama_decode` call — yet the *GPU kernel* dispatch for that same batch
   (measured via `rocprofv3`, see above) averages only ~1.64 of that
   possible width. The ~3.85-vs-~1.64 mismatch happens **inside
   `llama_decode` itself** — between fox correctly submitting a
   near-full batch and llama.cpp's ggml-cuda/HIP backend actually executing
   it as fused multi-sequence GEMV calls. This is confirmed to be llama.cpp/
   ggml-cuda internal behavior, not a fox scheduling or batch-construction
   defect, and not something fixable without patching or forking
   ggml-cuda — against fox's own "wrap llama.cpp, don't own its kernels"
   architecture. **Closed as structural** in `vllm-gap-analysis.md` §1.

Fox's own code was checked and is not at fault: `do_decode_batch`
(`src/engine/model/llama_cpp/batch.rs`) does build **one** `llama_batch`
covering all admitted requests and calls `llama_decode` **once** per step —
standard continuous batching, architecturally the same as what
`llama-server` does.

The split into per-token GEMV calls happens **inside llama.cpp's own ubatch
construction**, confirmed by rebuilding with a temporary log-callback patch
and running with `LLAMA_BATCH_DEBUG=1` (llama.cpp's own env-gated debug
tracing, `llama-batch.cpp`): the real decode ubatch for 4 concurrent
sequences prints `equal_seqs=false, n_tokens=4, n_seqs=4` — the exact
signature of `llama_batch_allocr::split_simple` (`ubatch_add(idxs,
idxs.size(), false)`), not `split_equal` (which always passes
`equal_seqs=true`).

**Contradiction resolved (2026-08-01, follow-up session): `split_equal` IS
chosen, not `split_simple`.** `llama_kv_cache::init_batch`
(`llama-kv-cache.cpp:725`) picks between the two splitters via `n_stream ==
1 ? split_simple : split_equal`, with `n_stream` set at construction from
`unified ? 1 : n_seq_max`. Confirmed by temporarily instrumenting
`llama_kv_cache::init_batch` itself with a raw `fprintf` (not a fox
change — a throwaway one-line edit to the vendored
`vendor/llama.cpp/src/llama-kv-cache.cpp`, rebuilt, exercised with a real
request, then reverted via `git -C vendor/llama.cpp checkout --` — never
committed, per this repo's submodule convention): the print at the exact
call site read **`n_stream=33 n_seq_max=33 n_ubatch=512`**, matching the
KV-cache startup log's `"33/33 seqs"` exactly. `n_stream == 1` is false, so
`split_equal` runs, which unconditionally passes `equal_seqs=true` to
`ubatch_add` — there is no code path in `split_equal` that produces
`equal_seqs=false`.

This means the earlier `LLAMA_BATCH_DEBUG=1` capture was **misread**: the
`equal_seqs=false, n_tokens=4, n_seqs=4` block attributed to "the real
4-way-concurrent decode ubatch" in the previous write-up of this doc was not
that — it must have been a different `ubatch_print` call (a `split_simple`
call from some other code path, e.g. one of the `graph_reserve`/warm-up
passes visible in that same debug capture, which construct synthetic
worst-case ubatches at startup, not real inference ubatches). The real
decode ubatch does have `equal_seqs=true`, exactly as `split_equal`'s code
guarantees — the splitter was never the problem; the throughput gap lives
in the batch-width-distribution and per-call-speed questions above instead.

**Side effect of this detour**: found and fixed a latent bug in
`Dockerfile.rocm` — it was missing `hipblas-dev`/`rocblas-dev` (only had the
runtime `hipblas`/`rocblas` packages plus `hipblas-common-dev`). Earlier
builds succeeded anyway via what was apparently incidental transitive
installation; a `--no-cache-filter builder` rebuild (needed to force Docker
to pick up the vendor instrumentation edit — a stale cached `COPY . .`
layer was silently reusing old vendor source across three separate rebuild
attempts) exposed it as a hard `Could not find hipblasConfig.cmake` CMake
failure. Fixed by adding the `-dev` packages explicitly; kept in the
Dockerfile going forward.

Whether Ollama achieves a fuller average batch width, a faster per-call GEMV,
or both, remains untested — Ollama's minimal image ships no profiling tools,
so it wasn't profiled the same way fox was.

## `n_batch`/`n_ubatch` experiment (tried, reverted)

Matched Ollama's exact `llama-server` launch values on fox's own context
creation: `ctx_params.n_batch = ctx_params.n_ubatch = effective_max_ctx.min(2048).max(max_batch_size)`
instead of deriving `n_batch` from the model's full context length (131072
here). Result: **54.4 t/s, a real +7-13% over the 48-51 t/s baseline — but
nowhere near closing the ~2x gap to Ollama's ~103-120 t/s.**

**Reverted anyway**, because capping `n_batch`/`n_ubatch` at 2048 has a
correctness risk that isn't cheap to guard against: fox's `get_embeddings`
path (`do_decode_batch`'s sibling, `do_get_embeddings` in `batch.rs`) submits
the **entire** input as one unchunked `llama_decode` call — no chunked
prefill involved. For a non-causal (encoder-style, BERT-family) embedding
model, `llama-context.cpp`'s decode() path asserts `n_ubatch >= n_tokens`
whenever `cparams.causal_attn == false` (resolved from the model's own
`hparams.causal_attn`, `llama-context.cpp:192`) — so an embedding input
longer than 2048 tokens on such a model would hit that assertion and crash,
where today (uncapped `n_batch`) it doesn't.

The blocker to gating this safely: **llama.cpp exposes no public API to query
whether a model is causal/non-causal before creating a context** —
`hparams.causal_attn` only resolves *during* `llama_init_from_model`, and
there's no `llama_model_*` getter for it (only `llama_set_causal_attn`,
which needs an existing context and is a setter). fox also has no existing
model-capability classification (`ModelInfo` doesn't track this) to lean on
instead. Doing this properly needs either new upstream-shaped FFI plumbing,
or a GGUF-metadata-based heuristic (read `general.architecture` and match
against a maintained list of known non-causal families — `bert`,
`nomic-bert`, `jina-bert`, etc.) — real work, not justified by a 7-13% gain
on a change that doesn't fix the actual bottleneck. If someone later revisits
this specifically to squeeze out that last 7-13% (e.g. as part of shipping
a real architecture classification for other reasons), the code to restore
is: cap `n_batch`/`n_ubatch` at `effective_max_ctx.min(2048).max(max_batch_size)`
in both `LlamaCppModel::load()` and `::new_context()` (`src/engine/model/llama_cpp/mod.rs`),
gated on the model being causal.

## Closed — if this is ever revisited

Every fox-owned layer was tested and ruled out: ROCm library version, an
llama.cpp version/patch gap vs Ollama, the split-selection logic
(`split_equal` confirmed correct), the `n_batch`/`n_ubatch` mismatch, and
scheduler admission pipelining (measured directly at ~3.85/4 batch width on
both CPU and ROCm — the scheduler is not starving the batch). The remaining
gap is confirmed to live inside llama.cpp's ggml-cuda/HIP `mul_mat_vec_q`
dispatch, and is closed as **structural** — see `vllm-gap-analysis.md` §1.
Not tracked as fox backlog. If revisited later:

1. Profile Ollama itself with the same `rocprofv3` methodology (needs
   installing profiling tooling into or against its container, since the
   official image ships none) to see whether its `llama-server` build hits
   the same `mul_mat_vec_q` dispatch pattern and just runs each call faster
   (a build flag, kernel-selection heuristic, or driver difference), or
   genuinely achieves a wider average kernel-launch batch.
2. Track whether a future llama.cpp release changes `mul_mat_vec_q`'s
   multi-sequence batching — this is the one condition that would reopen
   this as an achievable gap rather than a structural one.
3. The small independent 7-13% `n_batch`/`n_ubatch` win found and reverted
   earlier (`n_batch`/`n_ubatch`-experiment section above) is unrelated to
   this and still available if someone builds the causal/non-causal model
   detection it needs.

## Where to look

| Concern | File |
|---|---|
| `AMDGPU_TARGETS` passthrough | `build.rs` (ROCm/HIP auto-detection block) |
| ROCm Docker build | `Dockerfile.rocm` |
| fox's decode batch construction (verified correct) | `src/engine/model/llama_cpp/batch.rs` (`do_decode_batch`) |
| Scheduler admission order (verified correct — ~3.85/4 batch width) | `src/scheduler/schedule.rs`, `src/engine/run.rs` (`run_loop`) |
| llama.cpp's split selection (confirmed correct: `split_equal`) | `vendor/llama.cpp/src/llama-kv-cache.cpp:725` |
| llama.cpp's two splitters | `vendor/llama.cpp/src/llama-batch.cpp` (`split_simple`, `split_equal`) |
| llama.cpp's FFN/QKV graph building (confirmed: flat `n_tokens`, no forced `ncols_dst=1`) | `vendor/llama.cpp/src/llama-graph.cpp` (`build_ffn`, QKV reshapes ~line 1542) |
