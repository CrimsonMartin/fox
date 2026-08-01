# fox vs vLLM — gap analysis

What separates fox from [vLLM](https://github.com/vllm-project/vllm), the reference
high-throughput serving engine. The point of this doc is **not** to make fox into vLLM
— it's to be honest about which gaps are *worth closing* and which are consequences of
a deliberate architectural choice.

## The load-bearing difference

**vLLM is a native PyTorch/CUDA engine with custom kernels; fox wraps llama.cpp.** vLLM
owns its attention kernels (PagedAttention, FlashInfer), its CUDA graph capture, its
tensor-parallel all-reduce. fox rides llama.cpp's `llama_decode` — fox's scheduler
decides *what* to batch, but the compute is llama.cpp's.

That single fact sets the ceiling: **on a datacenter GPU, fox's tokens/s is bounded by
llama.cpp, not by fox.** Chasing vLLM's raw throughput there is not a feature backlog,
it's a rewrite. So the gaps below are tagged:

- **achievable** — wireable inside the wrapper model, often because llama.cpp already
  has the primitive; these are real backlog items.
- **structural** — tied to owning the kernels / being a datacenter engine; out of fox's
  niche (single static binary, CPU + consumer iGPU via Vulkan, Ollama-compatible,
  instant start).

**Legend:** ✅ has it · ⚠️ partial / with caveats · ❌ no

---

## 1. Throughput / batching core

| Capability | vLLM | fox | Kind |
|---|---|---|---|
| Continuous batching | ✅ fused in kernel | ✅ at scheduler level | — |
| PagedAttention | ✅ custom CUDA kernel | ⚠️ paged **accounting**; the attention itself is llama.cpp's | structural |
| Automatic prefix caching | ✅ | ✅ | — |
| Chunked prefill | ✅ | ✅ (0.13) | — |
| CUDA graphs / `torch.compile` | ✅ | ❌ (llama.cpp's domain) | structural |
| FlashAttention / FlashInfer selection | ✅ | ⚠️ FA=AUTO via llama.cpp | structural |

Behaviourally (interleaving, queueing, fairness) fox can match vLLM; on raw GPU
throughput it cannot, and that's fine — different hardware target.

## 2. Advanced decoding — **the highest-ROI gaps**

| Capability | vLLM | fox | Kind |
|---|---|---|---|
| Guided / structured decoding (JSON-schema, regex, grammar) | ✅ (outlines / xgrammar) | ❌ prompt-only | **achievable** |
| logprobs / prompt_logprobs / echo | ✅ | ❌ | **achievable** |
| Speculative decoding (draft / n-gram / EAGLE / Medusa) | ✅ | ⚠️ n-gram/prompt-lookup ✅ (0.15); draft-model ❌ | **achievable** |
| `n>1` / `best_of` | ✅ | ✅ (0.18) independent fan-out, capped at 8 — see `n-best-of-support.md` | — |
| beam search | ⚠️ demoted, not first-class | ❌ — deliberately out of scope (2026-08-01), see below | **structural, not achievable** |

**llama.cpp has native GBNF grammar support** → structured/JSON decoding is the single
biggest impact for the least effort. Speculative decoding is the largest *latency* win
that is actually within reach (llama.cpp has draft-model and n-gram primitives).

**Beam search — researched and deliberately closed as out of scope (2026-08-01), not
just "not yet done."** This row was tracked as "achievable" until actually
investigated:

- **llama.cpp removed `llama_beam_search()` from its public API in 2024** (dropped
  from the HTTP server in `a8c981b73`, then from the core library in `0cd6bd348`) —
  an ancestor of fox's currently pinned commit. There is no beam-search primitive left
  to build on; only the generic `llama_memory_seq_cp`/`seq_rm`/`seq_add` sequence
  primitives remain, and llama.cpp's own source marks true N-way KV cell sharing
  between sequences an unfinished refactor (`TAG_KV_CACHE_SHARE_CELLS` TODOs
  throughout `llama-kv-cache.cpp`), not a hardened feature.
- **vLLM itself demoted beam search out of its fast path** (RFC
  [#8306](https://github.com/vllm-project/vllm/issues/8306), 2024): pulled out of the
  core scheduler/PagedAttention-integrated path into a separate, offline-batch-oriented
  API (`LLM.beam_search()`), because it doesn't compose with continuous batching the
  way independent sampling does — `use_beam_search` was removed from `SamplingParams`
  entirely. The "vLLM has it" side of this gap-analysis row was already weaker than it
  looked.
- **OpenAI's API has never exposed real, token-level beam search** — the closest
  historical analog, the legacy `/v1/completions` `best_of` parameter, was always
  independent-sample-then-rank (exactly what fox's own `best_of` already does, see
  `n-best-of-support.md`), not joint per-token re-ranking. There is no natural
  OpenAI-compatible request shape to attach real beam search to; it would have to be a
  fox-only extension nobody's client library expects.
- **A *real*, KV-sharing-efficient implementation would need mechanics fox doesn't
  have and llama.cpp doesn't cleanly offer**: fox's existing block-sharing (used for
  prefix-cache donation) is a one-time, exclusive hand-off from a *finished* sequence
  to a *new* one — beam search needs a *live* fork of a *running* sequence into K
  siblings, repeated every decode step as beams are pruned and re-spawned. At the
  llama.cpp level, fox runs with `kv_unified = false` (the default), under which every
  cross-sequence `llama_memory_seq_cp` is the expensive, real-buffer-copy path, not a
  cheap metadata-only one — making per-step beam-forking cheap would additionally need
  a `kv_unified = true` architecture change with its own side effects on fox's
  existing context-sizing math.
- **A *naive* implementation (each beam as an independent request, re-ranked between
  coarse rounds)** would just be a more expensive, weaker variant of the `n`/`best_of`
  fan-out already shipped in 0.18 — no per-token joint re-ranking, redundant prefill
  cost scaling with beam width × generation length × rounds, without the actual
  benefit beam search is supposed to provide.

Given all of the above, this is scoped out as a considered decision, not a backlog
item — consistent with "major LLM APIs such as GPT, Gemini, and Claude" not supporting
it either. If llama.cpp or a client ecosystem ever brings back a real, efficient
primitive for this, it's worth revisiting; nothing here is expected to change on its
own.

## 3. Sampling

fox has: temperature, top_p, top_k, seed, repetition/frequency/presence penalties.

Missing vs vLLM (all **achievable** — pure sampling logic): **min_p, typical_p,
mirostat, logit_bias, min_tokens.**

## 4. LoRA / adapters

vLLM serves multi-LoRA with per-request hot-swap (per-sequence, kernel-mixed). fox:
✅ (0.18) single-base-model multi-LoRA via `--lora-modules`, adapter selected through
the `model` field, group-and-switch per llama.cpp's own context-level
`llama_set_adapters_lora` — see `lora-support.md`. **Not equivalent to vLLM's
per-sequence mixing**: llama.cpp has no punica-style kernel, so two concurrently
batched requests on different adapters are processed as separate sub-batches, and
adapter switches carry a real `sched_need_reserve` cost under heavy churn. Serving
LoRA on top of *multiple different* base models concurrently remains ❌.

## 5. Model architectures

| Class | vLLM | fox |
|---|---|---|
| Dense / GQA | ✅ | ✅ solid |
| MoE (Mixtral, DeepSeek-MoE, Qwen-MoE) | ✅ optimized | ⚠️ loads + CPU offload, approximate sizing |
| MLA / latent KV (DeepSeek V2/V3) | ✅ | ✅ (0.18) correctly sized via empirical create-then-shrink retry (no per-token formula), verified against real DeepSeek-V2-Lite; see `mla-recurrent-kv-sizing.md`. Context-rolling for MLA is a real, separate llama.cpp-side gap (`DSV4` compressed-cache shift unimplemented upstream), not fox's |
| Vision / multimodal (LLaVA, Qwen-VL) | ✅ | ✅ (0.17) via `mtmd` + `--mmproj`; base64 images only, see `vision-support.md` |
| Embeddings (BERT, nomic) | ✅ | ✅ dim + pooling fixed (0.11); mean-pool only, CLS not auto-detected |
| Encoder-decoder (T5) | ✅ | ❌ |
| Recurrent / hybrid (Mamba, RWKV) | ✅ | ✅ (0.18) correctly sized (same fix as MLA — no per-token formula applies); prefix caching correctly disabled via `llama_model_is_recurrent`/`llama_model_is_hybrid` (a real detection bug — the prior `llama_memory_can_shift`-based check silently returned the wrong answer for recurrent models — was found and fixed verifying against a real Mamba GGUF); see `mla-recurrent-kv-sizing.md` |

See [`engine-capabilities-checklist.md`](engine-capabilities-checklist.md) §2 for the
per-architecture detail and [`model-architecture-rework.md`](model-architecture-rework.md).

## 6. Quantization

vLLM: GPTQ, AWQ, FP8, INT8, bitsandbytes, Marlin kernels, KV-cache fp8.
fox: **GGUF only** (K-quants / legacy / IQ) + KV f16 / q8_0 / q4_0.

This is **not a real gap** — GGUF is exactly right for fox's CPU/consumer niche, and
non-GGUF safetensors formats are out of scope by design (fox is a GGUF engine).

## 7. Scale / parallelism (mostly structural)

| Capability | vLLM | fox | Kind |
|---|---|---|---|
| Tensor parallel (kernel-level all-reduce) | ✅ | ⚠️ layer/row split via llama.cpp | structural |
| Pipeline parallel / multi-node / distributed | ✅ | ❌ | structural |
| Disaggregated prefill/decode (P/D) | ✅ | ❌ | structural |

Datacenter-scale features, outside fox's single-node niche.

## 8. Serving robustness (achievable, medium value)

| Capability | vLLM | fox | Kind |
|---|---|---|---|
| OOM recovery (retry, degrade context) | ✅ | ✅ (0.16 batch-size-bisection retry + 0.18 reactive context-roll) | fail-fast (queue-depth cap → 429) + retry a recoverable `llama_decode` failure by shrinking the batch, then (0.18) one reactive context-roll on the remaining request before giving up — see `reactive-context-rolling.md` |
| Backpressure / rate-limit / max-queue | ✅ | ✅ (0.16) | — |
| Request priority (priority preemption) | ✅ | ⚠️ LIFO preemption only, no priority | achievable |
| KV offload / swap to CPU | ✅ | ⚠️ `--swap-fraction` placeholder, unimplemented | achievable |
| Tool calling with per-model parsers | ✅ | ✅ Hermes, Mistral, Llama3 (0.16) | — |

fox already has: continuous batching, disconnect cancellation, LIFO preemption,
context rolling (0.13), OpenAI + Ollama compat, Prometheus metrics, auth, health.

---

## Prioritized shortlist (best ROI given the llama.cpp wrapper)

Shipped since this analysis was written:

- ✅ **Guided / structured decoding via GBNF** (0.14) — `response_format` / Ollama
  `format`, JSON-schema→GBNF in Rust.
- ✅ **logprobs / top_logprobs** (0.14).
- ✅ **min_p, logit_bias, min_tokens** (0.14).
- ✅ **Embeddings** were already fixed back in 0.11 (correct `n_embd` length, mean-pool +
  L2, non-degenerate — golden-verified); the only remaining nuance is that dedicated
  embedding models' native pooling (CLS) isn't auto-detected (fox always mean-pools).
- ✅ **Speculative decoding — n-gram / prompt-lookup** (0.15) — exact (byte-identical
  output), off by default; 1.78× on repetitive output at 98% acceptance. Draft-model
  speculation is a later extension reusing the same verify/accept machinery.
- ✅ **Backpressure / max-queue + fail-fast** (0.16) — `--max-queue-depth` rejects new
  requests with HTTP 429 once the scheduler queue is full; a real engine failure
  (`StopReason::EngineError`) is now reported as an error instead of silently closing
  the response channel (which used to read as a fake empty 200).
- ✅ **OOM recovery — batch-size bisection retry** (0.16) — `llama_decode`'s return
  code was previously collapsed into a single fatal branch; `do_prefill`/`do_decode`
  now distinguish `1` ("no KV slot for the batch", per `llama.h`) from genuinely fatal
  codes and retry by splitting the batch in half and decoding each half
  independently — llama.cpp's own documented mitigation for that code — recursing
  down to a single request before falling back to the existing `EngineError` path.
  Observable via `ferrumox_decode_bisection_retries_total` + a per-event
  `tracing::warn!`.
- ✅ **Reactive context-rolling on OOM** (0.18) — once bisection retry bottoms out at a
  single request and it still fails, `engine/run.rs` performs one targeted context roll
  (reusing the existing `--context-shift` mechanism) on that request and retries the
  whole batch once more before falling back to `EngineError`. A typed error
  (`KvCacheFullAtMinimum`) carries the failing request id up from the model layer,
  which has no scheduler/config access, to the engine layer that does. In practice a
  narrow safety net — the existing proactive per-request threshold already prevents
  most aggregate exhaustion under normal load. The same stress-testing also found and
  fixed a more severe, unrelated bug: several requests' prefill chunks admitted into
  one scheduler step could together exceed `n_batch`, which llama.cpp enforces via a
  hard-abort `GGML_ASSERT` (no graceful return code, unlike `ret==1`) — a real
  process-crash reachable by ordinary concurrent load under a small
  `--max-context-len`, now capped before the call. See
  `docs/design/reactive-context-rolling.md`.
- ✅ **Hermes, Mistral, and Llama3 tool-call parsers** (0.16) — `tools` threaded into
  the Jinja render context, auto-detected from the model's own template
  (`--tool-call-parser auto\|generic\|hermes\|mistral\|llama3`). Mistral's parser
  handles both wire formats found in the wild: the classic `[TOOL_CALLS] [{"name":..,
  "arguments":..}]` JSON array (docs.mistral.ai, vLLM's `mistral` parser) and the
  newer per-call `[TOOL_CALLS]name[ARGS]{...}` format the currently-vendored
  llama.cpp's own PEG chat parser implements. Llama3 (`{"name":..,"parameters":..}`,
  optional `<|python_tag|>` prefix) is **explicit-opt-in only** — verified against the
  `llama-3.2-1b-instruct` GGUF already cached for e2e testing that real-world GGUF
  chat templates for Llama3 models routinely strip the tool-calling block entirely, so
  there's no reliable template marker to auto-detect it by. Models without a
  detected/selected native format keep the original generic prompt-based JSON parsing
  as the fallback.
- ✅ **Draft-model speculation** (0.16) — `--draft-model <name>` generalizes the 0.15
  n-gram win beyond context-echoing output via a second resident model; vocab
  compatibility is a hard load-time check, golden-verified exact via self-speculation.
  Loaded eagerly, no eviction pairing/VRAM budgeting (simple-scope decision, see
  `docs/design/speculative-roadmap.md` Level 2).
- ✅ **Vision / multimodal** (0.17) — `--mmproj <file>` loads a paired vision
  projector via llama.cpp's `mtmd` library; OpenAI `image_url` (base64 `data:` URI
  only) and Ollama `images` are encoded and answered. One global mmproj pairing
  (mirrors `--draft-model`); the image turn is prefilled atomically (no fox-level
  chunking, no prefix caching, no OOM bisection-retry on this path) — see
  `docs/design/vision-support.md` for why that scope was chosen.

Nothing left open in this section. Beam search (§2) was the last row still marked
"achievable" — investigated and reclassified as a deliberate non-goal, not a backlog
item; see the detailed reasoning under that table.

## What NOT to chase (outside the niche)

Disaggregated serving, pipeline / multi-node, kernel-level tensor parallelism,
FP8 / AWQ / GPTQ safetensors, CUDA graphs. That is vLLM-in-a-datacenter. fox's niche —
**one static binary, CPU + consumer iGPU (Vulkan), Ollama-compatible, instant start,
low memory** — is a place vLLM doesn't play. Winning there beats losing the throughput
race on an H100.
