# Fox — Feature & Correctness Status

A living inventory of **everything fox does** and an honest assessment of **what works
and what doesn't**. Use it to decide what to fix, in what order, and to track progress
per release.

- **Tracks through:** 0.18 (feature branch, in progress); `main`/`develop` are at 0.13.0
  as of this writing — 0.14 through 0.18 are done and closed but not yet merged up
  (deliberate, releases are being cut gradually). This file describes the code, not
  what's tagged.
- **Last updated:** 2026-08-01
- **Companion:** [Model-architecture correctness rework](docs/design/model-architecture-rework.md)
  — the design that resolved most of the ❌/⚠️ items below (see "Known issues" for what's
  still open). [`docs/design/vllm-gap-analysis.md`](docs/design/vllm-gap-analysis.md) is the
  maintained source of truth for the feature-gap comparison vs Ollama/vLLM — this file no
  longer duplicates it (see "Comparison" below).

### Assessment basis

Status is from **code review** (per-subsystem), not live runtime testing. "✅" means *no
defect found in review*, not *verified by running*. Items needing a running server or GPU
to confirm are marked ❓.

### Legend

| | Meaning |
|---|---|
| ✅ | Correct — no defect found in review |
| ⚠️ | Works with caveats / partial / footgun |
| ❌ | Incorrect for some models or inputs |
| 🚧 | Stub / parsed-but-unused / not wired |
| ❓ | Unconfirmed — needs a running/stress test |

---

## Serving runtime

| | Feature | Notes |
|---|---------|-------|
| ✅ | Axum HTTP server, startup, graceful shutdown, signal handling | |
| ✅ | Continuous batching (fox's own scheduler, not llama.cpp's) | prefill + decode per step |
| ✅ | LIFO preemption on KV pressure | frees blocks, returns seq_id, re-queues |
| ✅ | Request cancellation on client disconnect | `send()` fails → finished + KV freed immediately |
| ✅ | Multi-model registry with LRU + keep-alive eviction | engine loop aborted on `Drop` |
| ⚠️ | `max_models = 1` default | **0.18** — the *silent* part is fixed (a startup log now states the trade-off explicitly, `max_models_default_hint`); the default itself deliberately stays 1 — fox has no cross-model VRAM accounting (the per-load "does this fit" check compares against a static, whole-GPU figure from startup, never subtracting what's already resident), so raising the default without that accounting would trade a churn footgun for a real OOM-crash footgun. Real fix is proper multi-model VRAM budgeting — not done, tracked as its own follow-up, not this item |

## Model loading & architecture handling

> This is where most defects live — architecture facts derived by formula/literal,
> scattered across layers.

| | Feature | Notes |
|---|---------|-------|
| ✅ | GGUF load via FFI with actionable failure diagnosis | magic bytes / memory / GGUF version |
| ✅ | Runtime backend detection (CUDA/ROCm/Vulkan/Metal/CPU) | one binary |
| ✅ | `head_dim` from GGUF metadata (`<arch>.attention.key_length`) | **recently patched**; was `n_embd/n_head` (wrong for Gemma/MLA) |
| ✅ | Flash attention = AUTO | **recently patched**; was forced ENABLED → Gemma softcap garbage on CUDA |
| ✅ | `embedding_dim = n_embd` (read from `llama_model_n_embd`) | **fixed**; was `num_heads * head_dim` (wrong for Gemma/MLA + an out-of-bounds read). Stored on `ModelConfig.n_embd` |
| ✅ | `n_ctx` in `load()` no longer capped by a per-token formula | **0.18** — empirical create-then-shrink-on-failure retry loop (`shrink_n_ctx`) replaces the pre-creation byte-budget cap; the formula survives only as a soft first-guess ceiling under `--gpu-memory-fraction`. See `docs/design/mla-recurrent-kv-sizing.md` |
| ✅ | MLA & recurrent KV sizing correctness | **0.18** — the fix above applies uniformly (no per-arch branching); verified against real DeepSeek-V2-Lite (MLA) and Mamba (recurrent) models. Lightweight `KvMemoryClass` (Standard/Latent/Recurrent) added to `ModelInfo`/`fox probe` for observability |
| ✅ | Recurrent/hybrid detected (`llama_model_is_recurrent`/`llama_model_is_hybrid`); prefix caching disabled for them | historic fix (v0.3.1), **fixed again in 0.18** — the v0.3.1 implementation used `llama_memory_can_shift`, which real-model testing proved returns `true` for recurrent memory too ("trivial to shift", not "safe for fox's prefix cache"); silently enabled prefix caching for recurrent models until caught by testing a real Mamba GGUF end-to-end |
| ⚠️❓ | `n_ctx`/`n_batch`/`n_seq` heuristic | `.max(effective_ctx)` may size the pool for ~1 sequence while `n_seq_max=32` → possible tightness under concurrency; unconfirmed |

## Inference correctness (prefill / decode / sampling / output)

| | Feature | Notes |
|---|---------|-------|
| ✅ | Prefill/decode with stable `seq_id` (not batch slot), boundary-token resubmission | solid |
| ✅ | Sampling: rep-penalty → temp → top_k → stable softmax → top_p → draw; greedy if temp≤0; seeded | |
| ✅ | `frequency_penalty` / `presence_penalty` | **fixed (0.11)** — applied in the sampler with OpenAI semantics (`logit -= presence*seen + frequency*count`); was accepted but silently ignored |
| ✅ | UTF-8 reassembly across tokens (split emoji/CJK) | byte buffer, no `??` artifacts |
| ✅ | Multi-piece control-token holdback (BPE-split `<|im_end|>`) | |
| ✅ | User stop sequences | rolling buffer, cross-token-boundary |
| ⚠️ | Reasoning-block delimiters are per-model via `REASONING_FORMATS` (0.11) | not a hardcoded `<think>` literal anymore (`engine/output_filter.rs`'s `think_open`/`think_close` are configurable); but the registry has only one non-default entry (Gemma/GPT-OSS `<\|channel\|>`) so an unlisted model's real markers still fall through to the `<think>` default |
| ⚠️ | `U+2581 → space` applied unconditionally | SentencePiece assumption; would corrupt a BPE model containing that codepoint |
| ✅ | "supports thinking?" detection | **improved (0.11)** — checks the model's real Jinja template for `enable_thinking` first (`supports_thinking()`, `llama_cpp/mod.rs`); falls back to the old tokenize-`"<think>"`-heuristic only when the model has no template |

## APIs

| | Feature | Notes |
|---|---------|-------|
| ✅ | OpenAI: `/v1/chat/completions` (SSE + non-stream), `/v1/completions`, `/v1/models`, `/v1/embeddings`, `/health`, `/metrics` | |
| ✅ | Ollama: `/api/chat`, `/api/generate`, `/api/embed`, `/api/tags`, `/api/show`, `/api/ps`, `/api/pull`, `/api/delete`, `/api/copy`, `/api/create`, `/api/version`, load/unload | |
| ✅ | GGUF chat template rendered via real Jinja (`minijinja`) | **fixed (0.11, `cc12851`)** — was llama.cpp's legacy C engine, which doesn't run Jinja (see "Finding" below, kept as historical record); `render_chat_jinja`/`build_prompt_tokens_impl` (`engine/model/llama_cpp/vocab.rs`) render the model's actual embedded template, threading `enable_thinking`; environment compiled once per model, not per request (0.13). Falls back to the legacy built-in format only when a model has no embedded template or rendering fails. **Hardened (0.17)**: some GGUF conversions store a legacy template *name* (e.g. literally `"vicuna"`) in `tokenizer.chat_template` instead of real Jinja source — minijinja happily "renders" a no-tag string as itself, so the entire prompt silently collapsed to that one word (found via real e2e testing against `ggml-org/moondream2-20250414-GGUF` while validating vision support). `render_chat_jinja` now requires the template to contain `{{`/`{%` before trusting it as Jinja, falling through to `apply_chat_template_impl`'s name-based classifier otherwise |
| ⚠️ | Fallback template `"{role}: {content}"` when none present | may not match what the model expects |
| ✅ | Sampling defaults diverge between APIs | **intentional, documented (0.11 P4)** — centralized in `api/shared/sampling_defaults.rs`: `/v1/*` mirrors OpenAI (no `top_k`, no repeat penalty), `/api/*` mirrors Ollama (`top_k=40, repeat_penalty=1.1`); a unit test locks the divergence so it can't be "unified" by accident. Was previously undocumented duplicated literals, not a difference in behavior |
| ✅ | Optional Bearer auth (`FOX_API_KEY`), permissive CORS, OpenAI-style error mapping | |

## Product features

| | Feature | Notes |
|---|---------|-------|
| ✅ | Tool/function calling | **Hermes + Mistral + Llama3 parsers (0.16)** — `tools` is threaded into the Jinja render context, so a model whose real template natively formats tool calls (Hermes/Qwen `<tool_call>{...}</tool_call>`, Mistral `[TOOL_CALLS]`) renders and parses its own format instead of fox's generic listing; auto-detected from the model's own template (`--tool-call-parser auto\|generic\|hermes\|mistral\|llama3` to override). The Mistral parser handles both real-world wire formats (classic JSON array and the newer per-call `name[ARGS]{...}`). Llama3 (`{"name":..,"parameters":..}`, optional `<|python_tag|>`) is explicit-opt-in only — most GGUF chat templates for Llama3 models strip the tool-calling block entirely (verified against a cached `llama-3.2-1b-instruct` GGUF), so there's no reliable template signal to auto-detect it by. Models without a detected/selected native format keep the original generic prompt-based JSON parsing (`{"name","arguments"}` / `{"tool_calls":[…]}`) as the fallback |
| ✅ | JSON mode / structured output | **fixed (0.14)** — GBNF-constrained via `response_format`/`format`, JSON-schema→grammar in Rust, golden-verified; was prompt-instruction-only, no enforcement. Regex/choice-based grammar still absent |
| ✅ | Thinking / `--show-thinking` | **improved (0.11)** — `enable_thinking` is threaded through the real Jinja render, and detection uses the template's own `enable_thinking` marker before falling back to a literal-`<think>` tokenize check; the reasoning-delimiter registry (`REASONING_FORMATS`) knows Gemma/GPT-OSS's `<\|channel\|>` framing. Still whack-a-mole for any *other* model family whose real marker isn't `<think>` and isn't yet in the registry |
| ⚠️ | Vision / multimodal | **shipped (0.17)** — `--mmproj <file>` loads a paired vision projector via llama.cpp's `mtmd` library; OpenAI `image_url` (base64 `data:` URI only — no remote fetch) and Ollama `images` are encoded and answered. v1 scope: one global mmproj pairing (like `--draft-model`), no fox-level chunked-prefill/prefix-caching for the image turn (atomic `mtmd_helper_eval_chunks` call — a documented tradeoff, not a bug), no OOM bisection-retry on this path. Verified end-to-end on a real model (Gemma 4 E2B, 24/24 e2e checks). See `docs/design/vision-support.md` |
| ✅ | Embeddings | **fixed**: correct length (`n_embd`), mean-pooled + L2-normalized, non-degenerate (was all-zeros due to `pooling_type=NONE`) |

## Scheduler / KV / performance

| | Feature | Notes |
|---|---------|-------|
| ✅ | Paged KV cache (PagedAttention-style): block pool, ref-count, copy-on-write | |
| ✅ | Prefix caching by chained block hash; correct boundary resubmission | three related cross-request bugs found by real end-to-end testing and fixed in 0.15.1 (donated-sequence trimming, poisoned seq_id on failure, wrong resubmission position) |
| ✅ | KV quantization: `f16`/`q8_0`/`q4_0`, independent K/V | TurboQuant (`turbo2/3/4`) removed when migrating to upstream llama.cpp — see CHANGELOG |
| ✅ | Chunked prefill (0.13) | `--max-prefill-chunk` (default 512): a long prompt is prefilled in chunks across scheduler steps, interleaved with other requests' decode — closes the head-of-line-blocking gap vs vLLM |
| ✅ | Context rolling on full (0.13) | `--context-shift` (default on): drops the oldest KV window when a conversation fills `n_ctx` so generation continues instead of stopping with `length`; fixed in 0.15.1 to reserve headroom so it fires *before* the boundary, not exactly at it |
| ✅ | Speculative decoding — n-gram (0.15) + draft-model (0.16) | `--speculative`: byte-identical output regardless of proposer (golden-verified); n-gram 1.78× at 98% draft acceptance on repetitive output. `--draft-model <name>` (0.16) generalizes to any text via a second resident model — vocab-fingerprint checked at load time, fails loudly on mismatch; loaded eagerly, no eviction pairing/VRAM budgeting yet (documented limitation, see `docs/design/speculative-roadmap.md`) |
| ✅ | Prefix-cache block/seq_id leak on eviction | **resolved (0.12)** — was a suspected leak, closed by a dedicated stress test (`stress_prefix_cache_no_leak`) proving allocation returns to zero after draining; the original automated flag was a false positive |
| ✅ | Multi-GPU (layer/row split, manual or auto tensor-split) | |
| ✅ | MoE CPU offload (`--moe-cpu`) via expert-tensor regex | |
| 🚧 | `--swap-fraction` | parsed but unused (placeholder — real CPU↔GPU KV swap blocked on a missing llama.cpp API). **0.18**: no longer silently ignored — warns at startup when set to a nonzero value |
| ✅ | Backpressure / fail-fast (0.16) | `--max-queue-depth` rejects a full queue with HTTP 429 instead of queueing forever; a real engine failure gets a distinct `StopReason::EngineError` and an explicit terminal token instead of silently closing the response channel |
| ✅ | OOM recovery — batch-size bisection retry (0.16) + reactive context-roll (0.18) | `do_prefill`/`do_decode` distinguish `llama_decode`'s return codes (per `llama.h`) instead of treating any non-zero as fatal: `1` ("no KV slot for the batch") retries by splitting the batch in half, recursing down to a single request before giving up. **0.18** adds the "further degrade" step once bisection bottoms out: if that one remaining request has old context to discard, `engine/run.rs` performs one reactive context roll (reusing the existing `--context-shift` mechanism) and retries the whole batch once more before falling back to `EngineError`. See `docs/design/reactive-context-rolling.md`. Observable via `ferrumox_decode_bisection_retries_total` + `tracing::warn!`/`tracing::info!` per retry/roll |
| ✅ | Prefill batch-size overflow no longer crashes the process (0.18) | A real, more severe bug found while verifying the above: several requests admitted into the same prefill step each contributed their own chunk to one shared `llama_decode` call, and their **sum** could exceed `n_batch` — llama.cpp aborts via `GGML_ASSERT(n_tokens_all <= cparams.n_batch)` for this, a hard process crash with no graceful return code (unlike `ret==1`), reachable by ordinary concurrent load under a small `--max-context-len`. Fixed by capping the aggregate per-call submission against `llama_n_batch(ctx)` (`allocate_batch_budget`), spreading any excess to the next scheduler step. See `docs/design/reactive-context-rolling.md` |

## Model management / CLI

| | Feature | Notes |
|---|---------|-------|
| ✅ | Subcommands: `serve, run, pull, list, show, ps, rm, models, search, alias, bench, bench-kv`; implicit `fox <model> "prompt"` → `run` | |
| ✅ | `pull`/`search` from HuggingFace; `registry.json` (~14 curated models + aliases) | |
| ⚠️ | Ambiguous name resolution | two alias systems (registry.json vs `aliases.toml`), `:`→`-` normalization, prefix/substring match → can resolve to an unexpected file or trigger an unwanted `pull` |
| ⚠️ | VRAM estimate `file_size × 1.8` | informational warning only; does not prevent real OOM |

## Config / build / ops

| | Feature | Notes |
|---|---------|-------|
| ✅ | Config: flags + `FOX_*` env + `config.toml`, precedence flag > env > file | |
| ✅ | `build.rs`: builds llama.cpp with `GGML_BACKEND_DL`, auto-enables backends per host; ROCm FP8 patch | |
| ✅ | Prometheus metrics, JSON logs, Docker, systemd, installers | |
| ⚠️ | `vendor/llama.cpp` submodule required | without `--recurse-submodules` it won't build; stub build only via `FOX_SKIP_LLAMA=1` |

---

## Finding (2026-06-29): chat templates are not executed — no Jinja engine

> **RESOLVED in 0.11 (`cc12851`, "execute the model's real Jinja chat template", 2026-07-02).**
> Kept below as a historical record of the investigation — it documents *why* fox went with
> `minijinja` over llama.cpp's `minja`/`common_chat_*` path (bumping llama.cpp's own Jinja
> support would have meant tracking a moving upstream API; rendering in Rust with `minijinja`
> keeps the template-execution logic in fox's own tests/control). The fix shipped exactly as
> described in the "Fix" and "Implication" notes below, all three parts: (1) `minijinja` +
> `enable_thinking` threading — `engine/model/llama_cpp/vocab.rs:97-161`
> (`render_chat_jinja`/`build_prompt_tokens_impl`), template compiled once per model (0.13);
> (2) `tokenize_prompt_impl` (`vocab.rs`) parses the *template's* control tokens
> (`parse_special=true`) while user content still tokenizes literally
> (`tokenize_impl`, `add_special`/no `parse_special`) — the injection risk the finding
> flagged is handled by keeping the two tokenize paths separate; (3) `supports_thinking()`
> and `REASONING_FORMATS` (`llama_cpp/mod.rs`) detect the model's real reasoning markers
> instead of a hardcoded `<think>` literal (currently one non-default entry: Gemma/GPT-OSS's
> `<\|channel\|>` framing — an unlisted family still falls back to `<think>`). Golden test:
> `golden_chat_template_renders`. **Gap this finding did NOT originally cover, closed
> separately**: `tools` is now threaded into the Jinja context too (0.16's Hermes-parser
> work, see Product features above), so native tool-formatting macros are exercised.

fox applies chat templates through llama.cpp's **legacy C template engine**, which does
**not** run Jinja. The model's real template is detected by substring and replaced with a
hardcoded simplified format. Consequence: **thinking mode and native tool-calling are lost**
for any model whose behavior lives in its Jinja template (Gemma 4, Qwen3, …).

Verified on **Gemma 4 E2B** + pinned llama.cpp **`bc05a68`**:

- Gemma 4's GGUF ships a full Jinja template — `enable_thinking` toggle (×4), `<|think|>`
  token, tool-formatting macros.
- `apply_chat_template_impl` (`src/engine/model/llama_cpp/vocab.rs:144`) passes the template
  string to `llama_chat_apply_template`.
- That C API → `llm_chat_apply_template` (`vendor/llama.cpp/src/llama-chat.cpp:237`); **no
  `minja` exists in this commit**.
- It classifies by substring: `<start_of_turn>` → `LLM_CHAT_TEMPLATE_GEMMA`
  (`llama-chat.cpp:153`) → emits a simplified `<start_of_turn>…` format (`:372–392`) with
  **no thinking, no tools**.
- Also: `supports_thinking()` looks for the literal `<think>`, missing Gemma 4's
  `<|think|>` → reports `thinking:false`.
- Empirically: fox loaded gemma-4-E2B and answered coherently, but with `thinking:false`
  and no `<|think|>` ever emitted (the simplified template never enables it).

This is a **single root cause** behind two ⚠️ rows above (tool calling, thinking), and it
degrades fidelity for every model whose real behavior needs Jinja — so it ranks **above**
feature gaps like vision.

**Fix (architectural — belongs in the rework):** adopt a real Jinja engine — either bump
llama.cpp and use its `minja` + `common_chat_*`/`--jinja` path, or render templates in Rust
with `minijinja`, threading `enable_thinking`/tools — and detect the model's actual thinking
token (`<|think|>` vs `<think>`).

### Experiment (2026-06-29): minijinja + `enable_thinking` validates the fix

A standalone test confirmed the fix path end-to-end on the target machine (CPU,
`gemma-4-E2B`):

1. Extracted Gemma 4's real Jinja chat template from the GGUF.
2. Rendered it with **minijinja** (+ `minijinja-contrib` `pycompat`, needed for the template's
   `.get()` calls), passing `enable_thinking=true` → produced the correct
   `<|turn>system\n<|think|>\n…<|turn>model` prompt. With `enable_thinking=false` the
   `<|think|>` block is absent.
3. Temporarily patched fox to tokenize with `parse_special=true` (so `<|think|>` etc. encode as
   single control tokens, not literal text — confirmed: prompt token count dropped, `<|think|>`
   became 1 token) and fed the rendered prompt to `/v1/completions`.

**Result:** on a non-trivial problem (relative-speed word problem), Gemma 4 produced its
**native reasoning trace** in the `<|channel>thought … <channel|>` channel — thinking
activated. On trivial prompts or with `enable_thinking=false`, no thinking. The
`parse_special` patch was an experiment only and has been **reverted**.

**Implication — the thinking fix has three parts, not one:**

1. A real Jinja engine (minijinja, or llama.cpp `minja`) + thread `enable_thinking`/tools.
2. `parse_special` for the **template-added structure** so control tokens encode correctly —
   but *not* for user content (injection risk); the two must be tokenized separately.
3. Output-filter detection of the model's **actual** thinking markers — Gemma 4 uses
   `<|think|>` / `<|channel>thought`, **not** the `<think>` literal fox currently matches (so
   today fox would also leak the reasoning channel into the normal answer).

## Known issues, by severity

Mapped to the fix in the [design doc](docs/design/model-architecture-rework.md).

| # | Severity | Issue | Resolved by |
|---|----------|-------|-------------|
| 1 | ✅ Landed | `embedding_dim`→`n_embd`, embeddings pooling, KV pool follows `llama_n_ctx` | `ModelInfo` §4.1 + `fox probe` + golden tests (feature/0.11) |
| 2 | ✅ Resolved | Positional KV sizing applied to MLA/recurrent → instability in those families | **0.18** — §4.2's "ask llama.cpp, don't predict" applied via an empirical create-then-shrink retry loop at context creation (no per-arch formula, no per-arch branching); lightweight `KvMemoryClass` added for observability. Verified against real DeepSeek-V2-Lite (MLA) and Mamba (recurrent) models — also surfaced and fixed a real, separate bug where recurrent detection (`llama_memory_can_shift`) had been silently wrong since an upstream llama.cpp change. See `docs/design/mla-recurrent-kv-sizing.md` |
| 3 | ⚠️ Partial | Hardcoded control/think literals + thinking heuristic ("whack-a-mole") | Capabilities from model §4.3 — real Jinja detection + `REASONING_FORMATS` landed (0.11), but the registry covers only one non-default family; still whack-a-mole for the rest |
| 4 | ✅ Resolved | Sampling defaults diverge between APIs | **0.11 (P4)** — turned out to be a documentation/duplication problem, not a bug; centralized + the divergence is now intentional and test-locked |
| 5 | ✅ Resolved (simple scope) | Footguns: `max_models=1`, silent multimodal drop, ignored `frequency/presence_penalty`, dead `swap_fraction` | Phase P4 — multimodal drop now warns (0.11), `frequency/presence_penalty` now applied (0.11). **0.18** — `max_models=1`'s trade-off is now stated explicitly at startup instead of silent (default itself intentionally unchanged — no cross-model VRAM accounting exists yet, a separate, larger follow-up); `--swap-fraction` now warns when set to a nonzero value instead of silently doing nothing (real CPU↔GPU swap remains blocked on a missing llama.cpp API, unchanged) |
| 6 | ✅ Resolved | Prefix-cache eviction cleanup | **0.12** — stress test (`stress_prefix_cache_no_leak`) proved no leak; three *other* prefix-cache correctness bugs (unrelated to leak) were later found and fixed in 0.15.1 via real end-to-end testing |
| 7 | ✅ Resolved | Chat templates not executed (no Jinja) → thinking + native tool-calling lost (Gemma 4, Qwen3, …) | **0.11 (`cc12851`)** — real Jinja via `minijinja`, `enable_thinking` threaded, per-model reasoning-marker detection. `tools` threading (needed for native tool-*calling* specifically) landed separately in **0.16**, tracked as item 9 below |
| 8 | ✅ Resolved | No backpressure/OOM recovery — an admission-rejected or engine-crashed request silently closes its response channel (fake 200), and a real `llama_decode` failure always killed every request in the batch even when llama.cpp itself reports it as recoverable | **0.16** — `--max-queue-depth` limit + explicit error signaling + a distinct `StopReason::EngineError`, plus batch-size-bisection retry on `llama_decode` ret==1 ("no KV slot for batch"). **0.18** — added reactive context-rolling as the further "degrade" step once bisection bottoms out (see `docs/design/reactive-context-rolling.md`), and, found by the same real-concurrent-load testing, fixed a more severe *process-crash* bug: aggregate prefill tokens across several same-step requests could exceed `n_batch`, which llama.cpp enforces via a hard `GGML_ASSERT` abort with no graceful return code — now capped before the call (`allocate_batch_budget`) |
| 9 | ✅ Resolved | Tool calling was generic prompt-based only; native per-model formats were never exercised even though real Jinja rendering exists | **0.16** — Hermes, Mistral, and Llama3 parsers, `tools` threaded into the Jinja context. Llama3 is explicit-opt-in only (unreliable template auto-detection, see item above) |
| 10 | ✅ Resolved (simple scope) | Draft-model speculative decoding (generalizes 0.15's n-gram win beyond repetitive/context-echoing output) | **0.16** — `Proposer` trait + `--draft-model`. Deliberately no eviction pairing/VRAM budgeting (operator sizes both models to fit) — see `docs/design/speculative-roadmap.md` Level 2 |

**Bottom line:** the serving skeleton (batching, preemption, paged KV/CoW, prefix
caching, UTF-8/stop handling, multi-GPU, both APIs, CLI, ops) is solid, and the original
rework (items 1-7) closed most of the architecture-facts-scattered-across-layers class of
defect — real Jinja execution, centralized sampling defaults, fixed embeddings, a leak-free
prefix cache. Items 8-10 (backpressure, tool calling, draft-model speculation) are
0.16's feature work, now landed, plus vision/multimodal (0.17, see
`docs/design/vision-support.md`), LoRA adapters (0.18, see
`docs/design/lora-support.md`), multiple completions per request (0.18, see
`docs/design/n-best-of-support.md`), MLA/recurrent KV sizing (0.18, see
`docs/design/mla-recurrent-kv-sizing.md`), and reactive context-rolling plus a
process-crash fix in prefill batching (0.18, see
`docs/design/reactive-context-rolling.md`). What's left is genuine remaining
correctness debt with no work scheduled (the `REASONING_FORMATS` registry's narrow
coverage) — `docs/design/vllm-gap-analysis.md`'s feature-gap list is now fully closed;
its one remaining row (beam search) was investigated and reclassified as a deliberate
non-goal (2026-08-01): llama.cpp removed its beam-search API in 2024, vLLM itself
demoted beam search out of its fast serving path, and no major LLM API exposes
real token-level beam search today.

---

## Comparison & scope vs Ollama / vLLM

This section used to duplicate a full comparison table here; that copy drifted out of date
(it still said speculative decoding and chunked prefill were unshipped after both had
landed). The comparison is now maintained in one place —
**[`docs/design/vllm-gap-analysis.md`](docs/design/vllm-gap-analysis.md)** — and this file
just tracks the current bottom line:

Fox is a **single binary over llama.cpp/GGUF**: it competes *down* with Ollama (ease,
local-first) and looks *up* at vLLM (production throughput), and is **not** trying to become
a smaller vLLM (distributed serving, per-sequence mixed-adapter LoRA batching, non-GGUF
formats, kernel-level tensor parallel are explicit non-goals — see that doc's "What NOT
to chase").

**Already shipped since the gap analysis was last written up:** guided/structured decoding
via GBNF (0.14), logprobs/top_logprobs (0.14), min_p/logit_bias/min_tokens (0.14),
speculative decoding — n-gram (0.15) and draft-model (0.16), chunked prefill (0.13),
context rolling (0.13), backpressure/max-queue + fail-fast (0.16), Hermes/Mistral/Llama3
tool-call parsers (0.16), OOM recovery via batch-size-bisection retry (0.16),
vision/multimodal via `mtmd` (0.17, see `docs/design/vision-support.md`), single-base-model
multi-LoRA via `--lora-modules` (0.18, see `docs/design/lora-support.md`), `n`/`best_of`
multiple completions per request (0.18, see `docs/design/n-best-of-support.md`),
correct MLA/recurrent KV sizing (0.18, see `docs/design/mla-recurrent-kv-sizing.md`), and
reactive context-rolling on OOM (0.18, see `docs/design/reactive-context-rolling.md`).

**Nothing left open** on `vllm-gap-analysis.md`'s "Prioritized shortlist." Its one
remaining row, beam search, was investigated (2026-08-01) and closed as a deliberate
non-goal rather than a backlog item — see that doc for the full reasoning
(llama.cpp removed its beam-search API in 2024, vLLM demoted beam search out of its
own fast serving path, and no major LLM API exposes real token-level beam search
today; a naive fan-out approximation would just be a weaker, more expensive variant
of the `n`/`best_of` already shipped in 0.18).
