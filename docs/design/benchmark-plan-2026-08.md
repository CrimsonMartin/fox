# Four-engine benchmark and white paper — plan and current state

Handoff document. Everything a fresh session needs to continue the comparison without
re-deriving it. Written 2026-08-03, at the end of the 0.19.1 work.

## The goal

A white paper demonstrating what fox does differently, backed by a comparison of fox,
`llama-server`, Ollama and vLLM on this machine.

## Method, agreed

**Two axes, both reported.**

1. **Config-matched.** Same model, quantisation, context length and sampler settings
   across all four. Isolates the serving layer. This is the number nobody can argue with,
   and it is where fox sits at 96% of `llama-server` on decode-bound throughput.
2. **Best-effort per engine.** Each engine tuned with its own techniques — the comparison
   a user actually cares about when choosing one.

The discipline that makes axis 2 honest is that **the tuning effort must be equal**. If
fox gets speculative decoding, `llama-server` gets `--draft-model`. If fox quantises its
KV cache, so do the others. Every engine's exact configuration gets published. Tuning one
side and not the others is the failure mode to avoid, and it is easy to fall into by
accident because fox is the one whose flags we know best.

Report the workload where fox loses as prominently as the ones where it wins. A benchmark
page that only reports wins is marketing.

## Hardware

AMD Radeon 890M — **gfx1150**, RDNA 3.5, integrated. 123 GB system RAM, shared with the
GPU. Read from `/sys/class/kfd/kfd/topology/nodes/*/properties`.

## Engine status

| Engine | State | How |
|---|---|---|
| fox | ready | `Dockerfile.vulkan` → bundle; `make vulkan` |
| `llama-server` | ready, **flags audited** | `Dockerfile.llama-server-vulkan`, same vendored llama.cpp |
| vLLM | **runs**, verified | `rocm/vllm:latest` + `HSA_OVERRIDE_GFX_VERSION=11.0.0` |
| Ollama | **runs on GPU**, verified | `ollama/ollama:rocm` + `OLLAMA_IGPU_ENABLE=1`; `scripts/try_ollama_rocm.sh` |

All four engines run on this hardware. No engine has to be excluded from the comparison.

### Ollama — gate results, 2026-08-03

Run `scripts/try_ollama_rocm.sh`. It imports the same Q8_0 GGUF the published runs use
via a Modelfile rather than `ollama pull`, so axis 1 stays exact — `ollama pull
llama3.2:1b` would bring Q4_K_M and the comparison would no longer be of serving layers.

- **No `HSA_OVERRIDE_GFX_VERSION` needed.** Ollama recognises gfx1150 natively:
  `inference compute … library=ROCm compute=gfx1150 … type=iGPU`. Unlike vLLM, which
  needs the override, this is a difference worth stating in the write-up.
- **`OLLAMA_IGPU_ENABLE=1` is mandatory here.** By default Ollama *finds* the 890M and
  then discards it — `dropping integrated GPU; to enable, set OLLAMA_IGPU_ENABLE=1` —
  and falls back to CPU **without failing**. `ollama ps` reports `100% CPU` and it serves
  normally. Benchmarking that fallback against fox on Vulkan would have produced a huge,
  entirely fake win. With the flag: `100% GPU`.
- That silent fallback is why the gate asks *which processor the model is resident on*
  instead of *did the model load*. Every Ollama arm must assert `100% GPU` from
  `ollama ps` before its numbers count.

**Config-matching knobs for Ollama** — its defaults do not match what the other three
are given, and they are set by env var, not by request:

| knob | Ollama default | must be set to |
|---|---|---|
| `OLLAMA_CONTEXT_LENGTH` | `0` → chose **131072** for this model | `CTX_PER_SEQ` (4096) |
| `OLLAMA_NUM_PARALLEL` | `1` → 8 concurrent clients **serialise in a queue** | `CONC` |
| `OLLAMA_FLASH_ATTENTION` | `false` | match the other arms |
| `OLLAMA_KV_CACHE_TYPE` | unset (f16) | match `--kv-cache-type` |

`OLLAMA_NUM_PARALLEL` is the dangerous one: left at 1 it turns a concurrency benchmark
into a queueing benchmark, and the resulting TTFT curve looks exactly like the prefix-reuse
failure the paper is about. It would be a fabricated win in the direction of the thesis.

- The OpenAI-compatible surface works with `bench_burst.py` as written (streaming plus
  `stream_options.include_usage`), but usage carries **no `prompt_tokens_details`**, so
  `cached_tokens` reads 0 for Ollama. That means "not reported", not "no reuse", and the
  table has to say so. TTFT remains directly comparable.

### llama-server flag audit — done, numbers stand

- `--slot-prompt-similarity` defaults to `0.10`, same as fox. Its LCP slot affinity was
  active all along.
- `--cache-reuse` defaults to `0` and was not set in the published runs. Measured with
  `--cache-reuse 256`: cold TTFT 4376 → 4367 ms, warm 189 → 186 ms, cold `cached_tokens`
  0 either way. Inside the noise.

The comparison therefore stands with the reference configured in its favour.

### vLLM caveats

- `torch.cuda.get_device_name(0)` returns an **empty string**. ROCm allocates against the
  device without recognising it. Anything keying off the device name will misbehave.
- Do **not** quote the 5.22 tok/s from the feasibility gate. That was a 0.5B model under
  `enforce_eager=True`, which disables CUDA graphs and Inductor. Drop `enforce_eager` for
  any real measurement.
- The override belongs in vLLM's *documented configuration*, not a footnote — a reader
  with this hardware needs it too.

## Backend topology — no single run holds all four engines

Discovered while wiring the Ollama arm, and it changes the shape of the paper: **there is
no configuration in which all four engines run on the same compute backend against the
same model file.**

| engine | Vulkan | ROCm | consumes the GGUF |
|---|---|---|---|
| fox | yes (`Dockerfile.vulkan`) | yes (`Dockerfile.rocm`) | yes |
| `llama-server` | yes (`Dockerfile.llama-server-vulkan`) | yes | yes |
| Ollama | yes, but only the `:latest` image (0.30.10) | yes, only the `:rocm` image (0.32.5) | yes, via Modelfile |
| vLLM | **no Vulkan path at all** | yes | no — needs its own artifact |

So the bank splits in two, and each half must say what it is:

1. **Vulkan trio** — fox, `llama-server`, Ollama. Same backend, same GPU, same GGUF file.
   This is the config-matched axis, and it is where the serving-layer claim lives.
   `scripts/bench_engines.sh`.
2. **vLLM, separately, on ROCm, with its own model artifact.** Two variables move at
   once against the trio, so its number is not comparable to theirs at the serving-layer
   level and must not be put in the same column. What it *can* answer is the question a
   user actually asks: what does the best-known serving stack do on this hardware.

Reporting vLLM inside the trio's table would publish a backend difference as if it were
an engine difference — the exact failure mode the "equal tuning effort" rule exists to
prevent, arriving through the back door.

Two further caveats the trio table has to carry:

- fox and `llama-server` are built from the **same vendored llama.cpp**; Ollama ships its
  own fork (ggml 0.17 against the vendored 0.15.3). The fox↔`llama-server` comparison
  isolates the serving layer; the Ollama comparison does not isolate it as cleanly.
- fox runs `kv_unified = true`, both others `false`. That is not a tuning knob handed to
  fox — sharing cells via `seq_cp` under a unified KV *is* the mechanism under test — but
  it is a real difference and belongs in the configuration table, not in a footnote.

## Model

`qwen3.5:9b` (`unsloth/Qwen3.5-9B-GGUF`, 5.7 GB) for the main comparison — current, and
small enough that three rounds across four engines finishes. vLLM does not consume GGUF
natively in the same way; check whether it needs the safetensors repo instead, and record
whichever was used.

Note the architecture axis matters and is not covered by one model: sliding-window
attention (Gemma), hybrid attention/state-space (`falcon-h1` in the catalogue, where fox
disables prompt reuse entirely), and MoE all change the prefill/decode balance. A paper
measuring only dense GQA and concluding "4-6×" is refutable with a modern Gemma.

## Workloads

Built already:

- `scripts/ab_bench.sh` — decode-bound throughput, the neutral control.
- `scripts/ab_shared_prefix.sh` + `scripts/bench_burst.py` — concurrent burst behind a
  shared system prompt, cold and warm.

Still to build, in the order they are worth doing:

1. **Multi-turn chat** — reuses most of the burst driver, and backs the most-quoted
   product claim ("conversations get faster").
2. **RAG, cache-hostile** — shared system prompt, different retrieved context per query.
   Deliberately adverse to fox. Publishing where the advantage narrows is what makes the
   rest credible.
3. **Agentic** — long prefix, short fast turns, parallel sub-agents. Where fox should win
   most, and where n-gram speculative decoding should pay.
4. Code/FIM (`/infill`) and structured output (validity of produced JSON, not just speed).

## Benchmarking discipline — non-negotiable, learned the hard way

- **One server at a time.** ggml's thread pool spin-waits; an idle second server burns
  cores and skews the arm under test.
- **Alternate arms each round**, 3+ rounds, report median and range, and say plainly when
  ranges overlap.
- **Check the binary's timestamp against the commit.** A stale bundle once produced a
  confident, plausible, completely wrong result, and it was convincing because half the
  table still looked right.
- **Check the metric can move.** Pool usage read as the sum of per-slot block counts
  cannot fall when sharing works — it hid a real win across two measurements. Use
  `/slots`' `kv_blocks_used`.
- **Report measured prompt tokens.** An oversized prompt fails differently per engine:
  `llama-server` returns 400, fox rolls the context window and silently disables reuse.
- Kill servers and delete downloaded models after **every** test, not at the end.

## Results — Vulkan trio, 2026-08-03

`scripts/bench_engines.sh`, 3 rounds, arm order rotated each round so every engine leads
exactly once. 8 clients, 1856-token shared system prompt, 64 output tokens, 4096 ctx per
sequence, Llama-3.2-1B-Q8_0, Vulkan on the 890M. One server alive at a time.

| workload | fox | `llama-server` | Ollama |
|---|---|---|---|
| cold TTFT p50 | **1102 ms** | 4339 ms | 5377 ms |
| cold range | [1100, 1119] | [4312, 4341] | [5137, 5392] |
| cold burst wall | **3.00 s** | 8.43 s | 9.18 s |
| warm TTFT p50 | **50 ms** | 184 ms | 400 ms |
| warm range | [48, 53] | [184, 191] | [390, 411] |
| `cached_tokens`, cold | 12908 | 0 | not reported |

All ranges disjoint. fox is 3.94× `llama-server` and 4.88× Ollama cold; 3.68× and 8.00×
warm. The fox↔`llama-server` figures reproduce the earlier separate run (1129/4550 cold,
50/190 warm) on a freshly rebuilt bundle, which is the harness agreeing with itself.

Two things this table does **not** establish, and the write-up must not let it imply:

- **Ollama's warm TTFT is the odd number here**, 2.2× `llama-server`'s despite both being
  llama.cpp underneath, and nothing measured so far explains it. Its config was verified
  from its own log — `n_ctx = 32768`, `n_ctx_seq = 4096`, 8 slots, `flash_attn = auto`,
  matching the `llama-server` arm exactly — so it is not the obvious misconfiguration.
  Ollama ships a different llama.cpp (ggml 0.17 vs the vendored 0.15.3). Until the cause
  is found this is an observation, not a mechanism, and should be published as one.
- Nothing about **decode throughput**, which is the workload where fox has historically
  sat *below* `llama-server` at 96%. See the control below.

### The neutral control — where fox loses

`MODE=decode scripts/bench_engines.sh`, same rounds and rotation, 4 clients, 4 unrelated
short prompts, 128 output tokens each. Nothing to reuse, so this is the sampling and
batching path with prefill out of the picture. All three engines produced exactly 128
tokens per request, so they did the same work.

| metric | fox | `llama-server` | Ollama |
|---|---|---|---|
| per-request decode p50 | 45.3 tok/s | **49.6 tok/s** | 45.2 tok/s |
| range | [45.2, 45.4] | [49.0, 49.8] | [44.7, 46.0] |
| aggregate | 170.3 tok/s | **185.5 tok/s** | 158.3 tok/s |
| range | [170.0, 171.8] | [183.4, 186.9] | [155.3, 160.3] |

`llama-server` wins this one: **1.09× per request, 1.09× aggregate, ranges disjoint.**
fox and Ollama tie on the per-request rate (their ranges overlap, so no winner), but fox
finishes the batch 1.08× sooner on the aggregate with disjoint ranges — same per-stream
speed, better batching.

Note the gap against `llama-server` measured this way is **8-9%, not the 4%** quoted
elsewhere in this document. Different workload — 4 clients × 128 tokens here — so both
can be true, but the paper must quote the figure with the workload attached and should
not lead with the smaller one.

This is the table that has to sit next to the burst results, at the same prominence.
fox's case is "much faster when there is a prefix to share, slightly slower when there
is not", and stating the second half is what makes the first half credible.

## Results so far

fox vs `llama-server`, Vulkan, Llama-3.2-1B-Q8_0, both from the same vendored llama.cpp,
3 rounds, disjoint ranges:

| workload | fox | llama-server |
|---|---|---|
| 8 clients, shared 1856-token prompt, cold TTFT p50 | **1129 ms** | 4550 ms |
| 16 clients, cold TTFT p50 | **1402 ms** | 8064 ms |
| 16 clients, whole-burst wall | **3.8 s** | 16.2 s |
| 8 clients, warm TTFT p50 | **52 ms** | 193 ms |
| 4 clients, short unrelated prompts, throughput | 96% | baseline |

Doubling the clients costs fox 24% more cold TTFT and `llama-server` 79%.

The mechanism, which is the paper's actual thesis: `get_available_slot()` skips
`is_processing()` slots in both its similarity pass (`server-context.cpp:1609`) and its
LRU fallback (`:1652`), so concurrent arrivals cannot reuse from each other. fox copies a
shared prefix out of a sequence that is still decoding.

## Also outstanding, unrelated to the benchmark

- Merge to `develop` by milestones — **`main` is a squashed release-snapshot branch with
  no common ancestor**, so it is not the merge target. Three decisions are the user's:
  whether to integrate `origin/develop`'s 4 divergent commits, whether to push, and
  whether to create/push `v0.14.0`…`v0.19.1` tags (pushing tags triggers release
  workflows).
- Branch is `feature/0.19`; version is 0.19.1.
- `--moe-cpu` has no demo or guide, now that the catalogue has MoE models.
- The 4% decode gap: profile before acting. The `logits.to_vec()` the docs blamed is
  ~0.5% by arithmetic, so it is not the lever.
