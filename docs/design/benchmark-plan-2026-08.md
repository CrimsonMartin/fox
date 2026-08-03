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
| Ollama | not yet tried | `ollama/ollama:rocm` image already pulled |

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
