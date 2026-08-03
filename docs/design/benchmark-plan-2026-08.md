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
| vLLM | **serves, measured** | `rocm/vllm:latest` + `HSA_OVERRIDE_GFX_VERSION=11.0.0`; `scripts/bench_vllm.sh` |
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
small enough that three rounds across four engines finishes.

The measurements above are on **Llama-3.2-1B-Q8_0**, not qwen3.5:9b — it is the model the
earlier fox↔`llama-server` runs used, so reusing it let the new harness be checked
against a known answer before anything new was claimed. Repeat on the larger model before
publishing.

vLLM's artifact question is settled: it does not take the GGUF, and it was given
`unsloth/Llama-3.2-1B-Instruct` safetensors at BF16 (ungated, no HF token needed). That
is a real difference in what is being executed, recorded everywhere its numbers appear.

Note the architecture axis matters and is not covered by one model: sliding-window
attention (Gemma), hybrid attention/state-space (`falcon-h1` in the catalogue, where fox
disables prompt reuse entirely), and MoE all change the prefill/decode balance. A paper
measuring only dense GQA and concluding "4-6×" is refutable with a modern Gemma.

## Workloads

Built already:

- `scripts/ab_bench.sh` — decode-bound throughput, the neutral control.
- `scripts/ab_shared_prefix.sh` + `scripts/bench_burst.py` — concurrent burst behind a
  shared system prompt, cold and warm.

Built since (2026-08-03):

- `scripts/bench_engines.sh` — N engines, two backends, four modes (`burst`, `decode`,
  `sweep`, `noisy`), one server alive at a time, arm order rotated per round.
- `scripts/bench_decode.py` — the neutral decode-bound control.
- `scripts/bench_noisy.py` — noisy neighbour: a long prefill injected into live streams.
- `scripts/bench_vllm.sh` — vLLM on its own terms, separate table.
- `scripts/probe_cached_tokens.py` — tells "did not reuse" apart from "does not report".
- `scripts/try_ollama_rocm.sh` — Ollama feasibility gates, including GPU residency.

Still to build, in the order they are worth doing:

1. **Multi-turn chat** — reuses most of the burst driver, and backs the most-quoted
   product claim ("conversations get faster").
2. **RAG, cache-hostile** — shared system prompt, different retrieved context per query.
   Deliberately adverse to fox. Publishing where the advantage narrows is what makes the
   rest credible.
3. **Agentic** — long prefix, short fast turns, parallel sub-agents. Where fox should win
   most, and where n-gram speculative decoding should pay.
4. Code/FIM (`/infill`) and structured output (validity of produced JSON, not just speed).

KPIs worth adding next, in order of what they would reveal:

- **Extend the sweep to 64 and 128.** The current one stops before fox and `llama-server`
  bend, so no maximum can be quoted from it.
- **Goodput under an SLO** (fraction of requests meeting TTFT and ITL targets at each
  concurrency) — derivable from data already collected, no new runs.
- **Energy per 1000 tokens.** `power1_average` is exposed under the GPU's hwmon (~40 W
  idle here). For a product that runs on a laptop this is a differentiator nobody publishes.
- **Cold start and reload cost.** Ollama unloads after 5 minutes by default; fox has an
  LRU with `--keep-alive-secs`. A mid-session reload is invisible in every throughput table.
- **Reproducibility under concurrency.** fox is known to drift at `temperature=0` under
  concurrent load. Whether the other three do too decides if that is a property of
  continuous batching or a fox defect — it is currently an untested assumption.

Use cases still unmeasured: model switching (two models alternating — the most common
local setup), mid-generation cancellation, long-context single prompts (prefill-only,
where fox's cache cannot help), batch embedding, and offline bulk processing.

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

### Both backends, measured — there is no single winner

Decided to publish both rather than pick one. `BACKEND=vulkan|rocm scripts/bench_engines.sh`,
3 rounds each. Cold-burst TTFT p50:

| engine | Vulkan | ROCm | |
|---|---|---|---|
| fox | **1121 ms** | 2391 ms | Vulkan 2.1× |
| `llama-server` | **4327 ms** | 11315 ms | Vulkan 2.6× |
| Ollama | 5344 ms | **4645 ms** | ROCm 1.15× |

Warm TTFT reverses it — ROCm wins for all three (fox 48 vs 49, `llama-server` 140 vs 180,
Ollama 370 vs 405 ms) — and decode leans slightly Vulkan. So the older
`rocm-benchmarking-2026-08.md:107` line, "ROCm is ~15% faster than Vulkan", holds **only
for decode**; on cold prefill Vulkan is 2-2.6× better for both llama.cpp-derived engines.
Any backend recommendation has to name the workload.

Practical asymmetry worth stating alongside the numbers: gfx1150 is not officially
supported by ROCm. Both ROCm images compile for gfx1100 and `HSA_OVERRIDE_GFX_VERSION`
misrepresents the card to the runtime. Vulkan needs none of that and also runs on Intel
and NVIDIA. A 15% decode win does not buy that fragility for a default.

### Saturation curves — and the ceiling this sweep did not reach

`MODE=sweep`, decode workload at concurrency 1→32, 3 rounds. Aggregate tok/s and the
scaling efficiency against a single client:

| conc | fox (Vulkan) | `llama-server` (Vulkan) | Ollama (Vulkan) | fox (ROCm) | `llama-server` (ROCm) | Ollama (ROCm) |
|---|---|---|---|---|---|---|
| 1 | 53 | 54 | 48 | 52 | 54 | 48 |
| 4 | 170 | **192** | 158 | 174 | 184 | 129 |
| 8 | 249 | **277** | 248 | 277 | 299 | 221 |
| 16 | 376 | **429** | 337 | 416 | 434 | 140 |
| 32 | 584 | **663** | 460 | 496 | 496 | 133 |
| efficiency @32 | 35% | 38% | 30% | 30% | 28% | 9% |

Read honestly, three things come out of this:

- **The sweep never found fox's or `llama-server`'s knee on Vulkan.** Both were still
  climbing at 32, so "peak at concurrency 32" is the sweep's ceiling, not the engine's.
  Extend to 64 and 128 before quoting a maximum. Reporting the ceiling as a peak would
  be the same error class as a silent truncation.
- **`llama-server` leads the decode sweep at every level.** Consistent with the neutral
  control; fox's advantage is not throughput.
- **Ollama on ROCm collapses past 8 clients** — 221 tok/s at 8, then 140 at 16 and 133 at
  32, with efficiency down to 9% and ITL p99 at 204 ms. On Vulkan it scales normally to
  32. Something in the ROCm path degrades under concurrency; not root-caused, and it
  should be reproduced before it goes in a paper.

### Noisy neighbour — the workload where the gap is largest

`MODE=noisy`: 4 interactive clients streaming short chats continuously, then one ~4000-token
prompt injected. Everything is measured as inter-token latency inside the injection
window, because all three engines produce identical average throughput over the run — the
damage is a freeze in somebody else's stream, and an average cannot see it.

| | ITL p99 before | during | factor | long prefill |
|---|---|---|---|---|
| **Vulkan** | | | | |
| fox | 51 ms | **278 ms** | **5.5×** | 1972 ms |
| `llama-server` | 21 ms | 940 ms | 43.8× | 1760 ms |
| Ollama | 40 ms | 1059 ms | 26.3× | 2194 ms |
| **ROCm** | | | | |
| fox | 60 ms | **664 ms** | **11.1×** | 4819 ms |
| `llama-server` | 23 ms | 2329 ms | 100.8× | 4618 ms |
| Ollama | 46 ms | 900 ms | 19.5× | 1894 ms |

**RETRACTED as an architectural claim — it is a default.** See "the noisy-neighbour
advantage is a flag" below. The numbers stand; the interpretation does not.

This is the largest separation any workload here produces, and it is the one a user feels
most directly. But it comes with a finding that must be published next to it:

**fox has the worst baseline jitter of the three.** 51-60 ms ITL p99 at rest against
`llama-server`'s 21-23 ms. fox does not win by having a smoother stream; it wins by not
freezing the stream when a long prefill arrives. Stating only the factor would be
misleading — a reader who measures idle jitter would find fox 2.4× worse and conclude the
whole table was cooked.

### The noisy-neighbour advantage is a flag, not a design

Traced by arithmetic first: the stall an interactive stream suffers is the **prefill chunk
size × per-token prefill cost**. fox chunks at 512 tokens (`--max-prefill-chunk`, default
512); llama.cpp fills `n_batch = 2048` per `llama_decode` (`server-context.cpp:3051`).
Both interleave decode tokens with prefill — only the chunk differs, by 4×.

Tested from both directions, 3 rounds each:

| arm | ITL p99 before | during | factor | long prefill |
|---|---|---|---|---|
| fox, chunk 512 (default) | 50 ms | 273 ms | 5.5× | 1964 ms |
| `llama-server`, n_batch 2048 (default) | 21 ms | 933 ms | 44.3× | 1759 ms |
| **`llama-server` with `-b 512`** | 21 ms | **263 ms** | 12.4× | 1896 ms |
| **fox with chunk 2048** | 50 ms | **976 ms** | 19.5× | 1801 ms |

At a matched chunk the two engines stall **the same amount**: 263 ms against fox's 273 ms.
Give fox llama.cpp's chunk and it degrades to 976 ms, indistinguishable from
`llama-server`'s 933 ms. The advantage is entirely `--max-prefill-chunk 512` being a
quarter of `n_batch = 2048`, and `-b 512` hands it to the reference for free. Publishing
"fox degrades 5.5× where `llama-server` degrades 44×" as an architectural property would
have been exactly the failure the equal-tuning-effort rule exists to prevent — this time
in fox's favour.

It is not free for either: the smaller chunk costs ~8% on the long prefill itself (fox
1964 vs 1801 ms; `llama-server` 1896 vs 1759 ms). That is the real trade — interactive
smoothness against prefill throughput — and it is a flag both engines expose.

**The headline metric was also wrong.** The "factor" is a ratio against each engine's own
baseline, so it rewards an engine for having *worse* idle jitter. At a matched chunk
`llama-server` shows a worse factor (12.4× vs 5.5×) while suffering a *smaller* absolute
stall (263 vs 273 ms) — purely because its baseline is 21 ms against fox's 50 ms. Report
the **absolute stall**; the ratio flatters whoever starts out rougher.

### The KPIs that were missing

Two were added because their absence was hiding real behaviour, not to pad the table.

**Inter-token latency.** Adding ITL to the *existing* burst workload revealed what TTFT
alone had been reporting as a mere 4× difference: in the cold burst, `llama-server`'s ITL
p99 is **872 ms on Vulkan and 2304 ms on ROCm**, against fox's 74-76 ms. The interference
effect was in the data all along and no reported metric could see it.

**GPU memory, split VRAM/GTT.** This iGPU carves out only 2 GB as VRAM; everything else
lands in GTT, system RAM mapped for the GPU. Peaks above an idle baseline, burst workload:

| | VRAM | GTT | GPU busy |
|---|---|---|---|
| fox (Vulkan) | 237 MB | 2481 MB | 66% |
| `llama-server` (Vulkan) | 298 MB | 2069 MB | 84% |
| Ollama (Vulkan) | 238 MB | 2192 MB | 77% |
| fox (ROCm) | 2 MB | 2834 MB | 71% |
| `llama-server` (ROCm) | 3 MB | 2516 MB | 92% |

On ROCm the VRAM figure does not move at all. A VRAM-only memory column would have read
as "nothing allocated" for every engine on that backend.

**fox uses ~400 MB more GTT than `llama-server`, consistently**, while keeping the GPU
busy 66-71% against its 84-92%. Less redundant work, more memory held. Both halves belong
in the table; the second is the cost of the first.

The occupancy number doubles as the assertion that replaced a log that does not exist:
`llama-server` never states its backend anywhere, and this machine has a documented case
of `libggml-hip.so` failing to dlopen and the server falling back to CPU silently. Reading
the driver catches that for every engine. A busy percentage near zero aborts the row.

### The decode deficit: two hypotheses tested, both wrong

fox sits ~10% below `llama-server` on decode-bound throughput. Two explanations were
tested directly rather than argued about. Neither survived.

**Hypothesis 1 — `kv_unified = true` costs decode throughput.** Tested with
`FOX_KV_UNIFIED=0`, a runtime switch so both arms come from **one binary** (building two
would put the build into the comparison). Arm `fox-seq` in `scripts/bench_engines.sh`.

| decode, conc 4 | per request | aggregate |
|---|---|---|
| fox (unified) | 45 tok/s | 169.6 |
| fox-seq (not unified) | 46 tok/s | 172.8 |
| `llama-server` | **50 tok/s** | **184.6** |

Turning unified KV off recovers **2%**, not 10. It is not the lever. **Prediction was
wrong and is retracted.**

What the same run *did* price is the trade itself, and it is lopsided:

| | fox | fox-seq |
|---|---|---|
| cold burst TTFT p50 | **1108 ms** | 6300 ms |
| warm burst TTFT p50 | **51 ms** | 5987 ms |

Unified KV buys 5.7× cold and 117× warm TTFT for 2% of decode. Caveat that must travel
with those numbers: `fox-seq` has **no prompt reuse at all** — without a unified cache
fox cannot reuse, whereas `llama-server` without one still reuses from *idle* slots. So
this prices "fox with vs without its prefix cache", not "unified vs non-unified with
equivalent reuse". Only the 2% decode figure isolates unified KV cleanly, because that
workload has nothing to reuse either way.

Also from that run: `fox-seq` degrades 6.3× under a noisy neighbour against fox's 5.5×.
fox's resistance to prefill interference is therefore **not** coming from its prefix
cache — it survives with the cache off. That mechanism is still unidentified.

**Hypothesis 2 — fox fragments its decode batches.** Precedent existed: seq_id
fragmentation was once measured at 1.74 of a possible 4. Read from llama.cpp itself via
`LLAMA_BATCH_DEBUG=1` (never by patching `vendor/`); in decode each sequence contributes
exactly one token, so a ubatch's `n_tokens` *is* the batch fill.

fox at 4 clients: **3.89 of 4**, with 248 of 262 steps completely full. Essentially no
fragmentation; it accounts for ~3% at most. **Also not the lever.**

(`llama-server`'s equivalent trace was not captured — its log never reached DEBUG level
even at `-lv 4`. fox's own figure is enough to rule out gross fragmentation on fox's
side, but the comparison is one-sided and should be completed.)

**Found by profiling: it is the sampler's candidate selection.** `perf record` on both
servers under the same decode workload, 4 clients, one server at a time. (The `perf` on
`PATH` is a stub with no binary for this kernel; `/usr/lib/linux-tools-6.8.0-136/perf`
samples user space fine, which is all this needs.)

| self cost | fox | `llama-server` |
|---|---|---|
| waiting on the GPU fence | 78.7% | 82.8% |
| **sorting** | **6.61%** `quicksort::partition` | **1.39%** `llama_token_data_array_partial_sort_inplace` |
| sampler proper | 2.33% `sample_token` | 2.63% `common_sampler_sample` |
| output filter | 0.93% | — |
| **total CPU outside the GPU wait** | **~9.9%** | **~4.0%** |

The ~5.9% difference is the size of the unexplained gap. The mechanism is at
`src/engine/model/sampling.rs:189`, executed once per token **per sequence**:

```rust
let mut idx: Vec<usize> = (0..logits.len()).collect();   // 128256 × 8 B = 1 MB, per token
idx.select_nth_unstable_by(k - 1, |&a, &b| {
    logits[b].partial_cmp(&logits[a])                    // indirect: chases a 512 KB array
```

Two costs `llama-server` does not pay: a **1 MB allocation per token**, and a comparator
that **dereferences into a separate logits array** on every comparison while permuting the
index array. llama.cpp keeps `llama_token_data` (id and logit adjacent) contiguous and
partial-sorts it in place, so its comparisons read the value they are sorting by.

This also explains the shape of the curve, which nothing else did. The GPU decode step is
weight-bound and barely grows from 1 to 4 sequences, while this CPU cost is paid once per
sequence and grows linearly. So its *share* rises with concurrency: ~2% at 1-2 clients,
~10% from 4 upward — exactly the measured step.

Note what is **not** implicated: `logits.to_vec()`, which the older docs blamed, really is
~0.5%. The copy was never the problem; the selection over the copy is.

**Fixed and validated, 2026-08-03.** `select_top_n` (`src/engine/model/sampling.rs`)
keeps a sorted buffer of at most n entries and streams the logits once: the common case
per element is one `f32` compare against a running threshold, sequential, no indirection,
no allocation proportional to the vocabulary.

Validated end-to-end over 3 rounds, not with a micro-benchmark — this repo has a
precedent of a 4.6× sampling micro-benchmark win producing zero real throughput.

| decode, conc 4 | before | after |
|---|---|---|
| fox per request | 45 tok/s [45, 46] | **47 tok/s [47, 47]** |
| fox aggregate | 170 tok/s | **175 tok/s** |
| gap vs `llama-server` | 1.10× | **1.06×** |

No regression in the burst workload: cold TTFT 1108 → 1100 ms, warm 51 → 47 ms, and
`cached_tokens` identical at 12908/14840, so prefix reuse is untouched.

**The sweep-based claims first published for this fix were withdrawn** — see "the
neutral control was not neutral" below. Only the conc-4 decode figures above survive,
because they come from arms alternating inside one run with disjoint ranges. Validating
the fix at higher concurrency needs an old-sampler arm inside the same run, the way
`fox-seq` was done; comparing sweeps across sessions cannot carry it.

The new unit tests **do not run in CI**: the whole sampling module is
`#[cfg(not(fox_stub))]` and CI runs with `FOX_SKIP_LLAMA=1` (331 tests there against 430
in a real build). A sampler regression would not be caught by `make ci`.

### The neutral control was not neutral above 16 clients

`bench_decode.py` held 16 prompts and handed them out with `i % 16`, so from 32 clients
upward two clients got **byte-identical prompts** — precisely what fox's prefix cache
exists to reuse. The control turned into the favourable workload at exactly the
concurrencies where the sweep was making its strongest claims. Fixed by putting the
client index first, so two clients share one token instead of a whole prompt.

The bias was real and one-sided, which is itself a demonstration of the paper's thesis:

| conc 32, aggregate | duplicate prompts | unique prompts |
|---|---|---|
| fox | 641 | **570** (−11%) |
| `llama-server` | 664 | 673 (+1%) |

fox gained 11% from the duplicates and `llama-server` nothing, because `llama-server`
cannot reuse from a live sibling and fox can. **Every sweep figure at 32 published before
this fix is inflated in fox's favour and is retracted**, including "the deficit at 32
goes from 13.5% to 3.5%" — measured cleanly the deficit at 32 is **15%**.

A second lesson from the same comparison: two post-fix runs at 16 clients gave 423 and
400 tok/s, a 5.7% spread between sessions, while the within-run ranges were ±1%. Sweep
numbers are **not comparable across sessions** at better than ~6%, so any A/B on them
must run both arms inside one alternating run.

### Saturation, to 128 clients — fox has a ceiling and `llama-server` does not

3 rounds, alternating arms, unique prompts, Vulkan.

| conc | fox | range | `llama-server` | range |
|---|---|---|---|---|
| 1 | 53 | [53, 53] | 54 | [54, 54] |
| 4 | 176 | [174, 176] | 190 | [187, 190] |
| 8 | 250 | [249, 255] | 275 | [269, 278] |
| 16 | 400 | [399, 404] | 432 | [431, 435] |
| 32 | 570 | [570, 570] | 673 | [673, 675] |
| 64 | **610** | [604, 617] | 782 | [780, 789] |
| 128 | **416** | [414, 416] | **843** | [680, 871] |

**fox peaks at 64 clients and then collapses**: 610 → 416 tok/s, scaling efficiency down
to 6%, and ITL p99 at **400 ms** against `llama-server`'s 124 ms. `llama-server` never
bends inside this range — it is still climbing at 128, so its own knee is beyond what was
measured (and its 128 range, [680, 871], is wide enough that the level is unstable).

At 128 clients `llama-server` serves **2.03× fox's throughput**. This is a far more
important result than the sampler fix.

**Cause found: the unified KV cache.** Same sweep with the `fox-seq` arm
(`FOX_KV_UNIFIED=0`, same binary), 3 rounds, ranges disjoint against fox at every level:

| conc | fox | fox-seq | `llama-server` | cost of unified KV |
|---|---|---|---|---|
| 16 | 392 [391, 394] | 417 [415, 420] | 435 [431, 439] | 6% |
| 32 | 568 [561, 572] | 644 [641, 656] | 658 [647, 665] | 13% |
| 64 | 611 [611, 616] | 778 [771, 790] | 777 [772, 787] | 27% |
| 128 | **422** [420, 424] | **876** [869, 876] | 845 [794, 858] | **108%** |

`fox-seq` does not bend at all: it matches `llama-server` at 64 (778 vs 777) and passes it
at 128 (876 vs 845). **The entire collapse is the unified KV cache**, and nothing else in
fox is implicated — scheduler, admission budget and sampler are common to both arms.

The mechanism follows from `n_stream = 1`: with one shared cell pool, a decode step
attends over the union of every sequence's cells, so cost grows as N·(N·L) instead of
N·L. Measured decode-step time confirms the shape — doubling clients from 64 to 128
multiplies fox's step time by **2.93** and `llama-server`'s by 1.85.

ITL p99 at 128 tells the same story from the user's side: fox 391 ms, fox-seq 176 ms,
`llama-server` 125 ms.

**This corrects an earlier conclusion in this document.** "Turning unified KV off recovers
2%, it is not the lever" was measured at **4 clients**, where it is true. As a general
statement it was wrong: the cost is not a fixed percentage but a curve — 2% at 4 clients,
108% at 128.

So fox's central design choice is now priced on both sides, and the trade is sharp rather
than free:

| unified KV buys | unified KV costs |
|---|---|
| 5.7× cold TTFT, 117× warm (prefix reuse from a *live* sibling) | 6% throughput at 16 clients, 108% at 128 |
| the noisy-neighbour advantage is **not** among them — `fox-seq` degrades 6.3× vs fox's 5.5×, so that comes from somewhere else | a concurrency ceiling at ~64 |

The obvious follow-up is a design question, not a measurement: the mode is currently
compile-time-fixed, and `FOX_KV_UNIFIED` exists only as a measurement switch. Choosing it
per load — unified while concurrency is low and prefixes are shared, non-unified above the
knee — is not implemented and would need the switch to be safe to flip on a live model,
which it is not today.

It also bounds every "fox is within X% of `llama-server`" claim in this document to
**concurrency ≤ 16**. Above that the gap widens: 15% at 32, 22% at 64, 103% at 128.

Memory at 128 clients: `llama-server` peaks at **+17.5 GB of GTT**, which is the KV cache
for 128 × 4096 tokens. It fits only because this machine shares 123 GB of system RAM with
the GPU.

### vLLM — its own section, 2026-08-03

`scripts/bench_vllm.sh`, 3 rounds, server restarted per round. `rocm/vllm:latest`
(v0.11.2.dev), `HSA_OVERRIDE_GFX_VERSION=11.0.0`, `--max-model-len 4096 --max-num-seqs 8
--enable-prefix-caching --gpu-memory-utilization 0.55`, BF16 safetensors
(`unsloth/Llama-3.2-1B-Instruct`), ROCm. Startup 40-46 s per round. Same clients, same
workloads as the trio.

| workload | vLLM |
|---|---|
| cold TTFT p50 | 1995 ms, range [1975, 2058] |
| warm TTFT p50 | 669 ms, range [654, 711] |
| decode per request | 19.3 tok/s |
| decode aggregate | 75.6 tok/s |

**Do not put this column next to the trio's.** Backend and weight format both differ.
The decode figure in particular is mostly explained by the weights, not the serving
layer: BF16 moves roughly twice the bytes per token that Q8_0 does, and decode on this
iGPU is memory-bound, which is about the whole of the 45 → 19 tok/s difference. Saying
"fox decodes 2.3× faster than vLLM" from this table would be quoting a quantisation
choice as an engine result.

The first vLLM run of the day reported a **2758 ms** cold TTFT against the 1995 ms
measured here. The difference is `torch.compile`'s cache being cold on the very first
start; it is discarded rather than averaged in, and any future run should throw away its
first start for the same reason. Nothing equivalent applies to the other three.

### `cached_tokens` reads 0 for two different reasons — checked, not assumed

`scripts/probe_cached_tokens.py` sends the same prompt twice, streamed and non-streamed,
and reports whether `prompt_tokens_details` comes back at all.

| engine | `prompt_tokens_details` | so its 0 means |
|---|---|---|
| fox | present (12908 cold, 14840 warm) | real reuse, measured |
| `llama-server` | present (0 cold, 14840 warm) | real: none cold, full warm |
| Ollama | **absent**, both streamed and not | not reported |
| vLLM | **absent**, both streamed and not | not reported |

Both engines that report nothing show a large warm TTFT drop (Ollama 5377 → 400 ms, vLLM
1995 → 669 ms), so they are reusing prefixes and simply not exposing the counter.
Publishing their 0 in a "cached tokens" column would state the opposite of what happened,
which is why the column has to carry the distinction rather than the number alone.

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
