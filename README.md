<div align="center">

<img src="assets/fox.svg" alt="fox" width="420">

**A local LLM server built for concurrent work. Drop-in replacement for Ollama.**

[![CI](https://github.com/ferrumox/fox/actions/workflows/ci.yml/badge.svg)](https://github.com/ferrumox/fox/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)
[![Version](https://img.shields.io/badge/version-0.22.1-green.svg)](CHANGELOG.md)
[![Rust](https://img.shields.io/badge/rust-stable-brightgreen.svg)](https://rustup.rs/)
[![GitHub Stars](https://img.shields.io/github/stars/ferrumox/fox?style=social)](https://github.com/ferrumox/fox/stargazers)

[![Sponsor](https://img.shields.io/badge/❤️_Sponsor-ea4aaa?style=for-the-badge&logo=github-sponsors&logoColor=white)](https://github.com/sponsors/manuelslemos)

<img src="assets/demo.gif" alt="fox answering the same prompt over its OpenAI and Ollama APIs on one port" width="860">

</div>

Fox is dual-licensed MIT OR Apache-2.0 and stays that way. There is no paid tier and no plan for one.

---

## Try it in 30 seconds

```bash
# Linux x86_64 — picks the Vulkan build when a GPU is present, CPU otherwise
curl -fsSL https://github.com/ferrumox/fox/releases/latest/download/install.sh | sh
```

macOS and Windows: build from source (below), or run the Linux installer under WSL2.
Prebuilt binaries are Linux x86_64 for now.

```bash
# Pull a model and start (qwen3.6 is 22 GB; qwen3.5 is 2.7 GB if you want a quicker first run)
fox pull qwen3.6
fox serve

# Ask something (OpenAI-compatible)
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6","messages":[{"role":"user","content":"Hello!"}],"stream":true}'

# If you already use Ollama — just change the port from 11434 to 8080. That's it.
```

---

## Performance

Fox wraps llama.cpp, so a single request decoding on its own runs the same kernels
`llama-server` runs. There is no room for fox to be dramatically faster at that, and it
isn't. Where fox pulls ahead is when requests arrive together and share a prompt.

Radeon 890M, Vulkan, Llama-3.2-1B-Instruct-Q8_0, 1856-token shared system prompt.
Both servers built from the same vendored llama.cpp, one running at a time, arms
alternated across 3 rounds. All ranges below are disjoint.

| Workload | fox | llama-server |
|---|---|---|
| 8 clients, shared prompt, cold — TTFT p50 | **1129 ms** | 4550 ms |
| 16 clients, shared prompt, cold — TTFT p50 | **1402 ms** | 8064 ms |
| 16 clients, whole burst wall clock | **3.8 s** | 16.2 s |
| 4 clients, short unrelated prompts — throughput | 96% of llama-server | baseline |

Doubling the clients costs fox 24% more time to first token and `llama-server` 79%.

That last row is not a typo and it is not buried on purpose: on single-turn requests with
short prompts, fox is about 4% behind. That workload cannot see any of the work fox does,
because there is no prompt worth reusing. If your traffic looks like that, fox will not
make it faster.

Reproduce either one:

```bash
scripts/ab_shared_prefix.sh    # concurrent burst behind a shared prompt
scripts/ab_bench.sh            # decode-bound throughput
```

Full methodology, including two ways these benchmarks produced convincing wrong answers
before they produced right ones, is in `docs/design/rocm-benchmarking-2026-08.md`.

Numbers against Ollama are pending re-measurement on current hardware. The figures that
used to sit here were from an RTX 4060 with no recorded methodology, and this project's
rule is that a before/after claim comes from `scripts/ab_bench.sh` or it does not get
published.
---

## How it works

**Sequences remember what they hold.** Every sequence keeps the tokens resident in its
KV cache, including the tokens it generated. A new request is matched to the sequence
sharing the longest prefix with it and skips the prefill for that overlap. In a chat, the
second turn does not re-read the first.

**Requests can copy a prefix from a live sequence.** This is the part other llama.cpp
servers do not do. Slot affinity normally reuses an idle sequence, so when eight requests
carrying the same system prompt arrive at once, none of them can reuse anything and all
eight prefill the same tokens. Fox copies the shared prefix out of a sibling that is
already decoding. `llama-server` cannot: its slot selection skips busy slots in both its
similarity pass and its LRU fallback.

**A shared prefix is paid for once.** Sequences sharing a prefix share the block budget
for it instead of each reserving a copy, so the server admits as much concurrency as the
hardware actually holds.

**Requests do not queue behind each other.** Continuous batching decodes concurrent
requests in the same pass, so a long generation for one client does not delay a short
question from another.

More on the engine — prompt reuse, speculative decoding, structured output, vision, LoRA,
multi-GPU — in [`docs/features.md`](docs/features.md).

---

## Works with every tool you already use

**No code changes needed** — just change the base URL to `http://localhost:8080`.

| Client / Tool | Protocol | Status |
|---------------|----------|--------|
| Open WebUI | Ollama | ✓ Works out of the box |
| Continue.dev | Ollama | ✓ Works out of the box |
| LangChain | OpenAI | ✓ Works out of the box |
| LlamaIndex | OpenAI | ✓ Works out of the box |
| Cursor / Copilot Chat | OpenAI | ✓ Works out of the box |
| `ollama` CLI | Ollama | ✓ Works out of the box |
| `openai` Python SDK | OpenAI | ✓ Works out of the box |

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-local")

resp = client.chat.completions.create(
    model="qwen3.6",
    messages=[{"role": "user", "content": "Say hi in 5 words."}],
)
print(resp.choices[0].message.content)
```

JavaScript, LangChain, LlamaIndex and IDE configuration are in
[`docs/integrations.md`](docs/integrations.md); runnable snippets in
[`examples/`](examples/). Coming from Ollama, start at
[`docs/migration-from-ollama.md`](docs/migration-from-ollama.md).

---

## Installation

```bash
# Linux x86_64 — picks the Vulkan build when a GPU is present, CPU otherwise
curl -fsSL https://github.com/ferrumox/fox/releases/latest/download/install.sh | sh
```

It detects `/dev/dri`, verifies the published checksum, and tells you if `$PREFIX/bin` is
not on your `PATH`. Override with `--vulkan`, `--cpu`, `--version vX.Y.Z` or `--prefix`.

```bash
# From source — --recurse-submodules is not optional, llama.cpp is vendored
git clone --recurse-submodules https://github.com/ferrumox/fox
cd fox && cargo build --release

# Docker
docker run -p 8080:8080 -v ~/.cache/ferrumox/models:/root/.cache/ferrumox/models \
  ferrumox/fox serve
```

Prebuilt binaries are Linux x86_64 for now; macOS and Windows build from source or run the
installer under WSL2. Tarball layout, checksum verification and model storage are in
[`docs/installation.md`](docs/installation.md).

**One binary, any GPU.** Backends are compiled as shared libraries and chosen at runtime —
CUDA → ROCm → Vulkan → Metal → CPU, in that order of preference.

| Backend | Requirement |
|---------|-------------|
| CPU | x86_64 or arm64, AVX2 |
| CUDA | CUDA 12.x, Linux/Windows x86_64 |
| ROCm | ROCm 6.2+, Linux x86_64 |
| Metal | macOS 13+, Apple Silicon |
| Vulkan | Vulkan SDK 1.3+, Linux or Windows x86_64 |

There are no runtime dependencies beyond GPU drivers. The `.so` files in the release
tarball must stay beside the binary: `fox` is linked `RPATH=$ORIGIN` and looks nowhere
else.

---

## Everyday commands

```bash
fox search qwen coder        # search HuggingFace for GGUF models
fox pull qwen3.6             # or gemma3:12b-q4, or a full HF repo path
fox serve                    # lazy loading — no model needed upfront
fox run                      # interactive REPL
fox list / ps / show / rm    # manage what is on disk and what is loaded
fox bench qwen3.6            # measure it yourself
```

Every subcommand and flag is documented in [`docs/cli/`](docs/cli/); a guided first run is
in [`docs/quickstart.md`](docs/quickstart.md).

---

## APIs

Both families are served on the same port, so an existing client only needs its base URL
changed.

- **OpenAI-compatible** — `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`,
  `/v1/models`. Reference: [`docs/api/openai.md`](docs/api/openai.md).
- **Ollama-compatible** — `/api/chat`, `/api/generate`, `/api/embed`, `/api/tags`,
  `/api/ps`, `/api/show`, `/api/pull`. Reference:
  [`docs/api/ollama.md`](docs/api/ollama.md).
- **Beyond both** — `/infill`, `/rerank`, `/tokenize`, `/apply-template`, `/props`,
  `/slots`, `/lora-adapters`, `/health`, and `/metrics` for Prometheus.

What fox promises not to break across versions, and what it does not, is written down in
[`COMPATIBILITY.md`](COMPATIBILITY.md).

---

## Configuration

Every flag has a `FOX_*` environment variable and a key in
`~/.config/ferrumox/config.toml`. Precedence is flag > env > file.

| Flag | Default | What it does |
|------|---------|--------------|
| `--port` | `8080` | Bind port |
| `--max-models` | `1` | Models held in memory at once, LRU eviction |
| `--keep-alive-secs` | `300` | Evict an idle model after N seconds (0 = never) |
| `--gpu-memory-fraction` | `0.85` | Share of GPU RAM given to the KV cache |
| `--type-kv` | `f16` | KV cache quantization: `f16`, `q8_0`, `q4_0` |
| `--api-key` | — | Require `Authorization: Bearer <key>` |

The full table — multi-GPU split, MoE offload, batch and block sizing, aliases, logging —
is in [`docs/configuration.md`](docs/configuration.md).

---

## Documentation

| | |
|---|---|
| [Quick start](docs/quickstart.md) | First run, start to finish |
| [Installation](docs/installation.md) | Binaries, Docker, source, model storage |
| [Configuration](docs/configuration.md) | Every flag, env var and config key |
| [How fox works](docs/features.md) | The engine, in depth |
| [OpenAI API](docs/api/openai.md) · [Ollama API](docs/api/ollama.md) | Endpoint reference |
| [CLI](docs/cli/) | Every subcommand |
| [Integrations](docs/integrations.md) | Python, JS, LangChain, IDEs |
| [Migrating from Ollama](docs/migration-from-ollama.md) | What changes, what does not |
| [Benchmarks](docs/benchmarks.md) | How to measure fox yourself |
| [Troubleshooting](docs/troubleshooting.md) · [FAQ](docs/faq.md) | When it does not work |
| [Deployment](docs/deployment.md) | systemd, Docker, reverse proxies |
| [Feature status](STATUS.md) | What works, what does not, honestly |

---

## Community

- **Bug reports**: [GitHub Issues](https://github.com/ferrumox/fox/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ferrumox/fox/discussions)
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md) — build, test, architecture, and how a release is cut

```bash
FOX_SKIP_LLAMA=1 cargo test --all    # the suite, without needing llama.cpp or a GPU
```

---

## Support the project

Fox is built and maintained by [Manuel S. Lemos](https://github.com/manuelslemos) in his
spare time. Every feature is in the free build and will stay there.

If fox saves you time or replaces an API bill, sponsorship pays for the time that keeps it
maintained.

| Tier | What you get |
|---|---|
| $5 / month | Sponsor badge |
| $25 / month | Your issues get looked at first, and your name in [SPONSORS.md](SPONSORS.md) |
| $100 / month | Your logo in this README and a mention in each release |
| $500 / month | A direct line, and a say in what gets built next |

[GitHub Sponsors](https://github.com/sponsors/manuelslemos) · [Buy Me a Coffee](https://buymeacoffee.com/manuelslemos)

---

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE). Take either.
