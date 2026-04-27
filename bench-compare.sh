#!/usr/bin/env bash
set -euo pipefail

# Text inference benchmark: fox vs llama-server vs Ollama
# Runs each server sequentially on the same GPU, measures with fox-bench.
#
# Designed to run inside a CUDA devcontainer where all three binaries
# are installed natively (no docker-in-docker).
#
# Usage:
#   bash bench-compare.sh                            # defaults
#   MODEL_PATH=/path/to/model.gguf bash bench-compare.sh
#
# Expects: fox, fox-bench, llama-server, ollama installed.

MODEL_PATH="${MODEL_PATH:-/workspace/models/google_gemma-4-26B-A4B-it-Q4_K_M.gguf}"
MODEL_NAME="$(basename "$MODEL_PATH" .gguf)"
MODEL_DIR="$(dirname "$MODEL_PATH")"
FOX="${FOX:-/workspace/target/release/fox}"
BENCH="${BENCH:-/workspace/target/release/fox-bench}"
LLAMA_SERVER="${LLAMA_SERVER:-/workspace/vendor/llama.cpp/build/bin/llama-server}"

REQUESTS="${REQUESTS:-60}"
MAX_TOKENS="${MAX_TOKENS:-128}"
MAX_CTX="${MAX_CTX:-4096}"
MAX_BATCH="${MAX_BATCH:-4}"

FOX_PORT=8080
LLAMA_PORT=8082
OLLAMA_PORT=11434

RESULTS_DIR="${RESULTS_DIR:-/workspace/bench_results/compare}"
mkdir -p "$RESULTS_DIR"

CONCURRENCIES="${CONCURRENCIES:-1 2 4 8}"

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

# ── Prompt ─────────────────────────────────────────────────────────────────
read -r -d '' PROMPT << 'PROMPT_EOF' || true
You are a senior software architect reviewing a complex distributed system. Below is a detailed technical specification for a new microservice. Please analyze it thoroughly, identify potential issues, and provide recommendations.

SYSTEM SPECIFICATION: Real-Time Event Processing Pipeline

1. ARCHITECTURE OVERVIEW
The system processes financial transaction events from 47 regional data centers, aggregating them into a unified stream for fraud detection, compliance reporting, and real-time analytics. The expected throughput is 2.3 million events per second at peak, with a 99th percentile latency requirement of 50 milliseconds end-to-end.

2. DATA FLOW
Events arrive via Apache Kafka topics partitioned by region. Each event contains: transaction ID (UUID v7), timestamp (nanosecond precision), source account, destination account, amount, currency code (ISO 4217), merchant category code (MCC), device fingerprint hash, and geolocation coordinates. The average event size is 847 bytes after Protocol Buffers serialization.

The ingestion layer consists of 12 consumer groups, each with 8 consumers. Events are deserialized, validated against a JSON Schema, enriched with account metadata from a Redis Cluster (6 primary nodes, 18 replicas), and then routed to one of three processing pipelines based on risk score.

3. PROCESSING PIPELINES
Pipeline A (Low Risk, 78% of traffic): Events are batched in 100ms windows and written directly to Apache Parquet files on S3-compatible object storage. A separate Apache Spark job runs every 15 minutes to compute aggregate statistics.

Pipeline B (Medium Risk, 19% of traffic): Events pass through a rules engine implemented as a directed acyclic graph of 342 evaluation nodes. Each node applies a specific fraud detection heuristic. The rules engine maintains a sliding window of the last 1000 transactions per account in a custom B-tree backed by memory-mapped files.

Pipeline C (High Risk, 3% of traffic): Events undergo real-time ML inference using an ensemble of three models: a gradient boosted decision tree (XGBoost), a neural network (ONNX Runtime), and a graph neural network that analyzes transaction networks. Model inference must complete within 15ms.

4. STATE MANAGEMENT
The system maintains several categories of distributed state: per-account transaction history, per-merchant fraud scores updated every 30 seconds, global feature vectors refreshed every 5 minutes, and circuit breaker states for downstream dependencies.

Given this specification, provide a concise architectural review focusing on the most critical risks and the single most impactful improvement you would recommend. Be specific and reference the section numbers.
PROMPT_EOF

# ── Helpers ────────────────────────────────────────────────────────────────

cleanup_all() {
    pkill -f "fox serve" 2>/dev/null || true
    pkill -f llama-server 2>/dev/null || true
    pkill -f "ollama serve" 2>/dev/null || true
    sleep 2
}

wait_for_server() {
    local url="$1" max_wait="${2:-120}" waited=0
    while ! curl -sf "${url}/health" >/dev/null 2>&1 \
       && ! curl -sf "${url}/v1/models" >/dev/null 2>&1; do
        sleep 2; waited=$((waited + 2))
        if [ $waited -ge $max_wait ]; then
            echo "  ERROR: server did not start within ${max_wait}s" >&2
            return 1
        fi
    done
}

wait_for_model() {
    local url="$1" model="$2" max_wait="${3:-120}" waited=0
    echo -n "  waiting for model to load..."
    while true; do
        local resp
        resp=$(curl -s --max-time 5 "${url}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"${model}\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":1,\"stream\":false}" 2>&1) || true
        if echo "$resp" | grep -q '"completion_tokens"'; then
            echo " ready"
            return 0
        fi
        sleep 3; waited=$((waited + 3))
        if [ $waited -ge $max_wait ]; then
            echo " TIMEOUT"
            return 1
        fi
    done
}

stop_pid() {
    local pid="$1"
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    sleep 2
}

run_bench_sweep() {
    local label="$1" url="$2" model="$3"
    for conc in $CONCURRENCIES; do
        local reqs=$((REQUESTS / 4 * conc))
        [ "$reqs" -lt "$conc" ] && reqs=$conc
        local tag="${label}_c${conc}"
        echo "  bench $tag (c=$conc n=$reqs)"
        "$BENCH" \
            --url "$url" \
            --model "$model" \
            --concurrency "$conc" \
            --requests "$reqs" \
            --max-tokens "$MAX_TOKENS" \
            --prompt "$PROMPT" \
            --output json \
            > "${RESULTS_DIR}/${tag}.json" 2>&1 || true
    done
}

echo -e "${BOLD}════════════════════════════════════════════════${NC}"
echo -e "${BOLD} Inference Benchmark: fox vs llama.cpp vs Ollama${NC}"
echo -e "${BOLD}════════════════════════════════════════════════${NC}"
echo "  model:   $MODEL_NAME"
echo "  ctx:     $MAX_CTX  batch: $MAX_BATCH"
echo "  conc:    $CONCURRENCIES"
echo "  reqs:    $REQUESTS base"
echo "  tokens:  $MAX_TOKENS max"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# 1. Fox
# ═══════════════════════════════════════════════════════════════════════════
echo -e "${BOLD}[1/3] Fox${NC}"
cleanup_all
if [ ! -x "$FOX" ]; then
    echo "  SKIP: $FOX not found"
else
    "$FOX" serve \
        --model-path "$MODEL_PATH" \
        --models-dir "$MODEL_DIR" \
        --max-batch-size "$MAX_BATCH" \
        --max-context-len "$MAX_CTX" \
        --flash-attn true \
        --chunked-prefill-tokens 512 \
        --port "$FOX_PORT" \
        > "${RESULTS_DIR}/server_fox.log" 2>&1 &
    FOX_PID=$!
    wait_for_server "http://localhost:${FOX_PORT}"
    wait_for_model "http://localhost:${FOX_PORT}" "$MODEL_NAME"

    run_bench_sweep "fox" "http://localhost:${FOX_PORT}" "$MODEL_NAME"

    stop_pid "$FOX_PID"
    echo ""
fi

# ═══════════════════════════════════════════════════════════════════════════
# 2. llama-server (stock llama.cpp)
# ═══════════════════════════════════════════════════════════════════════════
echo -e "${BOLD}[2/3] llama-server${NC}"
cleanup_all
if [ ! -x "$LLAMA_SERVER" ]; then
    echo "  SKIP: $LLAMA_SERVER not found"
else
    # llama-server --ctx-size is TOTAL KV cache (shared across --parallel slots).
    # fox allocates max_ctx_per_seq * n_seq total. Match them:
    LLAMA_TOTAL_CTX=$((MAX_CTX * MAX_BATCH))
    "$LLAMA_SERVER" \
        --model "$MODEL_PATH" \
        --host 0.0.0.0 --port "$LLAMA_PORT" \
        --ctx-size "$LLAMA_TOTAL_CTX" \
        --flash-attn on \
        -ngl 99 \
        --parallel "$MAX_BATCH" \
        > "${RESULTS_DIR}/server_llama.log" 2>&1 &
    LLAMA_PID=$!
    wait_for_server "http://localhost:${LLAMA_PORT}" 180
    wait_for_model "http://localhost:${LLAMA_PORT}" "$MODEL_NAME" 180

    run_bench_sweep "llama" "http://localhost:${LLAMA_PORT}" "$MODEL_NAME"

    stop_pid "$LLAMA_PID"
    echo ""
fi

# ═══════════════════════════════════════════════════════════════════════════
# 3. Ollama
# ═══════════════════════════════════════════════════════════════════════════
echo -e "${BOLD}[3/3] Ollama${NC}"
cleanup_all
OLLAMA_MODEL="gemma4-bench"
if ! command -v ollama &>/dev/null; then
    echo "  SKIP: ollama not installed"
else
    # Start ollama serve in background
    OLLAMA_HOST="0.0.0.0:${OLLAMA_PORT}" ollama serve \
        > "${RESULTS_DIR}/server_ollama.log" 2>&1 &
    OLLAMA_PID=$!
    sleep 3
    wait_for_server "http://localhost:${OLLAMA_PORT}" 60

    # Create model from same GGUF for fair comparison
    if ! ollama list 2>/dev/null | grep -q "$OLLAMA_MODEL"; then
        echo "  creating model from GGUF..."
        cat > /tmp/Modelfile.bench <<EOF
FROM $MODEL_PATH
PARAMETER num_ctx $MAX_CTX
EOF
        ollama create "$OLLAMA_MODEL" -f /tmp/Modelfile.bench
        rm -f /tmp/Modelfile.bench
    fi
    echo "  server ready (pid $OLLAMA_PID)"

    # Warmup — ollama loads model on first request
    echo "  warming up (first request loads model)..."
    curl -s --max-time 120 "http://localhost:${OLLAMA_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$OLLAMA_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":4}" \
        > /dev/null 2>&1 || true
    echo "  model loaded"

    run_bench_sweep "ollama" "http://localhost:${OLLAMA_PORT}" "$OLLAMA_MODEL"

    stop_pid "$OLLAMA_PID"
    echo ""
fi

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD} RESULTS: $MODEL_NAME${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════════════════════════════════${NC}"
printf "%-12s  %4s  %9s  %8s  %8s  %8s  %8s  %6s\n" \
    "server" "conc" "tput(t/s)" "TTFT p50" "TTFT p95" "lat p50" "lat p95" "errors"
echo "───────────────────────────────────────────────────────────────────────────────"

for label in fox llama ollama; do
    for conc in $CONCURRENCIES; do
        tag="${label}_c${conc}"
        file="${RESULTS_DIR}/${tag}.json"
        if [ ! -f "$file" ] || ! grep -q "throughput_tokens_per_sec" "$file" 2>/dev/null; then
            printf "%-12s  %4s  %9s\n" "$label" "$conc" "SKIP"
            continue
        fi
        tput=$(grep -o '"throughput_tokens_per_sec": *[0-9.]*' "$file" | head -1 | sed 's/.*: *//')
        ttft50=$(grep -o '"ttft_p50_ms": *[0-9]*' "$file" | head -1 | sed 's/.*: *//')
        ttft95=$(grep -o '"ttft_p95_ms": *[0-9]*' "$file" | head -1 | sed 's/.*: *//')
        lat50=$(grep -o '"latency_p50_ms": *[0-9]*' "$file" | head -1 | sed 's/.*: *//')
        lat95=$(grep -o '"latency_p95_ms": *[0-9]*' "$file" | head -1 | sed 's/.*: *//')
        errs=$(grep -o '"requests_err": *[0-9]*' "$file" | head -1 | sed 's/.*: *//')
        tput_fmt=$(printf "%.1f" "$tput" 2>/dev/null || echo "$tput")
        printf "%-12s  %4s  %9s  %6sms  %6sms  %6sms  %6sms  %6s\n" \
            "$label" "$conc" "$tput_fmt" "$ttft50" "$ttft95" "$lat50" "$lat95" "$errs"
    done
done

echo "═══════════════════════════════════════════════════════════════════════════════"

# ── Bar chart: throughput at highest concurrency ──
TOP_CONC=$(echo $CONCURRENCIES | tr ' ' '\n' | sort -n | tail -1)
echo ""
echo -e "${BOLD}Throughput at c=$TOP_CONC (tok/s, higher is better)${NC}"
MAX_TPUT=0
declare -A TPUT_MAP
for label in fox llama ollama; do
    file="${RESULTS_DIR}/${label}_c${TOP_CONC}.json"
    if [ -f "$file" ] && grep -q "throughput_tokens_per_sec" "$file" 2>/dev/null; then
        t=$(grep -o '"throughput_tokens_per_sec": *[0-9.]*' "$file" | head -1 | sed 's/.*: *//')
        TPUT_MAP[$label]="$t"
        t_int=${t%.*}
        [ "${t_int:-0}" -gt "$MAX_TPUT" ] && MAX_TPUT=$t_int
    fi
done
BAR_WIDTH=40
for label in fox llama ollama; do
    if [ -n "${TPUT_MAP[$label]:-}" ]; then
        t="${TPUT_MAP[$label]}"
        t_int=${t%.*}
        if [ "$MAX_TPUT" -gt 0 ]; then
            bar_len=$(( t_int * BAR_WIDTH / MAX_TPUT ))
        else
            bar_len=0
        fi
        bar=$(printf '%0.s█' $(seq 1 $bar_len 2>/dev/null) 2>/dev/null || true)
        printf "  %-10s %s %.1f tok/s\n" "$label" "$bar" "$t"
    else
        printf "  %-10s SKIPPED\n" "$label"
    fi
done

# ── Bar chart: TTFT p50 at highest concurrency ──
echo ""
echo -e "${BOLD}TTFT p50 at c=$TOP_CONC (ms, lower is better)${NC}"
MAX_TTFT=0
declare -A TTFT_MAP
for label in fox llama ollama; do
    file="${RESULTS_DIR}/${label}_c${TOP_CONC}.json"
    if [ -f "$file" ] && grep -q "ttft_p50_ms" "$file" 2>/dev/null; then
        t=$(grep -o '"ttft_p50_ms": *[0-9]*' "$file" | head -1 | sed 's/.*: *//')
        TTFT_MAP[$label]="$t"
        [ "${t:-0}" -gt "$MAX_TTFT" ] && MAX_TTFT=$t
    fi
done
for label in fox llama ollama; do
    if [ -n "${TTFT_MAP[$label]:-}" ]; then
        t="${TTFT_MAP[$label]}"
        if [ "$MAX_TTFT" -gt 0 ]; then
            bar_len=$(( t * BAR_WIDTH / MAX_TTFT ))
        else
            bar_len=0
        fi
        bar=$(printf '%0.s█' $(seq 1 $bar_len 2>/dev/null) 2>/dev/null || true)
        printf "  %-10s %s %sms\n" "$label" "$bar" "$t"
    else
        printf "  %-10s SKIPPED\n" "$label"
    fi
done

echo ""
echo "Results saved to ${RESULTS_DIR}/"
