#!/usr/bin/env bash
# bench_suite.sh — run fox with different configurations and benchmark each.
# Usage: bash bench_suite.sh
# Expects to run inside the CUDA devcontainer with a model at /workspace/models/

set -euo pipefail

MODEL_PATH="/workspace/models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
MODEL_NAME="Meta-Llama-3.1-8B-Instruct-Q8_0"
MODEL_DIR="/workspace/models"
FOX="/workspace/target/release/fox"
BENCH="/workspace/target/release/fox-bench"
PORT=8080
URL="http://localhost:${PORT}"
RESULTS_DIR="/workspace/bench_results"
PROMPT="Write a short paragraph about the Rust programming language and its advantages for systems programming."

mkdir -p "$RESULTS_DIR"

wait_for_server() {
    local max_wait=120
    local waited=0
    while ! curl -sf "${URL}/health" >/dev/null 2>&1; do
        sleep 2
        waited=$((waited + 2))
        if [ $waited -ge $max_wait ]; then
            echo "ERROR: server did not start within ${max_wait}s"
            return 1
        fi
    done
    echo "Server ready after ~${waited}s"
}

stop_server() {
    if [ -n "${FOX_PID:-}" ]; then
        kill "$FOX_PID" 2>/dev/null || true
        wait "$FOX_PID" 2>/dev/null || true
        unset FOX_PID
        sleep 2
    fi
}

run_bench() {
    local label="$1"
    local concurrency="$2"
    local requests="$3"
    local max_tokens="$4"

    echo ""
    echo "=== Benchmark: ${label} (concurrency=${concurrency}, requests=${requests}) ==="
    "$BENCH" \
        --url "$URL" \
        --model "$MODEL_NAME" \
        --concurrency "$concurrency" \
        --requests "$requests" \
        --max-tokens "$max_tokens" \
        --prompt "$PROMPT" \
        --output json \
        > "${RESULTS_DIR}/${label}.json" 2>&1

    # Also print text summary
    "$BENCH" \
        --url "$URL" \
        --model "$MODEL_NAME" \
        --concurrency "$concurrency" \
        --requests "$requests" \
        --max-tokens "$max_tokens" \
        --prompt "$PROMPT" \
        --output text \
        2>&1 | tee "${RESULTS_DIR}/${label}.txt"
}

trap stop_server EXIT

# ============================================================
# Config 1: BASELINE — defaults (batch=1, f16 KV, auto context)
# ============================================================
echo ""
echo "######################################################"
echo "# CONFIG 1: BASELINE (defaults)"
echo "######################################################"

"$FOX" serve \
    --model-path "$MODEL_PATH" \
    --models-dir "$MODEL_DIR" \
    --max-batch-size 1 \
    --flash-attn true \
    --port "$PORT" \
    > "${RESULTS_DIR}/server_baseline.log" 2>&1 &
FOX_PID=$!
wait_for_server

run_bench "baseline_c1" 1 20 256
run_bench "baseline_c4" 4 40 256

stop_server

# ============================================================
# Config 2: KV QUANT — q8_0/q4_0, batch=4
# ============================================================
echo ""
echo "######################################################"
echo "# CONFIG 2: KV QUANT (q8_0/q4_0, batch=4)"
echo "######################################################"

"$FOX" serve \
    --model-path "$MODEL_PATH" \
    --models-dir "$MODEL_DIR" \
    --max-batch-size 4 \
    --type-k q8_0 \
    --type-v q4_0 \
    --flash-attn true \
    --port "$PORT" \
    > "${RESULTS_DIR}/server_kvquant.log" 2>&1 &
FOX_PID=$!
wait_for_server

run_bench "kvquant_c1" 1 20 256
run_bench "kvquant_c4" 4 40 256

stop_server

# ============================================================
# Config 3: OPTIMIZED — q8_0/q4_0, batch=8, ctx=4096
# ============================================================
echo ""
echo "######################################################"
echo "# CONFIG 3: OPTIMIZED (q8_0/q4_0, batch=8, ctx=4096)"
echo "######################################################"

"$FOX" serve \
    --model-path "$MODEL_PATH" \
    --models-dir "$MODEL_DIR" \
    --max-batch-size 8 \
    --type-k q8_0 \
    --type-v q4_0 \
    --max-context-len 4096 \
    --flash-attn true \
    --port "$PORT" \
    > "${RESULTS_DIR}/server_optimized.log" 2>&1 &
FOX_PID=$!
wait_for_server

run_bench "optimized_c1" 1 20 256
run_bench "optimized_c4" 4 40 256
run_bench "optimized_c8" 8 80 256

stop_server

echo ""
echo "######################################################"
echo "# ALL BENCHMARKS COMPLETE"
echo "######################################################"
echo ""
echo "Results in ${RESULTS_DIR}/"
ls -la "${RESULTS_DIR}/"
