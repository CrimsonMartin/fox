#!/usr/bin/env bash
# bench_chunked_prefill.sh — sweep chunked prefill token sizes to find the optimal value.
#
# Starts fox serve for each chunk size, runs fox-bench with a long prompt at
# multiple concurrency levels, and prints a summary table.
#
# Usage:
#   bash bench_chunked_prefill.sh                    # defaults
#   MODEL_PATH=/path/to/model.gguf bash bench_chunked_prefill.sh
#
# Expects: CUDA devcontainer with fox + fox-bench built at /workspace/target/release/

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/workspace/models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf}"
MODEL_NAME="$(basename "$MODEL_PATH" .gguf)"
MODEL_DIR="${MODEL_DIR:-/workspace/models}"
FOX="${FOX:-/workspace/target/release/fox}"
BENCH="${BENCH:-/workspace/target/release/fox-bench}"
PORT="${PORT:-8080}"
URL="http://localhost:${PORT}"
RESULTS_DIR="${RESULTS_DIR:-/workspace/bench_results/chunked_prefill}"
MAX_BATCH="${MAX_BATCH:-8}"
MAX_CTX="${MAX_CTX:-4096}"
MAX_TOKENS="${MAX_TOKENS:-128}"
REQUESTS_PER_LEVEL="${REQUESTS_PER_LEVEL:-20}"

# Chunk sizes to sweep (0 = disabled = full prompt in one pass)
CHUNK_SIZES="${CHUNK_SIZES:-0 128 256 512 1024 2048}"
# Concurrency levels
CONCURRENCIES="${CONCURRENCIES:-1 4 8}"

mkdir -p "$RESULTS_DIR"

# ── Long prompt (~1200 tokens) ──────────────────────────────────────────────
# This exercises chunked prefill meaningfully — short prompts fit in a single
# chunk regardless of the limit.
read -r -d '' LONG_PROMPT << 'PROMPT_EOF' || true
You are a senior software architect reviewing a complex distributed system. Below is a detailed technical specification for a new microservice. Please analyze it thoroughly, identify potential issues, and provide recommendations.

SYSTEM SPECIFICATION: Real-Time Event Processing Pipeline

1. ARCHITECTURE OVERVIEW
The system processes financial transaction events from 47 regional data centers, aggregating them into a unified stream for fraud detection, compliance reporting, and real-time analytics. The expected throughput is 2.3 million events per second at peak, with a 99th percentile latency requirement of 50 milliseconds end-to-end.

2. DATA FLOW
Events arrive via Apache Kafka topics partitioned by region. Each event contains: transaction ID (UUID v7), timestamp (nanosecond precision), source account, destination account, amount, currency code (ISO 4217), merchant category code (MCC), device fingerprint hash, and geolocation coordinates. The average event size is 847 bytes after Protocol Buffers serialization.

The ingestion layer consists of 12 consumer groups, each with 8 consumers. Events are deserialized, validated against a JSON Schema, enriched with account metadata from a Redis Cluster (6 primary nodes, 18 replicas), and then routed to one of three processing pipelines based on risk score.

3. PROCESSING PIPELINES
Pipeline A (Low Risk, 78% of traffic): Events are batched in 100ms windows and written directly to Apache Parquet files on S3-compatible object storage. A separate Apache Spark job runs every 15 minutes to compute aggregate statistics.

Pipeline B (Medium Risk, 19% of traffic): Events pass through a rules engine implemented as a directed acyclic graph of 342 evaluation nodes. Each node applies a specific fraud detection heuristic. The rules engine maintains a sliding window of the last 1000 transactions per account in a custom B-tree backed by memory-mapped files. If any rule triggers, the event is escalated to Pipeline C.

Pipeline C (High Risk, 3% of traffic): Events undergo real-time ML inference using an ensemble of three models: a gradient boosted decision tree (XGBoost), a neural network (ONNX Runtime), and a graph neural network that analyzes transaction networks. The ensemble requires the last 90 days of transaction history for the involved accounts, fetched from a distributed columnar store (Apache Druid). Model inference must complete within 15ms.

4. STATE MANAGEMENT
The system maintains several categories of distributed state: per-account transaction history (hot: last 24 hours in Redis, warm: last 90 days in Druid, cold: full history in Parquet on S3), per-merchant fraud scores updated every 30 seconds, global feature vectors for the GNN model refreshed every 5 minutes, and circuit breaker states for downstream dependencies.

Consistency is maintained using a hybrid approach: strong consistency for account-level mutations via Redis distributed locks with fencing tokens, eventual consistency for aggregate statistics with a maximum staleness of 30 seconds, and causal consistency for the ML feature store using vector clocks.

5. FAILURE MODES AND RECOVERY
The system must handle: Kafka broker failures (up to 2 simultaneous), Redis node failures (automatic failover within 3 seconds), network partitions between regions (continue processing with degraded accuracy), ML model serving failures (fallback to rules-only evaluation), and S3 write failures (local buffering with replay).

Dead letter queues capture events that fail processing after 3 retries. A separate reconciliation service runs hourly to reprocess DLQ events and identify systematic failures. All state changes are journaled to a write-ahead log for point-in-time recovery.

6. OBSERVABILITY
Metrics are emitted via OpenTelemetry to a Prometheus-compatible TSDB. Key SLIs include: event processing latency (histogram, 10ms buckets), pipeline throughput (counter per pipeline), error rate by category, ML model inference time, and cache hit ratios. Distributed tracing uses W3C Trace Context propagation across all service boundaries.

Given this specification, provide a concise architectural review focusing on the most critical risks and the single most impactful improvement you would recommend. Be specific and reference the section numbers.
PROMPT_EOF

# ── Helpers ─────────────────────────────────────────────────────────────────

wait_for_server() {
    local max_wait=120 waited=0
    while ! curl -sf "${URL}/health" >/dev/null 2>&1; do
        sleep 2; waited=$((waited + 2))
        if [ $waited -ge $max_wait ]; then
            echo "ERROR: server did not start within ${max_wait}s" >&2
            return 1
        fi
    done
}

stop_server() {
    if [ -n "${FOX_PID:-}" ]; then
        kill "$FOX_PID" 2>/dev/null || true
        wait "$FOX_PID" 2>/dev/null || true
        unset FOX_PID
        sleep 1
    fi
}

trap stop_server EXIT

# ── Collect results ─────────────────────────────────────────────────────────

echo "========================================"
echo " Chunked Prefill Tuning"
echo " model:  $MODEL_NAME"
echo " batch:  $MAX_BATCH  ctx: $MAX_CTX"
echo " chunks: $CHUNK_SIZES"
echo " conc:   $CONCURRENCIES"
echo " reqs:   $REQUESTS_PER_LEVEL per level"
echo "========================================"
echo ""

for chunk in $CHUNK_SIZES; do
    label="chunk${chunk}"
    echo "── chunk_size=$chunk ──"

    FOX_CHUNKED_PREFILL_TOKENS=$chunk "$FOX" serve \
        --model-path "$MODEL_PATH" \
        --models-dir "$MODEL_DIR" \
        --max-batch-size "$MAX_BATCH" \
        --max-context-len "$MAX_CTX" \
        --flash-attn true \
        --chunked-prefill-tokens "$chunk" \
        --port "$PORT" \
        > "${RESULTS_DIR}/server_${label}.log" 2>&1 &
    FOX_PID=$!
    wait_for_server
    echo "  server ready"

    for conc in $CONCURRENCIES; do
        reqs=$((REQUESTS_PER_LEVEL * conc))
        tag="${label}_c${conc}"
        echo "  bench $tag (c=$conc n=$reqs)"
        "$BENCH" \
            --url "$URL" \
            --model "$MODEL_NAME" \
            --concurrency "$conc" \
            --requests "$reqs" \
            --max-tokens "$MAX_TOKENS" \
            --prompt "$LONG_PROMPT" \
            --output json \
            > "${RESULTS_DIR}/${tag}.json" 2>&1
    done

    failures=$(grep -c "prefill failed" "${RESULTS_DIR}/server_${label}.log" 2>/dev/null || echo 0)
    echo "  prefill failures: $failures"
    stop_server
    echo ""
done

# ── Summary table ───────────────────────────────────────────────────────────

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo " RESULTS SUMMARY"
echo "═══════════════════════════════════════════════════════════════════════════════"
printf "%-12s  %4s  %9s  %8s  %8s  %8s  %8s  %6s\n" \
    "chunk" "conc" "tput(t/s)" "TTFT p50" "TTFT p95" "lat p50" "lat p95" "errors"
echo "───────────────────────────────────────────────────────────────────────────────"

for chunk in $CHUNK_SIZES; do
    label="chunk${chunk}"
    for conc in $CONCURRENCIES; do
        tag="${label}_c${conc}"
        file="${RESULTS_DIR}/${tag}.json"
        if [ ! -f "$file" ]; then
            printf "%-12s  %4s  %9s\n" "$chunk" "$conc" "MISSING"
            continue
        fi
        # Parse with grep + sed (no python dependency in container)
        tput=$(grep -o '"throughput_tokens_per_sec": *[0-9.]*' "$file" | head -1 | sed 's/.*: *//')
        ttft50=$(grep -o '"ttft_p50_ms": *[0-9]*' "$file" | head -1 | sed 's/.*: *//')
        ttft95=$(grep -o '"ttft_p95_ms": *[0-9]*' "$file" | head -1 | sed 's/.*: *//')
        lat50=$(grep -o '"latency_p50_ms": *[0-9]*' "$file" | head -1 | sed 's/.*: *//')
        lat95=$(grep -o '"latency_p95_ms": *[0-9]*' "$file" | head -1 | sed 's/.*: *//')
        errs=$(grep -o '"requests_err": *[0-9]*' "$file" | head -1 | sed 's/.*: *//')
        # Truncate throughput to 1 decimal
        tput_fmt=$(printf "%.1f" "$tput" 2>/dev/null || echo "$tput")
        printf "%-12s  %4s  %9s  %6sms  %6sms  %6sms  %6sms  %6s\n" \
            "$chunk" "$conc" "$tput_fmt" "$ttft50" "$ttft95" "$lat50" "$lat95" "$errs"
    done
done

echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Results saved to ${RESULTS_DIR}/"
