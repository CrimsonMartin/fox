#!/bin/bash
set -euo pipefail

MODEL_PATH="/workspace/models/google_gemma-4-26B-A4B-it-Q4_K_M.gguf"
MODEL_NAME="google_gemma-4-26B-A4B-it-Q4_K_M"
MODEL_DIR="$(dirname "$MODEL_PATH")"
FOX="/workspace/target/release/fox"
BENCH="/workspace/target/release/fox-bench"
RESULTS="/workspace/bench_results/compare_v3"
MAX_CTX=4096
MAX_BATCH=4

mkdir -p "$RESULTS"

PROMPT='You are a senior software architect reviewing a complex distributed system. Below is a detailed technical specification for a new microservice. Please analyze it thoroughly, identify potential issues, and provide recommendations. SYSTEM SPECIFICATION: Real-Time Event Processing Pipeline. 1. ARCHITECTURE OVERVIEW The system processes financial transaction events from 47 regional data centers, aggregating them into a unified stream for fraud detection, compliance reporting, and real-time analytics. The expected throughput is 2.3 million events per second at peak, with a 99th percentile latency requirement of 50 milliseconds end-to-end. 2. DATA FLOW Events arrive via Apache Kafka topics partitioned by region. Each event contains: transaction ID UUID v7, timestamp nanosecond precision, source account, destination account, amount, currency code ISO 4217, merchant category code MCC, device fingerprint hash, and geolocation coordinates. The average event size is 847 bytes after Protocol Buffers serialization. The ingestion layer consists of 12 consumer groups, each with 8 consumers. Events are deserialized, validated against a JSON Schema, enriched with account metadata from a Redis Cluster 6 primary nodes 18 replicas, and then routed to one of three processing pipelines based on risk score. Given this specification, provide a concise architectural review focusing on the most critical risks and the single most impactful improvement you would recommend.'

wait_for_server() {
    local url="$1" max_wait="${2:-120}" waited=0
    while ! curl -sf "${url}/health" >/dev/null 2>&1 \
       && ! curl -sf "${url}/v1/models" >/dev/null 2>&1; do
        sleep 2; waited=$((waited + 2))
        if [ $waited -ge $max_wait ]; then
            echo "  ERROR: timeout" >&2; return 1
        fi
    done
}

wait_for_model() {
    local url="$1" model="$2" max_wait="${3:-120}" waited=0
    while true; do
        resp=$(curl -s --max-time 10 "${url}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"${model}\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":1,\"stream\":false}" 2>&1) || true
        if echo "$resp" | grep -q '"completion_tokens"'; then return 0; fi
        sleep 3; waited=$((waited + 3))
        if [ $waited -ge $max_wait ]; then return 1; fi
    done
}

echo "=== Round 3: prefix cache fix ==="
pkill -f "fox serve" 2>/dev/null || true; sleep 1

"$FOX" serve \
    --model-path "$MODEL_PATH" \
    --models-dir "$MODEL_DIR" \
    --max-batch-size $MAX_BATCH \
    --max-context-len $MAX_CTX \
    --flash-attn true \
    --chunked-prefill-tokens 512 \
    --port 8080 \
    > "$RESULTS/server_fox.log" 2>&1 &
FOX_PID=$!

wait_for_server "http://localhost:8080"
echo "  warming up..."
wait_for_model "http://localhost:8080" "$MODEL_NAME"
echo "  model ready"

# Only test c=4 — that's where the fix matters
for conc in 1 4; do
    reqs=$((60 / 4 * conc))
    [ "$reqs" -lt "$conc" ] && reqs=$conc
    echo "  bench fox c=$conc n=$reqs"
    "$BENCH" \
        --url http://localhost:8080 \
        --model "$MODEL_NAME" \
        --concurrency $conc \
        --requests $reqs \
        --max-tokens 128 \
        --prompt "$PROMPT" \
        --output json \
        > "$RESULTS/fox_c${conc}.json" 2>&1 || echo "  WARN: bench c=$conc failed"
done

kill $FOX_PID 2>/dev/null; wait $FOX_PID 2>/dev/null || true

echo ""
echo "=== RESULTS ==="
for conc in 1 4; do
    file="$RESULTS/fox_c${conc}.json"
    if [ -f "$file" ] && grep -q "throughput_tokens_per_sec" "$file" 2>/dev/null; then
        tput=$(grep -o '"throughput_tokens_per_sec": *[0-9.]*' "$file" | head -1 | sed 's/.*: *//')
        ttft=$(grep -o '"ttft_p50_ms": *[0-9]*' "$file" | head -1 | sed 's/.*: *//')
        printf "  fox c=%s: %.1f tok/s  TTFT %sms\n" "$conc" "$tput" "$ttft"
    fi
done

# Check batch sizes
echo ""
echo "Decode batch sizes:"
sed 's/\x1b\[[0-9;]*m//g' "$RESULTS/server_fox.log" | grep "decode step" | sed 's/.*n: \([0-9]*\),.*/\1/' | sort | uniq -c | sort -rn

echo ""
echo "Seq IDs used:"
sed 's/\x1b\[[0-9;]*m//g' "$RESULTS/server_fox.log" | grep "admitted to batch" | grep -oP 'seq_id: \d+' | sort | uniq -c | sort -rn

echo "DONE"
