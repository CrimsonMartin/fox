#!/usr/bin/env bash
set -euo pipefail

MODEL="/workspace/models/google_gemma-4-26B-A4B-it-Q4_K_M.gguf"
MODEL_NAME="$(basename "$MODEL" .gguf)"
FOX=/workspace/target/release/fox
BENCH=/workspace/target/release/fox-bench
LLAMA_SERVER=/workspace/vendor/llama.cpp/build/bin/llama-server
RESULTS=/workspace/bench_results/compare_v4
mkdir -p "$RESULTS"

MAX_CTX=4096
MAX_BATCH=4
MAX_TOKENS=128
REQUESTS=60

PROMPT="You are a senior software architect reviewing a complex distributed system. Below is a detailed technical specification for a new microservice. Please analyze it thoroughly, identify potential issues, and provide recommendations. SYSTEM SPECIFICATION: Real-Time Event Processing Pipeline. 1. ARCHITECTURE OVERVIEW: The system processes financial transaction events from 47 regional data centers, aggregating them into a unified stream for fraud detection, compliance reporting, and real-time analytics. The expected throughput is 2.3 million events per second at peak, with a 99th percentile latency requirement of 50 milliseconds end-to-end. 2. DATA FLOW: Events arrive via Apache Kafka topics partitioned by region. Each event contains: transaction ID (UUID v7), timestamp (nanosecond precision), source account, destination account, amount, currency code (ISO 4217), merchant category code (MCC), device fingerprint hash, and geolocation coordinates. Given this specification, provide a concise architectural review focusing on the most critical risks and the single most impactful improvement you would recommend."

pkill -f "fox serve" 2>/dev/null || true
pkill -f llama-server 2>/dev/null || true
sleep 2

echo "======================================="
echo " Round 4: fox (min_p=0.05, no logits copy, block_in_place)"
echo "======================================="

"$FOX" serve \
    --model-path "$MODEL" \
    --models-dir "$(dirname "$MODEL")" \
    --max-batch-size "$MAX_BATCH" \
    --max-context-len "$MAX_CTX" \
    --flash-attn true \
    --chunked-prefill-tokens 512 \
    --port 8080 \
    > "$RESULTS/server_fox.log" 2>&1 &
FOX_PID=$!

echo "Waiting for fox to start..."
for i in $(seq 1 60); do
    if curl -sf http://localhost:8080/health >/dev/null 2>&1 || \
       curl -sf http://localhost:8080/v1/models >/dev/null 2>&1; then
        echo "  fox ready"
        break
    fi
    sleep 2
done

echo "Warming up fox..."
curl -s --max-time 60 http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":4,\"stream\":false}" \
    > /dev/null 2>&1 || true
sleep 2

echo "Benchmarking fox c=1..."
"$BENCH" --url http://localhost:8080 --model "$MODEL_NAME" \
    --concurrency 1 --requests 15 --max-tokens "$MAX_TOKENS" \
    --prompt "$PROMPT" --output json \
    > "$RESULTS/fox_c1.json" 2>&1 || true

echo "Benchmarking fox c=4..."
"$BENCH" --url http://localhost:8080 --model "$MODEL_NAME" \
    --concurrency 4 --requests "$REQUESTS" --max-tokens "$MAX_TOKENS" \
    --prompt "$PROMPT" --output json \
    > "$RESULTS/fox_c4.json" 2>&1 || true

kill $FOX_PID 2>/dev/null || true
wait $FOX_PID 2>/dev/null || true
sleep 2

echo ""
echo "======================================="
echo " Round 4: llama-server"
echo "======================================="

LLAMA_TOTAL_CTX=$((MAX_CTX * MAX_BATCH))
"$LLAMA_SERVER" \
    --model "$MODEL" \
    --host 0.0.0.0 --port 8082 \
    --ctx-size "$LLAMA_TOTAL_CTX" \
    --flash-attn on \
    -ngl 99 \
    --parallel "$MAX_BATCH" \
    > "$RESULTS/server_llama.log" 2>&1 &
LLAMA_PID=$!

echo "Waiting for llama-server to start..."
for i in $(seq 1 90); do
    if curl -sf http://localhost:8082/health >/dev/null 2>&1 || \
       curl -sf http://localhost:8082/v1/models >/dev/null 2>&1; then
        echo "  llama-server ready"
        break
    fi
    sleep 2
done

echo "Warming up llama-server..."
curl -s --max-time 120 http://localhost:8082/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":4,\"stream\":false}" \
    > /dev/null 2>&1 || true
sleep 2

echo "Benchmarking llama-server c=1..."
"$BENCH" --url http://localhost:8082 --model "$MODEL_NAME" \
    --concurrency 1 --requests 15 --max-tokens "$MAX_TOKENS" \
    --prompt "$PROMPT" --output json \
    > "$RESULTS/llama_c1.json" 2>&1 || true

echo "Benchmarking llama-server c=4..."
"$BENCH" --url http://localhost:8082 --model "$MODEL_NAME" \
    --concurrency 4 --requests "$REQUESTS" --max-tokens "$MAX_TOKENS" \
    --prompt "$PROMPT" --output json \
    > "$RESULTS/llama_c4.json" 2>&1 || true

kill $LLAMA_PID 2>/dev/null || true
wait $LLAMA_PID 2>/dev/null || true
sleep 2

echo ""
echo "======================================="
echo " Round 4: RESULTS"
echo "======================================="
for file in fox_c1 fox_c4 llama_c1 llama_c4; do
    f="$RESULTS/${file}.json"
    if [ -f "$f" ] && grep -q "throughput_tokens_per_sec" "$f" 2>/dev/null; then
        tput=$(grep -o '"throughput_tokens_per_sec": *[0-9.]*' "$f" | head -1 | sed 's/.*: *//')
        ttft=$(grep -o '"ttft_p50_ms": *[0-9]*' "$f" | head -1 | sed 's/.*: *//')
        lat=$(grep -o '"latency_p50_ms": *[0-9]*' "$f" | head -1 | sed 's/.*: *//')
        errs=$(grep -o '"requests_err": *[0-9]*' "$f" | head -1 | sed 's/.*: *//')
        printf "%-12s  tput=%7.1f tok/s  TTFT_p50=%5sms  lat_p50=%5sms  errs=%s\n" \
            "$file" "$tput" "$ttft" "$lat" "$errs"
    else
        printf "%-12s  SKIP\n" "$file"
    fi
done
echo "Done. Logs in $RESULTS/"
