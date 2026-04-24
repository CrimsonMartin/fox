#!/usr/bin/env bash
# Sweep --max-batch-size and --vision-contexts to find diminishing returns.
# Runs inside the container, outputs a TSV table.
set -euo pipefail

MODEL="/root/.cache/ferrumox/models/nanollava-text-model-f16.gguf"
PORT=8080
CTX=2048
CONCURRENCY=4
REQUESTS=20
# Small base64 PNG (1x1 red pixel)
IMG="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
PROMPT="Describe this image in one sentence."

BATCH_SIZES=(1 2 4 8)
VISION_CONTEXTS=(1 2 4)

echo -e "batch\tvision\tvram_mib\treqs\telapsed_s\treq_per_s\tavg_latency_s"

for vc in "${VISION_CONTEXTS[@]}"; do
  for bs in "${BATCH_SIZES[@]}"; do
    # Kill any running fox
    pkill -f "fox serve" 2>/dev/null || true
    sleep 2

    # Record baseline VRAM
    vram_before=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 0)

    # Start fox
    ./target/release/fox serve \
      --model-path "$MODEL" \
      --vision-contexts "$vc" \
      --max-context-len "$CTX" \
      --max-batch-size "$bs" \
      --port "$PORT" 2>/dev/null &
    FOX_PID=$!

    # Wait for fox to be ready
    for i in $(seq 1 60); do
      if curl -sf http://localhost:$PORT/api/tags >/dev/null 2>&1; then
        break
      fi
      sleep 1
    done

    # Record loaded VRAM
    vram_after=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 0)
    vram_net=$((vram_after - vram_before + vram_before))  # total used

    # Build request body
    BODY=$(cat <<ENDJSON
{
  "model": "nanollava-text-model-f16",
  "messages": [{"role":"user","content":[
    {"type":"image_url","image_url":{"url":"data:image/png;base64,$IMG"}},
    {"type":"text","text":"$PROMPT"}
  ]}],
  "max_tokens": 50
}
ENDJSON
)

    # Run concurrent requests and measure
    start_time=$(date +%s.%N)
    completed=0
    pids=()

    for r in $(seq 1 $REQUESTS); do
      (
        curl -sf -X POST "http://localhost:$PORT/v1/chat/completions" \
          -H "Content-Type: application/json" \
          -d "$BODY" -o /dev/null -w "%{time_total}\n" 2>/dev/null
      ) &
      pids+=($!)

      # Limit in-flight to CONCURRENCY
      if (( ${#pids[@]} >= CONCURRENCY )); then
        wait "${pids[0]}" 2>/dev/null && completed=$((completed+1)) || true
        pids=("${pids[@]:1}")
      fi
    done
    # Wait for remaining
    for p in "${pids[@]}"; do
      wait "$p" 2>/dev/null && completed=$((completed+1)) || true
    done

    end_time=$(date +%s.%N)
    elapsed=$(echo "$end_time - $start_time" | bc)
    req_per_s=$(echo "scale=2; $REQUESTS / $elapsed" | bc)
    avg_latency=$(echo "scale=3; $elapsed / $REQUESTS * $CONCURRENCY" | bc)

    echo -e "${bs}\t${vc}\t${vram_after}\t${REQUESTS}\t${elapsed}\t${req_per_s}\t${avg_latency}"

    # Cleanup
    kill $FOX_PID 2>/dev/null || true
    wait $FOX_PID 2>/dev/null || true
    sleep 2
  done
done
