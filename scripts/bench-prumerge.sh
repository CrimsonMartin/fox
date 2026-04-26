#!/usr/bin/env bash
set -euo pipefail

# PruMerge benchmark: compare vision throughput at different keep ratios.

FOX="${FOX:-/workspace/target/release/fox}"
FOX_PORT="${FOX_PORT:-8080}"
REQUESTS="${REQUESTS:-10}"
MODEL_DIR="${MODEL_DIR:-/workspace/models}"
TEXT_MODEL="${TEXT_MODEL:-nanollava-text-model-f16.gguf}"

BOLD='\033[1m'
GREEN='\033[0;32m'
NC='\033[0m'

TMPDIR=$(mktemp -d)
trap 'pkill -f "fox serve" 2>/dev/null||true; rm -rf "$TMPDIR"' EXIT

# Create test image request (2x2 BMP — CLIP will process as 729 tokens)
printf "\x42\x4d\x46\x00\x00\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00\x02\x00\x00\x00\x02\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\x00\x00\x00\xff\x00\x00\x00\x00\x00\xff\x00\x00\xff\x00\x00" > "$TMPDIR/test.bmp"
IMG_B64=$(base64 -w0 "$TMPDIR/test.bmp")

jq -n --arg img "$IMG_B64" \
  '{model: "'"$TEXT_MODEL"'", messages: [{role: "user", content: [{type: "text", text: "Describe what you see in one sentence."}, {type: "image_url", image_url: {url: ("data:image/bmp;base64," + $img)}}]}], max_tokens: 64, temperature: 0.1}' \
  > "$TMPDIR/req.json"

run_config() {
    local label="$1" keep="$2"
    echo -e "\n${BOLD}--- $label (keep=$keep) ---${NC}"

    pkill -f "fox serve" 2>/dev/null || true
    sleep 1

    FOX_PRUMERGE_KEEP="$keep" "$FOX" serve \
        --model-path "$MODEL_DIR/$TEXT_MODEL" \
        --port "$FOX_PORT" --max-context-len 2048 > "$TMPDIR/${label}.log" 2>&1 &

    for i in $(seq 1 60); do
        curl -s "http://localhost:${FOX_PORT}/health" >/dev/null 2>&1 && break
        sleep 1
    done

    # Warmup
    for i in $(seq 1 3); do
        curl -s --max-time 30 "http://localhost:${FOX_PORT}/v1/chat/completions" \
            -H "Content-Type: application/json" -d @"$TMPDIR/req.json" > /dev/null
    done

    # Measure
    local times=()
    local wall_start wall_end
    wall_start=$(date +%s%N)
    for i in $(seq 1 "$REQUESTS"); do
        local t
        t=$(curl -s -o "$TMPDIR/resp_${label}_$i.json" -w '%{time_total}' \
            --max-time 30 "http://localhost:${FOX_PORT}/v1/chat/completions" \
            -H "Content-Type: application/json" -d @"$TMPDIR/req.json")
        times+=("$t")
    done
    wall_end=$(date +%s%N)
    local wall_ms=$(( (wall_end - wall_start) / 1000000 ))

    # Stats
    IFS=$'\n' sorted=($(sort -g <<<"${times[*]}")); unset IFS
    local sum=0
    for t in "${times[@]}"; do sum=$(echo "$sum + $t" | bc); done
    local avg=$(echo "scale=3; $sum / $REQUESTS" | bc)
    local p50=${sorted[$(( REQUESTS / 2 ))]}
    local rps=$(echo "scale=2; $REQUESTS / ($wall_ms / 1000)" | bc)

    local output
    output=$(jq -r '.choices[0].message.content' "$TMPDIR/resp_${label}_1.json" 2>/dev/null)

    # Extract embd_decode from server logs (skip first warmup entry)
    local embd_avg
    embd_avg=$(grep "vision_decode_prefill_batch" "$TMPDIR/${label}.log" | \
        sed 's/\x1b\[[0-9;]*m//g' | \
        tail -"$REQUESTS" | \
        grep -oP 'embd_decode_us: \K[0-9]+' | \
        awk '{s+=$1; n++} END {if(n>0) printf "%.0f", s/n; else print "?"}')

    local kept
    kept=$(grep "prumerge" "$TMPDIR/${label}.log" | head -1 | \
        sed 's/\x1b\[[0-9;]*m//g' | grep -oP 'kept: \K[0-9]+' || echo "729")

    echo -e "  ${GREEN}Tokens:${NC}  729 → $kept"
    printf "  avg: %.3fs   p50: %.3fs   rps: %s\n" "$avg" "$p50" "$rps"
    printf "  embd_decode_avg: %s us\n" "$embd_avg"
    echo "  output: ${output:0:80}"

    pkill -f "fox serve" 2>/dev/null || true
    sleep 1
}

echo -e "${BOLD}=== PruMerge Benchmark ===${NC}"
echo "Requests per config: $REQUESTS"
echo "Model: $TEXT_MODEL (729 vision tokens per image)"

run_config "baseline" "1.0"
run_config "prumerge50" "0.5"
run_config "prumerge25" "0.25"

echo -e "\n${BOLD}=== Done ===${NC}"
