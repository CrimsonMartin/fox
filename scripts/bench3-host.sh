#!/usr/bin/env bash
# 3-way benchmark (text + vision) run from the HOST.
# Fox runs inside an existing devcontainer, llama.cpp and ollama in fresh docker containers.
set -euo pipefail

REQUESTS="${1:-50}"
CONCURRENCY="${2:-4}"
MAX_TOKENS="${3:-50}"
MODEL_DIR="${HOME}/.cache/ferrumox/models"
TEXT_MODEL="nanollava-text-model-f16.gguf"
MMPROJ="nanollava-mmproj-f16.gguf"
FOX_CONTAINER="crazy_chandrasekhar"
FOX_URL="http://172.17.0.2:8080"
LLAMACPP_PORT=8082
OLLAMA_PORT=11434

BOLD='\033[1m'
NC='\033[0m'

TMPDIR=$(mktemp -d)
trap 'docker rm -f llamacpp-bench ollama-bench 2>/dev/null; rm -rf "$TMPDIR"' EXIT

# Generate text-only + vision request payloads
echo -n "Generating $REQUESTS request payloads (max_tokens=$MAX_TOKENS)..."
python3 - "$REQUESTS" "$TMPDIR" "$MAX_TOKENS" << 'PYEOF'
import base64, hashlib, json, os, sys, io
try:
    from PIL import Image
    import numpy as np
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    import struct, zlib

PROMPTS = [
    "Write a short poem about the ocean.",
    "Explain how a CPU cache works in two sentences.",
    "List three fun facts about honey bees.",
    "What is the difference between TCP and UDP?",
    "Describe the taste of coffee to someone who has never tried it.",
    "Write a haiku about mountains.",
    "Explain recursion to a five-year-old.",
    "What causes thunder?",
    "Name three benefits of open source software.",
    "Summarize the plot of Romeo and Juliet in one sentence.",
]

def make_image_pil(seed):
    np.random.seed(seed)
    img = Image.fromarray(np.random.randint(0, 256, (768, 1024, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=95)
    return buf.getvalue()

def make_image_fallback(seed):
    width, height = 512, 512
    row_template = bytearray(width * 3)
    for x in range(width):
        row_template[x*3] = (seed * 7 + x * 3) % 256
        row_template[x*3+1] = (seed * 13 + x * 5) % 256
        row_template[x*3+2] = (seed * 19 + x * 7) % 256
    raw = bytearray()
    for y in range(height):
        raw += b'\x00'
        offset = (y * 3) % 256
        row = bytearray((b + offset) % 256 for b in row_template)
        raw += row
    def chunk(ctype, data):
        c = ctype + data
        return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)
    ihdr = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', ihdr) + chunk(b'IDAT', zlib.compress(bytes(raw), 0)) + chunk(b'IEND', b'')

make_image = make_image_pil if HAS_PIL else make_image_fallback
n = int(sys.argv[1])
outdir = sys.argv[2]
max_tokens = int(sys.argv[3])
for i in range(n):
    prompt = PROMPTS[i % len(PROMPTS)]
    # Text-only requests
    for model, suffix in [("nanollava-text-model-f16", "default"), ("qnguyen3/nanollava", "ollama")]:
        with open(os.path.join(outdir, f'txtreq_{suffix}_{i+1}.json'), 'w') as f:
            json.dump({"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0.8, "top_p": 0.95, "top_k": 40}, f)
    # Vision requests
    img_bytes = make_image(i)
    b64 = base64.b64encode(img_bytes).decode()
    if i == 0:
        print(f" (images: {len(img_bytes)/1024:.0f} KB each)", end="", flush=True)
    for model, suffix in [("nanollava-text-model-f16", "default"), ("qnguyen3/nanollava", "ollama")]:
        with open(os.path.join(outdir, f'req_{suffix}_{i+1}.json'), 'w') as f:
            f.write('{"model":"' + model + '","messages":[{"role":"user","content":[{"type":"text","text":"Describe this image in detail."},{"type":"image_url","image_url":{"url":"data:image/png;base64,' + b64 + '"}}]}],"max_tokens":' + str(max_tokens) + ',"temperature":0.8,"top_p":0.95,"top_k":40}')
PYEOF
echo " done"

# ---------------------------------------------------------------------------
# run_bench <label> <url> <model_suffix> [file_prefix]
#   file_prefix: "txt" for text-only requests, "" for vision requests (default)
run_bench() {
    local label="$1" url="$2" req_prefix="$3" file_prefix="${4:-}"
    local outdir="$TMPDIR/$label"
    mkdir -p "$outdir"

    local req_file_prefix="req"
    [ -n "$file_prefix" ] && req_file_prefix="${file_prefix}req"

    # Warmup
    echo "  Warming up..."
    curl -sf --max-time 600 -X POST "$url/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d @"$TMPDIR/${req_file_prefix}_${req_prefix}_1.json" -o /dev/null || { echo "  FAIL: warmup failed"; return 1; }

    # VRAM before
    local VRAM_LOADED
    VRAM_LOADED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")

    # Start VRAM sampler
    (while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 >> "$outdir/vram_samples.txt"; sleep 0.5; done) &
    local VRAM_PID=$!

    echo "  Running $REQUESTS requests at concurrency $CONCURRENCY..."
    local START END ELAPSED_MS
    START=$(date +%s%N)

    local running=0
    local pids=()
    for i in $(seq 1 "$REQUESTS"); do
        curl -sf --max-time 600 -X POST "$url/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d @"$TMPDIR/${req_file_prefix}_${req_prefix}_$i.json" \
            -o "$outdir/resp_$i.json" \
            -w "%{time_total}\n" > "$outdir/time_$i.txt" 2>/dev/null &
        pids+=($!)
        running=$((running + 1))

        if [ "$running" -ge "$CONCURRENCY" ]; then
            wait "${pids[0]}" 2>/dev/null || true
            pids=("${pids[@]:1}")
            running=$((running - 1))
        fi
    done
    for p in "${pids[@]}"; do wait "$p" 2>/dev/null || true; done

    END=$(date +%s%N)
    ELAPSED_MS=$(( (END - START) / 1000000 ))

    # Stop VRAM sampler
    kill "$VRAM_PID" 2>/dev/null; wait "$VRAM_PID" 2>/dev/null || true
    local VRAM_PEAK="$VRAM_LOADED"
    [ -f "$outdir/vram_samples.txt" ] && VRAM_PEAK=$(sort -n "$outdir/vram_samples.txt" | tail -1)

    # Collect latencies and token counts
    local TIMES=() ERRORS=0 TOTAL_TOKENS=0
    for i in $(seq 1 "$REQUESTS"); do
        [ -f "$outdir/time_$i.txt" ] && TIMES+=("$(head -1 "$outdir/time_$i.txt")")
        if [ -f "$outdir/resp_$i.json" ]; then
            grep -q '"error"' "$outdir/resp_$i.json" 2>/dev/null && ERRORS=$((ERRORS + 1))
            local toks
            toks=$(python3 -c "import json; d=json.load(open('$outdir/resp_$i.json')); print(d.get('usage',{}).get('completion_tokens',0))" 2>/dev/null || echo 0)
            TOTAL_TOKENS=$((TOTAL_TOKENS + toks))
        fi
    done

    printf '%s\n' "${TIMES[@]}" | sort -n > "$outdir/sorted.txt"
    local COUNT=${#TIMES[@]}
    local p50_idx=$(( COUNT * 50 / 100 )); [ "$p50_idx" -ge "$COUNT" ] && p50_idx=$((COUNT - 1))
    local p95_idx=$(( COUNT * 95 / 100 )); [ "$p95_idx" -ge "$COUNT" ] && p95_idx=$((COUNT - 1))
    local P50 P95 RPS ELAPSED_S TPS
    P50=$(sed -n "$((p50_idx + 1))p" "$outdir/sorted.txt")
    P95=$(sed -n "$((p95_idx + 1))p" "$outdir/sorted.txt")
    ELAPSED_S=$(echo "scale=2; $ELAPSED_MS / 1000" | bc)
    RPS=$(echo "scale=2; $REQUESTS / ($ELAPSED_MS / 1000)" | bc)
    TPS=$(echo "scale=1; $TOTAL_TOKENS / ($ELAPSED_MS / 1000)" | bc 2>/dev/null || echo "0")

    echo "${ELAPSED_S}|${RPS}|${P50}|${P95}|${ERRORS}|${VRAM_LOADED}|${VRAM_PEAK}|${TOTAL_TOKENS}|${TPS}" > "$outdir/summary.txt"
    echo "  Done: ${ELAPSED_S}s, ${RPS} req/s, ${TPS} tok/s (${TOTAL_TOKENS} tokens), P50=${P50}s, VRAM=${VRAM_LOADED}/${VRAM_PEAK} MiB, ${ERRORS} errors"
}

# ---------------------------------------------------------------------------
# 1. Fox (inside existing container)
# ---------------------------------------------------------------------------
echo -e "\n${BOLD}[1/3] Fox${NC}"
docker exec -d "$FOX_CONTAINER" bash -c \
    "/workspace/target/release/fox serve --model-path /root/.cache/ferrumox/models/nanollava-text-model-f16.gguf --max-context-len $((MAX_TOKENS + 1024)) --max-batch-size $CONCURRENCY --vision-contexts 1 --port 8080 2>&1 | tee /tmp/fox.log"

echo "  Waiting for fox to load..."
for i in $(seq 1 60); do
    curl -sf "$FOX_URL/api/tags" >/dev/null 2>&1 && break
    sleep 1
done
curl -sf "$FOX_URL/api/tags" >/dev/null 2>&1 || { echo "  FAIL: fox not ready"; exit 1; }
echo "  Fox ready."

echo -e "  ${BOLD}Text chat:${NC}"
run_bench "fox-text" "$FOX_URL" "default" "txt"
echo -e "  ${BOLD}Vision:${NC}"
run_bench "fox-vision" "$FOX_URL" "default" ""
docker exec "$FOX_CONTAINER" bash -c 'pkill -f "fox serve"' 2>/dev/null || true
sleep 3

# ---------------------------------------------------------------------------
# 2. llama.cpp
# ---------------------------------------------------------------------------
echo -e "\n${BOLD}[2/3] llama.cpp${NC}"
if docker image inspect ghcr.io/ggml-org/llama.cpp:server-cuda >/dev/null 2>&1; then
    docker run -d --gpus all --name llamacpp-bench \
        -p "$LLAMACPP_PORT":8080 \
        -v "$MODEL_DIR":/models \
        ghcr.io/ggml-org/llama.cpp:server-cuda \
        --model "/models/$TEXT_MODEL" \
        --mmproj "/models/$MMPROJ" \
        --host 0.0.0.0 --port 8080 \
        --ctx-size $((( MAX_TOKENS + 1024 ) * CONCURRENCY)) --flash-attn on --n-gpu-layers 99 \
        --parallel "$CONCURRENCY" > /dev/null 2>&1

    echo "  Waiting for llama.cpp to load..."
    for i in $(seq 1 60); do
        curl -sf "http://localhost:$LLAMACPP_PORT/health" >/dev/null 2>&1 && break
        sleep 1
    done
    sleep 2
    echo "  llama.cpp ready."

    echo -e "  ${BOLD}Text chat:${NC}"
    run_bench "llamacpp-text" "http://localhost:$LLAMACPP_PORT" "default" "txt"
    echo -e "  ${BOLD}Vision:${NC}"
    run_bench "llamacpp-vision" "http://localhost:$LLAMACPP_PORT" "default" ""
    docker rm -f llamacpp-bench > /dev/null 2>&1
    sleep 3
else
    echo "  SKIP: image not pulled"
fi

# ---------------------------------------------------------------------------
# 3. Ollama
# ---------------------------------------------------------------------------
echo -e "\n${BOLD}[3/3] Ollama${NC}"
if docker image inspect ollama/ollama:latest >/dev/null 2>&1; then
    docker run -d --gpus all --name ollama-bench \
        -p "$OLLAMA_PORT":11434 \
        -v "${HOME}/.ollama:/root/.ollama" \
        ollama/ollama:latest > /dev/null 2>&1

    echo "  Waiting for ollama..."
    for i in $(seq 1 30); do
        curl -sf "http://localhost:$OLLAMA_PORT/api/tags" >/dev/null 2>&1 && break
        sleep 1
    done
    # Load the model
    curl -sf -X POST "http://localhost:$OLLAMA_PORT/api/generate" \
        -d '{"model":"qnguyen3/nanollava","prompt":"hi","options":{"num_predict":1}}' >/dev/null 2>&1 || true
    sleep 3
    echo "  Ollama ready."

    echo -e "  ${BOLD}Text chat:${NC}"
    run_bench "ollama-text" "http://localhost:$OLLAMA_PORT" "ollama" "txt"
    echo -e "  ${BOLD}Vision:${NC}"
    run_bench "ollama-vision" "http://localhost:$OLLAMA_PORT" "ollama" ""
    docker rm -f ollama-bench > /dev/null 2>&1
    sleep 3
else
    echo "  SKIP: image not pulled"
fi

# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------
ALL_LABELS="fox-text fox-vision llamacpp-text llamacpp-vision ollama-text ollama-vision"

print_table() {
    local title="$1"; shift
    local labels=("$@")
    echo ""
    echo -e "${BOLD}=== $title ($REQUESTS reqs × concurrency $CONCURRENCY × max_tokens $MAX_TOKENS) ===${NC}"
    printf "%-16s %8s %8s %8s %8s %8s %8s %6s\n" \
        "Server" "Time(s)" "Req/s" "Tok/s" "Tokens" "P50(s)" "P95(s)" "Errs"
    printf "%-16s %8s %8s %8s %8s %8s %8s %6s\n" \
        "────────────────" "────────" "────────" "────────" "────────" "────────" "────────" "──────"
    for label in "${labels[@]}"; do
        summary="$TMPDIR/$label/summary.txt"
        if [ -f "$summary" ]; then
            IFS='|' read -r elapsed rps p50 p95 errs vram vpeak total_tokens tps < "$summary"
            printf "%-16s %8s %8s %8s %8s %8s %8s %6s\n" \
                "$label" "$elapsed" "$rps" "$tps" "$total_tokens" "$p50" "$p95" "$errs"
        else
            printf "%-16s %8s\n" "$label" "SKIPPED"
        fi
    done
}

print_table "Text Chat" fox-text llamacpp-text ollama-text
print_table "Vision"    fox-vision llamacpp-vision ollama-vision

# Bar charts
BAR=40

print_bars() {
    local title="$1" metric_idx="$2" unit="$3" lower_better="${4:-0}"; shift 4
    local labels=("$@")
    echo ""
    echo -e "${BOLD}${title}${NC}"
    declare -A VAL_MAP
    local MAX_VAL=0
    for label in "${labels[@]}"; do
        [ -f "$TMPDIR/$label/summary.txt" ] || continue
        IFS='|' read -r elapsed rps p50 p95 errs vram vpeak total_tokens tps < "$TMPDIR/$label/summary.txt"
        local vals=("$elapsed" "$rps" "$p50" "$p95" "$errs" "$vram" "$vpeak" "$total_tokens" "$tps")
        local v="${vals[$metric_idx]}"
        VAL_MAP[$label]="$v"
        local v_int=${v%.*}
        [ "$v_int" -gt "$MAX_VAL" ] 2>/dev/null && MAX_VAL=$v_int
    done
    for label in "${labels[@]}"; do
        [ -n "${VAL_MAP[$label]:-}" ] || { printf "  %-16s SKIPPED\n" "$label"; continue; }
        local v="${VAL_MAP[$label]}"; local v_int=${v%.*}
        local bar_len=$(( MAX_VAL > 0 ? v_int * BAR / MAX_VAL : 0 ))
        local bar=$(printf '%0.s█' $(seq 1 $bar_len) 2>/dev/null || true)
        printf "  %-16s %s %s %s\n" "$label" "$bar" "$v" "$unit"
    done
    unset VAL_MAP
}

print_bars "Text Throughput (req/s)" 1 "req/s" 0 fox-text llamacpp-text ollama-text
print_bars "Vision Throughput (req/s)" 1 "req/s" 0 fox-vision llamacpp-vision ollama-vision
print_bars "Vision Throughput (tok/s)" 8 "tok/s" 0 fox-vision llamacpp-vision ollama-vision
print_bars "Text Latency P50 (lower=better)" 2 "s" 1 fox-text llamacpp-text ollama-text
print_bars "Vision Latency P50 (lower=better)" 2 "s" 1 fox-vision llamacpp-vision ollama-vision
echo ""
