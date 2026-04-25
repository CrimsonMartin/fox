#!/usr/bin/env bash
set -euo pipefail

# Vision benchmark: fox vs llama-server vs Ollama (native, no Docker-in-Docker)
# Runs inside devcontainer where all three binaries are installed.

REQUESTS="${REQUESTS:-20}"
CONCURRENCY="${CONCURRENCY:-1}"
FOX_PORT=8080
LLAMA_PORT=8082
OLLAMA_PORT=11434

FOX="/workspace/target/release/fox"
LLAMA_SERVER="/workspace/vendor/llama.cpp/build/bin/llama-server"
MODEL_DIR="/workspace/models"
TEXT_MODEL="nanollava-text-model-f16.gguf"
MMPROJ="nanollava-mmproj-f16.gguf"

BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

TMPDIR=$(mktemp -d)
trap 'pkill -f "fox serve" 2>/dev/null||true; pkill -f llama-server 2>/dev/null||true; pkill -f "ollama serve" 2>/dev/null||true; rm -rf "$TMPDIR"' EXIT

echo -e "${BOLD}=== Vision Benchmark (native) ===${NC}"
echo "Requests:    $REQUESTS"
echo "Concurrency: $CONCURRENCY"
echo "Model:       nanollava"
echo ""

# Generate unique images
echo -n "Generating $REQUESTS unique images..."
python3 - "$REQUESTS" "$TMPDIR" << 'PYEOF'
import struct, zlib, base64, os, sys

def make_png(r, g, b):
    width, height = 256, 256
    raw = b''
    for y in range(height):
        raw += b'\x00' + bytes([r, g, b]) * width
    def chunk(ctype, data):
        c = ctype + data
        return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)
    ihdr = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', ihdr) + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b'')

n = int(sys.argv[1])
outdir = sys.argv[2]

for i in range(n):
    r = (i * 7 + 31) % 256
    g = (i * 13 + 97) % 256
    b = (i * 19 + 173) % 256
    b64 = base64.b64encode(make_png(r, g, b)).decode()
    for model in ["nanollava-text-model-f16", "qnguyen3/nanollava"]:
        suffix = "ollama" if "/" in model else "default"
        with open(os.path.join(outdir, 'req_%s_%d.json' % (suffix, i+1)), 'w') as f:
            f.write('{"model":"' + model + '","messages":[{"role":"user","content":[{"type":"text","text":"What color is this image? Answer in one word."},{"type":"image_url","image_url":{"url":"data:image/png;base64,' + b64 + '"}}]}],"max_tokens":16}')
PYEOF
echo " done"

# Benchmark runner
run_bench() {
    local label="$1" url="$2" req_prefix="$3"
    local outdir="$TMPDIR/$label"
    mkdir -p "$outdir"

    echo -n "  Waiting for server..."
    for i in $(seq 1 120); do
        if curl -s "$url/health" >/dev/null 2>&1 || curl -s "$url/v1/models" >/dev/null 2>&1; then
            echo " ready"; break
        fi
        [ "$i" -eq 120 ] && { echo " TIMEOUT"; return 1; }
        sleep 1
    done

    echo -n "  Warming up..."
    curl -s --max-time 60 "$url/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d @"$TMPDIR/req_${req_prefix}_1.json" -o /dev/null 2>&1 || true
    echo " done"

    echo "  Running $REQUESTS requests at concurrency $CONCURRENCY..."
    local START END ELAPSED_MS
    START=$(date +%s%N)

    local PIDS=()
    local ACTIVE=0
    for i in $(seq 1 "$REQUESTS"); do
        curl -s --connect-timeout 10 --max-time 60 -o "$outdir/resp_$i.json" \
            -w "%{time_total}\n" \
            "$url/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d @"$TMPDIR/req_${req_prefix}_$i.json" \
            > "$outdir/time_$i.txt" 2>&1 &
        PIDS+=($!)
        ACTIVE=$((ACTIVE + 1))
        if [ "$ACTIVE" -ge "$CONCURRENCY" ]; then
            wait -n 2>/dev/null || true
            ACTIVE=$((ACTIVE - 1))
        fi
    done
    for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done

    # Check server health after bench
    if ! curl -s --max-time 5 "$url/health" >/dev/null 2>&1 \
       && ! curl -s --max-time 5 "$url/v1/models" >/dev/null 2>&1; then
        echo -e "  ${RED}WARNING: server appears to have crashed${NC}"
    fi

    END=$(date +%s%N)
    ELAPSED_MS=$(( (END - START) / 1000000 ))

    local TIMES=() ERRORS=0 TOTAL_TOKENS=0
    for i in $(seq 1 "$REQUESTS"); do
        [ -f "$outdir/time_$i.txt" ] && TIMES+=("$(head -1 "$outdir/time_$i.txt")")
        if [ -f "$outdir/resp_$i.json" ]; then
            grep -q '"error"' "$outdir/resp_$i.json" 2>/dev/null && ERRORS=$((ERRORS + 1))
            local TOK
            TOK=$(jq -r '.usage.completion_tokens // 0' "$outdir/resp_$i.json" 2>/dev/null || echo 0)
            TOTAL_TOKENS=$((TOTAL_TOKENS + TOK))
        fi
    done

    printf '%s\n' "${TIMES[@]}" | sort -n > "$outdir/sorted.txt"
    local COUNT=${#TIMES[@]}
    local p50_idx=$(( COUNT * 50 / 100 ))
    local p95_idx=$(( COUNT * 95 / 100 ))
    [ "$p50_idx" -ge "$COUNT" ] && p50_idx=$((COUNT - 1))
    [ "$p95_idx" -ge "$COUNT" ] && p95_idx=$((COUNT - 1))

    local P50 P95 ELAPSED_S RPS
    P50=$(sed -n "$((p50_idx + 1))p" "$outdir/sorted.txt")
    P95=$(sed -n "$((p95_idx + 1))p" "$outdir/sorted.txt")
    ELAPSED_S=$(echo "scale=2; $ELAPSED_MS / 1000" | bc)
    RPS=$(echo "scale=2; $REQUESTS / ($ELAPSED_MS / 1000)" | bc)

    echo "${ELAPSED_S}|${RPS}|${P50}|${P95}|${ERRORS}|${TOTAL_TOKENS}" > "$outdir/summary.txt"
    echo "  Done: ${ELAPSED_S}s, ${RPS} req/s, P50=${P50}s, ${ERRORS} errors, ${TOTAL_TOKENS} tokens"
}

# ── 1. Fox ──
echo -e "\n${BOLD}[1/3] Fox${NC}"
pkill -f "fox serve" 2>/dev/null || true; sleep 1
$FOX serve --model-path "$MODEL_DIR/$TEXT_MODEL" \
    --port "$FOX_PORT" --max-context-len 2048 \
    --vision-contexts "$CONCURRENCY" > "$TMPDIR/fox_server.log" 2>&1 &
FOX_PID=$!
sleep 3
run_bench "fox" "http://localhost:$FOX_PORT" "default" || true
kill $FOX_PID 2>/dev/null; wait $FOX_PID 2>/dev/null || true; sleep 2

# ── 2. llama-server ──
echo -e "\n${BOLD}[2/3] llama-server${NC}"
pkill -f llama-server 2>/dev/null || true; sleep 1
$LLAMA_SERVER \
    --model "$MODEL_DIR/$TEXT_MODEL" \
    --mmproj "$MODEL_DIR/$MMPROJ" \
    --host 0.0.0.0 --port "$LLAMA_PORT" \
    --ctx-size 8192 --flash-attn on -ngl 99 \
    --parallel "$CONCURRENCY" > "$TMPDIR/llama_server.log" 2>&1 &
LLAMA_PID=$!
sleep 5
run_bench "llamacpp" "http://localhost:$LLAMA_PORT" "default" || true
kill $LLAMA_PID 2>/dev/null; wait $LLAMA_PID 2>/dev/null || true; sleep 2

# ── 3. Ollama ──
echo -e "\n${BOLD}[3/3] Ollama${NC}"
pkill -f "ollama serve" 2>/dev/null || true; sleep 1
OLLAMA_HOST="0.0.0.0:${OLLAMA_PORT}" ollama serve > "$TMPDIR/ollama_server.log" 2>&1 &
OLLAMA_PID=$!
sleep 3
run_bench "ollama" "http://localhost:$OLLAMA_PORT" "ollama" || true
kill $OLLAMA_PID 2>/dev/null; wait $OLLAMA_PID 2>/dev/null || true; sleep 2

# ── Summary ──
echo ""
echo -e "${BOLD}=== Vision Results ($REQUESTS requests, concurrency $CONCURRENCY) ===${NC}"
printf "%-16s %8s %8s %8s %8s %6s %6s\n" "Server" "Time(s)" "Req/s" "P50(s)" "P95(s)" "Errs" "Tokens"
printf "%-16s %8s %8s %8s %8s %6s %6s\n" "────────────────" "────────" "────────" "────────" "────────" "──────" "──────"

for label in fox llamacpp ollama; do
    summary="$TMPDIR/$label/summary.txt"
    if [ -f "$summary" ]; then
        IFS='|' read -r elapsed rps p50 p95 errs tokens < "$summary"
        printf "%-16s %8s %8s %8s %8s %6s %6s\n" "$label" "$elapsed" "$rps" "$p50" "$p95" "$errs" "$tokens"
    else
        printf "%-16s %8s\n" "$label" "SKIPPED"
    fi
done
echo ""
