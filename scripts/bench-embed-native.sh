#!/usr/bin/env bash
set -euo pipefail

# Embedding benchmark: fox vs Ollama (native, GPU)
# Runs inside devcontainer where both binaries are installed.

FOX="/workspace/target/release/fox"
FOX_PORT=8080
OLLAMA_PORT=11434

REQUESTS="${REQUESTS:-50}"
WARMUP="${WARMUP:-5}"

BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# nomic-embed-text pulled via ollama; fox needs the GGUF
# Ollama stores models in blobs; we need to find the actual GGUF path
OLLAMA_MODEL="nomic-embed-text"

SHORT_TEXT="The quick brown fox jumps over the lazy dog."
MEDIUM_TEXT="Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. These systems improve their performance on a specific task over time without being explicitly programmed. Common approaches include supervised learning, unsupervised learning, and reinforcement learning."
LONG_TEXT="${MEDIUM_TEXT} ${MEDIUM_TEXT} ${MEDIUM_TEXT} ${MEDIUM_TEXT}"

cleanup() {
    pkill -f "fox serve" 2>/dev/null || true
    pkill -f "ollama serve" 2>/dev/null || true
}
trap cleanup EXIT

time_request_ms() {
    local url="$1" data="$2"
    curl -s -o /dev/null -w '%{time_total}' \
        -X POST "$url" -H "Content-Type: application/json" -d "$data" \
        | awk '{printf "%.2f", $1 * 1000}'
}

run_bench() {
    local label="$1" url="$2" data="$3" n="$4" warmup="$5"
    echo -e "  ${BOLD}${label}${NC} (${n} requests, ${warmup} warmup)"

    for ((i=0; i<warmup; i++)); do
        curl -s -o /dev/null -X POST "$url" -H "Content-Type: application/json" -d "$data"
    done

    local times=()
    local start_all end_all
    start_all=$(date +%s%N)
    for ((i=0; i<n; i++)); do
        local t
        t=$(time_request_ms "$url" "$data")
        times+=("$t")
    done
    end_all=$(date +%s%N)
    local wall_ms=$(( (end_all - start_all) / 1000000 ))

    IFS=$'\n' sorted=($(sort -g <<<"${times[*]}")); unset IFS
    local sum=0
    for t in "${times[@]}"; do sum=$(echo "$sum + $t" | bc); done
    local avg=$(echo "scale=2; $sum / $n" | bc)
    local p50=${sorted[$(( n / 2 ))]}
    local p95=${sorted[$(( n * 95 / 100 ))]}
    local min=${sorted[0]}
    local max=${sorted[$(( n - 1 ))]}
    local rps=$(echo "scale=2; $n / ($wall_ms / 1000)" | bc)

    printf "    avg: %8s ms   p50: %8s ms   p95: %8s ms\n" "$avg" "$p50" "$p95"
    printf "    min: %8s ms   max: %8s ms   rps: %8s\n" "$min" "$max" "$rps"
}

echo -e "${BOLD}=== Embedding Benchmark (GPU, native) ===${NC}\n"

# Find nomic-embed-text GGUF path from Ollama's blob store
echo "Locating nomic-embed-text model file..."
GGUF_PATH=""
for f in /root/.ollama/models/blobs/*; do
    if [ -f "$f" ] && file "$f" 2>/dev/null | grep -qi "GGUF\|data"; then
        size=$(stat -c%s "$f" 2>/dev/null || echo 0)
        # nomic-embed-text Q4_0 is ~260MB, Q8_0 is ~520MB, f16 is ~520MB
        if [ "$size" -gt 100000000 ] && [ "$size" -lt 1000000000 ]; then
            GGUF_PATH="$f"
            echo "  Found: $f ($(( size / 1048576 )) MB)"
            break
        fi
    fi
done

if [ -z "$GGUF_PATH" ]; then
    echo "  Could not find nomic-embed-text GGUF in Ollama blobs."
    echo "  Trying to find via manifest..."
    MANIFEST=$(find /root/.ollama/models/manifests -path "*nomic-embed*" -type f 2>/dev/null | head -1)
    if [ -n "$MANIFEST" ]; then
        DIGEST=$(jq -r '.layers[] | select(.mediaType | contains("model")) | .digest' "$MANIFEST" 2>/dev/null | head -1)
        if [ -n "$DIGEST" ]; then
            GGUF_PATH="/root/.ollama/models/blobs/${DIGEST//:/-}"
            echo "  Found via manifest: $GGUF_PATH"
        fi
    fi
fi

# ── 1. Ollama ──
echo -e "\n${BOLD}[1/2] Ollama${NC}"
pkill -f "ollama serve" 2>/dev/null || true; sleep 1
OLLAMA_HOST="0.0.0.0:${OLLAMA_PORT}" ollama serve > /tmp/ollama_embed.log 2>&1 &
sleep 3

# Wait for readiness
for i in $(seq 1 30); do
    curl -s "http://localhost:${OLLAMA_PORT}/api/tags" >/dev/null 2>&1 && break
    sleep 1
done

# Warmup (loads model into GPU)
echo "  Loading model..."
curl -s -o /dev/null -X POST "http://localhost:${OLLAMA_PORT}/api/embed" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${OLLAMA_MODEL}\",\"input\":\"warmup\"}" || true
sleep 2

OLLAMA_URL="http://localhost:${OLLAMA_PORT}/api/embed"
OLL_SHORT="{\"model\":\"${OLLAMA_MODEL}\",\"input\":\"${SHORT_TEXT}\"}"
OLL_MED="{\"model\":\"${OLLAMA_MODEL}\",\"input\":\"${MEDIUM_TEXT}\"}"
OLL_LONG="{\"model\":\"${OLLAMA_MODEL}\",\"input\":\"${LONG_TEXT}\"}"

echo -e "\n${GREEN}Short text (${#SHORT_TEXT} chars)${NC}"
run_bench "Ollama" "$OLLAMA_URL" "$OLL_SHORT" "$REQUESTS" "$WARMUP"
echo -e "${GREEN}Medium text (${#MEDIUM_TEXT} chars)${NC}"
run_bench "Ollama" "$OLLAMA_URL" "$OLL_MED" "$REQUESTS" "$WARMUP"
echo -e "${GREEN}Long text (${#LONG_TEXT} chars)${NC}"
run_bench "Ollama" "$OLLAMA_URL" "$OLL_LONG" "$REQUESTS" "$WARMUP"

# Batch
BATCH_TEXTS='["The quick brown fox.","Machine learning enables computers.","Rust is a systems language.","Docker packages apps.","Neural networks mimic brains.","K8s orchestrates containers.","NLP handles language.","PostgreSQL is powerful.","WASM enables native perf.","Transformers changed NLU."]'
OLL_BATCH="{\"model\":\"${OLLAMA_MODEL}\",\"input\":${BATCH_TEXTS}}"
echo -e "${GREEN}Batch (10 texts)${NC}"
run_bench "Ollama (batch)" "$OLLAMA_URL" "$OLL_BATCH" "$((REQUESTS / 2))" "$WARMUP"

pkill -f "ollama serve" 2>/dev/null || true; sleep 2

# ── 2. Fox ──
echo -e "\n${BOLD}[2/2] Fox${NC}"
if [ -z "$GGUF_PATH" ]; then
    echo "  SKIP: no GGUF path found for nomic-embed-text"
else
    pkill -f "fox serve" 2>/dev/null || true; sleep 1
    FOX_MODEL="$(basename "$GGUF_PATH" .gguf)"
    $FOX serve --model-path "$GGUF_PATH" \
        --port "$FOX_PORT" > /tmp/fox_embed.log 2>&1 &
    FOX_PID=$!
    sleep 3

    for i in $(seq 1 30); do
        curl -s "http://localhost:${FOX_PORT}/health" >/dev/null 2>&1 && break
        sleep 1
    done

    # Warmup
    curl -s -o /dev/null -X POST "http://localhost:${FOX_PORT}/api/embed" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"${FOX_MODEL}\",\"input\":\"warmup\"}" || true
    sleep 1

    FOX_URL="http://localhost:${FOX_PORT}/api/embed"
    FOX_OPENAI_URL="http://localhost:${FOX_PORT}/v1/embeddings"
    FOX_SHORT="{\"model\":\"${FOX_MODEL}\",\"input\":\"${SHORT_TEXT}\"}"
    FOX_MED="{\"model\":\"${FOX_MODEL}\",\"input\":\"${MEDIUM_TEXT}\"}"
    FOX_LONG="{\"model\":\"${FOX_MODEL}\",\"input\":\"${LONG_TEXT}\"}"

    echo -e "\n${GREEN}Short text (${#SHORT_TEXT} chars)${NC}"
    run_bench "Fox (Ollama API)" "$FOX_URL" "$FOX_SHORT" "$REQUESTS" "$WARMUP"
    run_bench "Fox (OpenAI API)" "$FOX_OPENAI_URL" "$FOX_SHORT" "$REQUESTS" "$WARMUP"
    echo -e "${GREEN}Medium text (${#MEDIUM_TEXT} chars)${NC}"
    run_bench "Fox (Ollama API)" "$FOX_URL" "$FOX_MED" "$REQUESTS" "$WARMUP"
    echo -e "${GREEN}Long text (${#LONG_TEXT} chars)${NC}"
    run_bench "Fox (Ollama API)" "$FOX_URL" "$FOX_LONG" "$REQUESTS" "$WARMUP"

    # Batch
    FOX_BATCH="{\"model\":\"${FOX_MODEL}\",\"input\":${BATCH_TEXTS}}"
    echo -e "${GREEN}Batch (10 texts)${NC}"
    run_bench "Fox (batch)" "$FOX_URL" "$FOX_BATCH" "$((REQUESTS / 2))" "$WARMUP"

    kill $FOX_PID 2>/dev/null; wait $FOX_PID 2>/dev/null || true
fi

echo -e "\n${BOLD}=== Done ===${NC}"
