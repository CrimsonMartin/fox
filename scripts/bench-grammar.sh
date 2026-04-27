#!/usr/bin/env bash
set -euo pipefail

# Grammar benchmark: measure throughput with and without GBNF grammar constraint.
# Tests bounding-box JSON output format on vision and text requests.

FOX="${FOX:-/workspace/target/release/fox}"
FOX_PORT="${FOX_PORT:-8080}"
REQUESTS="${REQUESTS:-20}"
WARMUP="${WARMUP:-3}"

BOLD='\033[1m'
GREEN='\033[0;32m'
NC='\033[0m'

MODEL_DIR="${MODEL_DIR:-/workspace/models}"
TEXT_MODEL="${TEXT_MODEL:-nanollava-text-model-f16.gguf}"

TMPDIR=$(mktemp -d)
trap 'pkill -f "fox serve" 2>/dev/null||true; rm -rf "$TMPDIR"' EXIT

# GBNF grammar for bounding box JSON output
BBOX_GRAMMAR='root ::= "[" ws bbox ("," ws bbox)* ws "]"
bbox ::= "{" ws "\"label\"" ws ":" ws string "," ws "\"x\"" ws ":" ws number "," ws "\"y\"" ws ":" ws number "," ws "\"w\"" ws ":" ws number "," ws "\"h\"" ws ":" ws number ws "}"
string ::= "\"" [a-zA-Z0-9_ ]+ "\""
number ::= [0-9]+ ("." [0-9]+)?
ws ::= [ \t\n]*'

echo -e "${BOLD}=== Grammar Benchmark ===${NC}"
echo "Requests:    $REQUESTS"
echo "Warmup:      $WARMUP"
echo ""

# Start fox
pkill -f "fox serve" 2>/dev/null || true; sleep 1
$FOX serve --model-path "$MODEL_DIR/$TEXT_MODEL" \
    --port "$FOX_PORT" --max-context-len 2048 > "$TMPDIR/fox.log" 2>&1 &
FOX_PID=$!
sleep 3

for i in $(seq 1 60); do
    curl -s "http://localhost:${FOX_PORT}/health" >/dev/null 2>&1 && break
    sleep 1
done

URL="http://localhost:${FOX_PORT}/v1/chat/completions"
PROMPT="List 3 objects in this scene with bounding boxes. Output JSON array with label, x, y, w, h fields."

# Build request bodies
cat > "$TMPDIR/req_nogrm.json" <<REQEOF
{"model":"$TEXT_MODEL","messages":[{"role":"user","content":"$PROMPT"}],"max_tokens":256,"temperature":0.1}
REQEOF

python3 -c "
import json, sys
grammar = '''$BBOX_GRAMMAR'''
req = json.load(open('$TMPDIR/req_nogrm.json'))
req['grammar'] = grammar
json.dump(req, open('$TMPDIR/req_grm.json', 'w'))
"

run_bench() {
    local label="$1" reqfile="$2" n="$3" warmup="$4"
    echo -e "  ${GREEN}${label}${NC} (${n} requests, ${warmup} warmup)"

    for ((i=0; i<warmup; i++)); do
        curl -s -o /dev/null --max-time 30 "$URL" \
            -H "Content-Type: application/json" -d @"$reqfile"
    done

    local times=()
    local start_all end_all
    start_all=$(date +%s%N)
    for ((i=0; i<n; i++)); do
        local t
        t=$(curl -s -o "$TMPDIR/resp_${label}_$i.json" -w '%{time_total}' \
            --max-time 30 "$URL" -H "Content-Type: application/json" -d @"$reqfile")
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
    local rps=$(echo "scale=2; $n / ($wall_ms / 1000)" | bc)

    local total_tokens=0
    for ((i=0; i<n; i++)); do
        local tok
        tok=$(jq -r '.usage.completion_tokens // 0' "$TMPDIR/resp_${label}_$i.json" 2>/dev/null || echo 0)
        total_tokens=$((total_tokens + tok))
    done
    local avg_tokens=$((total_tokens / n))

    printf "    avg: %8s ms   p50: %8s ms   p95: %8s ms\n" "$avg" "$p50" "$p95"
    printf "    rps: %8s      avg_tokens: %d\n" "$rps" "$avg_tokens"

    echo "  Sample output:"
    jq -r '.choices[0].message.content' "$TMPDIR/resp_${label}_0.json" 2>/dev/null | head -5 | sed 's/^/    /'
    echo ""
}

echo -e "${BOLD}Text generation (no grammar)${NC}"
run_bench "no_grammar" "$TMPDIR/req_nogrm.json" "$REQUESTS" "$WARMUP"

echo -e "${BOLD}Text generation (bbox grammar)${NC}"
run_bench "grammar" "$TMPDIR/req_grm.json" "$REQUESTS" "$WARMUP"

kill $FOX_PID 2>/dev/null; wait $FOX_PID 2>/dev/null || true

echo -e "${BOLD}=== Done ===${NC}"
