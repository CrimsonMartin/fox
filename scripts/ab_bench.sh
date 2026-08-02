#!/usr/bin/env bash
# ab_bench.sh — alternating A/B benchmark of two server configurations.
#
# Companion to repeat_bench.sh. That script benchmarks servers you already have
# running and deliberately does not manage them; this one owns the whole cycle,
# because every false result recorded in docs/design/rocm-benchmarking-2026-08.md
# came from the parts it does not cover:
#
#   1. Two servers up at once. ggml's thread pool spin-waits, so an *idle*
#      second server still burns cores — and the distortion scales with each
#      server's thread count, i.e. it punishes exactly the variable under test.
#      (Measured: fox 79 vs llama-server 151 t/s with both up; 121 vs 153 alone.)
#      -> This script runs exactly one arm at a time and refuses to start if
#         something is already listening.
#
#   2. Comparing runs from different moments. Absolute numbers drift a lot over
#      a long session — one build measured 35.9 ms early in the day and 51.4 ms
#      hours later, unchanged. A single before/after pair is worthless.
#      -> This script alternates A/B/A/B and reports per-round values so drift
#         is visible, plus a warning when an arm drifts against itself.
#
#   3. A build change that silently did not apply. Stale libggml-cpu-*.so in
#      target/release get loaded regardless of what was just compiled, and a
#      libggml-hip.so that fails to dlopen falls back to CPU with no error.
#      -> This script captures what each arm actually loaded and prints it, and
#         warns when both arms load the same thing (i.e. you measured nothing).
#
#   4. Declaring a winner from overlapping noise.
#      -> The verdict is explicit: INCONCLUSIVE unless the arms' ranges are
#         disjoint. No p-values, just "these ranges overlap, this proves nothing".
#
# Usage:
#   ./scripts/ab_bench.sh \
#     --a-label generic --a-cmd './target/release/fox serve --model-path M --port 8097' \
#     --b-label zen4    --b-cmd './target/release/fox serve --model-path M --port 8097' \
#     --url http://localhost:8097 --model llama-3.2-1b-instruct-q8_0 \
#     [--rounds 3] [--metric ttft|throughput] [--prep-a CMD] [--prep-b CMD]
#
# --prep-a/--prep-b run before each start of that arm: use them to swap the
# thing under test (rebuild, move .so files, change an env var) so the two arms
# genuinely differ and you are not comparing a build against itself.
#
# Both arms should use the SAME url/port: only one runs at a time, and reusing
# the port makes that structurally obvious.
#
# Requires: curl, python3. fox-bench (or $BENCH_BIN) only for --metric throughput.

set -uo pipefail

export LC_ALL=C

die() { echo "ERROR: $*" >&2; exit 1; }
require() { command -v "$1" >/dev/null 2>&1 || die "'$1' not found in PATH"; }

A_LABEL=""; A_CMD=""; PREP_A=""
B_LABEL=""; B_CMD=""; PREP_B=""
URL=""; MODEL=""
ROUNDS=3
METRIC="ttft"
SAMPLES=10
READY_TIMEOUT=120
BENCH_BIN="${BENCH_BIN:-./target/release/fox-bench}"
CONCURRENCY=4
REQUESTS=40
MAX_TOKENS=256

while [[ $# -gt 0 ]]; do
    case "$1" in
        --a-label) A_LABEL="$2"; shift 2 ;;
        --a-cmd)   A_CMD="$2"; shift 2 ;;
        --prep-a)  PREP_A="$2"; shift 2 ;;
        --b-label) B_LABEL="$2"; shift 2 ;;
        --b-cmd)   B_CMD="$2"; shift 2 ;;
        --prep-b)  PREP_B="$2"; shift 2 ;;
        --url)     URL="$2"; shift 2 ;;
        --model)   MODEL="$2"; shift 2 ;;
        --rounds)  ROUNDS="$2"; shift 2 ;;
        --metric)  METRIC="$2"; shift 2 ;;
        --samples) SAMPLES="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --requests)    REQUESTS="$2"; shift 2 ;;
        --max-tokens)  MAX_TOKENS="$2"; shift 2 ;;
        -h|--help) sed -n '2,48p' "$0"; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

require curl; require python3
[[ -n "$A_LABEL" && -n "$A_CMD" ]] || die "--a-label/--a-cmd are required"
[[ -n "$B_LABEL" && -n "$B_CMD" ]] || die "--b-label/--b-cmd are required"
[[ -n "$URL" && -n "$MODEL" ]] || die "--url and --model are required"
[[ "$METRIC" == "ttft" || "$METRIC" == "throughput" ]] || die "--metric must be ttft or throughput"
[[ "$METRIC" == "ttft" ]] || [[ -x "$BENCH_BIN" ]] || die "fox-bench not found at $BENCH_BIN (needed for --metric throughput)"

PORT="${URL##*:}"; PORT="${PORT%%/*}"
LOGDIR=$(mktemp -d)
SERVER_PID=""

cleanup() {
    [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
    SERVER_PID=""
}
trap 'cleanup; exit 130' INT TERM

port_busy() { curl -sf -o /dev/null --max-time 2 "$URL/health" 2>/dev/null || curl -sf -o /dev/null --max-time 2 "$URL/api/tags" 2>/dev/null; }

# Refuse to run against a port someone else owns — see hazard 1 in the header.
if port_busy; then
    die "something is already serving on $URL. Stop it first: this script must be the only server running, or the numbers are meaningless."
fi

start_arm() {
    local label="$1" cmd="$2" prep="$3" log="$4"
    if [[ -n "$prep" ]]; then
        if ! bash -c "$prep" > "$log.prep" 2>&1; then
            echo "  prep for '$label' FAILED:" >&2; tail -5 "$log.prep" >&2; return 1
        fi
    fi
    # FOX_LLAMA_LOG surfaces llama.cpp's own startup lines, which is how we can
    # tell which backend .so actually got loaded (hazard 3).
    FOX_LLAMA_LOG=1 bash -c "$cmd" > "$log" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 "$READY_TIMEOUT"); do
        port_busy && return 0
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "  '$label' died on startup:" >&2; tail -5 "$log" >&2; return 1; }
        sleep 1
    done
    echo "  '$label' never became ready" >&2; return 1
}

# What did this arm actually load? Printed every round so a no-op change is
# visible instead of silently producing "no difference".
arm_fingerprint() {
    local log="$1"
    local backend so
    backend=$(grep -oE "backend: [A-Za-z0-9 ]+" "$log" 2>/dev/null | head -1 | sed 's/backend: //')
    so=$(grep -oE "libggml-[a-z0-9_.-]+\.so" "$log" 2>/dev/null | head -1)
    echo "${backend:-?} ${so:-?}" | tr -s ' '
}

measure_ttft() {
    local total=0 t
    for _ in $(seq 1 "$SAMPLES"); do
        t=$(curl -sf -o /dev/null -w "%{time_total}" --max-time 300 "$URL/v1/chat/completions" \
            -H 'Content-Type: application/json' \
            -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":1,\"temperature\":1.0}" 2>/dev/null) || return 1
        total=$(python3 -c "print($total+$t)")
    done
    python3 -c "print(f'{1000*$total/$SAMPLES:.2f}')"
}

measure_throughput() {
    "$BENCH_BIN" --url "$URL" --model "$MODEL" --concurrency "$CONCURRENCY" \
        --requests "$REQUESTS" --max-tokens "$MAX_TOKENS" 2>/dev/null \
        | awk '/Throughput/ {print $3}'
}

run_arm() {
    local label="$1" cmd="$2" prep="$3" round="$4"
    local log="$LOGDIR/${label}_r${round}.log"
    start_arm "$label" "$cmd" "$prep" "$log" || { cleanup; return 1; }
    local value
    if [[ "$METRIC" == "ttft" ]]; then value=$(measure_ttft); else value=$(measure_throughput); fi
    local fp; fp=$(arm_fingerprint "$log")
    cleanup
    sleep 2   # let the OS release the port and the thread pool wind down
    [[ -n "$value" ]] || { echo "  '$label' produced no measurement" >&2; return 1; }
    printf '%s\n' "$value" >> "$LOGDIR/$label.values"
    printf '%s\n' "$fp" >> "$LOGDIR/$label.fp"
    local unit; [[ "$METRIC" == "ttft" ]] && unit="ms" || unit="t/s"
    printf "  %-12s %8s %s   [%s]\n" "$label" "$value" "$unit" "$fp"
}

echo "=== ab_bench.sh ==="
echo "  A: $A_LABEL"
echo "  B: $B_LABEL"
echo "  Metric: $METRIC   Rounds: $ROUNDS   URL: $URL"
echo "  One server at a time; arms alternate to cancel drift."
echo

for r in $(seq 1 "$ROUNDS"); do
    echo "round $r/$ROUNDS:"
    # Alternate which arm goes first each round, so a systematic within-round
    # effect (warm page cache, thermal ramp) does not always favour the same arm.
    if (( r % 2 == 1 )); then
        run_arm "$A_LABEL" "$A_CMD" "$PREP_A" "$r" || die "arm '$A_LABEL' failed"
        run_arm "$B_LABEL" "$B_CMD" "$PREP_B" "$r" || die "arm '$B_LABEL' failed"
    else
        run_arm "$B_LABEL" "$B_CMD" "$PREP_B" "$r" || die "arm '$B_LABEL' failed"
        run_arm "$A_LABEL" "$A_CMD" "$PREP_A" "$r" || die "arm '$A_LABEL' failed"
    fi
done

echo
echo "=== verdict ==="
python3 - "$LOGDIR" "$A_LABEL" "$B_LABEL" "$METRIC" <<'PY'
import sys, statistics
logdir, a, b, metric = sys.argv[1:5]

def read(label):
    with open(f"{logdir}/{label}.values") as f:
        return [float(x) for x in f if x.strip()]

def fps(label):
    with open(f"{logdir}/{label}.fp") as f:
        return [x.strip() for x in f if x.strip()]

va, vb = read(a), read(b)
unit = "ms" if metric == "ttft" else "t/s"
lower_is_better = metric == "ttft"

for label, v in ((a, va), (b, vb)):
    print(f"  {label:12s} median={statistics.median(v):8.2f} {unit}  range=[{min(v):.2f}, {max(v):.2f}]  n={len(v)}")

# Hazard 3: if both arms fingerprint identically, the change under test never
# applied and any difference is pure noise.
fa, fb = set(fps(a)), set(fps(b))
if fa == fb:
    print(f"\n  WARNING: both arms loaded the same thing ({' / '.join(sorted(fa))}).")
    print("  The change under test did not apply — this comparison measures nothing.")

# Hazard 2: an arm drifting against itself means the session is not stable
# enough for the difference being claimed.
for label, v in ((a, va), (b, vb)):
    if len(v) > 1:
        spread = (max(v) - min(v)) / statistics.median(v)
        if spread > 0.10:
            print(f"\n  WARNING: '{label}' varies {spread*100:.0f}% against itself across rounds.")
            print("  The machine is not stable right now; treat any small difference as noise.")

# Hazard 4: overlapping ranges prove nothing, regardless of medians.
overlap = not (max(va) < min(vb) or max(vb) < min(va))
ma, mb = statistics.median(va), statistics.median(vb)
diff = (mb - ma) / ma * 100
better = None
if not overlap:
    better = (a if (ma < mb) == lower_is_better else b)

print()
if overlap:
    print(f"  VERDICT: INCONCLUSIVE — the ranges overlap ({diff:+.1f}% median difference).")
    print("  Do not claim a winner. Either the effect is smaller than this machine's")
    print("  noise, or there is no effect. More rounds may or may not separate them.")
else:
    print(f"  VERDICT: {better} wins — ranges are disjoint ({abs(diff):.1f}% median difference).")
PY

rm -rf "$LOGDIR"
