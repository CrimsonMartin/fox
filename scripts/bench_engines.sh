#!/usr/bin/env bash
# Concurrent burst behind a shared system prompt — fox vs llama-server vs Ollama.
#
# Generalises scripts/ab_shared_prefix.sh from two arms to N. Same workload, same
# client (bench_burst.py), same discipline: exactly one server alive at a time, and the
# arm order rotates every round so thermal drift and page-cache warming cannot
# systematically favour whichever engine happens to go first.
#
# WHY THESE THREE AND NOT FOUR. Every arm here runs on **Vulkan**, on the same GPU,
# against the same GGUF file. That is what makes it a comparison of serving layers
# rather than of compute backends. vLLM has no Vulkan path and does not consume this
# GGUF, so it cannot join this run without changing both variables at once; it needs
# its own round on ROCm with its own model artifact, documented separately. Putting it
# in this table would mean publishing a backend difference as if it were an engine
# difference.
#
# THE OLLAMA ARM NEEDS THREE THINGS THAT ARE NOT DEFAULTS, and each of them is a way to
# get a fake result:
#
#   OLLAMA_IGPU_ENABLE=1   Ollama discovers the 890M, recognises it, and then discards
#                          it for being integrated — "dropping integrated GPU" — and
#                          serves from CPU without failing. `ollama ps` says 100% CPU
#                          and the numbers look like a crushing win for fox.
#   OLLAMA_NUM_PARALLEL    Defaults to 1. Eight concurrent clients then queue behind
#                          each other, and the TTFT curve that produces looks exactly
#                          like the prefix-reuse failure this benchmark exists to
#                          measure. It would be a fabricated result pointing the way
#                          the hypothesis predicts, which is the worst kind.
#   OLLAMA_CONTEXT_LENGTH  Defaults to auto, which picked 131072 for this model — 32x
#                          what the other arms get.
#
# The residency check after loading is not a formality: it is the only place Ollama
# states plainly which processor it ended up on, and the CPU fallback is silent.
#
#   MODEL=~/.cache/ferrumox/models/llama-3.2-1b-instruct-q8_0.gguf \
#   FOX_BIN=./fox-vulkan/fox LLAMA_SERVER_BIN=./llama-server-vulkan/llama-server \
#     scripts/bench_engines.sh
set -uo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-$(mktemp -d -t bench-engines-XXXX)}"
mkdir -p "$OUT"

MODEL="${MODEL:?set MODEL to a .gguf}"
NAME="${NAME:-$(basename "$MODEL" .gguf)}"
FOX_BIN="${FOX_BIN:?set FOX_BIN}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:?set LLAMA_SERVER_BIN}"
# 0.30.10's image carries the Vulkan backend; the :rocm image (0.32.5) does not. Using
# the ROCm image here would swap the backend *and* the Ollama version in one step.
OLLAMA_IMAGE="${OLLAMA_IMAGE:-ollama/ollama:latest}"
OLLAMA_TAG="${OLLAMA_TAG:-foxbench}"

ENGINES="${ENGINES:-fox llama-server ollama}"
# burst  = concurrent clients behind a shared prefix (fox's favourable workload)
# decode = N unrelated short prompts, nothing to reuse (the neutral control)
# Both belong in the write-up. Publishing only the first would be marketing, and the
# second is where fox has historically sat *below* llama-server, at 96%.
MODE="${MODE:-burst}"
PORT="${PORT:-8360}"
URL="http://127.0.0.1:$PORT"
ROUNDS="${ROUNDS:-3}"
CONC="${CONC:-8}"
REPEATS="${REPEATS:-30}"
MAXTOK="${MAXTOK:-64}"
CTX_PER_SEQ="${CTX_PER_SEQ:-4096}"
CTX=$((CTX_PER_SEQ * CONC))   # llama-server splits -c across --parallel slots
CONT="fox-bench-ollama"
OLLAMA_DATA="${OLLAMA_DATA:-$OUT/ollama}"

[ -f "$MODEL" ] || { echo "no existe el modelo: $MODEL"; exit 1; }
[ -x "$FOX_BIN" ] || { echo "no ejecutable: $FOX_BIN"; exit 1; }
[ -x "$LLAMA_SERVER_BIN" ] || { echo "no ejecutable: $LLAMA_SERVER_BIN"; exit 1; }

# A stale bundle once produced a confident and completely wrong table, so the binary
# timestamps go in the log next to the numbers rather than being checked by eye.
echo "=== ráfaga concurrente, prompt de sistema compartido ==="
echo "    modelo      $MODEL"
echo "    fox         $(date -r "$FOX_BIN" '+%F %T')"
echo "    llama-serv  $(date -r "$LLAMA_SERVER_BIN" '+%F %T')"
echo "    ollama      $OLLAMA_IMAGE"
echo "    commit      $(git -C "$S/.." rev-parse --short HEAD 2>/dev/null)"
echo "    $CONC clientes · $MAXTOK tokens · ${CTX_PER_SEQ} ctx/secuencia · $ROUNDS rondas"
echo

stop_all() {
  docker rm -f "$CONT" >/dev/null 2>&1
  # Match by listening socket, never `pkill -f`: a pattern broad enough to catch the
  # server also matches this script's own command line.
  local p
  p=$(ss -lptn "sport = :$PORT" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1)
  [ -n "$p" ] && kill "$p" 2>/dev/null
  sleep 3
}
trap 'stop_all; exit 130' INT TERM

wait_up() {
  local path="$1"
  for _ in $(seq 1 90); do curl -sf -m 2 "$URL$path" >/dev/null 2>&1 && return 0; sleep 2; done
  return 1
}

start_fox() {
  env LD_LIBRARY_PATH="$(dirname "$FOX_BIN")" "$FOX_BIN" serve \
    --model-path "$MODEL" --host 127.0.0.1 --port "$PORT" \
    --max-context-len "$CTX_PER_SEQ" --max-batch-size "$CONC" \
    > "$OUT/server_fox.log" 2>&1 &
  disown
  wait_up /health
}

start_llama_server() {
  env LD_LIBRARY_PATH="$(dirname "$LLAMA_SERVER_BIN")" "$LLAMA_SERVER_BIN" \
    -m "$MODEL" --host 127.0.0.1 --port "$PORT" -c "$CTX" -ngl 99 --parallel "$CONC" \
    > "$OUT/server_llama-server.log" 2>&1 &
  disown
  wait_up /health
}

start_ollama() {
  mkdir -p "$OLLAMA_DATA"
  docker run -d --name "$CONT" \
    --device=/dev/dri --group-add video \
    -v "$OLLAMA_DATA:/root/.ollama" -v "$MODEL:/models/model.gguf:ro" \
    -p "127.0.0.1:$PORT:11434" \
    -e OLLAMA_VULKAN=1 -e OLLAMA_IGPU_ENABLE=1 -e OLLAMA_DEBUG=1 \
    -e OLLAMA_CONTEXT_LENGTH="$CTX_PER_SEQ" \
    -e OLLAMA_NUM_PARALLEL="$CONC" \
    -e OLLAMA_MAX_LOADED_MODELS=1 \
    -e OLLAMA_KEEP_ALIVE=30m \
    "$OLLAMA_IMAGE" >/dev/null 2>&1 || return 1
  wait_up /api/version || return 1
  docker exec "$CONT" sh -c "printf 'FROM /models/model.gguf\n' > /root/Modelfile" || return 1
  docker exec "$CONT" ollama create "$OLLAMA_TAG" -f /root/Modelfile >/dev/null 2>&1 || return 1
  # Force a load so `ollama ps` has something to report, then assert it went to the GPU.
  curl -sf -m 300 "$URL/api/generate" \
    -d "{\"model\":\"$OLLAMA_TAG\",\"prompt\":\"hi\",\"stream\":false,\"options\":{\"num_predict\":4}}" \
    >/dev/null 2>&1
  local proc
  proc=$(docker exec "$CONT" ollama ps 2>/dev/null | grep -oE '[0-9]+%/?[0-9]*%? *(CPU|GPU)(/GPU)?' | head -1)
  echo "    residencia: ${proc:-desconocida}" >&2
  case "$proc" in
    *"100% GPU"*) ;;
    *) echo "    ABORTADO: Ollama no quedó 100% en GPU ($proc) — medirlo sería medir el fallback" >&2
       return 1 ;;
  esac
  docker logs "$CONT" > "$OUT/server_ollama.log" 2>&1
  return 0
}

# The model name the client must send differs per engine: fox and llama-server accept
# the file's basename, Ollama only knows the tag it was imported under.
client_model() { [ "$1" = ollama ] && echo "$OLLAMA_TAG" || echo "$NAME"; }

run_arm() {
  local eng="$1"
  stop_all
  case "$eng" in
    fox)          start_fox ;;
    llama-server) start_llama_server ;;
    ollama)       start_ollama ;;
    *) echo "  motor desconocido: $eng"; return 1 ;;
  esac || { echo "  $eng: no arrancó (ver $OUT/server_$eng.log)"; stop_all; return 1; }

  local out
  if [ "$MODE" = decode ]; then
    out=$(python3 "$S/bench_decode.py" "$URL" "$(client_model "$eng")" "$CONC" "$MAXTOK" 2>&1) || {
      echo "  $eng: el cliente falló"; echo "$out" | tail -3; stop_all; return 1; }
    while read -r phase tps agg ctok; do
      echo "$tps $agg $ctok" >> "$OUT/${phase}_$eng.dat"
      printf "  %-13s %-6s decode p50 %6s tok/s  agregado %6s tok/s  salida %s tok\n" \
             "$eng" "$phase" "$tps" "$agg" "$ctok"
    done <<< "$out"
  else
    out=$(python3 "$S/bench_burst.py" "$URL" "$(client_model "$eng")" "$CONC" "$REPEATS" "$MAXTOK" 2>&1) || {
      echo "  $eng: el cliente falló"; echo "$out" | tail -3; stop_all; return 1; }
    while read -r phase p50 p90 wall cached ptok; do
      echo "$p50 $wall $cached" >> "$OUT/${phase}_$eng.dat"
      printf "  %-13s %-5s TTFT p50 %6s ms  p90 %6s ms  wall %5ss  cached %6s  prompt %s tok\n" \
             "$eng" "$phase" "$p50" "$p90" "$wall" "$cached" "$ptok"
      if (( ptok > CTX_PER_SEQ )); then
        echo "  AVISO: el prompt ($ptok) no cabe en el contexto por secuencia ($CTX_PER_SEQ)"
      fi
    done <<< "$out"
  fi
  stop_all
}

rm -f "$OUT"/cold_*.dat "$OUT"/warm_*.dat "$OUT"/decode_*.dat
read -r -a ARMS <<< "$ENGINES"
for r in $(seq 1 "$ROUNDS"); do
  echo "ronda $r/$ROUNDS:"
  # Rotate left by (r-1) so every engine leads a round: with 3 arms and 3 rounds each
  # one runs first exactly once, which is the multi-arm version of ab_shared_prefix's
  # alternation.
  n=${#ARMS[@]}
  for i in $(seq 0 $((n - 1))); do
    run_arm "${ARMS[$(( (i + r - 1) % n ))]}"
  done
done

echo
python3 - "$OUT" "$ENGINES" "$MODE" <<'PY'
import statistics, sys
d, engines, mode = sys.argv[1], sys.argv[2].split(), sys.argv[3]


def col(phase, eng, idx):
    try:
        return [float(l.split()[idx]) for l in open(f"{d}/{phase}_{eng}.dat") if l.strip()]
    except FileNotFoundError:
        return []


# In burst mode the headline is a latency (lower wins); in decode mode a rate (higher
# wins). Getting that backwards would not crash, it would just print the winner's name
# in the loser's place, so the direction is carried explicitly.
if mode == "decode":
    phases, unit, lower_is_better = ("decode",), "tok/s", False
    head, c2, c3 = "decode p50", "agregado", "salida"
else:
    phases, unit, lower_is_better = ("cold", "warm"), "ms", True
    head, c2, c3 = "TTFT p50", "wall", "cached"

for phase in phases:
    rows = [(e, col(phase, e, 0), col(phase, e, 1), col(phase, e, 2)) for e in engines]
    rows = [r for r in rows if r[1]]
    if not rows:
        continue
    print(f"  {phase.upper()}")
    print(f"    {'motor':<14}{head:>12}{'rango':>18}{c2:>10}{c3:>9}")
    for e, main, b, c in rows:
        print(f"    {e:<14}{statistics.median(main):>8.0f} {unit:<3}"
              f"{f'[{min(main):.0f}, {max(main):.0f}]':>18}"
              f"{statistics.median(b):>10.2f}{statistics.median(c):>9.0f}")
    # Ranges, not just medians: with 3 rounds a median difference that sits inside the
    # spread is not a result, and saying so is the whole point of running 3 rounds.
    base = rows[0]
    for e, main, _, _ in rows[1:]:
        mb, me = statistics.median(base[1]), statistics.median(main)
        base_wins = (mb < me) if lower_is_better else (mb > me)
        winner, factor = (base[0], max(mb, me) / min(mb, me)) if base_wins else (e, max(mb, me) / min(mb, me))
        disjoint = min(base[1]) > max(main) or min(main) > max(base[1])
        print(f"    {base[0]} vs {e}: {winner} {factor:.2f}x  "
              + ("(rangos disjuntos)" if disjoint else "SOLAPAN — no concluyente"))
    if mode == "decode":
        # The aggregate is the number a deployment feels, and it can disagree with the
        # per-request rate — a server can decode each stream at the same speed and
        # still finish the batch sooner. Reported as its own verdict rather than left
        # as an unchecked column, because only the headline column got a range test.
        for e, _, agg, _ in rows[1:]:
            mb, me = statistics.median(base[2]), statistics.median(agg)
            winner = base[0] if mb > me else e
            disjoint = min(base[2]) > max(agg) or min(agg) > max(base[2])
            print(f"    {base[0]} vs {e} (agregado): {winner} {max(mb, me)/min(mb, me):.2f}x  "
                  + ("(rangos disjuntos)" if disjoint else "SOLAPAN — no concluyente"))
    print()
if mode != "decode":
    print("  cached_tokens: Ollama no expone prompt_tokens_details, así que su 0 significa")
    print("  'no lo reporta', no 'no reutiliza'. El TTFT sí es comparable.")
else:
    print("  'salida' son tokens de salida medidos: si difieren mucho entre motores, no")
    print("  estaban haciendo el mismo trabajo y la comparación no vale.")
PY
echo
echo "datos y logs en $OUT"
echo "(borra $OLLAMA_DATA cuando acabes: el import de Ollama duplica el GGUF)"
