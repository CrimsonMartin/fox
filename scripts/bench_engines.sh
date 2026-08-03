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

# BACKEND=vulkan runs fox and llama-server as extracted native bundles; BACKEND=rocm
# runs them from their ROCm images, because a ROCm bundle needs the whole ROCm runtime
# next to it and the host deliberately does not have one installed.
#
# Both are worth publishing and they answer different questions. Vulkan is the only
# backend the three GGUF engines share, so it is where the serving-layer claim is
# isolated. ROCm is the only one all four share, and it is faster for fox on this GPU —
# but gfx1150 is not officially supported there: the images compile for gfx1100 and
# HSA_OVERRIDE_GFX_VERSION lies to the runtime about which card this is. A number
# obtained that way is real, and the configuration that produced it has to travel with
# it.
BACKEND="${BACKEND:-vulkan}"
case "$BACKEND" in vulkan|rocm) ;; *) echo "BACKEND debe ser vulkan o rocm"; exit 1 ;; esac

if [ "$BACKEND" = vulkan ]; then
  FOX_BIN="${FOX_BIN:?set FOX_BIN}"
  LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:?set LLAMA_SERVER_BIN}"
  # 0.30.10's image carries the Vulkan backend; the :rocm image (0.32.5) does not.
  OLLAMA_IMAGE="${OLLAMA_IMAGE:-ollama/ollama:latest}"
else
  FOX_IMAGE="${FOX_IMAGE:-fox:rocm-bench}"
  LS_IMAGE="${LS_IMAGE:-llamacpp:rocm}"
  # And the :rocm image is 0.32.5 against :latest's 0.30.10 — so the Ollama arm is not
  # only on a different backend between the two tables, it is a different Ollama. That
  # is forced by which image ships which backend, and it belongs in both tables.
  OLLAMA_IMAGE="${OLLAMA_IMAGE:-ollama/ollama:rocm}"
fi
OLLAMA_TAG="${OLLAMA_TAG:-foxbench}"

ENGINES="${ENGINES:-fox llama-server ollama}"
# burst  = concurrent clients behind a shared prefix (fox's favourable workload)
# decode = N unrelated short prompts, nothing to reuse (the neutral control)
# sweep  = the decode workload at rising concurrency, to find where each engine bends
# noisy  = one long prefill injected into streams that are already running
# All four belong in the write-up. Publishing only the first would be marketing, and
# `decode` is where fox has historically sat *below* llama-server.
MODE="${MODE:-burst}"
# A single concurrency is one gear of the gearbox. The knee — where added clients stop
# buying throughput and only buy latency — is the number that decides a deployment, and
# it is invisible at any fixed point.
SWEEP_LEVELS="${SWEEP_LEVELS:-1 2 4 8 16 32}"
PORT="${PORT:-8360}"
URL="http://127.0.0.1:$PORT"
ROUNDS="${ROUNDS:-3}"
CONC="${CONC:-8}"
REPEATS="${REPEATS:-30}"
MAXTOK="${MAXTOK:-64}"
# The noisy-neighbour long prompt does not fit in 4096, and an overflowing prompt fails
# differently per engine (400 vs a silently rolled window), so this mode gets its own
# default rather than a warning nobody reads.
[ "$MODE" = noisy ] && CTX_PER_SEQ="${CTX_PER_SEQ:-8192}"
CTX_PER_SEQ="${CTX_PER_SEQ:-4096}"
# Every arm must be *configured* for the largest concurrency it will be asked to serve,
# or the sweep would measure each engine's queue instead of its batching.
SRV_CONC="$CONC"
if [ "$MODE" = sweep ]; then
  for lvl in $SWEEP_LEVELS; do [ "$lvl" -gt "$SRV_CONC" ] && SRV_CONC="$lvl"; done
fi
CTX=$((CTX_PER_SEQ * SRV_CONC))   # llama-server splits -c across --parallel slots
NOISY_CLIENTS="${NOISY_CLIENTS:-4}"
CONT="fox-bench-ollama"
ENG_CONT="fox-bench-engine"     # fox / llama-server when they run from an image
OLLAMA_DATA="${OLLAMA_DATA:-$OUT/ollama}"
# gfx1150 has no ROCm kernels of its own; both images are compiled for gfx1100 and the
# override makes the runtime present the iGPU as one. Same value Dockerfile.rocm builds
# against — if they disagree the server loads and then faults mid-decode.
HSA_OVERRIDE="${HSA_OVERRIDE:-11.0.0}"
# Pass the render group by GID, not by name: minimal container images have no `render`
# entry in /etc/group and `--group-add render` fails there.
RENDER_GID="$(getent group render | cut -d: -f3)"

[ -f "$MODEL" ] || { echo "no existe el modelo: $MODEL"; exit 1; }

img_date() { docker image inspect -f '{{.Created}}' "$1" 2>/dev/null | cut -c1-19 | tr T ' '; }

if [ "$BACKEND" = vulkan ]; then
  [ -x "$FOX_BIN" ] || { echo "no ejecutable: $FOX_BIN"; exit 1; }
  [ -x "$LLAMA_SERVER_BIN" ] || { echo "no ejecutable: $LLAMA_SERVER_BIN"; exit 1; }
  FOX_STAMP="$(date -r "$FOX_BIN" '+%F %T')"
  LS_STAMP="$(date -r "$LLAMA_SERVER_BIN" '+%F %T')"
else
  docker image inspect "$FOX_IMAGE" >/dev/null 2>&1 || { echo "falta la imagen $FOX_IMAGE"; exit 1; }
  docker image inspect "$LS_IMAGE" >/dev/null 2>&1 || { echo "falta la imagen $LS_IMAGE"; exit 1; }
  FOX_STAMP="$FOX_IMAGE ($(img_date "$FOX_IMAGE"))"
  LS_STAMP="$LS_IMAGE ($(img_date "$LS_IMAGE"))"
fi

# A stale bundle once produced a confident and completely wrong table, so the build
# stamps go in the log next to the numbers rather than being checked by eye.
echo "=== ráfaga concurrente, prompt de sistema compartido ==="
echo "    backend     $BACKEND"
echo "    modelo      $MODEL"
echo "    fox         $FOX_STAMP"
echo "    llama-serv  $LS_STAMP"
echo "    ollama      $OLLAMA_IMAGE"
echo "    commit      $(git -C "$S/.." rev-parse --short HEAD 2>/dev/null)"
echo "    $CONC clientes · $MAXTOK tokens · ${CTX_PER_SEQ} ctx/secuencia · $ROUNDS rondas"
echo

stop_all() {
  docker rm -f "$CONT" "$ENG_CONT" >/dev/null 2>&1
  # Match by listening socket, never `pkill -f`: a pattern broad enough to catch the
  # server also matches this script's own command line.
  local p
  p=$(ss -lptn "sport = :$PORT" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1)
  [ -n "$p" ] && kill "$p" 2>/dev/null
  sleep 3
}
trap 'sampler_stop; stop_all; exit 130' INT TERM

wait_up() {
  local path="$1"
  for _ in $(seq 1 90); do curl -sf -m 2 "$URL$path" >/dev/null 2>&1 && return 0; sleep 2; done
  return 1
}

# ─── GPU sampling ─────────────────────────────────────────────────────────────
# Two jobs at once, and the second is why it is not optional.
#
#   KPI — on this iGPU only 2 GB is carved out as VRAM; everything above that spills
#   into GTT, which is system RAM mapped for the GPU. Reporting VRAM alone would show
#   every engine fitting comfortably and hide where the memory actually went.
#
#   ASSERTION — llama-server does not state its backend anywhere in its log, and
#   docs/design/rocm-benchmarking-2026-08.md records libggml-hip.so failing to dlopen
#   and the server falling back to CPU *with no error*. Reading the driver instead of
#   the log catches that for every engine, including the ones whose logs we cannot
#   parse. A CPU fallback shows up as a busy percentage near zero.
GPUDEV="$(ls -d /sys/class/drm/card*/device 2>/dev/null | head -1)"
SAMPLER_PID=""

gpu_read() { cat "$GPUDEV/$1" 2>/dev/null || echo 0; }

sampler_start() {
  [ -n "$GPUDEV" ] || return 0
  : > "$OUT/gpu_$1.samples"
  ( while :; do
      echo "$(gpu_read mem_info_vram_used) $(gpu_read mem_info_gtt_used) $(gpu_read gpu_busy_percent)" \
        >> "$OUT/gpu_$1.samples"
      sleep 0.5
    done ) &
  SAMPLER_PID=$!
}

sampler_stop() {
  [ -n "$SAMPLER_PID" ] && kill "$SAMPLER_PID" 2>/dev/null
  SAMPLER_PID=""
}

# Baseline is taken with no server running: the desktop compositor already holds ~1.7 GB
# of the 2 GB VRAM, so an absolute figure would be mostly other people's memory.
gpu_baseline() {
  [ -n "$GPUDEV" ] || return 0
  BASE_VRAM=$(gpu_read mem_info_vram_used)
  BASE_GTT=$(gpu_read mem_info_gtt_used)
}

gpu_report() {
  local eng="$1" f="$OUT/gpu_$1.samples"
  [ -s "$f" ] || return 0
  BUSY_MEAN=$(BV="${BASE_VRAM:-0}" BG="${BASE_GTT:-0}" python3 - "$f" "$OUT/mem_$eng.dat" <<'PY'
import os, sys
rows = [l.split() for l in open(sys.argv[1]) if l.strip()]
bv, bg = int(os.environ["BV"]), int(os.environ["BG"])
vram = max(int(r[0]) for r in rows) - bv
gtt = max(int(r[1]) for r in rows) - bg
busy = [int(r[2]) for r in rows]
mean = sum(busy) / len(busy)
mb = 1024 * 1024
with open(sys.argv[2], "a") as fh:
    fh.write(f"{vram/mb:.0f} {gtt/mb:.0f} {mean:.0f}\n")
print(f"{mean:.0f}|{vram/mb:.0f}|{gtt/mb:.0f}")
PY
)
  local mean="${BUSY_MEAN%%|*}" rest="${BUSY_MEAN#*|}"
  printf "  %-13s pico VRAM +%s MB · GTT +%s MB · GPU ocupada %s%%\n" \
         "$eng" "${rest%%|*}" "${rest##*|}" "$mean"
  # A server that never touched the GPU is not a slow arm, it is a different experiment.
  if [ "${mean:-0}" -lt 5 ]; then
    echo "  AVISO: la GPU estuvo al ${mean}% durante el arm '$eng' — probable fallback a CPU."
    echo "         No publiques esta fila sin comprobar qué backend cargó de verdad."
  fi
}

# One place where the ROCm passthrough is spelled out, so the two arms cannot drift
# apart in what they are handed. Both images were compiled for gfx1100; the override
# has to match or the server starts and then faults during decode.
rocm_run_env() {
  local image="$1"; shift
  local -a extra=()
  local kv
  for kv in "${FOX_ENV[@]}"; do extra+=(-e "$kv"); done
  docker run -d --name "$ENG_CONT" \
    --device=/dev/kfd --device=/dev/dri --group-add video --group-add "$RENDER_GID" \
    -e HSA_OVERRIDE_GFX_VERSION="$HSA_OVERRIDE" "${extra[@]}" \
    -v "$MODEL:/models/$(basename "$MODEL"):ro" \
    -p "127.0.0.1:$PORT:8080" \
    "$image" "$@" >/dev/null 2>&1
}

rocm_run() {
  local image="$1"; shift
  docker run -d --name "$ENG_CONT" \
    --device=/dev/kfd --device=/dev/dri --group-add video --group-add "$RENDER_GID" \
    -e HSA_OVERRIDE_GFX_VERSION="$HSA_OVERRIDE" \
    -v "$MODEL:/models/$(basename "$MODEL"):ro" \
    -p "127.0.0.1:$PORT:8080" \
    "$image" "$@" >/dev/null 2>&1
}

# fox-seq is fox with kv_unified off, from the SAME binary. It exists to price the
# trade: unified KV is what makes a partial seq_cp metadata-only (prefix sharing), and
# it is the leading suspect for the decode deficit. Running it as its own arm means the
# two configurations alternate inside each round instead of being compared across
# separate runs, where drift alone can manufacture a difference.
FOX_ENV=()

start_fox() {
  if [ "$BACKEND" = rocm ]; then
    rocm_run_env "$FOX_IMAGE" serve --model-path "/models/$(basename "$MODEL")" \
      --host 0.0.0.0 --port 8080 \
      --max-context-len "$CTX_PER_SEQ" --max-batch-size "$SRV_CONC" || return 1
    wait_up /health || return 1
    docker logs "$ENG_CONT" > "$OUT/server_fox.log" 2>&1
    return 0
  fi
  env LD_LIBRARY_PATH="$(dirname "$FOX_BIN")" "${FOX_ENV[@]}" "$FOX_BIN" serve \
    --model-path "$MODEL" --host 127.0.0.1 --port "$PORT" \
    --max-context-len "$CTX_PER_SEQ" --max-batch-size "$SRV_CONC" \
    > "$OUT/server_fox.log" 2>&1 &
  disown
  wait_up /health
}

start_llama_server() {
  if [ "$BACKEND" = rocm ]; then
    rocm_run "$LS_IMAGE" -m "/models/$(basename "$MODEL")" --host 0.0.0.0 --port 8080 \
      -c "$CTX" -ngl 99 --parallel "$SRV_CONC" || return 1
    wait_up /health || return 1
    docker logs "$ENG_CONT" > "$OUT/server_llama-server.log" 2>&1
    return 0
  fi
  env LD_LIBRARY_PATH="$(dirname "$LLAMA_SERVER_BIN")" "$LLAMA_SERVER_BIN" \
    -m "$MODEL" --host 127.0.0.1 --port "$PORT" -c "$CTX" -ngl 99 --parallel "$SRV_CONC" \
    > "$OUT/server_llama-server.log" 2>&1 &
  disown
  wait_up /health
}

start_ollama() {
  mkdir -p "$OLLAMA_DATA"
  # ROCm needs /dev/kfd and the render group; the Vulkan path needs neither, and giving
  # the Vulkan arm kfd anyway would let Ollama pick ROCm behind our back — the arm would
  # be labelled Vulkan while running something else.
  local dev=(--device=/dev/dri --group-add video)
  local backend_env=(-e OLLAMA_VULKAN=1)
  if [ "$BACKEND" = rocm ]; then
    dev=(--device=/dev/dri --device=/dev/kfd --group-add video --group-add "$RENDER_GID")
    backend_env=()
  fi
  docker run -d --name "$CONT" \
    "${dev[@]}" \
    -v "$OLLAMA_DATA:/root/.ollama" -v "$MODEL:/models/model.gguf:ro" \
    -p "127.0.0.1:$PORT:11434" \
    "${backend_env[@]}" -e OLLAMA_IGPU_ENABLE=1 -e OLLAMA_DEBUG=1 \
    -e OLLAMA_CONTEXT_LENGTH="$CTX_PER_SEQ" \
    -e OLLAMA_NUM_PARALLEL="$SRV_CONC" \
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
# fox and fox-seq write to the same server log path; they never run at the same time.
log_name() { [ "${1#fox}" != "$1" ] && echo fox || echo "$1"; }

run_arm() {
  local eng="$1"
  stop_all
  # Baseline after stop_all, before the server exists: anything else measures the
  # previous arm's memory as if it were this one's.
  gpu_baseline
  FOX_ENV=()
  case "$eng" in
    fox)          start_fox ;;
    fox-seq)      FOX_ENV=(FOX_KV_UNIFIED=0); start_fox ;;
    llama-server) start_llama_server ;;
    ollama)       start_ollama ;;
    *) echo "  motor desconocido: $eng"; return 1 ;;
  esac || { echo "  $eng: no arrancó (ver $OUT/server_$(log_name "$eng").log)"; sampler_stop; stop_all; return 1; }

  sampler_start "$eng"
  local out
  if [ "$MODE" = sweep ]; then
    for lvl in $SWEEP_LEVELS; do
      out=$(python3 "$S/bench_decode.py" "$URL" "$(client_model "$eng")" "$lvl" "$MAXTOK" 2>&1) || {
        echo "  $eng: el cliente falló en concurrencia $lvl"; echo "$out" | tail -3; continue; }
      read -r _ tps agg ctok itl99 <<< "$out"
      echo "$lvl $tps $agg $itl99" >> "$OUT/sweep_$eng.dat"
      printf "  %-13s c=%-3s decode p50 %6s tok/s  agregado %7s tok/s  ITL p99 %7s ms  salida %s tok\n" \
             "$eng" "$lvl" "$tps" "$agg" "$itl99" "$ctok"
    done
  elif [ "$MODE" = noisy ]; then
    out=$(python3 "$S/bench_noisy.py" "$URL" "$(client_model "$eng")" "$NOISY_CLIENTS" 2>&1) || {
      echo "  $eng: el cliente falló"; echo "$out" | tail -3; sampler_stop; stop_all; return 1; }
    read -r _ bp50 bp99 ip50 ip99 ratio lttft win nb ni lptok <<< "$out"
    echo "$bp99 $ip99 $ratio $lttft" >> "$OUT/noisy_$eng.dat"
    printf "  %-13s ITL p99 antes %6s ms → durante %8s ms  (x%s)  prefill largo %s ms / %s tok  ventana %ss  muestras %s/%s\n" \
           "$eng" "$bp99" "$ip99" "$ratio" "$lttft" "$lptok" "$win" "$nb" "$ni"
    if [ "${lptok:-0}" -gt "$CTX_PER_SEQ" ]; then
      echo "  AVISO: el prompt largo ($lptok) no cabe en el contexto por secuencia ($CTX_PER_SEQ)"
    fi
  elif [ "$MODE" = decode ]; then
    out=$(python3 "$S/bench_decode.py" "$URL" "$(client_model "$eng")" "$CONC" "$MAXTOK" 2>&1) || {
      echo "  $eng: el cliente falló"; echo "$out" | tail -3; sampler_stop; stop_all; return 1; }
    while read -r phase tps agg ctok itl99; do
      echo "$tps $agg $ctok $itl99" >> "$OUT/${phase}_$eng.dat"
      printf "  %-13s %-6s decode p50 %6s tok/s  agregado %6s tok/s  ITL p99 %6s ms  salida %s tok\n" \
             "$eng" "$phase" "$tps" "$agg" "$itl99" "$ctok"
    done <<< "$out"
  else
    out=$(python3 "$S/bench_burst.py" "$URL" "$(client_model "$eng")" "$CONC" "$REPEATS" "$MAXTOK" 2>&1) || {
      echo "  $eng: el cliente falló"; echo "$out" | tail -3; sampler_stop; stop_all; return 1; }
    while read -r phase p50 p90 wall cached ptok itl50 itl99; do
      echo "$p50 $wall $cached $itl50 $itl99" >> "$OUT/${phase}_$eng.dat"
      printf "  %-13s %-5s TTFT p50 %6s ms  p90 %6s ms  wall %5ss  cached %6s  ITL p50 %5s / p99 %6s ms  prompt %s tok\n" \
             "$eng" "$phase" "$p50" "$p90" "$wall" "$cached" "$itl50" "$itl99" "$ptok"
      if (( ptok > CTX_PER_SEQ )); then
        echo "  AVISO: el prompt ($ptok) no cabe en el contexto por secuencia ($CTX_PER_SEQ)"
      fi
    done <<< "$out"
  fi
  sampler_stop
  gpu_report "$eng"
  stop_all
}

rm -f "$OUT"/cold_*.dat "$OUT"/warm_*.dat "$OUT"/decode_*.dat "$OUT"/mem_*.dat "$OUT"/sweep_*.dat "$OUT"/noisy_*.dat
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
if mode == "sweep":
    # One table per engine: the curve matters more than any single point on it.
    for e in engines:
        try:
            rows = [[float(x) for x in l.split()] for l in open(f"{d}/sweep_{e}.dat")
                    if l.strip()]
        except FileNotFoundError:
            continue
        if not rows:
            continue
        by = {}
        for lvl, tps, agg, itl in rows:
            by.setdefault(lvl, []).append((tps, agg, itl))
        print(f"  {e}")
        print(f"    {'conc':>5}{'agregado':>12}{'rango':>16}{'por pet.':>11}{'ITL p99':>11}{'escala':>9}")
        base_agg = None
        best = (0, 0)
        for lvl in sorted(by):
            aggs = [r[1] for r in by[lvl]]
            tpss = [r[0] for r in by[lvl]]
            itls = [r[2] for r in by[lvl]]
            a = statistics.median(aggs)
            if base_agg is None:
                base_agg = a
            if a > best[1]:
                best = (lvl, a)
            # Scaling efficiency against the single-client number: 1.0 would mean each
            # added client bought a full client's worth of throughput.
            eff = a / (base_agg * lvl) if base_agg else 0
            # Ranges per level, not just medians: without them a 10% difference that
            # sits inside the round-to-round spread reads as a result.
            print(f"    {int(lvl):>5}{a:>10.0f} t/s"
                  f"{f'[{min(aggs):.0f}, {max(aggs):.0f}]':>16}"
                  f"{statistics.median(tpss):>9.1f} t/s"
                  f"{statistics.median(itls):>9.0f} ms{eff:>8.0%}")
        print(f"    pico de agregado en concurrencia {int(best[0])} ({best[1]:.0f} tok/s)")
        print()
    print("  'escala' compara con el cliente único: cuánto de un cliente entero compra")
    print("  cada cliente añadido. Cuando cae, los clientes de más sólo compran latencia.")
    raise SystemExit

if mode == "noisy":
    print("  UN PREFILL LARGO INYECTADO EN STREAMS YA EN CURSO")
    print(f"    {'motor':<14}{'ITL p99 antes':>15}{'durante':>12}{'factor':>9}{'prefill':>11}")
    for e in engines:
        try:
            rows = [[float(x) for x in l.split()] for l in open(f"{d}/noisy_{e}.dat")
                    if l.strip()]
        except FileNotFoundError:
            continue
        if not rows:
            continue
        b = statistics.median([r[0] for r in rows])
        i = statistics.median([r[1] for r in rows])
        ratio = statistics.median([r[2] for r in rows])
        lt = statistics.median([r[3] for r in rows])
        print(f"    {e:<14}{b:>12.0f} ms{i:>9.0f} ms{ratio:>8.1f}x{lt:>8.0f} ms")
    print()
    print("  El factor es lo que se lleva el usuario interactivo: 1x significa que no se")
    print("  enteró de que llegó el prompt largo; 20x que su stream se congeló.")
    raise SystemExit

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
# Memory and GPU occupancy, per engine across all rounds. VRAM and GTT are reported
# separately on purpose: this iGPU carves out only 2 GB as VRAM and everything else
# lands in GTT, so a VRAM-only figure reads as "nothing allocated" for every engine.
mem = []
for e in engines:
    try:
        rows = [[float(x) for x in l.split()] for l in open(f"{d}/mem_{e}.dat") if l.strip()]
    except FileNotFoundError:
        continue
    if rows:
        mem.append((e, max(r[0] for r in rows), max(r[1] for r in rows),
                    statistics.median([r[2] for r in rows])))
if mem:
    print("  MEMORIA Y OCUPACIÓN (pico sobre la línea base, sin servidor)")
    print(f"    {'motor':<14}{'VRAM':>9}{'GTT':>9}{'GPU':>8}")
    for e, v, g, b in mem:
        print(f"    {e:<14}{v:>7.0f} MB{g:>7.0f} MB{b:>7.0f}%")
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
