#!/usr/bin/env bash
# preflight.sh - complete pre-run validation (C++ engine + Python pipeline) and
# an estimate of how long a complete run would take. Linux mirror of
# preflight.ps1.
#
# USAGE:
#   ./preflight.sh
#   BENCH_GAMES=16 BENCH_SIMS=400 ./preflight.sh   # longer, more precise benchmark
set -uo pipefail   # NOT -e: the point is to go on and report every failure

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
REPORT="$ROOT/preflight_report_$TS.txt"
: > "$REPORT"

PASS=0
FAIL=0
log() { printf '%s\n' "$1" | tee -a "$REPORT"; }
hr()  { log ""; log "============================================================"; log "$1"; log "============================================================"; }
ok()  { log "  [PASS] $1"; PASS=$((PASS+1)); }
ko()  { log "  [FAIL] $1"; FAIL=$((FAIL+1)); }

hr "PREFLIGHT - $TS"
log "root=$ROOT"

# The interpreter is detected, not assumed: some Linux images have python3 and
# no python, and setup_pod.sh -- which detects it the same way -- ends by
# running this script. python first, so a machine that has both keeps using it.
PY=$(command -v python || command -v python3)
if [[ -z "$PY" ]]; then log "ERROR: no python interpreter on the PATH."; exit 1; fi
if ! command -v g++ >/dev/null 2>&1 && ! command -v clang++ >/dev/null 2>&1; then
    log "ERROR: no C++20 compiler (g++/clang++) on the PATH."; exit 1
fi

# ============================================================
# 0. ENVIRONMENT (python, torch, CUDA)
# ============================================================
hr "0. ENVIRONMENT (python, torch, CUDA)"
ENV_OUT="$ROOT/.preflight_env.txt"
"$PY" - > "$ENV_OUT" 2>&1 <<'PYEOF'
import os, sys
print(f"  python : {sys.version.split()[0]}")
try:
    import torch
except Exception as e:
    print(f"  torch  : CANNOT BE IMPORTED ({e})")
    sys.exit(2)
print(f"  torch  : {torch.__version__}")
avail = torch.cuda.is_available()
print(f"  CUDA available : {avail}")
if avail:
    print(f"  CUDA runtime   : {torch.version.cuda}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}          : {torch.cuda.get_device_name(i)}")
dev = os.environ.get("DAMA_DEVICE", "cpu")
srv = os.environ.get("DAMA_INFERENCE_SERVER", "auto")
print(f"  DAMA_DEVICE = {dev}   DAMA_INFERENCE_SERVER = {srv}")
if dev != "cpu" and not avail:
    print(f"  ERROR: DAMA_DEVICE={dev} but torch does NOT see any GPU.")
    print("  Install the right CUDA build of PyTorch (see requirements.txt).")
    sys.exit(1)
if dev != "cpu":
    print("  -> GPU path: the inference server will be ACTIVE (inference_server.py)")
else:
    print("  -> CPU path: pool of workers with one copy of the network each")
PYEOF
env_rc=$?
cat "$ENV_OUT" | tee -a "$REPORT"
rm -f "$ENV_OUT"
if [[ $env_rc -eq 0 ]]; then
    ok "environment consistent with DAMA_DEVICE"
else
    ko "environment NOT consistent (see above)"
fi

# ============================================================
# 1. BUILD + VALIDATION of engine_c/ (C++)
# ============================================================
hr "1. BUILD + VALIDATION of engine_c/ (C++ engine: rules, encoder, MCTS, concurrency, self-play)"
ENGC="$ROOT/engine_c"
BUILD_OUT="$ROOT/.preflight_build.txt"
if (cd "$ENGC" && ./build.sh) > "$BUILD_OUT" 2>&1; then
    build_rc=0
else
    build_rc=$?
fi
cat "$BUILD_OUT" >> "$REPORT"
if [[ $build_rc -eq 0 ]] && ! grep -qE "WRONG|MISMATCH|Segmentation|FAILED" "$BUILD_OUT"; then
    ok "engine_c: perft, encoder, MCTS, concurrency, multi-thread self-play"
    # mean batch size of the hub: if it were ~1, batching would be useless on a
    # GPU, so it is worth seeing it in the report
    grep -E "mean batch size" "$BUILD_OUT" | head -2 |
        while IFS= read -r line; do log "  $line"; done
else
    ko "engine_c build/validation did NOT pass (rc=$build_rc, see report)"
    tail -n 15 "$BUILD_OUT" | while IFS= read -r line; do log "    $line"; done
fi

# round-trip of the C++ dataset -> Python training (the C++ engine produces the
# samples, training stays in PyTorch: the format must match)
CPP_DS="$ENGC/selfplay_test.bin"
if [[ -f "$CPP_DS" ]]; then
    ds_out="$ROOT/.preflight_cppds.txt"
    if "$PY" "$ROOT/tests/test_cpp_dataset.py" "$CPP_DS" > "$ds_out" 2>&1; then ds_rc=0; else ds_rc=$?; fi
    cat "$ds_out" >> "$REPORT"
    if [[ $ds_rc -eq 0 ]]; then
        ok "C++ dataset readable by the Python training"
    else
        ko "round-trip of the C++ dataset did NOT pass"
        tail -n 8 "$ds_out" | while IFS= read -r line; do log "    $line"; done
    fi
    rm -f "$ds_out"
else
    ko "C++ dataset not produced by build.sh (expected in $CPP_DS)"
fi

rm -f "$ENGC"/perft "$ENGC"/perft.exe "$ENGC"/encoder_parity "$ENGC"/encoder_parity.exe \
      "$ENGC"/mcts_selfplay_test "$ENGC"/mcts_selfplay_test.exe \
      "$ENGC"/threading_test "$ENGC"/threading_test.exe \
      "$ENGC"/selfplay_mt_test "$ENGC"/selfplay_mt_test.exe \
      "$ENGC"/dama_engine "$ENGC"/dama_engine.exe \
      "$CPP_DS" "$BUILD_OUT"

# ============================================================
# 2. PYTHON TESTS
# ============================================================
hr "2. PYTHON TESTS (rules, AlphaZero end-to-end, inference server)"
# The tests are DISCOVERED, not listed: listed by hand they were eight out of
# the ten on disk, and a test added later was never run.
# test_cpp_dataset already runs above, with the dataset produced by the build.
for t in $(cd "$ROOT" && ls tests/test_*.py | sort); do
    [[ "$t" == *test_cpp_dataset.py ]] && continue
    out_file="$ROOT/.preflight_$(basename "$t").txt"
    if "$PY" "$ROOT/$t" > "$out_file" 2>&1; then py_rc=0; else py_rc=$?; fi
    cat "$out_file" >> "$REPORT"
    if [[ $py_rc -eq 0 ]] && ! grep -q "FAIL" "$out_file"; then
        ok "$t"
    else
        ko "$t (rc=$py_rc)"
        tail -n 8 "$out_file" | while IFS= read -r line; do log "    $line"; done
    fi
    rm -f "$out_file"
done

# ============================================================
# 3. CORES REALLY AVAILABLE
# ============================================================
hr "3. CORES REALLY AVAILABLE"
# On a shared machine the declared cores and the usable ones do not coincide,
# and the engine threads must be sized on the latter. Inside a container the
# quota is read from the system and is exact.
#
# Through benchmark.py, as in preflight.ps1, and not usable_cores.py directly:
# the measurement saturates the machine with dozens of processes and may not
# finish. benchmark.py runs it with a deadline and, if the deadline passes,
# closes the whole process tree instead of leaving the preflight frozen.
cores_out="$ROOT/.preflight_cores.txt"
"$PY" "$ROOT/benchmark.py" --cores-only --repeats 3 > "$cores_out" 2>&1 || true
cat "$cores_out" >> "$REPORT"
while IFS= read -r line; do log "$line"; done < "$cores_out"
THREADS=$(grep -oE "DAMA_CPP_THREADS=[0-9]+" "$cores_out" | grep -oE "[0-9]+$" | tail -1)
rm -f "$cores_out"
if [[ -n "$THREADS" ]]; then ok "cores measured ($THREADS threads suggested)"; else ko "core measurement did not succeed"; fi

# ============================================================
# 4. BENCHMARK: how long the complete run would take
# ============================================================
hr "4. BENCHMARK (estimated duration of the complete run)"
# It delegates to benchmark.py instead of redoing a measurement of its own here:
# a second implementation drifts from the first, and the one that lived here ran
# whole conductor cycles and measured the interpreted path even with the compiled
# engine available, that is, the wrong configuration.
#
# The arena is measured with ITS games (30, the real value), not with the
# self-play ones: the arena costs as much as self-play, and measuring it with
# fewer games makes the estimate come out too low.
BARGS=(--games "${BENCH_GAMES:-8}" --arena "${BENCH_ARENA_GAMES:-30}"
       --sims "${BENCH_SIMS:-400}" --no-cores)
if [[ -n "${DAMA_CPP_ENGINE:-}" && -x "${DAMA_CPP_ENGINE:-}" ]]; then
    BARGS+=(--engine "$DAMA_CPP_ENGINE")
    [[ -n "$THREADS" ]] && BARGS+=(--threads "$THREADS")
    [[ "${DAMA_DEVICE:-cpu}" != "cpu" ]] && BARGS+=(--fp16-compare)
else
    BARGS+=(--workers "${BENCH_WORKERS:-4}")
fi
[[ -n "${DAMA_DEVICE:-}" ]] && BARGS+=(--device "$DAMA_DEVICE")
[[ -n "${DAMA_WEIGHTS_FILE:-}" ]] && BARGS+=(--weights "$DAMA_WEIGHTS_FILE")
bench_out="$ROOT/.preflight_bench.txt"
if "$PY" "$ROOT/benchmark.py" "${BARGS[@]}" > "$bench_out" 2>&1; then b_rc=0; else b_rc=$?; fi
cat "$bench_out" >> "$REPORT"
while IFS= read -r line; do log "$line"; done < "$bench_out"
SUMMARY=$(grep -E "100 cycles =" "$bench_out" | tail -1 || true)
rm -f "$bench_out"
if [[ $b_rc -eq 0 ]]; then ok "benchmark completed"; else ko "benchmark did NOT complete (rc=$b_rc)"; fi

# ============================================================
# 5. RESUME TEST
# ============================================================
hr "5. RESUME TEST (interruption + restart)"
RESUME_DIR="$(mktemp -d)"
DAMA_WEIGHTS_DIR="$RESUME_DIR" DAMA_CYCLES=2 DAMA_GAMES=2 DAMA_SIMS=10 DAMA_ARENA_GAMES=2 \
    DAMA_SPRT_MAX_GAMES=2 DAMA_SPRT_CHUNK=2 \
    DAMA_CHANNELS=8 DAMA_BLOCKS=1 DAMA_WORKERS=1 DAMA_STRENGTH_EVERY=0 \
    "$PY" "$ROOT/conductor.py" > "$RESUME_DIR/run1.txt" 2>&1
r1rc=$?
DAMA_WEIGHTS_DIR="$RESUME_DIR" DAMA_CYCLES=4 DAMA_GAMES=2 DAMA_SIMS=10 DAMA_ARENA_GAMES=2 \
    DAMA_SPRT_MAX_GAMES=2 DAMA_SPRT_CHUNK=2 \
    DAMA_CHANNELS=8 DAMA_BLOCKS=1 DAMA_WORKERS=1 DAMA_STRENGTH_EVERY=0 \
    "$PY" "$ROOT/conductor.py" > "$RESUME_DIR/run2.txt" 2>&1
r2rc=$?
cat "$RESUME_DIR/run1.txt" "$RESUME_DIR/run2.txt" >> "$REPORT"
if [[ $r1rc -eq 0 && $r2rc -eq 0 ]] \
   && grep -q "resumed the state" "$RESUME_DIR/run2.txt" \
   && grep -q "\[cycle 003\]" "$RESUME_DIR/run2.txt" \
   && ! grep -q "\[cycle 001\]" "$RESUME_DIR/run2.txt"; then
    ok "resume: the second run restarts from cycle 3 (not from 1)"
else
    ko "resume: unexpected behavior (see report)"
fi
rm -rf "$RESUME_DIR"

# ============================================================
# 6. LibTorch backend of the C++ engine - a note, not a test
# ============================================================
hr "6. LibTorch backend of the C++ engine"
# The BINARY IS ASKED instead of deducing it from the environment variable.
#
# The previous version looked at $LIBTORCH and, if it was not set, declared "the
# engine was built WITHOUT the neural backend". It is a deduction, not a
# measurement, and on a pod just prepared by setup_pod.sh it was FALSE: the
# parity check on cuda had just passed at 0.000e+00, and two lines later the
# preflight announced the opposite. Whoever reads stops to ask which of the two
# to believe -- and is right to stop.
#
# The probe is the same as in run.sh: it tries to load a file that does not
# exist, and the error that comes back says whether the neural backend is there.
_engine=""
for _c in "$ENGC"/build/dama_engine "$ENGC"/dama_engine "$ENGC"/build/dama_engine.exe; do
    [[ -x "$_c" ]] && { _engine="$_c"; break; }
done
#
# The output is captured FIRST and compared afterwards, as run.sh does. Piping
# the engine into `grep -q` does not work under `set -o pipefail`: the probe makes
# the engine exit with an error on purpose, so the pipeline counted as failed even
# when grep found the text, and the `!` turned that into "neural backend present"
# for every binary -- a fake-backend engine included.
_probe=""
[[ -n "$_engine" ]] && _probe=$("$_engine" --mode selfplay --n-games 1 --n-threads 1 \
        --n-sims 2 --weights /nonexistent_.pt 2>&1 || true)
if [[ -n "$_engine" && "$_probe" != *"WITHOUT LibTorch"* ]]; then
    ok "the C++ engine has the neural backend (asked to the binary, not deduced)"
    log "  The check that counts is still parity_check, which compares the"
    log "  probabilities of the C++ side with those of PyTorch:"
    log "    python engine_c/tools/export_jit.py weights/champion.pt model_jit.pt cuda"
    log "    ./engine_c/build/parity_check model_jit.pt cuda"
elif [[ -n "${LIBTORCH:-}" ]]; then
    log "  LIBTORCH=$LIBTORCH"
    log "  The LibTorch backend is built and validated with:"
    log "    cmake -B engine_c/build -S engine_c -DWITH_LIBTORCH=ON \\"
    log "          -DCMAKE_PREFIX_PATH=\$LIBTORCH -DCMAKE_BUILD_TYPE=Release"
    log "    cmake --build engine_c/build -j"
    log "    python engine_c/tools/export_jit.py weights/champion.pt model_jit.pt cuda"
    log "    ./engine_c/build/parity_check model_jit.pt cuda"
    log "  The parity_check is MAKE-OR-BREAK: until it is green, the C++ engine"
    log "  must not be used to produce results."
else
    log "  [skip] the C++ engine does not have the neural backend, or the binary"
    log "  was not found. The rest -- rules, encoder, MCTS, multi-thread batching"
    log "  hub, self-play, dataset -- is already verified at point 1). To use a"
    log "  real network/the GPU, LibTorch and CMake are needed: see engine_c/README.md."
fi

# ============================================================
# SUMMARY
# ============================================================
hr "PREFLIGHT SUMMARY"
log "  PASS = $PASS    FAIL = $FAIL"
if [[ -n "${SUMMARY:-}" ]]; then
    log ""
    log "  Line to send back to whoever asked for the measurement:"
    log "  $SUMMARY"
fi
log "  report: $REPORT"
if [[ $FAIL -eq 0 ]]; then
    log ""; log ">>> PREFLIGHT PASSED."
    exit 0
else
    log ""; log ">>> PREFLIGHT FAILED ($FAIL checks) - do not start the run until they are all green."
    exit 1
fi
