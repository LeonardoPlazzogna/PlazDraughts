#!/usr/bin/env bash
# Launch of the production run, with checks BEFORE starting.
#
#   ./run.sh
#
# It comes from a real mistake: the run was launched with `python conductor.py`
# without environment variables, hence with the defaults -- the CPU device on a
# machine with a graphics card, the C++ engine disabled, and dozens of Python
# workers fighting over the processor. The conductor printed "device=cpu" and
# nobody noticed until the process died.
#
# The lesson is not "remember the variables": it is that a wrong configuration
# must FAIL AT ONCE AND LOUDLY, not degrade silently. Here every check that does
# not pass stops the launch with the explicit reason.
#
# To override a value, pass it in the environment:
#   DAMA_CYCLES=200 ./run.sh
set -u

cd "$(dirname "$0")"
PY=$(command -v python3 || command -v python)
[ -n "$PY" ] || { echo "ERROR: no python interpreter found."; exit 1; }

# --- configuration (can be overridden from the environment) -------------------
export DAMA_DEVICE="${DAMA_DEVICE:-cuda}"
# NB: `-` and not `:-`. With `:-` an EMPTY value passed on purpose
# (DAMA_CPP_ENGINE= ./run.sh, the documented way to disable the engine) would be
# replaced by the default, that is, the option would not work. With `-` the
# default applies only if the variable is not defined at all.
export DAMA_CPP_ENGINE="${DAMA_CPP_ENGINE-$PWD/engine_c/build/dama_engine}"
export DAMA_CPP_THREADS="${DAMA_CPP_THREADS:-$(nproc 2>/dev/null || echo 8)}"
export DAMA_WEIGHTS_DIR="${DAMA_WEIGHTS_DIR:-weights}"
export DAMA_CYCLES="${DAMA_CYCLES:-100}"
LOG="${DAMA_LOG:-run.log}"

fail() { echo; echo "  X  $*"; echo; echo "Launch aborted."; exit 1; }
ok()   { echo "  ok  $*"; }

echo "=== checks before the launch ==="

# 1. CUDA -- it is not enough that nvidia-smi answers. nvidia-smi talks to the
#    driver, while PyTorch needs the CUDA runtime: a pod has already been seen
#    with a perfectly working nvidia-smi and cuInit failing. The only check that
#    counts is whether torch can really allocate on the GPU.
if [ "$DAMA_DEVICE" != "cpu" ]; then
    gpu=$("$PY" - <<'PY' 2>&1
import torch
try:
    if not torch.cuda.is_available():
        print("NO: torch.cuda.is_available() is false"); raise SystemExit
    torch.zeros(8, device="cuda") * 2      # forces cuInit + a real allocation
    print("YES: " + torch.cuda.get_device_name(0))
except Exception as e:
    print(f"NO: {type(e).__name__}: {e}")
PY
)
    case "$gpu" in
        YES:*) ok "GPU${gpu#YES:}" ;;
        *)     fail "requested device '$DAMA_DEVICE' but the GPU is not usable.
      ${gpu#NO: }
      With DAMA_DEVICE=cpu the run goes all the same but orders of magnitude
      slower: it is not an acceptable fallback on this machine.
      If the GPU is really absent, launch explicitly: DAMA_DEVICE=cpu ./run.sh" ;;
    esac
fi

# 2. C++ engine -- it is what makes the run feasible. Without it the run starts
#    anyway but takes an order of magnitude longer: that must be a decision,
#    not something suffered.
if [ -n "$DAMA_CPP_ENGINE" ]; then
    [ -x "$DAMA_CPP_ENGINE" ] || fail "C++ engine not found or not executable:
      $DAMA_CPP_ENGINE
      Build it with:   cd engine_c && ./build.sh
      Or launch without it:   DAMA_CPP_ENGINE= ./run.sh"
    ok "C++ engine: $DAMA_CPP_ENGINE ($DAMA_CPP_THREADS threads)"

    # Can it load a network? Without LibTorch the engine exists and starts, but
    # breaks at every cycle and falls back to the Python path: the run goes all
    # the same, an order of magnitude slower, without anything shouting it.
    #
    # The ENGINE IS ASKED instead of looking at `ldd`. The ldd check that was
    # here before did not tell "built without LibTorch" from "nonexistent
    # binary": in both cases the count is zero, and on a new pod it led astray
    # for two rounds. Loading a file that does not exist is attempted: the error
    # that comes back says which of the two cases it is.
    probe=$("$DAMA_CPP_ENGINE" --mode selfplay --n-games 1 --n-threads 1 \
            --n-sims 2 --weights /nonexistent_.pt 2>&1 | head -5)
    case "$probe" in
        *"WITHOUT LibTorch"*)
            fail "the C++ engine was built WITHOUT LibTorch: it cannot load the
      network and would fall back to the Python path at every cycle.
      Rebuild it:
        cd engine_c && rm -rf build
        ARCH=\$(python -c \"import torch;c=torch.cuda.get_device_capability();print(f'{c[0]}.{c[1]}')\")
        cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DWITH_LIBTORCH=ON \\
              -DCMAKE_PREFIX_PATH=/workspace/libtorch -DTORCH_CUDA_ARCH_LIST=\$ARCH
        cmake --build build -j" ;;
        *)
            ok "can load a network (LibTorch present)" ;;
    esac

    # Is the binary UP TO DATE with respect to the Python code calling it?
    #
    # It is the most insidious defect of the series, because it does not show.
    # The conductor always passes --temp-plies to the arena; an engine built
    # before the option existed exits with "unknown option", the conductor
    # catches the error and FALLS BACK TO THE PYTHON PATH. The run starts, goes,
    # and runs an order of magnitude slower -- with the old arena on top, which
    # replayed the same games every time and promoted anything. All the checks
    # above would pass: the binary exists, is executable and can load a network.
    # It is just old.
    help=$("$DAMA_CPP_ENGINE" --help 2>&1)
    for option in --temp-plies --c-puct-b --n-sims-b; do
        case "$help" in
            *"$option"*) ;;
            *) fail "the C++ engine is OLDER than the Python code: it does not know
      $option, which the conductor passes to it. It would fall back to the
      Python path at every cycle, with the defective arena, without anything
      shouting it. Rebuild it:   cmake --build engine_c/build -j" ;;
        esac
    done
    ok "engine up to date (knows the options the conductor uses)"
else
    echo "  i   C++ engine disabled: self-play through the Python path (much slower)"
fi

# 3. Dry run left on by mistake: it produces samples WITHOUT playing value. A
#    whole run like that is GPU time thrown away, and the defect does not show in
#    the metrics -- the loss goes down anyway.
[ "${DAMA_CPP_FAKE:-0}" = "0" ] || fail "DAMA_CPP_FAKE is on: the engine would run WITHOUT a network and the samples
      would have no playing value. It is fine only to test the plumbing.
      Turn it off:   unset DAMA_CPP_FAKE"

# 4. Previous state: if present, the run RESUMES. It must be said before, not after.
st="$DAMA_WEIGHTS_DIR/train_state.pkl"
if [ -f "$st" ]; then
    c=$("$PY" -c "import pickle,sys;print(pickle.load(open(sys.argv[1],'rb'))['cycle'])" "$st" 2>/dev/null || echo "?")
    ok "state found: resumes from cycle $((c+1)) (folder $DAMA_WEIGHTS_DIR)"
else
    ok "no previous state: starting from cycle 1"
fi

echo
echo "=== start: $DAMA_CYCLES cycles, device=$DAMA_DEVICE, log in $LOG ==="
echo "    games/cycle=${DAMA_GAMES:-400}  arena up to ${DAMA_SPRT_MAX_GAMES:-400}"
echo "    follow with:  ./watch.sh -f     or   tail -f $LOG"
echo "    search calibration (with the run stopped):  python calibrate.py"
echo

# -u turns off Python's buffering: without it, the log arrives in chunks and
# `tail -f` looks stuck for minutes.
#
# No `exec`, and PIPESTATUS[0] is propagated: in a pipeline the exit status is
# the one of the LAST command, that is tee, which practically always succeeds. A
# crash of the conductor would then come out as "exit 0" -- and whoever checks
# the outcome (or a supervisor that restarts on error) would see nothing.
"$PY" -u conductor.py 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
[ "$rc" -eq 0 ] || echo "[run.sh] the conductor exited with status $rc (see $LOG)"
exit "$rc"
