#!/usr/bin/env bash
# Prepares a fresh pod for a run: checks, PyTorch, LibTorch, C++ engine.
#
#   ./setup_pod.sh
#
# It comes from a day lost rebuilding the environment by hand three times. The
# order of the steps is not random: the most important check is the FIRST, and if
# it does not pass the script stops there. On a pod with the GPU not exposed
# correctly, PyTorch was upgraded, LibTorch downloaded again and the engine
# rebuilt before it turned out that none of it mattered -- the container did not
# initialize CUDA and would not have done so in any case.
#
# THE RULE: `nvidia-smi` answering does NOT mean that CUDA works. nvidia-smi
# talks to the driver; PyTorch needs the runtime, and the two can diverge. The
# only proof that counts is really allocating a tensor on the GPU.
#
# Useful variables:
#   DAMA_CUDA_TAG=cu128     PyTorch build to use (default: deduced from the GPU)
#   DAMA_LIBTORCH=/workspace/libtorch    where to install LibTorch
#   DAMA_SKIP_PREFLIGHT=1   skips the final suite (faster, less safe)
set -u

cd "$(dirname "$0")"
ROOT="$PWD"
LIBTORCH="${DAMA_LIBTORCH:-/workspace/libtorch}"
PY=$(command -v python3 || command -v python)

step() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }
ok()   { echo "  ok  $*"; }
fail() { printf '\n  X  %s\n\nSetup aborted.\n' "$*"; exit 1; }

# --- 0. is the pod worth it? --------------------------------------------------
#
# Two checks that cost ten seconds and that BEFORE came later or were not there
# at all. Both come from real pods thrown away after minutes of work:
#
#   * CUDA not initializing: it was step 2, that is after a package installation
#     taking a minute. Now it is the first, because it is the most likely and
#     the most sudden -- nvidia-smi answers, the card is visible, and torch
#     allocates nothing.
#
#   * CPU QUOTA: it was not there. `nproc` reports the cores of the HOST, not the
#     ones the container can use, and the difference can be an order of
#     magnitude, with the same factor on the playing speed. The load of the
#     container must be sized on the quota, not on nproc.
step "0/6  is the pod usable?"

# Read in pure shell, without awk. It is not pedantry: the first version of this
# check used awk, which on one pod WAS NOT INSTALLED, and the command broke
# silently printing an empty line -- a check that checks nothing and does not
# say so is worse than no check. Both cgroup formats are read, because pods use
# now one, now the other.
NPROC=$(nproc 2>/dev/null || echo 1)
USABLE_CORES=""
if [ -r /sys/fs/cgroup/cpu.max ]; then                    # cgroup v2
    read -r _q _p < /sys/fs/cgroup/cpu.max
    [ "$_q" != "max" ] && USABLE_CORES=$(( _q / _p ))
elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then     # cgroup v1
    _q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
    _p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
    [ "$_q" -gt 0 ] 2>/dev/null && USABLE_CORES=$(( _q / _p ))
else
    echo "  !   CPU quota not readable: no known cgroup file."
fi
if [ -z "$USABLE_CORES" ]; then
    ok "no CPU quota: all $NPROC cores available"
    USABLE_CORES=$NPROC
else
    [ "$USABLE_CORES" -lt 1 ] && USABLE_CORES=1
    if [ "$USABLE_CORES" -lt $((NPROC / 2)) ]; then
        echo "  !   WARNING: nproc declares $NPROC cores but the quota grants $USABLE_CORES."
        echo "      Self-play is limited by the CPU, not by the GPU: the speed will"
        echo "      follow the QUOTA. Use DAMA_CPP_THREADS=$USABLE_CORES at launch, and"
        echo "      consider a pod with more vCPUs if the times do not add up."
    else
        ok "CPU quota: $USABLE_CORES cores out of $NPROC declared"
    fi
fi
echo "  suggested:  DAMA_CPP_THREADS=$USABLE_CORES"

# HISTORICAL NOTE, so that the mistake is not repeated: the number of the device
# node says NOTHING.
#
# The container receives one of the host's GPUs keeping its numbering, and it
# often happens to find /dev/nvidia2 or /dev/nvidia6 without any /dev/nvidia0.
# On a series of failed pods that condition recurred, and it was taken for the
# cause -- going as far as having the script REJECT the pod.
#
# It was wrong twice over. First a pod came with /dev/nvidia0 that broke all the
# same; then one with /dev/nvidia2 that works perfectly, with cuInit at zero.
# The check, had it stayed a rejection, would have thrown away a good pod
# because of a correlation seen in a handful of cases.
#
# What decides is the CODE returned by cuInit, which names the cause: it is
# collected by check_pod.sh, and it is the only thing to look at. Below, the
# equivalent test, which is a measurement and not a clue.

# CUDA: not "nvidia-smi answers" but "torch really allocates".
gpu=$("$PY" - <<'PY' 2>&1
import torch
try:
    torch.zeros(8, device="cuda") * 2
    print("YES: " + torch.cuda.get_device_name(0))
except Exception as e:
    print(f"NO: {type(e).__name__}: {e}")
PY
)
case "$gpu" in
    YES:*) ok "CUDA initializes${gpu#YES:}" ;;
    *)     fail "CUDA does NOT initialize on this pod.
      ${gpu#NO: }
      nvidia-smi can answer all the same: it talks to the driver, not to the runtime.
      Do NOT reinstall torch or LibTorch, they have nothing to do with it. Restart
      the pod from the provider's panel and launch again, or ask for another one.
      Quick checks before discarding it:
        echo \$CUDA_VISIBLE_DEVICES     (if it contains a strange index, fix it)
        ls /dev/nvidia0 /dev/nvidiactl  (if they are missing, the container has no GPU)" ;;
esac

# --- 1. is the GPU really usable? ---------------------------------------------
step "1/6  tools and GPU"
# tmux included: the run takes hours and must be detached from the terminal, so
# without it one does not get to the end. It was missing, and one found out only
# at launch time -- that is, after all the rest of the setup.
for t in git cmake g++ unzip wget tmux; do
    command -v $t >/dev/null || miss="${miss-} $t"
done
if [ -n "${miss-}" ]; then
    echo "  installing:${miss}"
    apt-get update -qq && apt-get install -y -qq ${miss} \
        || fail "cannot install:${miss}"
fi
ok "tools present"

command -v nvidia-smi >/dev/null || fail "nvidia-smi absent: this pod has no GPU."
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | sed 's/^/  /'
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
[ -n "$CAP" ] || fail "nvidia-smi does not report the capability: driver too old?"
ok "capability of the card: $CAP"

# The PyTorch build depends on the architecture: Blackwell (12.x) exists only in
# the cu128 builds onwards, the earlier ones do not have a single usable kernel.
# Two-digit capability = Blackwell or later (10.x datacenter, 12.x consumer such
# as the RTX 5090): they exist only from the cu128 builds onwards. An 8.x/9.x
# capability (Ada, Hopper) is fine with cu124 too.
case "$CAP" in
    1[0-9].*) DEFAULT_TAG=cu128 ;;
    *)        DEFAULT_TAG=cu124 ;;
esac
TAG="${DAMA_CUDA_TAG:-$DEFAULT_TAG}"
ok "PyTorch build to use: $TAG"

# --- 2. PyTorch, and the gate that counts -------------------------------------
step "2/6  PyTorch"
need_install=1
if "$PY" -c "import torch" 2>/dev/null; then
    cur=$("$PY" -c "import torch;print(torch.__version__)")
    echo "  already present: $cur"
    case "$cur" in *"+$TAG") need_install=0 ;; esac
fi
if [ "$need_install" = 1 ]; then
    echo "  installing torch $TAG (a few GB, it takes a few minutes)..."
    "$PY" -m pip install -q --upgrade torch --index-url \
        "https://download.pytorch.org/whl/$TAG" || fail "torch installation did not succeed"
fi

# THE GATE. is_available() is not enough, and not even allocating a tensor is: a
# CONVOLUTION is run, that is exactly the kind of operation the network runs on.
# This way the test covers both ways of breaking:
#   - the container does not initialize CUDA (cuInit breaks);
#   - CUDA starts but the PyTorch build has no kernels for this architecture, a
#     case in which trivial operations pass and convolutions do not.
# Looking at the list of torch.cuda.get_arch_list() would be more fragile: JIT
# compilation from PTX can cover architectures that are not listed, so an
# absence does not prove an incompatibility. Running the computation does.
probe=$("$PY" - <<'PY' 2>&1
import torch
import torch.nn.functional as F
try:
    x = torch.randn(2, 3, 8, 8, device="cuda")
    w = torch.randn(4, 3, 3, 3, device="cuda")
    v = float(F.conv2d(x, w).sum())
    if v != v:                       # NaN: kernel run but absurd result
        raise RuntimeError("the convolution produced NaN")
    print("OK", torch.cuda.get_device_name(0), "|", ",".join(torch.cuda.get_arch_list()))
except Exception as e:
    print(f"NO {type(e).__name__}: {e}")
PY
)
case "$probe" in
    OK*) ok "GPU usable -> ${probe#OK }" ;;
    *)   fail "CUDA does NOT initialize on this pod.
      ${probe#NO }
      It is not a version problem: PyTorch is installed correctly
      ($("$PY" -c 'import torch;print(torch.__version__)')) but the container does
      not expose the GPU to the runtime. nvidia-smi can answer all the same: it
      talks to the driver, not to the runtime.
      Do NOT waste time reinstalling torch or LibTorch. Restart the pod, or ask
      for another one, and launch this script again." ;;
esac

# Informational: the convolution above has already shown that the card is usable,
# so an architecture that is not listed is not an error -- it is printed only to
# have in the log what this build is made of.
arch_list=${probe#*| }
SM="sm_$(echo "$CAP" | tr -d '.')"
case "$arch_list" in
    *"$SM"*) ok "native $SM kernels present" ;;
    *) echo "  note: $SM is not among the compiled architectures ($arch_list)."
       echo "        The test convolution succeeded anyway, so the card works;"
       echo "        the first start can be slower." ;;
esac

"$PY" -m pip install -q -r requirements.txt || fail "requirements.txt cannot be installed"
ok "Python dependencies installed"

# --- 3. LibTorch, in the EXACT version of torch -------------------------------
step "3/6  LibTorch"
read -r TVER TTAG <<EOF
$("$PY" -c "import torch;print(torch.__version__.split('+')[0], 'cu'+torch.version.cuda.replace('.',''))")
EOF
URL="https://download.pytorch.org/libtorch/$TTAG/libtorch-cxx11-abi-shared-with-deps-$TVER%2B$TTAG.zip"
if [ -f "$LIBTORCH/build-version" ] && grep -q "$TVER" "$LIBTORCH/build-version" 2>/dev/null; then
    ok "already present and aligned ($TVER+$TTAG)"
else
    echo "  downloading $TVER+$TTAG ..."
    rm -rf "$LIBTORCH" /tmp/libtorch.zip
    wget -q --show-progress "$URL" -O /tmp/libtorch.zip \
        || fail "download did not succeed: $URL
      That version/CUDA combination does not exist as LibTorch.
      Take the closest URL from https://pytorch.org/get-started/locally/
      (LibTorch section, Linux, C++) and align torch to that version."
    mkdir -p "$(dirname "$LIBTORCH")"
    unzip -q /tmp/libtorch.zip -d "$(dirname "$LIBTORCH")" && rm -f /tmp/libtorch.zip
    ok "installed in $LIBTORCH"
fi

# --- 4. C++ engine ------------------------------------------------------------
step "4/6  C++ engine"
cd "$ROOT/engine_c"
rm -rf build
# TORCH_CUDA_ARCH_LIST declared: without it, the LibTorch configuration tries to
# detect the GPU by compiling a small CUDA program that inherits the C++20
# standard of this project, asks for the "CUDA20" dialect and breaks on every
# CMake older than CUDA 12. There is not a single CUDA file to compile here: the
# detection is useless and it is enough to skip it.
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DWITH_LIBTORCH=ON \
      -DCMAKE_PREFIX_PATH="$LIBTORCH" -DTORCH_CUDA_ARCH_LIST="$CAP" > /tmp/cmake.log 2>&1 \
    || { tail -25 /tmp/cmake.log; fail "CMake configuration did not succeed (full log: /tmp/cmake.log)"; }
cmake --build build -j > /tmp/build.log 2>&1 \
    || { tail -25 /tmp/build.log; fail "build did not succeed (full log: /tmp/build.log)"; }
ok "built: $ROOT/engine_c/build/dama_engine"

# Can it load a network? The ENGINE IS ASKED. Looking at `ldd` does not tell
# "built without LibTorch" from "nonexistent binary" -- both give zero matches,
# and that ambiguity cost two rounds of diagnosis.
probe=$(./build/dama_engine --mode selfplay --n-games 1 --n-threads 1 \
        --n-sims 2 --weights /nonexistent_.pt 2>&1 | head -5)
case "$probe" in
    *"WITHOUT LibTorch"*) fail "the engine was built without LibTorch despite
      WITH_LIBTORCH=ON. Look at /tmp/cmake.log." ;;
    *) ok "the engine can load a network" ;;
esac

# --- 5. parity with Python ----------------------------------------------------
step "5/6  parity check"
cd "$ROOT"
"$PY" engine_c/tools/export_jit.py weights/champion.pt /tmp/model_jit.pt cuda > /tmp/jit.log 2>&1 \
    || { tail -15 /tmp/jit.log; fail "TorchScript export did not succeed"; }
./engine_c/build/parity_check /tmp/model_jit.pt cuda || fail "PARITY NOT MET: the C++ engine does not
      reproduce the numbers of PyTorch. Do NOT use it to produce results."
ok "the C++ engine plays like PyTorch"

# --- 6. validation suite ------------------------------------------------------
if [ "${DAMA_SKIP_PREFLIGHT:-0}" = "0" ]; then
    step "6/6  preflight (a few minutes)"
    chmod +x ./*.sh engine_c/*.sh 2>/dev/null
    # DAMA_DEVICE=cuda is NOT a detail: without it, the preflight benchmark runs
    # on the Python path on the CPU with the production network (96x6, 400
    # simulations), where a single cycle takes several minutes and the preflight
    # looks stuck. The GPU has just been checked at step 2: might as well use it.
    DAMA_DEVICE=cuda ./preflight.sh 2>&1 | tail -14
else
    step "6/6  preflight skipped (DAMA_SKIP_PREFLIGHT=1)"
fi

cat <<NEXT

=== ready ===
  engine : $ROOT/engine_c/build/dama_engine
  GPU    : capability $CAP, PyTorch $(cd "$ROOT" && "$PY" -c 'import torch;print(torch.__version__)')

Next steps:
  1) (optional) calibrate c_puct on a trained champion:
       cp /path/to/champion.pt cal/
       DAMA_WEIGHTS_DIR=cal DAMA_CPP_ENGINE=\$PWD/engine_c/build/dama_engine \\
       DAMA_DEVICE=cuda DAMA_CAL_GAMES=400 python calibrate.py
  2) start the run inside tmux:
       tmux new -s dama
       ./run.sh                 (or  DAMA_C_PUCT=<value> ./run.sh)
  3) follow it:
       ./watch.sh -f
NEXT
