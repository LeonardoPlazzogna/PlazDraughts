#!/usr/bin/env bash
# Is a new pod usable? A verdict in fifteen seconds, before cloning or
# installing anything.
#
#   bash check_pod.sh
#
# IT COMES FROM EIGHT PODS IN A ROW, most of them unusable, each one found to be
# so after minutes of work. The message that comes back from PyTorch -- "CUDA
# unknown error" -- is the same for completely different causes, and it leads
# the diagnosis astray more than once.
#
# THE MOVE THAT CHANGES EVERYTHING: the CUDA library is asked for the CODE
# instead of reading PyTorch's message. cuInit() returns a number that names the
# cause, and every cause has a different answer -- some can be worked around from
# inside, others cannot. Without that number one guesses, and guessing wasted
# an hour.
set -u

PY=$(command -v python3 || command -v python)
[ -n "$PY" ] || { echo "no python interpreter"; exit 1; }

echo "=== device nodes ==="
ls -d /dev/nvidia* 2>/dev/null || echo "  NONE: the container has no GPU"

echo
echo "=== host driver and library in the container ==="
# A mismatch between the two is one of the causes of error, and it is not
# visible anywhere else: nvidia-smi shows the driver of the HOST, while the
# computation goes through the library injected INTO THE CONTAINER. If the
# runtime has not aligned them, cuInit breaks with a specific code (803).
cat /proc/driver/nvidia/version 2>/dev/null | head -1 || echo "  /proc/driver/nvidia absent"
ls -l /usr/lib/x86_64-linux-gnu/libcuda.so.* 2>/dev/null | head -3 || echo "  libcuda not found in the usual path"

echo
echo "=== enumeration used by CUDA ==="
# CUDA does not scan /dev in order: it enumerates from here and then opens the
# node with the number it finds. It is why a node called nvidia6 is not a
# problem in itself -- if this folder is populated consistently.
ls /proc/driver/nvidia/gpus/ 2>/dev/null || echo "  no GPU listed: that is the problem"

echo
echo "=== return code of cuInit ==="
"$PY" - <<'PY'
import ctypes
NAMES = {
    0:   ("SUCCESS", "all good"),
    100: ("NO_DEVICE", "the container does not see any GPU: change pod"),
    101: ("INVALID_DEVICE", "invalid device index: look at CUDA_VISIBLE_DEVICES"),
    200: ("INVALID_IMAGE", "corrupted library"),
    802: ("SYSTEM_NOT_READY", "driver starting or updating on the host: RETRY in a minute"),
    803: ("SYSTEM_DRIVER_MISMATCH", "the library in the container does not match the host driver: NOT fixable from inside, change template or pod"),
    804: ("COMPAT_NOT_SUPPORTED", "driver too old for this GPU"),
    999: ("UNKNOWN", "cause not declared by the driver: usually a dirty container state, try RESTARTING the pod once"),
}
try:
    lib = ctypes.CDLL("libcuda.so.1")
except OSError as e:
    print(f"  libcuda cannot be loaded: {e}")
    raise SystemExit
r = lib.cuInit(0)
name, meaning = NAMES.get(r, ("?", "code not catalogued"))
print(f"  cuInit -> {r}  {name}")
print(f"  {meaning}")
if r == 0:
    n = ctypes.c_int()
    lib.cuDeviceGetCount(ctypes.byref(n))
    print(f"  devices seen by CUDA: {n.value}")
PY

echo
echo "=== cores really available ==="
# The code must be written to a FILE, not passed on standard input: when
# multiprocessing RE-CREATES the children instead of forking them, it re-imports
# the main module -- and if that is standard input the process hangs forever. On
# Linux pods forking is the default behavior and the trouble would never show,
# but a check that hangs on the wrong machine is a check nobody uses. Verified:
# from stdin it hangs, from a file it does not.
cat > /tmp/_core.py <<'PYCORE'
import multiprocessing as mp, time, os
# How many processes to launch. They must be kept ABOVE the expected cores,
# otherwise the number of processes is measured instead of the available CPU. On
# Windows the limit is 63 handles per wait and beyond it one gets an exception
# instead of a measurement -- irrelevant on pods, but the check must be able to
# run on Windows too, otherwise it is never tried before use.
COUNT = 60 if os.name == "nt" else 96
def burn(_):
    t = time.time(); n = 0
    while time.time() - t < 1.5:
        n += sum(i * i for i in range(2000))
    return n
if __name__ == "__main__":
    one = burn(0)
    with mp.Pool(COUNT) as p:
        tot = sum(p.map(burn, range(COUNT)))
    real = tot / one
    print(f"  real cores: {real:.0f}   (nproc declares {os.cpu_count()})")
    print(f"  suggested:  DAMA_CPP_THREADS={max(1, int(real))}")
    if real < 35:
        print("  FEW: the cycles will be slow in proportion")
PYCORE
"$PY" /tmp/_core.py

echo
echo "=== verdict ==="
echo "  Keep the pod if cuInit gave 0 and the real cores are at least 35."
echo "  With 802 or 999 ONE restart of the pod from the panel is worth it, then change."
echo "  With 803 or 100 change at once: it cannot be fixed from inside."
