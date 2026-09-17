"""How many cores can REALLY be used, not how many the system declares.

    python usable_cores.py
    python usable_cores.py --repeats 5

It is needed because on rented machines the two numbers do not coincide: the
machine is shared and a quota limits how much each guest gets, so the effective
cores can be a fraction of the declared ones. Sizing the engine threads on the
declared number means asking for more than there are.

TWO ROUTES, and the first is much better than the second.

1. THE QUOTA DECLARED BY THE SYSTEM (Linux only). The limit imposed on the
   container is read from a file: it is exact, immediate and unaffected by the
   load. When it exists, it is the answer.

2. THE SATURATION TEST (everywhere). The machine is filled with processes and
   the work done is compared with that of a single process. It is the only
   route when there is no quota, but it is NOISY: it measures how free the
   machine is right now, not how much it could give you, and two consecutive
   measurements on an idle machine can differ by nearly a factor of two. That is
   why it is repeated and the MAXIMUM is kept -- the repetition least disturbed
   by other people's load -- and the spread is reported instead of hidden.

THE BARRIER. The processes must start TOGETHER: on Windows each one is created
from scratch and starting costs time, so without a barrier the first ones work
while the last ones do not exist yet, find little competition and inflate the
total -- to the point of reporting more usable cores than the machine declares,
which is absurd on its face.

WHY IT LIVES IN A SEPARATE FILE, and not inside benchmark.py where it is used:
on Windows the child processes re-import the main module. If that module imports
torch, dozens of children import it dozens of times and the measurement never
ends. Verified: inside the benchmark it hung, here it takes a few seconds.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time

DURATION = 1.0
WAIT_MAX = 30.0
# How long ONE measurement round is allowed before it is declared stuck, and how
# long the whole measurement. They are needed because a round may never finish:
# a pool of 32 processes was observed hanging for over twenty hours, the parent
# stuck inside map() and the children idle. Without a limit, whoever runs the
# preflight sees the command stuck and no message.
ROUND_MAX = WAIT_MAX + DURATION + 60.0
TOTAL_MAX = 420.0

_ready = None
_count = 0


def declared_quota() -> float | None:
    """Cores granted by the system, if the system declares them (Linux cgroup).

    Exact and instantaneous: when it exists, it is worth more than any
    measurement."""
    try:                                    # cgroup v2
        with open("/sys/fs/cgroup/cpu.max") as f:
            q, p = f.read().split()
            if q != "max":
                return int(q) / int(p)
    except Exception:
        pass
    try:                                    # cgroup v1
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            q = int(f.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            p = int(f.read().strip())
        if q > 0 and p > 0:
            return q / p
    except Exception:
        pass
    return None


def _init(ready, count):
    global _ready, _count
    _ready, _count = ready, count


def burn(_):
    """Checks in at the barrier, waits for the others, then works for DURATION."""
    with _ready.get_lock():
        _ready.value += 1
    limit = time.time() + WAIT_MAX
    while _ready.value < _count and time.time() < limit:
        time.sleep(0.005)
    t0 = time.time()
    n = 0
    while time.time() - t0 < DURATION:
        n += sum(i * i for i in range(2000))
    return n


def _solo_rate() -> float:
    t0 = time.time()
    n = 0
    while time.time() - t0 < DURATION:
        n += sum(i * i for i in range(2000))
    return n / (time.time() - t0)


def by_saturation(repeats: int = 3) -> tuple[list[float], int]:
    """Returns the results and how many processes could really be used.

    The number of processes ADAPTS. Asking for too many, Windows refuses with
    "the paging file is too small" (error 1455): every new process commits
    memory, and if the parent has already loaded torch the limit comes early. It
    happened when this measurement was called from the benchmark, where the
    parent has torch in memory: on its own it worked, from there it did not.

    Measuring with fewer processes is better than not measuring: it is enough
    that they are more than the expected cores, otherwise the processes would be
    counted instead of the CPU.
    """
    declared = os.cpu_count() or 1
    cap = 60 if os.name == "nt" else 96
    n_procs = max(8, min(cap, 2 * declared))
    ctx = mp.get_context("spawn")
    results = []
    deadline = time.time() + TOTAL_MAX
    while n_procs >= 4:
        try:
            for _ in range(repeats):
                if time.time() > deadline:
                    print(f"  (measurement stopped after {TOTAL_MAX:.0f}s: "
                          f"keeping the {len(results)} rounds already done)")
                    return results, n_procs
                rate = _solo_rate()
                ready = ctx.Value("i", 0)
                with ctx.Pool(n_procs, initializer=_init,
                              initargs=(ready, n_procs)) as p:
                    # map_async().get(timeout) and not map(): map waits forever,
                    # and if the pool breaks -- which on Windows happens when
                    # dozens of processes are requested -- it hangs, and so do
                    # the children. With the deadline, leaving the with block
                    # calls terminate() and closes them.
                    tot = sum(p.map_async(burn, range(n_procs))
                               .get(timeout=ROUND_MAX))
                results.append((tot / DURATION) / rate if rate else 0.0)
            return results, n_procs
        except (OSError, mp.TimeoutError) as e:
            results.clear()
            n_procs //= 2
            print(f"  ({type(e).__name__}: retrying with {n_procs} processes)")
    return results, n_procs


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--quota-only", action="store_true",
                    help="only read the system quota, without measuring")
    args = ap.parse_args()

    declared = os.cpu_count() or 1
    print(f"cores declared by the system : {declared}")

    quota = declared_quota()
    if quota is not None:
        print(f"container quota              : {quota:.1f}   (EXACT figure)")
        print(f"suggested                    : DAMA_CPP_THREADS={max(1, int(quota))}")
        if quota < 0.7 * declared:
            print(f"  only {100*quota/declared:.0f}% of the declared cores "
                  f"can be used")
        return
    print("container quota              : not declared "
          "(normal outside Linux containers)")

    if args.quota_only:
        return
    results, used = by_saturation(args.repeats)
    if not results:
        print("saturation test              : FAILED (not enough memory to start")
        print("  enough processes). The declared number still stands.")
        return
    best = max(results)
    print(f"saturation test              : "
          f"{', '.join(f'{e:.1f}' for e in results)}  ->  {best:.1f}"
          f"   ({used} processes)")
    if len(results) > 1 and min(results) < 0.75 * best:
        print("  SCATTERED MEASUREMENTS: the machine is doing something else. The")
        print("  highest value is the least disturbed, but repeat the test on an idle machine.")
    print(f"suggested                    : DAMA_CPP_THREADS={max(1, int(best))}")


if __name__ == "__main__":
    main()
