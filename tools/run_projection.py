"""How long until the end, read from the cycles already done.

    python tools/run_projection.py weights
    python tools/run_projection.py weights --cycles 100

It is the most precise way to estimate the duration, because it does not measure
a surrogate: it reads the times the run is really taking. And it costs nothing,
because the cycles it relies on are real cycles of the run, not a test thrown
away.

WHY TAKING THE MEAN AND MULTIPLYING IS NOT ENOUGH. The first cycles are
structurally the cheapest of the whole run, for three reasons acting together:

  1. the replay window fills up in about ten cycles, so training at the start
     sees far fewer samples;
  2. games get longer as the network learns (~84 plies measured with a random
     network, 150-159 with a trained one), and the cost of a game is
     proportional to its length;
  3. the PERIODIC phases have not run yet: the strength test starts at cycle 20,
     the generation ladder at 10, the deep anchor at 60.

HOW IT CORRECTS. The continuous phases (self-play, arena, training) are
estimated from the STEADY STATE -- the last cycles, not the mean from the start
-- and the periodic phases are amortized over their real cadence. What has never
been observed yet is declared as such instead of being silently left out.

HOW PRECISE IT IS. Below cycle 20 this tool does no better than the naive mean,
because the strength test has not run yet and nobody can guess its cost -- but
at least it SAYS so, instead of returning a low number that looks complete. From
cycle 20 on, once every periodic phase has been seen at least once, the error
shrinks and tends to err on the HIGH side, which for sizing a run is the right
direction to err in.

So: if a reliable estimate is needed, the mini-run must reach at least cycle 20.
And that is not time lost, because those cycles are cycles of the real run.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics as st


def _duration(s: float) -> str:
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.1f} min"
    if s < 3 * 86400:
        return f"{s / 3600:.1f} h"
    return f"{s / 86400:.1f} days"


def _num(r, k):
    try:
        return float(r.get(k) or 0)
    except (TypeError, ValueError):
        return 0.0


def read_run(folder: str) -> list[dict]:
    path = os.path.join(folder, "metrics.csv")
    if not os.path.exists(path):
        raise SystemExit(f"{path} does not exist: the run has not finished a cycle yet.")
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("cycle") or "").isdigit()]
    if not rows:
        raise SystemExit(f"{path} has no cycle rows.")
    return rows


def project(args) -> None:
    rows = read_run(args.folder)
    done = max(int(r["cycle"]) for r in rows)
    by_cycle = {int(r["cycle"]): r for r in rows}

    print("=" * 70)
    print(f"  PROJECTION FROM {done} CYCLES ALREADY DONE")
    print("=" * 70)

    # --- periodic phases: recognized by the columns that announce them --------
    # They must be separated BEFORE computing the steady state, otherwise the
    # cycles in which they run inflate the mean of the normal ones and the
    # cadence gets counted twice.
    strength_cycles = [c for c, r in by_cycle.items() if (r.get("strength") or "").strip()]
    ladder_cycles = [c for c, r in by_cycle.items() if (r.get("elo") or "").strip()]
    periodic = set(strength_cycles) | set(ladder_cycles)

    normal = [c for c in by_cycle if c not in periodic]
    if not normal:
        raise SystemExit("  every cycle contains a periodic phase: more cycles are needed.")

    # --- steady state: the last normal cycles, not the mean from the start ----
    window = [c for c in sorted(normal)[-args.window:]]
    def mean(cycles, key):
        return st.mean([_num(by_cycle[c], key) for c in cycles]) if cycles else 0.0

    base = mean(window, "duration_s")
    print(f"  normal cycle, steady state : {_duration(base)}  "
          f"(mean over the last {len(window)} cycles without periodic phases)")
    from_start = mean(sorted(normal), "duration_s")
    if base > 0 and from_start > 0:
        print(f"  mean from the start of run : {_duration(from_start)}  "
              f"({100*(base/from_start - 1):+.0f}% compared with the steady state)")
    print()

    print(f"  {'phase':<22}{'steady':>12}{'share':>8}")
    for label, key in (("self-play", "t_selfplay"), ("training", "t_train"),
                       ("arena", "t_arena"), ("evaluation", "t_eval")):
        v = mean(window, key)
        print(f"  {label:<22}{_duration(v):>12}{100*v/max(base,1e-9):>7.0f}%")
    print()

    # --- cost of the periodic phases, amortized --------------------------------
    #
    # THE CADENCES OVERLAP, and ignoring it counts twice. The generation ladder
    # runs every 10 cycles and the strength test every 20: so EVERY cycle with
    # the strength test also runs the ladder. Taking the surplus of both over a
    # normal cycle, the cost of the ladder ends up in both terms.
    #
    # Counted that way, the ladder is paid for twice and the projection comes out
    # inflated by hours over a hundred cycles.
    #
    # So they are separated like this: the ladder is measured on the cycles where
    # it runs ALONE, and the ladder it carries is subtracted from the strength test.
    ladder_only = sorted(set(ladder_cycles) - set(strength_cycles))
    extra = 0.0
    ladder_cost = 0.0
    print("  periodic phases")

    if ladder_cycles:
        if ladder_only:
            ladder_cost = max(0.0, mean(ladder_only, "duration_s") - base)
            note = f"over {len(ladder_only)} cycles where it runs alone"
        else:
            # No cycle with the ladder alone: it cannot be separated from the
            # strength test, and saying so is better than inventing a split.
            ladder_cost = max(0.0, mean(ladder_cycles, "duration_s") - base)
            note = "NOT separable from the strength test: coinciding cadences"
        amortized = ladder_cost / max(1, args.ladder_every)
        extra += amortized
        print(f"    {'generation ladder':<24} {_duration(ladder_cost):>10} "
              f"every {args.ladder_every:>2} cycles -> {_duration(amortized)}/cycle")
        print(f"      ({note})")
    else:
        print(f"    {'generation ladder':<24} NEVER SEEN YET "
              f"(runs every {args.ladder_every} cycles)")

    if strength_cycles:
        if not ladder_only and ladder_cycles:
            # Coinciding cadences: the cost of both phases has already been
            # counted in full in the row above, because the cycles are the same.
            # Adding it again here would count it twice -- exactly the mistake
            # this function exists to avoid.
            print(f"    {'strength test':<24} "
                  "already included in the row above (same cycles)")
        else:
            gross = max(0.0, mean(strength_cycles, "duration_s") - base)
            # Net of the ladder, which ran in those cycles too.
            net = max(0.0, gross - ladder_cost)
            amortized = net / max(1, args.strength_every)
            extra += amortized
            print(f"    {'strength test':<24} {_duration(net):>10} "
                  f"every {args.strength_every:>2} cycles -> {_duration(amortized)}/cycle")
            if gross > net:
                print(f"      (gross {_duration(gross)}, minus {_duration(ladder_cost)} "
                      "of ladder that runs in the same cycles)")
    else:
        print(f"    {'strength test':<24} NEVER SEEN YET "
              f"(runs every {args.strength_every} cycles)")

    unseen = []
    if not strength_cycles:
        unseen.append(f"strength test (cycle {args.strength_every})")
    if args.deep_every > 0 and done < args.deep_every:
        unseen.append(f"deep anchor (cycle {args.deep_every})")
    print()

    mean_cycle = base + extra
    print(f"  ESTIMATED MEAN CYCLE       : {_duration(mean_cycle)}")
    print()

    remaining = max(0, args.cycles - done)
    spent = sum(_num(r, "duration_s") for r in rows)
    missing = remaining * mean_cycle
    print("=" * 70)
    print(f"  COMPLETE RUN ({args.cycles} cycles)")
    print("=" * 70)
    print(f"  already spent              : {_duration(spent)}  ({done} cycles)")
    print(f"  remaining                  : {_duration(missing)}  ({remaining} cycles)")
    print(f"  TOTAL                      : {_duration(spent + missing)}")
    print()

    if unseen:
        print("  WARNING: these phases have never run yet, so their cost is")
        print("  NOT included in the estimate:")
        for v in unseen:
            print(f"    - {v}")
        print("  So the estimate is TOO LOW. Run this tool again after")
        print(f"  cycle {max(args.strength_every, args.deep_every if args.deep_every>0 else 0)}"
              " to have them all.")
        print()

    if done < args.min_cycles:
        print(f"  With only {done} cycles the estimate stays FRAGILE AND TOO LOW. In")
        print("  the first cycles games are short, the replay window is not full yet")
        print("  and not all the periodic phases have run.")
        print(f"  Measured on a real run: stopping at cycle 10 underestimates by")
        print(f"  28%, stopping at cycle 20 overestimates by 4%.")
        print(f"  Read this estimate again after cycle {args.min_cycles}.")


def main():
    ap = argparse.ArgumentParser(
        description="Estimates the duration of the run from the cycles already completed.")
    ap.add_argument("folder", help="weights folder (contains metrics.csv)")
    ap.add_argument("--cycles", type=int, default=100, help="cycles planned in total")
    ap.add_argument("--window", type=int, default=10,
                    help="how many recent cycles to use for the steady state")
    ap.add_argument("--ladder-every", type=int, default=10)
    ap.add_argument("--strength-every", type=int, default=20)
    ap.add_argument("--deep-every", type=int, default=60)
    ap.add_argument("--min-cycles", type=int, default=20,
                    help="below this number the estimate is declared fragile")
    args = ap.parse_args()
    project(args)


if __name__ == "__main__":
    main()
