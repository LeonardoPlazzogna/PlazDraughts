#!/usr/bin/env bash
# Dashboard of the run: reads weights/metrics.csv, weights/ladder.json and
# run.log and shows progress, trend and estimated end.
#
#   ./watch.sh              a snapshot
#   ./watch.sh -f           refreshes every 60 s
#   DAMA_WEIGHTS_DIR=... DAMA_CYCLES=... ./watch.sh    if they are not the defaults
#
# Dependencies: POSIX awk, date and the project's python interpreter.
# The tables are in PLAIN awk, without strftime/systime/bc: mawk -- the default
# awk on many Linux images -- does not have the first two, and bc is often not
# installed. A dashboard that collapses precisely on the machine that runs the
# training is of no use. Where dates must be read (the estimated end) python is
# used, which in this project is there by definition.
set -u

PY=$(command -v python3 || command -v python)
[ -n "$PY" ] || { echo "ERROR: no python interpreter found."; exit 1; }

DIR="${DAMA_WEIGHTS_DIR:-weights}"
TOT="${DAMA_CYCLES:-100}"
LOG="${DAMA_LOG:-run.log}"
CSV="$DIR/metrics.csv"

show() {
    printf '\n===== %s =====\n' "$(date '+%d/%m %H:%M:%S')"

    if [ ! -f "$CSV" ]; then
        echo "$CSV does not exist yet: the run has not completed its first cycle."
        [ -f "$LOG" ] && { echo; echo "--- tail of $LOG ---"; tail -n 15 "$LOG"; }
        return
    fi

    echo "--- latest cycles ---"
    awk -F, 'BEGIN{print " cycle  loss     p     v  arena pr  entr  qspr  vmae draw%   min"}
        $1=="cycle"{next}
        {printf "%6d %5.3f %5.3f %5.3f  %5.2f %2d %5.3f %5.3f %5.3f %5.2f %5.1f\n",
                 $1,$7,$8,$9,$10,$11,$23,$24,$25,$18,$3/60}' "$CSV" | tail -n 15

    # strength against fixed opponents: sparse rows (every DAMA_STRENGTH_EVERY cycles)
    if awk -F, '$1!="cycle"&&$27!=""{f=1} END{exit !f}' "$CSV"; then
        echo
        echo "--- strength against fixed opponents ---"
        awk -F, '$1!="cycle"&&$27!=""{printf "cycle %4d  elo_gen=%-8s %s\n",
                 $1,($26==""?"-":$26),$27}' "$CSV" | tail -n 6
    fi

    if [ -f "$DIR/ladder.json" ]; then
        echo
        echo "--- Elo across generations ---"
        "$PY" - "$DIR/ladder.json" <<'PY' 2>/dev/null || true
import json, sys
for e in json.load(open(sys.argv[1]))[-6:]:
    print(f"  gen {e['gen']:>3}  (cycle {e['cycle']:>4})  Elo {e['elo']:+8.1f}")
PY
    fi

    echo
    echo "--- progress ---"
    # The pace is measured from the TIMES of the rows, not from the duration
    # column. Two reasons, and the second is the one that matters:
    #
    #   1. the duration_s column long excluded the evaluation phase, so the
    #      estimates came out short. It is correct now, but the CSVs already
    #      written stay wrong;
    #   2. the times ALSO include the time that ends up in no measured phase --
    #      saving the resume buffer, I/O waits, the GPU shared with something
    #      else. It is the time that really passes, which is exactly what is
    #      needed to say when it ends.
    #
    # Mean over the last 20 intervals, and the window is not arbitrary.
    #
    # The strength measurements fire every DAMA_STRENGTH_EVERY cycles (20 by
    # default) and cost several times a normal cycle. A window of 10 contains
    # that cycle only half of the time, so the estimate swings between one
    # reading and the next with nothing changed. A window equal to the period
    # always contains exactly one.
    #
    # Nor is the mean from the start of the run good, which has the opposite
    # defect: the cycle gets longer as the network improves, because the games
    # get longer, and a global mean systematically underestimates the future.
    # Both are printed: if they diverge a lot, the run is slowing down.
    #
    # The other source of variation between cycles is the SPRT: a clearly better
    # or worse candidate closes the arena in a few games, an uncertain one
    # reaches the cap. That too is averaged over twenty cycles.
    eval "$("$PY" - "$CSV" "$TOT" "${DAMA_PACE_WINDOW:-20}" <<'PY'
import csv, sys
from datetime import datetime
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r.get("cycle", "").isdigit()]
tot = int(sys.argv[2])
if not rows:
    print("LAST=0 PCT=0 MIN=0 LEFT=0 HOURS=0 SEC=0 GLOB=0 NWIN=0"); raise SystemExit
last = int(rows[-1]["cycle"])
t = [datetime.fromisoformat(r["timestamp"]) for r in rows]
gaps = [(t[i] - t[i-1]).total_seconds() for i in range(1, len(t))]
window = int(sys.argv[3]) if len(sys.argv) > 3 else 20
gaps = [g for g in gaps if g > 0]
global_mean = (sum(gaps) / len(gaps)) if gaps else 0
gaps = gaps[-window:]
if gaps:
    m = sum(gaps) / len(gaps)
else:                       # a single row: fall back on the measured phases
    r = rows[-1]
    m = sum(float(r.get(k) or 0) for k in
            ("t_selfplay", "t_train", "t_arena", "t_eval")) or float(rows[-1]["duration_s"])
left = max(0, tot - last)
print(f"LAST={last} PCT={100*last//max(1,tot)} MIN={m/60:.1f} "
      f"LEFT={left} HOURS={left*m/3600:.1f} SEC={int(left*m)} "
      f"GLOB={global_mean/60:.1f} NWIN={len(gaps)}")
PY
)"
    printf "cycle %d/%s (%d%%)   %.1f min/cycle (last %s;  since the start %s)\n" \
           "$LAST" "$TOT" "$PCT" "$MIN" "$NWIN" "$GLOB"
    if [ "$LEFT" -gt 0 ]; then
        printf "%d cycles left = %s h   ->   estimated end ~ %s\n" \
               "$LEFT" "$HOURS" "$(date -d "+$SEC seconds" '+%d/%m %H:%M' 2>/dev/null || echo '?')"
    else
        echo "run completed."
    fi

    if [ -f "$LOG" ]; then
        echo
        echo "--- anomalies in the log ---"
        # no "|| echo 0": with no matches grep -c already prints 0 and exits with
        # 1, so the fallback would add a second 0 and the numeric comparison
        # below would break.
        n=$(grep -cE "failed|Traceback|Error|BrokenPipe|RESTART" "$LOG" 2>/dev/null)
        n=${n:-0}
        if [ "$n" -eq 0 ]; then
            echo "  none (0 fallbacks from the C++ engine, 0 exceptions, 0 restarts)"
        else
            echo "  $n suspicious lines, last 5:"
            grep -E "failed|Traceback|Error|BrokenPipe|RESTART" "$LOG" | tail -n 5 | sed 's/^/    /'
        fi
    fi
}

if [ "${1:-}" = "-f" ]; then
    while true; do clear; show; sleep "${DAMA_WATCH_EVERY:-60}"; done
else
    show
fi
