"""
Per-cycle metrics, written to CSV.

Whatever is only printed to the console is lost unless the output is
redirected, and even then it is text to be parsed again. Nothing can be plotted
that was not saved, and repeating a run of several hours to recover a number is
expensive.

One row per cycle is appended to `metrics.csv` in the weights folder, with the
header written only the first time: the file survives interruptions and
resumes like the rest of the state.

The schema is FIXED: adding columns later invalidates the existing files (a
file with a different header is archived, see MetricsCsv), so the diagnostic
columns (data quality, entropy, calibration, per-phase times) were planned from
the start, even though not all of them are available on every path -- in that
case they stay empty.
"""
from __future__ import annotations
import csv
import os
from dataclasses import dataclass, asdict, fields
from datetime import datetime, timezone


@dataclass
class GameStats:
    """Aggregate statistics over the games of ONE self-play cycle.

    They diagnose the QUALITY of the data, not just its quantity: if most games
    end in a no-progress draw, the value head gets very little signal, and
    without these numbers the symptom cannot be told apart from a training
    problem.
    """
    games: int = 0
    plies_mean: float = 0.0
    plies_max: int = 0
    white_wins: int = 0
    black_wins: int = 0
    draws: int = 0
    truncated: int = 0          # max_plies reached without an end
    no_progress: int = 0        # draw by the no-progress counter
    captures_mean: float = 0.0
    promotions_mean: float = 0.0
    entropy_mean: float = 0.0   # mean entropy of the visit distribution
    q_spread_mean: float = 0.0  # how much the search tells the root moves apart
    value_mae: float = 0.0      # |search value - outcome|, mean

    @property
    def draw_rate(self) -> float:
        return self.draws / self.games if self.games else 0.0


def aggregate(per_game: list[dict]) -> GameStats:
    """From a list of per-game dicts to the aggregate statistics."""
    st = GameStats()
    if not per_game:
        return st
    n = len(per_game)
    st.games = n
    st.plies_mean = sum(g["plies"] for g in per_game) / n
    st.plies_max = max(g["plies"] for g in per_game)
    st.white_wins = sum(1 for g in per_game if g["z_white"] > 0)
    st.black_wins = sum(1 for g in per_game if g["z_white"] < 0)
    st.draws = sum(1 for g in per_game if g["z_white"] == 0)
    st.truncated = sum(1 for g in per_game if g.get("truncated"))
    st.no_progress = sum(1 for g in per_game if g.get("no_progress"))
    st.captures_mean = sum(g.get("captures", 0) for g in per_game) / n
    st.promotions_mean = sum(g.get("promotions", 0) for g in per_game) / n
    ent = [g["entropy"] for g in per_game if g.get("entropy") is not None]
    st.entropy_mean = sum(ent) / len(ent) if ent else 0.0
    qs = [g["q_spread"] for g in per_game if g.get("q_spread") is not None]
    st.q_spread_mean = sum(qs) / len(qs) if qs else 0.0
    mae = [g["value_mae"] for g in per_game if g.get("value_mae") is not None]
    st.value_mae = sum(mae) / len(mae) if mae else 0.0
    return st


@dataclass
class CycleRow:
    """One row of metrics.csv. The field order is the column order."""
    cycle: int = 0
    timestamp: str = ""
    # TOTAL cycle time, evaluation included: it equals
    # t_selfplay + t_train + t_arena + t_eval. End-of-run estimates are built
    # on it, so it must include everything -- leaving the evaluation out lost
    # 28% of the real time.
    duration_s: float = 0.0
    # generated data
    samples: int = 0
    buffer: int = 0
    sims: int = 0               # self-play simulations per move in this cycle
    # training
    loss: float = 0.0
    loss_policy: float = 0.0
    loss_value: float = 0.0
    # arena and promotion
    arena_score: float = 0.0
    promoted: int = 0
    # quality of the self-play games
    games: int = 0
    plies_mean: float = 0.0
    plies_max: int = 0
    white_wins: int = 0
    black_wins: int = 0
    draws: int = 0
    draw_rate: float = 0.0
    truncated: int = 0
    no_progress: int = 0
    captures_mean: float = 0.0
    promotions_mean: float = 0.0
    # learning diagnostics
    entropy_mean: float = 0.0
    q_spread_mean: float = 0.0
    value_mae: float = 0.0
    # strength measurements (empty in the cycles where they are not taken)
    elo: str = ""
    strength: str = ""          # "random=1.00;greedy=0.85;ab2=0.60"
    # per-phase times
    t_selfplay: float = 0.0
    t_train: float = 0.0
    t_arena: float = 0.0
    t_eval: float = 0.0


class MetricsCsv:
    def __init__(self, weights_dir: str, filename: str = "metrics.csv"):
        self.path = os.path.join(weights_dir, filename)
        self._header = [f.name for f in fields(CycleRow)]
        os.makedirs(weights_dir, exist_ok=True)

        if os.path.exists(self.path):
            # If the schema changed (columns added by a newer version of the
            # code) appending is NOT possible: the rows would have more fields
            # than the header and the file would silently become unreadable.
            # The old file is archived with a suffix and a clean one is started.
            try:
                with open(self.path, newline="", encoding="utf-8") as f:
                    old = next(csv.reader(f), [])
            except OSError:
                old = []
            if old and old != self._header:
                ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                archived = f"{self.path}.{ts}.bak"
                os.replace(self.path, archived)
                print(f"[metrics] CSV schema changed: old file "
                      f"archived as {os.path.basename(archived)}")
            else:
                return          # compatible schema: just append

        with open(self.path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(self._header)

    def append(self, row: CycleRow) -> None:
        d = asdict(row)
        d["timestamp"] = d["timestamp"] or datetime.now(timezone.utc).isoformat(
            timespec="seconds")
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([d[k] for k in self._header])


def entropy_of(pi) -> float:
    """Entropy (in nats) of a probability distribution.

    On the visit policy it measures how DECISIVE the search is: high = visits
    spread over many moves, low = concentrated on a few. Expected to fall as
    the network learns to propose good candidates."""
    import math
    h = 0.0
    for p in pi:
        if p > 0.0:
            h -= float(p) * math.log(float(p))
    return h
