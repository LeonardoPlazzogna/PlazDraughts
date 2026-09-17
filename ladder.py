"""
GENERATIONAL strength meter: relative Elo between champions over time.

Why it is needed. The fixed opponents (players.py: random, greedy) are weak: as
soon as the champion beats them consistently the meter SATURATES and from then
on returns 1.00 forever, unable to tell a network that keeps improving from one
that is stuck or getting worse. The arena does not fill the gap: it compares the
candidate with the CURRENT champion, so it says whether a single step is an
improvement, not how much progress has been made since the start.

Here a snapshot of the champion is kept every N cycles (a "generation") and the
current champion plays against the previous generations. The result is an Elo
curve that never saturates, because the yardstick grows with the player.

The Elo is RELATIVE: generation 0 is worth 0 by definition. It is not comparable
with human Elo or with other programs -- it only measures the internal progress
of this run.
"""
from __future__ import annotations
import json
import math
import os

# Scores of 0.0 and 1.0 would give infinite Elo differences. The score is
# clamped to [1-CAP, CAP]: a crushing win becomes a LOWER BOUND on the strength
# difference, not an exact value. With CAP=0.99 the ceiling for a single
# comparison is ~+800 Elo.
SCORE_CAP = 0.99


def elo_diff(score: float, cap: float = SCORE_CAP) -> float:
    """Elo difference implied by a score in [0,1] (0.5 -> 0)."""
    s = min(max(score, 1.0 - cap), cap)
    return -400.0 * math.log10(1.0 / s - 1.0)


def expected_score(diff: float) -> float:
    """Inverse of elo_diff: the expected score for a given Elo difference."""
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


class Ladder:
    """List of generations with their Elo, persisted to disk.

    Each entry: {"gen": int, "cycle": int, "file": str, "elo": float}
    """

    def __init__(self, weights_dir: str, filename: str = "ladder.json"):
        self.dir = weights_dir
        self.path = os.path.join(weights_dir, filename)
        self.entries: list[dict] = []
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.entries = json.load(f)
            except Exception as e:
                print(f"[ladder] cannot read {self.path} ({e}); starting empty")
                self.entries = []

    # --- persistence ------------------------------------------------------
    def _save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, indent=1)
        os.replace(tmp, self.path)      # atomic

    # --- queries ----------------------------------------------------------
    def is_empty(self) -> bool:
        return not self.entries

    def latest_elo(self) -> float:
        return self.entries[-1]["elo"] if self.entries else 0.0

    def anchors(self, k: int) -> list[dict]:
        """The last `k` generations, most recent first. They are the most
        informative anchors: those close in strength give scores far from the
        extremes, hence more precise Elo estimates."""
        return list(reversed(self.entries[-k:])) if self.entries else []

    def path_of(self, entry: dict) -> str:
        return os.path.join(self.dir, entry["file"])

    # --- updates ----------------------------------------------------------
    def add(self, cycle: int, elo: float, save_net_fn) -> dict:
        """Records a new generation. `save_net_fn(path)` must write the weights
        to the given path."""
        gen = len(self.entries)
        fname = f"gen_{gen:03d}.pt"
        save_net_fn(os.path.join(self.dir, fname))
        entry = {"gen": gen, "cycle": cycle, "file": fname, "elo": float(elo)}
        self.entries.append(entry)
        self._save()
        return entry

    def estimate_elo(self, scores: list[tuple[dict, float]]) -> tuple[float, list[str]]:
        """Estimates the current champion's Elo from its scores against the anchors.

        `scores` is a list of (anchor_entry, champion_score). Each anchor gives
        an independent estimate (anchor Elo + the difference implied by the
        score); the mean is returned. Anchors beaten, or lost to, decisively
        are clamped by SCORE_CAP and so contribute a bound, not an exact value:
        this is flagged in the text details.
        """
        if not scores:
            return 0.0, []
        estimates, details = [], []
        for entry, sc in scores:
            est = entry["elo"] + elo_diff(sc)
            estimates.append(est)
            saturated = "" if (1.0 - SCORE_CAP) < sc < SCORE_CAP else " (saturated)"
            details.append(f"gen{entry['gen']}({entry['elo']:+.0f})={sc:.2f}{saturated}")
        return sum(estimates) / len(estimates), details
