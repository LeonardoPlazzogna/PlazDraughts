"""
Generates the search fixture the browser tests are held to.

    python web/test/fixtures/make_fixtures.py

It runs the PYTHON search (mcts.py) with the uniform evaluator on a handful of
positions and writes, for each one, the visit count and value of every root
move. The browser port must reproduce those numbers exactly.

Why exactly, and why it is possible: with a uniform evaluator and no Dirichlet
noise nothing in the search is random. The tree it builds is a deterministic
function of the rules, the PUCT formula, the leaf batching and the virtual
loss. So any difference in the counts means a difference in one of those --
which is precisely what a port is at risk of getting subtly wrong, and what no
test on the rules alone would catch.

The fixture is regenerated only if the search changes on purpose; when that
happens, the browser numbers have to move with it.
"""
from __future__ import annotations
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from dama import Position, WHITE, BLACK, W_MAN, W_KING, B_MAN, B_KING, sq   # noqa: E402
from evaluators import UniformEvaluator                                      # noqa: E402
from mcts import MCTS                                                        # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "search.json")


def build(pieces: dict, turn: int = WHITE, no_progress: int = 0) -> Position:
    b = [0] * 64
    for (r, c), p in pieces.items():
        b[sq(r, c)] = p
    return Position(b, turn, no_progress)


def playout(n_plies: int) -> Position:
    """A position reached by playing the lowest move (frm, then to), as the
    encoder parity check does: no seed involved, and it is a real position."""
    pos = Position()
    for _ in range(n_plies):
        if pos.is_terminal():
            break
        pos = pos.play(min(pos.legal_moves(), key=lambda m: (m.frm, m.to)))
    return pos


CASES = [
    ("opening", Position(), 128),
    ("after 12 plies", playout(12), 256),
    # Black to move, with a capture available: captures are mandatory, so this
    # also pins the priority rules inside the search.
    ("forced capture", build({(5, 2): W_MAN, (4, 3): B_MAN, (7, 0): W_MAN}, BLACK), 64),
    # A blocked king and a man: the side to move loses if it walks into the trap.
    ("endgame, king against man",
     build({(4, 3): W_KING, (0, 1): B_MAN, (1, 0): W_MAN, (2, 3): W_MAN}), 240),
]


def main():
    out = []
    for label, pos, sims in CASES:
        root = MCTS(UniformEvaluator(), n_sims=sims, batch_size=8).run(pos, add_noise=False)
        children = sorted(
            ({"frm": mv.frm, "to": mv.to, "N": st.N, "Q": round(st.Q, 9)}
             for mv, st in root.children.items()),
            key=lambda d: (-d["N"], d["frm"], d["to"]))
        out.append({
            "label": label,
            "board": list(pos.board),
            "turn": pos.turn,
            "no_progress": pos.no_progress,
            "sims": sims,
            "batch_size": 8,
            "children": children,
        })
        print(f"  {label:<26} {len(children)} root moves, {sims} simulations")
    # One case per line: a board written one number per line would bury four
    # short results under six hundred lines of noise.
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("[\n")
        fh.write(",\n".join(json.dumps(c, separators=(",", ":")) for c in out))
        fh.write("\n]\n")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
