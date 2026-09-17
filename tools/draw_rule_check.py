"""How far does the implemented draw rule diverge from the official FID one?

IMPLEMENTED RULE: draw after 80 consecutive plies without a capture or a
promotion, automatically.

OFFICIAL RULE (FID, art. 10):
  10.1 at least one king PER SIDE is required;
  10.2 it applies on request of a player or when imposed by the referee;
  10.3-4 40 KING moves of the counted player are counted;
  10.5 the count is RESET if either player moves a MAN or if the number of
       pieces on the board changes.

The difference that matters: the official rule resets at every man move, the
implemented one does not. Here games are played and both counts are kept in
parallel.

    python tools/draw_rule_check.py [WEIGHTS [GAMES [SIMS]]]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from dama import Position, NO_PROGRESS_DRAW
from dama.board import is_king
from evaluators import NetEvaluator
from model import PolicyValueNet
from mcts import MCTS

import argparse

_ap = argparse.ArgumentParser(
    description="How far the implemented draw rule diverges from the FID one.")
_ap.add_argument("weights", nargs="?", default=None,
                 help="trained champion; without it, games are played with a "
                      "uniform evaluator and are not representative")
_ap.add_argument("games", nargs="?", type=int, default=12)
_ap.add_argument("sims", nargs="?", type=int, default=60)
_args = _ap.parse_args()
WEIGHTS, GAMES, SIMS = _args.weights, _args.games, _args.sims

if WEIGHTS:
    net = PolicyValueNet(channels=96, n_blocks=6)
    net.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    net.eval()
    ev = NetEvaluator(net, "cpu")
else:
    from evaluators import UniformEvaluator
    ev = UniformEvaluator()

torch.set_num_threads(1)

def kings_per_side(b):
    white = sum(1 for p in b if p > 0 and is_king(p))
    black = sum(1 for p in b if p < 0 and is_king(p))
    return white, black

n_impl, n_off, n_other = 0, 0, 0
details = []
for g in range(GAMES):
    pos = Position()
    rng = np.random.default_rng(1000 + g)
    # official count: KING moves of the counted player, reset by any man move
    # or change in the number of pieces
    off = 0
    counted = None
    off_triggered = False
    plies = 0
    while not pos.is_terminal() and plies < 300:
        before = pos.board
        n_before = sum(1 for p in before if p != 0)
        # The first plies are sampled from the visits, with noise at the root:
        # without it a deterministic network always plays THE SAME game and ten
        # repetitions are not ten observations (see defect 5.1 in
        # docs/results.md).
        root = MCTS(ev, n_sims=SIMS, batch_size=8, rng=rng).run(
            pos, add_noise=(plies < 12))
        children = list(root.children.items())
        if plies < 12:
            weight = np.array([st.N for _, st in children], dtype=np.float64)
            weight = weight / weight.sum() if weight.sum() > 0 else None
            move = children[rng.choice(len(children), p=weight)][0]
        else:
            move = max(children, key=lambda kv: kv[1].N)[0]
        was_king = is_king(before[move.frm])
        turn = pos.turn
        pos = pos.play(move)
        plies += 1
        n_after = sum(1 for p in pos.board if p != 0)

        wk, bk = kings_per_side(pos.board)
        if wk == 0 or bk == 0:
            off, counted = 0, None              # 10.1: not applicable
        elif (not was_king) or n_after != n_before:
            off, counted = 0, None              # 10.5: count reset
        else:
            if counted is None:
                counted = turn
            if turn == counted:
                off += 1
                if off >= 40 and not off_triggered:
                    off_triggered = True

    end = "?"
    if pos.no_progress >= NO_PROGRESS_DRAW:
        end = "impl-80"; n_impl += 1
    elif plies >= 300:
        end = "cap-300"; n_other += 1
    else:
        end = "natural"; n_other += 1
    if off_triggered:
        n_off += 1
    details.append((g, plies, end, off_triggered, off))

print(f"{GAMES} games, {SIMS} simulations, network: {'trained' if WEIGHTS else 'uniform'}")
print()
print(f"{'game':>6}{'plies':>7}{'end':>12}{'official 40 reached':>21}{'final official count':>23}")
for g, p, f, s, u in details:
    print(f"{g:>6}{p:>7}{f:>12}{('YES' if s else 'no'):>21}{u:>23}")
print()
print(f"ended by the IMPLEMENTED rule (80 plies)       : {n_impl}/{GAMES}")
print(f"in which the OFFICIAL rule would have triggered : {n_off}/{GAMES}")
print(f"other                                          : {n_other}/{GAMES}")
