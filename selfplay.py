"""
Self-play: plays games with the (batched) MCTS and produces the training samples.

Each sample is (X, pi, mask, z):
  X    : [7,8,8] tensor of the position (canonical encoder: 4 piece planes +
         3 engineered ones)
  pi   : policy target [POLICY_SIZE] = the MCTS visit distribution (canonical
         frame, compact action 8*dark_square + 2*direction + mode)
  mask : [POLICY_SIZE] mask of the legal actions (used for MASKING in the loss)
  z    : final outcome from the point of view of the side to move (+1/-1/0,
         scaled by gamma^d when value_discount < 1, see play_game)
"""
from __future__ import annotations
import numpy as np

from dama import Position, BLACK, POLICY_SIZE, legal_mask, NO_PROGRESS_DRAW
from dama.encoder import encode, move_policy_index
from mcts import MCTS
from dama.board import W_MAN, W_KING, B_MAN, B_KING

_MAT = {W_MAN: 1, W_KING: 3, B_MAN: -1, B_KING: -3}


def _material_result(pos) -> int:
    """Adjudicates a TRUNCATED game (max_plies reached without an end) by
    material balance (king = 3). +1 White, -1 Black, 0 even -- from White's
    point of view, consistent with result()."""
    m = sum(_MAT.get(p, 0) for p in pos.board)
    return 1 if m > 0 else (-1 if m < 0 else 0)


def visit_policy_target(pos: Position, root) -> np.ndarray:
    """Visit distribution over the root's children, as a [POLICY_SIZE] vector
    of compact actions (same frame as the encoder)."""
    flip = pos.turn == BLACK
    target = np.zeros(POLICY_SIZE, dtype=np.float32)
    for move, child in root.children.items():
        target[move_policy_index(move, flip)] += child.N
    s = target.sum()
    if s > 0:
        target /= s
    return target


def select_move(root, temperature: float, rng: np.random.Generator):
    moves = list(root.children.keys())
    visits = np.array([root.children[m].N for m in moves], dtype=np.float64)
    if temperature <= 1e-6 or visits.sum() == 0:
        return moves[int(visits.argmax())]
    p = visits ** (1.0 / temperature)
    p /= p.sum()
    return moves[int(rng.choice(len(moves), p=p))]


def _root_q_spread(root) -> float:
    """How much the search TELLS APART the moves at the root: the difference
    between the best and the worst value among the visited children (from the
    point of view of the side to move, so Q is negated as in the PUCT descent).

    It is the number that says whether spending more on search makes sense.
    With a high policy entropy there are two opposite cases, which cannot be
    told apart without it:
      * SMALL spread -> the Q values are close and the value head does not
        discriminate; the visit distribution is necessarily diffuse, and
        neither more simulations nor better search algorithms (Gumbel,
        sequential halving) can sharpen it. The bottleneck is the network,
        not the search.
      * LARGE spread -> the Q values differ but the visits stay spread out:
        PUCT is exploring too much, and lowering c_puct fixes it for free.
    """
    qs = [-st.Q for st in root.children.values() if st.N > 0]
    return (max(qs) - min(qs)) if len(qs) >= 2 else 0.0


def _root_value(root) -> float:
    """Value estimated by the search at the root, from the point of view of the
    side to move. The children store Q from THEIR point of view, so it is
    negated; the average is weighted by visits, as in the PUCT descent."""
    tot = sum(st.N for st in root.children.values())
    if tot <= 0:
        return 0.0
    return sum(st.N * (-st.Q) for st in root.children.values()) / tot


def play_game(evaluator, n_sims: int = 100, temp_moves: int = 12,
              c_puct: float = 1.5, batch_size: int = 8, max_plies: int = 300,
              seed: int | None = None, value_discount: float = 1.0):
    """Plays one complete self-play game.

    Returns (samples, z_white, plies, stats), where `stats` is a dict with the
    game's diagnostic metrics (see metrics.py): length, outcome, termination
    reason, captures, promotions, mean entropy of the visit policy, and the gap
    between the value estimated by the search and the final outcome.

    `value_discount` (gamma) shrinks the value target in proportion to the
    DISTANCE from the end: a position d plies before the end gets z * gamma^d.
    With 1.0 (the default) nothing changes.

    What it is for. Without a discount, in a won position every move is worth
    +1 and the search has no way to prefer one: the visits spread almost
    uniformly and the network learns to be indifferent. With four kings
    against one, every legal move is valued +1.000 and the visits spread
    evenly among them, so the engine shuffles until the no-progress rule ends
    the game in a draw. With the discount, winning sooner is worth more than
    winning later, the tie is broken and the search finds a direction again.

    What it does NOT fix: drawn games are labelled zero, and a discounted zero
    is still zero. If a won position is squandered all the way to a
    no-progress draw, this mechanism does not recover it -- it acts only inside
    games that someone actually won, making it more attractive to win them
    sooner.
    """
    from metrics import entropy_of
    rng = np.random.default_rng(seed)
    mcts = MCTS(evaluator, n_sims=n_sims, c_puct=c_puct, batch_size=batch_size, rng=rng)
    pos = Position()
    history = []  # (X, pi, mask, turn)
    plies = 0
    captures = promotions = 0
    entropies, root_values, turns_at_root, q_spreads = [], [], [], []

    while not pos.is_terminal() and plies < max_plies:
        root = mcts.run(pos, add_noise=True)
        pi = visit_policy_target(pos, root)
        history.append((encode(pos), pi, legal_mask(pos), pos.turn))
        entropies.append(entropy_of(pi))
        root_values.append(_root_value(root))
        q_spreads.append(_root_q_spread(root))
        turns_at_root.append(pos.turn)
        temp = 1.0 if plies < temp_moves else 0.0
        move = select_move(root, temp, rng)
        if move.is_capture:
            captures += len(move.captured)
        if move.promotes:
            promotions += 1
        pos = pos.play(move)
        plies += 1

    natural = pos.result()
    truncated = natural is None
    # draw by the no-progress rule: really over, but by the counter
    no_progress = (not truncated) and pos.no_progress >= NO_PROGRESS_DRAW
    z_white = natural if natural is not None else _material_result(pos)

    # Calibration: how close the value estimated during the game is to the real
    # outcome. Both are from the point of view of the side to move at that
    # moment, so they compare directly.
    mae = None
    if root_values:
        mae = sum(abs(v - z_white * t) for v, t in zip(root_values, turns_at_root))
        mae /= len(root_values)

    stats = {
        "plies": plies,
        "z_white": z_white,
        "truncated": truncated,
        "no_progress": no_progress,
        "captures": captures,
        "promotions": promotions,
        "entropy": (sum(entropies) / len(entropies)) if entropies else None,
        "q_spread": (sum(q_spreads) / len(q_spreads)) if q_spreads else None,
        "value_mae": mae,
    }
    # d = plies left to the end; the last decision of the game has d=0 and
    # keeps full credit.
    n = len(history)
    samples = [(X, pi, mask,
                float(z_white * turn * (value_discount ** (n - 1 - i))))
               for i, (X, pi, mask, turn) in enumerate(history)]
    return samples, z_white, plies, stats
