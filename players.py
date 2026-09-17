"""
FIXED-strength opponents, to measure the champion's strength over time. They
use neither the network nor the MCTS: they pick the move directly -> stable,
fast and reproducible baselines.

  RandomPlayer    : a random legal move. The minimum baseline (the win rate
                    reaches ~100% as soon as the network learns anything:
                    useful at the start).
  GreedyPlayer    : 1 ply of material + "don't give pieces away" (subtracts the
                    opponent's best immediate recapture). Weak but not trivial:
                    it stays informative longer than random.
  AlphaBetaPlayer : fixed-depth alpha-beta search, a ladder of increasing and
                    reproducible strength.
"""
from __future__ import annotations
import numpy as np

from dama import W_MAN, W_KING, B_MAN, B_KING, Position, apply_move

_VAL = {W_MAN: 1, W_KING: 3, B_MAN: -1, B_KING: -3}   # signed (material)
_ABS = {W_MAN: 1, W_KING: 3, B_MAN: 1, B_KING: 3}     # absolute piece value


def _material(board) -> int:
    return sum(_VAL.get(p, 0) for p in board)


class RandomPlayer:
    name = "random"

    def move(self, pos: Position, rng: np.random.Generator):
        moves = pos.legal_moves()
        return moves[int(rng.integers(len(moves)))]


class AlphaBetaPlayer:
    """FIXED-depth alpha-beta search: a ladder of opponents of increasing and
    reproducible strength.

    Why it is needed. Chess has engines with a calibrated Elo to use as
    absolute anchors; Italian draughts does not. Not for lack of engines:
    Ed Gilbert's Kingsrow has a free Italian version, with a neural network and
    complete endgame databases up to ten pieces, and Martin Fierz's
    CheckerBoard hosts two more, Dama and Saltare. What is missing is the
    LADDER: none of them has a calibrated Elo, and they run as Windows DLLs
    inside a graphical interface.

    Fixed depths therefore remain the working anchor: in-process, with no
    dependencies, identical on every operating system, deterministic and with
    a strength that grows monotonically, so the rungs stay stable for the
    whole run. Kingsrow serves once, as the external check of where the
    champion really stands rather than of how it changes (tools/kingsrow.py
    drives its DLL directly).

    Compared with GreedyPlayer (in essence 1 ply of material) the useful
    measurement window is much longer: the champion keeps improving well after
    it has saturated greedy.

    The evaluation is deliberately simple and stated: material (king = 3),
    which accounts for almost everything, plus a small advancement term that
    encourages pushing towards promotion. It is not a strong engine and does
    not try to be: it must be STABLE, not good.
    """

    # Cap on the transposition table: beyond it the table is cleared. It only
    # keeps memory from growing without bound in a long game; the contents can
    # be regenerated, so clearing does not affect correctness.
    TT_MAX = 400_000

    def __init__(self, depth: int = 3, use_tt: bool = True):
        self.depth = depth
        self.name = f"ab{depth}"
        self.use_tt = use_tt          # False: only for the correctness tests
        # Transposition table: (board, side to move, depth) -> (value,
        # bound_type, best_move). Positions repeat a lot in the tree (different
        # move orders reach the same position), so storing them avoids
        # searching them again from scratch.
        self._tt: dict = {}

    # positional value: material + advancement of the men
    @staticmethod
    def _evaluate(board, color: int) -> float:
        score = 0.0
        for s, p in enumerate(board):
            if p == 0:
                continue
            score += _VAL.get(p, 0)
            if abs(p) == 1:                      # man: reward advancement
                row = s // 8
                # White advances towards row 0, Black towards row 7
                adv = (7 - row) if p > 0 else row
                score += 0.05 * adv * (1 if p > 0 else -1)
        return score * color                     # from the point of view of `color`

    # Quiescence: how many extra plies are allowed beyond the nominal depth
    # when the position is "noisy" (a capture is forced).
    MAX_EXT = 6

    @staticmethod
    def _order(moves):
        """Promising moves first: the more alpha-beta prunes, the fewer nodes
        it visits.

        With mandatory captures, when captures exist the legal moves are
        already filtered by capture priority and look alike; ordering matters
        mostly for quiet moves, where moves that promote come first (they
        change the material, hence the evaluation)."""
        return sorted(moves, key=lambda m: (m.promotes, len(m.captured)),
                      reverse=True)

    def _search(self, pos, depth: int, alpha: float, beta: float, color: int,
                ext: int = 0) -> float:
        # --- transposition table lookup ------------------------------------
        # Only at normal nodes (ext == 0): at quiescence nodes the value also
        # depends on how many extensions remain, which is not part of the key,
        # and storing them would give wrong results.
        alpha_orig = alpha
        key = None
        tt_move = None
        if self.use_tt and ext == 0 and depth > 0:
            key = (bytes((p + 2) for p in pos.board), pos.turn, depth)
            hit = self._tt.get(key)
            if hit is not None:
                val, flag, tt_move = hit
                if flag == 0:                    # exact value
                    return val
                elif flag == 1:                  # LOWER bound
                    alpha = max(alpha, val)
                else:                            # UPPER bound
                    beta = min(beta, val)
                if alpha >= beta:
                    return val

        moves = pos.legal_moves()
        if not moves:
            return -1000.0                       # the side that cannot move has lost
        if depth <= 0:
            # QUIESCENCE. Stopping with a mandatory capture pending distorts
            # the evaluation: the piece just lost would be counted without
            # seeing the recapture. Measured: without this, depth 2 LOSES to
            # greedy (0.40), because greedy does subtract the best immediate
            # recapture and is therefore less short-sighted. With the extension
            # the ladder is monotonic again.
            if ext >= self.MAX_EXT or not moves[0].is_capture:
                return self._evaluate(pos.board, color)
            ext += 1                             # noisy position: keep going
        # The best move found by an earlier search of this same position is
        # tried first: if it is still good, the cutoff happens at once and the
        # remaining branches are not even explored.
        ordered = self._order(moves)
        if tt_move is not None:
            for i, m in enumerate(ordered):
                if m == tt_move:
                    ordered.insert(0, ordered.pop(i))
                    break

        best, best_move = -1e9, None
        for m in ordered:
            # the child evaluates from the opponent's point of view: negate
            val = -self._search(pos.play(m), depth - 1, -beta, -alpha, -color, ext)
            if val > best:
                best, best_move = val, m
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break                            # cutoff

        # --- storage, with the BOUND TYPE ----------------------------------
        # A value produced under a cutoff is not exact: it is a bound. Without
        # this distinction the table would return wrong values when read back
        # with a different alpha-beta window.
        if key is not None:
            if best <= alpha_orig:
                flag = 2                         # UPPER bound (fail-low)
            elif best >= beta:
                flag = 1                         # LOWER bound (fail-high)
            else:
                flag = 0                         # exact
            if len(self._tt) >= self.TT_MAX:
                self._tt.clear()
            self._tt[key] = (best, flag, best_move)
        return best

    def move(self, pos: Position, rng: np.random.Generator):
        moves = pos.legal_moves()
        if len(moves) == 1:
            return moves[0]
        best_val, best = -1e9, []
        for m in self._order(moves):
            val = -self._search(pos.play(m), self.depth - 1, -1e9, 1e9, -pos.turn)
            if val > best_val:
                best_val, best = val, [m]
            elif val == best_val:
                best.append(m)
        # random tie-break: without it, repeated games would be identical
        return best[int(rng.integers(len(best)))] if len(best) > 1 else best[0]


class GreedyPlayer:
    name = "greedy"

    def move(self, pos: Position, rng: np.random.Generator):
        moves = pos.legal_moves()
        best_score, best = None, []
        for m in moves:
            nb = apply_move(pos.board, m, pos.turn)
            score = _material(nb) * pos.turn  # from the point of view of the side to move
            # penalty: the opponent's best immediate recapture
            opp = Position(nb, -pos.turn)
            caps = [mm for mm in opp.legal_moves() if mm.is_capture]
            if caps:
                score -= max(sum(_ABS.get(nb[c], 0) for c in mm.captured)
                             for mm in caps)
            if best_score is None or score > best_score:
                best_score, best = score, [m]
            elif score == best_score:
                best.append(m)
        return best[int(rng.integers(len(best)))] if len(best) > 1 else best[0]
