r"""
Bridge to a CheckerBoard engine: the champion plays against Kingsrow.

WHY. The project measures strength in two ways, and neither is absolute: the
generation ladder says how we change over time, the fixed alpha-beta depths say
how much better we are than a search we already know. Where we stand against the
engines that really play Italian draughts, neither says. Kingsrow Italian by Ed
Gilbert is itself a neural network plus search, with a 1.7-million-position
opening book and endgame databases up to ten pieces: it is the only external
reference available.

HOW. CheckerBoard engines are Windows DLLs exposing getmove and islegal. They are
loaded with ctypes and queried directly, without the graphical interface: a game
becomes a loop, and twenty games a quarter of an hour instead of a day spent
transcribing moves by hand.

THE DELICATE POINT is the mapping between the two board representations. The API
documents the format -- integers in an 8x8 matrix, with constants for color and
rank -- but does NOT say how the coordinates are oriented, nor which 32 squares
the Italian variant uses, which are the other ones compared with English
draughts. Getting that orientation wrong does not produce an error: it produces
normal-looking games between two engines that are looking at two different
boards.

So the mapping below was not deduced but MEASURED, and it can be measured again
at any time:

    python tools/kingsrow.py --validate

compares the whole set of legal moves between the two engines on positions
reached by random play. At the time of writing: 120 positions out of 120, 24 of
them with mandatory captures. It is a much stronger check than comparing replies
alone, because a wrong orientation can get a handful of moves right but not the
complete set on every position.

Note on the two equivalent conventions: rotating the board by 180 degrees and
swapping the colors leaves the game identical, so two different orientations give
the same results. Both pass the validation; one of them is used here.

USAGE
    python tools/kingsrow.py --validate                 check the mapping
    python tools/kingsrow.py --games 20 --time 0.1      match against the champion

    --weights FOLDER     where champion.pt is
    --sims N             simulations of our engine (default 800)
    --time SECONDS       Kingsrow's thinking time: it is its strength knob, and
                         the parameter to move to find the level at which the
                         two are equal
    --dll PATH           another CheckerBoard engine (it must be Italian)
    --save FILE          write the games played, move by move

Needs Windows and a 64-bit Python, like the DLL.
"""
from __future__ import annotations
import argparse
import ctypes
import math
import os
import sys
import time

import numpy as np
import torch

# The project modules live in the parent directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dama import (IN_PLANES, Position, WHITE, W_MAN, W_KING, B_MAN, B_KING,
                  rc, sq)
from model import PolicyValueNet
from evaluators import NetEvaluator
from mcts import MCTS

DLL_DEFAULT = r"C:\Program Files\CheckerBoard64\engines\kr_italian64.dll"

# CheckerBoard API constants (cbdeveloper).
CB_FREE, CB_WHITE, CB_BLACK, CB_MAN, CB_KING = 0, 1, 2, 4, 8
_RANK = {W_MAN: CB_MAN, W_KING: CB_KING, B_MAN: CB_MAN, B_KING: CB_KING}

# Orientation measured, not assumed: see --validate.
def _to_cb(r: int, c: int) -> tuple[int, int]:
    return (7 - c, r)


_FROM_CB = {_to_cb(r, c): (r, c) for r in range(8) for c in range(8)}


class _coor(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int), ("y", ctypes.c_int)]


class _CBmove(ctypes.Structure):
    _fields_ = [("jumps", ctypes.c_int), ("newpiece", ctypes.c_int),
                ("oldpiece", ctypes.c_int), ("from_", _coor), ("to", _coor),
                ("path", _coor * 12), ("del_", _coor * 12),
                ("delpiece", ctypes.c_int * 12)]


_Board8 = (ctypes.c_int * 8) * 8


class KingsRow:
    """A CheckerBoard engine seen as any other player."""

    def __init__(self, dll: str = DLL_DEFAULT):
        if not os.path.exists(dll):
            raise SystemExit(f"DLL not found: {dll}")
        root = os.path.dirname(os.path.dirname(dll))
        engines = os.path.dirname(dll)
        # egdb64.dll lives in CheckerBoard's root, not next to the engine:
        # without adding it to the search path loading fails with a generic
        # "module not found" that does not say which one.
        os.add_dll_directory(root)
        os.add_dll_directory(engines)
        # The opening book (23 MB, 1.7 million positions) is looked for in the
        # WORKING folder, and not at load time but at first use: stepping in for
        # a moment is not enough, one has to stay there. Measured trying both --
        # going back at once, 'about' declares "Not using an opening book" and
        # one would end up measuring an opponent weaker than the real one, with
        # no other sign.
        #
        # So the caller must have already made its own paths absolute: main()
        # does, and whoever uses this class from elsewhere is warned.
        os.chdir(engines)
        self.d = ctypes.WinDLL(dll)
        self.name = self._command("name")
        self.about = self._command("about")
        if "Not using an opening book" in self.about:
            print("[kingsrow] WARNING: opening book NOT loaded")
        game_type = self._command("get gametype").strip()
        if game_type != "22":
            raise SystemExit(
                f"the engine declares gametype {game_type}, not 22 (Italian draughts). "
                "An engine of another variant would play another game.")

    def _command(self, cmd: str) -> str:
        rep = ctypes.create_string_buffer(1024)
        self.d.enginecommand(ctypes.create_string_buffer(cmd.encode(), 256), rep)
        return rep.value.decode(errors="replace")

    def _board(self, pos: Position):
        b = _Board8()
        for s in range(64):
            p = pos.board[s]
            if p:
                x, y = _to_cb(*rc(s))
                b[x][y] = (CB_WHITE if p > 0 else CB_BLACK) | _RANK[p]
        return b

    def legal_moves(self, pos: Position) -> set[tuple[int, int]]:
        """(from, to) pairs that the DLL declares legal. Used by the validation."""
        b = self._board(pos)
        color = CB_WHITE if pos.turn == WHITE else CB_BLACK
        out = set()
        for f in range(1, 33):
            for a in range(1, 33):
                mv = _CBmove()
                if not self.d.islegal(b, ctypes.c_int(color), ctypes.c_int(f),
                                      ctypes.c_int(a), ctypes.byref(mv)):
                    continue
                src = _FROM_CB.get((mv.from_.x, mv.from_.y))
                dst = _FROM_CB.get((mv.to.x, mv.to.y))
                if src and dst:
                    out.add((sq(*src), sq(*dst)))
        return out

    def move(self, pos: Position, think_time: float = 0.1):
        """The move chosen by the engine, translated into one of our Moves."""
        b = self._board(pos)
        color = CB_WHITE if pos.turn == WHITE else CB_BLACK
        mv = _CBmove()
        playnow = ctypes.c_int(0)
        info = ctypes.create_string_buffer(1024)
        self.d.getmove(b, ctypes.c_int(color), ctypes.c_double(think_time), info,
                       ctypes.byref(playnow), ctypes.c_int(0), ctypes.c_int(0),
                       ctypes.byref(mv))
        src = _FROM_CB.get((mv.from_.x, mv.from_.y))
        dst = _FROM_CB.get((mv.to.x, mv.to.y))
        if src is None or dst is None:
            raise RuntimeError(f"move off the board: {mv.from_.x},{mv.from_.y}")
        frm, to = sq(*src), sq(*dst)
        taken = {sq(*_FROM_CB[(mv.del_[i].x, mv.del_[i].y)])
                 for i in range(min(mv.jumps, 12))
                 if (mv.del_[i].x, mv.del_[i].y) in _FROM_CB}

        # Among our legal moves the matching one is looked for. Start and end
        # alone are not always enough: two different jump chains can share them,
        # and then the captured pieces tell them apart.
        candidates = [m for m in pos.legal_moves() if m.frm == frm and m.to == to]
        if len(candidates) > 1:
            candidates = [m for m in candidates if set(m.captured) == taken] or candidates
        if not candidates:
            raise RuntimeError(
                f"the engine proposes {frm}->{to}, which is not legal for us. "
                "The mapping between the boards is broken: run --validate again.")
        return candidates[0]


# ------------------------------------------------------------------ validation
def validate(kr: KingsRow, n: int = 120, seed: int = 7) -> bool:
    rng = np.random.default_rng(seed)
    probes, pos = [], Position()
    while len(probes) < n:
        if pos.is_terminal():
            pos = Position()
            continue
        probes.append(pos)
        moves = pos.legal_moves()
        pos = pos.play(moves[int(rng.integers(len(moves)))])

    with_capture = sum(1 for p in probes if any(m.is_capture for m in p.legal_moves()))
    print(f"[validate] {n} positions from random play, {with_capture} with mandatory captures")
    same, differences = 0, []
    for p in probes:
        ours = {(m.frm, m.to) for m in p.legal_moves()}
        theirs = kr.legal_moves(p)
        if ours == theirs:
            same += 1
        else:
            differences.append((sorted(ours - theirs), sorted(theirs - ours)))
    for ours_only, theirs_only in differences[:3]:
        print(f"    DIFFERENCE: ours only {ours_only}, theirs only {theirs_only}")
    print(f"[validate] {same}/{n} positions with the same set of legal moves")
    if same == n:
        print("[validate] the two engines see the same board and the same game.")
    else:
        print("[validate] do NOT use this bridge: the representations diverge.")
    return same == n


# ----------------------------------------------------------------------- match
def _notation(m) -> str:
    """The move in dark-square numbers, as in play.py and rules_crosscheck.py."""
    from play import SQUARE_NUMBER
    if m.is_capture:
        s = "x".join(str(SQUARE_NUMBER[p]) for p in m.path)
    else:
        s = f"{SQUARE_NUMBER[m.frm]}-{SQUARE_NUMBER[m.to]}"
    return s + ("=D" if m.promotes else "")


def _write_header(f, args, kr) -> None:
    f.write(f"{kr.name.strip()} vs our champion\n")
    f.write(f"us: {args.sims} simulations per move\n")
    f.write(f"Kingsrow: {args.time}s per move"
            f"{', no book' if args.no_book else ''}"
            f"{f', hash {args.hash}mb' if args.hash else ''}\n")
    f.write("dark squares numbered 1..32; x = capture, =D = promotion\n\n")
    f.flush()


def _write_game(f, n, we_white, outcome, plies, reason, moves) -> None:
    """One game in plain text, written AS SOON AS it ends.

    The score says how much is lost, the moves say why. It is written at once
    and the buffer is flushed, instead of accumulating everything in memory and
    saving at the end: a fifty-game match lasts half an hour, and keeping it all
    in memory would lose everything on an interruption at the last game. This way
    the file can be read while the match is being played.
    """
    f.write("=" * 68 + "\n")
    f.write(f"Game {n}  --  we play {'White' if we_white else 'Black'}"
            f"  --  {outcome.upper()}  ({plies} plies, {reason})\n")
    f.write("=" * 68 + "\n")
    for i in range(0, len(moves), 2):
        pair = moves[i:i + 2]
        # The explicit space after the field is NOT redundant: a chain such as
        # 26x17x10x1=D fills exactly the field width, and without a separator
        # the two moves stick together into a token nobody can read back. It
        # really happened, in two games out of fifty.
        f.write(f"{i // 2 + 1:3d}. {pair[0]:<16} "
                f"{pair[1] if len(pair) > 1 else ''}\n")
    f.write("\n")
    f.flush()


def elo(s: float) -> float:
    s = min(max(s, 1e-6), 1 - 1e-6)
    return -400.0 * math.log10(1.0 / s - 1.0)


def match(kr: KingsRow, args):
    ckpt = os.path.join(args.weights, "champion.pt")
    net = PolicyValueNet(in_planes=IN_PLANES, channels=96, n_blocks=6,
                                 use_se=True)
    net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    net.eval()
    ev = NetEvaluator(net, args.device)
    print(f"[match] {args.games} games, us {args.sims} simulations, "
          f"Kingsrow {args.time}s per move, alternating colors")

    won = drawn = lost = 0
    f = open(args.save, "w", encoding="utf-8") if args.save else None
    if f:
        _write_header(f, args, kr)
    for g in range(args.games):
        rng = np.random.default_rng(args.seed + g)
        we_white = (g % 2 == 0)
        pos, plies, moves = Position(), 0, []
        t0 = time.time()
        while not pos.is_terminal() and plies < 300:
            if (pos.turn == WHITE) == we_white:
                root = MCTS(ev, n_sims=args.sims, c_puct=1.5, batch_size=8,
                            rng=rng).run(pos, add_noise=False)
                children = list(root.children.items())
                # Sampled openings: without them, two deterministic players
                # would replay the same game every time.
                if plies < args.temp_plies:
                    w = np.array([st.N for _, st in children], float)
                    move = children[int(rng.choice(len(children), p=w / w.sum()))][0]
                else:
                    move = max(children, key=lambda kv: kv[1].N)[0]
            else:
                move = kr.move(pos, args.time)
            moves.append(_notation(move))
            pos = pos.play(move)
            plies += 1
        r = pos.result() or 0
        ours = r if we_white else -r
        won += ours > 0
        drawn += ours == 0
        lost += ours < 0
        outcome = "win" if ours > 0 else ("draw" if ours == 0 else "loss")
        dt = time.time() - t0
        reason = ("no-progress rule" if pos.no_progress >= 80
                  else ("truncated" if plies >= 300 else "no legal moves"))
        if f:
            _write_game(f, g + 1, we_white, outcome, plies, reason, moves)
        print(f"   {g + 1:3d}/{args.games}  {outcome:5s} {plies:4d} plies "
              f"{dt:5.0f}s   [{won}W {drawn}D {lost}L]", flush=True)

    if f:
        f.close()
        print(f"\n  games written to {args.save}")

    n = won + drawn + lost
    score = (won + 0.5 * drawn) / max(1, n)
    var = (won * (1 - score) ** 2 + drawn * (0.5 - score) ** 2
           + lost * score ** 2) / max(1, n)
    half = 1.96 * math.sqrt(var / max(1, n))
    print(f"\n  {won} won, {drawn} drawn, {lost} lost out of {n}")
    print(f"  score: {score:.3f} [{score - half:.3f}, {score + half:.3f}]")
    # The Elo formula diverges only at the EXTREMES of the score, not when there
    # are no wins: four draws out of fifty make 0.040, a perfectly measurable
    # number. At 0 or 1 the number it would print is only the value it was
    # clipped to, and passing it off as a measurement would be worse than not
    # giving it: a whitewash says "beyond the resolution of this test", not
    # "minus two thousand four hundred".
    if score <= 0.0 or score >= 1.0:
        headline = "No point scored" if score <= 0.0 else "No point conceded"
        direction = "BELOW" if score <= 0.0 else "ABOVE"
        print(f"  {headline}: the gap is {direction} the resolution")
        print(f"  of {n} games. How much cannot be said -- the Elo of a")
        print(f"  whitewash is infinite, and the printed number would be the clipping.")
        print(f"  To get a figure a closer opponent is needed:")
        print(f"  lower --time, or try --no-book and --hash 1.")
    else:
        print(f"  Elo vs Kingsrow at {args.time}s: {elo(score):+.0f} "
              f"[{elo(score - half):+.0f}, {elo(score + half):+.0f}]")


def kingsrow_vs_alphabeta(kr: KingsRow, args):
    """Kingsrow against fixed-depth alpha-beta.

    It is the bridge between the two scales, and it is worth more than it seems.
    The champion has always been measured against the fixed depths, which
    however only say "better than a search we know". Finding which depth
    Kingsrow corresponds to at a given time gives every measurement already made
    against alpha-beta an external reference -- without needing our champion to
    win games against Kingsrow, which it has not done so far (it draws some and
    wins none).
    """
    from players import AlphaBetaPlayer
    opp = AlphaBetaPlayer(depth=args.vs_ab)
    print(f"[match] Kingsrow ({args.time}s) against {opp.name}, "
          f"{args.games} games, alternating colors")
    won = drawn = lost = 0
    for g in range(args.games):
        rng = np.random.default_rng(args.seed + g)
        kr_white = (g % 2 == 0)
        pos, plies = Position(), 0
        t0 = time.time()
        while not pos.is_terminal() and plies < 300:
            if (pos.turn == WHITE) == kr_white:
                move = kr.move(pos, args.time)
            else:
                move = opp.move(pos, rng)
            pos = pos.play(move)
            plies += 1
        r = pos.result() or 0
        theirs = r if kr_white else -r
        won += theirs > 0
        drawn += theirs == 0
        lost += theirs < 0
        print(f"   {g + 1:3d}/{args.games}  "
              f"{'Kingsrow wins' if theirs > 0 else ('draw' if theirs == 0 else opp.name + ' wins'):16s}"
              f" {plies:4d} plies {time.time() - t0:5.0f}s   [{won}W {drawn}D {lost}L]")
    n = max(1, won + drawn + lost)
    score = (won + 0.5 * drawn) / n
    print(f"\n  Kingsrow: {won} won, {drawn} drawn, {lost} lost -> score {score:.3f}")
    if 0 < score < 1:
        print(f"  Kingsrow's Elo vs {opp.name}: {elo(score):+.0f}")
    else:
        print(f"  whitewash: Kingsrow is off the scale against {opp.name}, "
              "try a greater depth")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dll", default=DLL_DEFAULT)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--weights", default=os.environ.get("DAMA_WEIGHTS_DIR", "weights"))
    ap.add_argument("--device", default=os.environ.get("DAMA_DEVICE", "cpu"))
    ap.add_argument("--sims", type=int, default=800)
    ap.add_argument("--time", type=float, default=0.1)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--temp-plies", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    # Knobs to BUILD A LADDER of weaker opponents. At the minimum time Kingsrow
    # is still very strong, and a score of 0.000 is not a measurement: it only
    # says "far below", without saying how far. Removing the book and reducing
    # the memory gives lower rungs, and the rung at which points start coming is
    # the real placement.
    ap.add_argument("--no-book", action="store_true",
                    help="turn off the 1.7M-position opening book")
    ap.add_argument("--hash", type=int, default=0, metavar="MB",
                    help="shrink the transposition table (default: 128mb)")
    ap.add_argument("--vs-ab", type=int, default=0, metavar="DEPTH",
                    help="Kingsrow against alpha-beta: links the two scales")
    ap.add_argument("--save", default=None, metavar="FILE",
                    help="write the games played, move by move")
    args = ap.parse_args()
    # ALL paths must be resolved here, before building KingsRow: its constructor
    # moves the working folder inside the engine's installation, which moreover
    # is not writable without privileges. A relative path left behind fails at
    # the end of the game, after half an hour of play, the worst moment to notice.
    args.weights = os.path.abspath(args.weights)
    if args.save:
        args.save = os.path.abspath(args.save)

    kr = KingsRow(args.dll)
    print(f"[kingsrow] {kr.name.strip()}")
    if args.no_book:
        print(f"[kingsrow] {kr._command('set book 0').strip()}")
    if args.hash:
        print(f"[kingsrow] {kr._command(f'set hashsize {args.hash}').strip()}")
    if args.validate:
        raise SystemExit(0 if validate(kr) else 1)
    if args.vs_ab:
        kingsrow_vs_alphabeta(kr, args)
    else:
        match(kr, args)


if __name__ == "__main__":
    main()
