r"""
Play against the champion, or measure its strength against a fixed opponent.

It answers two different questions that are easily confused. The first is
"does it play sensibly?", and it is qualitative: it is answered by looking at the
moves, not at numbers -- whether it opens reasonably, sees multiple captures,
does not give pieces away, knows how to push its men to promotion in the
endgame. The second is "how strong is it?", and it is quantitative: it is
answered only by making it play against something reproducible and counting the
points.

The first mode makes it play against you. The second against fixed-depth
alpha-beta search, the reproducible anchor for ITALIAN draughts: the well-known
checkers engines (Chinook, Cake) play English draughts, where a man can capture
a king and there is no capture-priority rule, and the one with an Italian
version, Kingsrow, has no calibrated Elo (tools/kingsrow.py uses it as an
external check).

USAGE
    python play.py                          you play White, the engine Black
    python play.py --side black             you play Black
    python play.py --sims 100               weaker engine (easier to beat)
    python play.py --vs-ab 4 --games 20     measurement: champion vs alpha-beta

    --weights FOLDER     where champion.pt is (default: the DAMA_WEIGHTS_DIR
                         environment variable, otherwise "weights")
    --device cpu|cuda    where the network runs
    --sims N             simulations per move; it is the strength knob

    --ascii              no colors, if the terminal does not render them well

DURING THE GAME type the row number of the move, or:
    h   the engine suggests what it would play in your place
    u   take back one of your moves (also undoes the reply)
    s N change the engine's simulations during the game
    q   quit

The dark squares are numbered 1 to 32 and the number stays visible when the
square is empty, so moves can be read straight off the board. The choice however
is made by the row number in the list: in Italian draughts capturing is
mandatory with capture priority, and two different jump chains can start and
end on the same squares: giving the start and end squares would not be enough to
tell them apart.
"""
from __future__ import annotations
import argparse
import os
import sys
import time

import numpy as np
import torch

from dama import (IN_PLANES, Position, WHITE, BLACK, W_MAN, W_KING, B_MAN,
                  B_KING, sq)
from model import PolicyValueNet
from evaluators import NetEvaluator
from mcts import MCTS
from players import AlphaBetaPlayer, RandomPlayer, GreedyPlayer

# --- drawing the board ---------------------------------------------------------
#
# The DARK squares are numbered 1 to 32 and the number stays visible when the
# square is empty: so the board reads itself and moves are written "22 -> 18"
# instead of with a pair of coordinates. The light squares never come into play
# in draughts and stay empty.
SQUARE_NUMBER = {}
_k = 0
for _r in range(8):
    for _c in range(8):
        if (_r + _c) % 2 == 1:
            _k += 1
            SQUARE_NUMBER[sq(_r, _c)] = _k

BG_DARK     = "\033[48;5;95m"      # dark wood: the squares in play
BG_LIGHT    = "\033[48;5;180m"     # light wood: always empty
BG_LAST     = "\033[48;5;108m"     # green: the squares touched by the last move
BG_CAPTURED = "\033[48;5;131m"     # red: the pieces just captured
FG_WHITE    = "\033[1;38;5;231m"
FG_BLACK    = "\033[1;38;5;16m"
FG_NUMBER   = "\033[38;5;138m"     # the square number, in a subdued tone
RESET       = "\033[0m"

ROUND = {W_MAN: "●", W_KING: "◉", B_MAN: "●", B_KING: "◉"}
PLAIN = {W_MAN: "o", W_KING: "O", B_MAN: "x", B_KING: "X"}


def setup_terminal(force_ascii: bool) -> bool:
    """Prepares the console and says whether to fall back on plain characters.

    Two separate problems, both on Windows only. The first: the console does not
    interpret ANSI sequences until someone asks for it, and os.system("") does
    so as a side effect. The second: standard output uses the local code page
    (cp1252 on Western European systems), which does not have the round symbols
    -- without noticing it here, the game dies with an encoding exception at the
    first board drawn.
    """
    if os.name == "nt":
        os.system("")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    if force_ascii:
        return True
    try:
        "".join(ROUND.values()).encode(sys.stdout.encoding or "utf-8")
        return False
    except (UnicodeEncodeError, LookupError):
        print("[play] the terminal cannot print the round symbols: "
              "using plain ones")
        return True


def draw_board(board, last=None, ascii_only=False) -> str:
    """The board as one looks at it, not as it is serialized.

    `last` is the move just played: its start and end squares are highlighted in
    green and the captured pieces in red, because in a jump chain working out
    what just happened by reading two text grids is needlessly tiring.
    """
    glyphs = PLAIN if ascii_only else ROUND
    touched = set()
    taken = set()
    if last is not None:
        touched = {last.frm, last.to}
        taken = set(last.captured)

    rows = ["    " + "".join(f" {c} " for c in range(8))]
    for r in range(8):
        cells = []
        for c in range(8):
            s = sq(r, c)
            dark = (r + c) % 2 == 1
            p = board[s]
            if ascii_only:
                text = f" {glyphs[p]} " if p else (
                    f"{SQUARE_NUMBER[s]:2d} " if dark else "   ")
                cells.append(text)
                continue
            if s in taken:
                bg = BG_CAPTURED
            elif s in touched:
                bg = BG_LAST
            else:
                bg = BG_DARK if dark else BG_LIGHT
            if p:
                fg = FG_WHITE if p > 0 else FG_BLACK
                cells.append(f"{bg}{fg} {glyphs[p]} {RESET}")
            elif dark:
                cells.append(f"{bg}{FG_NUMBER}{SQUARE_NUMBER[s]:2d} {RESET}")
            else:
                cells.append(f"{bg}   {RESET}")
        rows.append(f" {r}  " + "".join(cells) + f"  {r}")
    rows.append("    " + "".join(f" {c} " for c in range(8)))
    return "\n".join(rows)


def count_pieces(board):
    """Pieces left per color, to keep the material in view."""
    w = sum(1 for p in board if p == W_MAN), sum(1 for p in board if p == W_KING)
    b = sum(1 for p in board if p == B_MAN), sum(1 for p in board if p == B_KING)
    return w, b


def load_champion(folder: str, device: str):
    ckpt = os.path.join(folder, "champion.pt")
    if not os.path.exists(ckpt):
        raise SystemExit(f"{ckpt} not found. Give the folder with --weights.")
    net = PolicyValueNet(
        in_planes=IN_PLANES,
        channels=int(os.environ.get("DAMA_CHANNELS", "96")),
        n_blocks=int(os.environ.get("DAMA_BLOCKS", "6")),
        use_se=os.environ.get("DAMA_USE_SE", "1") != "0")
    net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    net.eval()
    print(f"[play] champion loaded from {ckpt}")
    return NetEvaluator(net, device)


def describe(m) -> str:
    """The move in the dark-square numbers, the same ones printed on the board.

    A jump chain is written in full (12x21x30) because with capture priority two
    chains can start and end on the same squares along different routes, and
    then start and end are not enough to tell them apart.
    """
    if m.is_capture:
        s = "x".join(str(SQUARE_NUMBER[p]) for p in m.path)
        s += f"   captures {len(m.captured)}"
    else:
        s = f"{SQUARE_NUMBER[m.frm]}-{SQUARE_NUMBER[m.to]}"
    if m.promotes:
        s += "   promotes to KING"
    return s


def show_board(pos: Position, move_no: int, last=None, ascii_only=False):
    (wm, wk), (bm, bk) = count_pieces(pos.board)
    print()
    print(draw_board(pos.board, last, ascii_only))
    glyphs = PLAIN if ascii_only else ROUND
    print(f"\n    White {glyphs[W_MAN]} {wm} men, {glyphs[W_KING]} {wk} kings"
          f"      Black {glyphs[B_MAN]} {bm} men, {glyphs[B_KING]} {bk} kings")
    side = "WHITE" if pos.turn == WHITE else "BLACK"
    warning = ""
    if pos.no_progress >= 60:
        warning = f"   <-- draw at 80"
    print(f"    move {move_no}, {side} to move"
          f"      no progress: {pos.no_progress}/80{warning}")


def search(ev, pos: Position, sims: int, rng):
    """A search without noise: the engine plays its best, it does not explore."""
    t0 = time.time()
    root = MCTS(ev, n_sims=sims, c_puct=1.5, batch_size=8, rng=rng).run(
        pos, add_noise=False)
    return root, time.time() - t0


def summarize(root, how_many: int = 3) -> str:
    """The most visited moves with their value, from the point of view of the side
    to move.

    The value stored in a child is from the point of view of the side that
    SUFFERS the move -- the tree's convention, which alternates the sign at
    every level -- so it is negated to read it from the side that is moving.
    """
    children = sorted(root.children.items(), key=lambda kv: kv[1].N, reverse=True)
    tot = sum(st.N for _, st in children) or 1.0
    rows = []
    for mv, st in children[:how_many]:
        rows.append(f"      {describe(mv):<34} {100.0 * st.N / tot:5.1f}% "
                    f"value {-st.Q:+.2f}")
    return "\n".join(rows)


def outcome_text(pos: Position) -> str:
    r = pos.result()
    if r is None:
        return "game interrupted"
    if r == 0:
        return "DRAW"
    return "WHITE wins" if r > 0 else "BLACK wins"


# ============================ interactive game ================================
def interactive(args):
    ev = load_champion(args.weights, args.device)
    rng = np.random.default_rng(args.seed)
    human = WHITE if args.side.lower().startswith("w") else BLACK
    sims = args.sims

    pos = Position()
    history: list[tuple] = []       # (position, move that led to it)
    last = None
    n = 1
    glyphs = PLAIN if args.ascii else ROUND
    print(f"\n  You play {'WHITE ' + glyphs[W_MAN] if human == WHITE else 'BLACK ' + glyphs[B_MAN]}."
          f"  The engine thinks {sims} simulations per move.")
    print("  The dark squares are numbered 1 to 32 and the number stays visible")
    print("  when the square is empty: moves can be read straight off the")
    print("  board. Choose by typing the row number in the list.")

    while not pos.is_terminal():
        show_board(pos, n, last, args.ascii)
        if pos.turn == human:
            moves = pos.legal_moves()
            for i, m in enumerate(moves):
                print(f"   {i:2d}) {describe(m)}")
            if len(moves) == 1:
                print("   (forced move)")
            choice = input("\n> ").strip().lower()

            if choice in ("q", "quit"):
                print("\nquitting.")
                return
            if choice in ("h", "hint"):
                print("   thinking...", end="", flush=True)
                root, dt = search(ev, pos, sims, rng)
                print(f"\r   in your place I would play ({dt:.1f}s):        ")
                print(summarize(root))
                continue
            if choice in ("u", "undo"):
                if len(history) < 2:
                    print("   not enough history to take back a move.")
                    continue
                pos, last = history[-2]   # undoes your move and the reply
                history = history[:-2]
                n -= 1
                continue
            if choice.startswith("s"):
                try:
                    sims = max(1, int(choice.split()[1]))
                    print(f"   engine simulations: {sims}")
                except (IndexError, ValueError):
                    print("   type for example:  s 200")
                continue

            try:
                k = int(choice)
                move = moves[k]
            except (ValueError, IndexError):
                print("   invalid number.")
                continue
            history.append((pos, last))
            pos = pos.play(move)
            last = move
        else:
            print(f"\n   the engine is thinking ({sims} simulations)...",
                  end="", flush=True)
            root, dt = search(ev, pos, sims, rng)
            move = max(root.children.items(), key=lambda kv: kv[1].N)[0]
            print(f"\r   the engine plays {describe(move)}"
                  f"   (it took {dt:.1f}s)" + " " * 12)
            print(summarize(root))
            history.append((pos, last))
            pos = pos.play(move)
            last = move
            n += 1

    show_board(pos, n, last, args.ascii)
    print(f"\n=== {outcome_text(pos)} ===")


# ============================ measurement against an anchor ===================
def play_vs_anchor(args):
    """Champion against a fixed opponent, alternating colors.

    The first plies are SAMPLED from the visits instead of played at best.
    Without that, all the games with the same colors would be the same game --
    both players are deterministic and there is no noise -- and playing twenty
    of them would carry the information of two. It is the same defect that made
    the intervals of the promotion arena too narrow.
    """
    ev = load_champion(args.weights, args.device)
    if args.vs_ab > 0:
        opp = AlphaBetaPlayer(depth=args.vs_ab)
    elif args.vs == "greedy":
        opp = GreedyPlayer()
    else:
        opp = RandomPlayer()

    print(f"[play] {args.games} games against {opp.name}, "
          f"{args.sims} simulations, alternating colors")
    if args.vs_ab >= 6:
        print("[play] note: deep alpha-beta is slow in Python, "
              "a game can take minutes")

    won = drawn = lost = 0
    for g in range(args.games):
        rng = np.random.default_rng(args.seed + g)
        net_white = (g % 2 == 0)
        pos, plies = Position(), 0
        while not pos.is_terminal() and plies < 300:
            if (pos.turn == WHITE) == net_white:
                root, _ = search(ev, pos, args.sims, rng)
                children = list(root.children.items())
                if plies < args.temp_plies:
                    w = np.array([st.N for _, st in children], dtype=np.float64)
                    move = children[int(rng.choice(len(children), p=w / w.sum()))][0]
                else:
                    move = max(children, key=lambda kv: kv[1].N)[0]
            else:
                move = opp.move(pos, rng)
            pos = pos.play(move)
            plies += 1

        r = pos.result() or 0
        for_net = r if net_white else -r
        won += for_net > 0
        drawn += for_net == 0
        lost += for_net < 0
        print(f"   game {g + 1:3d}/{args.games}: "
              f"{'won' if for_net > 0 else 'drawn' if for_net == 0 else 'lost':6s}"
              f"  ({plies} plies)   [{won}W {drawn}D {lost}L]")

    n = won + drawn + lost
    score = (won + 0.5 * drawn) / max(1, n)
    # Empirical variance, not binomial: draws contribute 0.5 without adding
    # spread, and assuming them binomial would inflate the interval.
    var = (won * (1 - score) ** 2 + drawn * (0.5 - score) ** 2
           + lost * score ** 2) / max(1, n)
    half = 1.96 * (var / max(1, n)) ** 0.5
    print(f"\n  {won} won, {drawn} drawn, {lost} lost out of {n}")
    print(f"  champion's score: {score:.3f} "
          f"[{score - half:.3f}, {score + half:.3f}]")
    if score - half > 0.5:
        print(f"  the champion is STRONGER than {opp.name}")
    elif score + half < 0.5:
        print(f"  the champion is weaker than {opp.name}")
    else:
        print(f"  indistinguishable from {opp.name} with this number of games")


def main():
    ap = argparse.ArgumentParser(
        description="play against the champion or measure its strength",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--weights", default=os.environ.get("DAMA_WEIGHTS_DIR", "weights"))
    ap.add_argument("--device", default=os.environ.get("DAMA_DEVICE", "cpu"))
    # 800 and not 400: against a human there is no hurry and the cost is less
    # than a second per move. The calibration measured that doubling the
    # simulations is worth about 150 Elo, so it is worth letting it think.
    ap.add_argument("--sims", type=int, default=800)
    ap.add_argument("--ascii", action="store_true",
                    help="no colors or round symbols, if the terminal "
                         "does not render them well")
    ap.add_argument("--side", default="white", choices=["white", "black"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--vs-ab", type=int, default=0, metavar="DEPTH",
                    help="measure against alpha-beta at this depth")
    ap.add_argument("--vs", default=None, choices=["random", "greedy"],
                    help="fixed opponent other than alpha-beta")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--temp-plies", type=int, default=10,
                    help="sampled opening plies, in measurement mode")
    args = ap.parse_args()
    args.ascii = setup_terminal(args.ascii)

    if args.vs_ab > 0 or args.vs is not None:
        play_vs_anchor(args)
    else:
        interactive(args)


if __name__ == "__main__":
    main()
