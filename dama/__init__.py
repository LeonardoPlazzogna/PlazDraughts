"""Italian draughts engine (the environment for AlphaZero)."""
from .board import (
    WHITE, BLACK, W_MAN, W_KING, B_MAN, B_KING, EMPTY,
    initial_board, render, rc, sq,
)
from .moves import Move, generate_legal_moves, apply_move
from .game import Position, NO_PROGRESS_DRAW
from .encoder import (
    encode, legal_actions, legal_mask, index_to_move,
    IN_PLANES, POLICY_SIZE,
)

__all__ = [
    "WHITE", "BLACK", "W_MAN", "W_KING", "B_MAN", "B_KING", "EMPTY",
    "initial_board", "render", "rc", "sq",
    "Move", "generate_legal_moves", "apply_move",
    "Position", "NO_PROGRESS_DRAW",
    "encode", "legal_actions", "legal_mask", "index_to_move",
    "IN_PLANES", "POLICY_SIZE",
]
