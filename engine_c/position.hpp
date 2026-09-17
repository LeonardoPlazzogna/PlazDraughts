// Game state, transitions and outcome. Faithful port of dama/game.py.
#pragma once
#include "dama.hpp"
#include <optional>
#include <vector>
#include <utility>
#include <functional>

namespace dama {

// Consecutive plies without a capture or a promotion after which the game is
// drawn (a simplification that differs from the official 40-move rule: see
// dama/game.py).
constexpr int NO_PROGRESS_DRAW = 80;

class Position {
public:
    Board board;
    int turn;
    int no_progress;

    Position() : board(initial_board()), turn(WHITE), no_progress(0) {}
    Position(Board b, int t, int np) : board(b), turn(t), no_progress(np) {}

    // Memoized: the (expensive) move generation runs once per state.
    const std::vector<Move>& legal_moves() const {
        if (!legal_computed_) {
            legal_cache_ = dama::legal_moves(board, turn);
            legal_computed_ = true;
        }
        return legal_cache_;
    }

    bool is_terminal() const {
        if (no_progress >= NO_PROGRESS_DRAW) return true;
        return legal_moves().empty();
    }

    // outcome from WHITE's point of view: +1 White wins, -1 Black wins,
    // 0 draw; nullopt if the game is not over. The side to move with no moves
    // has lost.
    std::optional<int> result() const {
        if (no_progress >= NO_PROGRESS_DRAW) return 0;
        if (legal_moves().empty()) return -turn;
        return std::nullopt;
    }

    float terminal_value(int perspective) const {
        auto r = result();
        if (!r) return 0.0f;
        return static_cast<float>(*r * perspective);
    }

    // new Position after `move` (this one is not modified; no push/pop,
    // unlike chess -- see chapter 1 of the thesis)
    Position play(const Move& m) const {
        Board nb = apply_move(board, m, turn);
        bool progressed = m.is_capture() || m.promotes;
        int npg = progressed ? 0 : no_progress + 1;
        return Position(nb, -turn, npg);
    }

private:
    mutable std::vector<Move> legal_cache_;
    mutable bool legal_computed_ = false;
};

// Key for caches/transpositions: exact equality on the content (board + side to
// move), not an incremental Zobrist hash -- the same choice as game.py:key().
// Like the Python key it ignores the no-progress counter.
struct PositionKey {
    Board board;
    int turn;
    bool operator==(const PositionKey& o) const {
        return turn == o.turn && board == o.board;
    }
};

struct PositionKeyHash {
    size_t operator()(const PositionKey& k) const {
        size_t h = std::hash<int>()(k.turn);
        for (int8_t v : k.board)
            h = h * 1000003u ^ std::hash<int>()(static_cast<int>(v));
        return h;
    }
};

inline PositionKey key_of(const Position& pos) { return {pos.board, pos.turn}; }

}  // namespace dama
