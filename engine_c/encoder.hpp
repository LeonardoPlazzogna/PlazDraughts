// Position -> tensor encoder for Italian draughts. Faithful port of
// dama/encoder.py. No LibTorch dependency: it produces a flat [7,8,8] buffer in
// row-major order (32-bit floats, like the numpy planes), which the LibTorch
// backend wraps with torch::from_blob without copying.
#pragma once
#include "dama.hpp"
#include <array>

namespace dama {

constexpr int ENC_IN_PLANES = 7;
constexpr int ENC_N_DIRS = 4;
constexpr int ENC_N_MODES = 2;
constexpr int ENC_SLOTS = ENC_N_DIRS * ENC_N_MODES;              // 8 slots per square
constexpr int ENC_POLICY_SIZE = 32 * ENC_SLOTS;                  // 256

using Planes = std::array<float, ENC_IN_PLANES * 8 * 8>;

// same fixed order as the Python: DIR_LIST = ((-1,-1),(-1,1),(1,-1),(1,1))
constexpr int ENC_DIR_LIST[4][2] = {{-1, -1}, {-1, 1}, {1, -1}, {1, 1}};

inline int enc_dir_to_idx(int dr, int dc) {
    for (int i = 0; i < 4; i++)
        if (ENC_DIR_LIST[i][0] == dr && ENC_DIR_LIST[i][1] == dc) return i;
    return -1;  // should never happen for a valid move
}

inline int canon_sq(int s, bool flip) { return flip ? (63 - s) : s; }

inline float& plane_at(Planes& p, int plane, int r, int c) {
    return p[(plane * 8 + r) * 8 + c];
}

// Position -> [7,8,8] tensor canonicalized to the point of view of the side to
// move. Planes: 0 our men, 1 our kings, 2 their men, 3 their kings, 4 dark-square
// mask, 5 capture available (broadcast), 6 normalized no-progress counter
// (broadcast). Same semantics as dama/encoder.py:encode().
inline Planes encode(const Board& board, int turn, int no_progress,
                     int no_progress_draw) {
    Planes planes{};  // zero-init
    bool flip = (turn == BLACK);

    for (int s = 0; s < 64; s++) {
        int8_t p = board[s];
        if (p == EMPTY) continue;
        int cs = canon_sq(s, flip);
        int r = cs / 8, c = cs % 8;
        int rel = color_of(p) * turn;   // +1 = ours, -1 = opponent's
        bool king = is_king(p);
        if (rel == 1) plane_at(planes, king ? 1 : 0, r, c) = 1.0f;
        else          plane_at(planes, king ? 3 : 2, r, c) = 1.0f;
    }

    // plane 4: geometric mask of the dark squares. Invariant under the
    // 180-degree rotation (the row+column parity does not change), so it can be
    // written directly in canonical coordinates.
    for (int r = 0; r < 8; r++)
        for (int c = 0; c < 8; c++)
            plane_at(planes, 4, r, c) = ((r + c) & 1) ? 1.0f : 0.0f;

    // plane 5: capture available (broadcast flag)
    if (has_capture(board, turn)) {
        for (int r = 0; r < 8; r++)
            for (int c = 0; c < 8; c++)
                plane_at(planes, 5, r, c) = 1.0f;
    }

    // plane 6: no-progress counter normalized to [0,1] (broadcast)
    float np = static_cast<float>(no_progress) / static_cast<float>(no_progress_draw);
    if (np > 1.0f) np = 1.0f;
    for (int r = 0; r < 8; r++)
        for (int c = 0; c < 8; c++)
            plane_at(planes, 6, r, c) = np;

    return planes;
}

// Compact action index (canonical frame): square*8 + direction*2 + mode, from
// the FIRST step of the move (`first_land`, added to Move in dama.hpp for this
// purpose). Same formula as dama/encoder.py:move_policy_index().
inline int move_policy_index(const Move& move, bool flip) {
    int cf = canon_sq(move.frm, flip);
    int c1 = canon_sq(move.first_land, flip);
    int r0 = cf / 8, col0 = cf % 8;
    int r1 = c1 / 8, col1 = c1 % 8;
    int dr = (r1 > r0) - (r1 < r0);
    int dc = (col1 > col0) - (col1 < col0);
    int di = enc_dir_to_idx(dr, dc);
    int mode = move.is_capture() ? 1 : 0;
    return dark_index()[cf] * ENC_SLOTS + di * ENC_N_MODES + mode;
}

}  // namespace dama
