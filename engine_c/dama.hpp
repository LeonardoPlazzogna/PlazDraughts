// Italian draughts game logic in C++ (a faithful port of dama/{board,moves,game}.py).
// Correct and fast rules with no external dependencies; the MCTS, the batched
// inference hub and the LibTorch backend are built on top of it.
#pragma once
#include <cstdint>
#include <vector>
#include <array>
#include <algorithm>

namespace dama {

// pieces (sign = color, magnitude = type): 0 empty, +-1 man, +-2 king
constexpr int8_t EMPTY = 0, W_MAN = 1, W_KING = 2, B_MAN = -1, B_KING = -2;
constexpr int WHITE = 1, BLACK = -1;

using Board = std::array<int8_t, 64>;

inline int rrow(int s) { return s >> 3; }
inline int ccol(int s) { return s & 7; }
inline int sq(int r, int c) { return r * 8 + c; }
inline bool on_board(int r, int c) { return r >= 0 && r < 8 && c >= 0 && c < 8; }
inline int color_of(int8_t p) { return (p > 0) - (p < 0); }
inline bool is_king(int8_t p) { return p == 2 || p == -2; }
inline bool is_man(int8_t p) { return p == 1 || p == -1; }
inline int promo_row(int color) { return color == WHITE ? 0 : 7; }
inline bool is_dark_square(int s) { return ((s / 8 + s % 8) & 1) == 1; }

// the 32 dark (playable) squares in index order, and the map square -> compact
// index 0..31 (-1 on the light squares): used by the compact action and encoding.
inline const std::array<int, 32>& dark_squares() {
    static const std::array<int, 32> ds = [] {
        std::array<int, 32> a{};
        int idx = 0;
        for (int s = 0; s < 64; s++)
            if (is_dark_square(s)) a[idx++] = s;
        return a;
    }();
    return ds;
}

inline const std::array<int, 64>& dark_index() {
    static const std::array<int, 64> di = [] {
        std::array<int, 64> a{};
        a.fill(-1);
        const auto& ds = dark_squares();
        for (int i = 0; i < 32; i++) a[ds[i]] = i;
        return a;
    }();
    return di;
}

constexpr int DIRS[4][2] = {{-1, -1}, {-1, 1}, {1, -1}, {1, 1}};

struct Move {
    int frm, to;
    int8_t captured[12];   // captured squares, in order
    int n_cap = 0;
    bool promotes = false;
    bool by_king = false;
    int kings_cap = 0;     // kings captured (for capture priority)
    int first_king = 999;  // index of the first captured king
    int first_land = -1;   // landing square of the FIRST step (jump or simple
                            // step): the encoder needs it for the compact move
                            // index, which depends on the first step only
    bool is_capture() const { return n_cap > 0; }
};

inline Board initial_board() {
    Board b{};
    for (int r = 0; r < 8; r++)
        for (int c = 0; c < 8; c++) {
            if (((r + c) & 1) != 1) continue;      // dark squares only
            if (r <= 2) b[sq(r, c)] = B_MAN;
            else if (r >= 5) b[sq(r, c)] = W_MAN;
        }
    return b;
}

// capture/move directions: king = 4 diagonals, man = 2 forward
inline int move_dirs(int8_t piece, int color, int out[4][2]) {
    if (is_king(piece)) {
        for (int i = 0; i < 4; i++) { out[i][0] = DIRS[i][0]; out[i][1] = DIRS[i][1]; }
        return 4;
    }
    int dr = (color == WHITE) ? -1 : 1;   // man: forward only
    out[0][0] = dr; out[0][1] = -1;
    out[1][0] = dr; out[1][1] = 1;
    return 2;
}

inline void emit(const Board& orig, std::vector<Move>& out, int frm, int to,
                 const int8_t* cap, int ncap, bool promotes, bool by_king,
                 int first_land) {
    Move m;
    m.frm = frm; m.to = to; m.n_cap = ncap;
    m.promotes = promotes; m.by_king = by_king;
    m.first_land = (first_land >= 0) ? first_land : to;  // simple move: a single step
    for (int i = 0; i < ncap; i++) {
        m.captured[i] = cap[i];
        if (is_king(orig[cap[i]])) {
            m.kings_cap++;
            if (m.first_king == 999) m.first_king = i;
        }
    }
    out.push_back(m);
}

// Capture recursion: emits only MAXIMAL chains into `out`. `work` has the origin
// lifted; jumped pieces stay on the board (they block) and are listed in `cap`
// (they cannot be jumped twice).
inline void captures_from(const Board& orig, Board& work, int cur, int8_t piece,
                          int color, int8_t* cap, int ncap, int origin, bool by_king,
                          int first_land, std::vector<Move>& out) {
    int r0 = rrow(cur), c0 = ccol(cur);
    int dirs[4][2];
    int nd = move_dirs(piece, color, dirs);
    bool extended = false;
    for (int d = 0; d < nd; d++) {
        int rm = r0 + dirs[d][0], cm = c0 + dirs[d][1];
        int rl = r0 + 2 * dirs[d][0], cl = c0 + 2 * dirs[d][1];
        if (!on_board(rl, cl)) continue;
        int mid = sq(rm, cm), land = sq(rl, cl);
        int8_t target = work[mid];
        if (color_of(target) != -color) continue;          // must be an enemy
        bool already = false;
        for (int i = 0; i < ncap; i++) if (cap[i] == mid) { already = true; break; }
        if (already) continue;                              // already captured
        if (is_man(piece) && is_king(target)) continue;     // a man does not capture a king
        if (work[land] != EMPTY) continue;                  // landing square free

        // if this is the first jump of the chain, its landing square is the
        // "first_land" that identifies the move in the compact index
        int this_first_land = (ncap == 0) ? land : first_land;

        cap[ncap] = (int8_t)mid;
        if (is_man(piece) && rl == promo_row(color)) {      // promotion -> STOP
            emit(orig, out, origin, land, cap, ncap + 1, true, by_king, this_first_land);
            extended = true;
            continue;
        }
        extended = true;
        captures_from(orig, work, land, piece, color, cap, ncap + 1, origin, by_king,
                      this_first_land, out);
    }
    if (!extended && ncap > 0)                              // chain complete
        emit(orig, out, origin, cur, cap, ncap, false, by_king, first_land);
}

inline void gen_captures(const Board& b, int color, std::vector<Move>& out) {
    for (int s = 0; s < 64; s++) {
        int8_t piece = b[s];
        if (color_of(piece) != color) continue;
        Board work = b; work[s] = EMPTY;    // the moving piece is "in hand"
        int8_t cap[12];
        captures_from(b, work, s, piece, color, cap, 0, s, is_king(piece), -1, out);
    }
}

// true if `color` has at least one capture available (used for the "capture
// available" plane of the encoder). Faithful port of moves.py:has_capture: it
// checks a single jump per piece and returns at the first one found -- much
// cheaper than generating every chain with gen_captures, which would run at
// every encode(), i.e. at every leaf evaluated by the MCTS (a version based on
// gen_captures was a real performance regression against the Python reference).
inline bool has_capture(const Board& b, int color) {
    for (int s = 0; s < 64; s++) {
        int8_t piece = b[s];
        if (color_of(piece) != color) continue;
        int r0 = rrow(s), c0 = ccol(s);
        int dirs[4][2];
        int nd = move_dirs(piece, color, dirs);
        for (int d = 0; d < nd; d++) {
            int rl = r0 + 2 * dirs[d][0], cl = c0 + 2 * dirs[d][1];
            if (!on_board(rl, cl)) continue;
            int8_t target = b[sq(r0 + dirs[d][0], c0 + dirs[d][1])];
            if (color_of(target) != -color) continue;
            if (is_man(piece) && is_king(target)) continue;
            if (b[sq(rl, cl)] != EMPTY) continue;
            return true;
        }
    }
    return false;
}

inline void gen_simple(const Board& b, int color, std::vector<Move>& out) {
    for (int s = 0; s < 64; s++) {
        int8_t piece = b[s];
        if (color_of(piece) != color) continue;
        int r0 = rrow(s), c0 = ccol(s);
        int dirs[4][2];
        int nd = move_dirs(piece, color, dirs);
        for (int d = 0; d < nd; d++) {
            int r1 = r0 + dirs[d][0], c1 = c0 + dirs[d][1];
            if (!on_board(r1, c1)) continue;
            int dd = sq(r1, c1);
            if (b[dd] != EMPTY) continue;
            Move m;
            m.frm = s; m.to = dd; m.n_cap = 0;
            m.promotes = is_man(piece) && r1 == promo_row(color);
            m.by_king = is_king(piece);
            m.first_land = dd;
            out.push_back(m);
        }
    }
}

// Capture priority (Italian rules 1-4)
inline void apply_priority(std::vector<Move>& c) {
    int mx = 0; for (auto& m : c) mx = std::max(mx, m.n_cap);
    std::vector<Move> f;
    for (auto& m : c) if (m.n_cap == mx) f.push_back(m);           // 1: most pieces
    bool anyk = false; for (auto& m : f) if (m.by_king) { anyk = true; break; }
    if (anyk) { std::vector<Move> g; for (auto& m : f) if (m.by_king) g.push_back(m); f.swap(g); }  // 2
    int mk = 0; for (auto& m : f) mk = std::max(mk, m.kings_cap);
    { std::vector<Move> g; for (auto& m : f) if (m.kings_cap == mk) g.push_back(m); f.swap(g); }    // 3
    if (mk > 0) {                                                   // 4: king earliest
        int fk = 999; for (auto& m : f) fk = std::min(fk, m.first_king);
        std::vector<Move> g; for (auto& m : f) if (m.first_king == fk) g.push_back(m); f.swap(g);
    }
    c.swap(f);
}

inline std::vector<Move> legal_moves(const Board& b, int color) {
    std::vector<Move> caps;
    gen_captures(b, color, caps);
    if (!caps.empty()) { apply_priority(caps); return caps; }
    std::vector<Move> simple;
    gen_simple(b, color, simple);
    return simple;
}

inline Board apply_move(const Board& b, const Move& m, int color) {
    Board nb = b;
    int8_t piece = nb[m.frm];
    nb[m.frm] = EMPTY;
    for (int i = 0; i < m.n_cap; i++) nb[m.captured[i]] = EMPTY;
    nb[m.to] = m.promotes ? (int8_t)(2 * color) : piece;
    return nb;
}

}  // namespace dama
