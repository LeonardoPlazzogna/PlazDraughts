// Validates engine_c/encoder.hpp for numerical parity against the Python
// reference (dama/encoder.py), on the same principle as perft: the same test
// cases, the same checksum (sum and index-weighted sum of the 448 plane values)
// computed ONCE with the Python reference and compared here. The expected
// values below were generated with dama.encoder.encode() on the very same
// cases.
#include "dama.hpp"
#include "encoder.hpp"
#include "position.hpp"
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <string>
#include <tuple>

using namespace dama;

struct Case {
    const char* label;
    Board board;
    int turn;
    int no_progress;
    double expect_sum, expect_wsum;
};

static Board make_empty() {
    Board b{};
    b.fill(EMPTY);
    return b;
}

static Board make_synth() {
    Board b = make_empty();
    b[sq(4, 3)] = W_MAN;    // can capture the B_MAN below, if White is to move
    b[sq(3, 2)] = B_MAN;
    b[sq(5, 4)] = W_KING;
    b[sq(2, 3)] = B_KING;
    return b;
}

int main() {
    Case cases[] = {
        {"start, White, np=0",       initial_board(), WHITE, 0,  56.0,    11492.0},
        {"start, Black, np=0",       initial_board(), BLACK, 0,  56.0,    11492.0},
        {"synthetic, White, np=20",  make_synth(),     WHITE, 20, 116.0,  38852.0},
        {"synthetic, Black, np=20",  make_synth(),     BLACK, 20, 52.0,   16360.0},
    };

    bool all_ok = true;
    for (auto& c : cases) {
        Planes p = encode(c.board, c.turn, c.no_progress, /*no_progress_draw=*/80);
        double sum = 0.0, wsum = 0.0;
        for (size_t i = 0; i < p.size(); i++) { sum += p[i]; wsum += p[i] * (double)i; }

        bool ok = (std::abs(sum - c.expect_sum) < 1e-4) &&
                  (std::abs(wsum - c.expect_wsum) < 1e-4);
        all_ok &= ok;
        printf("  %-28s sum=%10.4f (expected %10.4f)  wsum=%12.4f (expected %12.4f)  [%s]\n",
               c.label, sum, c.expect_sum, wsum, c.expect_wsum, ok ? "OK" : "WRONG");
    }

    // --- deterministic playout: checks the whole chain move generation ->
    // apply_move -> encode on REAL positions (not hand-built), choosing the
    // move by value (min frm, then min to) so that Python and C++ always pick
    // the same one even if the internal order of the move generator differed.
    // Expected values generated and verified with the Python reference.
    struct PlayoutCase { int ply; double expect_sum, expect_wsum; };
    PlayoutCase playout_cases[] = {
        {5,  55.799999,  11659.400391},
        {10, 53.799999,  11616.400391},
        {15, 118.399994, 34448.199219},
        {20, 114.000000, 33712.000000},
    };
    Position pos;
    int case_idx = 0;
    for (int ply = 0; ply < 20 && case_idx < 4; ply++) {
        if (pos.is_terminal()) break;
        const auto& moves = pos.legal_moves();
        auto best = std::min_element(moves.begin(), moves.end(),
            [](const Move& a, const Move& b) {
                return std::tie(a.frm, a.to) < std::tie(b.frm, b.to);
            });
        pos = pos.play(*best);
        if (ply + 1 == playout_cases[case_idx].ply) {
            Planes p = encode(pos.board, pos.turn, pos.no_progress, NO_PROGRESS_DRAW);
            double sum = 0.0, wsum = 0.0;
            for (size_t i = 0; i < p.size(); i++) { sum += p[i]; wsum += p[i] * (double)i; }
            auto& pc = playout_cases[case_idx];
            bool ok = (std::abs(sum - pc.expect_sum) < 1e-3) &&
                      (std::abs(wsum - pc.expect_wsum) < 1e-2);
            all_ok &= ok;
            printf("  %-28s sum=%10.4f (expected %10.4f)  wsum=%12.4f (expected %12.4f)  [%s]\n",
                   (std::string("playout ply=") + std::to_string(pc.ply)).c_str(),
                   sum, pc.expect_sum, wsum, pc.expect_wsum, ok ? "OK" : "WRONG");
            case_idx++;
        }
    }

    printf("\n%s\n", all_ok ? "ALL ENCODER CHECKSUMS MATCH THE PYTHON REFERENCE."
                            : "DISCREPANCY: the C++ encoder does NOT match the Python one.");
    return all_ok ? 0 : 1;
}
