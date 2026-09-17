// perft: counts the sequences of legal moves up to depth d from the starting
// position. It VALIDATES the C++ move generator against the Python reference
// engine. Expected values (identical to dama/moves.py): 7, 49, 302, 1469, 7361,
// 36473, 177532.
#include "dama.hpp"
#include <cstdio>
#include <cstdlib>
#include <chrono>

using namespace dama;

long long perft(const Board& b, int color, int depth) {
    if (depth == 0) return 1;
    std::vector<Move> moves = legal_moves(b, color);
    if (depth == 1) return (long long)moves.size();
    long long total = 0;
    for (const Move& m : moves)
        total += perft(apply_move(b, m, color), -color, depth - 1);
    return total;
}

int main(int argc, char** argv) {
    int max_depth = (argc > 1) ? std::atoi(argv[1]) : 7;
    // Python reference (dama/moves.py) for the starting position
    const long long ref[] = {1, 7, 49, 302, 1469, 7361, 36473, 177532};
    const int n_ref = sizeof(ref) / sizeof(ref[0]);

    Board b = initial_board();
    bool all_ok = true;
    for (int d = 1; d <= max_depth; d++) {
        auto t0 = std::chrono::steady_clock::now();
        long long nodes = perft(b, WHITE, d);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (d < n_ref) {
            bool ok = (nodes == ref[d]);
            all_ok &= ok;
            printf("  perft(%d) = %-10lld  expected %-10lld  [%s]  %8.2f ms\n",
                   d, nodes, ref[d], ok ? "OK" : "WRONG", ms);
        } else {
            printf("  perft(%d) = %-10lld  (no reference)            %8.2f ms\n",
                   d, nodes, ms);
        }
    }
    printf("\n%s\n", all_ok ? "ALL PERFT COUNTS MATCH THE PYTHON REFERENCE."
                            : "DISCREPANCY: the C++ move generator does NOT match the Python one.");
    return all_ok ? 0 : 1;
}
