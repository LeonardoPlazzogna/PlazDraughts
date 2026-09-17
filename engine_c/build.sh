#!/usr/bin/env bash
# Builds and validates the whole C++ engine. Linux/macOS.
# Requires a C++20 compiler (g++ or clang++) on the PATH.
#
# No -static here: on Linux dynamic linking against glibc/libstdc++ is the norm,
# and a fully static build needs static libraries that are not always
# installed. It is not needed anyway: these binaries are for this same machine
# or CI, not for redistribution.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CXX="${CXX:-g++}"

if ! command -v "$CXX" >/dev/null 2>&1; then
    echo "error: $CXX not found on the PATH (a C++20 compiler is needed)." >&2
    exit 1
fi
echo "compiler: $(command -v "$CXX")"

build_and_run() {
    local name="$1"
    shift
    echo
    echo "--- $name ---"
    "$CXX" -std=c++20 -O2 -march=native -pthread "$here/$name.cpp" -o "$here/$name"
    "$here/$name" "$@"
}

build_and_run perft
build_and_run encoder_parity
build_and_run mcts_selfplay_test
build_and_run threading_test 8 150 256
build_and_run selfplay_mt_test 6 4 50 "$here/selfplay_test.bin"
build_and_run arena_test 12 4 30

# main executable (not a test: it is only compiled)
echo
echo "--- dama_engine ---"
"$CXX" -std=c++20 -O2 -march=native -pthread "$here/main.cpp" -o "$here/dama_engine"
echo "built: $here/dama_engine (use --help for the options)"
