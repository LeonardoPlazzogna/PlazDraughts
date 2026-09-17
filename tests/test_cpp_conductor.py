"""
Integration test of the C++ engine <-> conductor bridge (cpp_engine.py).

It covers what can be checked WITHOUT LibTorch:
  1. the bridge runs the engine, reads its dataset back and returns samples in
     the form training expects (with the fake backend, net=None);
  2. those samples are usable by the real Python training;
  3. the bridge raises CppEngineError when the engine fails, which is what lets
     the conductor fall back on the Python self-play instead of stopping the run.

It does NOT cover (it needs LibTorch + a GPU, see engine_c/README.md) the path in
which the engine really loads a network. There only the backend behind the hub
changes, not the code exercised here.
"""
from __future__ import annotations
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                    # noqa: E402
import cpp_engine                                     # noqa: E402
from dama import IN_PLANES, POLICY_SIZE               # noqa: E402
from model import PolicyValueNet                      # noqa: E402
from train import train_on_samples                    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NAME = "dama_engine.exe" if os.name == "nt" else "dama_engine"


def _find_engine():
    """The engine to test, looked for where the project really puts it.

    Looking only in engine_c/ is not enough: the LibTorch build of build.ps1
    puts it in engine_c/build/. Where both paths exist, the test can end up
    testing the OTHER binary -- and if that one is a leftover built with g++, on
    Windows it dies with 0xC0000135 (missing DLL) without printing anything,
    while the real engine next to it works: a red test beside phases that ran
    green on a different executable.

    The order is the order of intent: if someone named an engine it is tested,
    otherwise the built one, otherwise it falls back.
    """
    chosen = os.environ.get("DAMA_CPP_ENGINE", "")
    if chosen and os.path.exists(chosen):
        return os.path.abspath(chosen)
    for c in (os.path.join(ROOT, "engine_c", "build", _NAME),
              os.path.join(ROOT, "engine_c", _NAME)):
        if os.path.exists(c):
            return c
    # Neither exists: it is built as a fallback, into engine_c/build/ because
    # that is where the rest of the project looks for it.
    return os.path.join(ROOT, "engine_c", "build", _NAME)


ENGINE = _find_engine()

ok = True


def check(cond, msg):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    if not cond:
        ok = False


def build_engine_if_needed():
    if os.path.exists(ENGINE):
        return True
    src = os.path.join(ROOT, "engine_c", "main.cpp")
    os.makedirs(os.path.dirname(ENGINE), exist_ok=True)
    print("  (building dama_engine...)")
    r = subprocess.run(["g++", "-std=c++20", "-O2", "-pthread", src, "-o", ENGINE],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  build failed:\n{r.stderr}")
        return False
    return True


def test_bridge_roundtrip():
    print("--- bridge cpp_engine -> dataset -> samples ---")
    with tempfile.TemporaryDirectory() as td:
        samples, out = cpp_engine.selfplay(
            ENGINE, None,                     # None = fake backend inside the engine
            n_games=6, n_sims=30, n_threads=4, device="cpu",
            mcts_batch=8, max_batch=64, seed=7, workdir=td)

    check(len(samples) > 0, f"samples produced ({len(samples)})")
    X, pi, mask, z = samples[0]
    check(X.shape == (IN_PLANES, 8, 8), f"shape of X = {X.shape}")
    check(pi.shape == (POLICY_SIZE,), f"shape of pi = {pi.shape}")
    check(mask.shape == (POLICY_SIZE,), f"shape of mask = {mask.shape}")
    check(all(abs(float(s[1].sum()) - 1.0) < 1e-4 for s in samples),
          "pi sums to 1 in every sample")
    check(all(float(s[2].sum()) > 0 for s in samples), "mask never empty")
    check(all(s[3] in (-1.0, 0.0, 1.0) for s in samples), "z in {-1,0,1}")
    check(all(not np.any(s[1][s[2] == 0] > 0) for s in samples),
          "no probability on illegal actions")
    check("games/hour" in out, "the engine reports its throughput")
    return samples


def test_training_accepts(samples):
    print("--- Python training accepts the C++ engine's samples ---")
    net = PolicyValueNet(channels=16, n_blocks=1, in_planes=IN_PLANES)
    m1 = train_on_samples(net, samples, epochs=1, batch_size=64, lr=1e-3)
    m2 = train_on_samples(net, samples, epochs=3, batch_size=64, lr=1e-3)
    check(np.isfinite(m1["loss"]), f"finite loss ({m1['loss']:.3f})")
    print(f"      loss: {m1['loss']:.3f} -> {m2['loss']:.3f}")
    check(m2["loss"] < m1["loss"], "the loss decreases")


def test_fallback_on_failure():
    # No "fail" in lines printed on success: preflight.ps1 flags any output
    # matching it, and PowerShell's -match ignores case.
    print("--- CppEngineError when the engine breaks ---")
    try:
        cpp_engine.selfplay("/nonexistent/path/dama_engine", None,
                            n_games=1, n_sims=10, n_threads=1, device="cpu")
        check(False, "missing executable -> CppEngineError")
    except cpp_engine.CppEngineError:
        check(True, "missing executable -> CppEngineError")

    # real engine but invalid arguments: it must fail cleanly
    try:
        cpp_engine.selfplay(ENGINE, None, n_games=1, n_sims=10, n_threads=1,
                            device="nonexistent_device")
        check(False, "invalid device -> CppEngineError")
    except cpp_engine.CppEngineError as e:
        check("code" in str(e) or "produce" in str(e),
              "invalid device -> CppEngineError with diagnostics")


def test_arena_bridge():
    print("--- arena bridge: score and game diversity ---")
    with tempfile.TemporaryDirectory() as td:
        score, wdl, out = cpp_engine.arena(
            ENGINE, None, None,           # fake backend for both sides
            n_games=12, n_sims=30, n_threads=4, device="cpu",
            mcts_batch=8, max_batch=64, seed=99, workdir=td)
    check(0.0 <= score <= 1.0, f"score in [0,1] ({score:.3f})")
    check("ARENA_SCORE" in out, "the engine reports ARENA_SCORE")

    # The counts feed the SPRT: they must be consistent with the score and add
    # up to the number of games requested.
    w, d, l = wdl
    check(w + d + l == 12, f"counts add up to the games ({w}+{d}+{l})")
    check(abs((w + 0.5 * d) / 12 - score) < 1e-6,
          "counts consistent with the reported score")

    # With two identical sides the score must stay near 0.5: if the color
    # assignment were wrong it would collapse to 0 or 1.
    check(abs(score - 0.5) <= 0.3, "score ~0.5 with identical backends")

    # Different seeds must be able to give different results: it is the proof
    # that the arena games are NOT all the same game (the defect found in the
    # Python arena, where the seed had no effect).
    with tempfile.TemporaryDirectory() as td:
        scores = [cpp_engine.arena(ENGINE, None, None, n_games=4, n_sims=30,
                                   n_threads=2, device="cpu", seed=s,
                                   workdir=td)[0]
                  for s in (1, 2, 3, 4, 5, 6)]
    check(len(set(scores)) > 1,
          f"different seeds -> different results (observed: {sorted(set(scores))})")


def main():
    if not build_engine_if_needed():
        print("\n>>> CANNOT BUILD THE ENGINE: tests skipped")
        sys.exit(2)
    samples = test_bridge_roundtrip()
    test_training_accepts(samples)
    test_arena_bridge()
    test_fallback_on_failure()
    print("\n>>> " + ("C++/CONDUCTOR INTEGRATION OK" if ok
                      else "C++/CONDUCTOR INTEGRATION FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
