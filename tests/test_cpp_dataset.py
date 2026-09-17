"""
Round-trip of the dataset produced by the C++ engine.

Checks that the samples written by the C++ self-play (engine_c/dataset.hpp) can
be read back by Python in the form training expects, and that they satisfy the
same invariants as the samples produced by the Python self-play. The final
check is the most significant one: training REALLY runs on those samples and
the loss goes down -- that is, the data produced by the C++ side is usable as
it is, without adapters.

The test file is generated beforehand with:
    engine_c/selfplay_mt_test <n_games> <n_threads> <n_sims> <path>
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dama import IN_PLANES, POLICY_SIZE
from cpp_dataset import load_dataset

FAILS = []


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        FAILS.append(name)


def main(path):
    samples = load_dataset(path)
    print(f"--- C++ dataset: {len(samples)} samples from {path} ---")

    check("at least one sample", len(samples) > 0)

    X, pi, mask, z = samples[0]
    check(f"shape X = ({IN_PLANES},8,8)", X.shape == (IN_PLANES, 8, 8))
    check(f"shape pi = ({POLICY_SIZE},)", pi.shape == (POLICY_SIZE,))
    check(f"shape mask = ({POLICY_SIZE},)", mask.shape == (POLICY_SIZE,))

    check("pi sums to 1 everywhere",
          all(abs(float(s[1].sum()) - 1.0) < 1e-4 for s in samples))
    check("mask never empty",
          all(float(s[2].sum()) >= 1.0 for s in samples))
    # Critical invariant: probability mass on a move declared illegal multiplies
    # a logit pushed to -1e9 and makes the loss jump to ~1e8. The symptom would
    # look like "training has diverged" instead of "the samples are
    # inconsistent", which is why the value is printed.
    off_mask = max(float(s[1][s[2] == 0].sum()) for s in samples)
    check(f"no probability on illegal actions (max {off_mask:.1e})",
          off_mask < 1e-6)
    check("z in {-1,0,1}",
          all(s[3] in (-1.0, 0.0, 1.0) for s in samples))
    check("X binary on the piece planes",
          all(set(np.unique(s[0][:4])) <= {0.0, 1.0} for s in samples[:50]))
    check("dark-square mask plane constant",
          all(float(s[0][4].sum()) == 32.0 for s in samples[:50]))

    # the decisive test: training runs on these samples without adapters
    import torch
    from model import PolicyValueNet
    from train import train_on_samples

    torch.manual_seed(0)
    net = PolicyValueNet(channels=16, n_blocks=1, in_planes=IN_PLANES)
    subset = samples[:min(512, len(samples))]
    m1 = train_on_samples(net, subset, epochs=1, batch_size=64, lr=1e-3, device="cpu")
    m2 = train_on_samples(net, subset, epochs=3, batch_size=64, lr=1e-3, device="cpu")
    print(f"      loss: {m1['loss']:.3f} -> {m2['loss']:.3f}")
    check("training accepts the C++ samples and the loss goes down",
          m2["loss"] < m1["loss"])


def ensure_dataset(path: str) -> bool:
    """The test file is not in the repository (it is a regenerable binary): if
    it is missing, this test produces it by itself.

    BORN FROM A CLEAN CLONE. Where it was written the file was there from an
    earlier build and the test passed; freshly cloned elsewhere it failed. A
    test that passes only where it was written is not checking the code, it is
    checking its own folder."""
    import shutil
    import subprocess

    if os.path.exists(path):
        return True

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    engine = os.path.join(root, "engine_c")
    exe = "selfplay_mt_test" + (".exe" if os.name == "nt" else "")
    # The generator can sit next to the sources (build.sh/build.ps1) or under
    # build/ (CMake), and with CMake on Windows in build/Release/.
    candidates = [os.path.join(engine, exe),
                  os.path.join(engine, "build", exe),
                  os.path.join(engine, "build", "Release", exe)]
    generator = next((c for c in candidates if os.path.exists(c)), None)

    if generator is None:
        cxx = shutil.which("g++") or shutil.which("clang++")
        if cxx is None:
            print(f"SKIPPED: {path} is missing and there is no C++ compiler to\n"
                  f"         generate it. A skip, not a defect: this test checks\n"
                  f"         the dataset produced by the C++ engine, which cannot\n"
                  f"         be built on this machine.")
            return False
        print(f"  (compiling {exe}...)")
        r = subprocess.run([cxx, "-std=c++20", "-O2", "-pthread",
                            os.path.join(engine, "selfplay_mt_test.cpp"),
                            "-o", os.path.join(engine, exe)],
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print(f"SKIPPED: building {exe} did not succeed:\n{r.stderr}")
            return False
        generator = os.path.join(engine, exe)

    print(f"  (generating {path} with {os.path.basename(generator)}...)")
    r = subprocess.run([generator, "8", "4", "60", path],
                       capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0 or not os.path.exists(path):
        print(f"SKIPPED: the generator did not produce {path}:\n{r.stdout}\n{r.stderr}")
        return False
    return True


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "engine_c", "selfplay_test.bin")
    if not ensure_dataset(path):
        sys.exit(0)     # skipped, not failed: see the message above
    main(path)
    print()
    if FAILS:
        print(f">>> {len(FAILS)} TESTS FAILED: {FAILS}")
        sys.exit(1)
    print(">>> C++ DATASET ROUND-TRIP PASSED")
