"""
Tests of the SPRT (sprt.py).

The check that matters is not that the formulas are written correctly, but that
the test REALLY has the statistical properties it promises. It is verified by
Monte Carlo simulation: thousands of matches are generated between opponents of
KNOWN strength and the real error rates are measured, comparing them with the
declared ones and with the fixed-threshold gate the SPRT replaces.
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                          # noqa: E402

import sprt                                                 # noqa: E402

ok = True


def check(cond, msg):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    if not cond:
        ok = False


def simulate_match(p_win, p_draw, rng, max_games, chunk=4,
                   s0=0.50, s1=0.55, alpha=0.05, beta=0.05, promote_min=0.55):
    """A complete match with early stopping. Returns (promoted, games)."""
    w = d = l = 0
    while w + d + l < max_games:
        for _ in range(chunk):
            u = rng.random()
            if u < p_win:
                w += 1
            elif u < p_win + p_draw:
                d += 1
            else:
                l += 1
        res = sprt.evaluate(w, d, l, s0, s1, alpha, beta)
        if res.decision != "continue":
            return res.decision == "H1", w + d + l
    return sprt.decide(w, d, l, promote_min, s0, s1, alpha, beta), w + d + l


def fixed_gate(p_win, p_draw, rng, n_games, promote_min=0.55):
    """The classic gate: n fixed games, fixed threshold."""
    w = d = l = 0
    for _ in range(n_games):
        u = rng.random()
        if u < p_win:
            w += 1
        elif u < p_win + p_draw:
            d += 1
        else:
            l += 1
    return (w + 0.5 * d) / n_games >= promote_min


def test_llr_basics():
    print("--- basic properties of the LLR ---")
    # for the same number of games, a better score must give a greater LLR
    a = sprt.llr(10, 10, 10)
    b = sprt.llr(15, 10, 5)
    check(b > a, f"better score -> greater LLR ({a:.2f} -> {b:.2f})")
    check(sprt.llr(0, 0, 0) == 0.0, "no games -> zero LLR")

    lo, up = sprt.bounds(0.05, 0.05)
    check(lo < 0 < up, f"bounds of opposite sign ({lo:.2f}, {up:.2f})")
    check(abs(up - 2.944) < 0.01, f"upper bound ~2.944 for alpha=0.05 ({up:.3f})")

    # all draws: no evidence of superiority, it must not promote
    r = sprt.evaluate(0, 40, 0)
    check(r.decision != "H1", f"40 draws -> does not promote (verdict {r.decision})")


def test_error_rates():
    print("--- real error rates (Monte Carlo simulation) ---")
    rng = np.random.default_rng(12345)
    N = 3000
    p_draw = 0.35          # realistic draw rate for this project

    # H0 TRUE: equivalent networks. Every promotion is a false positive.
    p_win = (1.0 - p_draw) / 2
    sprt_fp = sum(simulate_match(p_win, p_draw, rng, 200)[0] for _ in range(N)) / N
    rng2 = np.random.default_rng(12345)
    fixed_fp = sum(fixed_gate(p_win, p_draw, rng2, 30) for _ in range(N)) / N
    print(f"      equivalent networks -> spurious promotions: "
          f"SPRT {sprt_fp*100:.1f}%   fixed threshold {fixed_fp*100:.1f}%")
    check(sprt_fp < fixed_fp,
          f"SPRT errs less than the fixed gate ({sprt_fp*100:.1f}% < {fixed_fp*100:.1f}%)")
    check(sprt_fp <= 0.10, f"false positives within 10% ({sprt_fp*100:.1f}%)")

    # H1 TRUE: genuinely better candidate (score 0.60).
    score = 0.60
    p_win = score - p_draw / 2
    rng3 = np.random.default_rng(999)
    sprt_tp = sum(simulate_match(p_win, p_draw, rng3, 200)[0] for _ in range(N)) / N
    rng4 = np.random.default_rng(999)
    fixed_tp = sum(fixed_gate(p_win, p_draw, rng4, 30) for _ in range(N)) / N
    print(f"      better candidate    -> correct promotions: "
          f"SPRT {sprt_tp*100:.1f}%   fixed threshold {fixed_tp*100:.1f}%")
    check(sprt_tp > fixed_tp,
          f"SPRT recognizes more real improvements ({sprt_tp*100:.1f}% > {fixed_tp*100:.1f}%)")


def test_game_economy():
    print("--- games spent ---")
    rng = np.random.default_rng(7)
    p_draw = 0.35
    for label, score in (("equivalent", 0.50), ("better (0.65)", 0.65)):
        p_win = score - p_draw / 2
        games = [simulate_match(p_win, p_draw, rng, 200)[1] for _ in range(1000)]
        print(f"      {label:>16}: mean {np.mean(games):5.1f} games "
              f"(median {np.median(games):.0f}, max {max(games)})")
    # a clearly superior candidate must be recognized quickly
    p_win = 0.80 - p_draw / 2
    games = [simulate_match(p_win, p_draw, rng, 200)[1] for _ in range(500)]
    check(np.mean(games) < 30,
          f"clearly better candidate decided in <30 games (mean {np.mean(games):.1f})")


def test_score_ci():
    """The interval printed by calibrate.py and tools/compare.py.

    It used to be copied in both, with two different explanations; it lives in
    sprt.py now, so it is tested where every other statistic of this project is.
    """
    print("--- score and confidence interval ---")
    from sprt import score_ci

    s, ci = score_ci(0, 40, 0)
    check(abs(s - 0.5) < 1e-12, f"forty draws score exactly 0.5 ({s:.3f})")
    # The point of the regularization: a unanimous sample must NOT come back
    # with an interval of zero width.
    check(ci > 0.0, f"forty identical games still carry uncertainty (+-{ci:.3f})")

    s, ci = score_ci(10, 0, 0)
    check(abs(s - 1.0) < 1e-12 and ci > 0.0,
          f"ten wins score 1.000 and keep an interval (+-{ci:.3f})")

    # Distinct games are what the interval may be divided by: claiming more
    # than there are is how a difference that does not exist gets declared.
    _, wide = score_ci(17, 47, 36, n_eff=16)
    _, narrow = score_ci(17, 47, 36)
    check(wide > narrow,
          f"fewer distinct games widen the interval ({wide:.3f} > {narrow:.3f})")

    check(score_ci(0, 0, 0) == (0.0, 0.0), "no games: no score, no interval")


def main():
    test_llr_basics()
    test_error_rates()
    test_game_economy()
    test_score_ci()
    print("\n>>> " + ("SPRT OK" if ok else "SPRT FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
