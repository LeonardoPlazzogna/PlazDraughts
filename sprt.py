"""
Sequential probability ratio test (SPRT) for the arena promotion gate.

WHY. A gate that plays a FIXED number of games (30) and promotes when the score
exceeds a fixed threshold (0.55) is mostly noise. With 30 games the standard
deviation of the score under the null hypothesis is ~0.091, and the threshold is
only 0.55 sigma away from an even score. As a result:

    ~29% of EQUIVALENT networks get promoted by chance
    ~29% of REAL improvements (p=0.60) get rejected

that is, about a third of the decisions are noise. Over 100 cycles this injects
a random walk into the weight trajectory: networks that are not better get
promoted and real improvements get thrown away.

HOW. The SPRT (Wald) tests two competing hypotheses, accumulating evidence game
by game, and stops as soon as one of them is supported well enough:

    H0: the candidate is NOT better   (expected score s0, typically 0.50)
    H1: the candidate IS better       (expected score s1, typically 0.55)

Compared with a fixed number of games, type I and type II errors are CONTROLLED
by construction (alpha, beta), and on average FEWER games are needed -- clear
cases are decided early, and the budget is spent only where the result is
uncertain.

MODEL. The normal approximation on the score (GSPRT), with the variance
estimated from the observed win/draw/loss counts instead of assumed. That
matters in this game: draws are frequent (25-65% measured), and treating them
as half wins with a binomial variance would overestimate the uncertainty,
making the matches needlessly long.
"""
from __future__ import annotations
import math
from dataclasses import dataclass


@dataclass
class SprtResult:
    decision: str        # "H1" (promote), "H0" (reject), "continue"
    llr: float           # current log-likelihood ratio
    lower: float         # lower bound (accept H0)
    upper: float         # upper bound (accept H1)
    games: int
    score: float         # observed score in [0,1]


def bounds(alpha: float = 0.05, beta: float = 0.05) -> tuple[float, float]:
    """Wald bounds for the type I (alpha) and type II (beta) errors."""
    lower = math.log(beta / (1.0 - alpha))
    upper = math.log((1.0 - beta) / alpha)
    return lower, upper


def llr(wins: int, draws: int, losses: int,
        s0: float = 0.50, s1: float = 0.55) -> float:
    """Log-likelihood ratio between H1 (score s1) and H0 (score s0).

    Normal approximation with an EMPIRICAL variance: the spread of the score is
    estimated from the observed results, not assumed binomial. With many draws
    the real variance is much lower than the binomial one, and using it shortens
    the matches for the same guarantees.
    """
    n = wins + draws + losses
    if n == 0:
        return 0.0

    score = (wins + 0.5 * draws) / n

    # REGULARIZED VARIANCE, and the regularization is not a detail.
    #
    # Computing the variance on the raw counts and replacing a zero -- all
    # outcomes equal -- with 1e-6 "so as not to claim infinite certainty from a
    # degenerate sample" does exactly the opposite: with a typical per-game
    # variance around 0.1, that floor inflates the likelihood ratio a HUNDRED
    # THOUSAND times. Measured on that earlier version: a single win gave an
    # LLR of 23750 against a bound of 2.94, i.e. promotion after one game; a
    # single draw gave -1250, i.e. rejection after one game.
    #
    # The remedy is the classic one: half a fictitious outcome is added to each
    # category before estimating the spread. A unanimous sample stops looking
    # free of uncertainty, and the larger the sample the less the correction
    # weighs. With ten draws the verdict goes back to "continue" instead of a
    # rejection, which is the right answer: ten draws say practically nothing
    # about which side is stronger.
    e = 0.5
    nr = n + 3.0 * e
    sr = ((wins + e) + 0.5 * (draws + e)) / nr
    var = ((wins + e) * (1.0 - sr) ** 2 +
           (draws + e) * (0.5 - sr) ** 2 +
           (losses + e) * (0.0 - sr) ** 2) / nr

    # LLR of the normal approximation between two hypothesized means. The
    # regularized score is used here too, for consistency with the variance:
    # mixing a raw mean with a corrected spread would give sharper conclusions
    # than the two pieces together justify.
    return n * (s1 - s0) * (sr - 0.5 * (s0 + s1)) / var


def evaluate(wins: int, draws: int, losses: int,
             s0: float = 0.50, s1: float = 0.55,
             alpha: float = 0.05, beta: float = 0.05) -> SprtResult:
    """Current SPRT verdict on the counts accumulated so far."""
    lo, up = bounds(alpha, beta)
    value = llr(wins, draws, losses, s0, s1)
    n = wins + draws + losses
    score = (wins + 0.5 * draws) / n if n else 0.0

    if value >= up:
        decision = "H1"
    elif value <= lo:
        decision = "H0"
    else:
        decision = "continue"
    return SprtResult(decision, value, lo, up, n, score)


def decide(wins: int, draws: int, losses: int, promote_min: float,
           s0: float = 0.50, s1: float = 0.55,
           alpha: float = 0.05, beta: float = 0.05) -> bool:
    """FINAL promotion decision, even if the SPRT has not concluded.

    If the test has closed, its verdict is followed. If the maximum game budget
    ran out without a conclusion -- frequent when the two networks really are
    equivalent -- it falls back to the classic threshold, which is still a
    reasonable decision at that point.
    """
    res = evaluate(wins, draws, losses, s0, s1, alpha, beta)
    if res.decision == "H1":
        return True
    if res.decision == "H0":
        return False
    return res.score >= promote_min
def score_ci(w: int, d: int, l: int, n_eff: int | None = None) -> tuple[float, float]:
    """Score of a match and the half-width of its 95% interval.

    It lives here, and not in the two tools that print it, because it was
    written twice: the same arithmetic with two different explanations, which is
    how two copies start answering differently.

    The variance is EMPIRICAL, computed on the observed win/draw/loss counts
    rather than assumed binomial. Draws are a large share of the games in this
    game and each contributes exactly 0.5 without adding spread, so a binomial
    variance would inflate the interval by about a third and call
    "indistinguishable" a comparison that has in fact concluded.

    It is REGULARIZED with half a fictitious outcome per category. On raw counts
    a unanimous sample -- all draws, or all wins -- has zero variance and
    therefore an interval of exactly zero width: forty identical games would be
    declared a certain measurement. The reported score stays the observed one,
    because that is the data; only the uncertainty is corrected, and it cannot
    vanish.

    `n_eff` is the number of DIFFERENT games, which must be told apart from the
    number played: without noise and with deterministic networks, two games that
    start the same stay the same to the end, so the second adds nothing. Dividing
    by the games played instead of the distinct ones is exactly how a difference
    that does not exist gets declared significant.
    """
    n = w + d + l
    if n == 0:
        return 0.0, 0.0
    s = (w + 0.5 * d) / n
    e = 0.5
    nr = n + 3.0 * e
    sr = ((w + e) + 0.5 * (d + e)) / nr
    var = ((w + e) * (1.0 - sr) ** 2 + (d + e) * (0.5 - sr) ** 2 +
           (l + e) * sr ** 2) / nr
    return s, 1.96 * math.sqrt(var / max(1, min(n_eff or n, n)))
