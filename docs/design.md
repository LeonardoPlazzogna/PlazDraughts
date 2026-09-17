# Design

How the pipeline is built, and the measured reason behind each choice. The
measurements themselves, with their conditions, are in [`results.md`](results.md);
how to run everything is in [`running.md`](running.md).

The project ports the AlphaZero method of
[PlazChess](https://github.com/LeonardoPlazzogna/PlazChess) to Italian draughts. The
principle is the same — self-play guided by MCTS and a policy-value network — but
the game engine was written from scratch for the Italian rules, and the
architecture was resized for a game with a much smaller space of states and
actions than chess.

## 1. The rules

Implemented in `dama/moves.py` and `dama/game.py`, and identically in
`engine_c/dama.hpp`:

- A **man** moves and captures **one** square diagonally, **forward only**.
- A **king** moves and captures **one** square along **all four** diagonals: kings do
  not fly.
- **Capturing is mandatory**, and a capture chain continues as long as possible.
- **A man cannot capture a king.**
- **Promotion**: a man that reaches the last rank becomes a king and the **turn
  ends**, even in the middle of a capture chain.
- **Capture priority**, in order: (1) the most pieces captured; (2) with equal
  numbers, capturing with a king; (3) the most kings captured; (4) the king captured
  earliest in the sequence.
- A player with no legal moves **loses** (there is no distinction between having no
  pieces and being blocked).
- **Draw** after 80 consecutive plies without a capture or a promotion. This is a
  simplification of the federation's 40-move rule, and it ends games earlier than
  the official rule would: [results §9](results.md#9-the-implemented-draw-rule-is-not-the-official-one).

The move generator is validated by perft up to depth 7, identical between Python, C++
and the browser port (`web/engine/moves.js`), and against Kingsrow Italian: the same
set of legal moves on 120 positions out of 120, and the same answer on five positions
that isolate each step of the capture priority (`tools/kingsrow.py --validate`,
`tools/rules_crosscheck.py`).

## 2. Representation

- **Board**: a list of 64 signed integers (sign = color, magnitude = man or king).
  Only the 32 dark squares are ever occupied.
- **Encoder** (`dama/encoder.py`): a `[7, 8, 8]` tensor, **canonicalized** — rotated
  by 180° with "ours/theirs" swapped when Black is to move, so the network always
  sees itself moving up the board. Four piece planes (our men, our kings, their men,
  their kings) plus three engineered ones: the dark-square mask, a mandatory-capture
  flag, and the no-progress counter divided by the draw threshold. That last plane
  means the draw rule is an input of the model, not only of the adjudication
  ([results §9](results.md#the-rule-is-wired-into-the-network-not-only-into-the-adjudication)).
- **Compact action space**: instead of `from(64) × to(64) = 4096` indices, almost all
  illegal in draughts, a move is encoded by its first step:
  `dark square (32) × direction (4) × mode (2: step or jump) = 256`. Different capture
  chains can share their first step; the network cannot tell them apart, so the
  probability of a shared index is split evenly among the moves that share it
  (`evaluators.priors_from_logits`).

## 3. The network

`model.py`: a residual trunk (96 channels and 6 blocks by default, 1.13 M
parameters) with optional Squeeze-and-Excitation blocks.

- **Policy head**: fully convolutional and flat — a 1×1 convolution down to 8
  channels, reordered into the 256 actions. No from/to factorization is needed with
  an action space this small.
- **Value head**: three logits, win/draw/loss (WDL), trained with cross-entropy
  instead of regressing a scalar, which calibrates draws better. The scalar used by
  the search is derived from them: `P(win) − P(loss)`, with an optional contempt term
  on the draw. There is no global pooling in the value head, so the spatial structure
  (advancement, last rank) is not averaged away.

With a fractional value target (see the value discount below), the target
distribution puts its mass on the draw class and on the class of the target's sign,
with expected value equal to the target; with an integer target it is one-hot
([results §5.2](results.md#52-the-value-head-ignored-everything-but-the-sign)).

## 4. The search

`mcts.py`, `engine_c/mcts.hpp` in C++, and `web/engine/mcts.js` in the browser:

- **Flat tree**: parallel lists indexed by node id instead of one object per node,
  and every node caches its own position, so a move is applied once per edge and not
  once per simulation.
- **Leaf batching with virtual loss**: one network evaluation per wave of simulations
  (8 leaves by default).
- **Transposition cache** within a single search. The key is the board and the side
  to move; it ignores the no-progress counter, in all three implementations.
- **Root noise**: Dirichlet with α = 1.5 mixed at ε = 0.25. Positions with a single
  legal move are only 9.5% of those played, so the noise almost always acts
  ([results §1](results.md#1-the-game-measured-properties-of-italian-draughts)).
- **No** subtree reuse between moves, no progressive widening, no early stopping:
  choices compatible with a branching factor of about six.

**Leaf batching is not a free performance knob.** More leaves per wave means fewer
occasions for the tree to react to its own results, and in a game this narrow the
search spreads over almost everything: against 8 leaves, 16 cost 163 Elo, 32 cost
243 and 64 cost 293, while 4 is indistinguishable from 8. The default sits on the
knee of that curve ([results §6](results.md#leaf-batching-costs-playing-strength-and-a-lot-of-it)).
The promotion arena cannot detect this, because both sides use the same value and
the degradation cancels out; the engine's `--mcts-batch-b` exists to measure it.

## 5. Self-play and samples

`selfplay.py`: each position of a game yields a sample `(X, π, mask, z)` — the
encoded position, the visit distribution of the search, the legal-action mask and
the final outcome from the point of view of the side to move. The first 12 plies are
sampled from the visits (temperature 1), the rest played greedily. A game that
reaches 300 plies is adjudicated by material, with a king worth three men.

**Value discount** (`DAMA_VALUE_DISCOUNT`, 0.99). A position *d* plies from the end
gets the outcome multiplied by γ^d. Without it, in a won position every move is worth
+1 and the search has no reason to prefer one; with four kings against one, all
fifteen legal moves were valued +1.000 and the game ended in a no-progress draw.
Measured on paired games, the discount cuts the back-and-forth king moves from 44% to
30% and the game length by forty plies, at no cost in strength
([results §2.3](results.md#23-discount-of-the-value-target)). It cannot recover a
drawn game: a discounted zero is still zero.

## 6. Training

`train.py`: masked policy cross-entropy (log-softmax over the legal actions only)
plus WDL cross-entropy, 2 epochs per cycle over a replay buffer of 600,000 samples,
about ten cycles of games.

Three changes proposed after the first serious run were each tested with a
controlled A/B comparison, and two of them did not go as expected:

**Learning-rate schedule — adopted.** Linear warmup and cosine decay within each
training call, plus a second cosine along the run on the cycle number
(`DAMA_LR_SCHEDULE`, `DAMA_LR_FINAL`). Over 8 cycles:

| | promotions | mean arena score | value loss | policy loss |
|---|---|---|---|---|
| off | 3/8 | 0.41 | 0.537 | 1.465 |
| **on** | **4/8** | **0.57** | **0.460** | **1.428** |

With the schedule the candidates are on average *better* than the champion; without
it they are on average worse. A consequence to keep in mind: the number of cycles
also sets the length of the cosine, so it cannot be raised after the start without
the learning rate jumping back up
([results §7](results.md#7-final-configuration)).

**Weight EMA — rejected, measured as harmful.** Over 6 cycles: 5 promotions without
EMA against 1 with it, mean arena score 0.63 against 0.48, and a loss that drops to
1.757 instead of staying near 2.0. Each cycle is a short fine-tuning that moves away
from the champion's weights in a definite direction; averaging a directional
trajectory lags behind it, so the candidate comes out weaker and the arena rejects
it. EMA helps when training oscillates around a minimum, as in the long single-phase
training of the chess project. It stays available with `DAMA_EMA=1`. Note that
`train_on_samples` itself defaults to EMA on, so tools that call it without the flag
train with it.

**Mirror-symmetry augmentation — impossible, not implemented.** Italian draughts has
no usable spatial symmetry: the left-right mirror sends dark squares onto light ones,
so mirrored positions cannot even be encoded; the 180° rotation keeps the squares but
reverses the direction of play, and is already the canonicalization; the transpose
turns forward moves into backward ones, illegal for men. Verified by enumeration.

**The replay window.** Frozen-buffer experiments show that more epochs mostly
memorize: from 2 to 30 epochs the divergence drops by 0.067 on seen data and 0.005 on
new data ([results §2.4](results.md#24-training-epochs-per-cycle)). That pointed to
the number of distinct positions, and the window went from 200,000 to 600,000
samples; tripling it moved the policy head by 0.003
([results §3.5](results.md#35-the-window-was-widened-nothing-moved)).

## 7. Promotion

After training, the candidate plays the champion in an **arena**. Both sides search
without root noise, so the first 10 plies are sampled from the visits
(`DAMA_ARENA_TEMP_PLIES`): without that, every game with the same colors is the same
game, and the old default of 4 plies left about 16 distinct games out of 400
([results §5.1](results.md#51-the-arena-counted-repetitions-as-independent-trials)).
The engine reports the number of distinct games and warns when it drops below half.

**The gate is a sequential probability ratio test** (`sprt.py`), not a fixed
threshold. A gate of 30 games that promotes at a score of 0.55 decides wrongly about
a third of the time: with 30 games the standard deviation of the score is about
0.091, and the threshold is only 0.55σ from an even score. The SPRT tests H0 (score
0.50) against H1 (score 0.55) with α = β = 0.05, using a variance estimated from the
observed wins, draws and losses — draws are frequent, and a binomial variance would
overstate the uncertainty. Measured by Monte Carlo simulation (2,000 matches, 35%
draws):

| gate | false promotions | correct promotions | games (equivalent / better candidate) |
|---|---|---|---|
| fixed threshold, 30 games | 28.8% | 79.5% | always 30 |
| SPRT, cap 30 | 22.5% | 72.6% | 30 / 29 |
| SPRT, cap 60 | 19.2% | 83.8% | 57 / 54 |
| SPRT, cap 100 | 13.2% | 89.3% | 94 / 81 |
| SPRT, cap 200 | 6.5% | 94.8% | 173 / 108 |

The SPRT's own advantage at equal budget is modest (22.5% against 28.8%); most of the
gain comes from playing more games where they are needed. Its real value is that it
**allocates** games: a clearly better candidate is decided in a handful of games, an
ambiguous one goes to the cap. The default cap is **400 games**
(`DAMA_SPRT_MAX_GAMES`); on the definitive run, 37 arenas out of 100 reached it
without deciding, and those undecided matches account for more than half of the arena
time ([results §6](results.md#performance)).

**The yardstick moves with the player.** The arena only guarantees that the candidate
beats the *current* champion, and a chain of promotions each just above parity can
wander instead of climbing: it happened
([results §4.6](results.md#46-when-the-three-measurements-contradict-each-other-the-case-of-cycle-60)).
An optional safeguard (`DAMA_ANTIDRIFT`, off by default) makes an approved candidate
also play a frozen generation a few ladder steps back, and refuses the promotion if it
regresses.

## 8. The search budget

**`c_puct`** (`DAMA_C_PUCT`, 1.5) is honored by every path: sequential and parallel
Python self-play, the C++ engine and the arena. Lowering it sharpens the target as
much as doubling the simulations — measured entropy of the visit distribution with a
material evaluator on midgame positions, against 1.824 for a uniform distribution:

| simulations | c_puct 1.5 | c_puct 0.5 |
|---|---|---|
| 400 | 1.709 | **1.559** |
| 1600 | 1.422 | 1.174 |

But a sharper target is not a stronger player. Measured by direct strength comparison
(`calibrate.py`), 1.0 loses against 1.5, while 2.0 and 3.0 are indistinguishable from
it ([results §2.1](results.md#21-exploration-constant-of-the-search)). The default
stays 1.5.

**Scheduled simulations** (`DAMA_SIMS_FINAL`, off by default). The self-play
simulations can grow linearly from `DAMA_SIMS` to `DAMA_SIMS_FINAL` along the run,
spending search where the value head is good enough to use it. The evaluation
simulations — arena, ladder, fixed opponents — stay fixed on purpose: if they changed
over time, the Elo curve would measure the search budget instead of the network.

**Is search the bottleneck?** The `q_spread_mean` column of `metrics.csv` — the gap
between the best and the worst root move value — read together with `entropy_mean`
separates two cases that otherwise look alike:

| `q_spread` | `entropy` | diagnosis | what to do |
|---|---|---|---|
| small (< ~0.1) | high | the value head does not tell the moves apart | the bottleneck is the network: more search will not concentrate anything |
| large (> ~0.3) | high | the values differ but visits stay spread | PUCT explores too much: lower `c_puct` |
| — | low | the search is already decided | nothing to do here |

**Why not Gumbel AlphaZero.** Its advantage is largest when the budget is scarce
relative to the number of actions (in Go, 16-50 simulations over ~250 legal moves).
Here there are about six legal moves and 400 simulations, some 70 per action — the
opposite regime, where sequential halving has little to allocate. The measured budget
curve agrees: doubling the simulations still gains about 90-150 Elo up to 3200
([results §2.2](results.md#22-search-budget--the-scaling-curve)).

## 9. Measuring progress

The arena says whether one step improves; it does not say how far the run has come.
Three other instruments do, each with its own limit.

**Fixed-depth alpha-beta** (`players.AlphaBetaPlayer`): deterministic and
reproducible. The evaluation is material (king = 3) plus a small advancement term,
with **quiescence** on forced captures — without it depth 2 stopped right after the
opponent's capture, missed the recapture, and measured *weaker* than a greedy player
(0.40). A **transposition table that records the bound type** (exact, lower, upper)
keeps the deeper levels affordable; a table that stored values found under pruning
without their bound type would return wrong values under a different window, and the
anchor would silently change behavior (`tests/test_metrics.py` checks that the chosen
moves are identical with and without it). Cost per move, measured on real games:

| depth | without table | with table | gain |
|---|---|---|---|
| 3 | 2.9 ms | 3.0 ms | 0.96× |
| 4 | 12.4 ms | 11.9 ms | 1.04× |
| 5 | 44.7 ms | 36.4 ms | 1.23× |
| 6 | 208 ms | **131 ms** | **1.58×** |

Each level beats the previous one, with narrowing margins (12 games per pair: depth 4
against 3 0.71, 5 against 4 0.67, 6 against 5 0.62). The conductor measures depths 4
and 6 every 20 cycles (`DAMA_AB_DEPTHS`, `DAMA_STRENGTH_EVERY`, 60 games) and depth 8
every 60 cycles (`DAMA_AB_DEEP_*`). The limit: a strong model beats the whole scale,
and more depth does not help, because the weak static evaluation is the limit, not
the horizon ([results §4.3](results.md#43-against-fixed-depth-opponents)).

**The generation ladder** (`ladder.py`): every `DAMA_GEN_EVERY` cycles (10) the
champion is frozen to `gen_NNN.pt` and plays the previous `DAMA_GEN_ANCHORS` (3)
generations, `DAMA_GEN_GAMES` (100) games each, giving a relative Elo that grows with
the player. Generation 0 is 0 by definition; extreme scores are capped at 0.99, that is
±798 Elo per comparison, and flagged as saturated in the log. The history lives in
`ladder.json` and survives interruptions. The limit: the estimates are **chained**,
and the error accumulates — against direct matches the ladder overestimated by a
factor of four ([results §4.7](results.md#the-internal-ladder-says-the-opposite-and-by-a-lot)).
The saved generations are also the best offline yardstick: `tools/anchors.py` replays
all of them against a fixed opponent without training anything.

**An external engine**: Kingsrow Italian, played through the CheckerBoard interface
by `tools/kingsrow.py`. It is the only absolute reference, and it is far stronger:
against it the differences between models compress into a handful of draws
([results §4.1](results.md#41-against-an-external-engine)). Well-known engines such
as Chinook and Cake play English draughts, where a man can capture a king and there
is no capture priority; only an engine with an Italian variant plays the same game,
and the rules cross-check verifies that it does.

**Per-cycle metrics** (`weights/metrics.csv`, `metrics.py`): one row per cycle,
appended, surviving interruptions like the rest of the state.

| group | columns |
|---|---|
| cycle | `cycle`, `timestamp`, `duration_s` |
| data | `samples`, `buffer`, `sims` |
| training | `loss`, `loss_policy`, `loss_value` |
| arena | `arena_score`, `promoted` |
| games | `games`, `plies_mean`, `plies_max`, `white_wins`, `black_wins`, `draws`, `draw_rate`, `truncated`, `no_progress`, `captures_mean`, `promotions_mean` |
| diagnostics | `entropy_mean` (entropy of the visit distribution), `q_spread_mean`, `value_mae` (gap between the search value and the final outcome) |
| strength | `elo` (ladder), `strength` (e.g. `ab4=1.00;ab6=0.98`) |
| phase times | `t_selfplay`, `t_train`, `t_arena`, `t_eval` |

The game columns are there to diagnose a run that does not improve: if most games end
by the no-progress rule, the value head gets little signal, and without these numbers
that looks like a training problem. They are filled identically by the Python and the
C++ paths. Game length is the one signal that never saturates — and it says nothing
about strength ([results §4.4](results.md#game-length-is-the-only-signal-that-never-saturates)).

## 10. Two execution paths

**Python** (`parallel.py`, `inference_server.py`). Python self-play is bound by the
GIL, so a persistent pool of worker processes plays games in parallel, each with
PyTorch limited to one thread; the pool lives for the whole run and picks up new
weights from the file's modification time. On a GPU, one **inference server** process
owns the card and the network, and merges the requests of all the workers into one
batch (up to `DAMA_SERVER_BATCH`). The server receives only encoded planes and returns
only logits and values; the priors are built by the same function on both sides, so
the two paths give identical priors by construction, which
`tests/test_inference_server.py` checks. On the CPU the server path is about twice as
slow, since interprocess communication costs and there is nothing to gain; it is
enabled automatically only when the device is not the CPU (`DAMA_INFERENCE_SERVER`).

**C++** (`engine_c/`, `cpp_engine.py`). The same idea without the process boundary:
the workers are threads in one process, an evaluation request is a pointer and the
answer a write to shared memory, so the network gets large batches without IPC. Each
cycle the conductor exports the champion to TorchScript, runs the engine for self-play
and arena, reads its dataset back into the replay buffer and trains in PyTorch as
before. The engine uses the same search parameters as the Python self-play (c_puct
1.5, temperature for 12 plies, Dirichlet 1.5/0.25, 300 plies at most), so the buffer
stays homogeneous if the two paths alternate. **If the engine fails, the cycle falls
back on the Python path** instead of stopping the run. With the engine active the
Python pool and server are not started at all. The C++ backend is checked against
PyTorch by `parity_check` (maximum difference 0.000e+00); see
[`engine_c/README.md`](../engine_c/README.md).

Samples are bit-for-bit identical on the same machine whatever the number of engine
threads, and equivalent within tolerance across machines and compilers
([results §6](results.md#portability)).

## 11. Robustness

- **Resume.** The state — champion weights, replay buffer, last completed cycle — is
  written atomically at every cycle; on restart the conductor resumes from the next
  cycle. At most the cycle in progress is lost.
- **One run per folder.** `runlock.py` takes an operating-system lock on the weights
  folder, so a second run on the same folder is refused, and a run killed outright
  releases it (`fcntl` on Linux, `msvcrt` on Windows).
- **Launchers that refuse a wrong configuration.** A run once went on the CPU with an
  idle GPU next to it, and the conductor only wrote that in the log. `run.sh` and
  `run.ps1` check the device with a real allocation, whether the engine can load a
  network and knows the options the conductor passes, that the dry-run mode is off,
  and whether a previous state will be resumed — before starting
  ([`running.md`](running.md#3-launching)).
- **Decoding the engine's output** uses an explicit encoding and replaces unreadable
  characters: a single stray byte once made the arena score disappear without any
  error ([results §5.7](results.md#57-a-fault-that-leaves-no-trace-the-arena-outcome-read-and-lost)).
