# Results: data behind the thesis figures

Everything here was extracted from the files the runs and the measurement tools
produced. No number was copied by hand. The figures in the thesis draw these
files directly with `pgfplots`, so the plots contain no transcribed numbers.

Run A is the 100-cycle run, run B the 200-cycle run. Section numbers (§) refer
to [`docs/results.md`](../docs/results.md).

```
results/
  series/           CSV series for the figures of chapter 8
  games/            the 50 games of each run's champion against Kingsrow
  learning_curve/   30 games against Kingsrow for six saved generations
  previews/         quick SVG renderings of some series, for checking them by eye
```

## `series/`

| file | figure | content |
|---|---|---|
| `runA_loss.csv`, `runB_loss.csv` | F8.1 | total, policy and value loss, per cycle |
| `runA_game_shape.csv`, `runB_game_shape.csv` | F8.2 / F8.3 | mean plies and draw rate, per cycle |
| `runA_outcomes.csv`, `runB_outcomes.csv` | | self-play draws, white and black wins, no-progress endings, per cycle |
| `runA_fixed_depth.csv` | T8.2 | score against alpha-beta at depths 4, 6 and 8 |
| `runA_vs_generations.csv`, `runB_vs_generations.csv` | F8.4 | the champion against its own frozen generations |
| `runA_ladder_elo.csv` | F8.5 | Elo estimated by the generation ladder |
| `runA_ladder_vs_direct.csv`, `runB_ladder_vs_direct.csv` | | ladder Elo against the direct measurement, with its interval |
| `runA_arena.csv`, `runB_arena.csv` | F8.9 | arena score per cycle, with the promotion outcome |
| `runA_arena_cost.csv`, `runB_arena_cost.csv` | | share and seconds of the arena per cycle, next to how close the match was |
| `runA_phase_share.csv`, `runB_phase_share.csv` | | percentage of the cycle spent in self-play, training, arena and evaluation |
| `batch_speed_strength.csv` | F8.7 / T8.9 | leaf batching: speed and strength |
| `legal_moves_distribution.csv` | F8.8 | distribution of the number of legal moves |
| `champion_vs_ab8.csv` | | the final champion against depth 8, game by game, with the quiet material balance |
| `runA_quiet_peak_kr.csv`, `runB_quiet_peak_kr.csv`, `runA_quiet_peak_ab8.csv` | | games sorted by peak quiet material advantage (§4.5) |

The three `*_quiet_peak_*` files are regenerated, byte for byte, by
`python tools/quiet_material.py --csv`: the first two from the games in
`games/`, the third from `champion_vs_ab8.csv`.

### Notes on individual files

**`runA_vs_generations.csv`** — generation 0 has a score of 1.000, hence an
infinite Elo: in the plot it must be left out or marked separately, otherwise
the axis blows up. The first six rows have 24 games, the last four have 48: the
resolution is not uniform and the error bars must be computed accordingly. The
same holds for `runB_vs_generations.csv`, which samples every twenty cycles.

**`batch_speed_strength.csv`** — the two quantities come from different
experiments but **at the same number of simulations per move** (400), so they
can share an axis. Speed was measured on the production machine with a graphics
card; strength from a comparison in which the two sides use different batch
sizes and the same network. Empty cells (`nan`) are combinations that were not
measured, not zeros: 4 and 16 leaves have no speed measurement on the
production machine, and 4 leaves has no distinct Elo value because it is
indistinguishable from 8.

**`runA_fixed_depth.csv`** — the depth-8 column has been **complete since
2026-09-02**: during the run it had a single point, because that measurement
costs many times the others and ran on a longer cadence, and it was completed by
replaying the frozen generations. Mind the uneven resolution: depths 4 and 6
have 60 games per point, depth 8 has 30. Cycle 60 is now 0.983 over 30 games,
no longer 0.975 over 20.

**`runA_arena.csv`** — `promoted` is 1 or 0. For the figure it is worth telling
the points apart with two symbols and drawing the parity line at 0.5: the
cluster just above that line is the result.

## `games/`

The games of the run A and run B champions against Kingsrow Italian 1.19e,
deliberately weakened (0.01 s per move, no opening book, 1 MB hash table): 50
games each, alternating colors, 800 simulations per move on our side (§4.1).
Written by `tools/kingsrow.py --save`, read back by `tools/read_games.py` and
`tools/quiet_material.py`. Squares are numbered 1..32 as in `play.py`.

## `learning_curve/`

The same protocol at six points of the 300-cycle run — cycles 0 (random
network, the negative control), 50, 100, 150, 200 and 300 — with 30 games each
(§4.2). `champion.pt` is the final champion of that 300-cycle run (96x6): not
the strongest network of the project, but a trained one, included so that the
tools can be tried without training first.

The champions of run A and run B, and five of their frozen generations, are
published too, as ONNX files under [`web/models/`](../web/models): they are what
the browser page plays with, and `web/networks.html` says what each one
measured. They are the same weights, converted by `tools/export_onnx.py`, which
checks every conversion against PyTorch before writing it.

## `previews/`

SVG renderings of the loss, game shape, arena, batching and legal-move series,
for a quick visual check of the data. They are not the thesis figures, which are
drawn from the CSVs.
