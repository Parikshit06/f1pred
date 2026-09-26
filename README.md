# f1pred

Probabilistic forecasts for Formula 1 qualifying, races and the championship.
Each forecast is committed to `predictions/` before its session and graded
against the result.

**[Live forecast →](https://parikshit06.github.io/f1pred/)** ·
**[Method and accuracy →](https://parikshit06.github.io/f1pred/method.html)**

## Accuracy

Walk-forward over **2024–2026, 62 races**: each race is predicted by a model
trained only on races before it. Settings were fitted on 2022–23 and never on
the reported window.

| Approach | Top 5 | Podium | Winner | NDCG@5 | Log loss | Brier |
|---|---|---|---|---|---|---|
| Grid order | **3.92** | **2.06** | 58% | **0.898** | 1.493 | 0.617 |
| **This model** | 3.87 | 1.97 | **61%** | 0.894 | **1.164** | **0.566** |
| Championship leader | 3.45 | 1.69 | 31% | 0.809 | 2.197 | 0.833 |
| Recent driver form | 3.40 | 1.48 | 27% | 0.789 | 2.129 | 0.880 |
| Team form | 3.06 | 1.45 | 31% | 0.766 | 2.322 | 0.890 |

The starting grid is the baseline that matters, because it predicts a race well
with no model at all. The model calls more winners than the grid (38 of 62
against 36) and its probabilities are much better (log loss 1.164 against
1.493). It is slightly behind the grid on the rest of the order: most of a race
is decided in qualifying.

| Season | Top 5 | Winner | Log loss | Grid log loss | Races |
|---|---|---|---|---|---|
| 2024 | 4.08 | 50.0% | **1.577** | 1.720 | 24 |
| 2025 | 3.92 | 66.7% | **0.918** | 1.122 | 24 |
| 2026 | 3.43 | 71.4% | **0.876** | 1.738 | 14 |

Winner rates on 14–24 races move several points per race, so read the log loss
column across seasons rather than the winner column.

## Championship projection

**Who wins.** Graded at 27 checkpoints across seven completed seasons: the
favourite was right 78% of the time, Brier 0.114. Above 95% it is 12 for 12,
but the 95% interval on 12 from 12 runs down to 0.76, so a published 99.9%
means the maths says it's over, not one chance in a thousand. The six misses
are 2020's Bottas–Hamilton, two checkpoints of 2021, and three of 2025's
Piastri–Norris.

**How close.** Every points total comes with a 10th–90th percentile range,
graded against real final constructors' standings on seasons the calibration
never saw:

| Projection | Range held | Target | Width |
|---|---|---|---|
| Race luck only | 54% | 80% | 38 pts |
| **Current** | **82%** | 80% | 108 pts |

Race luck averages out over a season; error in the model's read of a car
doesn't. Each simulated season draws one pace offset per team, sized by
`make calibrate-spread` (swept on 2019–22, graded on 2023–25).

## Bugs that moved the numbers

- **Lapped finishers stored as retirements.** From 2023 jolpica writes a lapped
  car as `Lapped` instead of `+1 Lap`, and the parser matched on the `+`. 398
  finishes were flagged as retirements and fed the form and reliability
  features. `validate` now checks the flag against laps completed, and CI
  repairs flags before validating. The repair alone took winners from 34 to 36.
- **Retirements counted as slowness.** A car that stops is classified near
  last. Averaged into recent form, a broken gearbox read as a slow driver on
  top of counting against reliability. Form now uses finished races only.
- **Empty grid after qualifying.** The grid comes from race results, so between
  qualifying and the race it was blank, and the simulator ran on nothing. Win
  odds still looked plausible because they're blended with the closed form;
  podium and points were noise. The grid now comes from qualifying, and the
  simulator refuses a blank one.
- **Season projection scored from one race.** It reused the next race's scores
  — that weekend's grid and track — for every remaining round, so one
  qualifying session reshaped nine races. It now scores each driver on a
  typical weekend. At the same setting, coverage went from 70% to 77% with a
  slightly narrower range.

## Tried and cut

Measured on the same walk-forward and dropped: teammate race gap, time at
current team, form at circuits of similar character, blending in the team's
best recent grid slot, retirements split by cause, and pit-stop speed. None
cleared the noise on 62 races. Numbers are at the top of `features.py`.

Kept: `rank:ndcg` over `rank:pairwise` (NDCG@5 0.804 → 0.892), car-pace features
from rolling qualifying gaps, and a temperature refitted on a trailing 24-race
window so confidence drops when a season gets less predictable.

## Driver bias

`make bias` writes `reports/bias.json`. Raw per-driver error says every
front-runner is over-rated, but that's an artefact: a fast driver who retires is
classified near last, dragging their average down. It correlates 0.832 with
predicted rank and accounts for 69% of the variance. With that removed,
Hamilton (+0.75) and Verstappen (+0.68) are slightly under-rated, and the
largest biases are in the midfield: Tsunoda −2.27, Bottas −2.09, Pérez −1.62.
Tests check that predictions don't change when drivers are renamed.

## Run it

```bash
make setup                            # .venv, via uv if installed
make data-jolpica SEASONS=2018-2026   # ~15 min, rate-limited
make validate
make features
make predict
make dashboard
```

`make data-fastf1 SEASONS=2024-2026` is optional: slow, a few hundred MB a
season, and it only feeds practice pace to the qualifying model. No API keys
are needed.

`make backtest`, `make title-backtest`, `make calibrate-spread` and `make bias`
regenerate `reports/*.json`. `make test`, `make lint` and `make verify` are the
gates; `tests/test_readme_matches_reports.py` fails if a figure here drifts from
those files.

CI runs tests and lint on every push. `predict.yml` forecasts and deploys to
Pages hourly across race weekends; `evaluate.yml` re-measures accuracy monthly.

## How it works

```
jolpica ──┐
FastF1 ───┼─→ DuckDB ─→ validate ─→ features ─→ quali ranker
Open-Meteo┘                            │             │
                                       │      predicted grid
                                       ↓             ↓
                                  race ranker ←──────┘
                                       │
                Plackett–Luce ←────────┴────────→ Monte Carlo
                             └──── blend ────┘
                                     ↓
                            probabilities + championship
```

An XGBoost ranker scores the whole field, one group per race. Plackett–Luce
turns scores into win probabilities; a 10,000-run Monte Carlo adds pace
variance, safety cars, retirements and, before qualifying, an unknown grid.
Every rolling feature shifts back one race before aggregating, and
`tests/test_leakage.py` checks that no feature can see its own race.

```
src/f1pred/
  cli.py                 commands
  config.py              paths and fitted constants
  store.py               DuckDB schema and upserts
  http_cache.py          rate-limited, cached HTTP
  ingest/                jolpica, FastF1, Open-Meteo
  validate.py            data checks
  features.py            features, leakage guards, what was cut
  model.py               XGBoost rankers
  simulate.py            Plackett–Luce and Monte Carlo
  predict.py             forecast one race and log it
  backtest.py            walk-forward evaluation and tuning
  championship.py        season projection
  title_backtest.py      grades the title favourite
  spread_calibration.py  sizes the projection's range
  diagnostics.py         per-driver bias
  verify.py              audits behind make verify
  report.py              forecast page content
  report_render.py       CSS, SVG, colour
  method_page.py         method and accuracy page
predictions/             logged forecasts
reports/                 measured figures
```

## Data

[jolpica-f1](https://github.com/jolpica/jolpica-f1) for results, qualifying,
sprints, pit stops and standings; [FastF1](https://github.com/theOehrly/Fast-F1)
for session pace; [Open-Meteo](https://open-meteo.com/) for weather. Seasons
start at 2018, where FastF1 timing begins.

jolpica is volunteer-run, so the client stays well inside its limits: about 0.8
requests a second, a persisted hourly budget under 500, backoff on 429, and
everything cached.

## Limits

Not modelled: weather, tyre strategy, grid penalties, in-race penalties, team
orders, and which team will improve. The method page covers each.

## Licence

MIT. Data belongs to its sources; jolpica's data is CC BY-NC-SA 4.0, see their
[terms](https://github.com/jolpica/jolpica-f1/blob/main/TERMS.md).
