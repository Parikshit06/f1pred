# f1pred

Calibrated probabilities for Formula 1 qualifying, races and the championship.
Every forecast is written to `predictions/` with a timestamp before the session
runs, then graded against the result.

The predictions are not the point. The evaluation is: whether the probabilities
are honest, and whether the model beats the obvious answer.

**[Live forecast →](https://parikshit06.github.io/f1pred/)** ·
**[Method and accuracy →](https://parikshit06.github.io/f1pred/method.html)**

## Results

Walk-forward over **2024–2026: 62 races**, each predicted by a model trained
only on races before it. Settings fitted on **2022–23 only**, bounded at both
ends, so the tuner never saw the reported window.

| Approach | Top 5 | Podium | Winner | NDCG@5 | Log loss | Brier |
|---|---|---|---|---|---|---|
| Grid order | **3.92 / 5** | **2.07 / 3** | **58%** | **0.898** | 1.493 | 0.617 |
| **This model** | 3.86 / 5 | 1.97 / 3 | **58%** | 0.889 | **1.167** | **0.567** |
| Championship leader | 3.45 / 5 | 1.69 / 3 | 31% | 0.809 | 2.197 | 0.833 |
| Recent driver form | 3.40 / 5 | 1.48 / 3 | 27% | 0.789 | 2.129 | 0.880 |
| Team form | 3.07 / 5 | 1.45 / 3 | 31% | 0.766 | 2.322 | 0.890 |

**It matches the grid on winners and loses to it slightly on ordering.** It wins
on log loss and Brier, which measure whether a stated probability was true — a
grid says "pole wins" with no confidence attached, so when it is wrong it is
wrong absolutely.

That the grid is this strong is the actual finding: most of a race is decided on
Saturday.

| Season | Top 5 | Winner | Log loss | vs grid log loss | Races |
|---|---|---|---|---|---|
| 2024 | 4.04 | 45.8% | **1.576** | 1.720 | 24 |
| 2025 | 3.92 | 62.5% | **0.945** | 1.122 | 24 |
| 2026 | 3.43 | **71.4%** | **0.849** | 1.738 | 14 |

2026 is its best season; 2024 its worst, which is also the season with the most
one-sided car.

Not a fair fight against a betting market, and worth saying so: a closing price
aggregates people who know fuel loads, engine modes and whether a floor upgrade
worked. That gap is information, not technique. The largest closable piece is
Friday practice pace — already wired in, needs `make data-fastf1`.

## Is the championship projection honest?

Two separate claims, graded separately. `make title-backtest` and
`make calibrate-spread`.

**Naming the favourite.** 27 checkpoints across 7 completed seasons, Brier
**0.113**, favourite correct **85%** of the time.

| Claimed | Happened | n | 95% interval |
|---|---|---|---|
| 50–80% | 66.7% | 12 | 0.39–0.86 |
| over 95% | **100%** | 12 | **0.76–1.00** |

Twelve straight above 95% is consistent with a true rate as low as **76%**. So a
published 99.9% means *the arithmetic and the form both say this is over*, not a
calibrated one-in-a-thousand.

**The range, which was the real problem.** The projection publishes a 10th–90th
percentile band on every points total. Graded against real final constructors'
standings, it held **46%** of the time instead of 80%.

| Projection | Band held | Should be | Mean band width |
|---|---|---|---|
| Pace held fixed | 46% | 80% | 39 pts |
| **Current model** | **71%** | 80% | 90 pts |

The instinct is that this is about upgrades, and that is measurably wrong.
In-season development is real — fitted at **1.37 finishing positions** over a
full season (95% 0.90–1.83, 283 team-checkpoints, decaying to nothing by the
closing rounds), with the sampling noise in both means modelled rather than
counted as development. Adding exactly that much moved coverage 45% → 47%.

What was missing is duller: race-to-race luck averages out over a dozen races,
but *the model being wrong about a car today* does not — it rides into every
remaining race. Each simulated season now draws one pace offset per team, sized
by how much of the season is left. Swept on 2019–2022, graded on 2023–2025.

Still short of 80%, and the method page says so. The gain sits where the old
band was worst: a third of the way into a season, **25% → 90%**.

## Two bugs the evaluation caught

Neither raised an error. Both were found by chasing a forecast that looked wrong.

**398 finishes recorded as retirements.** The upstream status field is free
text. Until 2022 a lapped finisher reads `+1 Lap`; from 2023, `Lapped`. The
parser matched on the `+` prefix, so from 2023 it reclassified ordinary finishes
as retirements — concentrated in the midfield cars that get lapped. `make
validate` now cross-checks the flag against **laps completed**, a fact rather
than a label, so the next wording change fails loudly.

**Retirements averaged in as slowness.** A car that stops is classified near
last, and that went straight into rolling form. Reliability was already a
separate feature, so one event was charged twice — hardest on cars running at
the front when they stopped. Pace and reliability are now separate.

Together: log loss 1.281 → **1.167**, Brier 0.629 → **0.567**. The fitted
current-season weight fell from 4× to 1.5× — the 4× had been the model
compensating for corrupted form. *A settings value that moves a long way when a
bug is fixed is usually a symptom, not a finding.*

## What was tried and cut

Nine candidates measured, **eight cut**. Full numbers in the ledger at the top
of `features.py`.

- **Blending model rank with grid rank.** Looked like +0.09 top-five until the
  weight turned out to have been chosen on the test window. Re-fitted honestly,
  the gain vanished. Selection bias.
- **An unbounded tuner.** It refused to run unless the tuning season preceded
  the reported window — which looked airtight, and was not: no *end* season, so
  it ran straight through and fitted on the very races it then called
  out-of-sample. Bounded properly, honest log loss came out **worse** (1.254 →
  1.281). That gap is the size of the self-deception, and a test now enforces it.
- **Retirements split by cause**, with mechanical failures correlated inside a
  garage (fitted at 4.4× the independent rate). Identical to four decimal places
  — forced, since the split preserves each car's marginal retirement
  probability, so any metric scoring one driver at a time is blind to it. Graded
  on the constructors' band across 270 team-checkpoints: 50.0% either way.
- **Pit-crew speed.** Real spread (2026: Red Bull 0.49s a stop, Cadillac 3.24s).
  Better on two metrics, worse on three including log loss.
- **Circuit-character features**, **teammate-gap**, **team-switch**,
  **isotonic recalibration** — all neutral or worse.

What did work: `rank:ndcg` over `rank:pairwise` (NDCG@5 0.804 → 0.892);
explicit car-pace features (top-five 3.59 → 3.68); and re-fitting confidence
from a trailing 24-race window, so the model gets less certain when a season
stops being predictable.

## Is it favouring the famous drivers?

`make bias`. Raw per-driver error says it over-rates every front-runner —
Verstappen −1.90, Norris −1.98. That is an artifact: a fast driver who retires
is classified near last, dragging their *mean* below their true pace, so
comparing a fixed ranking against that mean manufactures a gradient down the
grid. It correlates 0.85 with predicted rank and explains **71% of the variance**.

Detrended, it inverts: **Hamilton +0.79 and Verstappen +0.65 are slightly
*under*-rated.** The real bias is Tsunoda (−2.30) and Pérez (−1.48) — quick
cars, drivers being beaten inside them. So it does conflate car with driver, by
about two places, in the midfield. At the sharp end residual bias is under one
position. Six tests assert the model is invariant to driver identity: rename
every driver and predictions must not move.

## Running it

```bash
make setup                            # creates .venv; uses uv if present, pip if not
make data-jolpica SEASONS=2018-2026   # ~15 min, rate-limited
make validate                         # data quality gate
make features
make predict                          # next unrun race
make dashboard                        # reports/index.html and method.html
```

`make which-python` prints the interpreter the targets use. No API keys — all
three sources are free and keyless.

Evidence regeneration is slower and deliberately separate:

```bash
make backtest START=2022
make title-backtest
make calibrate-spread
```

Quality gates: `make test` (138 tests), `make lint`, `make verify` (7 audits —
leakage, inputs, weighting, bias, accuracy, calibration, sanity).

`ci.yml` runs tests and lint on every push. `predict.yml` forecasts and deploys
to Pages on race weekends. `evaluate.yml` re-measures accuracy monthly and
commits the JSON the method page reads — kept out of the weekend pipeline so
published figures cannot drift from documented ones.

## How it works

```
jolpica ──┐
FastF1 ───┼─→ DuckDB (raw_*) ─→ validate ─→ features ─→ quali ranker
Open-Meteo┘                                    │             │
                                               │      predicted grid
                                               ↓             ↓
                                          race ranker ←──────┘
                                               │
                        Plackett–Luce ←────────┴────────→ Monte Carlo
                                     └──── blend ────┘
                                             ↓
                                    probabilities + championship
```

XGBoost ranker (`rank:ndcg`) scores the full field — one group per race, so each
race contributes every pairwise comparison inside it rather than one winner
label. Plackett–Luce converts scores to probabilities at a temperature fitted by
log loss. A 10,000-run Monte Carlo adds what a ranking cannot express: pace
variance, safety cars, retirements, and before qualifying an unknown grid. The
blend weight is fitted, not chosen.

Every rolling feature shifts by one race before aggregating, through one shared
helper. `tests/test_leakage.py` constructs cases where a leak would be visible
and asserts it is not.

## Data

| Source | Used for | Coverage |
|---|---|---|
| [jolpica-f1](https://github.com/jolpica/jolpica-f1) | results, grids, qualifying, sprints, pit stops, standings | 1950– |
| [FastF1](https://github.com/theOehrly/Fast-F1) | practice and qualifying pace, stints, compounds | 2018– |
| [Open-Meteo](https://open-meteo.com/) | race-day forecast and archive | — |

2018 is the floor: results go back to 1950, but without session timing those
seasons are not comparable.

jolpica is run by volunteers on donations, so the client is conservative — ~0.8
req/s against a documented 4/s, a sliding hourly window under their 500/hour,
backoff on 429, everything cached to disk. The hourly window is persisted
between runs, because the budget is per-IP and does not reset when your process
does.

## Layout

```
src/f1pred/
  config.py              paths, constants, fitted settings
  store.py               DuckDB schema and idempotent upserts
  ingest/                jolpica, FastF1, Open-Meteo
  validate.py            16 data quality checks
  features.py            feature engineering, leakage guards, rejected ledger
  model.py               XGBRanker wrappers
  simulate.py            Plackett–Luce and Monte Carlo
  backtest.py            walk-forward evaluation and hyperparameter fitting
  championship.py        season projection
  title_backtest.py      grades the title favourite
  spread_calibration.py  grades and re-fits the projection's range
  predict.py             one race, both models, logged output
  report.py              forecast page
  method_page.py         the evidence behind it
predictions/             timestamped forecasts, committed — the track record
reports/                 generated pages
```

## Known limits

About 80% of an F1 result is which car is fastest, and roughly one race in six
turns on something nothing knowable beforehand predicts. A well-built model
lands near 35–50% on winners in a dominant-car season and lower in a close one.
Anything claiming much more is leaking.

The 2026 regulation reset makes earlier seasons partially obsolete, handled by
weighting recent races more heavily — fitted, not assumed. Circuit features are
learned from prior visits, so a new track has none; the simulation falls back to
a neutral assumption and the page flags it.

Not modelled: weather, tyre strategy, in-race penalties, team orders, and which
team will improve. The method page lists these with what was tried against each.

## Licence

MIT. Data belongs to its sources; see jolpica's
[terms of use](https://github.com/jolpica/jolpica-f1/blob/main/TERMS.md).
