# f1pred

Probabilistic forecasts for Formula 1 qualifying, races and the championship.
Every forecast is timestamped into `predictions/` before the session runs, then
graded against the result.

Race probabilities are calibrated in aggregate: across 62 held-out races the
model claimed its favourite would win 54.7% of the time and they won 54.8%. The
championship band is not — it covers 71% where it should cover 80%, which the
method page states rather than rounds off.

**[Live forecast →](https://parikshit06.github.io/f1pred/)** ·
**[Method and accuracy →](https://parikshit06.github.io/f1pred/method.html)**

## Accuracy

Walk-forward over **2024–2026, 62 races**, each predicted by a model trained
only on earlier races. Settings fitted on 2022–23 only, bounded at both ends.

| Approach | Top 5 | Podium | Winner | NDCG@5 | Log loss | Brier |
|---|---|---|---|---|---|---|
| Grid order | **3.92** | **2.06** | **58%** | **0.898** | 1.493 | 0.617 |
| **This model** | 3.89 | 1.92 | 55% | 0.890 | **1.169** | **0.572** |
| Championship leader | 3.45 | 1.69 | 31% | 0.809 | 2.197 | 0.833 |
| Recent driver form | 3.40 | 1.48 | 27% | 0.789 | 2.129 | 0.880 |
| Team form | 3.06 | 1.45 | 31% | 0.766 | 2.322 | 0.890 |

**It loses to the grid on every ordering metric and wins on both probability
metrics.** Two fewer winners called across 62 races, a tenth of a place behind
on top-five — and log loss 1.169 against 1.493, Brier 0.572 against 0.617. Those
last two are the ones that measure whether a stated probability was true, which
a grid readout cannot do at all: it says "pole wins" with no confidence
attached, so when it is wrong it is wrong absolutely.

That the grid is this strong is the real finding, and it is worth stating
plainly rather than burying: most of a race is decided on Saturday. A model that
beat it on ordering, on this data, would be a model to distrust.

| Season | Top 5 | Winner | Log loss | Grid log loss | Races |
|---|---|---|---|---|---|
| 2024 | 4.08 | 41.7% | **1.581** | 1.720 | 24 |
| 2025 | 3.96 | 58.3% | **0.928** | 1.122 | 24 |
| 2026 | 3.43 | 71.4% | **0.875** | 1.738 | 14 |

2026 is the live season and only 14 races, so its 71.4% is the least reliable
number in this README; a single race moves it by seven points. The log loss
column is the one worth reading across seasons.

## Championship projection

**The favourite.** 27 checkpoints across 7 completed seasons: Brier **0.118**,
correct **85%** of the time. Above 95% claimed it is 12 for 12 — but the 95%
interval on 12 straight runs down to **0.76**. So a published 99.9% means
*the arithmetic says this is over*, not a calibrated one-in-a-thousand.

**The range.** It publishes a 10th–90th percentile band on every points total.
Graded against real final constructors' standings:

| Projection | Band held | Should be | Band width |
|---|---|---|---|
| Pace held fixed | 47% | 80% | 39 pts |
| **Current** | **71%** | 80% | 90 pts |

Upgrades were the obvious suspect and measurably not the cause: development
fits at **1.37 finishing positions** over a season (95% 0.90–1.83, 283
team-checkpoints), and adding exactly that much moved coverage barely at all.

The real gap is that race luck averages out over a dozen races but *the model
being wrong about a car today* does not — it rides into every remaining race.
Each simulated season now draws one pace offset per team, swept on 2019–2022
and graded on 2023–2025. Still short of 80%, and the method page says so. The
gain is concentrated early in the season, where the old band was worst;
`make calibrate-spread` prints the breakdown by how much of the season had run.

## Measured and cut

Nine candidates, **eight cut**. Full numbers in the ledger at the top of
`features.py`.

- **Blending model rank with grid rank** — the +0.09 top-five vanished once the
  weight was fitted off the test window. Selection bias.
- **An unbounded tuner** — it had a start season but no *end* season, so it ran
  through the reported window and fitted on the races it then called
  out-of-sample. Bounded properly, honest log loss came out **worse** (1.254 →
  1.281). A test now enforces the bound.
- **Retirements split by cause**, mechanical failures correlated inside a garage
  at a fitted 4.4× the independent rate. Identical to four decimal places —
  forced, since the split preserves each car's marginal retirement probability.
  Graded on the constructors' band too: 50.0% either way.
- **Pit-crew speed** — real spread (Red Bull 0.49s a stop, Cadillac 3.24s), but
  better on two metrics and worse on three including log loss.
- **Circuit-character, teammate-gap, team-switch, isotonic recalibration** — all
  neutral or worse.

Kept: `rank:ndcg` over `rank:pairwise` (NDCG@5 0.804 → 0.892), explicit car-pace
features (top-five 3.59 → 3.68), and re-fitting confidence from a trailing
24-race window so the model gets less certain when a season stops being
predictable.

## Two bugs worth knowing about

**398 finishes stored as retirements.** Upstream changed a lapped finisher from
`+1 Lap` to `Lapped` in 2023; the parser matched the `+` prefix. `make validate`
now checks the flag against **laps completed** — a fact, not a label.

**Retirements averaged in as slowness.** A car that stops is classified near
last, and that went into rolling form features while reliability was already
counted separately, charging one event twice.

Together: log loss 1.281 → **1.169**, Brier 0.629 → **0.567**. The fitted
current-season weight fell from 4× to 1.5×, which had been the model
compensating for corrupted form.

## Driver bias

`make bias`. Raw per-driver error says every front-runner is over-rated. That is
an artifact of comparing a fixed ranking against a mean dragged down by
retirements — it correlates 0.85 with predicted rank and explains **71% of the
variance**. Detrended it inverts: Hamilton **+0.79** and Verstappen **+0.65**
are slightly *under*-rated, and the real bias is midfield, Tsunoda −2.30 and
Pérez −1.48. Six tests assert predictions do not move when every driver is
renamed.

## Run it

```bash
make setup                            # creates .venv; uses uv if present, pip if not
make data-jolpica SEASONS=2018-2026   # ~15 min, rate-limited
make validate
make features
make predict
make dashboard
```

That is the whole pipeline. `make data-fastf1 SEASONS=2024-2026` is optional and
separate: it is slow, a few hundred MB a season, and it feeds practice pace to
the qualifying model only — worth roughly double the pole hit rate before
qualifying, and nothing at all once the grid is known. Without it
`practice_available` is 0 and everything else runs unchanged; `make verify`
reports its absence as a warning rather than a failure.

No API keys; all three sources are free and keyless. `make which-python` prints
the interpreter the targets use.

Slower, run separately: `make backtest`, `make title-backtest`,
`make calibrate-spread`. `make backtest` reproduces the table above exactly —
the reported window and the season the settings are fitted on are `START` and
`TUNE` in the Makefile, and the CLI refuses to report on a window it tuned on.

Gates: `make test` (177 tests), `make lint`, `make verify` (7 audits — leakage,
inputs, weighting, bias, accuracy, calibration, sanity).

`ci.yml` runs tests and lint on pushes to `main` and on every pull request;
`predict.yml` forecasts and deploys to Pages on race weekends; `evaluate.yml`
re-measures accuracy monthly, kept out of the weekend pipeline so published
figures cannot drift from documented ones.

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

XGBoost ranker (`rank:ndcg`) scores the full field, one group per race, so each
race contributes every pairwise comparison inside it rather than one winner
label. Plackett–Luce converts scores to probabilities at a temperature fitted by
log loss. A 10,000-run Monte Carlo adds what a ranking cannot express — pace
variance, safety cars, retirements, and an unknown grid before qualifying. The
blend weight is fitted, not chosen.

Every rolling feature shifts by one race before aggregating, through one shared
helper. `tests/test_leakage.py` builds cases where a leak would be visible and
asserts it is not.

```
src/f1pred/
  cli.py                 every command in this README
  config.py              paths, and the fitted constants with their workings
  store.py               DuckDB schema and idempotent upserts
  http_cache.py          rate-limited, resumable, on-disk HTTP
  ingest/                jolpica, FastF1, Open-Meteo
  validate.py            18 data quality checks
  features.py            feature engineering, leakage guards, rejected ledger
  model.py               XGBRanker wrappers
  simulate.py            Plackett–Luce and Monte Carlo
  predict.py             one race, forecast and logged
  backtest.py            walk-forward evaluation and hyperparameter fitting
  championship.py        season projection
  title_backtest.py      grades the title favourite
  spread_calibration.py  grades and re-fits the projection's range
  diagnostics.py         per-driver bias, and the artifact in it
  verify.py              the seven audits behind `make verify`
  report.py              forecast page: what goes on it
  report_render.py       and how it is drawn — CSS, SVG, colour
  method_page.py         the evidence behind it
predictions/             timestamped forecasts, committed — the track record
```

## Data

[jolpica-f1](https://github.com/jolpica/jolpica-f1) for results, grids,
qualifying, sprints, pit stops and standings;
[FastF1](https://github.com/theOehrly/Fast-F1) for session pace from 2018;
[Open-Meteo](https://open-meteo.com/) for weather. 2018 is the floor — results
go back to 1950, but without session timing those seasons are not comparable.

jolpica is run by volunteers on donations, so the client is deliberately
conservative: ~0.8 req/s against a documented 4/s, a sliding hourly window under
their 500/hour persisted between runs, backoff on 429, everything cached.

## Limits

Most of a race is settled before it starts. The measure of that here is the
grid baseline: starting order alone calls 58% of winners and scores 3.92 on
top-five, which is the bar any model has to clear. Retirements and safety cars
account for much of the rest, and neither is knowable beforehand.

Per-season winner accuracy swings between 42% and 71% on 14–24 races each. That
spread is what a sample this size looks like, not a trend — the 2026 figure in
particular rests on 14 races and should not be read as the model improving.

Not modelled: weather, tyre strategy, in-race penalties, team orders, and which
team will improve. The method page lists these with what was tried against each.

## Licence

MIT. Data belongs to its sources; see jolpica's
[terms of use](https://github.com/jolpica/jolpica-f1/blob/main/TERMS.md).
