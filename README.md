# F1 Prediction System

[![CI](https://github.com/Parikshit06/f1pred/actions/workflows/ci.yml/badge.svg)](https://github.com/Parikshit06/f1pred/actions/workflows/ci.yml)

Probabilistic forecasts for every Formula 1 Grand Prix. Before each session it
publishes each driver's chance to win, reach the podium and finish in the top 5
and top 10, along with a qualifying forecast and a championship projection.
Every forecast is saved before the session it forecasts and graded against the
result afterwards.

**[Live forecast](https://parikshit06.github.io/f1pred/)** ·
**[Method and accuracy](https://parikshit06.github.io/f1pred/method.html)** ·
**[Past races](https://parikshit06.github.io/f1pred/races.html)** ·
**[Full methodology](METHODOLOGY.md)**

## What it predicts

A race weekend produces three forecasts, each from more information than the
last, and a graded result:

| When | What it knows | What is published |
|---|---|---|
| Wednesday of race week | past races only | race, qualifying and championship forecasts |
| After practice | this weekend's practice times too | qualifying forecast, and a race forecast on the projected grid |
| After qualifying | the official starting grid | race forecast on the real grid |
| After the race | the result | the forecast graded, and kept on a page of its own |

Forecasts are committed to [`predictions/`](predictions/) as they are made and
never edited, so the track record cannot be tidied up afterwards.

## Why the approach is interesting

- **It ranks, rather than classifies.** A race is an ordering of about twenty
  drivers, so the model learns from every pairwise comparison inside a race
  instead of a single "winner" label.
- **Its probabilities cannot contradict each other.** Win, podium, top 5, top
  10 and expected finish are all read from one finishing-position distribution
  per race, so a driver can never be likelier to win than to reach the podium.
- **It cannot see the future, and tests prove it.** Every feature is built
  from earlier races only, and tests rebuild and tamper with the data to show
  that nothing leaks backwards.
- **It is judged against strong references.** The main one is the starting
  grid with calibrated probabilities, the hardest simple forecast to beat. Every
  comparison carries a 95% interval, and the results say where the model
  loses as well as where it wins.

## What the evaluation found

Every race from 2024 round 1 to 2026 round 15, **63 races**, was forecast by a
model trained only on races before it, with its settings fitted beforehand on
2022–23.

- **Before qualifying, it clearly beats the championship order**: its win
  chances are more accurate and it names more of the top five.
- **Its qualifying forecast clearly beats recent qualifying form**, and adding
  this weekend's practice times clearly improves it on held-out weekends.
- **After qualifying, it does not beat the calibrated starting grid.** It is
  level on probabilities, and the grid is clearly better at ordering the whole
  field. The grid carries most of what can be predicted.
- **Its win chances are well calibrated**, but its most confident podium and
  top-10 calls came true less often than stated.
- **Its title favourite was right 89% of the time**, but backing the points
  leader would have been right almost as often.

Each model is compared with a forecast that needs no model, with a 95%
interval over races. "Inconclusive" means the interval includes zero.

| Stage | Metric | Model | Reference | Difference (95% interval) | Verdict |
|---|---|---|---|---|---|
| After qualifying | NDCG@3 | 0.852 | 0.857 (grid) | -0.005 (-0.031 to +0.021) | inconclusive |
| | NDCG@5 | 0.895 | 0.897 (grid) | -0.002 (-0.019 to +0.015) | inconclusive |
| | Winner called | 67% | 59% (grid) | +8 pts (-3 to +19) | inconclusive |
| | Win log loss | 1.201 | 1.341 (grid) | -0.141 (-0.358 to +0.062) | inconclusive |
| | Podium Brier | 0.069 | 0.069 (grid) | -0.000 (-0.006 to +0.006) | inconclusive |
| | Kendall | 0.521 | 0.549 (grid) | -0.028 (-0.054 to -0.002) | **grid better** |
| Before qualifying | Win log loss | 1.723 | 2.128 (championship order) | -0.405 (-0.653 to -0.138) | **model better** |
| | Podium Brier | 0.084 | 0.096 (championship order) | -0.011 (-0.019 to -0.003) | **model better** |
| | Top-5 overlap | 3.70 of 5 | 3.44 of 5 (championship order) | +0.254 (+0.064 to +0.460) | **model better** |
| Qualifying | Pole log loss | 1.707 | 2.108 (recent qualifying form) | -0.402 (-0.678 to -0.133) | **model better** |
| | NDCG@5 | 0.854 | 0.810 (recent qualifying form) | +0.044 (+0.019 to +0.070) | **model better** |

Once the grid is known, the model's win log loss is 0.141 lower than the
grid's, but the interval runs from -0.36 to +0.06, and on one metric the grid
is clearly better. (An earlier version of this project claimed a large lead
over the grid. That grid baseline had never been calibrated.) The figures come
from [`reports/backtest.json`](reports/backtest.json), and a test checks that
this page quotes them exactly.

## How it works

```mermaid
flowchart TD
    A["jolpica-f1: results, qualifying, standings<br/>OpenF1: official grid, entry lists<br/>FastF1: this weekend's practice"] --> B[("DuckDB<br/>21 data checks")]
    B --> C["Temporal features<br/>(earlier races only)"]
    C --> Q["Qualifying model<br/>XGBRanker, one group per session<br/>+ this weekend's practice"]
    C --> R["Race model<br/>XGBRanker, one group per race"]
    Q -- "before qualifying:<br/>projected grid" --> R
    G["Official starting grid<br/>(penalties applied)"] -- "after qualifying" --> R
    R --> P["Plackett–Luce distribution<br/>(temperature refitted on past races)"]
    R --> M["Monte Carlo, 10,000 races<br/>retirements, safety cars, grid"]
    P --> F["One finishing-position distribution<br/>win · podium · top 5 · top 10"]
    M --> F
    F --> W["Forecast committed to predictions/<br/>→ website"]
    F --> E["Walk-forward evaluation<br/>→ reports/*.json"]
```

- **Data.** Results, qualifying and standings since 2018 from jolpica-f1, the
  official starting grid and entry lists from OpenF1, and this weekend's
  practice times from FastF1. Everything is checked before it is modelled.
- **Race-level learning-to-rank.** An XGBoost ranker (`rank:ndcg`) is trained
  with one query group per race. A second ranker forecasts qualifying, and
  before qualifying its forecast stands in for the grid.
- **No look-ahead.** Every feature is built only from earlier races. Rebuilding
  features from data that stops at a race changes nothing before it, and
  rewriting a result moves nothing up to that race.
- **Walk-forward evaluation.** No random splits. Settings are chosen on
  2022–23 and reported on 2024 onward, never on the races they are graded on.
- **Calibrated, coherent probabilities.** Scores become one finishing-position
  distribution per race, and the temperature that sharpens it is refitted only
  on past races.
- **Monte Carlo for race-day uncertainty.** 10,000 simulated races add
  retirements, safety cars and (before qualifying) an uncertain grid, blended
  with the ranker's own Plackett–Luce probabilities.

## Reproduce it

No API keys and no data download: the demo runs the whole pipeline on a seeded
synthetic championship.

```bash
git clone https://github.com/Parikshit06/f1pred.git && cd f1pred
make setup   # exact versions from uv.lock (Python 3.11–3.13)
make demo    # ingest → features → walk-forward → forecast → site, offline
make test    # unit, leakage and end-to-end tests
```

With real data:

```bash
make data-jolpica SEASONS=2018-2026   # ~15 min first time, rate-limited, cached
make data-openf1                      # official grids and entry lists
make repair validate features
make predict dashboard                # forecast the next race, render the site in reports/
make backtest title-backtest calibrate-spread verify   # regenerate reports/*.json
make experiments                      # the design experiments, about an hour
```

The scheduled workflows in [`.github/workflows`](.github/workflows) run the same
steps: `predict.yml` across each race weekend (and deploys the site),
`evaluate.yml` monthly.

## Limitations

- Win probabilities are well calibrated overall, but the few win calls above
  70% came true less often than stated (too few races to say by how much).
  The most confident podium and top-10 calls are over-confident: podium calls
  averaging 94% came true 78% of the time, and top-10 calls
  averaging 96% came true 87%, because the temperature is fitted on winners only.
- Not modelled: weather, tyre and pit strategy, in-race penalties, team orders,
  upgrades, safety-car timing, failures shared by a team's two cars.
- The qualifying forecast orders the back of the grid loosely, so
  backmarkers' chances of reaching the top ten in qualifying are too high. A
  fix that orders the whole field better cost accuracy at the front, so it
  is not used.
- The after-practice forecast needs this weekend's practice times from FastF1,
  and FastF1 cannot load F1's timing data from GitHub's runners. So no
  after-practice forecast has been logged in 2026. The practice history the
  qualifying model learns from is fetched on a local machine and reaches the
  scheduled runs through the published data snapshot.
- Until the official grid is published, penalties are unknown and the
  qualifying order stands in. The forecast says so.
- 63 test races separate the model from naive baselines. Against the
  calibrated starting grid it is level on probabilities and behind on ordering
  the field.
- The season projection assumes current form holds, widened by a calibrated
  pace-drift term.

---

The rest of this page is technical detail. [METHODOLOGY.md](METHODOLOGY.md)
has the full method.

## Results in detail

After qualifying, against the calibrated grid and championship order
(`reports/backtest.json`, generated by `make backtest`):

| Approach | NDCG@3 | NDCG@5 | Winner | Podium overlap | Win log loss | Podium Brier |
|---|---|---|---|---|---|---|
| **This model** | 0.852 | 0.895 | **67%** | 1.92 of 3 | **1.201** | **0.069** |
| Grid order, calibrated | **0.857** | **0.897** | 59% | **2.05 of 3** | 1.341 | **0.069** |
| Championship order | 0.756 | 0.808 | 30% | 1.68 of 3 | 2.128 | 0.096 |

On one metric, the grid is clearly better: Kendall rank correlation over the
whole field, -0.028 (-0.054 to -0.002). Every other interval includes zero.

**Calibration.** When the model says X%, the average gap to what happened
(expected calibration error, per driver) is 0.5% for wins, 2.8% for podiums
and 6.1% for the top ten. The method page shows it band by band.

## Leakage prevention

- Rolling windows shift a whole race before aggregating. Team features
  collapse to one value per team per race first, so a driver never sees the
  teammate's result from the same race.
- **Truncation test**: features built from data ending at race *k* equal
  features built from everything, for every race up to *k*. It caught a real
  leak: a season-progress feature divided by the number of races already run.
- **Tampering test**: reversing one race's result moves no feature of that
  race or any earlier one.
- The walk-forward trains on `race_seq < k` only, and a test spies on every fit
  to prove it. Before qualifying, every grid and qualifying input is replaced by
  its projection, so the real result can't reach a pre-qualifying forecast.

## What each piece is worth

`make experiments` writes `reports/experiments.json`. Every experiment is a
walk-forward with race-paired intervals. Choices are made on 2022–23 and
checked on 2024 onward (practice, below, has its own held-out split). Each was made
once, on the final configuration, and is not re-decided when the evaluation is
refreshed. Experiments fit one seed per model, so their figures sit slightly
off the headline's five-seed ones.

**Calibration and the simulation.** The same out-of-sample scores on 2024
onward, turned into probabilities five ways:

| Probability step | Win log loss | Podium Brier | Calibration error, win |
|---|---|---|---|
| Raw scores (temperature 1, no simulation) | 1.686 | 0.0747 | 3.8% |
| Temperature fitted on 2022–23, held | **1.226** | 0.0679 | 0.3% |
| Temperature refitted on the last 24 races (in use) | 1.229 | 0.0683 | 0.4% |
| Refitted, Plackett–Luce only | 1.301 | 0.0716 | 0.3% |
| Refitted, simulation only | 1.256 | **0.0651** | 0.5% |
| Refitted on the first three finishers, not just the winner | 1.261 | 0.0662 | 1.0% |

Calibration is most of the gain. A held and a refitted temperature are level,
and mixing the two probability models beats either alone on log loss. Fitting
the temperature on the first three finishers was tested against the
over-confident podium and top-10 calls: it improves their calibration on both
2022–23 and 2024 onward, but worsens win log loss on 2022–23 (0.928 against
0.883), so it is not used. It is a real trade-off, not a free fix.

**Ablation: what adds information once the grid is known.**

- **Everything together vs the grid alone**: log loss 0.840 vs 1.628 on
  2022–23 (difference +0.788, interval +0.462 to +1.155), but 1.229 vs 1.326 on
  2024 onward (not clear).
- **Adding one block at a time to the grid** (log loss on 2022–23): grid only
  1.628, + driver form 0.957, + team form 1.220, + both 0.914, full model
  0.840. Recent driver form carries most of what the model adds beyond the
  grid. On 2024 onward none of these differences is clear.
- **Driver form and team form** are the groups whose removal clearly costs
  log loss on 2022–23: +0.186 (+0.046 to +0.348) and +0.082 (+0.004 to
  +0.168). Removing **circuit history** clearly hurts ranking (NDCG@5 -0.011,
  -0.021 to -0.001).
- **Qualifying position and gap to pole** duplicate the grid (correlation
  0.95 with it): removing them changes nothing clearly. They stay because they
  are how a car that took a grid penalty is recognised as fast.
- **Tested and left out**: the driver's own qualifying record (average grid
  and qualifying place, pole rate, one-lap gap) made 2022–23 clearly worse
  after qualifying, +0.078 (+0.027 to +0.138), and before it, +0.043 (+0.009
  to +0.078). It reaches the race through the grid already, so inside the race
  model it counted twice. The teammate qualifying gap also made it worse,
  +0.049 (+0.010 to +0.089). Both, as rolling averages, still feed the
  qualifying model.
- On 2024 onward, no single group's removal or addition is clear either way.

**Practice pace (FP1–FP3): qualifying only, earned on held-out data.**
Practice is used as this weekend's evidence, never as a long-term driver
feature: each driver's gap to the fastest car in each session, their latest
session's gap and rank, the gap to their teammate, the team's best car, the
change from FP1 to FP2 to FP3, and how consistent those were. Stored as
compact numbers for 59 weekends of 2024–26, no telemetry. The design was
chosen on 2024, then graded on 35 untouched weekends of 2025–26 against the
same model without practice:

| Qualifying forecast, with practice minus without (2025–26) | Difference (95% interval) |
|---|---|
| Qualifying position error (places) | -0.280 (-0.429 to -0.128) |
| Pole log loss | -0.203 (-0.404 to -0.009) |
| NDCG@5 | +0.028 (+0.000 to +0.059) |
| NDCG@3 | +0.040 (-0.006 to +0.089) |

Practice clearly improves the qualifying forecast, so it feeds the qualifying
model. After qualifying it adds nothing beyond the official grid (race log loss
+0.029, -0.051 to +0.117), so the race model never sees it. (An earlier
version kept practice on a test that overlapped the reported window. A
simpler two-number design, re-tested properly, was not clearly better.)

**Qualifying: the front of the grid over the back.** The qualifying ranker
uses NDCG's exponential gains, so training concentrates on the front of the
grid and orders the back loosely. On 2024 onward it is no better than recent
qualifying form at ordering the whole field, and it gives backmarkers too good
a chance of the top ten. Two fixes were tested with the rule fixed in advance
(`quali_ordering` in `reports/experiments.json`). The median of recent
qualifying positions made no clear difference in either window. Linear gains
ordered the whole field clearly better on 2022–23 and again on 2024 onward
(position error -0.251 (-0.388 to -0.116) places), but on 2024 onward they were
clearly worse at the front, NDCG@5 -0.018 (-0.034 to -0.002). The race forecast
depends most on the front of the grid, so neither is used.

**Recent form: median, not mean.** Median finishing position over the last
races lowered 2022–23 log loss by 0.047 (interval 0.003 to 0.093). On 2024
onward the difference is not clear (-0.018 for the mean, -0.068 to +0.035). An
earlier configuration measured the same gain with an interval just touching
zero, so it is a marginal call, made once and not revisited.

**XGBoost settings** were compared on 2021 and on 2022–23 without intervals,
as a stability check rather than a search. No candidate is best on both:
depth 3 is best on 2022–23 and among the worst on 2021. The settings in use
(depth 4, 400 trees) are kept.

## Explainability

Every logged forecast stores, for each driver, SHAP contributions from XGBoost's
own TreeSHAP (`pred_contribs`, identical to `shap.TreeExplainer`). The published
score averages five seeds standardised within the race, so each seed's values
are centred and scaled the same way and averaged. The contributions then sum
exactly to the score that was published (tested). They describe what the model
relied on, not causes.

## Championship projection

The rest of the season is simulated 10,000 times, sprints included, from each
driver's median score over their last eight real weekends. Every driver with
points stays in the title race, including one sitting out a race. Graded on
27 checkpoints across 7 completed seasons (2019–2025), each projected with a
model trained only on races before it:

- **Who wins**: the favourite was right 89% of the time, Brier 0.077. That
  is a low bar: the favourite was simply the points leader at 26 of 27
  checkpoints, and backing the leader would have been right 85% of the time.
  Above 95% it is 14 for 14, but the 95% interval on that runs down to
  0.79, so a published 99.9% means the arithmetic says it's over, not one
  chance in a thousand. The misses are all in 2025's Piastri–Norris season
  (after rounds 8, 13 and 18).
- **How close**: every points total carries a 10th–90th percentile range,
  sized on 2019–22 and graded on 2023–25 (`make calibrate-spread`):

| Projection | Constructors inside the range | Width | Teammate gap inside | Target |
|---|---|---|---|---|
| Race luck only | 54% | 39 pts | 72% | 80% |
| **With the season-long pace term** | **86%** | 108 pts | 74% | 80% |

The pace term was sized on 2019–22, and on 2023–25 constructors land a little
wide of the target (86% against 80%). The teammate gap is still short of 80%:
on 2019–22 the ranges needed no extra driver term, but on 2023–25 they were too
narrow.

## Project structure

```
src/f1pred/
  ingest/            jolpica, OpenF1 (grid, entries), FastF1 (practice aggregates)
  store.py           DuckDB schema (raw tables are never overwritten)
  validate.py        21 data checks
  weekend.py         who is in the race, and where each car starts
  features.py        temporal feature layer
  model.py           XGBoost rankers, ensemble SHAP
  probability.py     Plackett–Luce, calibration, one coherent distribution
  simulate.py        Monte Carlo race simulation and forecast()
  predict.py         forecast one race, with provenance
  baselines.py       grid order and other no-model forecasts
  backtest.py        walk-forward, tuning
  metrics.py         ranking and probability metrics, bootstrap
  evaluation.py      headline backtest and experiments
  championship.py    season projection (sprints included)
  title_backtest.py, spread_calibration.py   grading the season projection
  verify.py          release gate: leakage, calibration, coherence
  report*.py, method_page.py   the website
  demo.py            synthetic championship for make demo and CI
predictions/         every logged forecast (the track record)
reports/             measured figures, which the README is tested against
tests/               unit, leakage, pipeline (end-to-end on the demo data)
```

## Data and licence

Code: MIT. Data belongs to its sources: [jolpica-f1](https://github.com/jolpica/jolpica-f1)
(CC BY-NC-SA 4.0, see its [terms](https://github.com/jolpica/jolpica-f1/blob/main/TERMS.md)),
[OpenF1](https://openf1.org), [FastF1](https://github.com/theOehrly/Fast-F1).
This is an unofficial project, not associated with Formula 1.
