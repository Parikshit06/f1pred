# f1pred

Top-ten qualifying and race forecasts for Formula 1, as calibrated
probabilities, plus a championship projection. Every prediction is written to
`predictions/` with the timestamp it was generated, then graded once the race
has run.

The predictions are not really the point. The point is the evaluation: whether
the probabilities are honest, and whether the model beats the obvious answer.

## What it does

Two models. The qualifying model predicts the grid; the race model predicts the
finish. Both rank the full field and publish the top ten as probabilities.

A third step projects the championship: the remaining calendar is run ten
thousand times from the race model's current strengths, carrying the points
already scored. That gives title odds, a projected final table for drivers and
constructors, and a 10th-90th percentile band around every projection - the
band matters, because a mean on its own reads as a promise. The leader in the
current standings is projected to win **4.5 of the remaining 9**, not all of
them; he retires 1.2 times on average. Title odds are apportioned so the column
sums to 100% and never prints 100% for a title that is not yet mathematically
won.

It assumes current form holds and does not re-forecast each weekend - stated on
the page, because it is the assumption that matters. Sprint points are not
projected, since the remaining sprint calendar is not in the data.

`make dashboard` renders `reports/index.html`, a single self-contained file
with no build step and no CDN, deployed to GitHub Pages by
`.github/workflows/predict.yml`. The page carries four things and nothing else:
the race board, the qualifying board, the championship projection and the
points race. `reports/method.html` sits one click behind it and carries the
working: the pipeline from score to probability, the feature list, the
walk-forward accuracy split by season, the calibration table, and the
championship projection graded against seven completed seasons. Bias
diagnostics and the record of what was tried and cut stay here in the README.

The two race models chain. Before qualifying the grid is unknown, so the qualifying model
produces a distribution over possible grids and each of the 10,000 simulated
races runs on its own sampled grid. Once qualifying has happened the real grid
goes in and everything is re-run. Both versions are kept, which is what lets you
measure how much of race prediction is just knowing where everyone starts.

## Results

Walk-forward over **2024–2026: 62 races**, each predicted by a model trained only
on races before it (187 by the end). Full field ranked, top ten published.

| Approach | Top 5 | Podium | Winner | NDCG@5 | Log loss | Brier |
|---|---|---|---|---|---|---|
| Grid order | **3.92 / 5** | **2.07 / 3** | **58%** | **0.898** | 1.493 | 0.617 |
| **This model** | 3.86 / 5 | 1.97 / 3 | **58%** | 0.889 | **1.167** | **0.567** |
| Championship leader | 3.45 / 5 | 1.69 / 3 | 31% | 0.809 | 2.197 | 0.833 |
| Recent driver form | 3.40 / 5 | 1.48 / 3 | 27% | 0.789 | 2.129 | 0.880 |
| Team form | 3.07 / 5 | 1.45 / 3 | 31% | 0.766 | 2.322 | 0.890 |

Settings (temperature, blend, recency) are fitted on **2022–23 only** and the
table reports **2024–26**, so the tuner never saw these races. That separation
is enforced in code — see the tuner note under "what was tried".

**It matches the grid baseline on winners and loses to it slightly on ordering.**
Where it wins is the probability: log loss 1.167 against 1.493 and Brier 0.567
against 0.617. Those two measure whether a stated probability was true, which is
the one thing a grid readout cannot do at all — a grid says "pole wins" with no
confidence attached, so when it is wrong it is wrong absolutely.

The grid being that strong a signal is the actual finding, and it is worth
stating plainly rather than burying: most of a race result is decided on
Saturday.

### How this compares to a bookmaker

The obvious benchmark is a betting market, and it is worth being precise about
why that is not a fair fight. A closing price is not a model output; it is the
aggregate of everyone who was willing to put money behind an opinion, including
people with information this project structurally cannot have — Friday fuel
loads, engine modes, a floor upgrade that did not work, a driver who is ill,
tyre allocation plans, and the local weather three hours out. Markets also move
until minutes before the lights, while this forecast is fixed and timestamped.

What *can* be compared fairly is the shape of the output, and that is the part
this project takes seriously: win probabilities that sum to one, are graded
against the outcome, and are checked against Wilson intervals rather than
asserted. On that measure it is close but not clean: pooled it says 0.538 where
0.597 happens, which `make verify` reports as a warning rather than hiding.

The realistic reading is that this system is competitive with a naive market
line and not with a sharp one, and the gap is information rather than
technique. The single largest closable piece of that gap is Friday practice
pace, which is already wired in and simply needs `make data-fastf1`.

### Does the championship projection deserve 99%?

`make title-backtest` grades it. Every completed season is stopped at four
checkpoints, the title is projected from the model as it stood at that moment,
and the claim is compared with who actually won. 27 checkpoints, 7 seasons,
Brier **0.112**, favourite correct **85%** of the time.

| Claimed | Happened | n | 95% interval |
|---|---|---|---|
| 50–80% | 55.6% | 9 | 0.27–0.81 |
| 80–95% | 100% | 2 | 0.34–1.00 |
| over 95% | **100%** | 16 | **0.81–1.00** |

Every time it has claimed more than 95%, the favourite has won — 16 out of 16.
That is a real record and a small one: sixteen straight is consistent with a
true rate as low as **81%**, which is what the interval says. So 99.9% should be
read as *the arithmetic and the form both say this is over*, not as a calibrated
one-in-a-thousand.

The misses are the interesting rows. In 2020 after round 15 it favoured Bottas
at 78%; Hamilton won. In 2021 after round 20 it favoured Hamilton at 62%;
Verstappen won. In 2025 it favoured Piastri at two checkpoints before Norris
took it. A projection that assumes current form holds is exactly wrong in the
seasons where form does not hold, and those three are the proof.

### Accuracy by season

A pooled average can hide a bad year, so the split is published too.

| Season | Approach | Top 5 | Podium | Winner | Log loss | Brier | Races |
|---|---|---|---|---|---|---|---|
| 2024 | Grid order | **4.00** | **2.00** | 45.8% | 1.720 | 0.695 | 24 |
| 2024 | This model | 4.04 | 1.71 | 45.8% | **1.576** | **0.694** | 24 |
| 2025 | Grid order | **4.00** | 2.25 | **66.7%** | 1.122 | 0.568 | 24 |
| 2025 | This model | 3.92 | **2.29** | 62.5% | **0.945** | **0.518** | 24 |
| 2026 | Grid order | **3.64** | 1.86 | 64.3% | 1.738 | 0.570 | 14 |
| 2026 | This model | 3.43 | 1.86 | **71.4%** | **0.849** | **0.436** | 14 |

2026 is the live season and the model's best one: it calls more winners than the
grid does and roughly halves the log loss. 2024 is its worst, which is also the
season with the most one-sided car — the grid tells you nearly everything when
one team is a second clear.

### Two bugs worth reading about

Both were found by chasing a forecast that looked wrong, and neither raised an
error. They are the reason the numbers in the table above moved.

**A third of all modern finishes were recorded as retirements.** The status
field is free text owned by the upstream API. Until 2022 a lapped finisher is
`+1 Lap`; from 2023 the same thing is written `Lapped`. The parser matched on
the prefix `+`, so from 2023 it silently reclassified **398 ordinary finishes as
retirements** — concentrated in exactly the midfield cars that get lapped, which
made some drivers look like they retired from every race. That fed recent form,
the reliability features and the simulator's retirement hazard. The raw status
text was stored, so it was repairable in place (`make repair`) without
re-downloading nine seasons, and `make validate` now cross-checks the flag
against laps completed — a fact, not a label — so the next wording change is
caught rather than absorbed.

**A retirement was being averaged in as genuine slowness.** A car that stops is
classified near last, and that position was going straight into the rolling form
features. A driver who retired from third read as a backmarker for the next
several races. Reliability was already a separate feature, so this charged for
the same event twice, and the second charge landed hardest on cars running at
the front when they stopped — the ones with furthest to fall. Pace and
reliability are now separate: a retirement counts once, as unreliability, and is
skipped by the pace windows rather than consuming a slot in them.

Fixing both took log loss from 1.281 to **1.167** and Brier from 0.629 to
**0.567**, and moved the fitted current-season weight from 4× back to 1.5× — the
4× had been the model compensating for corrupted form by leaning on the newest
races. A settings value that moves a long way when a bug is fixed is usually a
symptom rather than a finding.

### What was tried and did not work

Worth recording, because the failures constrain the conclusion:

- **Blending model rank with grid rank.** On 2024–26 a 50/50 blend scored 3.95
  top-five, beating both grid (3.92) and model (3.86) — a tempting result. But
  the weight was chosen by looking at the test window. Re-fitting it honestly on
  2022–23 picked 0.8, and at 0.8 the blend scores 3.86 on 2024–26: no better
  than the model alone and worse than the grid. **The apparent gain was
  selection bias**, and the rigorous version is in the repo history rather than
  in the pipeline.
- **An unbounded tuner, which was quietly cheating.** `--tune-season 2022`
  refused to run unless the tuning season preceded the reported window, which
  looked airtight. It was not: the tuners called the walk-forward with a start
  season and no *end* season, so they ran from 2022 straight through 2026 and
  fitted temperature, blend and recency on the very races the table then
  reported as out-of-sample. Bounded at both ends, the honest log loss came out
  *worse* than the contaminated one (1.254 → 1.281). That gap is the size of the
  self-deception, and the bound is now enforced by a test.
- **Form at circuits of the same character** — grouping tracks into power,
  downforce and balanced so "quick on the straights" was something the model
  could represent. The effect is visible in the raw numbers, but it rests on
  three or four races per driver per season, and adding it cost accuracy on
  both models — worst of all on the pole call it was built to sharpen (pole hit
  rate 25.8% → 22.6% expanding, 24.2% rolling). Recent form and the car-pace
  features already carry it with far more evidence behind them. Cut; the
  numbers are in the ledger at the top of `features.py`.
- **Seed-averaging across five fits** was accuracy-neutral (NDCG@5 0.894 either
  way). Kept anyway so a published forecast does not move when the model is
  re-run — a stability property, not an accuracy one.
- **Isotonic recalibration** of the win probabilities. Built, measured, cut —
  see the calibration section below.
- **Blending the qualifying score with the team's best recent grid slot.** That
  one-line baseline calls pole 35.5% of the time against the model's 27.4%, so
  the model looked like it was under-weighting the strongest car-pace signal.
  The weight was fitted on 2022–23 by NDCG, which picked 0.45 and improved every
  metric on that window. It did not transfer: on 2024–26 the pole rate went
  **0.274 → 0.226** and NDCG 0.831 → 0.829, with top-three and log loss slightly
  better. Worse on the number it was built to fix, so it is out — the second
  time a blend has looked good on the window it was fitted on and evaporated
  off it.

Seven candidate improvements were measured and six were cut. That is the honest
state of it: on this feature set and 187 training races the model sits close to
what the data supports, and the next real gain is more signal (FastF1 practice
pace) rather than more modelling.

### Is it just favouring the famous drivers?

Reasonable worry — most of the training data was won by a handful of people.
`make bias` answers it, and the first answer it gives is wrong in an
instructive way.

Averaging per-driver rank error says the model badly over-rates every
front-runner: Antonelli −2.47 positions, Norris −1.98, Verstappen −1.90. That is an artifact.
Finishing position is noisy asymmetrically — a fast driver who retires is
classified near last, dragging their *mean* result below their true pace, while
a slow driver inherits places when quicker cars break. Comparing a fixed
ranking against a mean pulled toward the middle manufactures a bias gradient
down the whole grid. It correlates 0.85 with predicted rank and explains **71%
of the variance**.

Fit that trend, take the residual, and the picture inverts:

| Driver | Raw bias | After correction |
|---|---|---|
| Tsunoda | −2.47 | **−2.30** |
| Pérez | −1.45 | **−1.48** |
| Hamilton | −0.56 | +0.79 |
| Verstappen | −1.90 | +0.65 |
| Ocon | +3.90 | +1.91 |

**Hamilton and Verstappen are slightly *under*-rated, not over-rated.** The real
bias is Tsunoda and Pérez — both quick cars, both drivers being beaten inside
them — so the model does conflate car with driver, by about two places, in the
midfield. It matters less than it looks: the model optimises and publishes the
sharp end, where residual bias is under one position.

Three fixes were tried and all three were cut. Teammate-gap features drew 3.3%
of model signal and made podium accuracy worse. Team-switch features (a driver's
history is earned in whatever car they were in — Hamilton at Ferrari is the live
case) moved top-five by +0.03, which is noise on 62 races, made top-one worse,
and *widened* the switcher bias they targeted. Shrinking rookie form toward the
car was neutral.

The structural reason they had nothing to add: the race model already sees this
weekend's grid, which encodes the current driver-and-car combination directly.
Grid and qualifying carry **48%** of the model's signal against 30% for driver
history. What survives is the diagnostic, not the fixes.

### What did work

- **`rank:ndcg` over `rank:pairwise`.** Pairwise spends capacity separating 15th
  from 16th. Switching lifted NDCG@5 from 0.804 to 0.892 on the same split.
- **Car-pace features.** Rolling qualifying gap to pole, as an explicit feature
  rather than something inferred from finishing positions. Top-five 3.59 → 3.68,
  log loss 1.41 → 1.37.
- **Rolling confidence calibration.** A fixed temperature cannot serve two eras.
  On 2022–23, with one dominant car, being decisive paid. Carried forward
  unchanged it was overconfident. Re-fitting temperature from a trailing 24-race
  window of finished races lets the model become less certain when the season
  stops being predictable, and is worth 1.287 → 1.281 log loss on 2024–26.
  Worth noting how much that shrank: under the leaky tuner the same change
  looked like 1.281 → 1.254, four times the gain. Most of what it was "fixing"
  was a badly fitted fixed temperature, and honest tuning fixed that directly.

Calibration is checked against Wilson intervals rather than by eye, because a
bucket of nine races will miss its target by 20 points on luck alone:

| Forecast bucket | Model says | Actually happened | n |
|---|---|---|---|
| 0.19–0.38 | 0.301 | 0.250 | 12 |
| 0.38–0.57 | 0.479 | 0.750 | 24 |
| 0.57–0.76 | 0.651 | 0.500 | 18 |
| 0.76–0.95 | 0.840 | 0.714 | 7 |

Pooled, the model says 0.538 against 0.597 observed (95% CI 0.473–0.710): it is
now mildly **under**-confident, where the pre-fix version was over-confident.
Two of five buckets fall outside their own interval, which is more than five
comparisons should produce by chance, so `make verify` flags this as a warning
rather than a pass — the honest state, printed rather than smoothed over. It is
also a small sample: the largest bucket holds 24 races. That shape is what isotonic regression exists to
fix, so it was built and measured (under the earlier settings, before the tuner
fix — the comparison is like-for-like within itself):

| | Log loss | Brier | Mean calibration gap |
|---|---|---|---|
| Rolling temperature only | **1.227** | **0.609** | 0.108 |
| \+ isotonic recalibration | 1.251 | 0.621 | 0.108 |

**Worse on both scoring rules, no better on calibration** — it pulled the low
tail in (gap 0.200 → 0.135) and pushed the high tail out (0.193 → 0.230). The
cause is sample size: a trailing window of ~37 races supplies 37 positive
examples to fit a flexible non-parametric curve on, and isotonic overfits the
placement of its own steps. Cut. It is worth revisiting at several hundred
races; it is not worth shipping at 62.

One bug found on the way is worth recording, because it nearly produced a false
negative. The first run showed isotonic changing *nothing* — identical to four
decimals. Isotonic is a step function, so most of the back of the grid maps to a
single value, and a monotonicity guard comparing `argsort` of the tied output
against the input was failing on tie-break order and silently returning the
input. The unit test missed it because it fed pre-sorted probabilities, where
index order and value order coincide. Fixed by breaking ties with a vanishing
multiple of the original probability; the test now asserts order preservation
across 500 randomly ordered fields.

## Before pushing

```bash
make test           # 111 tests
ruff check src tests && ruff format --check src tests
make validate       # data quality gate, exits non-zero on error
make verify         # 7 audits: leakage, inputs, weighting, bias, accuracy, calibration, sanity
```

`ci.yml` runs the first two on every push. `predict.yml` forecasts and deploys on
race weekends; `evaluate.yml` re-measures accuracy monthly and commits the
JSON the method page reads. Evaluation is deliberately *not* in the weekend
pipeline — re-running it three times a weekend with a different window than the
README reports is how published accuracy figures drift away from documented
ones.

## Setup

```bash
make setup          # stdlib venv + pip, needs Python 3.11+
# or, with uv installed:
make setup-uv
```

No API keys. All three data sources are free and keyless.

## Running it

```bash
make data-jolpica SEASONS=2018-2026   # ~15 min, rate-limited
make validate                         # data quality gate
make build-features
make backtest START=2022
make predict                          # next unrun race
make dashboard                        # reports/index.html
```

Optional, and slow — several GB and a few hours:

```bash
make data-fastf1 SEASONS=2024-2026    # practice pace and session weather
```

The model runs without it. Practice features are emitted as nulls and XGBoost
handles them natively, so FastF1 upgrades the forecast rather than gating it.

## Data

| Source | Used for | Coverage |
|---|---|---|
| [jolpica-f1](https://github.com/jolpica/jolpica-f1) | results, grids, qualifying, sprints, pit stops, standings | 1950– |
| [FastF1](https://github.com/theOehrly/Fast-F1) | practice and qualifying lap pace, stints, compounds, trackside weather | 2018– |
| [Open-Meteo](https://open-meteo.com/) | race-day forecast and historical archive | — |

2018 is the floor. Results go back to 1950, but without session timing those
seasons are not comparable.

### Rate limiting

jolpica is run by volunteers on donations, so the client is deliberately
conservative: ~0.8 requests/second against a documented 4/s, a sliding hourly
window under their 500/hour, exponential backoff on 429, and every response
cached to disk so a re-run costs nothing.

The hourly window is persisted to disk between runs. The budget is per-IP and
does not reset when your process does — without that, every restart during
development believes it has full quota and gets throttled from the first call.

## How it works

```
jolpica ──┐
FastF1 ───┼─→ DuckDB (raw_*) ─→ validate ─→ features ─→ quali ranker
Open-Meteo┘                                    │             │
                                               │      predicted grid
                                               │             ↓
                                               └───────→ race ranker
                                                             ↓
                                            Plackett–Luce + Monte Carlo
                                                             ↓
                                                  top 5, logged and graded
```

**Ranking, not classification.** Predicting only the winner gives one positive
example per race — about 170 since 2018. Ranking the field turns each race into
every pairwise comparison in it. Scores become probabilities via Plackett–Luce
with a temperature fitted on held-out races, because a ranker's raw score scale
means nothing probabilistically.

**The simulation earns its place.** It models what the ranker structurally
cannot see: per-team retirement hazard, race-to-race pace variance, safety cars,
and how much a given circuit lets pace overcome grid position. The blend weight
between ranker and simulation is chosen by log loss on validation races.

**Leakage control.** Every rolling feature goes through one helper that calls
`.shift(1)` before the window, so there is exactly one place to get it wrong.
`tests/test_leakage.py` constructs cases where a leak would be visible —
including the off-by-a-window bug you get from `.rolling().shift()` instead of
`.shift().rolling()` — and asserts it is not there.

## Validation

`make validate` runs 16 checks over the raw tables and exits non-zero on any
error, so it works as a CI gate. Contiguous round numbering, referential
integrity between results and races and drivers, exactly one winner per race,
no duplicate finishing positions, positions and points in range, races in
chronological order, and — the important one — that no stored weather forecast
was fetched after the session it describes.

`tests/test_validate.py` injects each defect and asserts the corresponding check
fires. A validation suite that has never caught anything is not evidence.

## Layout

```
src/f1pred/
  config.py       paths, constants, rate limits
  http_cache.py   rate-limited, disk-cached, resumable HTTP
  store.py        DuckDB schema and idempotent upserts
  ingest/         jolpica, FastF1, Open-Meteo
  validate.py     data quality checks
  features.py     feature engineering, leakage guards
  model.py        XGBRanker wrappers
  simulate.py     Plackett–Luce and Monte Carlo
  backtest.py     walk-forward evaluation and hyperparameter fitting
  predict.py      one race, both models, logged output
  report.py       static HTML dashboard
predictions/      timestamped predictions, committed — the track record
reports/          generated dashboard
```

## Known limits

Roughly 80% of an F1 result is which car is fastest, and about one race in six
turns on a safety car or a failure nothing knowable beforehand predicts. A
well-built model should land near 35–50% on winners in a season with a dominant
car and lower in a close one. Anything claiming much more is leaking data.

The 2026 regulation reset makes earlier seasons partially obsolete. That is
handled by weighting recent races more heavily, with the weight fitted rather
than assumed — see Results.

The circuit features are learned from prior visits, so a brand-new track has
none. The simulation falls back to a neutral overtaking assumption and the
dashboard flags the prediction as coming from an unseen circuit.

## Licence

MIT. Data belongs to its sources; see jolpica's
[terms of use](https://github.com/jolpica/jolpica-f1/blob/main/TERMS.md).
