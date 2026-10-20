# Methodology

How f1pred turns race data into forecasts, and how it checks them. Figures
quoted here and in the README come from `reports/*.json`, and are tested
against those files (`tests/test_readme_matches_reports.py`). Section 11 has
the full results.

## 1. The problem

A race is a ranking of ~20 drivers, so the target is the *order*, not a
yes/no per driver. The model is an XGBoost ranker (`rank:ndcg`), trained with
one query group per race: each race contributes comparisons between its
drivers (the race ranker pairs each driver with the eight it currently ranks
highest, the qualifying ranker with the whole field)
instead of a single "winner" label (about 190 races since 2018 would
be 190 positive examples for a classifier). Relevance is `24 - finishing
position`, so retirements keep their classified place.

Three forecasts are logged per weekend, from different information:

| Stage | Known | Model |
|---|---|---|
| Before practice | history only | qualifying ranker predicts the grid, race ranker scores the *projected* weekend |
| After practice | history and this weekend's FP1-FP3, plus sprint qualifying on a sprint weekend | as above, with practice feeding the qualifying ranker and sprint qualifying blended into its scores |
| After qualifying | the official starting grid and the qualifying result | race ranker on the real grid |

Once qualifying has run, the qualifying forecast is shown only for comparison.
It never stands in for the result.

Each stage is graded on its own. Before practice, the race being forecast is
scored with its practice columns hidden, and the live Wednesday forecast fits
its temperature on past races scored the same way. Against the championship
order on 2024 onward, win log loss is -0.302 (-0.559 to -0.039) before practice
and -0.414 (-0.667 to -0.154) after it. Both are clearly better.

## 2. Data

| Source | Used for |
|---|---|
| [jolpica-f1](https://github.com/jolpica/jolpica-f1) | results, qualifying, sprints, standings, calendar |
| [OpenF1](https://openf1.org) | the official starting grid (penalties applied) and per-session entry lists, 2023+ |
| [OpenF1](https://openf1.org) | practice pace: per-driver aggregates of FP1-FP3 laps (fetched with [FastF1](https://github.com/theOehrly/Fast-F1) before October 2026), and the sprint qualifying classification |

Everything lands in DuckDB. `raw_*` tables hold what the sources returned, and
derived data (features) never overwrites them. HTTP responses are cached, jolpica
is paced under its hourly budget, OpenF1 under its per-minute one, and network
errors are retried with backoff. A failure in a fallback source (OpenF1) never
stops a run.

**Who is in the race** (`weekend.race_entries`), most official first:
qualifying classification, then OpenF1's qualifying session, then the latest
FP2/FP3/sprint session, and only then the previous race's field, labelled as
an assumption. FP1 never sets the field, because teams must run rookies in it.
Otherwise a forecast built on the previous race's field can include a driver
who has been replaced and leave out the one who returned.

**Where each car starts** (`weekend.resolve_grid`): jolpica's results grid,
then OpenF1's official grid, then the qualifying order (provisional, penalties
missing, and recorded as such). Each candidate grid is checked: no driver twice,
no two cars in one slot, every driver entered in this race (otherwise it is a
stale grid from another session), at least 90% coverage. Pit-lane starters go
to the back of the grid, not read as grid 0, which would put them ahead of
pole.

**Penalties announced in advance** (`penalties.py`): a team can confirm days
ahead that a car takes a new power unit and starts from the back, long before
the official grid exists. `grid_penalties.json` lists such penalties, entered
by hand, each with the announcement it came from. They move the car down the
projected grid and every grid drawn before qualifying, and the qualifying
order after it, as the stewards would: N places back lands the car behind the
one that qualified there, and back-of-grid starters keep their qualifying
order. They are never applied to an official grid, which has them already, or
to the training history, where the real grid is known. A penalty announced
after a forecast was logged is new information, so the next run logs a new
forecast for that stage beside the earlier one, which stays as it was.

A back-of-grid penalty changes qualifying too, because there is little left to
qualify for. Of 21 front-runners (recent qualifying average in the top six)
who started from the back since 2018, two in three qualified three or more
places below their form, against one in six on a normal weekend, and only
half reached the top ten, against 94%. Their qualifying forecast is therefore
a mix of the usual lap and a lap held back to about 14th, weighted by that
excess (0.6). Place penalties are left alone, because the grid still depends
on the lap.

`make validate` runs 21 checks before anything is modelled: coverage,
referential integrity, positions, the retirement flag against laps completed,
grid slots, and time ordering (practice before qualifying and the race).

## 3. Features without leakage

One row per driver per race. Every feature uses only races strictly earlier:

* rolling windows shift one race before aggregating,
* team features collapse to one value per team per race *before* shifting,
  because shifting a team's rows would step back to the teammate in the same
  race,
* championship standing is the standing after the previous round,
* the season-progress feature divides by the calendar length, known before the
  season. Dividing by the rounds already run would give the same race a
  different value depending on when features were built.

Two tests treat the feature code as a black box:

* **truncation invariance**: build features from data that stops after race
  *k* and from all the data. Every feature of every race up to *k* must be
  identical (`tests/test_pipeline.py`, and `make verify` on the real data).
* **result tampering**: reverse one race's result. No feature of that race or
  an earlier one may move, and later ones must.

Race-model feature groups (27 features): driver form (median finish over the
last 3 and 5 races, points, podium and top-10 rates, places gained,
reliability), team form and car pace (rolling gap to pole of the team's best
car, the team's recent qualifying places), circuit history, and the grid,
qualifying position and gap to pole, which before qualifying are the
qualifying model's projection.

Two groups were tested and left out of the race model:

* the driver's own qualifying record (average grid and qualifying place, pole
  rate, one-lap gap to pole). It reaches the race through the grid already, and
  inside the race model it counted twice: adding it back made 2022-23 clearly
  worse, both after qualifying and before,
* the teammate qualifying gap, which also made 2022-23 worse after
  qualifying.

The qualifying model still uses the qualifying record and the teammate
head-to-head, where they are direct one-lap evidence.
Missing values stay missing (XGBoost routes them) rather than being filled with
a number the model would read as real: a driver with no teammate time has no
teammate gap, not a gap of zero.

## 4. From scores to probabilities

A ranker's score has no probabilistic meaning, so it is turned into one
**finishing-position distribution** `P[driver, position]` whose rows and
columns each sum to one. Win, podium, top-5, top-10 and expected finish are all
read from it, so they cannot contradict each other.

1. **Plackett-Luce**: `P(win) ∝ exp(score / T)`. The full order is sampled
   exactly with the Gumbel-max trick. The temperature `T` is fitted by maximum
   likelihood on the winners of the 24 races *before* each forecast (the
   2022-23 value until there are ten).
2. **Monte Carlo** (10,000 races): pace noise proportional to the field's
   score spread, a safety-car draw that widens it, an independent retirement
   draw per car from its driver and team DNF rates, a pull toward the grid that
   is stronger where the circuit's history says overtaking is hard, and, before
   qualifying, a grid drawn from the qualifying model for every run.
3. **Mixture**: `w · PL + (1 - w) · MC`, with `w` fitted on 2022-23. A mixture of
   two such matrices keeps rows and columns summing to one. A 0.2% uniform
   share is mixed in last, so no driver is ever at exactly zero.

This is a stochastic layer over the learned ranking, not a physics simulation.
It does not model tyres, pit stops, weather, safety-car timing, penalties, team
orders or failures shared by a team's two cars.

## 5. Evaluation

Walk-forward throughout: each race is forecast by models trained only on races
before it (`backtest.oos_scores` is the one place that enforces this, and a
test checks that no model ever sees its target race). There is no random split
anywhere.

| Window | Use |
|---|---|
| 2018-2021 | training only |
| 2022-2023 | settings fitted here: recency weight, temperature, mixture weight |
| 2024-now | reported. Never used to fit settings, only to veto a change that is clearly worse (section 6), and 2024 picked the practice design |

**Baselines** get probabilities the same way the model does: Plackett-Luce over
`-log(rank)` with a temperature fitted on earlier races. The grid baseline is
the one that matters. A fixed, hand-picked decay would make the model's lead
on log loss look larger than it is.

**Metrics.** Ranking: NDCG@3, NDCG@5, winner called, podium and top-5 overlap,
Spearman, Kendall. Probability: win log loss, win Brier (over the field),
podium and top-10 Brier (per driver), and reliability diagrams with expected
calibration error. Ranking and probability are reported separately. The
calibration error averages over every driver in every race, most of them with
chances near zero, so it reads small. The most confident bands are reported
separately (section 10).

**Uncertainty.** Model and baseline are compared race by race, with a 95%
bootstrap interval over races for each difference. When the interval includes
zero, the report says neither is ahead.

**Calibration.** Temperature scaling is the calibrator: it fits the within-race
softmax structure and is refitted only on past races. Isotonic regression was
considered and not used, because calibrating each driver's probability on its
own breaks the constraint that a race's win probabilities sum to one.
`make experiments` compares no calibration, a fixed temperature and the rolling
one, and the mixture against each of its halves. It also tests fitting the
temperature on the order of the first three finishers instead of the winner
alone, aimed at the over-confident podium and top-10 calls: their calibration
improves in both windows, but win log loss on 2022-23 gets worse (0.928 against
0.883), so the winner-only fit stays.

## 6. Experiments (`make experiments` → `reports/experiments.json`)

Each uses the same walk-forward. Decisions are read off 2022-23 and checked on
2024 onward. A check on 2024 onward can veto a change but never adopt one.
The rule for a change is an improvement on 2022-23 with a 95% interval clear
of zero. Many comparisons are made, so a marginal pass can be luck. To
stop choices flipping with every refresh of the data, each was taken once, on
the final configuration, and later runs report without re-deciding.

* **Ablation**: the grid alone, the grid plus each feature group, the full
  model, and the full model minus each group. A group whose removal costs
  nothing duplicates another.
* **Redundancy**: feature pairs with |Spearman| ≥ 0.8, and grouped permutation
  importance (each group shuffled between drivers within a race, out of sample)
  beside mean |SHAP| shares.
* **Pre-qualifying ablation**: the groups left out of the race model, added
  back when the grid columns hold the qualifying model's projection. A group
  can be harmless once the real grid is known and still count twice here.
* **Feature definitions**: median against mean recent form. The median lowered
  2022-23 log loss by 0.047 (interval 0.003 to 0.093) and is in use. On 2024 onward
  the difference is not clear, so this is the most marginal decision in the
  project. A teammate gap measured within one session instead
  of on each driver's best lap was also tried, made no difference, and was
  dropped.
* **Practice**: does this weekend's FP1-FP3 say anything about this weekend's
  qualifying beyond history, and about the race beyond the official grid?
  Practice is treated as current-weekend evidence only: gaps to each
  session's fastest car (over 7% counts as no representative lap), the latest
  session's gap and rank, the teammate and team gaps, the FP1-FP2-FP3 trend and
  consistency. Stored for 59 weekends of 2024-26, compact numbers only, and
  only sessions that started before qualifying. The rule was fixed first: pick
  the design (compact or detailed) on 2024 by NDCG@5, then keep it only if it
  is clearly better than history alone on the untouched 2025-26 weekends on
  NDCG@5 or position error and clearly worse on nothing. The detailed design
  passed: position error -0.280 places (-0.429 to -0.128), pole log loss
  -0.203 (-0.404 to -0.009), NDCG@5 +0.028 (+0.000 to +0.059). It feeds the
  qualifying model only. It adds nothing to the race once the grid is known.
  The live pipeline fetches the current weekend's sessions. Each weekend logs
  three forecasts: before practice (history only, logged on the Wednesday of
  race week), after practice and after qualifying.
* **Qualifying order** (`quali_ordering`): the qualifying ranker is trained
  with NDCG's exponential gains, which concentrate on the front of the grid,
  so it orders the back loosely and overstates backmarkers' top-ten chances.
  Two fixes were tested with the rule fixed in advance: kept only if clearly
  better on 2022-23 on pole log loss, NDCG@5, Spearman or position error, and
  clearly worse on none of those, NDCG@3 or top-ten overlap in either window.
  The median of recent qualifying positions made no clear difference. Linear
  gains ordered the whole field clearly better in both windows but were
  clearly worse on NDCG@5 on 2024 onward, so neither is used.
* **Retirements in both halves** (`retirements`): Plackett-Luce has no
  retirements, so with a 0.7 blend only 30% of each car's retirement risk
  reaches the published chances. Rule fixed in advance: used only if clearly
  better on 2022-23 on podium or top-10 Brier, and clearly worse on none of
  those or win log loss in either window. Calibration error fell on 2022-23
  (podium 2.9% to 1.5%, top ten 7.2% to 4.9%), but podium
  Brier was not clearly better (-0.0015, -0.0033 to +0.0001), so
  it is not used.
* **Qualifying pairing** (`quali_pairing`): XGBoost's default pairs each
  driver only with the eight it currently ranks highest. Pairs across the whole
  field, with the same exponential gains, were judged on the qualifying-order
  rule above. On 2022-23 position error was -0.198 (-0.332 to -0.066) and Spearman
  +0.027 (+0.009 to +0.046), both clearly better, with nothing clearly worse in either
  window, so the qualifying ranker uses it.
* **Sprint qualifying** (`sprint_quali`): from 2024, sprint qualifying runs
  before Grand Prix qualifying. Its order is blended into the qualifying
  model's scores (`backtest.sprint_blend`, rescaled to the model's spread so the
  temperature still holds), using the OpenF1 classification the live forecast
  uses. Rule fixed in advance: weight picked from 0.25, 0.5 and 1 on the 2024
  sprint weekends by position error, then kept only if clearly better on
  2025-26 on position error or NDCG@5 and clearly worse on none of those, NDCG@3
  or pole log loss. Weight 0.5 was picked. On the twelve 2025-26 sprint
  weekends position error was -0.713 (-1.120 to -0.329) and pole log loss
  -0.248 (-0.392 to -0.118), so it is in use. On a sprint weekend the after-practice
  forecast waits for sprint qualifying.
* **Circuit type** (`circuit_type`): street and temporary circuits against
  permanent ones, as each driver's and each team's recent finishing at
  circuits of the same type, so a circuit seen once or never borrows from the
  rest of its type. Rule fixed in advance: kept only if clearly better on
  2022-23 on win log loss or NDCG@5, before or after qualifying, and clearly
  worse on none of those, NDCG@3 or podium Brier in either window. It made no
  clear difference on 2022-23 (win log loss +0.012, -0.039 to +0.065, after
  qualifying), so it is not used. Before qualifying on 2024 onward it was slightly
  better, -0.061 (-0.130 to +0.005), but not clearly.
* **Season projection**: how to score a driver for the rest of the season,
  graded on every title-backtest checkpoint against final driver and team
  points, the final gap between teammates and the champion's probability.
  Decided on 2019-22, checked on 2023 onward.
* **XGBoost settings**: a handful of candidates on two separate validation
  windows, without intervals. Not a search: no candidate is best on both
  windows (depth 3 leads on 2022-23 and trails on 2021), so the settings in
  use are kept.

## 7. Explanations

SHAP values come from XGBoost's own TreeSHAP (`pred_contribs=True`, identical
to `shap.TreeExplainer`). The forecast averages five seeds, each standardised
within the race, so each seed's contributions are centred and scaled the same
way and then averaged. The result sums exactly to the published score
(`tests/test_model.py`), so the explanation is of the forecast that was made.
Near-duplicate features are grouped before a sentence is written.

SHAP describes what the model relied on. It is not causal: "helped by starting
position" means the model scored the driver up for it, not that the grid slot
caused a result.

## 8. Championship projection

The rest of the season is simulated 10,000 times with the race noise model,
sprints included (3-2-1 in 2021, 8-1 from 2022, from the calendar). Each
driver's strength is the median of the model's own scores for their last eight
real weekends, each scored on that race's full field. The alternative was to
score a synthetic "typical weekend", which copies recent form into the
circuit columns (counting it twice) and carries each driver's teammate gap.
Against that alternative, on the 2019-22 title-backtest checkpoints the new one
was clearly better on final driver points and on the champion's probability,
and on 2023 onward it was clearly better on the final gap between teammates and level
on the rest (`season_projection` in `reports/experiments.json`). Points
already scored are carried. Constructors' points are summed from the results as
awarded, because summing the current drivers' totals credited a team with points
its new driver scored elsewhere. A per-team and a smaller per-driver pace
offset, held all season, widen the range to its calibrated 10th-90th coverage
(`make calibrate-spread`: fitted on 2019-22, graded on 2023-25). The clinch
arithmetic counts every point still available, sprints and bonus points
included.

## 9. Reproducibility and provenance

* `uv.lock` pins every dependency, and CI installs from it on Python 3.11-3.13.
  The reports were generated on 3.12 (pinned in `.python-version`) with
  XGBoost 3.4.1, and each report records its versions. XGBoost 3.4 needs
  Python 3.12+, so on 3.11 the lock resolves 3.2 and figures may differ in the
  last decimals.
* `make demo` runs the whole pipeline on a seeded synthetic championship with
  no network. The same fixture drives `tests/test_pipeline.py` in CI.
* Every forecast records its stage, grid and entry sources, the model and
  feature versions (hashes), its training cut-off, a hash of the results it was
  trained on, the git commit and the library versions.

Where the code lives:

```
src/f1pred/
  ingest/            jolpica, OpenF1 (grid, entries), practice (OpenF1 lap aggregates)
  store.py           DuckDB schema (raw tables are never overwritten)
  validate.py        21 data checks
  weekend.py         who is in the race, and where each car starts
  penalties.py       grid penalties announced before the official grid
  features.py        temporal feature layer
  model.py           XGBoost rankers, ensemble SHAP
  probability.py     Plackett-Luce, calibration, one coherent distribution
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
reports/             measured figures, which the docs are tested against
tests/               unit, leakage, pipeline (end-to-end on the demo data)
```

## 10. Limitations

* Win probabilities are well calibrated overall (expected calibration error
  under 1%), though the few win calls above 70% came true less often than
  stated. The most confident podium and top-10 calls are over-confident:
  podium calls averaging 94% came true 79% of the time, and top-10 calls
  averaging 96% came true 87% (`reliability` in `reports/backtest.json`),
  and mid-range top-ten chances are under-confident. The temperature is fitted
  on winners only. Fitting it on the first three places improves those calls
  but worsens win log loss on 2022-23 (section 5), so it is not used.
* GitHub runs scheduled jobs late when it is busy (every four to six hours
  over the Singapore weekend instead of hourly), so forecasts can be logged
  hours later than planned, though still before the session.
* Races are noisy and the sample is small. 65 test races is enough to
  separate the model from naive baselines. Against the calibrated starting
  grid it is level on probabilities and behind on ordering the whole field
  (Kendall rank correlation), once qualifying has run.
* Not modelled: weather, tyre and pit strategy, in-race penalties, team orders,
  upgrades, correlated failures, safety-car timing.
* The qualifying forecast orders the back of the grid loosely, so
  backmarkers' top-ten chances in qualifying are too high (section 6).
* Practice before October 2026 was fetched with FastF1, which cannot reach
  F1's timing data from GitHub's runners (every session came back without
  laps there), so no after-practice forecast was logged before then. Practice
  now comes from OpenF1 with the same lap definitions (`ingest/practice.py`).
  Recomputed from OpenF1 on 13 past sessions, every best lap matched to the
  millisecond and 95% of long-run medians were within 0.1 s (largest gap
  0.32 s). The stored history is kept as fetched.
* Before the official grid is published, only penalties announced in advance
  and entered in `grid_penalties.json` are applied. Any other penalty is
  missed until the official grid appears.
* A fast car starting from the back is rare in training: 21 front-runners
  since 2018. The ranker barely holds the start against it and the
  simulation does, so the blend relies on the two balancing out. Against
  those 21 races (2 wins, 6 podiums, 15 top tens) they roughly do, but that
  is too few to say how well calibrated such a forecast is.
* The season projection assumes current form holds, widened by a calibrated
  but simple pace-drift term.
* Practice data covers 59 weekends (2024-26). Its gain is clear on position
  error and pole probability, borderline on NDCG@5.

## 11. Results in detail

Every figure here comes from `reports/*.json`, and
`tests/test_readme_matches_reports.py` checks it against those files. The
headline evaluation covers the 65 races from 2024 round 1 to 2026 round 17.

### What the measures mean

- **Log loss**: how surprised the forecast was by what happened. Giving the
  winner a 2% chance costs far more than giving them 40%. Lower is better.
- **Brier score**: the average squared gap between a forecast chance and what
  happened (1 or 0). Lower is better.
- **NDCG@3, NDCG@5**: how well the top three or five were put in order, from 0
  to 1, with the top places counting most.
- **Kendall**: how well the whole predicted order matches the result, from -1
  to 1.
- **Calibration error**: the average gap between a stated chance and how often
  it came true.
- **95% interval**: the range the true difference probably lies in. If it
  includes zero, the two forecasts cannot be told apart.
- **Walk-forward**: each race is forecast by a model trained only on races
  before it, the way it would have run live.

### Race and qualifying

After qualifying, against the calibrated grid and championship order
(`reports/backtest.json`, generated by `make backtest`):

| Approach | NDCG@3 | NDCG@5 | Winner | Podium overlap | Win log loss | Podium Brier |
|---|---|---|---|---|---|---|
| **This model** | 0.856 | 0.897 | **68%** | 1.94 of 3 | **1.173** | **0.067** |
| Grid order, calibrated | **0.861** | **0.900** | 60% | **2.06 of 3** | 1.309 | 0.068 |
| Championship order | 0.751 | 0.803 | 29% | 1.68 of 3 | 2.169 | 0.095 |

Each comparison is paired race by race, with a 95% bootstrap interval.
"Inconclusive" means the interval includes zero.

| Stage | Metric | Model | Reference | Difference (95% interval) | Verdict |
|---|---|---|---|---|---|
| After qualifying | NDCG@3 | 0.856 | 0.861 (grid) | -0.005 (-0.030 to +0.020) | inconclusive |
|  | NDCG@5 | 0.897 | 0.900 (grid) | -0.003 (-0.019 to +0.014) | inconclusive |
|  | Winner called | 68% | 60% (grid) | +8 pts (-3 to +18) | inconclusive |
|  | Win log loss | 1.173 | 1.309 (grid) | -0.136 (-0.349 to +0.065) | inconclusive |
|  | Podium Brier | 0.067 | 0.068 (grid) | -0.000 (-0.006 to +0.005) | inconclusive |
|  | Kendall | 0.522 | 0.549 (grid) | -0.027 (-0.052 to -0.002) | **grid better** |
| After practice | Win log loss | 1.755 | 2.169 (championship order) | -0.414 (-0.667 to -0.154) | **model better** |
|  | Podium Brier | 0.085 | 0.095 (championship order) | -0.010 (-0.017 to -0.002) | **model better** |
|  | Top-5 overlap | 3.66 of 5 | 3.43 of 5 (championship order) | +0.231 (+0.031 to +0.462) | **model better** |
| Before practice | Win log loss | 1.867 | 2.169 (championship order) | -0.302 (-0.559 to -0.039) | **model better** |
|  | Podium Brier | 0.089 | 0.095 (championship order) | -0.006 (-0.014 to +0.002) | inconclusive |
|  | Top-5 overlap | 3.66 of 5 | 3.43 of 5 (championship order) | +0.231 (+0.015 to +0.462) | **model better** |
| Qualifying | Pole log loss | 1.681 | 2.105 (recent qualifying form) | -0.424 (-0.653 to -0.194) | **model better** |
|  | NDCG@5 | 0.857 | 0.810 (recent qualifying form) | +0.046 (+0.025 to +0.069) | **model better** |

After qualifying the model's win log loss is 0.136 lower than the grid's, but
the interval runs from -0.35 to +0.07. Over the whole field (Kendall rank
correlation), the grid is clearly better.

**Calibration.** When the model says X%, the average gap to what happened
(expected calibration error, per driver) is 0.5% for wins, 2.4% for podiums
and 6.0% for the top ten. That average covers every driver in every race, most
of them with chances near zero, so it reads small. The most confident bands
are the weak spot (section 10), and the method page shows every band.

### What each choice is worth

**Calibration and the simulation.** The same out-of-sample scores on 2024
onward, turned into probabilities six ways:

| Probability step | Win log loss | Podium Brier | Calibration error, win |
|---|---|---|---|
| Raw scores (temperature 1, no simulation) | 1.686 | 0.0747 | 3.8% |
| Temperature fitted on 2022-23, held | **1.226** | 0.0679 | 0.3% |
| Temperature refitted on the last 24 races (in use) | 1.229 | 0.0683 | 0.4% |
| Refitted, Plackett-Luce only | 1.301 | 0.0716 | 0.3% |
| Refitted, simulation only | 1.256 | **0.0651** | 0.5% |
| Refitted on the first three finishers, not just the winner | 1.261 | 0.0662 | 1.0% |

Calibration is most of the gain. A held and a refitted temperature are level,
and mixing the two probability models beats either alone on log loss. Fitting
the temperature on the first three finishers was tested against the
over-confident podium and top-10 calls: it improves their calibration on both
2022-23 and 2024 onward, but worsens win log loss on 2022-23 (0.928 against
0.883), so it is not used.

**Ablation: what adds information once the grid is known.**

- **Everything together vs the grid alone**: log loss 0.840 vs 1.628 on
  2022-23 (difference +0.788, interval +0.462 to +1.155), but 1.229 vs 1.326 on
  2024 onward (not clear).
- **Adding one block at a time to the grid** (log loss on 2022-23): grid only
  1.628, + driver form 0.957, + team form 1.220, + both 0.914, full model
  0.840. Recent driver form carries most of what the model adds beyond the
  grid. On 2024 onward none of these differences is clear.
- **Driver form and team form** are the groups whose removal clearly costs
  log loss on 2022-23: +0.186 (+0.046 to +0.348) and +0.082 (+0.004 to
  +0.168). Removing **circuit history** clearly hurts ranking (NDCG@5 -0.011,
  -0.021 to -0.001).
- **Qualifying position and gap to pole** duplicate the grid (correlation
  0.95 with it): removing them changes nothing clearly. They stay because they
  are how a car that took a grid penalty is recognised as fast.
- **Tested and left out**: the driver's own qualifying record (average grid
  and qualifying place, pole rate, one-lap gap) made 2022-23 clearly worse
  after qualifying, +0.078 (+0.027 to +0.138), and before it, +0.043 (+0.009
  to +0.078). It reaches the race through the grid already, so inside the race
  model it counted twice. The teammate qualifying gap also made it worse,
  +0.049 (+0.010 to +0.089). Both, as rolling averages, still feed the
  qualifying model.
- On 2024 onward, no single group's removal or addition is clear either way.

**Practice pace**, with and without, on the 35 untouched weekends of 2025-26
(section 6):

| Qualifying forecast, with practice minus without (2025-26) | Difference (95% interval) |
|---|---|
| Qualifying position error (places) | -0.280 (-0.429 to -0.128) |
| Pole log loss | -0.203 (-0.404 to -0.009) |
| NDCG@5 | +0.028 (+0.000 to +0.059) |
| NDCG@3 | +0.040 (-0.006 to +0.089) |

Practice clearly improves the qualifying forecast, so it feeds the qualifying
model. After qualifying it adds nothing beyond the official grid (race log loss
+0.029, -0.051 to +0.117), so the race model never sees it.

**Qualifying order: tested, not used.** Linear gains against the exponential
gains in use, on 2024 onward: position error -0.251 (-0.388 to -0.116) places,
but NDCG@5 -0.018 (-0.034 to -0.002). The race forecast depends most on the
front of the grid, so they are not used.

**Qualifying pairing: in use.** Pairing across the whole field: position error
-0.198 (-0.332 to -0.066) places and Spearman +0.027 (+0.009 to +0.046) on
2022-23, nothing clear on 2024 onward.

**Sprint qualifying: in use on sprint weekends.** On the twelve sprint
weekends of 2025-26, qualifying position error went from 3.44 to 2.72 places,
-0.713 (-1.120 to -0.329), and pole log loss from 1.899 to 1.650, -0.248
(-0.392 to -0.118).

**Circuit type: tested, not used.** No clear difference on 2022-23 (section 6).

### Championship projection

Graded on 27 checkpoints across 7 completed seasons (2019-2025), each
projected with a model trained only on races before it:

- **Who wins**: the favourite was right 89% of the time, Brier 0.077. That
  is a low bar: the favourite was simply the points leader at 26 of 27
  checkpoints, and backing the leader would have been right 85% of the time.
  Above 95% it is 14 for 14, but the 95% interval on that runs down to
  0.79, so a published 99.9% means the arithmetic says it's over, not one
  chance in a thousand. The misses are all in 2025's Piastri-Norris season
  (after rounds 8, 13 and 18).
- **How close**: every points total carries a 10th-90th percentile range,
  sized on 2019-22 and graded on 2023-25 (`make calibrate-spread`):

| Projection | Constructors inside the range | Width | Teammate gap inside | Target |
|---|---|---|---|---|
| Race luck only | 54% | 39 pts | 72% | 80% |
| **With the season-long pace term** | **86%** | 108 pts | 74% | 80% |

The pace term was sized on 2019-22, and on 2023-25 constructors land a little
wide of the target (86% against 80%). The teammate gap is still short of 80%:
on 2019-22 the ranges needed no extra driver term, but on 2023-25 they were too
narrow.
