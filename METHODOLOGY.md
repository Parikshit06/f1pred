# Methodology

How f1pred turns race data into forecasts, and how it checks them. Figures
quoted here come from `reports/*.json`, and the README's headline numbers are
tested against those files (`tests/test_readme_matches_reports.py`).

## 1. The problem

A race is a ranking of ~20 drivers, so the target is the *order*, not a
yes/no per driver. The model is an XGBoost ranker (`rank:ndcg`), trained with
one query group per race: each race contributes every pairwise comparison
inside it instead of a single "winner" label (about 190 races since 2018 would
be 190 positive examples for a classifier). Relevance is `24 - finishing
position`, so retirements keep their classified place.

Three forecasts are logged per weekend, from different information:

| Stage | Known | Model |
|---|---|---|
| Before practice | history only | qualifying ranker predicts the grid, race ranker scores the *projected* weekend |
| After practice | history and this weekend's FP1-FP3 | as above, with practice feeding the qualifying ranker |
| After qualifying | the official starting grid and the qualifying result | race ranker on the real grid |

Once qualifying has run, the qualifying forecast is shown only for comparison.
It never stands in for the result.

## 2. Data

| Source | Used for |
|---|---|
| [jolpica-f1](https://github.com/jolpica/jolpica-f1) | results, qualifying, sprints, standings, calendar |
| [OpenF1](https://openf1.org) | the official starting grid (penalties applied) and per-session entry lists, 2023+ |
| [FastF1](https://github.com/theOehrly/Fast-F1) | practice pace: per-driver aggregates of FP1-FP3 laps, this weekend only |

Everything lands in DuckDB. `raw_*` tables hold what the sources returned, and
derived data (features) never overwrites them. HTTP responses are cached, jolpica
is paced under its hourly budget, OpenF1 under its per-minute one, and network
errors are retried with backoff. A failure in a fallback source (OpenF1) never
stops a run.

**Who is in the race** (`weekend.race_entries`), most official first:
qualifying classification, then OpenF1's qualifying session, then the latest
FP2/FP3/sprint session, and only then the previous race's field, labelled as
an assumption. FP1 never sets the field, because teams must run rookies in it.
This fixed a real failure: a forecast built on the previous race's field
included a driver who had been replaced and left out the one who returned.

**Where each car starts** (`weekend.resolve_grid`): jolpica's results grid,
then OpenF1's official grid, then the qualifying order (provisional, penalties
missing, and recorded as such). Each candidate grid is checked: no driver twice,
no two cars in one slot, every driver entered in this race (otherwise it is a
stale grid from another session), at least 90% coverage. Pit-lane starters go
to the back of the grid. They had been read as grid 0, ahead of pole.

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
  season. It used to divide by the number of rounds already run, so the same race
  got a different value depending on when features were built.

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
| 2024-now | reported, never used to choose anything |

**Baselines** get probabilities the same way the model does: Plackett-Luce over
`-log(rank)` with a temperature fitted on earlier races. The grid baseline is
the one that matters. An earlier version gave it a fixed, hand-picked decay,
which made the model's lead on log loss look larger than it was.

**Metrics.** Ranking: NDCG@3, NDCG@5, winner called, podium and top-5 overlap,
Spearman, Kendall. Probability: win log loss, win Brier (over the field),
podium and top-10 Brier (per driver), and reliability diagrams with expected
calibration error. Ranking and probability are reported separately.

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
2024-. The rule for a change is an improvement on 2022-23 with a 95% interval
clear of zero. Many comparisons are made, so a marginal pass can be luck. To
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
  2022-23 log loss by 0.047 (interval 0.003 to 0.093) and is in use. On 2024-
  the difference is not clear. An earlier configuration measured the same
  gain with an interval just touching zero, so this is the most marginal
  decision in the project. A teammate gap measured within one session instead
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
  clearly worse on NDCG@5 on 2024-, so neither is used.
* **Season projection**: how to score a driver for the rest of the season,
  graded on every title-backtest checkpoint against final driver and team
  points, the final gap between teammates and the champion's probability.
  Decided on 2019–22, checked on 2023–.
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
real weekends, each scored on that race's full field. An earlier version
scored a synthetic "typical weekend", which copied recent form into the
circuit columns (counting it twice) and carried each driver's teammate gap.
Against that old method, on the 2019–22 title-backtest checkpoints the new one
was clearly better on final driver points and on the champion's probability,
and on 2023– it was clearly better on the final gap between teammates and level
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

## 10. Limitations

* Win probabilities are well calibrated overall (expected calibration error
  under 1%), though the few win calls above 70% came true less often than
  stated. The most confident podium and top-10 calls are over-confident:
  podium calls averaging 94% came true 78% of the time, and top-10 calls
  averaging 96% came true 87% (`reliability` in `reports/backtest.json`),
  and mid-range top-ten chances are under-confident. The temperature is fitted
  on winners only. Fitting it on the first three places improves those calls
  but worsens win log loss on 2022-23 (section 5), so it is not used.
* Races are noisy and the sample is small. 63 test races is enough to
  separate the model from naive baselines. Against the calibrated starting
  grid it is level on probabilities and behind on ordering the whole field
  (Kendall rank correlation), once qualifying has run.
* Not modelled: weather, tyre and pit strategy, in-race penalties, team orders,
  upgrades, correlated failures, safety-car timing.
* The qualifying forecast orders the back of the grid loosely, so
  backmarkers' top-ten chances in qualifying are too high (section 6).
* Before the official grid is published, penalties are unknown and the
  qualifying order stands in.
* The season projection assumes current form holds, widened by a calibrated
  but simple pace-drift term.
* Practice data covers 59 weekends (2024-26). Its gain is clear on position
  error and pole probability, borderline on NDCG@5.
