"""Feature engineering.

One row per (season, round, driver). Every feature is derived only from races
that happened strictly earlier, which is enforced by sorting chronologically
and calling .shift(1) before every rolling window. If you add a feature, it
must go through the same discipline or tests/test_leakage.py will fail.

Features are grouped by when they become knowable, because the two models see
different slices:

    BASE      known before the weekend starts (form, reliability, circuit history)
    PRACTICE  known after Friday running (FastF1 pace; optional, may be null)
    GRID      known only after qualifying (grid slot, quali gaps)

    quali model : BASE + PRACTICE          -> predicts the grid
    race model  : BASE + PRACTICE + GRID   -> predicts the finish
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .store import connect

log = logging.getLogger(__name__)

BASE_FEATURES = [
    "drv_avg_finish_3",
    "drv_avg_finish_5",
    "drv_avg_grid_5",
    "drv_points_rate_5",
    "drv_dnf_rate_10",
    "drv_experience",
    "team_avg_finish_3",
    "team_avg_finish_5",
    "team_points_rate_5",
    "team_dnf_rate_10",
    "champ_position_before",
    "champ_points_before",
    "drv_circuit_avg_finish",
    "team_circuit_avg_finish",
    "circuit_overtaking_score",
    "circuit_dnf_rate",
    "drv_teammate_quali_edge",
    "season_progress",
    # Car pace. About 80% of an F1 result is which car is fastest, and until
    # these were added the model had no direct way to express that before
    # qualifying - it had to infer pace from finishing positions, which are a
    # lagging and much noisier proxy. Rolling qualifying gap to pole is the
    # cleanest public signal of car performance there is.
    "team_pace_gap_pct",
    "drv_pace_gap_pct",
    "team_pace_trend",
    # Racecraft, separated from pace: who actually gains places on Sunday.
    "drv_positions_gained_5",
    "drv_podium_rate_10",
    "drv_top10_rate_10",
    "drv_circuit_starts",
    "circuit_pole_win_rate",
    "drv_avg_quali_5",
    "drv_pole_rate_10",
    "team_avg_quali_5",
]

# Tried and cut, all measured on the 2024-26 walk-forward:
#   drv_teammate_race_edge / drv_beats_teammate_rate - gap to the other car in
#     the same garage. Drew 3.3% of model signal, podium accuracy 2.016 -> 1.968.
#   drv_races_at_team / drv_first_season_at_team - how settled a driver is in
#     the current car. Top-5 3.855 -> 3.887 (noise on 62 races), top-1 0.500 ->
#     0.484, and the switcher bias it targeted got wider, not narrower.
#   drv_form_shrunk / drv_is_rookie - form shrunk toward the car by experience.
#     Neutral.
#   drv_type_avg_finish / team_type_avg_finish / drv_type_edge and the
#     qualifying pair - form at circuits of the same character (power,
#     downforce, balanced), on the theory that "quick on power tracks" is a
#     real property no other feature could express. Tested as a career-long
#     expanding mean and as a six-visit rolling mean, against a run with the
#     features removed entirely:
#
#                   race: top5  podium  winner  ndcg    quali: pole  top5   ndcg
#       removed          3.839   1.984   0.581  0.889          0.258  3.710  0.833
#       expanding        3.823   1.968   0.565  0.887          0.226  3.597  0.819
#       rolling(6)       3.839   1.919   0.565  0.885          0.242  3.613  0.825
#
#     They drew 3-6% of the model's attention and cost accuracy on both
#     models, worst of all on the pole call they were meant to sharpen. The
#     effect they chase is real in the raw numbers but rests on three or four
#     races per driver per season; recent form and the car-pace features
#     already carry it with far more evidence behind them. Cut, and the
#     circuit-character map with them.
#   Blending the qualifying score with the team's best recent grid slot - the
#     one-line baseline that calls pole 35.5% of the time against the model's
#     27.4%, so it looked like the model was under-weighting the strongest
#     car-pace signal. Weight fitted on 2022-23 by NDCG, which picked 0.45 and
#     improved every metric on that window. On the held-out 2024-26 window it
#     did not transfer: pole 0.274 -> 0.226, NDCG 0.831 -> 0.829, with top-three
#     and log loss slightly better. Worse on the number it was built to fix, so
#     it is out. Second time a blend has looked good on the window it was fitted
#     on and evaporated off it.
#   drv_incident_rate_10 / team_mechanical_rate_10 - retirements split by cause,
#     so that "the driver crashed" and "the engine let go" stop sharing one
#     number. Built from an explicit status vocabulary; 43% of retirements carry
#     a generic status and classified as neither. The simulator drew the two as
#     separate channels with the car-caused half correlated inside a garage, at
#     a shared-cause fraction fitted from the data (both cars of a team have
#     suffered a mechanical retirement 16 times against 3.6 expected under
#     independence - 4.4x, phi 0.316).
#
#     It changed nothing that could be measured. Single-race metrics were
#     identical to four decimal places, which in hindsight is forced: the split
#     preserves each car's marginal retirement probability exactly, so anything
#     that scores one driver at a time is blind to it by construction.
#
#                          top5   podium  winner  log loss  Brier
#       one blended rate   3.903   1.952   0.581    1.1880  0.5741
#       split, correlated  3.903   1.952   0.581    1.1879  0.5742
#
#     The joint distribution was the honest place to look, so it was graded
#     there too - coverage of the constructors' 10th-90th band against real
#     final standings, 270 team-checkpoints. 50.0% either way, mean band width
#     38.3 against 38.5. Correlating a 4%-per-car event adds almost nothing next
#     to a season of pace variance. Cut, and the fitted shared-cause constant
#     with it.
#   team_pit_gap_5 - the team's fastest pit stop of a weekend against the
#     fastest anybody managed, rolled over five races: the one part of race
#     strategy that is in the public data and is a property of the team rather
#     than a per-race decision. 99.1% coverage, and a real spread (2026: Red
#     Bull 0.49s to Cadillac 3.24s), so the signal exists.
#
#                      top5   podium  winner  ndcg    log loss  Brier
#       without       3.855   1.984   0.581  0.8930    1.1837  0.5769
#       with          3.903   1.952   0.581  0.8902    1.1879  0.5742
#
#     Better on two metrics, worse on three, including log loss - the number
#     the settings are tuned against. On 62 races that is noise, and noise does
#     not earn a column. Cut.
#
# All of them are gone rather than kept "in case": the race model already sees this
# weekend's grid, which encodes the current driver-and-car combination
# directly, and grid/quali carries 48% of the signal against 30% for driver
# history. See `f1pred.cli bias` for the diagnostic that settled it.

PRACTICE_FEATURES = [
    "fp_long_run_gap_pct",
    "fp_best_gap_pct",
    "practice_available",
]

GRID_FEATURES = [
    "grid",
    "quali_position",
    "quali_gap_to_pole_pct",
    "quali_gap_to_teammate_pct",
]

# Practice feeds the QUALIFYING model only, and reaches the race forecast
# through the predicted grid rather than directly. Measured on 2025-26, with
# practice simulated at a range of correlations to true qualifying pace:
#
#   pre-qualifying forecast    no practice   rho=0.85   rho=0.95
#     pole called                   18%        32%        37%
#     rank correlation             0.71       0.82       0.83
#
#   post-qualifying forecast   no practice   rho=0.70   rho=0.90
#     top five                     3.82       3.82       3.79
#     winner called                 61%        61%        68%
#
# Real FP3-to-qualifying order correlation sits around 0.85-0.9, so the middle
# column is the realistic one: practice roughly doubles the pole hit rate and
# does nothing measurable once the actual grid is known. Including it in the
# race model would add three mostly-null columns of noise for no gain.
# Direct one-lap evidence. These lead the qualifying model, because
# qualifying history predicts qualifying and race results do not: a race
# outcome bundles strategy, traffic, reliability and incidents on top of pace.
QUALI_HISTORY_FEATURES = [
    "drv_avg_quali_3",
    "drv_avg_quali_5",
    "drv_pole_rate_10",
    "drv_quali_top3_rate_10",
    "team_avg_quali_5",
    "drv_pace_gap_pct",
    "team_pace_gap_pct",
    "team_pace_trend",
    "drv_teammate_quali_edge",
]

# Race-outcome context the qualifying model may still use, but only as
# background. Everything heavier - podium rate, points rate, championship
# position - was making the qualifying model a race-form model wearing a
# different hat: it drew 59% of its signal from race outcomes and put a driver
# with zero poles in three seasons ahead of one with six poles that year.
QUALI_CONTEXT_FEATURES = [
    "drv_experience",
    "team_avg_finish_5",
    "drv_circuit_avg_finish",
    "team_circuit_avg_finish",
    "circuit_pole_win_rate",
    "season_progress",
]

QUALI_FEATURES = QUALI_HISTORY_FEATURES + QUALI_CONTEXT_FEATURES + PRACTICE_FEATURES
RACE_FEATURES = BASE_FEATURES + GRID_FEATURES

# Rolling windows never see the current race, so early-career rows are sparse.
# XGBoost handles NaN natively, so we leave them rather than imputing a value
# the model would read as real.
MAX_FIELD = 24


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _load() -> pd.DataFrame:
    with connect(read_only=True) as con:
        results = con.execute(
            """
            SELECT r.season, r.round, r.driver_id, r.constructor_id, r.grid,
                   r.position, r.classified, r.points, r.dnf, r.laps,
                   ra.circuit_id, ra.race_date, ra.race_start_utc
            FROM raw_results r
            JOIN raw_races ra USING (season, round)
            """
        ).fetchdf()

        quali = con.execute(
            """
            SELECT season, round, driver_id,
                   position AS quali_position, best_ms
            FROM raw_qualifying
            """
        ).fetchdf()

        standings = con.execute(
            """
            SELECT season, round, driver_id,
                   position AS champ_position, points AS champ_points
            FROM raw_standings
            """
        ).fetchdf()

        pace = con.execute(
            """
            SELECT season, round, driver_id,
                   min(CASE WHEN session IN ('FP1','FP2','FP3') THEN long_run_ms END) AS fp_long_run_ms,
                   min(CASE WHEN session IN ('FP1','FP2','FP3') THEN best_lap_ms END)  AS fp_best_ms
            FROM raw_session_pace
            GROUP BY 1,2,3
            """
        ).fetchdf()

    return results, quali, standings, pace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _prior_rolling(df: pd.DataFrame, by: str | list[str], col: str, window: int, how: str = "mean"):
    """Rolling stat over the previous `window` races, current race excluded.

    ONLY valid when the grouping key has exactly one row per race - i.e. keys
    built on driver_id. .shift(1) steps back one ROW, so with two rows per race
    (a constructor has two cars) it steps back to the teammate in the SAME
    race and leaks that result. Team-level features must use
    _prior_rolling_by_race instead; tests/test_leakage.py enforces this.
    """
    grouped = df.groupby(by, sort=False)[col]
    return grouped.transform(lambda s: getattr(s.shift(1).rolling(window, min_periods=1), how)())


def _prior_expanding(df: pd.DataFrame, by: str | list[str], col: str, how: str = "mean"):
    """Expanding stat over all previous races. Same one-row-per-race caveat."""
    grouped = df.groupby(by, sort=False)[col]
    return grouped.transform(lambda s: getattr(s.shift(1).expanding(min_periods=1), how)())


def _roll_over_valid(s: pd.Series, window: int | None) -> pd.Series:
    """Mean of the last `window` NON-NULL values strictly before each row.

    A null neither contributes a value nor consumes a slot in the window - it
    is skipped, not counted as a bad result. `window=None` is expanding.

    Mechanics, because this is easy to get subtly wrong: roll over the non-null
    subsequence only, put each answer back at its own row, carry it forward
    across the skipped rows, then step back one row. The final step is what
    keeps the current race out of its own feature, exactly as .shift(1) does in
    the plain helpers.
    """
    valid = s.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=s.index)
    roll = (
        valid.expanding(min_periods=1).mean()
        if window is None
        else valid.rolling(window, min_periods=1).mean()
    )
    return roll.reindex(s.index).ffill().shift(1)


def _prior_pace(df: pd.DataFrame, by: str | list[str], col: str, window: int | None):
    """Recent form over races the entry actually ran to the end of.

    A retirement is recorded as last place, because that is where the
    classification puts it. Averaged into recent form it reads as "this driver
    was slow", which is usually false - the car broke, or someone hit them. The
    model already has a reliability channel of its own (drv_dnf_rate_10,
    team_dnf_rate_10, and the simulation's retirement hazard), so letting
    retirements into the form average charges for the same event twice: once
    honestly as unreliability, once again as fake slowness.

    The second charge is not evenly spread. It lands hardest on cars that were
    running near the front when they stopped, because they fall furthest, which
    means the feature systematically understates exactly the pace it is there
    to measure. Pace and reliability are separate questions and get separate
    features.

    One-row-per-race keys only; see _prior_rolling.
    """
    return df.groupby(by, sort=False)[col].transform(lambda s: _roll_over_valid(s, window))


def _prior_pace_by_race(df: pd.DataFrame, group: str, col: str, window: int | None) -> pd.Series:
    """_prior_pace for a group fielding several cars. See _prior_rolling_by_race
    for why the per-race collapse has to happen first."""
    per_race = (
        df.groupby([group, "race_seq"], sort=True)[col]  # mean() skips nulls
        .mean()
        .reset_index()
        .sort_values([group, "race_seq"])
    )
    per_race["_v"] = per_race.groupby(group, sort=False)[col].transform(lambda s: _roll_over_valid(s, window))
    merged = df[[group, "race_seq"]].merge(
        per_race[[group, "race_seq", "_v"]], on=[group, "race_seq"], how="left"
    )
    return pd.Series(merged["_v"].to_numpy(), index=df.index)


def _prior_pace_at_circuit(df: pd.DataFrame, group: str, col: str) -> pd.Series:
    """_prior_pace_by_race scoped to previous visits to this circuit."""
    per_race = (
        df.groupby([group, "circuit_id", "race_seq"], sort=True)[col]
        .mean()
        .reset_index()
        .sort_values([group, "circuit_id", "race_seq"])
    )
    per_race["_v"] = per_race.groupby([group, "circuit_id"], sort=False)[col].transform(
        lambda s: _roll_over_valid(s, None)
    )
    merged = df[[group, "circuit_id", "race_seq"]].merge(
        per_race[[group, "circuit_id", "race_seq", "_v"]],
        on=[group, "circuit_id", "race_seq"],
        how="left",
    )
    return pd.Series(merged["_v"].to_numpy(), index=df.index)


def _prior_circuit_by_race(df: pd.DataFrame, group: str, col: str, how: str = "mean") -> pd.Series:
    """Expanding stat at a circuit for a multi-car group, race-stepped.

    Same hazard as _prior_rolling_by_race, scoped to visits to one circuit.
    """
    per_race = (
        df.groupby([group, "circuit_id", "race_seq"], sort=True)[col]
        .mean()
        .reset_index()
        .sort_values([group, "circuit_id", "race_seq"])
    )
    per_race["_value"] = per_race.groupby([group, "circuit_id"], sort=False)[col].transform(
        lambda s: getattr(s.shift(1).expanding(min_periods=1), how)()
    )
    merged = df[[group, "circuit_id", "race_seq"]].merge(
        per_race[[group, "circuit_id", "race_seq", "_value"]],
        on=[group, "circuit_id", "race_seq"],
        how="left",
    )
    return pd.Series(merged["_value"].to_numpy(), index=df.index)


def _prior_rolling_by_race(
    df: pd.DataFrame,
    group: str,
    col: str,
    window: int | None,
    how: str = "mean",
    race_agg: str = "mean",
) -> pd.Series:
    """Rolling stat for a group that fields several cars, e.g. a constructor.

    Collapse to one value per (group, race) first, step back a whole race, then
    broadcast the answer to every driver in that group. Without the collapse,
    the second car in a garage reads its teammate's finishing position from the
    race being predicted - which is the future.

    Pass window=None for an expanding window.
    """
    per_race = (
        df.groupby([group, "race_seq"], sort=True)[col]
        .agg(race_agg)
        .reset_index()
        .sort_values([group, "race_seq"])
    )

    def roll(s: pd.Series) -> pd.Series:
        shifted = s.shift(1)
        window_obj = (
            shifted.expanding(min_periods=1) if window is None else shifted.rolling(window, min_periods=1)
        )
        return getattr(window_obj, how)()

    per_race["_value"] = per_race.groupby(group, sort=False)[col].transform(roll)
    merged = df[[group, "race_seq"]].merge(
        per_race[[group, "race_seq", "_value"]], on=[group, "race_seq"], how="left"
    )
    return pd.Series(merged["_value"].to_numpy(), index=df.index)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def _upcoming_entries(results: pd.DataFrame) -> pd.DataFrame:
    """Placeholder rows for races that have not run yet.

    The feature frame is driven by results, so a future race has no rows and
    therefore nothing to predict. We synthesise an entry list from the most
    recent completed round of that season - which is what a person would do -
    and leave position/points null so the rows are never used for training.
    """
    with connect(read_only=True) as con:
        scheduled = con.execute(
            """
            SELECT ra.season, ra.round, ra.circuit_id, ra.race_date, ra.race_start_utc
            FROM raw_races ra
            LEFT JOIN raw_results r USING (season, round)
            WHERE r.driver_id IS NULL AND ra.race_date >= current_date - 1
            GROUP BY ALL
            ORDER BY ra.season, ra.round
            """
        ).fetchdf()

    if scheduled.empty or results.empty:
        return pd.DataFrame()

    rows = []
    for _, race in scheduled.iterrows():
        season = int(race["season"])
        in_season = results[results["season"] == season]
        if in_season.empty:
            continue
        latest = in_season[in_season["round"] == in_season["round"].max()]
        entry = latest[["driver_id", "constructor_id"]].drop_duplicates()
        for _, e in entry.iterrows():
            rows.append(
                {
                    "season": season,
                    "round": int(race["round"]),
                    "driver_id": e["driver_id"],
                    "constructor_id": e["constructor_id"],
                    "circuit_id": race["circuit_id"],
                    "race_date": race["race_date"],
                    "race_start_utc": race["race_start_utc"],
                    "grid": np.nan,
                    "position": np.nan,
                    "classified": None,
                    "points": np.nan,
                    "dnf": None,
                    "laps": np.nan,
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        log.info(
            "Added %d placeholder entries for %d upcoming race(s)",
            len(out),
            out.groupby(["season", "round"]).ngroups,
        )
    return out


def build(include_upcoming: bool = True) -> pd.DataFrame:
    results, quali, standings, pace = _load()
    if results.empty:
        raise RuntimeError("raw_results is empty - run the ingest first")

    if include_upcoming:
        upcoming = _upcoming_entries(results)
        if not upcoming.empty:
            results = pd.concat([results, upcoming], ignore_index=True)

    df = results.merge(quali, on=["season", "round", "driver_id"], how="left")
    df = df.merge(pace, on=["season", "round", "driver_id"], how="left")

    # Chronological ordering drives every rolling window below.
    df = df.sort_values(["season", "round", "position"], na_position="last").reset_index(drop=True)
    df["race_seq"] = df.groupby(["season", "round"], sort=False).ngroup()

    # Championship standing going INTO the race = standing after the previous
    # round. Joining on the current round would leak the result we predict.
    prev = standings.copy()
    prev["round"] = prev["round"] + 1
    df = df.merge(
        prev.rename(
            columns={"champ_position": "champ_position_before", "champ_points": "champ_points_before"}
        ),
        on=["season", "round", "driver_id"],
        how="left",
    )
    # Round 1 has no prior standing: everyone starts level.
    first_round = df["round"] == 1
    df.loc[first_round, "champ_points_before"] = df.loc[first_round, "champ_points_before"].fillna(0.0)

    # ---- driver form -----------------------------------------------------
    df["finish_or_last"] = df["position"].fillna(MAX_FIELD)
    df["scored"] = (df["points"] > 0).astype(float)
    df["dnf_flag"] = df["dnf"].astype(float)

    # Position on the days the car made it to the flag. Null on a retirement,
    # which is what keeps a broken gearbox out of the pace average - see
    # _prior_pace. finish_or_last stays as it is: outcome features (points,
    # podium rate, the training label) should absolutely count a retirement,
    # because not finishing is a real and costly outcome.
    df["finish_when_running"] = df["finish_or_last"].where(~df["dnf"].astype(bool))

    df["drv_avg_finish_3"] = _prior_pace(df, "driver_id", "finish_when_running", 3)
    df["drv_avg_finish_5"] = _prior_pace(df, "driver_id", "finish_when_running", 5)
    df["drv_avg_grid_5"] = _prior_rolling(df, "driver_id", "grid", 5)
    df["drv_points_rate_5"] = _prior_rolling(df, "driver_id", "points", 5)
    df["drv_dnf_rate_10"] = _prior_rolling(df, "driver_id", "dnf_flag", 10)
    df["drv_experience"] = df.groupby("driver_id", sort=False).cumcount()

    # ---- team form -------------------------------------------------------
    # Grouped by (constructor, driver) then averaged would double-count the
    # better car; grouping by constructor alone mixes both drivers, which is
    # what we want for a car-performance signal.
    # Both cars collapse to one number per race BEFORE the window steps back,
    # otherwise the second car reads its teammate's result from this race.
    df["team_avg_finish_3"] = _prior_pace_by_race(df, "constructor_id", "finish_when_running", 3)
    df["team_avg_finish_5"] = _prior_pace_by_race(df, "constructor_id", "finish_when_running", 5)
    df["team_points_rate_5"] = _prior_rolling_by_race(df, "constructor_id", "points", 5, race_agg="sum")
    df["team_dnf_rate_10"] = _prior_rolling_by_race(df, "constructor_id", "dnf_flag", 10)

    # ---- circuit history -------------------------------------------------
    df["drv_circuit_avg_finish"] = _prior_pace(df, ["driver_id", "circuit_id"], "finish_when_running", None)
    df["team_circuit_avg_finish"] = _prior_pace_at_circuit(df, "constructor_id", "finish_when_running")

    df = _add_circuit_profile(df)

    # ---- qualifying-derived ---------------------------------------------
    df = _add_quali_features(df)

    # ---- car pace and racecraft -----------------------------------------
    df = _add_pace_features(df)

    # ---- practice --------------------------------------------------------
    df = _add_practice_features(df)

    # ---- season context --------------------------------------------------
    rounds_per_season = df.groupby("season")["round"].transform("max")
    df["season_progress"] = df["round"] / rounds_per_season

    # ---- labels ----------------------------------------------------------
    # XGBRanker wants higher = better, so invert position into a relevance
    # score. Retirements keep their classification order rather than all
    # collapsing to the same value, which preserves "retired on lap 50" being
    # a better outcome than "retired on lap 2".
    # Null where the race has not happened, so training silently skips it
    # rather than learning that every future entry finishes last.
    df["race_relevance"] = np.where(
        df["position"].notna(), (MAX_FIELD - df["finish_or_last"]).clip(lower=0), np.nan
    )
    df["quali_relevance"] = np.where(
        df["quali_position"].notna(),
        (MAX_FIELD - df["quali_position"].fillna(MAX_FIELD)).clip(lower=0),
        np.nan,
    )
    # Grid becomes known only after qualifying; before that it is genuinely
    # unknown and must stay null rather than being filled with a guess.
    df["grid"] = df["grid"].replace(0, np.nan)  # 0 = pit lane start in Ergast

    df = df.sort_values(["season", "round", "driver_id"]).reset_index(drop=True)
    log.info("Built %d rows across %d races", len(df), df["race_seq"].nunique())
    return df


def _add_circuit_profile(df: pd.DataFrame) -> pd.DataFrame:
    """How much a circuit reshuffles the grid, and how punishing it is.

    Computed from prior visits only. A circuit nobody has raced at gets NaN,
    which is the honest answer - see cold_start() for how a new track is
    handled at prediction time.
    """
    df["abs_pos_change"] = (df["grid"] - df["finish_or_last"]).abs()

    per_race = (
        df.groupby(["circuit_id", "race_seq"], sort=True)
        .agg(pos_change=("abs_pos_change", "mean"), dnf=("dnf_flag", "mean"))
        .reset_index()
        .sort_values(["circuit_id", "race_seq"])
    )
    per_race["circuit_overtaking_score"] = per_race.groupby("circuit_id")["pos_change"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    per_race["circuit_dnf_rate"] = per_race.groupby("circuit_id")["dnf"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )

    return df.merge(
        per_race[["circuit_id", "race_seq", "circuit_overtaking_score", "circuit_dnf_rate"]],
        on=["circuit_id", "race_seq"],
        how="left",
    )


def _add_quali_features(df: pd.DataFrame) -> pd.DataFrame:
    """Gaps expressed as percentages, not milliseconds.

    A 0.3s gap means something different at Monaco (1:11) and Spa (1:42).
    Percentage of pole time is comparable across circuits and eras, which
    matters when the training set spans nine seasons of regulation changes.
    """
    pole = df.groupby(["season", "round"])["best_ms"].transform("min")
    df["quali_gap_to_pole_pct"] = (df["best_ms"] / pole - 1.0) * 100

    team_best = df.groupby(["season", "round", "constructor_id"])["best_ms"].transform("min")
    df["quali_gap_to_teammate_pct"] = (df["best_ms"] / team_best - 1.0) * 100

    # Qualifying history, measured from QUALIFYING POSITION rather than grid.
    # Grid carries penalties: a driver who takes pole and starts tenth reads as
    # a poor qualifier on any grid-based measure. To predict who is fast over
    # one lap, the only honest record is where they actually qualified.
    df["pole_flag"] = (df["quali_position"] == 1).astype(float)
    df.loc[df["quali_position"].isna(), "pole_flag"] = np.nan
    df["quali_top3_flag"] = (df["quali_position"] <= 3).astype(float)
    df.loc[df["quali_position"].isna(), "quali_top3_flag"] = np.nan

    df["drv_avg_quali_3"] = _prior_rolling(df, "driver_id", "quali_position", 3)
    df["drv_avg_quali_5"] = _prior_rolling(df, "driver_id", "quali_position", 5)
    df["drv_pole_rate_10"] = _prior_rolling(df, "driver_id", "pole_flag", 10)
    df["drv_quali_top3_rate_10"] = _prior_rolling(df, "driver_id", "quali_top3_flag", 10)
    df["team_avg_quali_5"] = _prior_rolling_by_race(df, "constructor_id", "quali_position", 5, race_agg="min")

    # Rolling head-to-head against the other side of the garage. Strong signal:
    # it isolates the driver from the car, which almost nothing else here does.
    df = df.sort_values(["season", "round"]).reset_index(drop=True)
    df["drv_teammate_quali_edge"] = _prior_rolling(df, "driver_id", "quali_gap_to_teammate_pct", 5)
    return df


def _add_pace_features(df: pd.DataFrame) -> pd.DataFrame:
    """Car strength and racecraft, both knowable before the weekend starts.

    These use this race's own qualifying gap only through .shift(1) inside
    _prior_rolling, so a prediction made on Thursday can use them.
    """
    df = df.sort_values(["season", "round"]).reset_index(drop=True)

    # Best of the two cars each weekend is a better read on the machine than
    # either driver alone - it strips out one driver having a scruffy lap.
    df["team_round_pace"] = df.groupby(["season", "round", "constructor_id"])[
        "quali_gap_to_pole_pct"
    ].transform("min")

    df["team_pace_gap_pct"] = _prior_rolling_by_race(
        df, "constructor_id", "team_round_pace", 5, race_agg="min"
    )
    df["drv_pace_gap_pct"] = _prior_rolling(df, "driver_id", "quali_gap_to_pole_pct", 5)

    # Is the car coming to us or going away? Negative means improving.
    recent = _prior_rolling_by_race(df, "constructor_id", "team_round_pace", 3, race_agg="min")
    longer = _prior_rolling_by_race(df, "constructor_id", "team_round_pace", 8, race_agg="min")
    df["team_pace_trend"] = recent - longer

    # Racecraft: places gained from the grid, independent of how fast the car
    # qualified. This is the part of a result that belongs to the driver.
    # Null on a retirement for the same reason the form averages are: a car
    # that stops from third is recorded as losing nineteen places, which says
    # nothing about the driver's racecraft.
    df["positions_gained"] = (df["grid"] - df["finish_when_running"]).where(~df["dnf"].astype(bool))
    df["drv_positions_gained_5"] = _prior_pace(df, "driver_id", "positions_gained", 5)

    df["podium_flag"] = (df["position"] <= 3).astype(float)
    df["top10_flag"] = (df["position"] <= 10).astype(float)
    df["drv_podium_rate_10"] = _prior_rolling(df, "driver_id", "podium_flag", 10)
    df["drv_top10_rate_10"] = _prior_rolling(df, "driver_id", "top10_flag", 10)

    df["drv_circuit_starts"] = df.groupby(["driver_id", "circuit_id"], sort=False).cumcount()

    # How reliably pole converts to a win here. Monaco and Monza are opposite
    # ends of this, and it is exactly what "does the grid matter" means.
    pole_won = (
        df.assign(pole_win=((df["grid"] == 1) & (df["position"] == 1)).astype(float))
        .groupby(["circuit_id", "race_seq"], sort=True)["pole_win"]
        .max()
        .reset_index()
        .sort_values(["circuit_id", "race_seq"])
    )
    pole_won["circuit_pole_win_rate"] = pole_won.groupby("circuit_id")["pole_win"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    return df.merge(
        pole_won[["circuit_id", "race_seq", "circuit_pole_win_rate"]],
        on=["circuit_id", "race_seq"],
        how="left",
    )


def _add_practice_features(df: pd.DataFrame) -> pd.DataFrame:
    if "fp_long_run_ms" not in df or df["fp_long_run_ms"].notna().sum() == 0:
        # No FastF1 data ingested. Emit the columns as null so the model
        # schema is identical either way, and flag their absence.
        df["fp_long_run_gap_pct"] = np.nan
        df["fp_best_gap_pct"] = np.nan
        df["practice_available"] = 0.0
        return df

    fastest_long = df.groupby(["season", "round"])["fp_long_run_ms"].transform("min")
    fastest_best = df.groupby(["season", "round"])["fp_best_ms"].transform("min")
    df["fp_long_run_gap_pct"] = (df["fp_long_run_ms"] / fastest_long - 1.0) * 100
    df["fp_best_gap_pct"] = (df["fp_best_ms"] / fastest_best - 1.0) * 100
    df["practice_available"] = df["fp_best_ms"].notna().astype(float)
    return df


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save(df: pd.DataFrame) -> None:
    with connect() as con:
        con.execute("DROP TABLE IF EXISTS features")
        con.register("_f", df)
        con.execute("CREATE TABLE features AS SELECT * FROM _f")
        con.unregister("_f")
    log.info("Wrote %d feature rows", len(df))


def load() -> pd.DataFrame:
    with connect(read_only=True) as con:
        return con.execute("SELECT * FROM features ORDER BY season, round, driver_id").fetchdf()
