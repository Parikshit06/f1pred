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
    race model  : BASE + GRID              -> predicts the finish

Two columns that are easy to conflate are kept apart: `quali_position` is the
qualifying classification and `grid` is where the car actually started, after
penalties. The grid comes from weekend.resolve_grid, which records its source.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import weekend
from .store import connect

log = logging.getLogger(__name__)

BASE_FEATURES = [
    "drv_avg_finish_3",
    "drv_avg_finish_5",
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
    "season_progress",
    # Car pace: rolling qualifying gap to pole, the cleanest public signal of
    # how fast the car is. Without these the model inferred pace from results.
    "team_pace_gap_pct",
    "team_pace_trend",
    # Racecraft, separated from pace: who actually gains places on Sunday.
    "drv_positions_gained_5",
    "drv_podium_rate_10",
    "drv_top10_rate_10",
    "drv_circuit_starts",
    "circuit_pole_win_rate",
    "team_avg_quali_5",
]

# Tried and cut, on the 2024-26 walk-forward (62 races), shown without -> with.
# Numbers are from when each was tried. None cleared the noise, so none stay.
#
#   teammate race edge, beat-teammate rate     podium 2.016 -> 1.968
#   races at current team, first season there  winner 0.500 -> 0.484; switcher bias wider
#   form shrunk toward the car by experience   neutral
#   form at circuits of the same character     worse on both models, pole 0.258 -> 0.226
#   quali score blended with the team's best   fitted at 0.45 on 2022-23, then off-window
#     recent grid slot                           pole 0.274 -> 0.226
#   retirements split by cause, correlated     identical to 4 d.p. - the split keeps each
#     inside a garage                            car's marginal DNF rate; band coverage
#                                                50.0% either way
#   team pit-stop gap, 5-race                  better on two metrics, worse on three
#                                                including log loss
#
# And measured with race-paired bootstrap intervals (reports/experiments.json):
#   teammate qualifying gap in the race model  win log loss +0.007 on 2022-23,
#                                                interval -0.047 to +0.060: nothing
#   teammate gap in the same session, signed   no difference from the best-lap one
#   median instead of mean recent form         2022-23 gain's interval spans zero

PRACTICE_FEATURES = [
    "fp_long_run_gap_pct",
    "fp_best_gap_pct",
    "practice_available",
]

GRID_FEATURES = [
    "grid",
    "quali_position",
    "quali_gap_to_pole_pct",
]

# Measured and left out of the race model: the teammate qualifying gap, this
# weekend's and its five-race average. Removing them changed win log loss by
# -0.005 on 2022-23 and +0.001 on 2024-, both intervals straddling zero
# (reports/experiments.json, ablation). drv_teammate_quali_edge still feeds the
# qualifying model, where head-to-head record is direct one-lap evidence.
TEAMMATE_FEATURES = ["quali_gap_to_teammate_pct", "drv_teammate_quali_edge"]

# Also measured and left out of the race model: the driver's own qualifying
# record. It is the qualifying model's core input, and reaches the race through
# the grid - projected before qualifying, official after. Inside the race model
# it counted twice: removing it improved pre-qualifying win log loss on 2022-23
# by 0.060 (interval 0.020 to 0.104) and cost nothing after qualifying.
QUALI_FORM_FEATURES = ["drv_avg_grid_5", "drv_avg_quali_5", "drv_pole_rate_10", "drv_pace_gap_pct"]

# Practice feeds the qualifying model only and reaches the race through the
# predicted grid. Measured on the weekends with practice data
# (reports/experiments.json, practice): it sharpens the qualifying forecast and
# adds nothing to the race once the real grid is known.

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

# Race-outcome features the qualifying model may use, kept deliberately light.
# With points rate and championship position in, 59% of its signal came from
# race results and it ranked a driver with no poles above one with six.
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

# Bumped whenever a feature's definition changes, and written into every
# forecast, so a logged prediction says which definitions produced it.
FEATURE_VERSION = "2026.09.5"

# How recent finishing form (drv_/team_avg_finish_*) is summarised: "mean" or
# "median". The median resists one freak result, but it did not earn its place:
# on the tuning seasons its gain had an interval including zero, and on 2024-
# it was worse (reports/experiments.json, form_statistic).
FORM_STAT = "mean"

# Rolling windows never see the current race, so early-career rows are sparse.
# XGBoost handles NaN natively, so we leave them rather than imputing a value
# the model would read as real.
MAX_FIELD = 24
TEAMMATE_GAP_CLIP = 3.0  # % of a lap


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
@dataclass
class Raw:
    """The raw tables features are built from."""

    results: pd.DataFrame
    quali: pd.DataFrame
    standings: pd.DataFrame
    pace: pd.DataFrame
    races: pd.DataFrame
    openf1_grid: pd.DataFrame
    openf1_entries: pd.DataFrame


def _load() -> Raw:
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
            SELECT season, round, driver_id, constructor_id,
                   position AS quali_position, q1_ms, q2_ms, q3_ms, best_ms
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

        # Practice sessions only: all of them run before qualifying.
        pace = con.execute(
            """
            SELECT season, round, driver_id,
                   min(long_run_ms) AS fp_long_run_ms,
                   min(best_lap_ms) AS fp_best_ms
            FROM raw_session_pace
            WHERE session IN ('FP1', 'FP2', 'FP3')
            GROUP BY 1, 2, 3
            """
        ).fetchdf()

        races = con.execute(
            "SELECT season, round, circuit_id, race_date, race_start_utc FROM raw_races"
        ).fetchdf()
        openf1_grid = con.execute("SELECT season, round, driver_id, position FROM raw_openf1_grid").fetchdf()
        openf1_entries = con.execute(
            "SELECT season, round, session, driver_id, constructor_id FROM raw_openf1_entries"
        ).fetchdf()

    return Raw(results, quali, standings, pace, races, openf1_grid, openf1_entries)


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


def _roll_over_valid(s: pd.Series, window: int | None, stat: str = "mean") -> pd.Series:
    """Mean (or median) of the last `window` non-null values strictly before each row.

    Nulls are skipped rather than counted. Rolls over the non-null values only,
    puts each result back at its row, carries it forward across the gaps, then
    shifts one row so the current race never feeds its own feature.
    `window=None` is expanding.
    """
    valid = s.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=s.index)
    roller = valid.expanding(min_periods=1) if window is None else valid.rolling(window, min_periods=1)
    roll = getattr(roller, stat)()
    return roll.reindex(s.index).ffill().shift(1)


def _prior_pace(df: pd.DataFrame, by: str | list[str], col: str, window: int | None, stat: str = "mean"):
    """Recent finishing form over races the car actually finished.

    A retirement is classified near last; averaged in, it reads as slowness when
    the car broke. Reliability already has its own features and the simulator's
    retirement hazard, so counting it here too would charge the same event twice
    - hardest on cars that stopped while running near the front.

    One-row-per-race keys only; see _prior_rolling.
    """
    return df.groupby(by, sort=False)[col].transform(lambda s: _roll_over_valid(s, window, stat))


def _prior_pace_by_race(
    df: pd.DataFrame, group: str, col: str, window: int | None, stat: str = "mean"
) -> pd.Series:
    """_prior_pace for a group fielding several cars. See _prior_rolling_by_race
    for why the per-race collapse has to happen first."""
    per_race = (
        df.groupby([group, "race_seq"], sort=True)[col]  # mean() skips nulls
        .mean()
        .reset_index()
        .sort_values([group, "race_seq"])
    )
    per_race["_v"] = per_race.groupby(group, sort=False)[col].transform(
        lambda s: _roll_over_valid(s, window, stat)
    )
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
def _upcoming_entries(raw: Raw) -> pd.DataFrame:
    """Placeholder rows for races that have not run yet.

    The feature frame is driven by results, so a future race has no rows and
    nothing to predict. Its entry list comes from weekend.race_entries: the
    weekend's own qualifying or session lists when they exist, the previous
    race's field only as a labelled last resort. Position and points stay null
    so the rows are never used for training.
    """
    results, races = raw.results, raw.races
    if races.empty or results.empty:
        return pd.DataFrame()
    run = set(zip(results["season"], results["round"]))
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    dates = pd.to_datetime(races["race_date"])
    not_run = np.array([(s, r) not in run for s, r in zip(races["season"], races["round"])], dtype=bool)
    scheduled = races[not_run & (dates >= today - pd.Timedelta(days=1)).to_numpy()].sort_values(
        ["season", "round"]
    )

    frames = []
    for race in scheduled.itertuples():
        season, rnd = int(race.season), int(race.round)
        entry, source = weekend.race_entries(season, rnd, results, raw.quali, raw.openf1_entries)
        if entry.empty:
            continue
        frames.append(
            entry.assign(
                season=season,
                round=rnd,
                circuit_id=race.circuit_id,
                race_date=race.race_date,
                race_start_utc=race.race_start_utc,
                entry_source=source,
            )
        )
    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    for col, value in (("grid", np.nan), ("position", np.nan), ("points", np.nan), ("laps", np.nan)):
        out[col] = value
    out["classified"] = pd.Series([None] * len(out), dtype=object)
    out["dnf"] = pd.Series([None] * len(out), dtype=object)
    log.info(
        "Added %d placeholder entries for %d upcoming race(s); next race's field from %s",
        len(out),
        out.groupby(["season", "round"]).ngroups,
        out["entry_source"].iloc[0],
    )
    return out


def _check_integrity(df: pd.DataFrame) -> None:
    """Fail loudly on the two mistakes that silently corrupt every rolling feature."""
    dupes = df.duplicated(["season", "round", "driver_id"])
    if dupes.any():
        raise ValueError(f"{int(dupes.sum())} duplicate (season, round, driver) rows")
    order = df.drop_duplicates("race_seq").sort_values("race_seq")
    if not (order[["season", "round"]].apply(tuple, axis=1).is_monotonic_increasing):
        raise ValueError("race_seq is not in (season, round) order")


def build(include_upcoming: bool = True, form_stat: str | None = None) -> pd.DataFrame:
    raw = _load()
    stat = form_stat or FORM_STAT
    results = raw.results.copy()
    if results.empty:
        raise RuntimeError("raw_results is empty - run the ingest first")
    results["entry_source"] = "results"

    if include_upcoming:
        upcoming = _upcoming_entries(raw)
        if not upcoming.empty:
            results = pd.concat([results, upcoming[results.columns]], ignore_index=True)

    df = results.merge(
        raw.quali.drop(columns=["constructor_id"]), on=["season", "round", "driver_id"], how="left"
    )
    df = df.merge(raw.pace, on=["season", "round", "driver_id"], how="left")

    # Where each car started: the results grid, else the official grid, else
    # the qualifying order - see weekend.resolve_grid.
    resolved = weekend.resolve_grid(df[["season", "round", "driver_id", "grid"]], raw.quali, raw.openf1_grid)
    df["grid"] = resolved["grid"].to_numpy()
    df["grid_source"] = resolved["grid_source"].to_numpy()

    # Chronological ordering drives every rolling window below.
    df = df.sort_values(["season", "round", "position"], na_position="last").reset_index(drop=True)
    df["race_seq"] = df.groupby(["season", "round"], sort=False).ngroup()
    _check_integrity(df)

    # Championship standing going INTO the race = standing after the previous
    # round. Joining on the current round would leak the result we predict.
    prev = raw.standings.copy()
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
    ran = df["position"].notna()
    df["finish_or_last"] = df["position"].fillna(MAX_FIELD)
    # Outcome flags are null for a race not yet run, not zero: a placeholder
    # row must never read as a race without points.
    df["scored"] = (df["points"] > 0).astype(float).where(ran)
    df["dnf_flag"] = df["dnf"].astype(float)

    # Finishing position when the car finished; null on a retirement, so a
    # failure doesn't read as slowness. Outcome features and the label still
    # use finish_or_last, because not finishing is a real result.
    retired = df["dnf"].astype("boolean").fillna(True).astype(bool)  # unknown counts as not running
    df["finish_when_running"] = df["finish_or_last"].where(ran & ~retired)

    df["drv_avg_finish_3"] = _prior_pace(df, "driver_id", "finish_when_running", 3, stat)
    df["drv_avg_finish_5"] = _prior_pace(df, "driver_id", "finish_when_running", 5, stat)
    df["drv_avg_grid_5"] = _prior_rolling(df, "driver_id", "grid", 5)
    df["drv_points_rate_5"] = _prior_rolling(df, "driver_id", "points", 5)
    df["drv_dnf_rate_10"] = _prior_rolling(df, "driver_id", "dnf_flag", 10)
    df["drv_experience"] = df.groupby("driver_id", sort=False).cumcount()

    # ---- team form -------------------------------------------------------
    # One number per car per race, collapsed before the window shifts -
    # otherwise one driver's row sees the teammate's result from the same race.
    df["team_avg_finish_3"] = _prior_pace_by_race(df, "constructor_id", "finish_when_running", 3, stat)
    df["team_avg_finish_5"] = _prior_pace_by_race(df, "constructor_id", "finish_when_running", 5, stat)
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
    # Calendar length is known before a season starts. Counting the rounds with
    # results instead - as this once did - gave the same race a different value
    # depending on how much of the season had been run when features were built.
    calendar = raw.races.groupby("season")["round"].max()
    df["season_progress"] = df["round"] / df["season"].map(calendar).fillna(df["round"])

    # ---- labels ----------------------------------------------------------
    # XGBRanker wants higher = better. Retirements keep their classified order,
    # and unrun races stay null so training skips them.
    df["race_relevance"] = np.where(ran, (MAX_FIELD - df["finish_or_last"]).clip(lower=0), np.nan)
    df["quali_relevance"] = np.where(
        df["quali_position"].notna(),
        (MAX_FIELD - df["quali_position"].fillna(MAX_FIELD)).clip(lower=0),
        np.nan,
    )

    df = df.sort_values(["season", "round", "driver_id"]).reset_index(drop=True)
    log.info("Built %d rows across %d races", len(df), df["race_seq"].nunique())
    return df


def _add_circuit_profile(df: pd.DataFrame) -> pd.DataFrame:
    """How much a circuit reshuffles the grid, and how punishing it is.

    Computed from prior visits only. A circuit nobody has raced at gets NaN,
    which is the honest answer - see cold_start() for how a new track is
    handled at prediction time.
    """
    df["abs_pos_change"] = (df["grid"] - df["finish_or_last"]).abs().where(df["position"].notna())

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

    df["quali_gap_to_teammate_pct"] = teammate_gap(df)

    # Measured from qualifying position, not grid: grid carries penalties, and a
    # driver who takes pole and starts tenth isn't a poor qualifier.
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
    # Over the last five weekends with a comparison; one without isn't a tie.
    # Clipped at 3%: beyond that the gap is a crash or a failure in Q1, not the
    # drivers, and one such lap would otherwise dominate a five-race average.
    df = df.sort_values(["season", "round"]).reset_index(drop=True)
    df["_edge"] = df["quali_gap_to_teammate_pct"].clip(-TEAMMATE_GAP_CLIP, TEAMMATE_GAP_CLIP)
    df["drv_teammate_quali_edge"] = _prior_pace(df, "driver_id", "_edge", 5)
    return df.drop(columns="_edge")


def teammate_gap(df: pd.DataFrame) -> pd.Series:
    """Qualifying gap to the teammate, as a % of a lap: each driver's best lap of
    the day against the team's best, so the faster driver is at 0.

    Null, never 0, when there is no teammate time to compare with: a missing
    teammate is not a tie. (Filling it with 0 read as "beat the teammate" for 51
    drivers who had nobody to beat.)
    """
    keys = ["season", "round", "constructor_id"]
    team_best = df.groupby(keys)["best_ms"].transform("min")
    timed = df.groupby(keys)["best_ms"].transform("count")
    return ((df["best_ms"] / team_best - 1.0) * 100).where(timed >= 2)


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

    # Places gained from the grid. Null on a retirement - stopping from third
    # isn't a racecraft result.
    df["positions_gained"] = (df["grid"] - df["finish_when_running"]).where(~df["dnf"].astype(bool))
    df["drv_positions_gained_5"] = _prior_pace(df, "driver_id", "positions_gained", 5)

    ran = df["position"].notna()
    df["podium_flag"] = (df["position"] <= 3).astype(float).where(ran)
    df["top10_flag"] = (df["position"] <= 10).astype(float).where(ran)
    df["drv_podium_rate_10"] = _prior_rolling(df, "driver_id", "podium_flag", 10)
    df["drv_top10_rate_10"] = _prior_rolling(df, "driver_id", "top10_flag", 10)

    df["drv_circuit_starts"] = df.groupby(["driver_id", "circuit_id"], sort=False).cumcount()

    # How reliably pole converts to a win here. Monaco and Monza are opposite
    # ends of this, and it is exactly what "does the grid matter" means.
    pole_won = (
        df.assign(
            pole_win=((df["grid"] == 1) & (df["position"] == 1)).astype(float).where(df["position"].notna())
        )
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
