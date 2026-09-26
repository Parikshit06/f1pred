"""Forecast one race and log it.

Before qualifying the grid is unknown, so each simulated race samples its own
grid from the qualifying model. After qualifying the real grid is used. Both
calls are logged to predictions/ with the time they were made and are never
overwritten - that folder is the track record.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from . import backtest, championship, config, features, model, probability, provenance, simulate, weekend
from .store import connect

log = logging.getLogger(__name__)


@dataclass
class DriverLine:
    driver_id: str
    name: str
    short: str  # surname, for column widths that cannot hold the full name
    team: str
    p_win: float
    p_podium: float
    p_top5: float
    p_top10: float
    exp_position: float
    grid: int | None
    why: str = ""


@dataclass
class Prediction:
    season: int
    round: int
    race_name: str
    circuit_id: str
    race_start_utc: str | None
    generated_at_utc: str
    grid_known: bool
    quali_board: list[dict] = field(default_factory=list)
    race_board: list[dict] = field(default_factory=list)
    field_probs: list[dict] = field(default_factory=list)
    position_matrix: list[list[float]] = field(default_factory=list)
    season_outlook: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    quali_start_utc: str | None = None

    def path(self) -> Path:
        stamp = self.generated_at_utc.replace(":", "").replace("-", "")[:15]
        tag = "postquali" if self.grid_known else "prequali"
        return config.PREDICTIONS / f"{self.season}-{self.round:02d}-{tag}-{stamp}.json"

    def existing(self) -> Path | None:
        """An earlier forecast for this race at this stage, if there is one."""
        tag = "postquali" if self.grid_known else "prequali"
        matches = sorted(config.PREDICTIONS.glob(f"{self.season}-{self.round:02d}-{tag}-*.json"))
        return matches[0] if matches else None

    def days_out(self) -> float | None:
        """How far ahead of the race this forecast is being made."""
        return _days_until(self.race_start_utc)

    def waiting_for_practice(self) -> bool:
        """A pre-qualifying call made before practice, with qualifying still
        more than PRACTICE_WAIT_HOURS away: worth holding for the pace data."""
        if self.grid_known or self.meta.get("practice_data"):
            return False
        to_quali = _days_until(self.quali_start_utc)
        return to_quali is not None and to_quali * 24 > config.PRACTICE_WAIT_HOURS

    def save(self, force: bool = False) -> Path | None:
        """Write the forecast unless this race already has one at this stage.

        One file per race per stage keeps the record honest - a second pre-quali
        file from a later run would look like the forecast was retried until it
        looked good. It also lets the workflow run often without littering.
        """
        prior = None if force else self.existing()
        if prior is not None:
            log.info(
                "forecast already logged for %d r%d at this stage: %s",
                self.season,
                self.round,
                prior.name,
            )
            return prior

        # Too far out to be a race-week call. The page still renders; the file isn't
        # written, so it can't block the later forecast for this stage.
        ahead = self.days_out()
        if not force and not self.grid_known and ahead is not None and ahead > config.LOG_WINDOW_DAYS:
            log.info(
                "not logging %d r%d yet: %.1f days out, window is %.0f",
                self.season,
                self.round,
                ahead,
                config.LOG_WINDOW_DAYS,
            )
            return None

        if not force and self.waiting_for_practice():
            log.info("not logging %d r%d yet: waiting for practice pace", self.season, self.round)
            return None

        p = self.path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2, default=str))
        return p


def _days_until(stamp: str | None) -> float | None:
    if not stamp:
        return None
    when = pd.to_datetime(stamp, errors="coerce", utc=True)
    if pd.isna(when):
        return None
    return float((when - pd.Timestamp(datetime.now(UTC))).total_seconds()) / 86400


# ---------------------------------------------------------------------------
def _driver_names() -> dict[str, tuple[str, str]]:
    with connect(read_only=True) as con:
        df = con.execute(
            """
            SELECT driver_id,
                   max(given_name || ' ' || family_name) AS full_name,
                   max(family_name) AS surname
            FROM raw_drivers GROUP BY driver_id
            """
        ).fetchdf()
    return dict(zip(df["driver_id"], zip(df["full_name"], df["surname"])))


def next_race(df: pd.DataFrame) -> tuple[int, int]:
    """The earliest race that hasn't started yet.

    Checked against the start time, not the absence of results: jolpica can lag
    a race by hours, and in that window a finished race would otherwise get a
    "pre-race" forecast written after the flag.
    """
    pending = df[df["position"].isna()]
    if "race_start_utc" in pending.columns:
        # utc=True on both sides: naive vs aware comparison raises, and whether the
        # column carries a timezone depends on the ingest.
        now = pd.Timestamp(datetime.now(UTC))
        starts = pd.to_datetime(pending["race_start_utc"], errors="coerce", utc=True)
        # A race with no published start time is kept: better to forecast one
        # we cannot date than to silently skip a round.
        pending = pending[starts.isna() | (starts > now)]
    if pending.empty:
        raise RuntimeError(
            "No upcoming race found - every scheduled race has either run or started. "
            "Is the schedule ingested for this season?"
        )
    row = pending.sort_values(["season", "round"]).iloc[0]
    return int(row["season"]), int(row["round"])


def grid_is_known(race: pd.DataFrame) -> bool:
    """Post-qualifying once nearly every car has a slot from qualifying or the
    official grid (weekend.resolve_grid). The projected order is never enough."""
    if "grid_source" in race:
        return bool(race["grid_source"].notna().sum() >= len(race) * weekend.MIN_GRID_COVERAGE)
    return bool(race["quali_position"].notna().sum() >= len(race) * weekend.MIN_GRID_COVERAGE)


def _dominant(series: pd.Series) -> str | None:
    values = series.dropna()
    return str(values.mode().iloc[0]) if not values.empty else None


def feature_labels() -> dict[str, str]:
    """Plain-English phrase for every feature, used in the published rationale.

    Kept as its own function so a test can assert no feature is missing - an
    unlabelled one would appear on the page as a raw column name.
    """
    return {
        "grid": "starting position",
        "quali_position": "qualifying position",
        "quali_gap_to_pole_pct": "pace gap to pole",
        "quali_gap_to_teammate_pct": "gap to teammate in qualifying",
        "drv_avg_finish_3": "recent finishing form",
        "drv_avg_finish_5": "typical finish over five races",
        "drv_avg_grid_5": "recent qualifying form",
        "drv_points_rate_5": "recent points scoring",
        "drv_dnf_rate_10": "reliability record",
        "drv_experience": "experience",
        "team_avg_finish_3": "team's recent form",
        "team_avg_finish_5": "team's form",
        "team_points_rate_5": "team's points rate",
        "team_dnf_rate_10": "team reliability",
        "champ_position_before": "championship position",
        "champ_points_before": "championship points",
        "drv_circuit_avg_finish": "record at this circuit",
        "team_circuit_avg_finish": "team's record at this circuit",
        "circuit_overtaking_score": "how much this circuit reshuffles the order",
        "circuit_dnf_rate": "how punishing this circuit is",
        "drv_teammate_quali_edge": "head-to-head against a teammate",
        "season_progress": "point in the season",
        "fp_long_run_gap_pct": "long-run practice pace",
        "fp_best_gap_pct": "single-lap practice pace",
        "practice_available": "practice data",
        "team_pace_gap_pct": "the car's recent one-lap pace",
        "drv_pace_gap_pct": "their own recent one-lap pace",
        "team_pace_trend": "which way the car is trending",
        "drv_positions_gained_5": "places usually made up on Sunday",
        "drv_podium_rate_10": "how often they reach the podium",
        "drv_top10_rate_10": "how often they score",
        "drv_circuit_starts": "starts at this circuit",
        "circuit_pole_win_rate": "how often pole converts here",
        "drv_avg_quali_3": "qualifying form over three races",
        "drv_avg_quali_5": "qualifying form over five races",
        "drv_pole_rate_10": "how often they take pole",
        "drv_quali_top3_rate_10": "how often they qualify in the top three",
        "team_avg_quali_5": "the car's recent qualifying form",
    }


def _season_outlook(race: pd.DataFrame, names: dict, season: int, rnd: int) -> dict:
    """Where the championship lands if current form holds. See championship.py."""
    try:
        out = championship.project(
            race["driver_id"].tolist(),
            race["constructor_id"].tolist(),
            race["season_score"].to_numpy(),
            simulate.dnf_probability(race),
            season,
            rnd - 1,
        )
    except Exception as exc:  # noqa: BLE001 - a missing standings row must not
        log.warning("season projection unavailable: %s", exc)
        return {}
    if not out:
        return {}

    drivers = out["drivers"].head(5).copy()
    drivers["name"] = [names.get(d, (d, d))[1] for d in drivers["driver_id"]]
    history = championship.points_history(season)
    top_ids = out["drivers"].head(5)["driver_id"].tolist()

    # One series per driver: what happened, then where it goes.
    series = []
    for did in top_ids:
        past = history[history.driver_id == did]
        points = {int(r): float(v) for r, v in zip(past["round"], past["points"])}
        points.update({int(r): float(v) for r, v in out["projection"][did].items()})
        last = rnd - 1
        anchor = points.get(last)
        low = {int(r): float(v) for r, v in out["projection_low"][did].items()}
        high = {int(r): float(v) for r, v in out["projection_high"][did].items()}
        # The band starts pinched at the last race actually run - there is no
        # uncertainty about points already scored.
        if anchor is not None:
            low[last] = high[last] = anchor
        series.append(
            {
                "label": names.get(did, (did, did))[1],
                "team": out["drivers"].set_index("driver_id").loc[did, "team"],
                "points": points,
                "low": low,
                "high": high,
            }
        )
    # Teammates share a constructor colour; the chart tints the second car in
    # a garage lighter. Dash is reserved for "projected".
    seen: set[str] = set()
    for entry in series:
        if entry["team"] in seen:
            entry["second_car"] = True
        seen.add(entry["team"])

    top = out["drivers"].iloc[0]
    return {
        "leader_wins": float(top["exp_wins"]),
        "drivers": drivers.to_dict("records"),
        "constructors": out["constructors"].head(5).assign(team_name=lambda d: d["team"]).to_dict("records"),
        "series": series,
        "last_actual_round": rnd - 1,
        "n_races": out["n_races"],
        "n_sims": out["n_sims"],
        "points_available": out["points_available"],
        "sprints_left": out.get("sprints_left", 0),
        "lead": out["lead"],
        "clinch_in": out["clinch_in"],
        "clinched": out["clinched"],
        "after_round": out["after_round"],
    }


# Near-duplicate features split credit, sometimes with opposite signs, which
# reads as a contradiction ("helped by grid, held back by qualifying"). They are
# summed within these groups before anything is said about them.
EXPLANATION_GROUPS = {
    "grid": "grid",
    "quali_position": "grid",
    "drv_avg_grid_5": "quali_form",
    "drv_avg_quali_3": "quali_form",
    "drv_avg_quali_5": "quali_form",
    "drv_pole_rate_10": "quali_form",
    "drv_avg_finish_3": "race_form",
    "drv_avg_finish_5": "race_form",
    "drv_points_rate_5": "race_form",
    "drv_podium_rate_10": "race_form",
    "drv_top10_rate_10": "race_form",
    "team_avg_finish_3": "team_form",
    "team_avg_finish_5": "team_form",
    "team_points_rate_5": "team_form",
    "drv_pace_gap_pct": "one_lap_pace",
    "team_pace_gap_pct": "one_lap_pace",
    "quali_gap_to_pole_pct": "one_lap_pace",
    "team_avg_quali_5": "one_lap_pace",
    "champ_position_before": "championship",
    "champ_points_before": "championship",
    "drv_dnf_rate_10": "reliability",
    "team_dnf_rate_10": "reliability",
    "quali_gap_to_teammate_pct": "teammate",
    "drv_teammate_quali_edge": "teammate",
}
GROUP_LABELS = {
    "grid": "starting position",
    "quali_form": "recent qualifying form",
    "race_form": "recent race results",
    "team_form": "the team's recent results",
    "one_lap_pace": "one-lap pace",
    "championship": "championship position",
    "reliability": "reliability record",
    "teammate": "head-to-head against a teammate",
}


def grouped_contributions(ranker: model.Ranker, race: pd.DataFrame) -> pd.DataFrame:
    """SHAP contributions to each driver's score, summed within feature groups.

    Rows are drivers (race order), columns are groups; each row sums to the
    driver's score relative to the field. See model.Ranker.contributions.
    """
    phi = pd.DataFrame(ranker.contributions(race), columns=ranker.feature_names)
    groups = [EXPLANATION_GROUPS.get(f, f) for f in ranker.feature_names]
    return phi.T.groupby(groups, sort=False).sum().T


def explain(ranker: model.Ranker, race: pd.DataFrame) -> tuple[list[str], list[dict]]:
    """One plain sentence per driver, plus the numbers behind it.

    "Helped by" the largest group pushing the driver up the order, "held back
    by" the largest pulling them down. SHAP says what the model leaned on for
    this forecast; it is not a claim about what causes a result.
    """
    labels = {**feature_labels(), **GROUP_LABELS}
    unlabelled = [f for f in ranker.feature_names if EXPLANATION_GROUPS.get(f, f) not in labels]
    if unlabelled:
        log.warning("no plain-English label for: %s", ", ".join(unlabelled))

    grouped = grouped_contributions(ranker, race)
    lines, details = [], []
    for _, row in grouped.iterrows():
        up = row[row > 0].sort_values(ascending=False)
        down = row[row < 0].sort_values()
        parts = []
        if not up.empty:
            parts.append(f"helped by {labels.get(up.index[0], up.index[0])}")
        if not down.empty:
            parts.append(f"held back by {labels.get(down.index[0], down.index[0])}")
        line = "; ".join(parts)
        # Not .capitalize(): that lowercases the rest of the sentence.
        lines.append(line[:1].upper() + line[1:] if line else "")
        top = row.reindex(row.abs().sort_values(ascending=False).index).head(4)
        details.append({labels.get(k, k): round(float(v), 3) for k, v in top.items()})
    return lines, details


# ---------------------------------------------------------------------------
def run(
    season: int | None = None,
    rnd: int | None = None,
    n_sims: int = config.N_SIMULATIONS,
    settings: backtest.Settings | None = None,
) -> Prediction:
    """Forecast one race with the pipeline the walk-forward grades.

    Before qualifying: the qualifying model forecasts the grid, the race model
    scores the projected weekend, and every simulated race draws its own grid.
    After qualifying: the official grid (or, until it is published, the
    qualifying order - recorded as such) and the real qualifying result go to
    the race model. The qualifying forecast is then shown for comparison only;
    it never stands in for the result.
    """
    s = settings or backtest.load_settings()
    df = features.build(include_upcoming=True)
    features.save(df)

    if season is None or rnd is None:
        season, rnd = next_race(df)

    race = df[(df.season == season) & (df["round"] == rnd)].copy().reset_index(drop=True)
    if race.empty:
        raise RuntimeError(f"No entries for {season} round {rnd}")
    seq = int(race["race_seq"].iloc[0])

    history = df[df["race_seq"] < seq]
    if history["race_seq"].nunique() < backtest.MIN_TRAIN_RACES:
        raise RuntimeError("Not enough completed races to train on")

    known_grid = grid_is_known(race)
    stage = "post_quali" if known_grid else "pre_quali"
    quali_model = backtest.quali_trainer(s)(history)
    race_model = backtest.race_trainer(s)(history)
    t_race, t_quali = backtest.trailing_temperatures(df, seq, s, stage)

    # ---- qualifying ------------------------------------------------------
    race["quali_score"] = quali_model.score(race)
    q = simulate.ranking_forecast(race["driver_id"].tolist(), race["quali_score"].to_numpy(), t_quali)
    for col in ("p_win", "p_top5", "p_top10", "exp_position"):
        race[f"q_{col}"] = q.column(col)

    # ---- race ------------------------------------------------------------
    if known_grid:
        scored = race
    else:
        scored = backtest.projected_weekend(race, race["quali_score"].to_numpy())
        race["grid"] = scored["grid"].to_numpy()  # shown as the projected start
    race["score"] = race_model.score(scored)
    inputs = simulate.race_inputs(
        scored, race["score"].to_numpy(), grid_known=known_grid, quali_scores=race["quali_score"].to_numpy()
    )
    # Grids drawn before qualifying use the tuned qualifying temperature, and
    # the qualifying board its trailing refit: each exactly as it is graded
    # (backtest.walk_forward and backtest.walk_forward_quali).
    fc = simulate.forecast(inputs, t_race, s.blend_weight, n_sims, grid_temperature=s.quali_temperature)
    problems = probability.check_distribution(fc.matrix)
    if problems:
        raise RuntimeError(f"forecast is not a coherent distribution: {problems}")
    for col in ("p_win", "p_podium", "p_top5", "p_top10", "exp_position"):
        race[col] = fc.column(col)

    # This race is scored on this race - its grid, its track. The rest of the
    # season isn't, so each driver's season strength comes from their own
    # recent weekends instead. See championship.season_strength.
    race["season_score"] = championship.season_strength(race_model, race, history)
    race["why"], why_detail = explain(race_model, scored)
    race["why_detail"] = why_detail

    # ---- assemble --------------------------------------------------------
    names = _driver_names()
    meta_race = race.iloc[0]

    def lines(sort_col: str, win_col: str, pod: str, top5: str, top10: str, exp: str) -> list[dict]:
        d = race.sort_values(sort_col, ascending=False).head(config.TOP_N)
        return [
            asdict(
                DriverLine(
                    driver_id=r.driver_id,
                    name=names.get(r.driver_id, (r.driver_id, r.driver_id))[0],
                    short=names.get(r.driver_id, (r.driver_id, r.driver_id))[1],
                    team=r.constructor_id,
                    p_win=round(float(getattr(r, win_col)), 4),
                    p_podium=round(float(getattr(r, pod)), 4),
                    p_top5=round(float(getattr(r, top5)), 4),
                    p_top10=round(float(getattr(r, top10)), 4),
                    exp_position=round(float(getattr(r, exp)), 2),
                    grid=int(r.grid) if pd.notna(r.grid) else None,
                    why=getattr(r, "why", "") if sort_col == "p_win" else "",
                )
            )
            for r in d.itertuples()
        ]

    with connect(read_only=True) as con:
        race_name, quali_start = con.execute(
            "SELECT race_name, quali_start_utc FROM raw_races WHERE season = ? AND round = ?", [season, rnd]
        ).fetchone() or (None, None)

    return Prediction(
        season=season,
        round=rnd,
        race_name=race_name or f"{season} round {rnd}",
        circuit_id=str(meta_race["circuit_id"]),
        race_start_utc=str(meta_race["race_start_utc"]) if pd.notna(meta_race["race_start_utc"]) else None,
        quali_start_utc=str(quali_start) if quali_start is not None and pd.notna(quali_start) else None,
        generated_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        grid_known=known_grid,
        quali_board=lines("q_p_win", "q_p_win", "q_p_top5", "q_p_top5", "q_p_top10", "q_exp_position"),
        race_board=lines("p_win", "p_win", "p_podium", "p_top5", "p_top10", "exp_position"),
        field_probs=[
            {
                "driver_id": r.driver_id,
                "name": names.get(r.driver_id, (r.driver_id, r.driver_id))[1],
                "team": r.constructor_id,
                "p_win": round(float(r.p_win), 4),
                "p_podium": round(float(r.p_podium), 4),
                "p_top5": round(float(r.p_top5), 4),
                "p_top10": round(float(r.p_top10), 4),
                "exp_position": round(float(r.exp_position), 2),
                "grid": int(r.grid) if pd.notna(r.grid) else None,
                "quali_position": int(r.quali_position) if pd.notna(r.quali_position) else None,
                "q_exp_position": round(float(r.q_exp_position), 2),
                "why": r.why,
                "why_detail": r.why_detail,
            }
            for r in race.sort_values("p_win", ascending=False).itertuples()
        ],
        position_matrix=[[round(float(v), 4) for v in row] for row in fc.matrix],
        season_outlook=_season_outlook(race, names, season, rnd),
        meta={
            "stage": stage,
            "grid_source": _dominant(race["grid_source"]) if known_grid else "projected",
            "entry_source": _dominant(race["entry_source"]),
            "trained_through": list(race_model.trained_through),
            "n_training_races": int(history["race_seq"].nunique()),
            "n_simulations": n_sims,
            "temperature": t_race,
            "quali_temperature": t_quali,
            "blend_weight": s.blend_weight,
            "recency_weight": s.recency,
            "practice_data": bool(race["practice_available"].max() > 0),
            "circuit_seen_before": bool(pd.notna(meta_race["circuit_overtaking_score"])),
            # position_matrix rows follow this order, which is the order the
            # simulation ran in - not the sorted output order.
            "matrix_driver_ids": race["driver_id"].tolist(),
            "provenance": provenance.record(
                model_version=provenance.model_version(
                    race_model.feature_names, model.PARAMS, features.FEATURE_VERSION
                ),
                quali_model_version=provenance.model_version(
                    quali_model.feature_names, model.PARAMS, features.FEATURE_VERSION
                ),
                feature_version=features.FEATURE_VERSION,
                training_cutoff=list(race_model.trained_through),
            ),
        },
    )
