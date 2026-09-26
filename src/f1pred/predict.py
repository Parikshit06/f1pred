"""Forecast one race and log it.

Before qualifying the grid is unknown, so each simulated race samples its own
grid from the qualifying model. After qualifying the real grid is used. Both
calls are logged to predictions/ with the time they were made and are never
overwritten - that folder is the track record.
"""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import championship, config, features, model, simulate
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
        if not self.race_start_utc:
            return None
        start = pd.to_datetime(self.race_start_utc, errors="coerce", utc=True)
        if pd.isna(start):
            return None
        return float((start - pd.Timestamp(datetime.now(UTC))).total_seconds()) / 86400

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

        p = self.path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2, default=str))
        return p


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
    return bool(race["quali_position"].notna().sum() >= len(race) * 0.8)


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
        "drv_avg_finish_5": "form over five races",
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
        "lead": out["lead"],
        "clinch_in": out["clinch_in"],
        "clinched": out["clinched"],
        "after_round": out["after_round"],
    }


def explain(ranker: model.Ranker, race: pd.DataFrame, top_k: int = 2) -> list[str]:
    """One plain sentence per driver, from SHAP contributions.

    A ranking score tells you nothing on its own. "The model liked this driver
    mostly because of the car's recent pace, despite a poor grid slot" is
    something a reader can disagree with, which is the point.
    """
    try:
        import shap
    except ImportError:
        return [""] * len(race)

    labels = feature_labels()
    unlabelled = [f for f in ranker.feature_names if f not in labels]
    if unlabelled:
        log.warning("no plain-English label for: %s", ", ".join(unlabelled))

    try:
        explainer = shap.TreeExplainer(ranker.booster)
        values = explainer.shap_values(race[ranker.feature_names])
    except Exception as exc:  # noqa: BLE001
        log.debug("SHAP unavailable: %s", exc)
        return [""] * len(race)

    # Near-duplicate features split credit with opposite signs, which reads as a
    # contradiction ("helped by grid, held back by qualifying"). Group them.
    synonyms = {
        "grid": "grid",
        "quali_position": "grid",
        "drv_avg_grid_5": "quali_form",
        "drv_avg_quali_3": "quali_form",
        "drv_avg_quali_5": "quali_form",
        "drv_avg_finish_3": "race_form",
        "drv_avg_finish_5": "race_form",
        "team_avg_finish_3": "team_form",
        "team_avg_finish_5": "team_form",
        "drv_pace_gap_pct": "one_lap_pace",
        "team_pace_gap_pct": "one_lap_pace",
    }

    def pick(order, sign, used: set[str]) -> list[int]:
        """Strongest contributors in one direction, one per synonym group."""
        chosen = []
        for i in order:
            if sign * row[i] <= 0:
                continue
            name = ranker.feature_names[i]
            group = synonyms.get(name, name)
            if group in used:
                continue
            used.add(group)
            chosen.append(i)
            if len(chosen) >= (top_k - 1 if sign > 0 else 1):
                break
        return chosen

    out = []
    for row in values:
        # Lead with the strongest thing in the driver's favour, then the
        # strongest thing against. Taking the top-k by magnitude regardless of
        # sign made every line read the same when one feature dominated.
        used: set[str] = set()
        helped = pick(np.argsort(-row), 1, used)
        hurt = pick(np.argsort(row), -1, used)

        parts = [f"helped by {labels.get(ranker.feature_names[i], ranker.feature_names[i])}" for i in helped]
        parts += [
            f"held back by {labels.get(ranker.feature_names[i], ranker.feature_names[i])}" for i in hurt
        ]
        line = "; ".join(parts)
        # Not .capitalize() - that lowercases the rest, turning "Sunday" into
        # "sunday" halfway through the sentence.
        out.append(line[:1].upper() + line[1:] if line else "")
    return out


# ---------------------------------------------------------------------------
def run(
    season: int | None = None,
    rnd: int | None = None,
    n_sims: int = config.N_SIMULATIONS,
    temperature: float = config.DEFAULT_TEMPERATURE,
    blend_weight: float = config.DEFAULT_BLEND_WEIGHT,
) -> Prediction:
    df = features.build(include_upcoming=True)
    features.save(df)

    if season is None or rnd is None:
        season, rnd = next_race(df)

    race = df[(df.season == season) & (df["round"] == rnd)].copy().reset_index(drop=True)
    if race.empty:
        raise RuntimeError(f"No entries for {season} round {rnd}")

    history = df[df["race_seq"] < race["race_seq"].iloc[0]]
    if history["race_seq"].nunique() < 20:
        raise RuntimeError("Not enough completed races to train on")

    quali_model = model.train_quali(history)
    race_model = model.train_race(history)

    known_grid = grid_is_known(race)

    # ---- qualifying ------------------------------------------------------
    race["quali_score"] = quali_model.score(race)
    q_probs = simulate.plackett_luce(race["quali_score"].to_numpy(), temperature)
    q_sim = simulate.simulate(
        simulate.SimInputs(
            driver_ids=race["driver_id"].tolist(),
            scores=race["quali_score"].to_numpy(),
            # Qualifying rarely ends a weekend; only a crash or failure does.
            dnf_prob=np.full(len(race), 0.02),
            # No grid in qualifying. Passing one here would have applied a
            # starting-position penalty in dataframe order, which is nonsense -
            # leaving both None gives every driver the same neutral position.
            grid=None,
            grid_scores=None,
            overtaking_score=8.0,  # nothing to overtake; pace alone decides
            safety_car_prob=0.10,
        ),
        n_sims=n_sims,
        temperature=temperature,
    )
    race["q_p_win"] = simulate.blend(q_probs, q_sim["p_win"].to_numpy(), blend_weight)
    race["q_p_top5"] = q_sim["p_top5"].to_numpy()
    race["q_p_top10"] = q_sim["p_points"].to_numpy()
    race["q_exp_pos"] = q_sim["exp_position"].to_numpy()

    # ---- race ------------------------------------------------------------
    # grid comes from results, which don't exist until the race is run, so after
    # qualifying it's empty. The starting grid is the qualifying order (grid
    # penalties aren't modelled).
    if known_grid:
        g = race["grid"].fillna(race["quali_position"]).to_numpy(dtype=float)
        if np.isnan(g).any():
            # grid_is_known only asks for 80% of the field, so a driver who set
            # no time can still be missing here. In a real race they line up at
            # the back, which is also the honest default: last, in field order.
            back = np.nanmax(g) if np.isfinite(g).any() else 0.0
            g[np.isnan(g)] = back + 1.0 + np.arange(int(np.isnan(g).sum()))
        race["grid"] = g
    if not known_grid:
        # Use the qualifying model's expected order as the grid. Assigned as an array
        # so it doesn't rely on race's index matching the simulator's.
        race["grid"] = q_sim["exp_position"].rank(method="first").to_numpy()

    race["score"] = race_model.score(race)
    # This race is scored on this race - its grid, its track. The rest of the
    # season is not at this track or from this grid, so it gets a separate
    # score for an ordinary weekend. See championship.typical_weekend.
    race["season_score"] = race_model.score(championship.typical_weekend(race, history))
    r_probs = simulate.plackett_luce(race["score"].to_numpy(), temperature)

    ot = race["circuit_overtaking_score"].iloc[0]
    sim = simulate.simulate(
        simulate.SimInputs(
            driver_ids=race["driver_id"].tolist(),
            scores=race["score"].to_numpy(),
            dnf_prob=simulate.dnf_probability(race),
            grid=race["grid"].to_numpy() if known_grid else None,
            grid_scores=None if known_grid else race["quali_score"].to_numpy(),
            overtaking_score=float(ot) if pd.notna(ot) else 3.0,
            safety_car_prob=float(np.clip(race["circuit_dnf_rate"].iloc[0] * 2, 0.2, 0.7))
            if pd.notna(race["circuit_dnf_rate"].iloc[0])
            else 0.35,
        ),
        n_sims=n_sims,
        temperature=temperature,
    )
    race["p_win"] = simulate.blend(r_probs, sim["p_win"].to_numpy(), blend_weight)
    for col in ("p_podium", "p_top5", "exp_position"):
        race[col] = sim[col].to_numpy()
    race["p_top10"] = sim["p_points"].to_numpy()

    # Only p_win is blended with the closed form; the wider bands come straight
    # from the simulation, so the blend can lift p_win above p_podium. Floor each
    # band at the one inside it. Blending the whole distribution would be better.
    bands = ["p_win", "p_podium", "p_top5", "p_top10"]
    for inner, outer in itertools.pairwise(bands):
        race[outer] = np.maximum(race[outer].to_numpy(), race[inner].to_numpy())
    race["why"] = explain(race_model, race)

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
        race_name = con.execute(
            "SELECT race_name FROM raw_races WHERE season = ? AND round = ?", [season, rnd]
        ).fetchone()

    pred = Prediction(
        season=season,
        round=rnd,
        race_name=race_name[0] if race_name else f"{season} round {rnd}",
        circuit_id=str(meta_race["circuit_id"]),
        race_start_utc=str(meta_race["race_start_utc"]) if pd.notna(meta_race["race_start_utc"]) else None,
        generated_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        grid_known=known_grid,
        quali_board=lines("q_p_win", "q_p_win", "q_p_top5", "q_p_top5", "q_p_top10", "q_exp_pos"),
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
                "q_exp_position": round(float(r.q_exp_pos), 2),
                "why": r.why,
            }
            for r in race.sort_values("p_win", ascending=False).itertuples()
        ],
        position_matrix=[[round(float(v), 4) for v in row] for row in sim["position_dist"]],
        season_outlook=_season_outlook(race, names, season, rnd),
        meta={
            "trained_through": list(race_model.trained_through),
            "n_training_races": int(history["race_seq"].nunique()),
            "n_simulations": n_sims,
            "temperature": temperature,
            "blend_weight": blend_weight,
            "practice_data": bool(race["practice_available"].max() > 0),
            "circuit_seen_before": bool(pd.notna(ot)),
            # position_matrix rows follow this order, which is the order the
            # simulation ran in - not the sorted output order.
            "matrix_driver_ids": race["driver_id"].tolist(),
        },
    )
    return pred
