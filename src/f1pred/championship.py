"""Season projection: run the rest of the calendar many times from current form.

Each simulated race uses the same noise model as simulate.simulate - pace
variance, safety cars, per-driver retirement hazard - so title odds stay
consistent with the published race probabilities. Sprint weekends run a
shorter race on the Saturday with its own points, taken from the calendar, so
remaining sprint points are in the projection.

Assumes current form holds; it can't see an upgrade coming. The fastest-lap
point (2019-24) is counted in what a driver could still score, but not
simulated: at most a point a race, and not in the 2026 rules.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config
from .simulate import PACE_NOISE, SAFETY_CAR_SPREAD
from .store import connect

log = logging.getLogger(__name__)

# 2010-present scoring, top ten.
POINTS = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1], dtype=float)
# A sprint is about a third of the distance, so a third of the retirement risk.
SPRINT_DISTANCE_SHARE = 1 / 3


def sprint_points(season: int) -> np.ndarray:
    """Sprint scoring by season: top three in 2021, top eight from 2022."""
    if season < 2021:
        return np.array([], dtype=float)
    if season == 2021:
        return np.array([3, 2, 1], dtype=float)
    return np.array([8, 7, 6, 5, 4, 3, 2, 1], dtype=float)


def fastest_lap_point(season: int) -> float:
    """One point for the fastest lap if finishing in the top ten, 2019-2024."""
    return 1.0 if 2019 <= season <= 2024 else 0.0


def round_maximum(season: int, sprint: np.ndarray, cars: int = 1) -> np.ndarray:
    """Most points one driver (cars=1) or one team (cars=2) can take from each round."""
    race = POINTS[:cars].sum() + fastest_lap_point(season)
    sprint_max = sprint_points(season)[:cars].sum()
    return race + np.asarray(sprint, dtype=float) * sprint_max


def standings_after(season: int, rnd: int) -> pd.DataFrame:
    """Driver standings as they stand, including any sprint points scored."""
    with connect(read_only=True) as con:
        return con.execute(
            """
            SELECT driver_id, constructor_id, points, position
            FROM raw_standings
            WHERE season = ? AND round = ? AND driver_id IS NOT NULL
            ORDER BY position
            """,
            [season, rnd],
        ).fetchdf()


def points_history(season: int) -> pd.DataFrame:
    """Cumulative points per driver after every round run so far."""
    with connect(read_only=True) as con:
        return con.execute(
            """
            SELECT round, driver_id, points
            FROM raw_standings
            WHERE season = ? AND driver_id IS NOT NULL
            ORDER BY round
            """,
            [season],
        ).fetchdf()


def remaining_rounds(season: int, after_round: int) -> int:
    return len(remaining_schedule(season, after_round))


def remaining_schedule(season: int, after_round: int) -> pd.DataFrame:
    """Rounds still to run, and which of them have a sprint.

    A sprint weekend is known from the calendar (jolpica lists the Sprint
    session) or, for a round already run, from its sprint results.
    """
    with connect(read_only=True) as con:
        return con.execute(
            """
            SELECT ra.round,
                   (ra.sprint_start_utc IS NOT NULL
                    OR EXISTS (SELECT 1 FROM raw_sprint s
                               WHERE s.season = ra.season AND s.round = ra.round)) AS sprint
            FROM raw_races ra
            WHERE ra.season = ? AND ra.round > ?
            ORDER BY ra.round
            """,
            [season, after_round],
        ).fetchdf()


def constructor_points(season: int, after_round: int) -> dict[str, float]:
    """Constructors' points as awarded: each result counts for the team the car
    was entered by. Summing the current drivers' totals instead credits a team
    with points its new driver scored somewhere else."""
    with connect(read_only=True) as con:
        rows = con.execute(
            """
            SELECT constructor_id, sum(points) FROM (
                SELECT constructor_id, points FROM raw_results WHERE season = ? AND round <= ?
                UNION ALL
                SELECT constructor_id, points FROM raw_sprint WHERE season = ? AND round <= ?
            ) GROUP BY 1
            """,
            [season, after_round, season, after_round],
        ).fetchall()
    return {c: float(p or 0.0) for c, p in rows if c is not None}


def simulate_seasons(
    scores: np.ndarray,
    dnf_prob: np.ndarray,
    base_points: np.ndarray,
    n_races: int,
    team_index: np.ndarray | None = None,
    n_sims: int = 10_000,
    safety_car_prob: float = 0.35,
    season_fraction_left: float = 0.0,
    seed: int = config.RANDOM_SEED,
    sprint: np.ndarray | None = None,
    sprint_table: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Final totals per simulation, plus per-round mean, 10th and 90th percentile
    points and expected wins (what the progression chart draws).

    Same race noise as simulate.simulate, minus the grid term. On top, each
    simulated season draws one pace offset per team and holds it all year:
    race luck averages out over a dozen races, but error in today's read of a
    car doesn't. Sized by config.SEASON_PACE_UNCERTAINTY. It models how far a
    team could drift, not which team will.
    """
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    score_sd = float(np.std(scores)) or 1.0

    # One offset per team, shared by teammates, plus a smaller one per driver so
    # teammates can still drift apart. Both scale with the share of the season
    # still to run.
    left = float(np.clip(season_fraction_left, 0.0, 1.0))
    to_score = left * score_sd / config.POSITIONS_PER_SCORE_SD
    team_sd = config.SEASON_PACE_UNCERTAINTY * to_score
    driver_sd = config.SEASON_DRIVER_UNCERTAINTY * to_score
    drift = np.zeros((1, n))
    if team_sd > 0:
        if team_index is None:
            drift = drift + rng.normal(0.0, team_sd, size=(n_sims, n))
        else:
            drift = drift + rng.normal(0.0, team_sd, size=(n_sims, int(team_index.max()) + 1))[:, team_index]
    if driver_sd > 0:
        drift = drift + rng.normal(0.0, driver_sd, size=(n_sims, n))

    totals = np.tile(np.asarray(base_points, dtype=float), (n_sims, 1))
    track = np.zeros((n_races, n))
    lo_track = np.zeros((n_races, n))
    hi_track = np.zeros((n_races, n))
    wins = np.zeros(n)
    rows = np.arange(n_sims)[:, None]
    sprint = np.zeros(n_races, dtype=bool) if sprint is None else np.asarray(sprint, dtype=bool)
    sprint_table = sprint_points(config.CURRENT_SEASON) if sprint_table is None else sprint_table

    def run(table: np.ndarray, dnf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One race in every simulated season: points awarded, and the order."""
        chaos = np.where(rng.random((n_sims, 1)) < safety_car_prob, SAFETY_CAR_SPREAD, 1.0)
        pace = scores + drift + rng.normal(0.0, 1.0, size=(n_sims, n)) * (PACE_NOISE * score_sd * chaos)
        retired = rng.random((n_sims, n)) < dnf
        pace = np.where(retired, -np.inf, pace)
        order = np.argsort(-pace, axis=1)
        take = min(len(table), n)
        awarded = np.zeros((n_sims, n))
        awarded[rows, order[:, :take]] = table[:take]
        # A car that stopped scores nothing, even in a thin field.
        return np.where(retired, 0.0, awarded), order

    for r in range(n_races):
        if sprint[r] and len(sprint_table):
            sprint_awarded, _ = run(sprint_table, np.asarray(dnf_prob) * SPRINT_DISTANCE_SHARE)
            totals += sprint_awarded
        awarded, order = run(POINTS, np.asarray(dnf_prob))
        np.add.at(wins, order[:, 0], 1)

        totals += awarded
        track[r] = totals.mean(axis=0)
        lo_track[r] = np.percentile(totals, 10, axis=0)
        hi_track[r] = np.percentile(totals, 90, axis=0)

    # Wins are counted per simulated race, so this is expected wins over the
    # remaining calendar.
    return totals, track, lo_track, hi_track, wins / n_sims


def allocate_odds(
    probs: np.ndarray,
    clinched: np.ndarray | None = None,
    rank_by: np.ndarray | None = None,
) -> np.ndarray:
    """Round title odds to 0.1% so the column still sums to 100.

    Largest-remainder rounding, and a leader who hasn't clinched is never shown
    at 100% - the simulation can't resolve past 1 in n_sims, and the real
    uncertainty is whether form holds.
    """
    probs = np.asarray(probs, dtype=float)
    total = probs.sum()
    if total <= 0:
        return np.zeros_like(probs)

    # Largest remainder, in tenths of a percent.
    exact = probs / total * 1000.0
    tenths = np.floor(exact).astype(int)
    short = 1000 - tenths.sum()
    if short > 0:
        for i in np.argsort(-(exact - np.floor(exact)))[:short]:
            tenths[i] += 1

    leader = int(np.argmax(tenths))
    is_clinched = bool(clinched[leader]) if clinched is not None else False
    if tenths[leader] >= 1000 and not is_clinched:
        # Everyone else is on a simulated zero, so ordering by probability would
        # pick an arbitrary rival. Order by points when given them.
        order = np.argsort(-(rank_by if rank_by is not None else probs))
        rival = int(order[1]) if len(probs) > 1 else leader
        if rival != leader:
            tenths[leader] -= 1
            tenths[rival] += 1

    return tenths / 1000.0


def clinch_round(
    leader_points: float, rival_points: float, races_left: int | np.ndarray, win: float = 25.0
) -> int | None:
    """Rounds from now until the leader could mathematically seal the title.

    Leader takes the maximum from each round and the rival nothing: settled
    after k rounds once L + M[:k] > R + M[k:], where M holds each remaining
    round's maximum (sprints and bonus points included). Pass the rounds'
    maxima, or a count of plain race weekends worth `win` each. None if not
    before the finale.
    """
    maxima = np.full(int(races_left), win) if np.isscalar(races_left) else np.asarray(races_left, dtype=float)
    if len(maxima) == 0:
        return None
    gap = leader_points - rival_points
    total = maxima.sum()
    for k in range(1, len(maxima) + 1):
        taken = maxima[:k].sum()
        if gap + taken > total - taken + 1e-9:
            return k
    return None


def project(
    driver_ids: list[str],
    teams: list[str],
    scores: np.ndarray,
    dnf_prob: np.ndarray,
    season: int,
    after_round: int,
    n_sims: int = 10_000,
    safety_car_prob: float = 0.35,
) -> dict:
    """Title projection for drivers and constructors."""
    standings = standings_after(season, after_round)
    if standings.empty:
        return {}

    have = dict(zip(standings["driver_id"], standings["points"]))
    base = np.array([have.get(d, 0.0) for d in driver_ids], dtype=float)
    schedule = remaining_schedule(season, after_round)
    n_races = len(schedule)
    if n_races == 0:
        return {}
    sprint = schedule["sprint"].to_numpy(dtype=bool)
    driver_max = round_maximum(season, sprint)
    team_max = round_maximum(season, sprint, cars=2)

    names = {t: i for i, t in enumerate(dict.fromkeys(teams))}
    team_index = np.array([names[t] for t in teams])
    totals, track, lo_track, hi_track, wins = simulate_seasons(
        scores,
        dnf_prob,
        base,
        n_races,
        n_sims=n_sims,
        safety_car_prob=safety_car_prob,
        team_index=team_index,
        season_fraction_left=n_races / max(after_round + n_races, 1),
        sprint=sprint,
        sprint_table=sprint_points(season),
    )

    # Everyone with points stays in the title race, entered this weekend or
    # not: a driver sitting out (illness, a stand-in in their car) keeps what
    # they have. Dropping them once handed a title already clinched - Hamilton,
    # 2020, out with COVID - to their teammate at 90%. They score nothing more in
    # the simulation, which for a driver who returns is conservative.
    n_field = len(driver_ids)
    absent = standings[~standings["driver_id"].isin(driver_ids)]
    if not absent.empty:
        held = absent["points"].to_numpy(dtype=float)
        totals = np.hstack([totals, np.tile(held, (n_sims, 1))])
        track, lo_track, hi_track = (
            np.hstack([a, np.tile(held, (n_races, 1))]) for a in (track, lo_track, hi_track)
        )
        wins = np.concatenate([wins, np.zeros(len(held))])
        base = np.concatenate([base, held])
        driver_ids = [*driver_ids, *absent["driver_id"].tolist()]
        teams = [*teams, *absent["constructor_id"].fillna("unknown").tolist()]

    # Ties for the title are vanishingly rare and are broken on countback in
    # reality; argmax picks one, which is close enough at this precision.
    champion = np.argmax(totals, axis=1)
    raw_title = np.bincount(champion, minlength=len(driver_ids)) / n_sims
    rank = (-totals).argsort(axis=1).argsort(axis=1) + 1

    # Mathematically alive: could still reach the leader by taking everything
    # left - sprints and bonus points included - while the leader scores nothing.
    # Anyone in the standings counts as a leader, raced this weekend or not.
    leader_points = max(base.max(), float(standings["points"].max()))
    alive = base + driver_max.sum() >= leader_points
    clinched = alive.sum() == 1
    title = allocate_odds(raw_title, alive & clinched, totals.mean(axis=0))

    drivers = pd.DataFrame(
        {
            "driver_id": driver_ids,
            "team": teams,
            "now": base,
            "projected": totals.mean(axis=0),
            "low": np.percentile(totals, 10, axis=0),
            "high": np.percentile(totals, 90, axis=0),
            "exp_wins": wins,
            "p_title": title,
            "alive": alive,
            "p_top3": (rank <= 3).mean(axis=0),
        }
    ).sort_values("projected", ascending=False)

    # Constructors: both cars, summed inside each simulated season, so a team's
    # title odds account for the correlation between its two drivers.
    # Each team starts from the points it was actually awarded, then adds what
    # its current drivers score in each simulated season.
    team_names = sorted(set(teams))
    idx = {t: [i for i, x in enumerate(teams) if x == t] for t in team_names}
    awarded = constructor_points(season, after_round)
    team_base = np.array([awarded.get(t, base[idx[t]].sum()) for t in team_names])
    team_totals = np.stack(
        [team_base[j] + (totals[:, idx[t]] - base[idx[t]]).sum(axis=1) for j, t in enumerate(team_names)],
        axis=1,
    )
    t_champ = np.argmax(team_totals, axis=1)
    t_raw = np.bincount(t_champ, minlength=len(team_names)) / n_sims
    t_alive = team_base + team_max.sum() >= team_base.max()
    constructors = pd.DataFrame(
        {
            "team": team_names,
            "now": team_base,
            "projected": team_totals.mean(axis=0),
            "low": np.percentile(team_totals, 10, axis=0),
            "high": np.percentile(team_totals, 90, axis=0),
            "p_title": allocate_odds(t_raw, t_alive & (t_alive.sum() == 1), team_totals.mean(axis=0)),
            "alive": t_alive,
        }
    ).sort_values("projected", ascending=False)

    # Points gap between teammates, first-listed minus second. Graded by
    # calibrate-spread; the team offset cancels here, so only the per-driver
    # term can widen it.
    pairs = []
    for t in team_names:
        field = [i for i in idx[t] if i < n_field]  # this weekend's two cars
        if len(field) == 2:
            a, b = field
            gap = totals[:, a] - totals[:, b]
            pairs.append(
                {
                    "team": t,
                    "first": driver_ids[a],
                    "second": driver_ids[b],
                    "low": float(np.percentile(gap, 10)),
                    "high": float(np.percentile(gap, 90)),
                }
            )

    return {
        "drivers": drivers,
        "constructors": constructors,
        "teammates": pd.DataFrame(pairs, columns=["team", "first", "second", "low", "high"]),
        "projection": pd.DataFrame(
            track, columns=driver_ids, index=range(after_round + 1, after_round + 1 + n_races)
        ),
        "projection_low": pd.DataFrame(
            lo_track, columns=driver_ids, index=range(after_round + 1, after_round + 1 + n_races)
        ),
        "projection_high": pd.DataFrame(
            hi_track, columns=driver_ids, index=range(after_round + 1, after_round + 1 + n_races)
        ),
        "n_races": n_races,
        "n_sims": n_sims,
        "after_round": after_round,
        "points_available": int(driver_max.sum()),
        "sprints_left": int(sprint.sum()),
        "lead": float(np.sort(base)[-1] - np.sort(base)[-2]) if len(base) > 1 else 0.0,
        "clinch_in": clinch_round(float(np.sort(base)[-1]), float(np.sort(base)[-2]), driver_max)
        if len(base) > 1
        else None,
        "clinched": bool(clinched),
    }


# ---------------------------------------------------------------------------
# What a driver is worth on a weekend that is not this one
# ---------------------------------------------------------------------------
TYPICAL_WINDOW = 8
MIN_OWN_RACES = 3


def season_strength(
    ranker, race: pd.DataFrame, history: pd.DataFrame, window: int = TYPICAL_WINDOW
) -> np.ndarray:
    """Each driver's strength for the rest of the season: the median of the
    model's scores for their own last `window` real weekends.

    The rest of the season isn't at this track or from this grid, so the next
    race's score is the wrong input. An earlier version built a synthetic
    "typical weekend" and scored that, which copied recent form into the circuit
    columns (counting it twice) and set each driver's teammate gap to their
    average. Scoring the weekends that actually happened avoids inventing inputs
    the model has never seen, and the median keeps one wrecked weekend from
    setting a driver's season. It projected final points and title odds better
    on 2019-22, across every driver and team (reports/experiments.json,
    season_projection); on 2023- the three approaches were level.

    Scores are standardised within each race, so a median across races compares
    like with like. A driver with fewer than MIN_OWN_RACES recent weekends (a
    rookie, a returning reserve) is scored on a typical weekend instead.
    """
    names = race["driver_id"].tolist()
    run = history[history["position"].notna()]
    seqs = sorted(run.loc[run["driver_id"].isin(names), "race_seq"].unique())[-window:]
    per_race = []
    for seq in seqs:
        # Scored on that race's whole field: scores are standardised within a
        # race, so leaving out drivers who have since gone would shift them.
        r = run[run["race_seq"] == seq]
        if len(r) >= 5:
            per_race.append(pd.Series(ranker.score(r), index=r["driver_id"].to_numpy()).reindex(names))
    table = pd.concat(per_race, axis=1) if per_race else pd.DataFrame(index=names)
    enough = table.notna().sum(axis=1) >= MIN_OWN_RACES
    typical = table[enough].median(axis=1)
    if all(d in typical.index for d in names):
        return typical.reindex(names).to_numpy(dtype=float)
    fallback = pd.Series(ranker.score(typical_weekend(race, history, method="median8")), index=names)
    return np.array([typical[d] if d in typical.index else fallback[d] for d in names], dtype=float)


# Ten of the race model's features describe the weekend rather than the driver:
# where they start, and the track they start at. Together they carry about half
# the model - qualifying position alone is a third of it.
WEEKEND_GRID = ("grid", "quali_position", "quali_gap_to_pole_pct", "quali_gap_to_teammate_pct")
WEEKEND_CIRCUIT = (
    "drv_circuit_avg_finish",
    "team_circuit_avg_finish",
    "circuit_overtaking_score",
    "circuit_dnf_rate",
    "drv_circuit_starts",
    "circuit_pole_win_rate",
)


def typical_weekend(race: pd.DataFrame, history: pd.DataFrame, method: str = "mean5") -> pd.DataFrame:
    """The same field on an ordinary weekend at an ordinary track.

    The rest of the season isn't at this circuit or from this grid, so the
    projection shouldn't be scored on either. Scoring it with the next race's
    features let one qualifying session reshape nine races.

    method (compared in evaluation.season_projection_experiment):
      mean5    grid and qualifying from the driver's 5-race means, gaps from
               their 5-race means, track record replaced by recent form
      neutral  as mean5, but track record and the weekend's teammate gap set
               to the field's typical values: recent form and the teammate
               record are already features, so copying them into these
               columns counted them twice
      median8  as neutral, with medians over the last 8 races, so one bad
               weekend doesn't set a driver's typical grid slot
    """
    t = race.copy()
    cols = [c for c in WEEKEND_GRID + WEEKEND_CIRCUIT if c in history.columns]
    med = history[cols].median(numeric_only=True)
    window, stat = (8, "median") if method == "median8" else (5, "mean")

    def col(name: str) -> pd.Series:
        return t[name] if name in t.columns else pd.Series(np.nan, index=t.index)

    recent = history.sort_values("race_seq").groupby("driver_id").tail(window)

    def per_driver(c: str) -> pd.Series:
        if c not in recent.columns:
            return pd.Series(np.nan, index=t.index)
        return t["driver_id"].map(recent.groupby("driver_id")[c].agg(stat))

    # .fillna with a Series aligns on index; every Series here is built from
    # t's own index, so they line up by construction.
    if method == "median8":
        t["quali_position"] = per_driver("quali_position").fillna(med.get("quali_position"))
        t["grid"] = per_driver("grid").fillna(t["quali_position"])
    else:
        t["quali_position"] = (
            col("drv_avg_quali_5").fillna(col("drv_avg_grid_5")).fillna(med.get("quali_position"))
        )
        t["grid"] = col("drv_avg_grid_5").fillna(t["quali_position"])

    if "quali_gap_to_pole_pct" in history.columns:
        t["quali_gap_to_pole_pct"] = per_driver("quali_gap_to_pole_pct").fillna(
            med.get("quali_gap_to_pole_pct")
        )
    if "quali_gap_to_teammate_pct" in history.columns:
        t["quali_gap_to_teammate_pct"] = (
            med.get("quali_gap_to_teammate_pct")
            if method != "mean5"
            else per_driver("quali_gap_to_teammate_pct").fillna(med.get("quali_gap_to_teammate_pct"))
        )

    for c, form in (
        ("drv_circuit_avg_finish", "drv_avg_finish_5"),
        ("team_circuit_avg_finish", "team_avg_finish_5"),
    ):
        if c in t.columns:
            t[c] = col(form).fillna(med.get(c)) if method == "mean5" else med.get(c)
    for c in ("circuit_overtaking_score", "circuit_dnf_rate", "circuit_pole_win_rate", "drv_circuit_starts"):
        if c in t.columns:
            t[c] = med.get(c)
    return t
