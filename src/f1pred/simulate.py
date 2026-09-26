"""Stochastic race-outcome simulation over the learned ranking model.

Not a physics model. Each simulated race perturbs the ranker's scores:

    pace noise   normal, sd = PACE_NOISE x the spread of scores in the field
    safety car   with the circuit's probability, the noise widens by
                 SAFETY_CAR_SPREAD - a stand-in for a neutralised race
                 handing out luck, not a model of when one happens
    grid         a pull toward the starting order, stronger where the
                 circuit's history says overtaking is hard
    retirement   each car independently, at its driver and team DNF rates
    unknown grid before qualifying, each run draws its own grid from the
                 qualifying model

Not modelled: tyre and pit strategy, weather, safety-car timing, penalties,
team orders, first-lap incidents, and failures shared by a team's two cars.

forecast() is the one entry point. The walk-forward evaluation and the live
forecast both call it, so the numbers that are graded are produced exactly the
way the published ones are.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config, probability

PACE_NOISE = 0.55
SAFETY_CAR_SPREAD = 1.9
DEFAULT_OVERTAKING = 3.0  # mean |grid - finish| at a circuit with no history
DEFAULT_SAFETY_CAR = 0.35


@dataclass
class SimInputs:
    driver_ids: list[str]
    scores: np.ndarray  # higher = better
    dnf_prob: np.ndarray  # per driver, 0..1
    grid: np.ndarray | None = None  # known grid, or None before qualifying
    grid_scores: np.ndarray | None = None  # quali-model scores, used when grid is None
    overtaking_score: float = DEFAULT_OVERTAKING
    safety_car_prob: float = DEFAULT_SAFETY_CAR


def grid_pull(overtaking_score: float) -> float:
    """How much a track lets pace show: where nobody overtakes, the grid decides."""
    ot = DEFAULT_OVERTAKING if not np.isfinite(overtaking_score) else float(overtaking_score)
    return float(np.clip(1.6 - ot / 8.0, 0.15, 1.4))


def simulate_matrix(
    inputs: SimInputs,
    n_sims: int = config.N_SIMULATIONS,
    grid_temperature: float = 1.0,
    seed: int = config.RANDOM_SEED,
) -> np.ndarray:
    """Run the race n_sims times; return shares as a matrix [driver, position]."""
    rng = np.random.default_rng(seed)
    scores = np.asarray(inputs.scores, dtype=float)
    n = len(scores)
    score_sd = float(np.std(scores)) or 1.0

    if inputs.grid is not None:
        grid_given = np.asarray(inputs.grid, dtype=float)
        # A grid with gaps doesn't raise on its own - pace goes NaN and the
        # simulation averages into a near-uniform field. Fail instead.
        if not np.isfinite(grid_given).all():
            missing = int((~np.isfinite(grid_given)).sum())
            raise ValueError(
                f"grid has {missing} missing entries of {n}. A race that has not run yet "
                "carries no grid in results - fill it from qualifying before simulating."
            )
        grid = np.broadcast_to(grid_given, (n_sims, n))
    elif inputs.grid_scores is not None:
        orders = probability.sample_orders(rng, inputs.grid_scores, grid_temperature, n_sims)
        grid = np.empty((n_sims, n))
        grid[np.arange(n_sims)[:, None], orders] = np.arange(1, n + 1)[None, :]
    else:
        grid = np.full((n_sims, n), (n + 1) / 2.0)

    chaos = np.where(rng.random((n_sims, 1)) < inputs.safety_car_prob, SAFETY_CAR_SPREAD, 1.0)
    pace = (
        scores[None, :]
        + rng.normal(0.0, 1.0, size=(n_sims, n)) * (PACE_NOISE * score_sd * chaos)
        - grid_pull(inputs.overtaking_score) * (grid - 1) * (score_sd / 6.0)
    )
    # Retirements are classified behind every finisher, ordered by how far they
    # got - which a random draw stands in for.
    retired = rng.random((n_sims, n)) < np.asarray(inputs.dnf_prob, dtype=float)[None, :]
    pace = np.where(retired, -1e6 + rng.random((n_sims, n)), pace)

    return probability.position_counts(np.argsort(-pace, axis=1))


def simulate(
    inputs: SimInputs,
    n_sims: int = config.N_SIMULATIONS,
    temperature: float = 1.0,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """The simulation on its own, summarised per driver.

    `temperature` only shapes the grid drawn before qualifying.
    """
    matrix = simulate_matrix(inputs, n_sims, grid_temperature=temperature, seed=seed)
    out = probability.summarise(matrix)
    out.insert(0, "driver_id", inputs.driver_ids)
    out["p_points"] = out["p_top10"]
    out["position_dist"] = list(matrix)
    return out


# ---------------------------------------------------------------------------
# Inputs from a feature frame
# ---------------------------------------------------------------------------
def dnf_probability(df: pd.DataFrame, floor: float = 0.02, cap: float = 0.35) -> np.ndarray:
    """Blend the driver's and the team's recent retirement rates.

    Team reliability dominates - an engine does not care who is steering - but
    driver rate carries crash risk, so both go in.
    """
    drv = df["drv_dnf_rate_10"].fillna(0.12).to_numpy()
    team = df["team_dnf_rate_10"].fillna(0.12).to_numpy()
    return np.clip(0.35 * drv + 0.65 * team, floor, cap)


def safety_car_probability(circuit_dnf_rate: float) -> float:
    """A heuristic: tracks that stop cars also bring out safety cars."""
    if circuit_dnf_rate is None or not np.isfinite(circuit_dnf_rate):
        return DEFAULT_SAFETY_CAR
    return float(np.clip(circuit_dnf_rate * 2, 0.2, 0.7))


def starting_grid(grid: pd.Series | np.ndarray) -> np.ndarray:
    """A complete grid: anyone without a slot starts at the back, in field order."""
    g = np.asarray(grid, dtype=float).copy()
    missing = np.isnan(g)
    if missing.any():
        back = np.nanmax(g) if np.isfinite(g).any() else 0.0
        g[missing] = back + 1.0 + np.arange(int(missing.sum()))
    return g


def race_inputs(
    race: pd.DataFrame, scores: np.ndarray, grid_known: bool, quali_scores: np.ndarray | None = None
) -> SimInputs:
    """Everything the simulation needs about one race, from its feature rows."""
    first = race.iloc[0]
    return SimInputs(
        driver_ids=race["driver_id"].tolist(),
        scores=np.asarray(scores, dtype=float),
        dnf_prob=dnf_probability(race),
        grid=starting_grid(race["grid"]) if grid_known else None,
        grid_scores=None if grid_known else quali_scores,
        overtaking_score=float(first.get("circuit_overtaking_score", np.nan)),
        safety_car_prob=safety_car_probability(float(first.get("circuit_dnf_rate", np.nan))),
    )


# ---------------------------------------------------------------------------
# The forecast
# ---------------------------------------------------------------------------
@dataclass
class RaceForecast:
    driver_ids: list[str]
    matrix: np.ndarray  # [driver, position]; rows and columns sum to 1
    table: pd.DataFrame  # driver_id, p_win, p_podium, p_top5, p_top10, exp_position

    def column(self, name: str) -> np.ndarray:
        return self.table[name].to_numpy()


def forecast(
    inputs: SimInputs,
    temperature: float,
    blend_weight: float,
    n_sims: int = config.N_SIMULATIONS,
    grid_temperature: float | None = None,
    seed: int = config.RANDOM_SEED,
) -> RaceForecast:
    """One coherent finishing-position distribution for a race.

    Plackett-Luce at the calibrated temperature, mixed with the Monte Carlo at
    the fitted weight; see probability.py for why the mixture stays coherent.
    """
    pl = probability.pl_position_matrix(inputs.scores, temperature, n_samples=max(n_sims, 10_000), seed=seed)
    sim = simulate_matrix(
        inputs, n_sims, grid_temperature=grid_temperature if grid_temperature else temperature, seed=seed + 1
    )
    matrix = probability.mix(pl, sim, blend_weight)
    table = probability.summarise(matrix)
    table.insert(0, "driver_id", inputs.driver_ids)
    return RaceForecast(inputs.driver_ids, matrix, table)


def ranking_forecast(
    driver_ids: list[str],
    scores: np.ndarray,
    temperature: float,
    n_samples: int = 20_000,
    seed: int = config.RANDOM_SEED,
) -> RaceForecast:
    """Plackett-Luce alone, for qualifying: one lap has no retirements or safety
    cars worth simulating, so its distribution is the ranking's own."""
    matrix = probability.smooth(probability.pl_position_matrix(scores, temperature, n_samples, seed))
    table = probability.summarise(matrix)
    table.insert(0, "driver_id", driver_ids)
    return RaceForecast(driver_ids, matrix, table)
