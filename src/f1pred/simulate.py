"""Turning ranking scores into probabilities.

Two routes, deliberately different, then blended:

  Plackett-Luce   closed form. Treats each driver's score as a strength and
                  reads off the probability of finishing first. Fast, smooth,
                  and blind to anything the ranker did not see.

  Monte Carlo     simulates the race. Adds the things a ranker structurally
                  cannot express: pace varies race to race, cars break, safety
                  cars compress the field, and before qualifying the grid
                  itself is unknown.

The blend weight is not a taste decision - backtest.tune_blend() picks it by
minimising log loss on held-out races.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# Plackett-Luce
# ---------------------------------------------------------------------------
def plackett_luce(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """P(each entry finishes first), from ranking scores.

    Temperature controls how decisive the model is. Below 1 sharpens toward the
    favourite, above 1 flattens. Fitted on validation data, because a ranker's
    raw score scale carries no probabilistic meaning on its own.
    """
    z = np.asarray(scores, dtype=float) / max(temperature, 1e-6)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


ADAPTIVE_WINDOW = 24  # roughly one season of races
MIN_CALIBRATION_RACES = 10


def fit_temperature(
    score_groups: list[np.ndarray], winner_idx: list[int], grid: np.ndarray | None = None
) -> float:
    """Choose the temperature that minimises log loss on observed winners."""
    if grid is None:
        grid = np.concatenate([np.linspace(0.05, 2.0, 40), np.linspace(2.2, 8.0, 30)])

    best_t, best_loss = 1.0, np.inf
    for t in grid:
        losses = []
        for scores, win in zip(score_groups, winner_idx):
            if win is None or win < 0 or win >= len(scores):
                continue
            p = plackett_luce(scores, t)
            losses.append(-np.log(max(p[win], 1e-12)))
        if losses:
            loss = float(np.mean(losses))
            if loss < best_loss:
                best_t, best_loss = float(t), loss
    return best_t


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
@dataclass
class SimInputs:
    driver_ids: list[str]
    scores: np.ndarray  # higher = better
    dnf_prob: np.ndarray  # per driver, 0..1
    grid: np.ndarray | None = None  # known grid, or None before qualifying
    grid_scores: np.ndarray | None = None  # quali-model scores, used when grid is None
    overtaking_score: float = 3.0  # mean |grid - finish| at this circuit
    safety_car_prob: float = 0.35


def _softmax_sample_order(rng: np.random.Generator, scores: np.ndarray, temperature: float) -> np.ndarray:
    """Sample a full ordering via sequential Plackett-Luce draws."""
    remaining = list(range(len(scores)))
    order = []
    z = scores / max(temperature, 1e-6)
    while remaining:
        sub = z[remaining]
        sub = sub - sub.max()
        p = np.exp(sub)
        p /= p.sum()
        pick = rng.choice(len(remaining), p=p)
        order.append(remaining.pop(pick))
    return np.array(order)


def simulate(
    inputs: SimInputs,
    n_sims: int = config.N_SIMULATIONS,
    temperature: float = 1.0,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Run the race n_sims times. Returns per-driver outcome probabilities and
    the full position distribution the dashboard draws as a spread."""
    rng = np.random.default_rng(seed)
    n = len(inputs.driver_ids)
    positions = np.zeros((n, n), dtype=np.int32)  # [driver, finishing position]

    # How much a circuit lets pace express itself. At a track where nobody
    # overtakes, the grid dominates the result; at one where they do, race
    # pace matters more. This is the mechanism the reel's flat "track score"
    # feature was reaching for.
    ot = 3.0 if not np.isfinite(inputs.overtaking_score) else float(inputs.overtaking_score)
    grid_pull = float(np.clip(1.6 - ot / 8.0, 0.15, 1.4))

    score_sd = float(np.std(inputs.scores)) or 1.0

    # A grid with a hole in it does not raise, it degrades: the pace term goes
    # NaN, argsort returns an arbitrary order, and 10,000 runs of that average
    # into a near-uniform field. The output still looks like probabilities -
    # every driver on about the same chance - so nothing downstream notices.
    # This is the shape of the bug that published a 26% favourite with a 9%
    # podium, so it fails here instead.
    grid_given = None
    if inputs.grid is not None:
        grid_given = np.asarray(inputs.grid, dtype=float)
        if not np.isfinite(grid_given).all():
            missing = int((~np.isfinite(grid_given)).sum())
            raise ValueError(
                f"grid has {missing} missing entries of {n}. A race that has not run yet "
                "carries no grid in results - fill it from qualifying before simulating."
            )

    for i in range(n_sims):
        # Grid: known after qualifying, sampled from the quali model before it.
        if inputs.grid is not None:
            grid = grid_given
        elif inputs.grid_scores is not None:
            order = _softmax_sample_order(rng, inputs.grid_scores, temperature)
            grid = np.empty(n, dtype=float)
            grid[order] = np.arange(1, n + 1)
        else:
            grid = np.full(n, (n + 1) / 2.0)

        # A safety car compresses the field and hands out luck.
        chaos = 1.0 + (0.9 if rng.random() < inputs.safety_car_prob else 0.0)

        race_pace = (
            inputs.scores
            + rng.normal(0.0, 0.55 * score_sd * chaos, size=n)
            - grid_pull * (grid - 1) * (score_sd / 6.0)
        )

        retired = rng.random(n) < inputs.dnf_prob
        # Retirements are classified behind every finisher, ordered by how far
        # they got - which we proxy with a random draw.
        race_pace = np.where(retired, -1e6 + rng.random(n), race_pace)

        order = np.argsort(-race_pace)
        positions[order, np.arange(n)] += 1

    pos_prob = positions / n_sims
    return pd.DataFrame(
        {
            "driver_id": inputs.driver_ids,
            "p_win": pos_prob[:, 0],
            "p_podium": pos_prob[:, :3].sum(axis=1),
            "p_top5": pos_prob[:, :5].sum(axis=1),
            "p_points": pos_prob[:, :10].sum(axis=1),
            "exp_position": (pos_prob * np.arange(1, len(inputs.driver_ids) + 1)).sum(axis=1),
            "position_dist": list(pos_prob),
        }
    )


def rolling_temperature(
    score_groups: list[np.ndarray],
    winner_idx: list[int],
    fallback: float,
    window: int = ADAPTIVE_WINDOW,
) -> float:
    """Temperature fitted on the most recent races only.

    A single fixed temperature cannot be right across eras. On 2022-23, when
    one car was dominant, the sharpest setting tested (0.5) minimised log loss
    - being decisive paid. Carried into 2024-26, where the field is closer,
    that same setting is overconfident: races it called at 29% happened 10% of
    the time.

    So confidence is re-fitted from a trailing window instead of frozen. This
    uses finished races only, so it stays honest in a walk-forward, and it is
    what a live system would do anyway - the model should get less sure when
    the season stops being predictable.
    """
    if len(score_groups) < MIN_CALIBRATION_RACES:
        return fallback
    return fit_temperature(score_groups[-window:], winner_idx[-window:])


def blend(p_model: np.ndarray, p_sim: np.ndarray, weight: float) -> np.ndarray:
    """weight = how much to trust the closed-form model over the simulation."""
    out = weight * np.asarray(p_model) + (1.0 - weight) * np.asarray(p_sim)
    total = out.sum()
    return out / total if total > 0 else out


def dnf_probability(df: pd.DataFrame, floor: float = 0.02, cap: float = 0.35) -> np.ndarray:
    """Blend the driver's and the team's recent retirement rates.

    Team reliability dominates - an engine does not care who is steering - but
    driver rate carries crash risk, so both go in.
    """
    drv = df["drv_dnf_rate_10"].fillna(0.12).to_numpy()
    team = df["team_dnf_rate_10"].fillna(0.12).to_numpy()
    return np.clip(0.35 * drv + 0.65 * team, floor, cap)
