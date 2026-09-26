"""Invariants on a published forecast.

The backtest measures whether forecasts are good. These check that they're
possible at all - the backtest only reads the winner column, so an
internally contradictory board would pass it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from f1pred import predict
from f1pred.store import database_exists


def test_a_simulator_column_assigned_as_a_series_lands_as_nan():
    """Pins a silent pandas trap: a features slice keeps its original index while
    simulator frames are 0..n-1, so assigning a Series fills the column with NaN.
    predict.run resets the index and assigns arrays; this keeps it honest.
    """
    race = pd.DataFrame({"driver_id": ["a", "b", "c"]}, index=[3766, 3767, 3768])
    sim = pd.DataFrame({"exp_position": [2.0, 1.0, 3.0]})

    race["as_series"] = sim["exp_position"].rank(method="first")
    race["as_array"] = sim["exp_position"].rank(method="first").to_numpy()

    assert race["as_series"].isna().all(), "pandas started aligning these - revisit predict.run"
    assert not race["as_array"].isna().any()
    assert race["as_array"].tolist() == [2.0, 1.0, 3.0]


@pytest.fixture(scope="module")
def forecast():
    if not database_exists():
        pytest.skip("no database; run the pipeline first")
    return predict.run(n_sims=800)


def test_the_projected_grid_reaches_the_race_model(forecast):
    """Before qualifying the grid is the quali model's expected order, and it is
    the race model's strongest single feature. If it arrives empty the model is
    flying blind and the page says "-" in the Start column for every driver."""
    if forecast.grid_known:
        pytest.skip("grid is real, not projected")
    missing = [r["short"] for r in forecast.race_board if r["grid"] is None]
    assert not missing, f"no projected grid for {missing}"


def test_published_probabilities_cannot_contradict_each_other(forecast):
    """Winning is a podium, a podium is a top five, a top five scores. Any row
    that breaks that chain is impossible however good the model is - and it is
    the first thing a reader notices, because they can check it by eye."""
    for r in forecast.race_board + forecast.quali_board:
        chain = [r["p_win"], r["p_podium"], r["p_top5"], r["p_top10"]]
        assert chain == sorted(chain), f"{r['short']}: {chain}"


def test_the_field_is_not_forecast_to_finish_in_a_heap(forecast):
    """A model with no usable features gives everyone roughly the same chance.
    The favourite should be clearly ahead of the tenth-placed driver; when the
    projected grid went missing this gap collapsed and midfielders came out
    with a quarter chance of a podium."""
    board = forecast.race_board
    assert board[0]["p_win"] > 3 * board[min(9, len(board) - 1)]["p_win"]


def test_a_grid_with_holes_is_refused_rather_than_averaged_into_noise():
    """Before a race is run, grid is empty in the features table. Simulated as-is,
    pace went NaN and the output averaged into a near-uniform field that still
    looked like probabilities.
    """
    import numpy as np

    from f1pred import simulate

    n = 4
    args = {
        "driver_ids": [f"d{i}" for i in range(n)],
        "scores": np.linspace(2.0, -2.0, n),
        "dnf_prob": np.full(n, 0.05),
        "grid_scores": None,
        "overtaking_score": 3.0,
        "safety_car_prob": 0.3,
    }
    with pytest.raises(ValueError, match="missing"):
        simulate.simulate(
            simulate.SimInputs(grid=np.array([1.0, np.nan, 3.0, 4.0]), **args),
            n_sims=50,
            temperature=0.4,
        )

    ok = simulate.simulate(
        simulate.SimInputs(grid=np.arange(1.0, n + 1), **args), n_sims=2000, temperature=0.4
    )
    assert ok["p_win"].iloc[0] > ok["p_win"].iloc[-1], "pace order should survive a valid grid"


def test_the_starting_grid_is_taken_from_qualifying_before_the_race_is_run(forecast):
    """Once qualifying is in, every driver has a start position - it is the
    qualifying classification. An empty Start column on the page means the
    simulator was handed nothing."""
    if not forecast.grid_known:
        pytest.skip("qualifying not in yet")
    missing = [r["short"] for r in forecast.race_board if r["grid"] is None]
    assert not missing, f"grid known but no starting position for {missing}"
