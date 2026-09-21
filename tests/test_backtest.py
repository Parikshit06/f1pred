"""Scoring and the calibration table.

The calibration table is the one place the model's own confidence is graded,
so a race silently missing from it is worse than a wrong number: nothing about
the output says anything is absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1pred import backtest


def _races(p_top_pick: list[float], p_winner: list[float] | None = None) -> pd.DataFrame:
    """One row per graded race, in the shape walk_forward emits."""
    p_winner = p_winner if p_winner is not None else p_top_pick
    return pd.DataFrame(
        {
            "method": "model",
            "season": 2025,
            "round": range(1, len(p_top_pick) + 1),
            "p_top_pick": p_top_pick,
            "p_winner": p_winner,
            "top1_hit": [1] * len(p_top_pick),
        }
    )


def test_calibration_keeps_every_race():
    """The bins are cut on p_top_pick, so they have to reach the top of it.

    They were built from p_winner - the probability the model gave whoever
    actually won - which is the smaller of the two on every race the favourite
    lost. Any race more confident than the best-covered winner fell past the
    last edge, came back NaN from pd.cut, and was dropped by the groupby. The
    published table said 62 races and summed to 61.
    """
    # The confident race was wrong about the winner, so p_winner stays low.
    races = _races(p_top_pick=[0.20, 0.35, 0.50, 0.94], p_winner=[0.20, 0.35, 0.50, 0.02])
    cal = backtest.BacktestResult(races).calibration()
    assert cal["n"].sum() == len(races)


def test_calibration_buckets_span_the_stated_probabilities():
    races = _races([0.05, 0.25, 0.45, 0.65, 0.85, 0.99])
    cal = backtest.BacktestResult(races).calibration()
    assert cal["n"].sum() == len(races)
    # Every bucket that survives must hold at least one race.
    assert (cal["n"] > 0).all()


def test_calibration_of_an_empty_result_is_empty():
    assert backtest.BacktestResult(pd.DataFrame()).calibration().empty


def test_ndcg_rewards_putting_the_real_winner_first():
    truth = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    perfect = backtest.ndcg_at_k(["a", "b", "c", "d", "e"], truth)
    swapped = backtest.ndcg_at_k(["b", "a", "c", "d", "e"], truth)
    assert perfect == pytest.approx(1.0)
    assert swapped < perfect


def test_baselines_produce_a_distribution():
    race = pd.DataFrame(
        {
            "driver_id": list("abcde"),
            "grid": [1, 2, 3, 4, 5],
            "champ_position_before": [5, 4, 3, 2, 1],
            "drv_avg_finish_3": [2.0, 3.0, 1.0, 5.0, 4.0],
            "team_avg_finish_3": [1.0, 1.0, 4.0, 4.0, 9.0],
        }
    )
    for kind in ("grid", "championship", "recent_form", "team_form"):
        b = backtest.baseline_predictions(race, kind)
        assert b["p_win"].sum() == pytest.approx(1.0)
        assert np.all(np.diff(b.sort_values("pred_rank")["p_win"]) <= 0)
