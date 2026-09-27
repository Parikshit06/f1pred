"""The walk-forward and what it is graded with.

The walk-forward is where leakage would hide: a model allowed to train on the
race it forecasts looks brilliant in the backtest and is useless live. These
tests pin the training cut-off, the separation of the tuning and reporting
windows, and the scoring arithmetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1pred import backtest, baselines, features, metrics, model, simulate


def _frame(n_races: int = 40, n_drivers: int = 6, seed: int = 0) -> pd.DataFrame:
    """A made-up history with every column the race and quali models need."""
    rng = np.random.default_rng(seed)
    rows = []
    for race in range(n_races):
        order = rng.permutation(n_drivers)
        for pos, d in enumerate(order, start=1):
            rows.append(
                {
                    "race_seq": race,
                    "season": 2020 + race // 10,
                    "round": race % 10 + 1,
                    "driver_id": f"d{d}",
                    "constructor_id": f"t{d // 2}",
                    "position": float(pos),
                    "quali_position": float(pos),
                    "grid": float(pos),
                }
            )
    df = pd.DataFrame(rows)
    for col in set(features.RACE_FEATURES + features.QUALI_FEATURES) - set(df.columns):
        df[col] = rng.normal(size=len(df))
    df["race_relevance"] = 24 - df["position"]
    df["quali_relevance"] = 24 - df["quali_position"]
    return df


# ---------------------------------------------------------------------------
# The training cut-off
# ---------------------------------------------------------------------------
def test_no_model_ever_trains_on_the_race_it_forecasts_or_later():
    df = _frame()
    seen: list[tuple[int, int]] = []

    def spy(history: pd.DataFrame):
        seen.append((int(history["race_seq"].max()), len(seen)))
        return model.train(history, ["drv_avg_finish_3"], "race_relevance", n_seeds=1)

    targets = list(range(32, 40))
    out = backtest.oos_scores(df, targets, spy, retrain_every=1)
    assert sorted(out) == targets
    for (last_trained, _), target in zip(seen, targets):
        assert last_trained == target - 1, (
            f"race {target} was forecast by a model that saw race {last_trained}"
        )


def test_a_reused_model_is_only_ever_older_than_the_race():
    df = _frame()
    trained_through = []

    def spy(history):
        trained_through.append(int(history["race_seq"].max()))
        return model.train(history, ["drv_avg_finish_3"], "race_relevance", n_seeds=1)

    backtest.oos_scores(df, list(range(32, 40)), spy, retrain_every=3)
    # Retrained before 32, 35 and 38 - each time on everything before that race.
    assert trained_through == [31, 34, 37]


def test_the_tuning_window_stops_before_the_reported_one():
    df = _frame()
    tuned = backtest.completed_races(df, 2021, 2022)
    reported = backtest.completed_races(df, 2023)
    assert set(tuned["race_seq"]).isdisjoint(set(reported["race_seq"]))
    assert tuned["season"].max() <= 2022


def test_a_pre_qualifying_forecast_cannot_see_the_qualifying_result():
    """Every grid-stage feature is replaced by its projection."""
    race = _frame().query("race_seq == 5").reset_index(drop=True)
    projected = backtest.projected_weekend(race, quali_scores=np.arange(len(race), 0, -1.0))
    for col in features.GRID_FEATURES:
        assert col in projected
    assert projected["grid"].tolist() == list(range(1, len(race) + 1))
    assert (projected["quali_gap_to_pole_pct"] == race["drv_pace_gap_pct"]).all()
    assert (projected["quali_position"] == projected["grid"]).all()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def test_ndcg_rewards_putting_the_real_winner_first():
    truth = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    perfect = metrics.ndcg_at_k(["a", "b", "c", "d", "e"], truth)
    swapped = metrics.ndcg_at_k(["b", "a", "c", "d", "e"], truth)
    assert perfect == pytest.approx(1.0)
    assert swapped < perfect


def test_ranking_metrics_on_a_perfect_call():
    actual = pd.Series([1, 2, 3, 4, 5, 6], index=list("abcdef"), dtype=float)
    out = metrics.ranking(actual, actual)
    assert out["winner_hit"] == 1 and out["podium_overlap"] == 3 and out["top5_overlap"] == 5
    assert out["ndcg3"] == pytest.approx(1.0) and out["spearman"] == pytest.approx(1.0)


def test_probability_metrics_reward_confidence_in_the_right_driver():
    actual = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"])
    sure = pd.DataFrame({"p_win": [0.9, 0.05, 0.05], "p_podium": 1.0, "p_top10": 1.0}, index=actual.index)
    wrong = pd.DataFrame({"p_win": [0.05, 0.05, 0.9], "p_podium": 1.0, "p_top10": 1.0}, index=actual.index)
    assert (
        metrics.probability(sure, actual)["win_logloss"] < metrics.probability(wrong, actual)["win_logloss"]
    )
    assert metrics.probability(sure, actual)["win_brier"] < metrics.probability(wrong, actual)["win_brier"]


def test_reliability_keeps_every_prediction():
    """A probability falling outside the bucket edges used to vanish silently."""
    p = pd.Series([0.0, 0.01, 0.3, 0.99, 1.0])
    y = pd.Series([0, 0, 1, 1, 1])
    assert metrics.reliability(p, y)["n"].sum() == len(p)


def test_calibration_error_is_zero_when_stated_equals_observed():
    p = pd.Series([0.25] * 400)
    y = pd.Series(([1] + [0] * 3) * 100)
    assert metrics.expected_calibration_error(p, y) == pytest.approx(0.0)


def test_paired_comparison_only_uses_races_both_were_graded_on():
    races = pd.DataFrame(
        {
            "method": ["model"] * 3 + ["grid"] * 2,
            "season": [2025] * 5,
            "round": [1, 2, 3, 1, 2],
            "win_logloss": [0.5, 0.5, 9.0, 1.0, 1.0],
        }
    )
    out = metrics.paired_comparison(races, "model", "grid", ["win_logloss"], n_boot=500)
    row = out.iloc[0]
    assert row["n_races"] == 2
    assert row["difference"] == pytest.approx(-0.5)
    assert row["better"] == "model"


def test_an_interval_straddling_zero_is_called_unclear():
    rng = np.random.default_rng(0)
    races = pd.DataFrame(
        {
            "method": ["model"] * 30 + ["grid"] * 30,
            "season": 2025,
            "round": list(range(30)) * 2,
            "ndcg5": np.r_[rng.normal(0.8, 0.1, 30), rng.normal(0.8, 0.1, 30)],
        }
    )
    assert metrics.paired_comparison(races, "model", "grid", ["ndcg5"])["better"].iloc[0] == "unclear"


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
RACE = pd.DataFrame(
    {
        "driver_id": list("abcde"),
        "grid": [1, 2, 3, 4, np.nan],
        "champ_position_before": [5, 4, 3, 2, 1],
        "drv_avg_finish_3": [2.0, 3.0, 1.0, 5.0, 4.0],
        "team_avg_finish_3": [1.0, 1.0, 4.0, 4.0, 9.0],
    }
)


@pytest.mark.parametrize("kind", baselines.KINDS)
def test_baselines_produce_a_coherent_distribution(kind):
    fc = simulate.ranking_forecast(RACE["driver_id"].tolist(), baselines.scores(RACE, kind), temperature=0.5)
    assert fc.table["p_win"].sum() == pytest.approx(1.0)
    order = baselines.order(RACE, kind).sort_values()
    p = fc.table.set_index("driver_id").loc[order.index, "p_win"].to_numpy()
    assert np.all(np.diff(p) <= 0.01), "a baseline's favourite must be the likeliest winner"


def test_a_car_without_a_grid_slot_starts_at_the_back_in_the_grid_baseline():
    assert baselines.order(RACE, "grid")["e"] == 5


def test_tied_baseline_values_get_equal_chances():
    fc = simulate.ranking_forecast(
        RACE["driver_id"].tolist(), baselines.scores(RACE, "team_form"), temperature=0.5, n_samples=100_000
    )
    t = fc.table.set_index("driver_id")
    assert t.loc["a", "p_win"] == pytest.approx(t.loc["b", "p_win"], abs=0.01)
