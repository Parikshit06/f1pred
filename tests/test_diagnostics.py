"""The bias diagnostic has to survive its own correction being unnecessary."""

from __future__ import annotations

import pandas as pd

from f1pred import diagnostics


def _entries(errors_by_driver: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    for driver, errs in errors_by_driver.items():
        for i, e in enumerate(errs):
            rows.append(
                {
                    "season": 2024,
                    "round": i + 1,
                    "driver_id": driver,
                    "constructor_id": "team",
                    "predicted_rank": 5.0,
                    "actual_rank": 5.0 - e,
                    "error": e,
                }
            )
    return pd.DataFrame(rows)


def test_flags_a_genuinely_over_rated_driver():
    """One driver consistently placed better than they finish, everyone else
    unbiased, all at the same predicted rank so there is no trend to remove."""
    data = {f"d{i}": [0.0] * 20 for i in range(6)}
    data["flattered"] = [-3.0] * 20
    bias = diagnostics.driver_bias(_entries(data), min_races=10)
    assert bias.index[0] == "flattered"
    assert bias.loc["flattered", "raw_bias"] < -2


def test_removes_the_rank_trend_rather_than_reporting_it_as_driver_bias():
    """Construct pure artifact: bias is a clean linear function of predicted
    rank and nothing else. Driver-specific bias must come out near zero."""
    rows = []
    for i, driver in enumerate([f"d{i}" for i in range(10)]):
        predicted = float(i + 1)
        err = 0.32 * predicted - 3.49  # the trend measured on real data
        for r in range(20):
            rows.append(
                {
                    "season": 2024,
                    "round": r,
                    "driver_id": driver,
                    "constructor_id": "t",
                    "predicted_rank": predicted,
                    "actual_rank": predicted - err,
                    "error": err,
                }
            )
    entries = pd.DataFrame(rows)

    trend = diagnostics.trend_strength(entries, min_races=10)
    assert trend["variance_explained"] > 0.99, "a pure trend must be recognised as one"

    bias = diagnostics.driver_bias(entries, min_races=10)
    assert bias["raw_bias"].abs().max() > 1.0, "raw bias should look alarming"
    assert bias["driver_specific_bias"].abs().max() < 0.01, (
        "after correction nothing should remain - it was all positional"
    )


def test_real_driver_bias_survives_the_correction():
    """A trend plus one genuinely biased driver: the trend goes, the driver stays."""
    rows = []
    for i, driver in enumerate([f"d{i}" for i in range(10)]):
        predicted = float(i + 1)
        err = 0.32 * predicted - 3.49
        if driver == "d5":
            err -= 2.5
        for r in range(20):
            rows.append(
                {
                    "season": 2024,
                    "round": r,
                    "driver_id": driver,
                    "constructor_id": "t",
                    "predicted_rank": predicted,
                    "actual_rank": predicted - err,
                    "error": err,
                }
            )
    bias = diagnostics.driver_bias(pd.DataFrame(rows), min_races=10)
    assert bias.index[0] == "d5"
    assert bias.loc["d5", "driver_specific_bias"] < -1.5
    others = bias.drop(index="d5")["driver_specific_bias"].abs()
    assert others.max() < 0.75, "the rest should be close to unbiased"


def test_min_races_filter_excludes_thin_samples():
    data = {"regular": [0.0] * 30, "cameo": [-5.0] * 3}
    bias = diagnostics.driver_bias(_entries(data), min_races=15)
    assert "cameo" not in bias.index
    assert "regular" in bias.index
