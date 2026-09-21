from __future__ import annotations

import numpy as np
import pytest

from f1pred import simulate


def test_plackett_luce_is_a_distribution():
    p = simulate.plackett_luce(np.array([3.0, 1.0, 0.5, -2.0]))
    assert p.sum() == pytest.approx(1.0)
    assert (p > 0).all()
    assert p[0] == p.max(), "highest score must get the highest probability"


def test_temperature_controls_sharpness():
    scores = np.array([2.0, 1.0, 0.0])
    sharp = simulate.plackett_luce(scores, temperature=0.3)
    flat = simulate.plackett_luce(scores, temperature=3.0)
    assert sharp[0] > flat[0]
    assert flat.std() < sharp.std()


def test_plackett_luce_handles_large_scores_without_overflow():
    p = simulate.plackett_luce(np.array([900.0, 800.0, 100.0]))
    assert np.isfinite(p).all()
    assert p.sum() == pytest.approx(1.0)


def test_fit_temperature_prefers_sharp_when_favourite_always_wins():
    groups = [np.array([3.0, 1.0, 0.0]) for _ in range(30)]
    winners = [0] * 30
    assert simulate.fit_temperature(groups, winners) < 1.0


def test_fit_temperature_prefers_flat_when_outcomes_are_random():
    rng = np.random.default_rng(1)
    groups = [np.array([3.0, 1.0, 0.0]) for _ in range(60)]
    winners = [int(rng.integers(0, 3)) for _ in range(60)]
    assert simulate.fit_temperature(groups, winners) > 1.0


def _inputs(n=6, **kw):
    defaults = {
        "driver_ids": [f"d{i}" for i in range(n)],
        "scores": np.linspace(2.0, -2.0, n),
        "dnf_prob": np.full(n, 0.05),
        "grid": np.arange(1, n + 1, dtype=float),
    }
    defaults.update(kw)
    return simulate.SimInputs(**defaults)


def test_simulation_probabilities_are_coherent():
    out = simulate.simulate(_inputs(), n_sims=2000)
    assert out["p_win"].sum() == pytest.approx(1.0, abs=1e-6)
    assert (out["p_podium"] >= out["p_win"] - 1e-9).all()
    assert (out["p_top5"] >= out["p_podium"] - 1e-9).all()
    assert (out["p_top5"] <= 1.0 + 1e-9).all()


def test_position_distribution_rows_sum_to_one():
    out = simulate.simulate(_inputs(), n_sims=1000)
    for row in out["position_dist"]:
        assert float(np.sum(row)) == pytest.approx(1.0, abs=1e-6)


def test_stronger_car_wins_more_often():
    out = simulate.simulate(_inputs(), n_sims=3000)
    assert out["p_win"].iloc[0] > out["p_win"].iloc[-1]


def test_unreliable_car_wins_less_often():
    n = 4
    base = simulate.simulate(_inputs(n, dnf_prob=np.full(n, 0.01)), n_sims=3000)
    broken = np.full(n, 0.01)
    broken[0] = 0.35
    worse = simulate.simulate(_inputs(n, dnf_prob=broken), n_sims=3000)
    assert worse["p_win"].iloc[0] < base["p_win"].iloc[0]


def test_simulation_is_reproducible():
    a = simulate.simulate(_inputs(), n_sims=500, seed=7)
    b = simulate.simulate(_inputs(), n_sims=500, seed=7)
    assert (a["p_win"].to_numpy() == b["p_win"].to_numpy()).all()


def test_unknown_grid_is_sampled_not_assumed():
    """Without a grid the simulation must still produce a valid distribution,
    driven by the qualifying model's scores."""
    n = 6
    out = simulate.simulate(_inputs(n, grid=None, grid_scores=np.linspace(2.0, -2.0, n)), n_sims=2000)
    assert out["p_win"].sum() == pytest.approx(1.0, abs=1e-6)
    assert out["p_win"].iloc[0] > out["p_win"].iloc[-1]


def test_blend_returns_a_distribution():
    a = np.array([0.6, 0.3, 0.1])
    b = np.array([0.2, 0.4, 0.4])
    for w in (0.0, 0.35, 1.0):
        out = simulate.blend(a, b, w)
        assert out.sum() == pytest.approx(1.0)


def test_dnf_probability_is_bounded():
    import pandas as pd

    df = pd.DataFrame({"drv_dnf_rate_10": [0.0, 1.0, np.nan], "team_dnf_rate_10": [0.0, 1.0, np.nan]})
    p = simulate.dnf_probability(df)
    assert (p >= 0.02).all() and (p <= 0.35).all()
