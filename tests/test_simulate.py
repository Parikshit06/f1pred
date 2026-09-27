from __future__ import annotations

import numpy as np
import pytest

from f1pred import probability, simulate


def test_plackett_luce_is_a_distribution():
    p = probability.plackett_luce(np.array([3.0, 1.0, 0.5, -2.0]))
    assert p.sum() == pytest.approx(1.0)
    assert (p > 0).all()
    assert p[0] == p.max(), "highest score must get the highest probability"


def test_temperature_controls_sharpness():
    scores = np.array([2.0, 1.0, 0.0])
    sharp = probability.plackett_luce(scores, temperature=0.3)
    flat = probability.plackett_luce(scores, temperature=3.0)
    assert sharp[0] > flat[0]
    assert flat.std() < sharp.std()


def test_plackett_luce_handles_large_scores_without_overflow():
    p = probability.plackett_luce(np.array([900.0, 800.0, 100.0]))
    assert np.isfinite(p).all()
    assert p.sum() == pytest.approx(1.0)


def test_fit_temperature_prefers_sharp_when_favourite_always_wins():
    groups = [np.array([3.0, 1.0, 0.0]) for _ in range(30)]
    winners = [0] * 30
    assert probability.fit_temperature(groups, winners) < 1.0


def test_fit_temperature_prefers_flat_when_outcomes_are_random():
    rng = np.random.default_rng(1)
    groups = [np.array([3.0, 1.0, 0.0]) for _ in range(60)]
    winners = [int(rng.integers(0, 3)) for _ in range(60)]
    assert probability.fit_temperature(groups, winners) > 1.0


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


def test_mix_returns_a_distribution():
    a = np.array([0.6, 0.3, 0.1])
    b = np.array([0.2, 0.4, 0.4])
    for w in (0.0, 0.35, 1.0):
        out = probability.mix(a, b, w)
        assert out.sum() == pytest.approx(1.0)
        assert (out > 0).all()


# ---------------------------------------------------------------------------
# One coherent distribution
# ---------------------------------------------------------------------------
def _forecast(n=8, grid=True, **kw):
    inputs = _inputs(n, **({} if grid else {"grid": None, "grid_scores": np.linspace(2.0, -2.0, n)}), **kw)
    return simulate.forecast(inputs, temperature=0.5, blend_weight=0.6, n_sims=2000, seed=3)


@pytest.mark.parametrize("grid", [True, False])
def test_the_forecast_is_one_coherent_distribution(grid):
    fc = _forecast(grid=grid)
    assert probability.check_distribution(fc.matrix) == []
    t = fc.table
    assert t["p_win"].sum() == pytest.approx(1.0)
    assert t["p_podium"].sum() == pytest.approx(3.0)
    assert t["p_top5"].sum() == pytest.approx(5.0)
    assert t["p_top10"].sum() == pytest.approx(min(10, len(t)))
    assert t["exp_position"].mean() == pytest.approx((len(t) + 1) / 2)


def test_every_band_contains_the_one_inside_it():
    """Winning is a podium, a podium is a top five: true for every driver by
    construction, not by flooring after the fact."""
    t = _forecast().table
    assert (t["p_win"] <= t["p_podium"] + 1e-12).all()
    assert (t["p_podium"] <= t["p_top5"] + 1e-12).all()
    assert (t["p_top5"] <= t["p_top10"] + 1e-12).all()


def test_no_driver_is_ever_given_exactly_zero():
    """A Monte Carlo count of zero is infinitely wrong the one time it happens."""
    t = _forecast(scores=np.array([9.0, 8.0, -9.0, -9.5, -10.0, -10.5, -11.0, -12.0])).table
    assert (t["p_win"] > 0).all()
    assert np.isfinite(t.drop(columns="driver_id").to_numpy()).all()


def test_derived_numbers_agree_with_the_matrix():
    fc = _forecast()
    m = fc.matrix
    assert np.allclose(fc.table["p_win"], m[:, 0])
    assert np.allclose(fc.table["p_podium"], m[:, :3].sum(axis=1))
    assert np.allclose(fc.table["exp_position"], m @ np.arange(1, m.shape[1] + 1))


def test_the_forecast_is_reproducible():
    a, b = _forecast(), _forecast()
    assert np.array_equal(a.matrix, b.matrix)


def test_gumbel_sampling_matches_plackett_luce():
    """Sorting Gumbel-perturbed scores draws a Plackett-Luce ranking exactly."""
    scores = np.array([1.5, 0.7, 0.0, -0.4])
    m = probability.pl_position_matrix(scores, temperature=0.8, n_samples=200_000, seed=1)
    assert np.allclose(m[:, 0], probability.plackett_luce(scores, 0.8), atol=0.004)


def test_check_distribution_catches_a_broken_matrix():
    bad = np.full((3, 3), 1 / 3)
    bad[0, 0] = 0.9
    assert probability.check_distribution(bad)
    assert probability.check_distribution(np.full((2, 2), np.nan))


def test_temperature_is_refitted_only_from_enough_history():
    groups = [np.array([3.0, 1.0, 0.0])] * 5
    assert probability.rolling_temperature(groups, [0] * 5, fallback=0.77) == 0.77
    many = [np.array([3.0, 1.0, 0.0])] * 30
    assert probability.rolling_temperature(many, [0] * 30, fallback=0.77) < 1.0


def test_dnf_probability_is_bounded():
    import pandas as pd

    df = pd.DataFrame({"drv_dnf_rate_10": [0.0, 1.0, np.nan], "team_dnf_rate_10": [0.0, 1.0, np.nan]})
    p = simulate.dnf_probability(df)
    assert (p >= 0.02).all() and (p <= 0.35).all()
