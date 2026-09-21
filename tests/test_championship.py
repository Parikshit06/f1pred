"""Checks on the title projection.

The projection compounds nine races of noise, so an error here does not look
like an error - it looks like a plausible table. These pin the properties that
must hold whatever the numbers come out as.
"""

from __future__ import annotations

import numpy as np
import pytest

from f1pred import championship as champ


def _sim(scores, dnf=None, base=None, races=5, sims=2000):
    scores = np.asarray(scores, dtype=float)
    dnf = np.zeros(len(scores)) if dnf is None else np.asarray(dnf, dtype=float)
    base = np.zeros(len(scores)) if base is None else np.asarray(base, dtype=float)
    out = champ.simulate_seasons(scores, dnf, base, races, n_sims=sims, safety_car_prob=0.0)
    return out[0], out[1]


def test_a_faster_car_scores_more():
    totals, _ = _sim([3.0, 0.0, -3.0])
    mean = totals.mean(axis=0)
    assert mean[0] > mean[1] > mean[2]


def test_points_already_scored_are_carried_forward():
    totals, _ = _sim([0.0, 0.0], base=[100.0, 0.0], races=1)
    assert totals.mean(axis=0)[0] - totals.mean(axis=0)[1] == pytest.approx(100.0, abs=3)


def test_a_car_that_never_finishes_never_scores():
    totals, _ = _sim([5.0, 0.0, 0.0], dnf=[1.0, 0.0, 0.0])
    assert totals[:, 0].max() == 0.0, "a retirement scored points"


def test_no_season_awards_more_than_the_points_on_offer():
    races = 4
    totals, _ = _sim([1.0] * 12, races=races)
    assert totals.sum(axis=1).max() <= champ.POINTS.sum() * races + 1e-9


def test_every_race_awards_the_full_points_when_everyone_finishes():
    races = 3
    totals, _ = _sim([0.5] * 15, races=races, sims=200)
    assert totals.sum(axis=1) == pytest.approx(champ.POINTS.sum() * races)


def test_the_progression_track_ends_where_the_totals_do():
    """The chart and the table are drawn from the same simulation, so the last
    point on the line must be the projected final score."""
    totals, track = _sim([2.0, 1.0, 0.0], races=6)
    assert track[-1] == pytest.approx(totals.mean(axis=0), abs=1e-9)


def test_the_progression_track_only_goes_up():
    _, track = _sim([2.0, 1.0, 0.0], races=6)
    assert (np.diff(track, axis=0) >= -1e-9).all(), "points went backwards"


def test_a_big_enough_lead_cannot_be_overturned():
    """Sanity on the title probability: with one race left and more points in
    hand than are on offer, the leader is champion in every simulation."""
    totals, _ = _sim([0.0, 3.0], base=[100.0, 0.0], races=1)
    assert (totals[:, 0] > totals[:, 1]).all()


def test_unreliability_costs_the_title():
    """Same pace, one car breaks a third of the time."""
    totals, _ = _sim([2.0, 2.0], dnf=[0.33, 0.0], races=9, sims=4000)
    wins = (totals[:, 0] > totals[:, 1]).mean()
    assert wins < 0.35, f"a car retiring a third of the time won {wins:.0%} of titles"


# ---------------------------------------------------------------------------
# Printing the odds
# ---------------------------------------------------------------------------
def test_odds_always_sum_to_one_hundred():
    """A title column that reads 99.9% across the whole field looks like a leak."""
    for probs in ([0.9999, 0.0001, 0.0, 0.0], [0.4, 0.35, 0.25], [1.0, 0.0], [0.33, 0.33, 0.34]):
        out = champ.allocate_odds(np.array(probs))
        assert out.sum() == pytest.approx(1.0, abs=1e-9), probs


def test_an_unclinched_title_never_prints_as_certain():
    out = champ.allocate_odds(np.array([1.0, 0.0, 0.0]))
    assert out[0] == pytest.approx(0.999)
    assert out[1] == pytest.approx(0.001), "the spare tenth went nowhere"


def test_a_clinched_title_does_print_as_certain():
    """Once nobody else can reach the leader, 100% is the honest number."""
    out = champ.allocate_odds(np.array([1.0, 0.0]), np.array([True, False]))
    assert out[0] == pytest.approx(1.0)


def test_allocation_keeps_the_ordering():
    out = champ.allocate_odds(np.array([0.5, 0.3, 0.2]))
    assert list(out) == sorted(out, reverse=True)


def test_clinch_needs_more_races_when_the_lead_is_smaller():
    assert champ.clinch_round(292, 211, 9) == 3
    assert champ.clinch_round(292, 280, 9) > champ.clinch_round(292, 211, 9)


def test_clinch_with_one_race_left_is_that_race():
    """Level on points into the last round: it is settled there and nowhere
    earlier, so the answer is the finale rather than None."""
    assert champ.clinch_round(100, 100, 1) == 1


def test_clinch_is_none_when_the_season_is_over():
    assert champ.clinch_round(100, 100, 0) is None


def test_clinch_is_immediate_when_the_lead_already_exceeds_what_is_left():
    assert champ.clinch_round(400, 100, 2) == 1


def test_the_projection_carries_a_range_not_just_a_mean():
    """A single line reads as certainty. The 10-90 band is what shows that a
    leader retiring twice is an ordinary outcome, not an upset."""
    totals, track, lo, hi, wins = champ.simulate_seasons(
        np.array([2.0, 1.0, 0.0]), np.array([0.15, 0.15, 0.15]), np.zeros(3), 9, n_sims=3000
    )
    assert (lo[-1] < track[-1]).all(), "the low band is not below the mean"
    assert (hi[-1] > track[-1]).all(), "the high band is not above the mean"
    assert (hi[-1] - lo[-1] > 0).all()


def test_expected_wins_cannot_exceed_the_races_left():
    _, _, _, _, wins = champ.simulate_seasons(np.array([9.0, 0.0]), np.zeros(2), np.zeros(2), 6, n_sims=800)
    assert (wins <= 6 + 1e-9).all()
    assert wins.sum() == pytest.approx(6.0), "six races must award exactly six wins"


def test_retirements_cost_the_leader_wins():
    """The question this answers: does the projection actually let the
    favourite break down?"""
    _, _, _, _, clean = champ.simulate_seasons(np.array([3.0, 0.0]), np.zeros(2), np.zeros(2), 9, n_sims=3000)
    _, _, _, _, fragile = champ.simulate_seasons(
        np.array([3.0, 0.0]), np.array([0.25, 0.0]), np.zeros(2), 9, n_sims=3000
    )
    assert fragile[0] < clean[0] - 0.1, "a 25% retirement rate did not cost any wins"


def test_the_spare_tenth_goes_to_the_nearest_rival_not_a_random_name():
    """With everyone else on a simulated zero, ordering by probability picks an
    arbitrary index - which once put 0.1% against the ninth-placed team."""
    probs = np.array([1.0, 0.0, 0.0, 0.0])
    points = np.array([800.0, 540.0, 500.0, 110.0])
    out = champ.allocate_odds(probs, rank_by=points)
    assert out[1] == pytest.approx(0.001), "the tenth did not go to second on points"
    assert out[3] == 0.0
