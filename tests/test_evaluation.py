"""The experiment definitions: they must stay in step with the model they test."""

from __future__ import annotations

from f1pred import evaluation, features


def test_every_race_feature_belongs_to_exactly_one_ablation_group():
    """A feature added to the model but not to a group would never be ablated."""
    grouped = [f for cols in evaluation.GROUPS.values() for f in cols] + ["grid"]
    assert sorted(grouped) == sorted(set(grouped)), "a feature sits in two groups"
    assert set(grouped) == set(features.RACE_FEATURES)


def test_the_ablation_includes_the_grid_alone_and_the_full_model():
    sets = evaluation.feature_sets()
    assert sets["grid only"] == ["grid"]
    assert sets["full model"] == list(features.RACE_FEATURES)
    for name, cols in evaluation.GROUPS.items():
        assert set(sets[f"full - {name}"]) == set(features.RACE_FEATURES) - set(cols)
    for name, cols in evaluation.CANDIDATES.items():
        assert not set(cols) & set(features.RACE_FEATURES), f"candidate {name} is already in the model"
        assert set(sets[f"full + {name} (not in model)"]) == set(features.RACE_FEATURES) | set(cols)


def test_practice_never_reaches_the_race_model_by_default():
    """Practice is a pre-qualifying input: it feeds the qualifying model, and
    reaches the race only through the projected grid."""
    assert not set(features.PRACTICE_FEATURES) & set(features.RACE_FEATURES)


def test_practice_feeds_the_qualifying_model_only():
    """Held out (design picked on 2024, graded on 2025-26), current-weekend
    practice improved the qualifying forecast; it added nothing to the race
    beyond the official grid, so the race model never sees it."""
    practice = set(features.PRACTICE_FEATURES) | set(features.PRACTICE_DETAIL_FEATURES)
    assert practice <= set(features.QUALI_FEATURES)
    assert not practice & set(features.RACE_FEATURES)
    assert "practice" in evaluation.EXPERIMENTS


def test_a_difference_is_only_clear_when_its_interval_excludes_zero():
    """The verdict behind every kept or rejected experiment, including which
    way is better for scores where lower wins."""
    assert evaluation._clear({"ndcg5_ci": [0.01, 0.05]}, "ndcg5") == "better"
    assert evaluation._clear({"ndcg5_ci": [-0.03, -0.002]}, "ndcg5") == "worse"
    assert evaluation._clear({"position_error_ci": [-0.3, -0.1]}, "position_error") == "better"
    assert evaluation._clear({"pole_logloss_ci": [-0.01, 0.19]}, "pole_logloss") == "unclear"
