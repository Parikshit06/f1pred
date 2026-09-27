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


def test_the_left_out_groups_are_also_tested_before_qualifying():
    """A group can be harmless once the real grid is known and still count twice
    before it, when the grid is the qualifying model's projection of the same
    record. The pre-qualifying ablation is what caught that."""
    assert "pre_quali_ablation" in evaluation.EXPERIMENTS


def test_the_method_page_shows_the_pre_qualifying_ablation():
    from f1pred import method_page

    ex = {
        "pre_quali_ablation": {
            "tuning 2022-2023": [
                {"variant": "full model", "win_logloss": 1.8},
                {
                    "variant": "full + qualifying form (not in model)",
                    "win_logloss": 1.86,
                    "win_logloss_diff": 0.06,
                    "win_logloss_ci": [0.02, 0.10],
                },
            ]
        }
    }
    html = method_page._pre_quali_ablation(ex)
    assert "qualifying form" in html and "+0.020 to +0.100" in html and "worse" in html
    assert method_page._pre_quali_ablation({}) == ""
