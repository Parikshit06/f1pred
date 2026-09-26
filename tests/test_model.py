"""The ranker: grouping, reproducibility, and explanations that add up."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1pred import model

FEATS = ["skill", "car", "noise"]


def _history(n_races: int = 50, n_drivers: int = 8, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    skill = np.linspace(2, -2, n_drivers)
    rows = []
    for race in range(n_races):
        car = rng.normal(0, 0.5, n_drivers)
        noise = rng.normal(0, 1, n_drivers)
        finish = np.argsort(np.argsort(-(skill + car + 0.5 * noise))) + 1
        for d in range(n_drivers):
            rows.append(
                {
                    "race_seq": race,
                    "season": 2024 + race // 24,
                    "round": race % 24 + 1,
                    "driver_id": f"d{d}",
                    "skill": skill[d] + rng.normal(0, 0.1),
                    "car": car[d],
                    "noise": noise[d],
                    "race_relevance": 24.0 - finish[d],
                }
            )
    # Shuffled on purpose: training must group by race whatever the input order.
    return pd.DataFrame(rows).sample(frac=1.0, random_state=0).reset_index(drop=True)


@pytest.fixture(scope="module")
def ranker():
    return model.train(_history(), FEATS, "race_relevance", n_seeds=3)


def test_training_rows_are_grouped_by_race(monkeypatch):
    """XGBRanker needs each race's rows contiguous and qids sorted."""
    captured = {}
    real_fit = model.xgb.XGBRanker.fit

    def spy(self, X, y, qid=None, **kw):
        captured["qid"] = np.asarray(qid)
        return real_fit(self, X, y, qid=qid, **kw)

    monkeypatch.setattr(model.xgb.XGBRanker, "fit", spy)
    model.train(_history(), FEATS, "race_relevance", n_seeds=1)
    qid = captured["qid"]
    assert np.all(np.diff(qid) >= 0), "qid must be sorted so each race is one contiguous group"
    assert len(np.unique(qid)) == 50


def test_one_weight_per_race_not_per_row():
    h = _history().drop_duplicates("race_seq")
    w = model.recency_weights(h, 2.0)
    assert len(w) == 50 and (w > 0).all() and w.max() <= 1.0


def test_scores_are_reproducible():
    a = model.train(_history(), FEATS, "race_relevance", n_seeds=2)
    b = model.train(_history(), FEATS, "race_relevance", n_seeds=2)
    race = _history().query("race_seq == 7")
    assert np.array_equal(a.score(race), b.score(race))


def test_scores_are_centred_within_the_race(ranker):
    race = _history().query("race_seq == 7")
    assert ranker.score(race).mean() == pytest.approx(0.0, abs=1e-6)  # float32 predictions


def test_the_model_learns_the_signal(ranker):
    race = _history().query("race_seq == 7").sort_values("skill", ascending=False)
    s = ranker.score(race)
    assert s[0] > s[-1]


def test_contributions_add_up_to_the_published_score(ranker):
    """The explanation is of the ensemble forecast, not of one seed."""
    race = _history().query("race_seq == 11")
    phi = ranker.contributions(race)
    assert phi.shape == (len(race), len(FEATS))
    assert np.allclose(phi.sum(axis=1), ranker.score(race), atol=1e-4)


def test_contributions_credit_the_feature_that_matters(ranker):
    race = _history().query("race_seq == 11")
    phi = np.abs(ranker.contributions(race)).mean(axis=0)
    assert FEATS[int(np.argmax(phi))] == "skill"


def test_importance_is_averaged_over_seeds(ranker):
    imp = model.feature_importance(ranker)
    assert set(imp["feature"]) <= set(FEATS)
    assert imp["share_pct"].sum() == pytest.approx(100.0, abs=0.5)


def test_missing_feature_columns_are_refused(ranker):
    with pytest.raises(KeyError):
        ranker.score(_history().drop(columns="car"))
