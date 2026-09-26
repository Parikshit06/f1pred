"""Ranking models for qualifying and race outcomes.

Framed as learning-to-rank rather than classification. Predicting "who wins"
gives one positive example per race - about 170 since 2018. Ranking the field
turns each race into ~190 pairwise comparisons, which is the difference
between a model that learns and one that memorises the fastest car.

XGBRanker requires rows grouped by query, sorted by query id. Here a query is
one race, so qid = race_seq.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from . import config, features

log = logging.getLogger(__name__)

# rank:ndcg rather than pairwise: it optimises the metric that's reported, and
# beat pairwise on the 2021 hold-out (NDCG@5 0.892 vs 0.804). Pairwise spends
# effort separating 15th from 16th.
PARAMS = {
    "objective": "rank:ndcg",
    "eval_metric": "ndcg@5",
    "lambdarank_num_pair_per_sample": 8,
    "n_estimators": 400,
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_child_weight": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.8,
    "reg_lambda": 2.0,
    "random_state": config.RANDOM_SEED,
    "n_jobs": 4,
}


# Seed averaging was accuracy-neutral on the 2021 hold-out. It stays so a
# published forecast doesn't change just because the model was refit.
N_SEEDS = 5


@dataclass
class Ranker:
    """An ensemble of identically-configured rankers, plus the schema it expects.

    Scores are standardised within each race before averaging. Raw ranker
    outputs sit on arbitrary and inconsistent scales between seeds, so a plain
    mean would let whichever seed happened to produce the widest spread decide
    the result.
    """

    boosters: list[xgb.XGBRanker]
    feature_names: list[str]
    label: str
    trained_through: tuple[int, int]  # (season, round) of the last race seen

    @property
    def booster(self) -> xgb.XGBRanker:
        """First member. SHAP explains one tree ensemble, not an average of them."""
        return self.boosters[0]

    def score(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.feature_names if c not in df.columns]
        if missing:
            raise KeyError(f"missing feature columns: {missing}")

        X = df[self.feature_names]
        stacked = []
        for booster in self.boosters:
            s = booster.predict(X)
            sd = s.std()
            stacked.append((s - s.mean()) / sd if sd > 1e-9 else s - s.mean())
        return np.mean(stacked, axis=0)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        for i, booster in enumerate(self.boosters):
            booster.save_model(str(path.with_suffix(f".{i}.json")))
        path.with_suffix(".meta.json").write_text(
            json.dumps(
                {
                    "feature_names": self.feature_names,
                    "label": self.label,
                    "trained_through": list(self.trained_through),
                    "n_boosters": len(self.boosters),
                }
            )
        )

    @classmethod
    def load(cls, path: Path) -> Ranker:
        meta = json.loads(path.with_suffix(".meta.json").read_text())
        boosters = []
        for i in range(meta.get("n_boosters", 1)):
            b = xgb.XGBRanker(**PARAMS)
            b.load_model(str(path.with_suffix(f".{i}.json")))
            boosters.append(b)
        return cls(boosters, meta["feature_names"], meta["label"], tuple(meta["trained_through"]))


def recency_weights(df: pd.DataFrame, current_season_weight: float) -> np.ndarray:
    """Weight recent seasons more heavily.

    2026 is a regulation reset, so a 2019 result says less about 2026 pace than
    a 2026 result does. The weight decays one step per season back and is
    floored so old races still contribute; the exact value is chosen by
    backtest.tune_recency() rather than picked by hand.
    """
    latest = int(df["season"].max())
    age = latest - df["season"].astype(int)
    return np.maximum(current_season_weight ** (-age.to_numpy() / 2.0), 0.15)


def _prepare(df: pd.DataFrame, feature_names: list[str], label: str):
    d = df.dropna(subset=[label]).sort_values("race_seq").reset_index(drop=True)
    X = d[feature_names]
    y = d[label].to_numpy()
    qid = d["race_seq"].to_numpy()
    return d, X, y, qid


def train(
    df: pd.DataFrame,
    feature_names: list[str],
    label: str,
    current_season_weight: float = config.DEFAULT_CURRENT_SEASON_WEIGHT,
    n_seeds: int = N_SEEDS,
) -> Ranker:
    d, X, y, qid = _prepare(df, feature_names, label)
    if d.empty:
        raise RuntimeError(f"no training rows with label {label}")

    # XGBRanker weights are per group, not per row.
    per_race = d.groupby("race_seq", sort=True).head(1)
    group_weights = recency_weights(per_race, current_season_weight)

    boosters = []
    for i in range(max(1, n_seeds)):
        params = {**PARAMS, "random_state": config.RANDOM_SEED + i * 101}
        booster = xgb.XGBRanker(**params)
        booster.fit(X, y, qid=qid, sample_weight=group_weights, verbose=False)
        boosters.append(booster)

    last = d.iloc[-1]
    return Ranker(boosters, feature_names, label, (int(last["season"]), int(last["round"])))


def train_quali(df: pd.DataFrame, **kw) -> Ranker:
    return train(df, features.QUALI_FEATURES, "quali_relevance", **kw)


def train_race(df: pd.DataFrame, **kw) -> Ranker:
    return train(df, features.RACE_FEATURES, "race_relevance", **kw)


def feature_importance(ranker: Ranker, top: int = 15) -> pd.DataFrame:
    gain = ranker.booster.get_booster().get_score(importance_type="gain")
    rows = [{"feature": k, "gain": v} for k, v in gain.items()]
    out = pd.DataFrame(rows).sort_values("gain", ascending=False)
    total = out["gain"].sum()
    out["share_pct"] = (out["gain"] / total * 100).round(1) if total else 0.0
    return out.head(top).reset_index(drop=True)
