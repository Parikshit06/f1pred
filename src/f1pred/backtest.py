"""Walk-forward evaluation and tuning.

Every race in the window is forecast by models trained only on races before it
(race_seq strictly lower), then graded. That one rule lives in oos_scores(),
which every evaluation here goes through:

    train on races 1..k-1  ->  forecast race k  ->  grade it  ->  k+1

Scoring (fitting XGBoost) is the expensive step and turning scores into
probabilities is cheap, so the two are kept apart: tuning the temperature or
the model/simulation mix reuses the same out-of-sample scores.

Three windows, never overlapping in what they are used for:

    <= TUNE_START-1    training only
    TUNE_START..START-1  settings are fitted here (make backtest: 2022-23)
    >= START             reported, and never used to choose anything (2024-)

Two stages are graded separately, because they are different forecasts:

    post-qualifying  the official grid is known         (the headline numbers)
    pre-qualifying   the grid is drawn from the qualifying model's forecast,
                     and every grid and qualifying feature of the race is
                     replaced by its projection, so the actual qualifying
                     result can't leak in
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import baselines, config, features, metrics, model, probability, simulate

log = logging.getLogger(__name__)

MIN_TRAIN_RACES = 30
CALIBRATION_WARMUP = 24  # races before the window, scored only to seed the rolling calibration
RANKING = ["ndcg3", "ndcg5", "winner_hit", "podium_overlap", "top5_overlap", "spearman", "kendall"]
PROBABILITY = ["win_logloss", "win_brier", "podium_brier", "top10_brier"]


@dataclass
class Settings:
    """Everything tuned, in one place, so the backtest and the live forecast
    can't drift apart. Defaults are the values fitted on 2022-23 (config)."""

    temperature: float = config.DEFAULT_TEMPERATURE
    blend_weight: float = config.DEFAULT_BLEND_WEIGHT
    recency: float = config.DEFAULT_CURRENT_SEASON_WEIGHT
    quali_temperature: float = config.DEFAULT_QUALI_TEMPERATURE
    adaptive_temperature: bool = True
    n_sims: int = 3000


# ---------------------------------------------------------------------------
# Out-of-sample scores
# ---------------------------------------------------------------------------
def completed_races(df: pd.DataFrame, start_season: int, end_season: int | None = None) -> pd.DataFrame:
    keys = df[df["position"].notna()][["race_seq", "season", "round"]].drop_duplicates()
    keys = keys[keys["season"] >= start_season]
    if end_season is not None:
        keys = keys[keys["season"] <= end_season]
    return keys.sort_values("race_seq").reset_index(drop=True)


def oos_scores(
    df: pd.DataFrame,
    targets: list[int],
    train: Callable[[pd.DataFrame], model.Ranker],
    retrain_every: int = 1,
    score: Callable[[model.Ranker, pd.DataFrame, int], np.ndarray] | None = None,
) -> dict[int, tuple[pd.DataFrame, np.ndarray]]:
    """race_seq -> (that race's rows, scores from a model that never saw it).

    The training rows are exactly those with race_seq below the target's.
    Retrains every `retrain_every` targets; a model is only ever reused for a
    later race, never an earlier one. `score` overrides how a race is scored
    (the pre-qualifying stage scores a projected weekend).
    """
    out: dict[int, tuple[pd.DataFrame, np.ndarray]] = {}
    ranker, since = None, 10**9
    for seq in sorted(targets):
        history = df[df["race_seq"] < seq]
        if history["race_seq"].nunique() < MIN_TRAIN_RACES:
            continue
        if ranker is None or since >= retrain_every:
            ranker, since = train(history), 0
        since += 1
        race = df[df["race_seq"] == seq].reset_index(drop=True)
        out[seq] = (race, score(ranker, race, seq) if score else ranker.score(race))
    return out


def race_trainer(
    settings: Settings,
    feature_names: list[str] | None = None,
    n_seeds: int = model.N_SEEDS,
    params: dict | None = None,
):
    names = feature_names or features.RACE_FEATURES
    return lambda h: model.train(
        h, names, "race_relevance", current_season_weight=settings.recency, n_seeds=n_seeds, params=params
    )


def quali_trainer(settings: Settings, feature_names: list[str] | None = None, n_seeds: int = model.N_SEEDS):
    names = feature_names or features.QUALI_FEATURES
    return lambda h: model.train(
        h, names, "quali_relevance", current_season_weight=settings.recency, n_seeds=n_seeds
    )


def winner_index(race: pd.DataFrame, column: str = "position") -> int | None:
    pos = race[column].to_numpy(dtype=float)
    if not np.isfinite(pos).any():
        return None
    return int(np.nanargmin(pos))


def projected_weekend(race: pd.DataFrame, quali_scores: np.ndarray) -> pd.DataFrame:
    """The race's rows as the race model sees them before qualifying.

    The qualifying model's order stands in for the grid and the qualifying
    position; the driver's recent averages stand in for this weekend's gaps.
    Every GRID feature is overwritten, so the real result can't reach a
    pre-qualifying forecast - in the backtest, where it exists, or live.
    """
    t = race.copy()
    projected = pd.Series(-np.asarray(quali_scores)).rank(method="first").to_numpy()
    t["grid"] = projected
    t["quali_position"] = projected
    t["quali_gap_to_pole_pct"] = t["drv_pace_gap_pct"]
    t["quali_gap_to_teammate_pct"] = t["drv_teammate_quali_edge"]
    assert set(features.GRID_FEATURES) <= {
        "grid",
        "quali_position",
        "quali_gap_to_pole_pct",
        "quali_gap_to_teammate_pct",
    }
    return t


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------
@dataclass
class BacktestResult:
    races: pd.DataFrame = field(default_factory=pd.DataFrame)  # one row per (method, race)
    outcomes: pd.DataFrame = field(default_factory=pd.DataFrame)  # one row per (method, race, driver)
    settings: Settings = field(default_factory=Settings)
    stage: str = "post_quali"

    def summary(self) -> pd.DataFrame:
        if self.races.empty:
            return pd.DataFrame()
        cols = [c for c in RANKING + PROBABILITY if c in self.races]
        out = self.races.groupby("method")[cols].mean()
        out["n_races"] = self.races.groupby("method").size()
        order = ["model", *baselines.KINDS]
        return out.reindex([m for m in order if m in out.index])

    def by_season(self, methods: tuple[str, ...] = ("model", "grid")) -> pd.DataFrame:
        sub = self.races[self.races["method"].isin(methods)]
        cols = [c for c in RANKING + PROBABILITY if c in sub]
        out = sub.groupby(["season", "method"])[cols].mean()
        out["n_races"] = sub.groupby(["season", "method"]).size()
        return out.reset_index()

    def reliability(self, method: str = "model") -> dict[str, pd.DataFrame]:
        o = self.outcomes[self.outcomes["method"] == method]
        return {
            event: metrics.reliability(o[f"p_{event}"], o[flag])
            for event, flag in (("win", "won"), ("podium", "podium"), ("top10", "top10"))
            if not o.empty
        }

    def calibration_error(self, method: str = "model") -> dict[str, float]:
        o = self.outcomes[self.outcomes["method"] == method]
        return {
            event: metrics.expected_calibration_error(o[f"p_{event}"], o[flag])
            for event, flag in (("win", "won"), ("podium", "podium"), ("top10", "top10"))
        }

    def compare(self, a: str = "model", b: str = "grid") -> pd.DataFrame:
        return metrics.paired_comparison(self.races, a, b, RANKING + PROBABILITY)


def walk_forward(
    df: pd.DataFrame,
    start_season: int,
    end_season: int | None = None,
    retrain_every: int = 1,
    settings: Settings | None = None,
    stage: str = "post_quali",
    race_features: list[str] | None = None,
    quali_features: list[str] | None = None,
    include_baselines: bool = True,
    n_seeds: int = model.N_SEEDS,
    warmup: int = CALIBRATION_WARMUP,
    scored: dict | None = None,
) -> BacktestResult:
    """Forecast every completed race from start_season on, exactly as live.

    The `warmup` races before the window are scored too, but only to give the
    rolling temperature its trailing history from the first reported race;
    they are not graded. `scored` reuses post-qualifying scores from
    oos_scores() when only the probability step is being varied.
    """
    s = settings or Settings()
    df = df.sort_values(["race_seq", "driver_id"]).reset_index(drop=True)
    window = completed_races(df, start_season, end_season)
    if window.empty:
        return BacktestResult(settings=s, stage=stage)
    earlier = completed_races(df, 0)
    earlier = earlier[earlier["race_seq"] < window["race_seq"].min()].tail(warmup)
    targets = earlier["race_seq"].tolist() + window["race_seq"].tolist()
    reported = set(window["race_seq"])

    quali_scored: dict = {}
    score = None
    if stage == "pre_quali":
        quali_scored = oos_scores(df, targets, quali_trainer(s, quali_features, n_seeds), retrain_every)
        targets = [t for t in targets if t in quali_scored]

        def score(ranker, race, seq):
            return ranker.score(projected_weekend(race, quali_scored[seq][1]))

    if scored is None:
        scored = oos_scores(df, targets, race_trainer(s, race_features, n_seeds), retrain_every, score)
    else:  # scores computed once and reused, e.g. to compare calibration choices
        scored = {k: v for k, v in scored.items() if k in set(targets)}

    rows, outcomes = [], []
    seen: dict[str, tuple[list, list]] = {}  # method -> (score groups, winner indices)

    def temperature_for(method: str, fallback: float) -> float:
        groups, winners = seen.setdefault(method, ([], []))
        return (
            probability.rolling_temperature(groups, winners, fallback) if s.adaptive_temperature else fallback
        )

    def remember(method: str, sc: np.ndarray, win: int | None) -> None:
        if win is not None:
            seen[method][0].append(np.asarray(sc))
            seen[method][1].append(win)

    for seq in sorted(scored):
        race, sc = scored[seq]
        if race["position"].notna().sum() < 5:
            continue
        season, rnd = int(race["season"].iloc[0]), int(race["round"].iloc[0])
        actual = pd.Series(race["position"].to_numpy(), index=race["driver_id"].to_numpy())
        win = winner_index(race)

        # ---- the model ---------------------------------------------------
        t = temperature_for("model", s.temperature)
        if seq in reported:
            grid_known = stage == "post_quali"
            inputs = simulate.race_inputs(
                race if grid_known else projected_weekend(race, quali_scored[seq][1]),
                sc,
                grid_known=grid_known,
                quali_scores=None if grid_known else quali_scored[seq][1],
            )
            fc = simulate.forecast(
                inputs,
                t,
                s.blend_weight,
                s.n_sims,
                grid_temperature=s.quali_temperature,
                seed=config.RANDOM_SEED + seq,
            )
            table = fc.table.set_index("driver_id")
            pred_rank = pd.Series(-sc, index=race["driver_id"].to_numpy()).rank(method="first")
            rows.append(
                {
                    "method": "model",
                    "season": season,
                    "round": rnd,
                    "temperature": t,
                    "p_top_pick": float(table.loc[pred_rank.idxmin(), "p_win"]),
                    **metrics.ranking(pred_rank, actual),
                    **metrics.probability(table, actual),
                }
            )
            outcomes.append(metrics.outcome_rows(table, actual, method="model", season=season, round=rnd))
        remember("model", sc, win)

        # ---- baselines ---------------------------------------------------
        if not include_baselines:
            continue
        for kind in baselines.KINDS:
            if kind == "grid" and stage == "pre_quali":
                continue  # there is no grid before qualifying
            b_scores = baselines.scores(race, kind)
            t_b = temperature_for(kind, 1.0)
            if seq in reported:
                fc = simulate.ranking_forecast(
                    race["driver_id"].tolist(), b_scores, t_b, seed=config.RANDOM_SEED + seq
                )
                table = fc.table.set_index("driver_id")
                pred_rank = baselines.order(race, kind).rank(method="first")
                rows.append(
                    {
                        "method": kind,
                        "season": season,
                        "round": rnd,
                        "temperature": t_b,
                        "p_top_pick": float(table.loc[pred_rank.idxmin(), "p_win"]),
                        **metrics.ranking(pred_rank, actual),
                        **metrics.probability(table, actual),
                    }
                )
                outcomes.append(metrics.outcome_rows(table, actual, method=kind, season=season, round=rnd))
            remember(kind, b_scores, win)

    return BacktestResult(
        pd.DataFrame(rows),
        pd.concat(outcomes, ignore_index=True) if outcomes else pd.DataFrame(),
        s,
        stage,
    )


# ---------------------------------------------------------------------------
# Qualifying
# ---------------------------------------------------------------------------
def walk_forward_quali(
    df: pd.DataFrame,
    start_season: int,
    end_season: int | None = None,
    retrain_every: int = 1,
    settings: Settings | None = None,
    quali_features: list[str] | None = None,
    races: list[int] | None = None,
    n_seeds: int = model.N_SEEDS,
    warmup: int = CALIBRATION_WARMUP,
) -> pd.DataFrame:
    """Grade the qualifying forecast: one row per race.

    `races` restricts grading to those race_seq values (e.g. weekends with
    practice data), while the warm-up still seeds the temperature.
    """
    s = settings or Settings()
    df = df.sort_values(["race_seq", "driver_id"]).reset_index(drop=True)
    window = df[df["quali_position"].notna()][["race_seq", "season"]].drop_duplicates()
    window = window[window["season"] >= start_season]
    if end_season is not None:
        window = window[window["season"] <= end_season]
    if races is not None:
        window = window[window["race_seq"].isin(races)]
    if window.empty:
        return pd.DataFrame()
    before = df[df["quali_position"].notna() & (df["race_seq"] < window["race_seq"].min())]
    warm = sorted(before["race_seq"].unique())[-warmup:]
    targets = list(warm) + window["race_seq"].tolist()
    scored = oos_scores(df, targets, quali_trainer(s, quali_features, n_seeds), retrain_every)

    graded = set(window["race_seq"])
    seen: dict[str, tuple[list, list]] = {}
    rows = []
    for seq in sorted(scored):
        race, sc = scored[seq]
        pole = winner_index(race, "quali_position")
        # The no-model reference: recent qualifying average, best first.
        recent = race["drv_avg_quali_3"].fillna(race["drv_avg_quali_3"].max() + 1).fillna(1.0)
        candidates = {"model": np.asarray(sc), "recent_quali_form": -np.log(recent.rank().to_numpy())}
        for method, method_scores in candidates.items():
            groups, poles = seen.setdefault(method, ([], []))
            if seq in graded and pole is not None:
                fallback = s.quali_temperature if method == "model" else 1.0
                t = probability.rolling_temperature(groups, poles, fallback)
                p = probability.plackett_luce(method_scores, t)
                actual = pd.Series(race["quali_position"].to_numpy(), index=race["driver_id"].to_numpy())
                pred_rank = pd.Series(-method_scores, index=race["driver_id"].to_numpy()).rank(method="first")
                rank = metrics.ranking(pred_rank, actual)
                rows.append(
                    {
                        "method": method,
                        "season": int(race["season"].iloc[0]),
                        "round": int(race["round"].iloc[0]),
                        "race_seq": seq,
                        "pole_hit": rank.get("winner_hit"),
                        "pole_logloss": float(-np.log(max(p[pole], metrics.EPS))),
                        "ndcg5": rank.get("ndcg5"),
                        "top10_overlap": len(
                            set(pred_rank.nsmallest(10).index) & set(actual.nsmallest(10).index)
                        ),
                        "spearman": rank.get("spearman"),
                        "temperature": t,
                    }
                )
            if pole is not None:
                groups.append(method_scores)
                poles.append(pole)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tuning: fitted on the tuning window, never on the reported one
# ---------------------------------------------------------------------------
def _winner_loss(scored: dict, temperature: float) -> float:
    losses = []
    for race, sc in scored.values():
        win = winner_index(race)
        if win is not None:
            losses.append(-np.log(max(probability.plackett_luce(sc, temperature)[win], metrics.EPS)))
    return float(np.mean(losses)) if losses else float("inf")


def _fit_temperature(scored: dict, column: str = "position") -> float:
    groups, wins = [], []
    for race, sc in scored.values():
        win = winner_index(race, column)
        if win is not None:
            groups.append(sc)
            wins.append(win)
    return probability.fit_temperature(groups, wins)


def tune(
    df: pd.DataFrame,
    tune_start: int,
    tune_end: int,
    retrain_every: int = 4,
    recency_candidates: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0, 4.0),
    blend_candidates: tuple[float, ...] = tuple(np.round(np.linspace(0.0, 1.0, 11), 2)),
    n_sims: int = 1500,
) -> tuple[Settings, dict]:
    """Fit the tuned settings on tune_start..tune_end.

    1. recency weight: the one that gives the lowest winner log loss at its own
       best temperature (each candidate retrains the models)
    2. temperature: maximum likelihood on that window's winners
    3. model/simulation mix: lowest winner log loss of the full forecast
    4. qualifying temperature: maximum likelihood on that window's pole-sitters
    """
    keys = completed_races(df, tune_start, tune_end)["race_seq"].tolist()
    trace: dict = {"window": [tune_start, tune_end], "n_races": len(keys), "recency": [], "blend": []}

    best_recency, best_loss, best_scored = None, np.inf, None
    for w in recency_candidates:
        scored = oos_scores(df, keys, race_trainer(Settings(recency=w)), retrain_every)
        t = _fit_temperature(scored)
        loss = _winner_loss(scored, t)
        trace["recency"].append({"recency": w, "temperature": t, "win_logloss": loss})
        log.info("  recency %.1f -> T %.2f, log loss %.4f", w, t, loss)
        if loss < best_loss:
            best_recency, best_loss, best_scored = w, loss, scored

    temperature = _fit_temperature(best_scored)
    best_w, best_blend_loss = 0.5, np.inf
    for w in blend_candidates:
        losses = []
        for seq, (race, sc) in best_scored.items():
            win = winner_index(race)
            if win is None:
                continue
            fc = simulate.forecast(
                simulate.race_inputs(race, sc, grid_known=True),
                temperature,
                float(w),
                n_sims,
                seed=config.RANDOM_SEED + seq,
            )
            losses.append(-np.log(max(fc.column("p_win")[win], metrics.EPS)))
        loss = float(np.mean(losses))
        trace["blend"].append({"blend_weight": float(w), "win_logloss": loss})
        if loss < best_blend_loss:
            best_w, best_blend_loss = float(w), loss

    quali_scored = oos_scores(df, keys, quali_trainer(Settings(recency=best_recency)), retrain_every)
    quali_t = _fit_temperature(quali_scored, "quali_position")

    settings = Settings(
        temperature=temperature, blend_weight=best_w, recency=best_recency, quali_temperature=quali_t
    )
    trace["chosen"] = asdict(settings)
    log.info("tuned on %d-%d: %s", tune_start, tune_end, asdict(settings))
    return settings, trace


# ---------------------------------------------------------------------------
# The live forecast uses exactly these
# ---------------------------------------------------------------------------
def load_settings() -> Settings:
    """The settings the last `make backtest` fitted and reported with, else the
    defaults in config. The live forecast and the report can't disagree."""
    import json

    path = config.REPORTS / "backtest.json"
    try:
        saved = json.loads(path.read_text()).get("settings") or {}
    except (OSError, json.JSONDecodeError):
        saved = {}
    known = {k: v for k, v in saved.items() if k in Settings.__dataclass_fields__}
    return Settings(**known)


def trailing_temperatures(
    df: pd.DataFrame, seq: int, settings: Settings, stage: str, retrain_every: int = 1
) -> tuple[float, float]:
    """Race and qualifying temperatures for a forecast of race `seq`, fitted the
    way the walk-forward fits them: on out-of-sample scores for the completed
    races just before it. Falls back to the tuned values when adaptive
    calibration is off or there is too little history."""
    if not settings.adaptive_temperature:
        return settings.temperature, settings.quali_temperature
    done = completed_races(df, 0)
    trail = done[done["race_seq"] < seq]["race_seq"].tail(probability.ADAPTIVE_WINDOW).tolist()
    quali = oos_scores(df, trail, quali_trainer(settings), retrain_every)

    score = None
    if stage == "pre_quali":

        def score(ranker, race, s):
            return ranker.score(projected_weekend(race, quali[s][1]))

        trail = [t for t in trail if t in quali]
    race = oos_scores(df, trail, race_trainer(settings), retrain_every, score)

    def fitted(scored: dict, column: str, fallback: float) -> float:
        groups, wins = [], []
        for r, sc in scored.values():
            w = winner_index(r, column)
            if w is not None:
                groups.append(sc)
                wins.append(w)
        return probability.rolling_temperature(groups, wins, fallback)

    return fitted(race, "position", settings.temperature), fitted(
        quali, "quali_position", settings.quali_temperature
    )
