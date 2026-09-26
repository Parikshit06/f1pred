"""The evaluation behind every number and design choice the project reports.

    run_backtest()     reports/backtest.json     make backtest
    run_experiments()  reports/experiments.json  make experiments

Every experiment is a walk-forward: each race forecast by models trained only
on races before it. Comparisons are paired by race with bootstrap intervals
(metrics.paired_comparison), because a few points on 60 races is often noise
and the report should say when it is.

Where an experiment decides something - a setting, whether a feature group
stays - the decision is read off the tuning seasons (2022-23), and the test
seasons (2024-) are reported alongside so the choice can be checked, not made,
on them.
"""

from __future__ import annotations

import logging
from dataclasses import asdict

import numpy as np
import pandas as pd

from . import backtest, baselines, config, features, metrics, model, probability, provenance

log = logging.getLogger(__name__)

HEADLINE = ["ndcg3", "ndcg5", "winner_hit", "podium_overlap", "top5_overlap", "spearman", "kendall"]
PROBABILITY = ["win_logloss", "win_brier", "podium_brier", "top10_brier"]


def _records(df: pd.DataFrame, digits: int = 4) -> list[dict]:
    if df is None or df.empty:
        return []
    return df.round(digits).replace({np.nan: None}).to_dict("records")


def _summary(res: backtest.BacktestResult) -> list[dict]:
    s = res.summary()
    return _records(s.rename_axis("method").reset_index())


# ---------------------------------------------------------------------------
# Headline backtest
# ---------------------------------------------------------------------------
def run_backtest(
    df: pd.DataFrame,
    start_season: int,
    tune_season: int,
    n_sims: int = 3000,
    retrain_every: int = 1,
) -> dict:
    """Tune on tune_season..start_season-1, then grade start_season onward."""
    if tune_season >= start_season:
        raise ValueError(f"tune season {tune_season} must precede the first reported season {start_season}")
    settings, trace = backtest.tune(df, tune_season, start_season - 1)
    settings.n_sims = n_sims

    log.info("post-qualifying walk-forward from %d", start_season)
    post = backtest.walk_forward(df, start_season, retrain_every=retrain_every, settings=settings)
    log.info("pre-qualifying walk-forward from %d", start_season)
    pre = backtest.walk_forward(
        df, start_season, retrain_every=retrain_every, settings=settings, stage="pre_quali"
    )
    log.info("qualifying walk-forward from %d", start_season)
    quali = backtest.walk_forward_quali(df, start_season, retrain_every=retrain_every, settings=settings)

    races = post.races[post.races["method"] == "model"]
    last = races.sort_values(["season", "round"]).iloc[-1]
    quali_summary = quali.groupby("method")[
        ["pole_hit", "pole_logloss", "ndcg5", "top10_overlap", "spearman"]
    ].mean()
    quali_summary["n_races"] = quali.groupby("method").size()

    return {
        "window": {
            "start_season": start_season,
            "tuned_on": [tune_season, start_season - 1],
            "n_races": len(races),
            "through": [int(last["season"]), int(last["round"])],
        },
        "settings": asdict(settings),
        "tuning": trace,
        "summary": _summary(post),
        "by_season": _records(post.by_season()),
        "comparison_vs_grid": _records(post.compare("model", "grid")),
        "calibration_error": {k: round(v, 4) for k, v in post.calibration_error().items()},
        "grid_calibration_error": {k: round(v, 4) for k, v in post.calibration_error("grid").items()},
        "reliability": {k: _records(v) for k, v in post.reliability().items()},
        "races": _records(races[["season", "round", *HEADLINE, *PROBABILITY, "temperature", "p_top_pick"]]),
        "pre_quali": {
            "summary": _summary(pre),
            "comparison_vs_championship": _records(pre.compare("model", "championship")),
            "calibration_error": {k: round(v, 4) for k, v in pre.calibration_error().items()},
        },
        "qualifying": {
            "summary": _records(quali_summary.rename_axis("method").reset_index()),
            "comparison": _records(
                metrics.paired_comparison(
                    quali,
                    "model",
                    "recent_quali_form",
                    ["pole_hit", "pole_logloss", "ndcg5", "top10_overlap", "spearman"],
                )
            ),
        },
        "provenance": provenance.record(
            model_version=provenance.model_version(
                features.RACE_FEATURES, model.PARAMS, features.FEATURE_VERSION
            ),
            feature_version=features.FEATURE_VERSION,
        ),
    }


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
DRIVER_FORM = [
    "drv_avg_finish_3",
    "drv_avg_finish_5",
    "drv_points_rate_5",
    "drv_podium_rate_10",
    "drv_top10_rate_10",
    "drv_positions_gained_5",
    "drv_dnf_rate_10",
    "drv_experience",
    "champ_position_before",
    "champ_points_before",
]
QUALI_FORM = features.QUALI_FORM_FEATURES
TEAM_FORM = [
    "team_avg_finish_3",
    "team_avg_finish_5",
    "team_points_rate_5",
    "team_dnf_rate_10",
    "team_pace_gap_pct",
    "team_pace_trend",
    "team_avg_quali_5",
]
TEAMMATE = features.TEAMMATE_FEATURES
CIRCUIT = [
    "drv_circuit_avg_finish",
    "team_circuit_avg_finish",
    "circuit_overtaking_score",
    "circuit_dnf_rate",
    "drv_circuit_starts",
    "circuit_pole_win_rate",
    "season_progress",
]
QUALIFYING = ["quali_position", "quali_gap_to_pole_pct"]
GROUPS = {
    "qualifying": QUALIFYING,
    "driver form": DRIVER_FORM,
    "team form": TEAM_FORM,
    "circuit": CIRCUIT,
}
# Tested and not in the model; re-tested by adding them to the full model.
CANDIDATES = {"qualifying form": QUALI_FORM, "teammate": TEAMMATE}


def _assert_groups_cover_the_model() -> None:
    grouped = {f for cols in GROUPS.values() for f in cols} | {"grid"}
    missing = set(features.RACE_FEATURES) - grouped
    extra = grouped - set(features.RACE_FEATURES)
    if missing or extra:
        raise AssertionError(f"feature groups out of date: missing {missing}, unknown {extra}")


def feature_sets() -> dict[str, list[str]]:
    """Grid alone, grid plus each group, the full model, the full model minus each
    group, and plus each candidate group that was tested and left out."""
    _assert_groups_cover_the_model()
    full = list(features.RACE_FEATURES)
    sets = {"grid only": ["grid"]}
    for name, cols in GROUPS.items():
        sets[f"grid + {name}"] = ["grid", *cols]
    sets["full model"] = full
    for name, cols in GROUPS.items():
        sets[f"full - {name}"] = [f for f in full if f not in cols]
    for name, cols in CANDIDATES.items():
        sets[f"full + {name} (not in model)"] = [*full, *cols]
    return sets


def _variant(df, start, end, settings, feats, n_seeds, retrain_every, stage="post_quali") -> pd.DataFrame:
    res = backtest.walk_forward(
        df,
        start,
        end_season=end,
        retrain_every=retrain_every,
        settings=settings,
        race_features=feats,
        include_baselines=False,
        n_seeds=n_seeds,
        stage=stage,
    )
    return res.races


def _paired_table(results: dict[str, pd.DataFrame], reference: str, cols: list[str]) -> list[dict]:
    """Every variant against the reference, race by race."""
    stacked = pd.concat([r.assign(method=name) for name, r in results.items()], ignore_index=True)
    out = []
    for name, result in results.items():
        row = {"variant": name, "n_races": len(result)}
        row.update({m: float(result[m].mean()) for m in cols})
        if name != reference:
            comp = metrics.paired_comparison(stacked, name, reference, cols, n_boot=3000)
            for r in comp.itertuples():
                row[f"{r.metric}_diff"] = r.difference
                row[f"{r.metric}_ci"] = [r.ci_low, r.ci_high]
        out.append(row)
    return out


ABLATION_METRICS = ["ndcg3", "ndcg5", "winner_hit", "spearman", "win_logloss", "podium_brier", "top10_brier"]


def ablation(df, windows, settings, n_seeds=1, retrain_every=1) -> dict:
    """Which information improves out-of-sample forecasts, and which duplicates other features."""
    out = {}
    for label, (start, end) in windows.items():
        results = {}
        for name, feats in feature_sets().items():
            log.info("ablation %s: %s", label, name)
            results[name] = _variant(df, start, end, settings, feats, n_seeds, retrain_every)
        out[label] = _paired_table(results, "full model", ABLATION_METRICS)
    return out


def pre_quali_ablation(df, windows, settings, n_seeds=1, retrain_every=1) -> dict:
    """The left-out candidates again, before qualifying, where the grid columns
    hold the qualifying model's projection. A group the projection already
    carries can count twice here even when it is harmless after qualifying."""
    full = list(features.RACE_FEATURES)
    out = {}
    for label, (start, end) in windows.items():
        results = {
            "full model": _variant(df, start, end, settings, full, n_seeds, retrain_every, "pre_quali")
        }
        for name, cols in CANDIDATES.items():
            log.info("pre-qualifying ablation %s: + %s", label, name)
            results[f"full + {name} (not in model)"] = _variant(
                df, start, end, settings, [*full, *cols], n_seeds, retrain_every, "pre_quali"
            )
        out[label] = _paired_table(results, "full model", ABLATION_METRICS)
    return out


def calibration_variants(df, start, settings, n_seeds, retrain_every) -> list[dict]:
    """Does calibration help, and does the simulation earn its place?

    raw          ranker scores read directly as Plackett-Luce strengths (T=1)
    fixed        temperature fitted on the tuning seasons, held all window
    rolling      refitted before each race on the 24 before it (deployed)
    no simulation / simulation only   the two halves of the mixture alone
    All share one set of out-of-sample scores, so only the probability step differs.
    """
    targets_df = backtest.completed_races(df, 0)
    first = backtest.completed_races(df, start)["race_seq"].min()
    targets = targets_df[targets_df["race_seq"] >= first - backtest.CALIBRATION_WARMUP]["race_seq"].tolist()
    scored = backtest.oos_scores(df, targets, backtest.race_trainer(settings, n_seeds=n_seeds), retrain_every)

    variants = {
        "raw (T=1, no simulation)": {"temperature": 1.0, "adaptive_temperature": False, "blend_weight": 1.0},
        "fixed temperature": {"adaptive_temperature": False},
        "rolling temperature (deployed)": {},
        "rolling, no simulation": {"blend_weight": 1.0},
        "rolling, simulation only": {"blend_weight": 0.0},
    }
    rows = []
    for name, change in variants.items():
        s = backtest.Settings(**{**asdict(settings), **change})
        res = backtest.walk_forward(df, start, settings=s, include_baselines=False, scored=scored)
        m = res.races
        rows.append(
            {
                "variant": name,
                "n_races": len(m),
                **{c: float(m[c].mean()) for c in PROBABILITY},
                **{f"ece_{k}": v for k, v in res.calibration_error().items()},
            }
        )
    return rows


def definition_experiment(
    frames: dict[str, pd.DataFrame], reference: str, windows, settings, n_seeds=1, retrain_every=2
) -> dict:
    """The same model on feature tables built two ways, graded race by race.

    Used for choices about how a feature is defined rather than whether it is
    in, such as mean or median recent form.
    """
    cols = ["ndcg3", "ndcg5", "winner_hit", "spearman", "win_logloss", "podium_brier"]
    out = {}
    for label, (start, end) in windows.items():
        results = {
            name: _variant(df, start, end, settings, None, n_seeds, retrain_every)
            for name, df in frames.items()
        }
        out[label] = _paired_table(results, reference, cols)
    return out


def practice_experiment(df, settings, n_seeds=1, retrain_every=1) -> dict:
    """Does practice pace improve the qualifying forecast, and the race forecast
    beyond the official grid? Graded only on weekends with practice data."""
    with_practice = df[df["practice_available"] > 0]["race_seq"].unique().tolist()
    if not with_practice:
        return {"note": "no practice data ingested"}
    no_practice = [f for f in features.QUALI_FEATURES if f not in features.PRACTICE_FEATURES]
    start = int(df[df["race_seq"].isin(with_practice)]["season"].min())
    cols = ["pole_hit", "pole_logloss", "ndcg5", "top10_overlap", "spearman"]

    def quali(feats):
        r = backtest.walk_forward_quali(
            df,
            start,
            retrain_every=retrain_every,
            settings=settings,
            quali_features=feats,
            races=with_practice,
            n_seeds=n_seeds,
        )
        return r[r["method"] == "model"]

    quali_results = {"with practice": quali(features.QUALI_FEATURES), "without practice": quali(no_practice)}

    race_cols = ["ndcg5", "winner_hit", "spearman", "win_logloss", "podium_brier"]
    race_results = {}
    for name, feats in (
        ("race model", features.RACE_FEATURES),
        ("race model + practice", features.RACE_FEATURES + features.PRACTICE_FEATURES),
    ):
        r = _variant(df, start, None, settings, feats, n_seeds, retrain_every)
        keys = df[df["race_seq"].isin(with_practice)][["season", "round"]].drop_duplicates()
        race_results[name] = r.merge(keys, on=["season", "round"])
    seasons = sorted(df[df["race_seq"].isin(with_practice)]["season"].unique().tolist())
    return {
        "weekends_with_practice": len(with_practice),
        "seasons": [int(x) for x in seasons],
        "qualifying": _paired_table(quali_results, "without practice", cols),
        "race": _paired_table(race_results, "race model", race_cols),
    }


def redundancy(df, start, settings, threshold: float = 0.8) -> dict:
    """Feature pairs that carry nearly the same information, and how much each
    group matters when shuffled within the race (permutation importance)."""
    train = df[(df["position"].notna()) & (df["season"] < start)]
    corr = train[features.RACE_FEATURES].corr(method="spearman")
    pairs = []
    for i, a in enumerate(corr.columns):
        for b in corr.columns[i + 1 :]:
            r = corr.loc[a, b]
            if np.isfinite(r) and abs(r) >= threshold:
                pairs.append({"a": a, "b": b, "spearman": round(float(r), 3)})
    pairs.sort(key=lambda p: -abs(p["spearman"]))

    # Permutation importance, out of sample: each race scored by its own
    # walk-forward model with one group's values shuffled between drivers.
    rng = np.random.default_rng(config.RANDOM_SEED)
    drops: dict[str, list[tuple[float, float]]] = {g: [] for g in [*GROUPS, "grid"]}
    shap_share: dict[str, list[float]] = {g: [] for g in [*GROUPS, "grid"]}
    group_of = {f: g for g, cols in GROUPS.items() for f in cols} | {"grid": "grid"}

    def score(ranker, race, seq):
        base = ranker.score(race)
        actual = pd.Series(race["position"].to_numpy(), index=race["driver_id"].to_numpy())
        win = backtest.winner_index(race)
        base_rank = metrics.ranking(pd.Series(-base, index=actual.index).rank(method="first"), actual)
        base_ll = -np.log(max(probability.plackett_luce(base, settings.temperature)[win], metrics.EPS))
        for g, record in drops.items():
            cols = [f for f in (GROUPS.get(g) or ["grid"]) if f in ranker.feature_names]
            shuffled = race.copy()
            perm = rng.permutation(len(race))
            for c in cols:  # one column at a time keeps each a float column
                shuffled[c] = race[c].to_numpy(dtype=float, na_value=np.nan)[perm]
            s = ranker.score(shuffled)
            r = metrics.ranking(pd.Series(-s, index=actual.index).rank(method="first"), actual)
            ll = -np.log(max(probability.plackett_luce(s, settings.temperature)[win], metrics.EPS))
            record.append((base_rank["ndcg5"] - r["ndcg5"], ll - base_ll))
        phi = np.abs(ranker.contributions(race)).sum(axis=0)
        total = phi.sum() or 1.0
        for g, shares in shap_share.items():
            shares.append(sum(v for f, v in zip(ranker.feature_names, phi) if group_of.get(f) == g) / total)
        return base

    targets = backtest.completed_races(df, start)["race_seq"].tolist()
    backtest.oos_scores(df, targets, backtest.race_trainer(settings, n_seeds=1), retrain_every=1, score=score)
    n = len(targets)
    importance = []
    for g, vals in drops.items():
        arr = np.array(vals)
        nd = metrics.bootstrap_mean(arr[:, 0])
        ll = metrics.bootstrap_mean(arr[:, 1])
        importance.append(
            {
                "group": g,
                "ndcg5_drop": nd[0],
                "ndcg5_drop_ci": [nd[1], nd[2]],
                "logloss_rise": ll[0],
                "logloss_rise_ci": [ll[1], ll[2]],
                "mean_abs_shap_share": float(np.mean(shap_share[g])) if shap_share[g] else None,
            }
        )
    importance.sort(key=lambda r: -r["logloss_rise"])
    return {"correlated_pairs": pairs, "permutation_importance": importance, "n_races": n}


def parameter_stability(df, windows, settings, retrain_every=2) -> dict:
    """A handful of XGBoost settings on two separate validation windows.

    Not a search: the question is whether the settings in use are among the
    best on both windows, or whether the ranking of candidates flips between
    them (in which case the difference is noise).
    """
    candidates = {
        "depth 4, 400 trees (in use)": {},
        "depth 3, 400 trees": {"max_depth": 3},
        "depth 6, 400 trees": {"max_depth": 6},
        "depth 4, 200 trees": {"n_estimators": 200},
        "depth 4, 800 trees": {"n_estimators": 800},
        "pairwise objective": {"objective": "rank:pairwise"},
    }
    out = {}
    for label, (start, end) in windows.items():
        rows = []
        targets = backtest.completed_races(df, start, end)["race_seq"].tolist()
        for name, params in candidates.items():
            scored = backtest.oos_scores(
                df, targets, backtest.race_trainer(settings, n_seeds=1, params=params), retrain_every
            )
            ndcg, ll = [], []
            groups, wins = [], []
            for race, sc in scored.values():
                actual = pd.Series(race["position"].to_numpy(), index=race["driver_id"].to_numpy())
                ndcg.append(
                    metrics.ranking(pd.Series(-sc, index=actual.index).rank(method="first"), actual)["ndcg5"]
                )
                groups.append(sc)
                wins.append(backtest.winner_index(race))
            t = probability.fit_temperature(groups, wins)
            ll = [-np.log(max(probability.plackett_luce(g, t)[w], metrics.EPS)) for g, w in zip(groups, wins)]
            rows.append({"params": name, "ndcg5": float(np.mean(ndcg)), "win_logloss": float(np.mean(ll))})
        out[label] = rows
    return out


def season_projection_experiment(
    df: pd.DataFrame,
    seasons: tuple[int, ...] = (2019, 2020, 2021, 2022, 2023, 2024, 2025),
    methods: tuple[str, ...] = ("mean5", "neutral", "median8", "rescored8"),
    n_sims: int = 3000,
) -> dict:
    """How should a driver be scored for 'an ordinary weekend' in the season projection?

    At each title-backtest checkpoint one model is trained on races before it;
    every method scores the field with that same model and the rest of the
    season is simulated. Graded against every driver's and team's final points,
    the final gap between teammates (the case where a driver beside a strong
    teammate is marked down), and the champion's probability.

    rescored8 skips the synthetic weekend: a driver's score is the median of
    the model's scores for their own last 8 real weekends
    (championship.season_strength, the method in use).
    """
    from . import championship, simulate, spread_calibration, title_backtest

    rows = []
    for season in seasons:
        drivers_final, teams_final = spread_calibration.final_points(season)
        champion = title_backtest.actual_champion(season)
        rounds = sorted(df[df.season == season]["round"].unique())
        for frac in spread_calibration.CHECKPOINTS:
            after = round(len(rounds) * frac)
            nxt = df[(df.season == season) & (df["round"] == after + 1)]
            if after < 3 or after >= len(rounds) or nxt.empty:
                continue
            train = df[df["race_seq"] < int(nxt["race_seq"].iloc[0])]
            ranker = model.train_race(train)
            race = nxt.reset_index(drop=True)
            for method in methods:
                if method == "rescored8":
                    scores = championship.season_strength(ranker, race, train)
                else:
                    scores = ranker.score(championship.typical_weekend(race, train, method=method))
                out = championship.project(
                    race["driver_id"].tolist(),
                    race["constructor_id"].tolist(),
                    scores,
                    simulate.dnf_probability(race),
                    season,
                    after,
                    n_sims=n_sims,
                )
                if not out:
                    continue
                d = out["drivers"].set_index("driver_id")
                err = [d.loc[k, "projected"] - v for k, v in drivers_final.items() if k in d.index]
                c = out["constructors"].set_index("team")
                team_err = [c.loc[k, "projected"] - v for k, v in teams_final.items() if k in c.index]
                gap_err = []
                for _, members in race.groupby("constructor_id")["driver_id"]:
                    m = [x for x in members if x in d.index and x in drivers_final]
                    if len(m) == 2:
                        projected = d.loc[m[0], "projected"] - d.loc[m[1], "projected"]
                        gap_err.append(abs(projected - (drivers_final[m[0]] - drivers_final[m[1]])))
                p_champ = float(d.loc[champion, "p_title"]) if champion in d.index else 0.0
                rows.append(
                    {
                        "method": method,
                        "season": season,
                        "round": after,
                        "driver_points_mae": float(np.mean(np.abs(err))),
                        "team_points_mae": float(np.mean(np.abs(team_err))),
                        "teammate_gap_mae": float(np.mean(gap_err)) if gap_err else np.nan,
                        "champion_logloss": float(-np.log(max(p_champ, 1e-3))),
                    }
                )
            log.info("projection experiment %d after r%d done", season, after)
    frame = pd.DataFrame(rows)
    cols = ["driver_points_mae", "team_points_mae", "teammate_gap_mae", "champion_logloss"]
    out = {"n_checkpoints": int((frame["method"] == methods[0]).sum()), "seasons": list(seasons)}
    # Decided on the earlier seasons, checked on the later ones.
    for label, part in (
        ("decide 2019-2022", frame[frame["season"] <= 2022]),
        ("check 2023-", frame[frame["season"] >= 2023]),
        ("all", frame),
    ):
        by_method = {m: part[part["method"] == m] for m in methods}
        out[label] = _paired_table(by_method, "mean5", cols)
    return out


def run_experiments(
    df: pd.DataFrame,
    settings: backtest.Settings,
    start_season: int,
    tune_season: int,
    only: list[str] | None = None,
    save=None,
) -> dict:
    """Run each experiment in turn; `save(result)` is called after every one, so
    an hour of work isn't lost to a failure in the last. `only` re-runs a subset."""
    tune_window = (tune_season, start_season - 1)
    windows = {
        f"tuning {tune_window[0]}-{tune_window[1]}": tune_window,
        f"test {start_season}-": (start_season, None),
    }
    early = (tune_season - 1, tune_season - 1)
    frames: dict[str, pd.DataFrame] = {}

    def frame(name: str, **kw) -> pd.DataFrame:
        if name not in frames:
            frames[name] = features.build(include_upcoming=False, **kw)
        return frames[name]

    experiments = {
        "ablation": lambda: ablation(df, windows, settings, retrain_every=2),
        "pre_quali_ablation": lambda: pre_quali_ablation(df, windows, settings, retrain_every=2),
        "calibration": lambda: calibration_variants(df, start_season, settings, n_seeds=1, retrain_every=1),
        "form_statistic": lambda: definition_experiment(
            {stat: frame(stat, form_stat=stat) for stat in ("mean", "median")},
            features.FORM_STAT,
            windows,
            settings,
        ),
        "practice": lambda: practice_experiment(df, settings),
        "redundancy": lambda: redundancy(df, start_season, settings),
        "season_projection": lambda: season_projection_experiment(df),
        "parameter_stability": lambda: parameter_stability(
            df, {f"{early[0]}": early, f"{tune_window[0]}-{tune_window[1]}": tune_window}, settings
        ),
    }
    result: dict = {
        "note": (
            "Walk-forward throughout; one XGBoost seed per model (the headline uses five) and, "
            "for the ablation and feature-definition runs, a refit every second race, so that "
            "every arm of an experiment is fitted identically. Differences are race-paired "
            "with 95% bootstrap intervals."
        ),
        "settings": asdict(settings),
    }
    for name, run in experiments.items():
        if only and name not in only:
            continue
        log.info("experiment: %s", name)
        result[name] = run()
        result["provenance"] = provenance.record(feature_version=features.FEATURE_VERSION)
        if save:
            save(result)
    return result


EXPERIMENTS = (
    "ablation",
    "pre_quali_ablation",
    "calibration",
    "form_statistic",
    "practice",
    "redundancy",
    "season_projection",
    "parameter_stability",
)


def render_backtest(result: dict) -> str:
    """The backtest as text, for the terminal."""
    w = result["window"]
    names = baselines.NAMES
    lines = [
        (
            f"Walk-forward {w['start_season']}-, {w['n_races']} races through "
            f"{w['through'][0]} r{w['through'][1]}; settings fitted on {w['tuned_on'][0]}-{w['tuned_on'][1]}"
        ),
        f"settings: {result['settings']}",
        "",
        "post-qualifying (official grid known)",
    ]
    summary = pd.DataFrame(result["summary"]).set_index("method").rename(index=names)
    lines.append(summary.round(3).to_string())
    lines += ["", "model vs grid, paired by race (95% bootstrap interval of the difference)"]
    comp = pd.DataFrame(result["comparison_vs_grid"])
    if not comp.empty:
        lines.append(
            comp[["metric", "model", "grid", "difference", "ci_low", "ci_high", "better"]]
            .round(3)
            .to_string(index=False)
        )
    lines += [
        "",
        f"calibration error (ECE) model {result['calibration_error']}, grid {result['grid_calibration_error']}",
    ]
    lines += ["", "pre-qualifying (grid projected by the qualifying model)"]
    lines.append(
        pd.DataFrame(result["pre_quali"]["summary"])
        .set_index("method")
        .rename(index=names)
        .round(3)
        .to_string()
    )
    lines += ["", "qualifying forecast"]
    lines.append(pd.DataFrame(result["qualifying"]["summary"]).set_index("method").round(3).to_string())
    return "\n".join(lines)
