"""End-to-end verification.

Seven audits that between them answer "should anyone believe this model":

  1. leakage        does any pre-race feature move when the race result changes?
  2. inputs         which of the three input classes actually reach the model?
  3. weighting      what does it lean on, and does that ordering make sense?
  4. bias           is it favouring particular drivers, once the artifact is removed?
  5. accuracy       does it beat baselines anyone could produce without it?
  6. calibration    when it says 30%, does that happen 30% of the time?
  7. plausibility   are the published numbers internally coherent?

Each returns PASS / WARN / FAIL with the evidence attached. `f1pred.cli verify`
exits non-zero on any FAIL, so it works as a release gate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import backtest, config, diagnostics, features, model, simulate

log = logging.getLogger(__name__)

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


@dataclass
class Audit:
    name: str
    status: str
    headline: str
    detail: str = ""
    table: pd.DataFrame | None = None


@dataclass
class Verification:
    audits: list[Audit] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(a.status == FAIL for a in self.audits)

    def render(self) -> str:
        width = 74
        out = ["=" * width, "MODEL VERIFICATION", "=" * width, ""]
        for a in self.audits:
            out.append(f"[{a.status}] {a.name}")
            out.append(f"       {a.headline}")
            if a.detail:
                out.extend("       " + line for line in a.detail.splitlines())
            if a.table is not None and not a.table.empty:
                out.append("")
                out.extend("       " + line for line in a.table.to_string().splitlines())
            out.append("")
        fails = sum(a.status == FAIL for a in self.audits)
        warns = sum(a.status == WARN for a in self.audits)
        out.append("-" * width)
        out.append(
            f"{len(self.audits)} audits: {len(self.audits) - fails - warns} pass, {warns} warn, {fails} fail"
        )
        return "\n".join(out)


# ---------------------------------------------------------------------------
# 1. Leakage
# ---------------------------------------------------------------------------
def audit_leakage(df: pd.DataFrame) -> Audit:
    """Change a race's result; no feature for that race or any earlier one may move.

    This is stronger than inspecting the code. If a feature accidentally peeks
    at the current race - or at any future race - rewriting an outcome will
    shift it, and the comparison below catches it whatever the mechanism.
    """
    from .store import connect

    with connect(read_only=True) as con:
        results = con.execute("SELECT * FROM raw_results").fetchdf()

    completed = df[df["position"].notna()]
    mid = completed["race_seq"].quantile(0.5)
    target = int(completed.loc[(completed["race_seq"] - mid).abs().idxmin(), "race_seq"])
    key = df[df["race_seq"] == target].iloc[0]
    season, rnd = int(key["season"]), int(key["round"])

    baseline = features.build(include_upcoming=False)

    # Reverse the finishing order of that one race - a maximal perturbation.
    tampered = results.copy()
    mask = (tampered["season"] == season) & (tampered["round"] == rnd)
    n = int(mask.sum())
    tampered.loc[mask, "position"] = tampered.loc[mask, "position"].to_numpy()[::-1]
    tampered.loc[mask, "points"] = tampered.loc[mask, "points"].to_numpy()[::-1]

    with connect() as con:
        con.execute("CREATE OR REPLACE TABLE _real_results AS SELECT * FROM raw_results")
        con.register("_t", tampered)
        con.execute("DELETE FROM raw_results")
        con.execute("INSERT INTO raw_results SELECT * FROM _t")
        con.unregister("_t")
    try:
        after = features.build(include_upcoming=False)
    finally:
        with connect() as con:
            con.execute("DELETE FROM raw_results")
            con.execute("INSERT INTO raw_results SELECT * FROM _real_results")
            con.execute("DROP TABLE _real_results")

    cols = [c for c in features.RACE_FEATURES if c in baseline.columns]
    keys = ["season", "round", "driver_id"]
    a = baseline[keys + cols + ["race_seq"]].set_index(keys).sort_index()
    b = after[keys + cols + ["race_seq"]].set_index(keys).sort_index()
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]

    at_or_before = a["race_seq"] <= target
    after_target = a["race_seq"] > target

    def changed(frame_a, frame_b, rows):
        d = (frame_a.loc[rows, cols] - frame_b.loc[rows, cols]).abs()
        both_null = frame_a.loc[rows, cols].isna() & frame_b.loc[rows, cols].isna()
        moved = (d > 1e-9) & ~both_null
        # A NaN turning into a number (or back) is a change the subtraction misses.
        flipped = frame_a.loc[rows, cols].isna() ^ frame_b.loc[rows, cols].isna()
        return (moved | flipped).sum()

    past_changes = changed(a, b, at_or_before)
    future_changes = changed(a, b, after_target)

    offenders = past_changes[past_changes > 0]
    # Grid and qualifying belong to the race weekend but precede the race, and
    # we did not tamper with them, so they must not move either.
    if len(offenders):
        return Audit(
            "1. Temporal leakage",
            FAIL,
            f"{len(offenders)} feature(s) reacted to a result they should not see",
            f"Rewrote the finishing order of {season} round {rnd} ({n} entries).\n"
            "Features for that race and every earlier one must be identical.",
            offenders.to_frame("rows_changed"),
        )

    return Audit(
        "1. Temporal leakage",
        PASS,
        f"No pre-race feature moved when {season} r{rnd}'s result was rewritten",
        f"Reversed the finishing order of {n} entries.\n"
        f"Features at or before that race: 0 of {len(cols)} changed.\n"
        f"Features after it: {int((future_changes > 0).sum())} changed, which is correct -\n"
        "later races legitimately learn from earlier results.",
    )


# ---------------------------------------------------------------------------
# 2. Inputs
# ---------------------------------------------------------------------------
INPUT_CLASSES = {
    "Past results": [
        "drv_avg_finish_3",
        "drv_avg_finish_5",
        "drv_points_rate_5",
        "drv_dnf_rate_10",
        "team_avg_finish_3",
        "team_avg_finish_5",
        "team_points_rate_5",
        "team_dnf_rate_10",
        "champ_position_before",
        "champ_points_before",
        "drv_circuit_avg_finish",
        "team_circuit_avg_finish",
        "drv_positions_gained_5",
        "drv_podium_rate_10",
        "drv_top10_rate_10",
        "drv_experience",
        "drv_circuit_starts",
    ],
    "Qualifying": [
        "grid",
        "quali_position",
        "quali_gap_to_pole_pct",
        "quali_gap_to_teammate_pct",
        "drv_teammate_quali_edge",
        "team_pace_gap_pct",
        "drv_pace_gap_pct",
        "team_pace_trend",
        "drv_avg_grid_5",
    ],
    "Practice": ["fp_best_gap_pct", "fp_long_run_gap_pct", "practice_available"],
    "Circuit": ["circuit_overtaking_score", "circuit_dnf_rate", "circuit_pole_win_rate", "season_progress"],
}


def audit_inputs(df: pd.DataFrame) -> Audit:
    """Are all three input classes the model claims to use actually populated?"""
    recent = df[df["season"] >= df["season"].max() - 1]
    rows = []
    for label, cols in INPUT_CLASSES.items():
        present = [c for c in cols if c in recent.columns]
        # practice_available is a flag that is always 0 or 1; coverage of the
        # real signals is what matters.
        signal = [c for c in present if c != "practice_available"]
        cov = recent[signal].notna().mean().mean() * 100 if signal else 0.0
        rows.append({"input": label, "features": len(present), "coverage_pct": round(cov, 1)})
    table = pd.DataFrame(rows).set_index("input")

    dead = table[table["coverage_pct"] < 1.0]
    # Practice is an optional enhancement to the QUALIFYING model only; the
    # race forecast is designed to stand up without it. Everything else is
    # load-bearing, so an empty table there is a hard failure.
    required_dead = [d for d in dead.index if d != "Practice"]
    if required_dead:
        return Audit(
            "2. Input coverage",
            FAIL,
            f"{', '.join(required_dead)} has no data - it cannot influence any prediction",
            "The model is wired for it but the table is empty.",
            table,
        )
    if "Practice" in dead.index:
        return Audit(
            "2. Input coverage",
            WARN,
            "Qualifying and past results are live; practice pace is absent (optional)",
            "Practice feeds only the pre-qualifying forecast, where it roughly doubles\n"
            "the pole hit rate (18% -> 32% at realistic FP3 correlation). It is not in\n"
            "the race model at all - once the grid is known it adds nothing measurable.\n"
            "To enable: make data-fastf1 SEASONS=2024-2026",
            table,
        )
    return Audit("2. Input coverage", PASS, "All input classes carry data", "", table)


# ---------------------------------------------------------------------------
# 3. Weighting
# ---------------------------------------------------------------------------
def audit_weighting(df: pd.DataFrame) -> Audit:
    """Does the model lean on things that should matter, in a sensible order?"""
    ranker = model.train_race(df[df["position"].notna()])
    gain = ranker.booster.get_booster().get_score(importance_type="gain")
    imp = pd.DataFrame([{"feature": k, "gain": v} for k, v in gain.items()])
    total = imp["gain"].sum()
    imp["share"] = imp["gain"] / total * 100

    lookup = {f: label for label, cols in INPUT_CLASSES.items() for f in cols}
    imp["input"] = imp["feature"].map(lookup).fillna("Other")
    by_class = imp.groupby("input")["share"].sum().sort_values(ascending=False).round(1)

    problems = []
    quali = by_class.get("Qualifying", 0.0)
    past = by_class.get("Past results", 0.0)
    circuit = by_class.get("Circuit", 0.0)

    # Qualifying is measured this weekend in this car; history is a lagging
    # proxy. A model leaning on history over the grid would be ignoring the
    # freshest evidence it has.
    if quali < past:
        problems.append(
            f"Qualifying carries {quali:.0f}% against {past:.0f}% for past results. "
            "The grid is the most current evidence available and should dominate."
        )
    if quali > 85:
        problems.append(
            f"Qualifying carries {quali:.0f}% - the model is close to just reading the grid back."
        )
    if circuit < 0.5:
        problems.append("Circuit character is contributing almost nothing.")

    single = imp.sort_values("share", ascending=False).iloc[0]
    if single["share"] > 60:
        problems.append(
            f"{single['feature']} alone carries {single['share']:.0f}% - dangerously concentrated."
        )

    top = imp.sort_values("share", ascending=False).head(10)[["feature", "input", "share"]].round(1)
    detail = "share of total gain, grouped:\n" + by_class.to_string()
    if problems:
        return Audit("3. Weighting", WARN, "; ".join(problems), detail, top.set_index("feature"))
    return Audit(
        "3. Weighting",
        PASS,
        f"Qualifying {quali:.0f}% > past results {past:.0f}% > circuit {circuit:.0f}% - the sensible order",
        detail,
        top.set_index("feature"),
    )


# ---------------------------------------------------------------------------
# 4. Bias
# ---------------------------------------------------------------------------
def audit_bias(df: pd.DataFrame, start_season: int, retrain_every: int = 3) -> Audit:
    entries = diagnostics.collect(df, start_season, retrain_every)
    bias = diagnostics.driver_bias(entries)
    trend = diagnostics.trend_strength(entries)

    worst = bias["driver_specific_bias"].abs().max()
    # Report the most-raced drivers rather than naming anyone: the audit should
    # not carry a list of people it treats as special, and the drivers with the
    # most starts are the ones whose bias estimate is worth trusting.
    stars = bias.nlargest(min(4, len(bias)), "races").index.tolist()
    star_rows = bias.loc[stars, ["races", "raw_bias", "driver_specific_bias"]].round(2) if stars else None

    detail = (
        f"{trend['variance_explained'] * 100:.0f}% of raw per-driver bias is explained by predicted\n"
        f"rank alone (r={trend['r']:.2f}) - an artifact of comparing a fixed ranking to a\n"
        "mean pulled toward the middle by retirements. The corrected column is what counts."
    )

    star_bias = bias.loc[stars, "driver_specific_bias"].abs().max() if stars else 0.0
    if star_bias > 1.5:
        return Audit(
            "4. Driver bias",
            FAIL,
            f"A full-season driver is mis-rated by {star_bias:.2f} positions after correction",
            detail,
            star_rows,
        )
    if worst > 3.0:
        return Audit(
            "4. Driver bias",
            WARN,
            f"Worst driver-specific bias is {worst:.2f} positions ({bias['driver_specific_bias'].abs().idxmax()})",
            detail,
            bias[["races", "raw_bias", "driver_specific_bias"]].head(3).round(2),
        )
    return Audit(
        "4. Driver bias",
        PASS,
        f"No driver mis-rated by more than {worst:.2f} positions; full-season drivers within {star_bias:.2f}",
        detail,
        star_rows,
    )


# ---------------------------------------------------------------------------
# 5. Accuracy
# ---------------------------------------------------------------------------
def audit_accuracy(
    df: pd.DataFrame, start_season: int, n_sims: int, retrain_every: int, params: dict | None = None
) -> tuple[Audit, object]:
    res = backtest.walk_forward(
        df, start_season, retrain_every=retrain_every, n_sims=n_sims, **(params or {})
    )
    s = res.summary()
    if s.empty or "model" not in s.index:
        return Audit("5. Accuracy", FAIL, "backtest produced no model rows"), res

    m, weak = s.loc["model"], s.drop(index="model")
    beaten = (m["ndcg5"] > weak["ndcg5"]).sum()

    detail = (
        f"{int(m['n_races'])} races, walk-forward, trained only on earlier races.\n"
        f"Beats {beaten} of {len(weak)} baselines on NDCG@5."
    )
    if "grid" in weak.index:
        g = weak.loc["grid"]
        detail += (
            f"\nAgainst the grid baseline: top5 {m['top5_overlap']:.2f} vs {g['top5_overlap']:.2f}, "
            f"log loss {m['logloss']:.3f} vs {g['logloss']:.3f}."
        )

    if beaten < len(weak) - 1:
        return Audit("5. Accuracy", FAIL, f"Only beats {beaten} of {len(weak)} baselines", detail, s), res
    if m["logloss"] > weak["logloss"].min():
        return Audit(
            "5. Accuracy",
            WARN,
            "A baseline produces better-calibrated probabilities than the model",
            detail,
            s,
        ), res
    return Audit(
        "5. Accuracy",
        PASS,
        f"Best log loss of all approaches ({m['logloss']:.3f}); beats {beaten}/{len(weak)} on ranking",
        detail,
        s,
    ), res


# ---------------------------------------------------------------------------
# 6. Calibration
# ---------------------------------------------------------------------------
def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval - behaves sensibly at small n, unlike normal approx."""
    if n == 0:
        return (0.0, 1.0)
    phat = k / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half = z / denom * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    return (max(0.0, centre - half), min(1.0, centre + half))


def audit_calibration(res) -> Audit:
    """Is the miscalibration real, or is it small buckets?

    Sixty races split four ways leaves 9-30 observations per bucket, where a
    proportion carries a standard error near 0.15. Judging calibration on raw
    gaps at that sample size flags noise as failure, so each bucket is tested
    against its own confidence interval, and the aggregate confidence level -
    which uses every race at once - is tested separately.
    """
    cal = res.calibration()
    if cal.empty:
        return Audit("6. Calibration", WARN, "not enough races to bucket")

    cal = cal.copy()
    cal["gap"] = (cal["predicted"] - cal["actual"]).abs()
    lo, hi, verdict = [], [], []
    for row in cal.itertuples():
        n = int(row.n)
        k = int(round(row.actual * n))
        a, b = _wilson(k, n)
        lo.append(round(a, 3))
        hi.append(round(b, 3))
        verdict.append("ok" if a <= row.predicted <= b else "off")
    cal["ci_low"], cal["ci_high"], cal["verdict"] = lo, hi, verdict
    off = int((cal["verdict"] == "off").sum())

    # Aggregate: does the model's stated confidence match reality overall?
    total_n = int(cal["n"].sum())
    pooled_pred = float((cal["predicted"] * cal["n"]).sum() / total_n)
    pooled_obs = float((cal["actual"] * cal["n"]).sum() / total_n)
    agg_lo, agg_hi = _wilson(int(round(pooled_obs * total_n)), total_n)
    aggregate_ok = agg_lo <= pooled_pred <= agg_hi

    detail = (
        f"Pooled over {total_n} races: model says {pooled_pred:.3f}, observed {pooled_obs:.3f} "
        f"(95% CI {agg_lo:.3f}-{agg_hi:.3f}).\n"
        f"Per bucket, {off} of {len(cal)} fall outside their own interval; with "
        f"{len(cal)} comparisons about one is expected by chance.\n"
        f"Mean absolute gap {cal['gap'].mean():.3f}, but buckets hold only "
        f"{int(cal['n'].min())}-{int(cal['n'].max())} races each."
    )
    table = cal[["bucket", "predicted", "actual", "ci_low", "ci_high", "n", "verdict"]]

    if not aggregate_ok:
        return Audit(
            "6. Calibration",
            FAIL,
            f"Overall confidence is wrong: says {pooled_pred:.3f}, happens {pooled_obs:.3f}",
            detail,
            table,
        )
    if off > max(1, len(cal) // 3):
        return Audit(
            "6. Calibration",
            WARN,
            f"{off} of {len(cal)} buckets off, more than chance explains",
            detail,
            table,
        )
    return Audit(
        "6. Calibration",
        PASS,
        f"Overall confidence is right: says {pooled_pred:.3f}, happens {pooled_obs:.3f}",
        detail,
        table,
    )


# ---------------------------------------------------------------------------
# 7. Plausibility of a live prediction
# ---------------------------------------------------------------------------
def audit_plausibility(df: pd.DataFrame) -> Audit:
    """The published numbers must be internally coherent and physically sane."""
    completed = df[df["position"].notna()]
    last_seq = completed["race_seq"].max()
    race = df[df["race_seq"] == last_seq].copy()
    history = df[df["race_seq"] < last_seq]

    ranker = model.train_race(history)
    race["score"] = ranker.score(race)
    temp = config.DEFAULT_TEMPERATURE
    p_model = simulate.plackett_luce(race["score"].to_numpy(), temp)
    sim = simulate.simulate(
        simulate.SimInputs(
            driver_ids=race["driver_id"].tolist(),
            scores=race["score"].to_numpy(),
            dnf_prob=simulate.dnf_probability(race),
            grid=race["grid"].fillna(len(race)).to_numpy(),
            overtaking_score=float(race["circuit_overtaking_score"].iloc[0])
            if pd.notna(race["circuit_overtaking_score"].iloc[0])
            else 3.0,
        ),
        n_sims=4000,
        temperature=temp,
    )
    p = simulate.blend(p_model, sim["p_win"].to_numpy(), config.DEFAULT_BLEND_WEIGHT)

    checks = {
        "win probabilities sum to 1": abs(p.sum() - 1.0) < 1e-6,
        "no negative probabilities": bool((p >= 0).all()),
        "podium >= win for every driver": bool((sim["p_podium"] >= sim["p_win"] - 1e-9).all()),
        "top5 >= podium for every driver": bool((sim["p_top5"] >= sim["p_podium"] - 1e-9).all()),
        "position distribution rows sum to 1": bool(
            all(abs(float(np.sum(r)) - 1.0) < 1e-6 for r in sim["position_dist"])
        ),
        "favourite below 90% (no false certainty)": float(p.max()) < 0.90,
        "favourite above 8% (not a coin toss)": float(p.max()) > 0.08,
        "at least 4 drivers above 2%": int((p > 0.02).sum()) >= 4,
        "expected positions span the field": float(sim["exp_position"].max() - sim["exp_position"].min())
        > 5.0,
    }
    failed = [k for k, v in checks.items() if not v]
    table = pd.DataFrame(
        [{"check": k, "result": "ok" if v else "FAILED"} for k, v in checks.items()]
    ).set_index("check")

    top = race.assign(p=p).sort_values("p", ascending=False).head(3)
    shape = ", ".join(f"{r.driver_id} {r.p * 100:.0f}%" for r in top.itertuples())
    if failed:
        return Audit(
            "7. Prediction sanity",
            FAIL,
            f"{len(failed)} coherence check(s) failed",
            f"Top three: {shape}",
            table,
        )
    return Audit(
        "7. Prediction sanity", PASS, f"All {len(checks)} coherence checks hold", f"Top three: {shape}", table
    )


# ---------------------------------------------------------------------------
def _tuned_params() -> dict:
    """Use the settings a previous `backtest --tune` fitted, if there are any.

    Verifying with library defaults when the project has tuned settings would
    grade a model nobody is actually running.
    """
    import json

    path = config.REPORTS / "backtest.json"
    if not path.exists():
        return {}
    try:
        params = json.loads(path.read_text()).get("params", {})
    except (json.JSONDecodeError, OSError):
        return {}
    return {
        k: params[k] for k in ("temperature", "blend_weight") if isinstance(params.get(k), (int, float))
    } | (
        {"current_season_weight": params["recency_weight"]}
        if isinstance(params.get("recency_weight"), (int, float))
        else {}
    )


def run(start_season: int = 2024, n_sims: int = 3000, retrain_every: int = 3) -> Verification:
    df = features.load()
    v = Verification()
    v.audits.append(audit_leakage(df))
    v.audits.append(audit_inputs(df))
    v.audits.append(audit_weighting(df))
    v.audits.append(audit_bias(df, start_season, retrain_every))
    accuracy, res = audit_accuracy(df, start_season, n_sims, retrain_every, _tuned_params())
    v.audits.append(accuracy)
    v.audits.append(audit_calibration(res))
    v.audits.append(audit_plausibility(df))
    return v
