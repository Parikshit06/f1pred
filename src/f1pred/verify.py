"""End-to-end verification: a release gate, `make verify`.

  1. leakage        does any pre-race feature move when a result is rewritten,
                    or when later races are added?
  2. inputs         which input classes actually reach the model?
  3. contributions  what the model leans on, out of sample (SHAP). Reported,
                    not judged against an expected ordering: which inputs
                    matter is for the model and the ablation to establish
  4. bias           is it favouring particular drivers, once the artefact is removed?
  5. accuracy       what does it add over the starting grid, with intervals?
  6. calibration    when it says 30%, does that happen 30% of the time?
  7. plausibility   is a live forecast one coherent distribution?

Each returns PASS / WARN / FAIL with the evidence attached. `f1pred.cli verify`
exits non-zero on any FAIL.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import backtest, diagnostics, features, model, probability, simulate

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


def audit_truncation(cuts: int = 3) -> Audit:
    """Build features from data that stops at a race, and from everything:
    every feature of every race up to the cut must come out identical."""
    from dataclasses import replace

    raw = features._load()
    done = raw.results[["season", "round"]].drop_duplicates().sort_values(["season", "round"])
    picks = done.iloc[np.linspace(len(done) * 0.4, len(done) - 2, cuts).astype(int)]
    full = features.build(include_upcoming=False)
    cols = sorted(set(features.RACE_FEATURES + features.QUALI_FEATURES))
    keys = ["season", "round", "driver_id"]
    moved: dict[str, int] = {}
    original = features._load
    try:
        for cut in picks.itertuples():
            limit = cut.season * 100 + cut.round

            def keep(t, limit=limit):
                return t if t.empty else t[(t["season"] * 100 + t["round"]) <= limit]

            trimmed = replace(
                raw,
                results=keep(raw.results),
                quali=keep(raw.quali),
                standings=keep(raw.standings),
                pace=keep(raw.pace),
                openf1_grid=keep(raw.openf1_grid),
                openf1_entries=keep(raw.openf1_entries),
            )
            features._load = lambda trimmed=trimmed: trimmed
            early = features.build(include_upcoming=False).set_index(keys)[cols].sort_index()
            late = full.set_index(keys)[cols].loc[early.index]
            diff = ~np.isclose(early.astype(float), late.astype(float), equal_nan=True)
            for c, n in zip(cols, diff.sum(axis=0)):
                if n:
                    moved[c] = moved.get(c, 0) + int(n)
    finally:
        features._load = original
    where = ", ".join(f"{r.season} r{r.round}" for r in picks.itertuples())
    if moved:
        return Audit(
            "1b. Truncation invariance",
            FAIL,
            f"{len(moved)} feature(s) change when later races are added",
            f"Cut after {where}.",
            pd.Series(moved, name="rows_changed").to_frame(),
        )
    return Audit(
        "1b. Truncation invariance",
        PASS,
        f"All {len(cols)} features identical whether later races exist or not",
        f"Cut after {where}; compared every row up to each cut.",
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
        "drv_avg_quali_5",
        "drv_pole_rate_10",
        "team_avg_quali_5",
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
# 3. Contributions
# ---------------------------------------------------------------------------
def audit_contributions(df: pd.DataFrame, n_races: int = 12) -> Audit:
    """Share of the model's attention by input class, out of sample.

    The model is trained on races before the last `n_races`, and SHAP shares
    are averaged over those races. No ordering is expected: this is what the
    model learned. Only an input class with no contribution at all is flagged,
    because that means data isn't reaching it.
    """
    done = sorted(df[df["position"].notna()]["race_seq"].unique())
    held = done[-n_races:]
    ranker = model.train_race(df[df["race_seq"] < held[0]])
    lookup = {f: label for label, cols in INPUT_CLASSES.items() for f in cols}
    shares: dict[str, list[float]] = {}
    for seq in held:
        race = df[df["race_seq"] == seq]
        phi = np.abs(ranker.contributions(race)).sum(axis=0)
        total = phi.sum() or 1.0
        by_class: dict[str, float] = {}
        for f, v in zip(ranker.feature_names, phi):
            by_class[lookup.get(f, "Other")] = by_class.get(lookup.get(f, "Other"), 0.0) + v / total
        for k, v in by_class.items():
            shares.setdefault(k, []).append(v)
    table = (
        pd.Series({k: float(np.mean(v)) * 100 for k, v in shares.items()}, name="share_pct")
        .sort_values(ascending=False)
        .round(1)
        .to_frame()
    )
    dead = [k for k in ("Past results", "Qualifying") if table["share_pct"].get(k, 0.0) < 0.5]
    detail = (
        f"Mean |SHAP| share over the last {len(held)} races, from a model trained before them.\n"
        "Which input matters is established by the ablation in reports/experiments.json;\n"
        "this checks that the learned contributions are live and not concentrated by accident."
    )
    if dead:
        return Audit("3. Contributions", WARN, f"{', '.join(dead)} contribute nothing", detail, table)
    top = table.index[0]
    return Audit("3. Contributions", PASS, f"Largest share: {top} ({table.iloc[0, 0]:.0f}%)", detail, table)


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
def audit_accuracy(df: pd.DataFrame, start_season: int, retrain_every: int) -> tuple[Audit, object]:
    """Against the starting grid, race by race, with bootstrap intervals."""
    res = backtest.walk_forward(
        df, start_season, retrain_every=retrain_every, settings=backtest.load_settings()
    )
    if res.races.empty:
        return Audit("5. Accuracy", FAIL, "backtest produced no model rows"), res
    comp = res.compare("model", "grid").set_index("metric")
    worse = [m for m, r in comp.iterrows() if r["better"] == "grid"]
    better = [m for m, r in comp.iterrows() if r["better"] == "model"]
    table = comp[["model", "grid", "difference", "ci_low", "ci_high", "better"]].round(3)
    detail = (
        f"{int(comp['n_races'].iloc[0])} races, walk-forward, trained only on earlier races.\n"
        "Both sides calibrated the same way; 'better' needs a 95% interval clear of zero."
    )
    if "win_logloss" in worse:
        return Audit("5. Accuracy", FAIL, "The grid alone gives better win probabilities", detail, table), res
    if worse:
        return Audit(
            "5. Accuracy",
            WARN,
            f"Better than the grid on {len(better)} metric(s), worse on {', '.join(worse)}",
            detail,
            table,
        ), res
    return Audit(
        "5. Accuracy", PASS, f"Better than the grid on {len(better)} metric(s), worse on none", detail, table
    ), res


# ---------------------------------------------------------------------------
# 6. Calibration
# ---------------------------------------------------------------------------
def audit_calibration(res) -> Audit:
    """Per-driver reliability of win and podium probabilities.

    Each bucket is judged against its own Wilson interval, because a bucket of
    a dozen races carries a standard error near 0.15 and a raw gap there is
    noise, not miscalibration.
    """
    rel = res.reliability()
    if not rel:
        return Audit("6. Calibration", WARN, "nothing to bucket")
    rows = []
    for event, table in rel.items():
        for r in table.itertuples():
            rows.append(
                {
                    "event": event,
                    "bucket": r.bucket,
                    "stated": round(r.stated, 3),
                    "observed": round(r.observed, 3),
                    "n": int(r.n),
                    "verdict": "ok" if r.ci_low <= r.stated <= r.ci_high else "off",
                }
            )
    table = pd.DataFrame(rows).set_index(["event", "bucket"])
    off = int((table["verdict"] == "off").sum())
    ece = res.calibration_error()
    detail = (
        "Expected calibration error: "
        + ", ".join(f"{k} {v:.3f}" for k, v in ece.items())
        + f"\n{off} of {len(table)} buckets fall outside their own 95% interval; "
        f"about {len(table) * 0.05:.1f} would by chance."
    )
    if off > max(2, len(table) // 4):
        return Audit("6. Calibration", WARN, f"{off} of {len(table)} buckets off", detail, table)
    return Audit(
        "6. Calibration",
        PASS,
        f"{len(table) - off} of {len(table)} buckets within their interval",
        detail,
        table,
    )


# ---------------------------------------------------------------------------
# 7. Plausibility of a live forecast
# ---------------------------------------------------------------------------
def audit_plausibility(df: pd.DataFrame) -> Audit:
    """The most recent race, forecast exactly as live: one coherent distribution."""
    completed = df[df["position"].notna()]
    last_seq = completed["race_seq"].max()
    race = df[df["race_seq"] == last_seq].copy().reset_index(drop=True)
    history = df[df["race_seq"] < last_seq]
    s = backtest.load_settings()

    ranker = backtest.race_trainer(s)(history)
    scores = ranker.score(race)
    fc = simulate.forecast(
        simulate.race_inputs(race, scores, grid_known=True), s.temperature, s.blend_weight, 4000
    )
    t = fc.table
    checks = {
        "a coherent position distribution": not probability.check_distribution(fc.matrix),
        "win probabilities sum to 1": abs(t["p_win"].sum() - 1.0) < 1e-6,
        "podium probabilities sum to 3": abs(t["p_podium"].sum() - 3.0) < 1e-6,
        "win <= podium <= top 5 <= top 10 for every driver": bool(
            (
                (t["p_win"] <= t["p_podium"] + 1e-12)
                & (t["p_podium"] <= t["p_top5"] + 1e-12)
                & (t["p_top5"] <= t["p_top10"] + 1e-12)
            ).all()
        ),
        "no driver at exactly zero": bool((t["p_win"] > 0).all()),
        "favourite below 90% (no false certainty)": float(t["p_win"].max()) < 0.90,
        "favourite above 8% (not a coin toss)": float(t["p_win"].max()) > 0.08,
        "expected positions span the field": float(t["exp_position"].max() - t["exp_position"].min()) > 5.0,
    }
    failed = [k for k, v in checks.items() if not v]
    table = pd.DataFrame(
        [{"check": k, "result": "ok" if v else "FAILED"} for k, v in checks.items()]
    ).set_index("check")
    top = t.sort_values("p_win", ascending=False).head(3)
    shape = ", ".join(f"{r.driver_id} {r.p_win * 100:.0f}%" for r in top.itertuples())
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
def run(start_season: int = 2024, n_sims: int = 3000, retrain_every: int = 3) -> Verification:
    df = features.load()
    v = Verification()
    v.audits.append(audit_leakage(df))
    v.audits.append(audit_truncation())
    v.audits.append(audit_inputs(df))
    v.audits.append(audit_contributions(df))
    v.audits.append(audit_bias(df, start_season, retrain_every))
    accuracy, res = audit_accuracy(df, start_season, retrain_every)
    v.audits.append(accuracy)
    v.audits.append(audit_calibration(res))
    v.audits.append(audit_plausibility(df))
    return v
