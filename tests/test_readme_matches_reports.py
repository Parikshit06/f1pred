"""The README's figures must match the reports/*.json they come from.

The README is written by hand, so nothing updates it when an evaluation
refreshes the evidence. This catches the drift; a failure means the README
needs updating, and the message says to what.
"""

from __future__ import annotations

import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
REPORTS = ROOT / "reports"


def _load(name: str) -> dict:
    path = REPORTS / name
    if not path.exists():
        pytest.skip(f"{name} has not been generated in this checkout")
    return json.loads(path.read_text())


def _readme() -> str:
    if not README.exists():
        pytest.skip("no README in this checkout")
    return README.read_text()


def _summary(method: str, section: str | None = None) -> dict:
    data = _load("backtest.json")
    rows = (data.get(section) or {}).get("summary", []) if section else data.get("summary", [])
    for row in rows:
        if row["method"] == method:
            return row
    pytest.skip(f"no {method} row in backtest.json")
    raise AssertionError  # unreachable, keeps the type checker happy


def _comparison(key: str, section: str | None = None) -> dict:
    data = _load("backtest.json")
    rows = (data.get(section) or {}).get(key, []) if section else data.get(key, [])
    return {r["metric"]: r for r in rows}


# ---------------------------------------------------------------------------
# Race accuracy
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "method,column,fmt",
    [
        ("model", "win_logloss", "{:.3f}"),
        ("grid", "win_logloss", "{:.3f}"),
        ("model", "ndcg3", "{:.3f}"),
        ("grid", "ndcg3", "{:.3f}"),
        ("model", "ndcg5", "{:.3f}"),
        ("grid", "ndcg5", "{:.3f}"),
        ("model", "podium_brier", "{:.3f}"),
        ("championship", "win_logloss", "{:.3f}"),
    ],
)
def test_headline_accuracy_figures_appear_in_the_readme(method, column, fmt):
    value = fmt.format(_summary(method)[column])
    assert value in _readme(), (
        f"backtest.json says {method} {column} is {value}, which is not in the README. "
        "Re-run the evaluation and update the results table."
    )


@pytest.mark.parametrize("method", ["model", "grid", "championship"])
def test_winner_rates_are_current(method):
    rate = f"{_summary(method)['winner_hit'] * 100:.0f}%"
    assert rate in _readme(), f"{method} called {rate} of winners"


def test_the_race_count_is_current():
    n = int(_summary("model")["n_races"])
    assert f"{n} races" in _readme(), f"the backtest covers {n} races; the README says otherwise"


def test_the_log_loss_interval_against_the_grid_is_quoted():
    ll = _comparison("comparison_vs_grid").get("win_logloss")
    if not ll:
        pytest.skip("no paired comparison recorded")
    text = _readme()
    assert f"{abs(ll['difference']):.3f}" in text
    assert f"{ll['ci_low']:.2f} to {ll['ci_high']:+.2f}" in text, (
        "the interval quoted is not the measured one"
    )


def test_the_readme_claims_no_lead_over_the_grid_the_evidence_lacks():
    """The claim that is an interpretation rather than a number, and the one
    most tempting to overstate. It may only say the model beats the grid on a
    metric whose paired interval excludes zero."""
    comp = _comparison("comparison_vs_grid")
    if not comp:
        pytest.skip("no paired comparison recorded")
    text = _readme().lower()
    clear_wins = [m for m, r in comp.items() if r["better"] == "model"]
    clear_losses = [m for m, r in comp.items() if r["better"] == "grid"]
    for phrase in ("beats the grid", "better than the grid", "outperforms the grid"):
        if phrase in text:
            assert clear_wins, f"the README says the model {phrase!r}, but no paired interval supports it"
    if not clear_wins and not clear_losses:
        assert "nothing is statistically clear" in text, "the README no longer admits the grid is level"
    else:
        assert "nothing is statistically clear" not in text, "some intervals exclude zero; say which"


def test_every_clear_loss_to_the_grid_is_admitted():
    """The mirror image: where the grid is ahead with an interval clear of zero,
    the README has to say so and quote the interval."""
    comp = _comparison("comparison_vs_grid")
    losses = {m: r for m, r in comp.items() if r["better"] == "grid"}
    if not losses:
        pytest.skip("the grid is not clearly ahead on any metric")
    text = _readme()
    assert "the grid is clearly better" in text.lower()
    for metric, r in losses.items():
        quoted = f"{r['difference']:+.3f} ({r['ci_low']:+.3f} to {r['ci_high']:+.3f})"
        assert quoted in text, f"{metric}: the grid is ahead by {quoted}, not quoted in the README"


@pytest.mark.parametrize(
    "section,key,metric",
    [
        ("pre_quali", "comparison_vs_championship", "win_logloss"),
        ("qualifying", "comparison", "pole_logloss"),
        ("qualifying", "comparison", "ndcg5"),
    ],
)
def test_the_clear_wins_are_quoted_with_their_intervals(section, key, metric):
    r = _comparison(key, section).get(metric)
    if not r:
        pytest.skip(f"no {section} {metric} comparison")
    quoted = f"{r['difference']:+.3f} ({r['ci_low']:+.3f} to {r['ci_high']:+.3f})"
    assert quoted in _readme(), f"{section} {metric} should read {quoted}"


def test_calibration_errors_are_current():
    ece = _load("backtest.json").get("calibration_error") or {}
    text = _readme()
    for event in ("win", "podium", "top10"):
        if event in ece:
            assert f"{ece[event] * 100:.1f}%" in text, f"{event} calibration error is {ece[event]:.1%}"


# ---------------------------------------------------------------------------
# Championship projection
# ---------------------------------------------------------------------------
def test_title_brier_and_hit_rate_are_current():
    tb = _load("title_backtest.json")
    checkpoints = tb.get("checkpoints") or []
    if not checkpoints:
        pytest.skip("no graded checkpoints")
    text = _readme()
    hit = sum(c["favourite_was_right"] for c in checkpoints)
    assert f"{tb['brier']:.3f}" in text, f"title Brier is {tb['brier']:.3f}"
    assert f"{hit / len(checkpoints):.0%}" in text, f"favourite correct {hit / len(checkpoints):.0%}"
    assert f"{len(checkpoints)} checkpoints" in text, f"{len(checkpoints)} checkpoints were graded"


def test_the_high_confidence_bucket_is_current():
    """n above 95% and its Wilson floor match the report."""
    buckets = _load("title_backtest.json").get("calibration") or []
    over = [b for b in buckets if b["bucket"] == "over 95%"]
    if not over:
        pytest.skip("no over-95% bucket")
    text, b = _readme(), over[0]
    assert f"{b['n']} for {b['n']}" in text, f"the over-95% bucket holds {b['n']} checkpoints"
    assert f"{float(b['ci_low']):.2f}" in text, f"its Wilson floor is {float(b['ci_low']):.2f}"


@pytest.mark.parametrize("arm", ["before", "after"])
def test_band_coverage_is_current(arm):
    held = _load("spread_calibration.json").get("held_out", {})
    if arm not in held:
        pytest.skip(f"no {arm} arm recorded")
    value = f"{held[arm]['coverage']:.0%}"
    assert value in _readme(), (
        f"spread_calibration.json says the {arm} coverage is {value}; the README disagrees"
    )


def test_the_shortfall_is_still_admitted():
    """Coverage is below target. If a future run reaches 80% this test should be
    deleted along with the sentence - but while it is short, the README has to
    keep saying so rather than quietly dropping the admission."""
    held = _load("spread_calibration.json").get("held_out", {})
    if "after" not in held:
        pytest.skip("no graded arm")
    if held["after"]["coverage"] < 0.80:
        assert "short of 80%" in _readme(), (
            f"coverage is {held['after']['coverage']:.0%}, below the stated 80%, "
            "and the README no longer says so"
        )


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def _experiments() -> dict:
    return _load("experiments.json")


def test_the_calibration_comparison_is_current():
    rows = {r["variant"]: r for r in _experiments().get("calibration") or []}
    if not rows:
        pytest.skip("no calibration experiment recorded")
    text = _readme()
    for r in rows.values():
        assert f"{r['win_logloss']:.3f}" in text, f"{r['variant']}: log loss {r['win_logloss']:.3f}"


def test_the_ablation_headline_is_current():
    ab = _experiments().get("ablation") or {}
    tuning = next((rows for w, rows in ab.items() if w.startswith("tuning")), None)
    if not tuning:
        pytest.skip("no ablation recorded")
    rows = {r["variant"]: r for r in tuning}
    text = _readme()
    assert f"{rows['full model']['win_logloss']:.3f}" in text
    grid = rows["grid only"]
    assert f"{grid['win_logloss']:.3f}" in text
    assert f"{grid['win_logloss_ci'][0]:+.3f} to {grid['win_logloss_ci'][1]:+.3f}" in text


def test_the_practice_result_is_current():
    practice = _experiments().get("practice") or {}
    with_practice = next((r for r in practice.get("qualifying") or [] if r.get("pole_logloss_ci")), None)
    if not with_practice:
        pytest.skip("no practice experiment recorded")
    lo, hi = with_practice["pole_logloss_ci"]
    assert f"{with_practice['pole_logloss_diff']:+.3f} ({lo:+.3f} to {hi:+.3f})" in _readme()
    assert f"{practice['weekends_with_practice']} weekends" in _readme()
