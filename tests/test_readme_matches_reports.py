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


def _summary(method: str) -> dict:
    for row in _load("backtest.json").get("summary", []):
        if row["method"] == method:
            return row
    pytest.skip(f"no {method} row in backtest.json")
    raise AssertionError  # unreachable, keeps the type checker happy


# ---------------------------------------------------------------------------
# Race accuracy
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "method,column,fmt",
    [
        ("model", "logloss", "{:.3f}"),
        ("model", "brier", "{:.3f}"),
        ("model", "top5_overlap", "{:.2f}"),
        ("grid", "logloss", "{:.3f}"),
        ("grid", "top5_overlap", "{:.2f}"),
    ],
)
def test_headline_accuracy_figures_appear_in_the_readme(method, column, fmt):
    value = fmt.format(_summary(method)[column])
    assert value in _readme(), (
        f"backtest.json says {method} {column} is {value}, which is not in the README. "
        "Re-run the evaluation and update the accuracy table."
    )


def test_the_race_count_is_current():
    n = int(_summary("model")["n_races"])
    assert f"{n} races" in _readme(), f"the backtest covers {n} races; the README says otherwise"


def test_the_readme_does_not_claim_to_match_the_grid_on_winners_unless_it_does():
    """The one claim that is an interpretation rather than a number, and the one
    that quietly stopped being true when a re-run moved the model from 58% to
    55% while the grid stayed at 58%."""
    model, grid = _summary("model")["top1_hit"], _summary("grid")["top1_hit"]
    claims_parity = "Matches the grid on winners" in _readme()
    if claims_parity:
        assert abs(model - grid) < 0.005, (
            f"the README claims parity on winners but the model is {model:.1%} against the grid's {grid:.1%}"
        )


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
# Driver bias
# ---------------------------------------------------------------------------
def test_driver_bias_figures_are_current():
    bias = _load("bias.json")
    text = _readme()
    assert f"{bias['r']:.3f}" in text, f"bias correlation is {bias['r']:.3f}"
    assert f"{bias['variance_explained']:.0%}" in text, f"artefact share is {bias['variance_explained']:.0%}"
    named = {d["driver_id"]: d["driver_specific_bias"] for d in bias["drivers"]}
    for driver, shown in (
        ("hamilton", "+{:.2f}"),
        ("max_verstappen", "+{:.2f}"),
        ("tsunoda", "\u2212{:.2f}"),
    ):
        value = named.get(driver)
        if value is not None:
            assert shown.format(abs(value)) in text, f"{driver} is {value:+.2f}"


# ---------------------------------------------------------------------------
# Counts the README asserts about the repo itself
# ---------------------------------------------------------------------------
def test_the_test_count_is_plausible():
    """Deliberately loose. Collecting the suite from inside it is circular, and
    a defined `def test_` can expand into several cases through parametrise - 140
    definitions currently collect as 164. So this only catches the count being
    stale downwards, which is the direction it drifts: tests get added and the
    number in the README does not.
    """
    import re

    text = _readme()
    claimed = re.search(r"\((\d+) tests\)", text)
    if not claimed:
        pytest.skip("the README does not state a test count")
    defined = sum(
        len(re.findall(r"^def test_", p.read_text(), re.MULTILINE))
        for p in (ROOT / "tests").glob("test_*.py")
    )
    if not defined:
        pytest.skip("no test files found from here")
    stated = int(claimed.group(1))
    assert stated >= defined, (
        f"the README claims {stated} tests but {defined} are defined before parametrisation, "
        "so the figure is stale"
    )
