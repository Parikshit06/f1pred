"""Checks on the published page.

Two things are worth testing about a report: that the colour ramp is valid
(a line that outruns its axis, a colour nobody can see), and that nothing
internal leaks into a page other people will read.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

from f1pred import report
from f1pred import report_render as rr


def _srgb_to_linear(channel: int) -> float:
    c = channel / 255
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _oklab_lightness(hex_colour: str) -> float:
    r, g, b = (_srgb_to_linear(int(hex_colour[i : i + 2], 16)) for i in (1, 3, 5))
    long_ = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    med = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    short = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return 0.2104542553 * long_ + 0.7936177850 * med - 0.0040720468 * short


def _relative_luminance(hex_colour: str) -> float:
    r, g, b = (_srgb_to_linear(int(hex_colour[i : i + 2], 16)) for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(a: str, b: str) -> float:
    la, lb = _relative_luminance(a), _relative_luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def _token(name: str, block: str = ":root{") -> str:
    """Pull one custom property out of the stylesheet, from a named block."""
    start = rr.CSS.index(block) + len(block)
    body = rr.CSS[start : rr.CSS.index("}", start)]
    match = re.search(rf"--{name}\s*:\s*(#[0-9a-fA-F]{{6}})", body)
    assert match, f"--{name} not defined in {block}"
    return match.group(1)


THEMES = [
    ("light", ":root{"),
    ("dark", ':root[data-theme="dark"]{'),
]


@pytest.mark.parametrize("theme,block", THEMES)
def test_body_text_meets_contrast_in_both_themes(theme, block):
    assert _contrast(_token("paper", block), _token("ink", block)) >= 7.0, theme


@pytest.mark.parametrize("theme,block", THEMES)
def test_secondary_text_still_meets_contrast(theme, block):
    """--ink-2 carries every explanatory paragraph on the page, so it has to
    clear normal-text contrast, not just large-text."""
    assert _contrast(_token("paper", block), _token("ink-2", block)) >= 4.5, theme


@pytest.mark.parametrize("theme,block", THEMES)
def test_accent_is_legible_on_its_own_ground(theme, block):
    assert _contrast(_token("paper", block), _token("accent", block)) >= 4.5, theme


@pytest.mark.parametrize("theme,block", THEMES)
def test_every_token_is_defined_in_every_theme(theme, block):
    """A token defined only in the light block renders as nothing in dark -
    the classic unreadable-artifact bug."""
    base = rr.CSS.index(":root{") + len(":root{")
    light = set(re.findall(r"--([a-z0-9-]+)\s*:", rr.CSS[base : rr.CSS.index("}", base)]))
    start = rr.CSS.index(block) + len(block)
    here = set(re.findall(r"--([a-z0-9-]+)\s*:", rr.CSS[start : rr.CSS.index("}", start)]))
    if theme == "light":
        return
    missing = {t for t in here} - light
    assert not missing, f"{theme} defines tokens the base :root never does: {missing}"


@pytest.mark.parametrize(
    "theme,block,mode", [("light", ":root{", 0), ("dark", ':root[data-theme="dark"]{', 1)]
)
def test_constructor_colours_clear_the_non_text_contrast_floor(theme, block, mode):
    """A livery is the only thing identifying a car before you read the name,
    so every team gets a per-surface variant that clears 3:1. This is the check
    that caught Mercedes petrol at 1.8:1 on white."""
    paper = _token("paper", block)
    weak = {
        team: round(_contrast(paper, pair[mode]), 2)
        for team, pair in rr.TEAM_COLOURS.items()
        if _contrast(paper, pair[mode]) < 3.0
    }
    assert not weak, f"below the non-text floor on {theme}: {weak}"


def test_every_team_has_a_variant_for_each_surface():
    for team, pair in rr.TEAM_COLOURS.items():
        assert len(pair) == 2, team
        assert pair[0] != pair[1], f"{team} uses one value on both surfaces"


def test_team_colours_are_emitted_as_theme_tokens():
    """team_colour returns a var() so the value follows the viewer's theme; a
    literal hex in the markup would be one theme's colour on both."""
    assert rr.team_colour("ferrari") == "var(--t-ferrari)"
    for mode, block in ((0, ":root{"), (1, ':root[data-theme="dark"]{')):
        start = rr.CSS.index(block)
        chunk = rr.CSS[start : rr.CSS.index("}", start)]
        assert f"--t-ferrari:{rr.TEAM_COLOURS['ferrari'][mode]}" in chunk


def test_unknown_constructor_still_gets_a_colour():
    """A new team appears every couple of seasons; no row may render without an
    identifying mark."""
    assert rr.team_colour("brand_new_team") == "var(--t-default)"
    assert rr.team_colour(None) == "var(--t-default)"
    assert "--t-default:" in rr.CSS


@pytest.fixture
def no_track_record(monkeypatch):
    """Isolate the page from whatever is sitting in predictions/ on this
    machine - otherwise the test passes or fails depending on local state."""
    monkeypatch.setattr(report, "scoreboard", lambda: pd.DataFrame())


def test_report_renders_without_data(no_track_record):
    """The page must build before any prediction exists, or the first CI run
    fails on an empty repository."""
    out = report.build(prediction=None, backtest_summary=None, calibration=None)
    assert out.startswith("<!doctype html>")
    assert "No race scheduled" in out
    assert "<h1>" in out


def test_report_escapes_driver_names(no_track_record):
    fake = {
        "race_name": "<script>alert(1)</script>",
        "grid_known": True,
        "race_board": [
            {
                "name": "A & B",
                "team": "t",
                "p_win": 0.5,
                "p_podium": 0.7,
                "p_top5": 0.9,
                "p_top10": 0.95,
                "grid": 3,
                "exp_position": 2.4,
                "why": "",
            }
        ],
        "quali_board": [],
        "field_probs": [],
        "position_matrix": [],
        "meta": {},
    }
    out = report.build(prediction=fake)
    assert "<script>alert(1)</script>" not in out
    assert "A &amp; B" in out


def test_scoreboard_handles_predictions_for_unrun_races(monkeypatch, tmp_path):
    """The normal state right after publishing a forecast: a prediction exists
    but the race has not happened, so nothing is gradeable yet."""
    import json

    from f1pred import config

    pred = {
        "season": 2099,
        "round": 1,
        "race_name": "Future Grand Prix",
        "generated_at_utc": "2099-01-01T00:00:00+00:00",
        "grid_known": False,
        "race_board": [{"driver_id": "someone"}],
        "field_probs": [],
    }
    monkeypatch.setattr(config, "PREDICTIONS", tmp_path)
    (tmp_path / "p.json").write_text(json.dumps(pred))
    monkeypatch.setattr(report.config, "PREDICTIONS", tmp_path)

    board = report.scoreboard()
    assert board.empty, "a race that has not run must not be graded"


SERIES = [
    {"label": "Alpha", "team": "ferrari", "points": {1: 10, 2: 25, 3: 40, 4: 52, 5: 64}},
    {"label": "Beta", "team": "mercedes", "points": {1: 18, 2: 30, 3: 55, 4: 70, 5: 88}},
]


def test_the_chart_labels_every_series():
    """Teammates share a constructor colour and the red/orange pair sits in the
    CVD floor band, so identity may never rest on hue alone."""
    out = rr.progression_chart(SERIES, 3)
    for s in SERIES:
        assert out.count(s["label"]) >= 2, f"{s['label']} is not both labelled and in the legend"


def test_the_chart_separates_what_happened_from_what_is_projected():
    out = rr.progression_chart(SERIES, 3)
    assert "proj" in out, "projected segment is not distinguishable from the actual one"
    assert "class='now'" in out, "nothing marks where the actual data stops"


def test_chart_axis_labels_reach_the_data():
    """Every gridline must name a value the chart spans, and the top of the
    scale must not cut a series off."""
    import re

    out = rr.progression_chart(SERIES, 3)
    ticks = [int(v) for v in re.findall(r"text-anchor='end'>(\d+)</text>", out)]
    assert ticks, "no y-axis labels"
    assert max(ticks) >= max(v for s in SERIES for v in s["points"].values())


def test_chart_survives_empty_input():
    assert rr.progression_chart([{"label": "X", "team": "haas", "points": {}}], 3) == ""
    assert rr.progression_chart([], 3) == ""


def test_chart_accepts_round_numbers_that_came_back_from_json():
    """The page renders from a logged prediction, where dict keys are strings.
    That is the shape that actually reaches the renderer."""
    out = rr.progression_chart([{"label": "A", "team": "ferrari", "points": {"1": 10, "2": 20, "3": 30}}], 2)
    assert "<svg" in out


def test_long_odds_are_never_printed_as_certainty():
    """Nine races out a simulation cannot resolve below 1 in n_sims; printing
    100% would claim a precision the method does not have."""
    assert rr._odds(1.0) == "99.9%"
    assert rr._odds(0.9999) == "99.9%"
    assert rr._odds(0.0) == "0.0%"
    assert rr._odds(0.372) == "37.2%"
    assert "<" not in rr._odds(0.0004) and ">" not in rr._odds(1.0)


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------
def test_wilson_interval_widens_at_small_n():
    """The point of using Wilson: 5/9 and 500/900 are the same proportion but
    nothing like the same evidence, and a calibration audit must not treat
    them alike."""
    from f1pred import verify

    small = verify._wilson(5, 9)
    large = verify._wilson(500, 900)
    assert (small[1] - small[0]) > (large[1] - large[0]) * 3
    assert small[0] < 5 / 9 < small[1]


def test_wilson_handles_degenerate_counts():
    from f1pred import verify

    assert verify._wilson(0, 0) == (0.0, 1.0)
    lo, hi = verify._wilson(0, 10)
    assert lo == 0.0 and 0 < hi < 0.5
    lo, hi = verify._wilson(10, 10)
    assert hi == 1.0 and 0.5 < lo < 1.0


def test_hidden_elements_are_actually_hidden():
    """SVG does not honour the HTML hidden attribute on its own. Without the
    rule, the chart's hover crosshair and dots sit at the viewBox origin and
    render as stray marks in the corner."""
    assert "[hidden]{display:none!important}" in rr.CSS


def test_only_the_probability_cells_animate_their_value():
    """The count-up rewrites textContent, so it must not touch a cell that
    contains a child element - the championship totals carry a range span."""
    board = rr.race_board(
        [
            {
                "name": "A",
                "team": "ferrari",
                "p_win": 0.5,
                "p_podium": 0.8,
                "p_top10": 0.9,
                "grid": 3,
                "driver_id": "a",
            }
        ]
    )
    assert "data-count" in board
    panel = rr.championship_panel(
        "Drivers",
        [
            {
                "name": "A",
                "team": "ferrari",
                "now": 100.0,
                "projected": 200.0,
                "low": 180.0,
                "high": 220.0,
                "p_title": 0.5,
            }
        ],
        "name",
        "team",
    )
    assert "data-count" not in panel, "the count-up would destroy the range span"
    assert "180–220" in panel


def test_links_in_the_inverted_bar_take_the_bar_ink():
    """The top strap inverts the page. An accent-coloured link on it is
    unreadable, which is what shipped the first time."""
    assert ".bar a{color:color-mix(in srgb, var(--paper)" in rr.CSS


def test_the_method_page_builds_and_links_back():
    from f1pred import method_page

    html = method_page.build()
    assert html.startswith("<!doctype html>")
    assert "index.html" in html, "no way back to the forecast"
    assert "Method and accuracy" in html


def test_the_forecast_page_links_to_the_evidence():
    """The forecast page is deliberately thin; if the link to the working is
    missing, the thinness reads as having nothing to show."""
    out = report.build(prediction=None, backtest_summary=None, calibration=None)
    assert "method.html" in out
