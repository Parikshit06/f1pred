"""Checks on the published page.

Two things are worth testing about a report: that the colour ramp is valid
(a line that outruns its axis, a colour nobody can see), and that nothing
internal leaks into a page other people will read.
"""

from __future__ import annotations

import re

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
    monkeypatch.setattr(report, "race_history", list)


def test_report_renders_without_data(no_track_record):
    """The page must build before any prediction exists, or the first CI run
    fails on an empty repository."""
    out = report.build(prediction=None)
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

    assert report.history_of([pred], lambda s, r: {}, {}) == [], "a race that has not run must not be graded"


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


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------
def test_wilson_interval_widens_at_small_n():
    """The point of using Wilson: 5/9 and 500/900 are the same proportion but
    nothing like the same evidence, and a calibration audit must not treat
    them alike."""
    from f1pred import metrics

    small = metrics.wilson(5, 9)
    large = metrics.wilson(500, 900)
    assert (small[1] - small[0]) > (large[1] - large[0]) * 3
    assert small[0] < 5 / 9 < small[1]


def test_wilson_handles_degenerate_counts():
    from f1pred import metrics

    assert metrics.wilson(0, 0) == (0.0, 1.0)
    lo, hi = metrics.wilson(0, 10)
    assert lo == 0.0 and 0 < hi < 0.5
    lo, hi = metrics.wilson(10, 10)
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
    """The top strap has its own surface, so its link uses the strap's ink, not the accent
    (unreadable on the strap) and not a transparent mix (which resolved close
    to the strap's own background)."""
    rule = rr.CSS.split(".bar a{")[1].split("}")[0]
    colour = next(d for d in rule.split(";") if d.strip().startswith("color:"))
    assert "var(--bar-ink)" in colour, colour
    assert "color-mix" not in colour, "a transparent text colour resolved near the strap itself"


def test_the_method_page_builds_and_links_back():
    from f1pred import method_page

    html = method_page.build()
    assert html.startswith("<!doctype html>")
    assert "index.html" in html, "no way back to the forecast"
    assert "href='method.html' aria-current='page'>Method" in html


def test_the_forecast_page_links_to_the_evidence():
    """The forecast page is deliberately thin; if the link to the working is
    missing, the thinness reads as having nothing to show."""
    out = report.build(prediction=None)
    assert "method.html" in out


@pytest.mark.parametrize("theme,block", THEMES)
def test_tertiary_ink_clears_aa_on_both_surfaces(theme, block):
    """--ink-3 carries every column header, team name and range on the page, so
    it is body text and must clear 4.5:1 — on the tinted band as well as on the
    paper, because alternate sections sit on the band."""
    ink3 = _token("ink-3", block)
    for surface in ("paper", "band"):
        ratio = _contrast(_token(surface, block), ink3)
        assert ratio >= 4.5, f"{theme} --ink-3 on --{surface} is {ratio:.2f}"


@pytest.mark.parametrize("theme,block", THEMES)
def test_the_band_surface_is_defined(theme, block):
    """Alternate sections paint --band full-bleed; if it is missing they render
    on the page ground and the rhythm disappears."""
    assert _token("band", block)


# ---------------------------------------------------------------------------
# The alternating band
# ---------------------------------------------------------------------------
def _lstar(hex_colour: str) -> float:
    """CIE L*, which tracks perceived lightness where a raw RGB average does not."""
    r, g, b = (int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5))

    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    y = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
    return 116 * (y ** (1 / 3) if y > 0.008856 else 7.787 * y + 16 / 116) - 16


@pytest.mark.parametrize("theme,block", THEMES)
def test_the_band_is_a_second_ink_pass_not_a_grey_box(theme, block):
    """Alternate sections sit on a slightly darker band. Under ~1 L* it disappears;
    over ~3 it reads as a grey box rather than the same sheet.
    """
    paper = _token("paper", block)
    band = _token("band", block)
    step = abs(_lstar(paper) - _lstar(band))
    assert 1.0 <= step <= 3.0, f"{theme}: paper-to-band step is {step:.1f} in L*"


@pytest.mark.parametrize("theme,block", THEMES)
def test_the_band_is_the_same_paper_not_a_different_colour(theme, block):
    """A band that drifts in hue reads as dingy against the paper rather than
    as the same stock. Both surfaces must lean the same way."""
    paper, band = _token("paper", block), _token("band", block)

    def warmth(c):  # red minus blue: positive is warm, negative is cool
        return int(c[1:3], 16) - int(c[5:7], 16)

    assert (warmth(paper) >= 0) == (warmth(band) >= 0), (
        f"{theme}: paper {paper} and band {band} lean opposite ways"
    )
    assert abs(warmth(paper) - warmth(band)) <= 6, f"{theme}: hue drifts between paper and band"


@pytest.mark.parametrize("theme,block", THEMES)
def test_inline_code_still_reads_against_its_own_surface(theme, block):
    """Code chips have their own token, distinct from both paper and band."""
    chip = _token("chip", block)
    for surface in ("paper", "band"):
        step = abs(_lstar(chip) - _lstar(_token(surface, block)))
        assert step >= 1.0, f"{theme}: code chip is invisible on --{surface}"


def test_the_page_script_is_a_raw_string():
    """It contains regex escapes. As a plain string those are invalid escape
    sequences - a DeprecationWarning today and a SyntaxError in a later Python."""
    import importlib
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        importlib.reload(rr)
    assert r"[\d.]+" in rr.SCRIPT, "the escape was 'fixed' by removing the regex"


# ---------------------------------------------------------------------------
# A clean checkout has no database
# ---------------------------------------------------------------------------
def test_both_pages_build_with_no_database(tmp_path, monkeypatch):
    """A fresh clone - and CI - has no database. The two readers that only
    decorate the page (track record, driver surnames) must fall back to empty.
    """
    from f1pred import config, method_page, store

    monkeypatch.setattr(config, "DB_PATH", tmp_path / "absent.duckdb")
    assert not store.database_exists()

    assert report.race_history() == [], "a missing database invented a track record"
    assert "method.html" in report.build(prediction=None)
    assert "How the forecast works" in method_page.build()


def test_a_missing_database_is_not_confused_with_an_empty_one(tmp_path, monkeypatch):
    """If the file exists the readers must go through to it, so that a genuinely
    empty database still surfaces as an error rather than a silent blank page."""
    import duckdb

    from f1pred import config, store

    db = tmp_path / "present.duckdb"
    duckdb.connect(str(db)).close()
    monkeypatch.setattr(config, "DB_PATH", db)
    assert store.database_exists(), "an existing database was treated as absent"


# ---------------------------------------------------------------------------
# Figures quoted in prose have to come from the data beside them
# ---------------------------------------------------------------------------
def test_the_method_page_sets_the_winner_rate_against_the_grid(tmp_path, monkeypatch):
    """The headline rate always comes with the pole-sitter's, and with the
    admission that the gap could be luck when the interval says so."""
    import json

    from f1pred import config, method_page

    monkeypatch.setattr(config, "REPORTS", tmp_path)
    (tmp_path / "backtest.json").write_text(
        json.dumps(
            {
                "window": {"n_races": 63, "start_season": 2024},
                "summary": [
                    {"method": "model", "winner_hit": 0.667},
                    {"method": "grid", "winner_hit": 0.587},
                ],
                "comparison_vs_grid": [{"metric": "winner_hit", "better": "unclear"}],
            }
        )
    )
    html = method_page.build()
    assert "67% of races" in html and "won 59%" in html
    assert "could be luck" in html


def test_the_title_claim_names_the_leader_it_mostly_follows(tmp_path, monkeypatch):
    """A title favourite that is nearly always the points leader is a low bar;
    the page has to say so, with the leader's own record."""
    import json

    from f1pred import config, method_page

    monkeypatch.setattr(config, "REPORTS", tmp_path)
    (tmp_path / "backtest.json").write_text(json.dumps({"reliability": {}}))
    cp = {"favourite": "a", "leader": "a", "favourite_was_right": 1}
    (tmp_path / "title_backtest.json").write_text(
        json.dumps({"checkpoints": [{**cp, "leader_was_champion": 1}, {**cp, "leader_was_champion": 0}]})
    )
    html = method_page.build()
    assert "win 100% of the time" in html
    assert "already leading the championship (2 times in 2)" in html
    assert "right 50% of the time" in html


def test_the_calibration_section_is_drawn_from_the_report(tmp_path, monkeypatch):
    """Reliability charts and the calibration error come off backtest.json,
    so the page can't show a calibration nobody measured."""
    import json

    from f1pred import config, method_page

    monkeypatch.setattr(config, "REPORTS", tmp_path)
    bucket = {
        "bucket": "(0.2, 0.35]",
        "stated": 0.27,
        "observed": 0.31,
        "n": 40,
        "ci_low": 0.19,
        "ci_high": 0.46,
    }
    (tmp_path / "backtest.json").write_text(
        json.dumps(
            {
                "summary": [],
                "reliability": {"win": [bucket], "podium": [bucket]},
                "calibration_error": {"win": 0.0123, "podium": 0.0456},
            }
        )
    )
    html = method_page.build()
    assert html.count("class='cpt") == 2, "one point per band with enough cases, per event"
    assert "Said 27%, happened 31% (40 cases" in html
    assert "class='diag'" in html, "no perfect-calibration reference line"


def test_an_old_forecast_without_the_new_fields_still_renders():
    """Logged forecasts are never rewritten, so the page must read every vintage."""
    old = {
        "season": 2026,
        "round": 1,
        "race_name": "Old Grand Prix",
        "circuit_id": "old",
        "grid_known": False,
        "race_board": [
            {
                "name": "A Driver",
                "short": "Driver",
                "team": "ferrari",
                "p_win": 0.3,
                "p_podium": 0.6,
                "p_top5": 0.8,
                "p_top10": 0.95,
                "exp_position": 3.2,
                "grid": None,
            }
        ],
        "quali_board": [],
        "field_probs": [],
    }
    html = report.build(prediction=old)
    assert "Old Grand Prix" in html and "30.0%" in html


def test_a_probability_too_small_to_print_is_not_shown_as_zero():
    assert rr.pct(0.0) == "&lt;0.1%"
    assert rr.pct(0.0004) == "&lt;0.1%"
    assert rr.pct(0.0006) == "0.1%"
    assert rr.pct(0.004, 0) == "&lt;1%"
    assert rr.pct(0.542) == "54.2%"


def test_a_probability_short_of_certain_is_not_shown_as_100():
    assert rr.pct(0.998, 0) == "&gt;99%"
    assert rr.pct(0.9996) == "&gt;99.9%"
    assert rr.pct(0.994, 0) == "99%"
    assert rr.pct(1.0, 0) == "100%"


def test_the_second_car_in_a_garage_is_tinted():
    first = rr.series_colour({"team": "mercedes"})
    second = rr.series_colour({"team": "mercedes", "second_car": True})
    assert first == rr.team_colour("mercedes")
    assert second != first and "color-mix" in second


def test_after_qualifying_the_board_shows_where_each_driver_qualified():
    rows = [
        {
            "driver_id": "a",
            "name": "A",
            "short": "A",
            "team": "mercedes",
            "p_win": 0.3,
            "p_podium": 0.7,
            "p_top10": 0.97,
            "grid": 16,
        }
    ]
    before, after = rr.quali_board(rows), rr.quali_board(rows, qualified=True)
    assert "Top 10" in before and "P16" not in before
    assert "Starts" in after and "P16" in after


def test_the_championship_panel_uses_the_latest_projection(tmp_path, monkeypatch):
    """The logged forecast fixes the race boards; the season panel should move
    with results, so the dashboard reads the latest projection when there is one."""
    import json

    from f1pred import cli, config
    from f1pred import method_page as mp

    for name in ("PREDICTIONS", "REPORTS"):
        (tmp_path / name).mkdir()
        monkeypatch.setattr(config, name, tmp_path / name)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "f1.duckdb")
    monkeypatch.setattr(config, "SEASON_NOW", tmp_path / "season.json")

    logged = {
        "season": 2026,
        "round": 15,
        "race_name": "Test GP",
        "circuit_id": "x",
        "race_start_utc": None,
        "generated_at_utc": "2026-01-01T00:00:00+00:00",
        "grid_known": False,
        "quali_board": [],
        "race_board": [],
        "field_probs": [],
        "season_outlook": {"marker": "logged"},
    }
    (tmp_path / "PREDICTIONS" / "2026-15-prequali-x.json").write_text(json.dumps(logged))
    config.SEASON_NOW.write_text(json.dumps({"marker": "latest"}))

    rendered = {}
    monkeypatch.setattr(report, "write", lambda p, *a, **k: rendered.update(p) or tmp_path / "i.html")
    monkeypatch.setattr(mp, "write", lambda *a, **k: tmp_path / "m.html")
    cli.main(["dashboard"])
    assert rendered["season_outlook"] == {"marker": "latest"}
    assert rendered["race_board"] == [], "the logged boards must be left as they were"


def test_title_odds_say_what_the_simulation_can_and_cannot_resolve():
    assert rr._odds(0.0) == "&lt;0.1%"  # still mathematically alive
    assert rr._odds(0.0, alive=False) == "out"
    assert rr._odds(1.0) == "clinched"
    assert rr._odds(0.9995) == "99.9%"
    assert rr._odds(0.008) == "0.8%"


def test_the_record_reads_race_by_race_and_merges_reruns():
    """A race card lists its forecasts in order - re-runs of the same step
    merged - and ignores anything made after the start."""

    def fc(when, stage, pick, p):
        return {
            "season": 2026,
            "round": 3,
            "race_name": "Test GP",
            "race_start_utc": "2026-03-10 14:00:00",
            "generated_at_utc": when,
            "meta": {"stage": stage},
            "race_board": [{"driver_id": pick, "p_win": p}],
            "field_probs": [
                {"driver_id": "a", "name": "A", "p_win": p if pick == "a" else 0.2},
                {"driver_id": "b", "name": "B", "p_win": p if pick == "b" else 0.1},
            ],
        }

    preds = [
        fc("2026-03-07T09:00:00+00:00", "pre_quali", "b", 0.5),
        fc("2026-03-08T09:00:00+00:00", "pre_quali", "b", 0.55),
        fc("2026-03-09T20:00:00+00:00", "post_quali", "a", 0.6),
        fc("2026-03-10T15:00:00+00:00", "post_quali", "b", 0.9),
    ]
    [race] = report.history_of(preds, lambda s, r: {"a": 1, "b": 2}, {})
    assert [st["stage"] for st in race["steps"]] == ["pre_quali", "post_quali"]
    assert race["steps"][0]["runs"] == 2 and race["steps"][0]["p_pick_high"] == 0.55
    assert race["final_hit"] and race["winner"] == "A"
    html = rr.record_table([race])
    assert "Before practice" in html and "After qualifying" in html and "won by A" in html
    assert html.count("&#10003;") == 1 and html.count("&#10007;") == 1, "one miss before, one hit after"


def _field(n):
    return [
        {
            "driver_id": f"d{i}",
            "name": f"D{i}",
            "team": "ferrari",
            "p_win": 0.5 / (i + 1),
            "p_podium": 0.6 / (i + 1),
            "p_top5": 0.9 / (i + 1),
            "exp_position": i + 1.5,
            "grid": i + 1,
            "why": "Helped by starting position",
            "why_detail": {"starting position": 0.5, "reliability record": -0.1},
        }
        for i in range(n)
    ]


def test_the_board_shows_the_whole_field_leading_with_the_contenders(no_track_record):
    html = rr.race_board(_field(20))
    assert "Show the other 10 drivers" in html
    assert html.count("<div class='row") == 20


def test_a_finished_race_shows_the_result_beside_the_unchanged_forecast(monkeypatch, no_track_record):
    monkeypatch.setattr(report, "actual_result", lambda season, rnd: {"d1": 1, "d0": 2})
    pred = {
        "season": 2026,
        "round": 3,
        "race_name": "Test GP",
        "grid_known": True,
        "generated_at_utc": "2026-03-09T20:00:00+00:00",
        "race_board": [],
        "quali_board": [],
        "field_probs": _field(5),
        "meta": {"stage": "post_quali"},
    }
    html = report.build(prediction=pred)
    assert "Race finished" in html and "<span>Result</span>" in html
    assert "Result in" in html, "the stage shown is the result, not a live forecast"


def test_each_outcome_bar_adds_up_and_agrees_with_the_table():
    """The hero bar is the table's own chances taken apart: win, 2nd-3rd,
    4th-5th, 6th-10th, the rest. It sums to one and can't disagree."""
    r = {"p_win": 0.3, "p_podium": 0.6, "p_top5": 0.8, "p_top10": 0.95}
    shares = rr.outcome_shares(r)
    assert abs(sum(shares) - 1) < 1e-9
    assert [round(s, 2) for s in shares] == [0.3, 0.3, 0.2, 0.15, 0.05]
    html = rr.outcome_bars([{"name": "A", "team": "ferrari", **r, "finished": 4}], finished=True)
    assert "class='b2 hit'" in html, "a P4 finish rings the 4th-5th segment"


def test_a_forecast_whose_win_chances_disagree_with_its_distribution_is_flagged():
    """Some early logged forecasts took the win chance from somewhere other than
    the finishing distribution. They are never rewritten, so the page says so."""
    base = {"meta": {"matrix_driver_ids": ["a", "b"]}, "position_matrix": [[0.6, 0.4], [0.4, 0.6]]}
    ok = {**base, "field_probs": [{"driver_id": "a", "p_win": 0.6}, {"driver_id": "b", "p_win": 0.4}]}
    off = {**base, "field_probs": [{"driver_id": "a", "p_win": 0.5}, {"driver_id": "b", "p_win": 0.4}]}
    assert report.coherent(ok) and not report.coherent(off)
    line = report.status_line({**off, "race_start_utc": None}, "post_quali", [], {})
    assert "earlier version of the pipeline" in line


def test_each_race_keeps_its_final_forecast_made_before_the_start():
    """The archived page shows the last forecast logged before the race, never
    one made after it started."""
    preds = [
        {"season": 2026, "round": 3, "race_start_utc": "2026-03-10 14:00:00", "generated_at_utc": t}
        for t in ("2026-03-08T09:00:00+00:00", "2026-03-09T20:00:00+00:00", "2026-03-10T15:00:00+00:00")
    ]
    final = report.final_forecasts(preds)
    assert final[(2026, 3)]["generated_at_utc"] == "2026-03-09T20:00:00+00:00"
    assert report.page_name(2026, 3) == "race-2026-03.html"


def test_the_track_record_links_each_race_to_its_page():
    race = {
        "season": 2026,
        "round": 3,
        "race": "Test GP",
        "winner": "A",
        "winner_team": "ferrari",
        "steps": [],
    }
    assert "href='race-2026-03.html'" in rr.record_table([race])


def test_past_races_are_one_click_from_every_page(no_track_record):
    """The nav reaches the list of graded races, and an archived race page says
    it is archived and links back to the latest forecast."""
    assert "href='races.html'" in report.build(prediction=None)
    assert "Past races" in report.races_page()
    archived = report.build(
        prediction={"season": 2026, "round": 3, "race_name": "Test GP", "field_probs": _field(3)},
        archived=True,
    )
    assert "Archived forecast" in archived and "href='index.html'" in archived
