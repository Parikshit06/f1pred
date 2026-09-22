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
    """The top strap inverts the page, so its link uses --paper, not the accent
    (unreadable on the strap) and not a transparent mix (which resolved close
    to the strap's own background)."""
    rule = rr.CSS.split(".bar a{")[1].split("}")[0]
    colour = next(d for d in rule.split(";") if d.strip().startswith("color:"))
    assert "var(--paper)" in colour, colour
    assert "color-mix" not in colour, "a transparent text colour resolved near the strap itself"


def test_the_method_page_builds_and_links_back():
    from f1pred import method_page

    html = method_page.build()
    assert html.startswith("<!doctype html>")
    assert "index.html" in html, "no way back to the forecast"
    assert "Method and accuracy" in html


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


def test_the_method_page_reports_whether_the_range_holds(tmp_path, monkeypatch):
    """The projection's band is the one claim on the page that is a promise with
    a number attached, and it is currently NOT being kept - 71% against a stated
    80%. A page that quietly dropped that section when the calibration file was
    renamed would be presenting a shortfall as a clean bill of health."""
    import json

    from f1pred import config, method_page

    monkeypatch.setattr(config, "REPORTS", tmp_path)
    (tmp_path / "spread_calibration.json").write_text(
        json.dumps(
            {
                "best": 6.0,
                "fit_seasons": [2019, 2022],
                "grade_seasons": [2023, 2025],
                "held_out": {
                    "before": {"coverage": 0.458, "width": 38.8, "n": 120},
                    "after": {"coverage": 0.708, "width": 89.7, "n": 120},
                },
            }
        )
    )
    html = method_page.build()
    # The section was folded into "The championship" - grading who wins and
    # grading how close are the same question, and splitting them made a reader
    # hold the first to follow the second.
    assert "The championship" in html
    assert "46%" in html and "71%" in html, "the graded coverage is not on the page"
    assert "80%" in html, "the target the band is being held to is not stated"


def test_the_method_page_omits_the_range_section_when_it_has_not_been_graded():
    """No calibration file means no claim - better a missing section than one
    quoting numbers from a run that never happened."""
    import pytest as _pytest

    from f1pred import config, method_page

    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "REPORTS", config.REPORTS / "does-not-exist")
        assert "Does the range hold up" not in method_page.build()


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
    """Sections alternate between paper and a slightly darker band. Every band
    edge already carries a hairline, so the tone only has to suggest a change of
    register - it does not have to separate anything.

    Below about 1 in L* nobody sees it and the page goes flat; above about 3 it
    stops reading as the same sheet under different light and starts reading as
    a grey rectangle dropped on top, which is exactly the note this pins. It
    shipped once at 4.2.
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
    """Code chips used to borrow the band colour. Softening the band would have
    made them vanish, so they have their own token - and it has to be a step
    away from BOTH surfaces they can sit on."""
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
    """The condition CI runs in, and the one these tests never had locally.

    `data/*.duckdb` is rebuildable, so it is gitignored and a fresh clone does
    not have one. Two readers reach for it while rendering - the track record
    and the driver-surname lookup - and a read-only connect to a missing file
    raises rather than creating it. Both now have an empty answer instead,
    because they decorate a page rather than compute it.

    This failed on the repo's very first CI run, which is the honest argument
    for the test: the suite passed on every machine that had already run the
    pipeline, which was every machine anyone had tried it on.
    """
    from f1pred import config, method_page, store

    monkeypatch.setattr(config, "DB_PATH", tmp_path / "absent.duckdb")
    assert not store.database_exists()

    assert report.scoreboard().empty, "a missing database invented a track record"
    assert "method.html" in report.build(prediction=None)
    assert "Model specification" in method_page.build()


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
def test_the_high_end_paragraph_counts_the_table_it_sits_under(tmp_path, monkeypatch):
    """It used to be typed out, and it drifted: the page read "16 of 16 correct
    above 95%" at a true rate "as low as 81%" while the table directly above it
    showed twelve checkpoints and a Wilson floor of 0.76. A sentence that
    interprets a table has to be computed from that table."""
    import json

    from f1pred import config, method_page

    monkeypatch.setattr(config, "REPORTS", tmp_path)
    (tmp_path / "title_backtest.json").write_text(
        json.dumps(
            {
                "brier": 0.1,
                "checkpoints": [
                    {
                        "season": 2024,
                        "after_round": 8,
                        "races_left": 10,
                        "favourite": "hamilton",
                        "p_favourite": 0.97,
                        "champion": "hamilton",
                        "favourite_was_right": 1,
                        "p_on_actual_champion": 0.97,
                    },
                    {
                        "season": 2025,
                        "after_round": 9,
                        "races_left": 9,
                        "favourite": "piastri",
                        "p_favourite": 0.61,
                        "champion": "norris",
                        "favourite_was_right": 0,
                        "p_on_actual_champion": 0.3,
                    },
                ],
                "calibration": [
                    {"bucket": "50-80%", "claimed": "0.61", "happened": "0.0", "n": "1", "ci_low": "0.0"},
                    {"bucket": "over 95%", "claimed": "0.97", "happened": "1.0", "n": "1", "ci_low": "0.2"},
                ],
            }
        )
    )
    html = method_page.build()
    assert "right 1 times out of 1" in html, "the count is not read off the table"
    assert "true rate of 20%" in html, "the Wilson floor is not read off the table"
    assert "16 of 16" not in html
    # The one miss is named, with the probability the table recorded for it.
    assert "2025 r9 favoured Piastri at 61.0%" in html


def test_the_pooled_calibration_line_adds_up_the_buckets(tmp_path, monkeypatch):
    """The pooled figure was prose too, and quoted 62 races over a table that
    summed to 61 - which was how the dropped-bucket bug stayed invisible."""
    import json

    from f1pred import config, method_page

    monkeypatch.setattr(config, "REPORTS", tmp_path)
    (tmp_path / "backtest.json").write_text(
        json.dumps(
            {
                "summary": [],
                "by_season": [],
                "calibration": [
                    {"bucket": "(0.0, 0.5]", "predicted": "0.400", "actual": "0.500", "n": "10"},
                    {"bucket": "(0.5, 1.0]", "predicted": "0.800", "actual": "0.900", "n": "30"},
                ],
                "params": {},
            }
        )
    )
    html = method_page.build()
    assert "Pooled over 40 races" in html
    assert "stated 0.700, observed 0.800" in html
    assert "under-confident" in html
    assert "Buckets hold 10&ndash;30 races" in html


def test_a_column_heading_is_aligned_the_same_way_as_its_own_cells():
    """The bug this catches is visible from across the room and was invisible
    in the code: cells were right-aligned by position and headings were not
    aligned at all, so every numeric column sat under a left-flush label."""
    df = pd.DataFrame({"approach": ["Grid order"], "top 5": [3.919], "races": [62]})
    html = rr.table(df)

    heads = re.findall(r"<th(?: class='([^']*)')?>", html)
    cells = re.findall(r"<td class='([^']*)'>", html)
    assert len(heads) == len(cells) == 3
    for head, cell in zip(heads, cells):
        assert ("n" in (head or "").split()) == ("n" in cell.split()), (
            f"heading {head!r} disagrees with cell {cell!r}"
        )


def test_a_column_of_words_is_not_right_aligned():
    """Alignment used to be chosen by column position, which right-aligned a
    column of names in a monospaced face whenever a table carried two text
    columns - the title-projection listing did, and it read as broken."""
    df = pd.DataFrame(
        {"season": [2021], "favourite": ["Hamilton"], "champion": ["Verstappen"], "claimed": [0.63]}
    )
    cells = re.findall(r"<td class='([^']*)'>", rr.table(df))
    assert cells[1] == "" and cells[2] == "", "a column of driver names was right-aligned"
    assert "n" in cells[3].split(), "a column of probabilities was not right-aligned"


def test_a_quantity_carrying_its_unit_still_counts_as_a_number():
    """The range table publishes "90 pts" and "71%" as strings. They belong
    under a right-aligned heading with the figures they sit beside."""
    df = pd.DataFrame({"projection": ["Current model"], "band held": ["71%"], "width": ["90 pts"]})
    cells = re.findall(r"<td class='([^']*)'>", rr.table(df))
    assert "n" in cells[1].split() and "n" in cells[2].split()


def test_the_emphasised_row_starts_where_every_other_row_starts():
    """The marker on the highlighted row used to be drawn by giving that one
    cell an extra 10px of padding, which pushed its text out of line with the
    column it belonged to. The gutter is reserved on every row now."""
    css = rr.document("", standalone=True)
    rule = re.search(r"tr\.me td:first-child\{([^}]*)\}", css)
    assert rule, "the emphasis rule is gone"
    assert "padding" not in rule.group(1), "the emphasised row is still padded out of line"
    assert re.search(r"th:first-child,\s*td:first-child\{[^}]*padding-left", css), (
        "the first-column gutter is not reserved on every row"
    )


def test_probability_buckets_are_published_as_ranges_not_pandas_intervals():
    from f1pred.method_page import _band

    assert _band("(0.189, 0.378]") == "19–38%"
    assert _band("(-0.001, 0.189]") == "0–19%", "a negative edge leaked into the page"
    assert _band("over 95%") == "over 95%", "a label that is already readable was mangled"
    assert "&ndash;" not in _band("(0.189, 0.378]"), "an HTML entity will be escaped and shown raw"
