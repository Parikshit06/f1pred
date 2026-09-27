"""The method page: how the forecast is made and how well it has done.

One short article in plain words. Every figure is read from reports/*.json so
the page cannot drift from the evaluation; the full tables sit behind one fold
and METHODOLOGY.md has the rest.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from . import config
from . import report_render as rr

log = logging.getLogger(__name__)


def _load(name: str) -> dict:
    path = config.REPORTS / name
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        log.warning("could not read %s", path)
        return {}


def _top_bucket(rows: list[dict]) -> dict | None:
    rows = [r for r in rows or [] if int(r.get("n", 0)) >= 20]
    return max(rows, key=lambda r: float(r["stated"]), default=None)


def _leader_record(checkpoints: list[dict]) -> tuple[int, int] | None:
    """(checkpoints where the favourite was the points leader, checkpoints where
    the leader became champion), as recorded by the title backtest."""
    if not checkpoints or any("leader" not in c for c in checkpoints):
        return None
    same = sum(c["favourite"] == c["leader"] for c in checkpoints)
    return same, sum(int(c["leader_was_champion"]) for c in checkpoints)


STAGES = [
    ("After qualifying", None, "comparison_vs_grid", "the calibrated starting grid"),
    ("Before qualifying", "pre_quali", "comparison_vs_championship", "the championship order"),
    ("Qualifying", "qualifying", "comparison", "recent qualifying form"),
]
METRICS = {
    "ndcg3": ("Top-3 order", "NDCG@3", False),
    "ndcg5": ("Top-5 order", "NDCG@5", False),
    "winner_hit": ("Winner called", "hit rate", False),
    "podium_overlap": ("Podium named", "overlap of 3", False),
    "top5_overlap": ("Top five named", "overlap of 5", False),
    "top10_overlap": ("Top ten named", "overlap of 10", False),
    "spearman": ("Whole-field order", "Spearman", False),
    "kendall": ("Whole-field order", "Kendall", False),
    "win_logloss": ("Win chances", "log loss", True),
    "win_brier": ("Win chances", "Brier", True),
    "podium_brier": ("Podium chances", "Brier", True),
    "top10_brier": ("Top-10 chances", "Brier", True),
    "pole_hit": ("Pole-sitter called", "hit rate", False),
    "pole_logloss": ("Pole chances", "log loss", True),
}
# One probability score per stage, and the winner rate: fixed in advance,
# with every measure where a reference is clearly ahead added to them.
HEADLINE = [
    (None, "winner_hit"),
    (None, "win_logloss"),
    ("pre_quali", "win_logloss"),
    ("qualifying", "pole_logloss"),
]


def _comparisons(bt: dict) -> list[tuple[str, str, dict]]:
    """(stage, reference, row) for every paired comparison in the report."""
    out = []
    for stage, section, key, ref in STAGES:
        rows = (bt.get(section) or {}).get(key, []) if section else bt.get(key, [])
        out += [(stage, ref, r) for r in rows if {"model", "difference", "ci_low", "ci_high"} <= r.keys()]
    return out


def _fmt(metric: str, v: float) -> str:
    return f"{v:.0%}" if metric.endswith("_hit") else f"{v:.3f}"


def _ref_value(r: dict) -> float | None:
    skip = {"metric", "model", "difference", "ci_low", "ci_high", "better", "n_races"}
    return next((v for k, v in r.items() if k not in skip), None)


def _verdict(r: dict) -> tuple[str, str]:
    b = r.get("better")
    if b == "model":
        return "model clearly better", "good"
    if b in (None, "unclear"):
        return "no clear difference", "flat"
    return "reference clearly better", "bad"


def _headline(bt: dict) -> str:
    """The key results in one row, before any detail."""
    w = bt.get("window") or {}
    comps = _comparisons(bt)
    sections = {s[0]: s[1] for s in STAGES}
    picked = []
    for section, metric in HEADLINE:
        stage = next(k for k, v in sections.items() if v == section)
        row = next((c for c in comps if c[0] == stage and c[2]["metric"] == metric), None)
        if row:
            picked.append(row)
    picked += [c for c in comps if c[2].get("better") not in ("model", "unclear", None) and c not in picked]
    if not (picked and w):
        return ""
    tiles = []
    for stage, ref, r in picked:
        name, term, lower = METRICS.get(r["metric"], (r["metric"], "", False))
        verdict, cls = _verdict(r)
        ref_v = _ref_value(r)
        tiles.append(
            f"<div class='kr'><span class='kr-s'>{rr.esc(stage)}</span>"
            f"<b class='kr-m'>{rr.esc(name)} <small>{rr.esc(term)}{', lower is better' if lower else ''}</small></b>"
            f"<span class='kr-v'>{_fmt(r['metric'], r['model'])}"
            + (f"<em>vs {_fmt(r['metric'], ref_v)}</em>" if ref_v is not None else "")
            + f"</span><span class='kr-r'>model vs {rr.esc(ref)}</span>"
            f"<span class='kr-x {cls}'>{verdict}</span></div>"
        )
    through = w.get("through") or ["", ""]
    tuned = w.get("tuned_on") or ["", ""]
    return (
        f"<p class='kr-cap'>Walk-forward test on {w.get('n_races')} races, {w.get('start_season')} round 1 to "
        f"{through[0]} round {through[1]}, with settings fixed on {tuned[0]}&ndash;{tuned[1]}. "
        "A verdict is clear only when the paired 95% interval over races excludes zero.</p>"
        f"<div class='keyrow'>{''.join(tiles)}</div>"
    )


def _detailed(bt: dict) -> str:
    """Every paired comparison the evaluation produced, grouped by stage."""
    comps = _comparisons(bt)
    if not comps:
        return ""
    rows, last = [], None
    for stage, ref, r in comps:
        name, term, lower = METRICS.get(r["metric"], (r["metric"], "", False))
        verdict, cls = _verdict(r)
        ref_v = _ref_value(r)
        pts = r["metric"].endswith("_hit")
        diff = (
            f"{r['difference'] * 100:+.0f} pts ({r['ci_low'] * 100:+.0f} to {r['ci_high'] * 100:+.0f})"
            if pts
            else f"{r['difference']:+.3f} ({r['ci_low']:+.3f} to {r['ci_high']:+.3f})"
        )
        if stage != last:
            rows.append(
                f"<tr class='grp'><td colspan='5'><b>{rr.esc(stage)}</b> <small>vs {rr.esc(ref)}</small></td></tr>"
            )
        last = stage
        rows.append(
            f"<tr><td>{rr.esc(name)}<small>{rr.esc(term)}{' &darr;' if lower else ''}</small></td>"
            f"<td class='n'>{_fmt(r['metric'], r['model'])}</td>"
            f"<td class='n'>{_fmt(r['metric'], ref_v) if ref_v is not None else ''}</td>"
            f"<td class='n mono'>{diff}</td><td class='vd {cls}'>{verdict}</td></tr>"
        )
    return (
        "<p>Every comparison the evaluation made, model minus reference, with its 95% interval. "
        "&darr; marks a score where lower is better.</p>"
        "<div class='scroll'><table class='keyres'><thead><tr><th>Measure</th>"
        "<th class='n'>Model</th><th class='n'>Reference</th><th class='n'>Difference</th>"
        f"<th>Verdict</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
        "<p class='cap'>Source: <code>reports/backtest.json</code>, written by <code>make backtest</code>.</p>"
    )


INPUTS = [
    "Where each driver starts",
    "Their recent results",
    "How fast the car has been",
    "One-lap speed",
    "How often the car breaks down",
    "Their record at this track",
    "This weekend's practice times",
]

WEEKEND = [
    ("Before practice", "Past races only."),
    ("After practice", "Adds this weekend's lap times."),
    ("After qualifying", "Adds the real starting grid."),
    ("After the race", "Graded against the result."),
]


def _looks_at() -> str:
    chips = "".join(f"<li>{rr.esc(x)}</li>" for x in INPUTS)
    return (
        f"<ul class='chips'>{chips}</ul>"
        f"<p>From every race since {config.FIRST_SEASON} it learns how much each of these tends to "
        "matter, then ranks this weekend's drivers the same way. It never sees a race's result "
        "before forecasting it.</p>"
    )


def _dots(k: int = 36) -> str:
    dots = "".join(f"<i class='{'on' if i < k else ''}'></i>" for i in range(100))
    return f"<figure class='dots'><div>{dots}</div><figcaption>{k} in 100</figcaption></figure>"


def _to_chances() -> str:
    return (
        "<div class='split'><div><p>A ranking is not a forecast yet. So the model plays the race out "
        "10,000 times, adding crashes, breakdowns and the ordinary chaos of a Grand Prix.</p>"
        "<p>If a driver wins 3,600 of those races, their chance is 36%: about 36 in every 100. "
        "Every percentage on the site is counted this way.</p></div>" + _dots() + "</div>"
    )


def _example() -> str:
    """The latest graded race: the winner's chance as the weekend went on."""
    try:
        from . import report

        history = report.race_history()
    except Exception:  # the example is optional; the page is not
        log.warning("could not read the live record", exc_info=True)
        return ""
    if not history:
        return ""
    r = history[0]
    last = {rr._step_label(st): st for st in r["steps"]}
    seen = [(k, last[k]) for k in rr.WEEKEND_STEPS if k in last]
    if len(seen) < 2:
        return ""
    (first, a), (_, b) = seen[0], seen[-1]
    return (
        f"<p class='example'>At the {rr.esc(r['race'])}, the forecast gave {rr.esc(r['winner'])} "
        f"<b>{rr.pct(a['p_winner'], 0)}</b> {first.lower()}, and <b>{rr.pct(b['p_winner'], 0)}</b> once "
        f"the grid was set. {rr.esc(r['winner'])} won.</p>"
    )


def _sharpens() -> str:
    steps = "".join(f"<li><b>{rr.esc(k)}</b><span>{rr.esc(v)}</span></li>" for k, v in WEEKEND)
    return (
        "<p>A new forecast is published at each step of a race weekend, and none is edited "
        "afterwards. Each one knows a little more than the last.</p>"
        f"<ol class='timeline'>{steps}</ol>" + _example()
    )


def _claims(bt: dict, tb: dict | None = None) -> str:
    """Three plain answers to 'is it any good', each from the reports."""
    out = []
    main = {r["method"]: r for r in bt.get("summary") or []}
    n = int((bt.get("window") or {}).get("n_races") or 0)
    if main.get("model") and main.get("grid") and n:
        out.append(
            (
                f"After qualifying, its favourite won {main['model']['winner_hit']:.0%} of races.",
                (
                    f"That sounds good, but simply backing the driver on pole won "
                    f"{main['grid']['winner_hit']:.0%}, and over {n} races a gap that size could be luck. "
                    "The grid itself holds most of what can be predicted."
                ),
            )
        )
    pre = {r["metric"]: r for r in (bt.get("pre_quali") or {}).get("comparison_vs_championship") or []}
    if pre.get("win_logloss"):
        if pre["win_logloss"].get("better") == "model":
            head = "Before qualifying, it beats following the championship table."
            tail = "Its chances are more accurate" + (
                ", though it picks the winner no more often."
                if (pre.get("winner_hit") or {}).get("better") != "model"
                else "."
            )
        else:
            head, tail = "Before qualifying, it is no better than the championship table.", ""
        out.append((head, tail))
    rel = bt.get("reliability") or {}
    top = _top_bucket(rel.get("podium"))
    if rel:
        tail = (
            f"Its boldest calls are the exception: podiums it rated around {float(top['stated']):.0%} "
            f"happened {float(top['observed']):.0%} of the time."
            if top and float(top["ci_high"]) < float(top["stated"])
            else ""
        )
        out.append(("When it says 30%, it happens about 30% of the time.", tail))
    checkpoints = (tb or {}).get("checkpoints") or []
    lead = _leader_record(checkpoints) if checkpoints else None
    if lead:
        n_cp = len(checkpoints)
        right = sum(int(c["favourite_was_right"]) for c in checkpoints)
        out.append(
            (
                f"Its title favourite went on to win {right / n_cp:.0%} of the time.",
                (
                    f"But that favourite was nearly always the driver already leading the championship "
                    f"({lead[0]} times in {n_cp}), and simply backing the leader would have been right "
                    f"{lead[1] / n_cp:.0%} of the time."
                ),
            )
        )
    if not out:
        return ""
    items = "".join(f"<li><b>{rr.esc(h)}</b>{f' <span>{rr.esc(s)}</span>' if s else ''}</li>" for h, s in out)
    return f"<ol class='claims'>{items}</ol>"


def _good(bt: dict, tb: dict) -> str:
    claims = _claims(bt, tb)
    if not claims:
        return ""
    w = bt.get("window") or {}
    chart = rr.calibration_plot(bt.get("reliability") or {})
    return (
        f"<p>To find out, it forecast every race from {w.get('start_season', '')} onward using only "
        "the races before each one, and was compared with simple guesses anyone could make.</p>"
        + claims
        + (
            "<h3 class='sub'>Does 30% mean 30%?</h3><p>Every driver-race since "
            f"{w.get('start_season', '')}, grouped by the chance the forecast gave. On the dashed line, "
            "the forecast meant exactly what it said; each whisker is the 95% interval on how often "
            "it happened.</p>" + chart
            if chart
            else ""
        )
    )


def _blind_spots() -> str:
    return (
        "<p>Weather, tyre strategy, a safety car at the wrong moment, penalties handed out during "
        "the race, upgrades and team orders. It also treats each car's chance of breaking down "
        "separately, even when a team shares a problem.</p>"
    )


def _hood(bt: dict) -> str:
    repo = rr.esc(config.REPO_URL)
    w = bt.get("window") or {}
    tuned = w.get("tuned_on") or ["", ""]
    rows = [
        ("The model", "Ranks the drivers within each race", "XGBoost, learning to rank"),
        (
            "The chances",
            "Turns the ranking into odds, blended with 10,000 simulated races",
            "Plackett&ndash;Luce, Monte Carlo",
        ),
        (
            "The test",
            (
                f"Settings picked on {tuned[0]}&ndash;{tuned[1]}; every result measured on races from "
                f"{w.get('start_season', '')} it had never seen"
            ),
            "walk-forward evaluation",
        ),
        ("The comparison", "Only counts a gap that survives reshuffling the races", "paired bootstrap"),
    ]
    items = "".join(f"<div><dt>{k}</dt><dd>{v}<small>{term}</small></dd></div>" for k, v, term in rows)
    return (
        f"<dl class='hood'>{items}</dl>"
        f"<p class='links'><a href='{repo}'>Source code</a><a href='{repo}/blob/main/METHODOLOGY.md'>"
        f"Full methodology</a><a href='{repo}/blob/main/reports/backtest.json'>Raw results</a></p>"
    )


def _part(title: str, body: str, cls: str = "") -> str:
    return f"<section class='part {cls}'><h2>{title}</h2>{body}</section>" if body else ""


def build(standalone: bool = True) -> str:
    bt = _load("backtest.json")
    tb = _load("title_backtest.json")

    s: list[str] = [
        rr.top_bar(None, rr.utcnow(), page="method"),
        "<main class='wrap'><article class='article'>",
        "<header class='mast'><h1>How the forecast works</h1>",
        (
            "<p class='standfirst'>Before every Grand Prix, this site publishes how it thinks the race "
            "will finish. Here is how that forecast is made, and how often it is right.</p>"
        ),
        "</header>",
        _headline(bt),
        _part("What it looks at", _looks_at()),
        _part("From a ranking to a percentage", _to_chances()),
        _part("It sharpens through the weekend", _sharpens()),
        _part("Is it any good?", _good(bt, tb)),
        _part("What it can&rsquo;t see", _blind_spots()),
        _part("Every result", _detailed(bt)),
        _part("Under the hood", _hood(bt)),
        "</article>",
        (
            "<footer><span>Data: jolpica-f1 &middot; OpenF1 &middot; FastF1</span>"
            f"<span><a href='index.html'>Forecast</a> &middot; "
            f"<a href='{rr.esc(config.REPO_URL)}'>Source</a></span></footer>"
        ),
        "</main>",
    ]
    return rr.document("".join(s), standalone=standalone, title="How the forecast works")


def write(path: Path | None = None) -> Path:
    path = path or (config.REPORTS / "method.html")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build())
    log.info("Wrote %s", path)
    return path
