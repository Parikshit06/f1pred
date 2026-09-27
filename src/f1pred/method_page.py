"""Model documentation: specification, features, measured performance.

Written as a spec sheet, not an essay. Each section names the command that
regenerates it. Earlier drafts of this page explained why each choice was good;
that is the register that makes a page read as generated, and it is also not
what a reader checking the work needs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

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


def _surnames() -> dict[str, str]:
    """driver_id -> family name, so the graded table reads as names and not keys."""
    from .store import connect, database_exists

    if not database_exists():
        return {}

    with connect(read_only=True) as con:
        return dict(con.execute("SELECT driver_id, max(family_name) FROM raw_drivers GROUP BY 1").fetchall())


def _range_block(sc: dict) -> str:
    """How close the projection's published range has been.

    Split out because the range is graded by its own command and its own JSON:
    the title backtest can be missing while this is present, and the section
    should still say what it knows.
    """
    held = sc.get("held_out")
    if not held:
        return ""
    rows = pd.DataFrame(
        [
            {
                "projection": label,
                "teams held": f"{held[key]['coverage']:.0%}",
                "width": f"{held[key]['width']:.0f} pts",
                "teammate gap held": (
                    f"{held[key]['teammate_coverage']:.0%}" if "teammate_coverage" in held[key] else "–"
                ),
                "should be": "80%",
            }
            for label, key in (("Race luck only", "before"), ("Current", "after"))
        ]
    )
    return (
        "<p class='cap' style='margin-top:26px'><b>How close.</b> Every points total is "
        "published with a range the real answer should land inside eight times in ten. "
        "Graded against constructors' final points, and against the final gap between "
        "teammates:</p>"
        + rr.table(rows)
        + "<p class='cap'>Race luck averages out over a dozen races. What doesn't is the model "
        "being wrong about a car <em>now</em>, which carries into every remaining race, so each "
        f"simulated season draws one pace offset per team (sized {config.SEASON_PACE_UNCERTAINTY:g} "
        "positions on 2019&ndash;22)."
        + (
            f" A per-driver offset is sized the same way ({config.SEASON_DRIVER_UNCERTAINTY:g})."
            if config.SEASON_DRIVER_UNCERTAINTY
            else " A per-driver offset was swept too and came out at zero on 2019&ndash;22; "
            "on later seasons the teammate gap still lands inside its range less often than 80%."
        )
        + (" Constructors are still short of 80%." if held["after"]["coverage"] < 0.80 else "")
        + "</p>"
    )


NAMES = {
    "model": "This model",
    "grid": "Grid order",
    "championship": "Championship order",
    "recent_form": "Recent driver form",
    "team_form": "Team form",
}
METRIC_NAMES = {
    "ndcg3": "NDCG@3",
    "ndcg5": "NDCG@5",
    "winner_hit": "winner called",
    "podium_overlap": "podium overlap",
    "top5_overlap": "top-5 overlap",
    "spearman": "Spearman",
    "kendall": "Kendall",
    "win_logloss": "win log loss",
    "win_brier": "win Brier",
    "podium_brier": "podium Brier",
    "top10_brier": "top-10 Brier",
}


def _row(rows: list[dict], key: str, value: str) -> dict:
    return next((r for r in rows if r.get(key) == value), {})


KEY_ROWS = [
    (
        "After qualifying",
        "comparison_vs_grid",
        None,
        "calibrated grid",
        ["ndcg3", "ndcg5", "winner_hit", "win_logloss", "win_brier", "podium_brier"],
    ),
    (
        "Before qualifying",
        "comparison_vs_championship",
        "pre_quali",
        "championship order",
        ["ndcg5", "win_logloss", "podium_brier"],
    ),
    ("Qualifying", "comparison", "qualifying", "recent qualifying form", ["ndcg5", "pole_logloss"]),
]


def _key_results(bt: dict) -> str:
    """Model against its no-model reference at each stage, with a verdict per
    row. Metrics where the reference is clearly ahead are always shown."""
    names = {**METRIC_NAMES, "pole_logloss": "pole log loss"}
    rows = []
    for stage, key, section, ref, wanted in KEY_ROWS:
        comp = (bt.get(section) or {}).get(key, []) if section else bt.get(key, [])
        comp = {r["metric"]: r for r in comp}
        shown = wanted + [
            m for m, r in comp.items() if r.get("better") not in ("unclear", None) and m not in wanted
        ]
        for m in shown:
            r = comp.get(m)
            if not r:
                continue
            model_side = r.get("better") == "model"
            verdict = (
                "inconclusive"
                if r.get("better") == "unclear"
                else ("clear: model better" if model_side else "clear: reference better")
            )
            fmt = (lambda v: f"{v * 100:.0f}%") if m == "winner_hit" else (lambda v: f"{v:.3f}")
            diff = (
                f"{r['difference'] * 100:+.0f} pts ({r['ci_low'] * 100:+.0f} to {r['ci_high'] * 100:+.0f})"
                if m == "winner_hit"
                else f"{r['difference']:+.3f} ({r['ci_low']:+.3f} to {r['ci_high']:+.3f})"
            )
            model_v = r.get("model")
            ref_v = next(
                (
                    v
                    for k, v in r.items()
                    if k not in ("metric", "model", "difference", "ci_low", "ci_high", "better", "n_races")
                ),
                None,
            )
            cls = "" if verdict == "inconclusive" else ("model" if model_side else "grid")
            first = m == shown[0]
            rows.append(
                f"<tr{' class=grp' if first else ''}><td>{f'<b>{stage}</b><small>vs {rr.esc(ref)}</small>' if first else ''}</td>"
                f"<td>{rr.esc(names.get(m, m))}</td>"
                f"<td class='n'>{fmt(model_v) if model_v is not None else ''}</td>"
                f"<td class='n'>{fmt(ref_v) if ref_v is not None else ''}</td>"
                f"<td class='n mono'>{diff}</td>"
                f"<td><span class='pill {cls}'>{verdict}</span></td></tr>"
            )
    if not rows:
        return ""
    return (
        "<p class='cap' style='margin-top:22px'><b>Key results</b>, model against a forecast that "
        "needs no model. The difference is model minus reference with its 95% range; "
        "&ldquo;inconclusive&rdquo; means that range includes zero. Lower is better for log "
        "loss and Brier.</p>"
        "<div class='scroll'><table class='keyres'><thead><tr><th>Stage</th><th>Metric</th>"
        "<th class='n'>Model</th><th class='n'>Reference</th>"
        "<th class='n'>Difference</th><th>Verdict</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _in_short(bt: dict, sc: dict) -> str:
    """Two short paragraphs, computed from the report files so they can't drift."""
    window = bt.get("window") or {}
    rows = bt.get("summary") or []
    model, grid = _row(rows, "method", "model"), _row(rows, "method", "grid")
    if not (model and grid and window):
        return ""
    n = int(model["n_races"])
    mw, gw = round(model["winner_hit"] * n), round(grid["winner_hit"] * n)
    return (
        f"<p class='cap'>Every race from {window['start_season']} to {window['through'][0]} round "
        f"{window['through'][1]} ({n} races) was forecast by models trained only on the races "
        f"before it, with settings chosen on {window['tuned_on'][0]}&ndash;{window['tuned_on'][1]}. "
        f"After qualifying it called {mw} winners against {gw} for simply backing the car on pole, "
        "but the starting grid is a strong forecast on its own and most differences are within "
        "chance. The model's clear gains come earlier: before qualifying, and in forecasting "
        "qualifying itself.</p>"
    )


def _accuracy_table(rows: list[dict]) -> str:
    b = pd.DataFrame(rows)
    if b.empty:
        return ""
    b["approach"] = b["method"].map(lambda x: NAMES.get(x, x))
    b["winner %"] = b["winner_hit"] * 100
    cols = {
        "approach": "approach",
        "ndcg3": "ndcg@3",
        "ndcg5": "ndcg@5",
        "winner %": "winner %",
        "podium_overlap": "podium",
        "win_logloss": "log loss",
        "win_brier": "brier",
        "podium_brier": "podium brier",
    }
    t = b[[c for c in cols if c in b]].rename(columns=cols)
    return rr.table(
        t.round({"winner %": 1}).round(3),
        emphasise="This model",
        best_cols={
            "ndcg@3": "max",
            "ndcg@5": "max",
            "winner %": "max",
            "podium": "max",
            "top 5": "max",
            "spearman": "max",
            "log loss": "min",
            "brier": "min",
            "podium brier": "min",
        },
    )


def _step(n: int, title: str, what: str, why: str, detail: str) -> str:
    return (
        f"<li><div class='sn'>{n}</div><div><h3>{rr.esc(title)}</h3>"
        f"<dl class='ww'><dt>What</dt><dd>{what}</dd><dt>Why</dt><dd>{why}</dd></dl>"
        + (
            f"<details class='tech'><summary>Technical detail</summary><p>{detail}</p></details>"
            if detail
            else ""
        )
        + "</div></li>"
    )


def _calibration_numbers(ex: dict) -> dict[str, float]:
    rows = {r["variant"]: r["win_logloss"] for r in ex.get("calibration") or []}
    return {
        "raw": rows.get("raw (T=1, no simulation)"),
        "rolling": rows.get("rolling temperature (deployed)"),
        "pl": rows.get("rolling, no simulation"),
        "sim": rows.get("rolling, simulation only"),
    }


def pipeline(ex: dict) -> list[tuple[str, str, str, str]]:
    """The nine stages, each with what it does, why it is there, and the detail
    a technical reader wants. Figures come from reports/experiments.json."""
    from . import features, validate

    c = _calibration_numbers(ex)
    cal_why = (
        f"Raw ranker scores read as probabilities are badly over-confident: win log loss "
        f"{c['raw']:.3f} on the reported races, against {c['rolling']:.3f} once calibrated."
        if c["raw"] and c["rolling"]
        else "Raw ranker scores have no probabilistic meaning; the temperature gives them one."
    )
    mc_why = (
        f"The ranker knows nothing of retirements or safety cars. Mixed, the two models beat "
        f"either alone: log loss {c['rolling']:.3f}, against {c['pl']:.3f} for Plackett&ndash;Luce "
        f"only and {c['sim']:.3f} for the simulation only."
        if c["rolling"] and c["pl"] and c["sim"]
        else "The ranker knows nothing of retirements or safety cars; the simulation adds that uncertainty."
    )
    return [
        (
            "Historical data",
            (
                "Results, qualifying, sprints and standings since 2018 from jolpica-f1; the official "
                "starting grid and each session's entry list from OpenF1; this weekend's practice "
                "laps from FastF1, reduced to a few numbers per driver. Stored in DuckDB."
            ),
            (
                "A forecast is only as good as knowing who is racing and where they start. "
                f"{len(validate._CHECKS)} data checks run before anything is modelled."
            ),
            (
                "Raw tables are never overwritten. The field for a race comes from the most official "
                "source available (qualifying, then OpenF1's session entries, then the previous race, "
                "labelled as an assumption); FP1 never sets it, because teams run rookies there."
            ),
        ),
        (
            "Temporal features",
            (
                "One row per driver per race: recent finishing form (median), team form and car pace, "
                "reliability, circuit history and championship standing, each computed only from "
                "earlier races."
            ),
            (
                "Anything computed with knowledge of the race it predicts makes a backtest look better "
                "than the live forecast can be."
            ),
            (
                "Rolling windows shift a whole race before aggregating; team values collapse to one per "
                "team per race first. Two black-box tests: features built from data that stops at race "
                "<i>k</i> must equal features built from everything, for every race up to <i>k</i>; "
                "rewriting one race's result must move no feature of that race or an earlier one."
            ),
        ),
        (
            "Qualifying forecast",
            (
                "An XGBoost ranker (<code>rank:ndcg</code>, one group per qualifying session) orders "
                "the field from qualifying history and, once they have run, this weekend's practice "
                "sessions."
            ),
            (
                "Before qualifying the grid is unknown, and it is the strongest single input to a race. "
                "Practice sharpens this forecast; it adds nothing to the race once the grid is known, so "
                "it is used only here."
            ),
            (
                f"{len(features.QUALI_FEATURES)} features. Practice was tested on the 2024 and 2026 "
                "weekends with data, which fall inside the reported window, so that one choice is not "
                "held out. Before qualifying, every simulated race draws its own grid from this "
                "forecast (Plackett&ndash;Luce with its own fitted temperature)."
            ),
        ),
        (
            "Official starting grid",
            (
                "After qualifying, the grid the race actually starts from: penalties applied, from "
                "jolpica or OpenF1, checked (no driver twice, no two cars in one slot, every car entered, "
                "90% coverage). Until it is published the qualifying order stands in, and the page says so."
            ),
            (
                "Penalties move cars, sometimes a long way. The predicted qualifying order is never used "
                "in place of the real result."
            ),
            (
                "Pit-lane starters go to the back of the grid; they had been read as grid 0, ahead of "
                "pole. Grid and qualifying position are separate inputs, which is how a fast car that "
                "took a penalty is recognised."
            ),
        ),
        (
            "Race learning-to-rank",
            (
                f"An XGBoost ranker (<code>rank:ndcg</code>) with one query group per race, on "
                f"{len(features.RACE_FEATURES)} features including the grid. Five seeds, each "
                "standardised within the race, averaged."
            ),
            (
                "A race is an ordering. Grouping by race trains on every pairwise comparison inside it, "
                "not on about 190 winner labels."
            ),
            (
                "Relevance is 24 &minus; finishing position, so retirements keep their classified place. "
                "Hyper-parameters are fixed (depth 4, 400 trees); a handful of alternatives were compared "
                "on 2021 and 2022&ndash;23 and none was best on both."
            ),
        ),
        (
            "Probability distribution",
            (
                "Scores become a full finishing-position matrix <i>P</i>[driver, position] whose rows "
                "and columns each sum to one. Win, podium, top 5, top 10 and expected finish are all "
                "read from it."
            ),
            (
                "Separately modelled probabilities contradict each other (a podium chance below the win "
                "chance). One distribution cannot."
            ),
            (
                "Plackett&ndash;Luce: <i>P</i>(<i>i</i> leads) = exp(<i>s<sub>i</sub></i>/<i>T</i>) / "
                "&Sigma;<sub><i>j</i></sub> exp(<i>s<sub>j</sub></i>/<i>T</i>), applied down the order. "
                "Full orders are sampled exactly with the Gumbel-max trick. A 0.2% uniform share is "
                "mixed in last so no driver sits at exactly zero."
            ),
        ),
        (
            "Calibration",
            (
                "The temperature <i>T</i> is fitted by maximum likelihood on the winners of the 24 races "
                "before each forecast. Baselines are calibrated the same way."
            ),
            cal_why,
            (
                "Temperature scaling keeps a race's probabilities summing to one; isotonic regression "
                "per driver would not. Fitted only on past races, never on the race being forecast. "
                "Fitted on winners only, which leaves podium and top-10 chances over-confident at the "
                "top end (see Calibration below)."
            ),
        ),
        (
            "Monte Carlo simulation",
            (
                "10,000 simulated races per forecast: pace noise proportional to the field's score "
                "spread, a safety-car draw, an independent retirement draw per car, a pull toward the "
                "grid where overtaking is hard, and before qualifying an uncertain grid."
            ),
            mc_why,
            (
                "Final matrix = <i>w</i>&middot;PL + (1 &minus; <i>w</i>)&middot;MC with <i>w</i> fitted on "
                "2022&ndash;23; a mixture of two doubly-stochastic matrices is still one. This is a "
                "stochastic layer over the learned ranking, not a physics model: no tyres, pit stops, "
                "weather or team orders."
            ),
        ),
        (
            "Evaluation",
            (
                "Walk-forward: every race from 2024 forecast by models trained only on races before it, "
                "scored against baselines that need no model, with 95% race-level bootstrap intervals."
            ),
            (
                "A random split would train on the future. Settings chosen on the reported races would "
                "flatter them."
            ),
            (
                "Settings fitted on 2022&ndash;23, reported from 2024. Ranking: NDCG@3/5, winner called, "
                "podium and top-5 overlap, Spearman, Kendall. Probability: log loss, Brier, reliability "
                "and expected calibration error. A difference counts only when its interval excludes zero."
            ),
        ),
    ]


def _top_bucket(rows: list[dict]) -> dict | None:
    rows = [r for r in rows or [] if int(r.get("n", 0)) >= 20]
    return max(rows, key=lambda r: float(r["stated"]), default=None)


def limitations(bt: dict) -> list[tuple[str, str]]:
    """Limits of the forecast. The evidence-based ones are read off the reports."""
    out = []
    n = int((bt.get("window") or {}).get("n_races") or 0)
    if n:
        out.append(
            (
                "Sample size",
                (
                    f"{n} test races. Enough to separate the model from naive baselines, not enough to "
                    "separate it from the calibrated starting grid on most metrics."
                ),
            )
        )
    rel = bt.get("reliability") or {}
    tops = [(label, _top_bucket(rel.get(key))) for label, key in (("podium", "podium"), ("top ten", "top10"))]
    tops = [(label, b) for label, b in tops if b and float(b["observed"]) < float(b["stated"])]
    if tops:
        out.append(
            (
                "Calibration",
                "Near-certain calls are over-confident: "
                + "; ".join(
                    f"a stated {float(b['stated']):.0%} {label} chance happened {float(b['observed']):.0%} "
                    f"of the time (n={int(b['n'])})"
                    for label, b in tops
                )
                + ". The temperature is fitted on winners only.",
            )
        )
    out.append(
        (
            "Practice data",
            (
                "Only this weekend's sessions are fetched, and the historical test covers the 2024 and "
                "2026 weekends with data. That choice is the least held-out in the project."
            ),
        )
    )
    return out + NOT_MODELLED


NOT_MODELLED = [
    ("Weather", "Not a feature. The model does not know it will rain."),
    (
        "Strategy",
        (
            "No tyre model, no undercut, no stop count. Pit-crew speed was built as a feature and "
            "measured; it cost log loss and was cut."
        ),
    ),
    (
        "Penalties",
        (
            "Grid penalties are in once the official grid is published; until then the grid is the "
            "qualifying order, and the forecast says so. In-race penalties are not modelled."
        ),
    ),
    (
        "Upgrades",
        (
            "Which team improves is not forecast. How far a team's pace can move is in the "
            "championship projection's range."
        ),
    ),
    ("Team orders", "Not represented."),
    (
        "Correlated failures",
        "Each car retires independently; a failure shared by both cars of a team is not modelled.",
    ),
]


def _section(label: str, body: str, note: str = "", band: bool = False) -> str:
    cls = " class='band'" if band else ""
    return (
        f"<section{cls}><div class='lab'><b>{label}</b>{note}</div><div class='body'>{body}</div></section>"
    )


def _top_end_note(rel: dict) -> str:
    """Name the most confident band where the forecast was over-confident beyond chance."""
    misses = []
    for key, label in (("podium", "podium"), ("top10", "top-10")):
        b = _top_bucket(rel.get(key))
        if b and float(b["ci_high"]) < float(b["stated"]):
            misses.append(
                f"{label} calls averaging {float(b['stated']):.0%} came true {float(b['observed']):.0%} of the time"
            )
    if not misses:
        return ""
    return " The clearest miss is at the top end: the most confident " + " and ".join(misses) + "."


def _calibration(bt: dict) -> str:
    rel = bt.get("reliability") or {}
    chart = rr.calibration_tabs(rel)
    if not chart:
        return ""
    points = []
    rows = [r for r in rel.get("podium") or [] if int(r.get("n", 0)) >= 10]
    if rows:
        ok = sum(float(r["ci_low"]) <= float(r["stated"]) <= float(r["ci_high"]) for r in rows)
        points.append(
            f"For podium chances, {ok} of {len(rows)} bands land within what chance alone explains."
        )
    ece = (bt.get("calibration_error") or {}).get("win")
    if ece is not None:
        points.append(
            f"Win chances are the best calibrated: on average the stated chance is {ece * 100:.1f} "
            "percentage points from what happened."
        )
    note = _top_end_note(rel).strip()
    if note:
        points.append(
            note.replace("The clearest miss is at the top end: the", "The weak spot is the").rstrip(".")
        )
    return (
        "<p class='cap'>A forecast that says 30% should come true about 30% of the time. Each row "
        "groups every driver-race since 2024 by the chance the forecast gave, and shows how often it "
        "actually happened. On the ring means the forecast meant what it said.</p>"
        + chart
        + "<ul class='takeaways'>"
        + "".join(f"<li>{p}.</li>" if not p.endswith(".") else f"<li>{p}</li>" for p in points)
        + "</ul>"
    )


def _championship(tb: dict, sc: dict) -> str:
    body = ""
    if tb.get("checkpoints"):
        f = pd.DataFrame(tb["checkpoints"])
        cal = pd.DataFrame(tb.get("calibration") or [])
        body = (
            f"<p class='cap'><b>Who wins.</b> Every season from {int(f['season'].min())} to "
            f"{int(f['season'].max())} was stopped at four points and the title projected from "
            f"the model as it stood then. Across those {len(f)} checkpoints it named the eventual "
            f"champion {f['favourite_was_right'].mean() * 100:.0f}% of the time."
        )
        if not cal.empty:
            top = cal.loc[[pd.to_numeric(cal["claimed"], errors="coerce").idxmax()]]
            n_top = int(pd.to_numeric(top["n"], errors="coerce").iloc[0])
            lo = float(pd.to_numeric(top["ci_low"], errors="coerce").iloc[0])
            body += (
                f" Above 95% it was right {n_top} times out of {n_top}, which is still consistent "
                f"with a true rate as low as {lo:.0%}: read a published 99.9% as "
                "<em>the arithmetic says it's over</em>, not one chance in a thousand."
            )
        missed = f[f["favourite_was_right"] == 0]
        if not missed.empty:
            seasons = sorted({int(x) for x in missed["season"]})
            body += f" Its misses were all in {', '.join(map(str, seasons))}."
        body += "</p>"
    return body + _range_block(sc)


IDEA_FLOW = [
    ("Before qualifying", "Past races + this weekend's practice", "a forecast of the starting grid"),
    ("After qualifying", "The official starting grid + past races", "a forecast of the race"),
    ("Every forecast", "10,000 simulated races", "a chance for every driver and position"),
]

CONCEPTS = [
    (
        "Ranking, not yes/no",
        (
            "Instead of asking of each driver separately &ldquo;will they win?&rdquo;, the model "
            "compares the drivers in the same race and learns their likely order "
            "(learning-to-rank)."
        ),
    ),
    (
        "Chances, not a single pick",
        (
            "It does not only name a winner. It estimates how likely each driver is to finish in "
            "each position, and every percentage on the site is read from that one table."
        ),
    ),
    (
        "Honest percentages",
        (
            "If the forecast says 70%, that should happen about 70% of the time across many "
            "forecasts. Checking this is called calibration, and it is shown below."
        ),
    ),
    (
        "Uncertainty",
        (
            "Races have crashes, retirements and safety cars. Each race is simulated 10,000 times "
            "with that randomness, and the percentages are how often each outcome came up."
        ),
    ),
]


def _idea() -> str:
    flow = "".join(
        f"<li><span class='k'>{rr.esc(when)}</span><b>{rr.esc(inp)}</b><span class='to'>&rarr; {rr.esc(out)}</span></li>"
        for when, inp, out in IDEA_FLOW
    )
    concepts = "".join(f"<div><h3>{rr.esc(t)}</h3><p>{b}</p></div>" for t, b in CONCEPTS)
    return (
        "<p class='lede'>The forecast gets better as more information becomes available.</p>"
        f"<ol class='idea'>{flow}</ol>"
        f"<div class='concepts'>{concepts}</div>"
    )


def build(standalone: bool = True) -> str:
    bt = _load("backtest.json")
    tb = _load("title_backtest.json")
    ex = _load("experiments.json")
    sc = _load("spread_calibration.json")

    s: list[str] = [
        rr.top_bar(None, rr.utcnow(), page="method"),
        "<main class='wrap'>",
        "<header class='mast'>",
        "<div class='kicker'>F1 prediction system &middot; method and accuracy</div>",
        "<h1>How the forecast works</h1>",
        (
            "<p class='sub'>From race results to a published probability in nine steps, then how "
            "well it has done. Every number here is read from the project's evaluation reports.</p>"
        ),
        "</header>",
    ]
    s.append(_section("The idea", _idea()))
    s.append(_section("Results", _in_short(bt, sc) + _key_results(bt), band=True))
    s.append(
        _section(
            "How it works",
            "<ol class='steps'>"
            + "".join(_step(i + 1, *step) for i, step in enumerate(pipeline(ex)))
            + "</ol>",
            "<span>Open a step for the technical detail</span>",
        )
    )
    if bt.get("summary"):
        s.append(
            _section(
                "Race accuracy",
                "<p class='cap'>After qualifying, each race forecast by models trained only on "
                "earlier races, beside simple forecasts that need no model. NDCG rewards the right "
                "drivers in the right order at the top; log loss and Brier grade the probabilities, "
                "and lower is better. The starting grid is hard to beat.</p>"
                + _accuracy_table(bt["summary"]),
            )
        )
    cal = _calibration(bt)
    if cal:
        s.append(_section("Calibration", cal, "<span>Does 30% mean 30%?</span>", band=True))
    champ = _championship(tb, sc)
    if champ:
        s.append(_section("Championship", champ))
    s.append(
        _section(
            "Explanations",
            "<p class='cap'>Each driver's row on the forecast page opens to the inputs that moved "
            "the model's ranking of them most, up and down (SHAP values from XGBoost, summed over "
            "related features, averaged over the five models). They show what the prediction was "
            "associated with, not what causes a result: &ldquo;pushed up by starting "
            "position&rdquo; means the model scored the driver higher for it.</p>",
            band=True,
        )
    )
    s.append(
        _section(
            "Limitations",
            "<div class='scroll'><table class='kv'>"
            + "".join(f"<tr><td>{rr.esc(k)}</td><td class='feat'>{v}</td></tr>" for k, v in limitations(bt))
            + "</table></div>",
        )
    )
    s.append(
        "<footer><span>Data: jolpica-f1 &middot; OpenF1 &middot; FastF1</span>"
        f"<span><a href='index.html'>Forecast</a> &middot; "
        f"<a href='{rr.esc(config.REPO_URL)}'>Source</a></span></footer>"
    )
    s.append("</main>")
    return rr.document("".join(s), standalone=standalone, title="How the forecast works")


def write(path: Path | None = None) -> Path:
    path = path or (config.REPORTS / "method.html")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build())
    log.info("Wrote %s", path)
    return path
