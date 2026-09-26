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
        "simulated season draws one pace offset per team, and a smaller one per driver so "
        "teammates can drift apart."
        + (" Still short of 80%." if held["after"]["coverage"] < 0.80 else "")
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


def _clear_side(comp: dict, side: str, lead: str) -> str:
    """A sentence naming every metric where `side` is ahead with an interval clear of zero."""
    names = [METRIC_NAMES.get(m, m) for m, r in comp.items() if r.get("better") == side]
    return f" {lead}{', '.join(names)}." if names else ""


def _in_short(bt: dict, sc: dict) -> str:
    """The summary paragraphs, computed from the report files so they can't drift."""
    window = bt.get("window") or {}
    rows = bt.get("summary") or []
    model, grid = _row(rows, "method", "model"), _row(rows, "method", "grid")
    out = []
    if model and window:
        n = int(model["n_races"])
        out.append(
            f"<p class='cap'>Every race from {window['start_season']} on was forecast by models "
            f"trained only on races before it: {n} races through {window['through'][0]} round "
            f"{window['through'][1]}. The settings were fitted on {window['tuned_on'][0]}&ndash;"
            f"{window['tuned_on'][1]} and never on the races reported.</p>"
        )
    if model and grid:
        n = int(model["n_races"])
        comp = {r["metric"]: r for r in bt.get("comparison_vs_grid") or []}
        ll = comp.get("win_logloss", {})
        clear = ll.get("better") == "model"
        mw, gw = round(model["winner_hit"] * n), round(grid["winner_hit"] * n)
        out.append(
            "<p class='cap'>The bar is the starting grid, given probabilities calibrated the same "
            f"way as the model's. The model called {mw} winners of {n} against the grid's {gw}; "
            f"its win log loss is {model['win_logloss']:.3f} against {grid['win_logloss']:.3f}"
            + (
                f", a difference whose 95% interval ({ll['ci_low']:+.3f} to {ll['ci_high']:+.3f}) "
                f"{'excludes' if clear or ll.get('better') == 'grid' else 'includes'} zero"
                if ll
                else ""
            )
            + ". Most of a race is decided in qualifying, and the numbers say so."
            + _clear_side(comp, "grid", "The grid is clearly better on ")
            + _clear_side(comp, "model", "The model is clearly better on ")
            + "</p>"
        )
    ece = bt.get("calibration_error") or {}
    if ece:
        out.append(
            "<p class='cap'>Calibration, per driver: when it says a driver has an X% chance, the "
            f"average gap between X and what happened is {ece.get('win', 0):.1%} for wins, "
            f"{ece.get('podium', 0):.1%} for podiums and {ece.get('top10', 0):.1%} for the top ten "
            "(expected calibration error).</p>"
        )
    held = (sc.get("held_out") or {}).get("after")
    if held:
        cov = held["coverage"]
        tail = "slightly too narrow" if cov < 0.78 else "slightly too wide" if cov > 0.82 else "about right"
        out.append(
            "<p class='cap'>The season projection's range should hold the final total eight times in "
            f"ten. On seasons it wasn't tuned on it held {cov:.0%}, so it is {tail}.</p>"
        )
    out.append(
        "<p class='cap'>Each forecast is committed to <code>predictions/</code> before its session, "
        "with the time it was made, the data it saw and the code that made it recorded in the file.</p>"
    )
    return "".join(out)


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
        "top5_overlap": "top 5",
        "spearman": "spearman",
        "win_logloss": "log loss",
        "win_brier": "brier",
        "podium_brier": "podium brier",
        "n_races": "races",
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


def _comparison(rows: list[dict], a: str, b: str) -> str:
    """Race-paired differences with intervals, and a verdict per metric."""
    c = pd.DataFrame(rows)
    if c.empty:
        return ""
    pills = []
    for r in c.itertuples():
        side = {"model": "model", a: "model", b: "grid"}.get(r.better, "")
        who = NAMES.get(r.better, r.better)
        text = "no clear difference" if r.better == "unclear" else f"{who} better"
        pills.append(f"<span class='pill {side}'>{METRIC_NAMES.get(r.metric, r.metric)}: {text}</span>")
    t = pd.DataFrame(
        {
            "metric": [METRIC_NAMES.get(m, m) for m in c["metric"]],
            NAMES.get(a, a): c[a].round(3),
            NAMES.get(b, b): c[b].round(3),
            "difference": c["difference"].round(3),
            "95% interval": [f"{lo:+.3f} to {hi:+.3f}" for lo, hi in zip(c["ci_low"], c["ci_high"])],
        }
    )
    return "<div class='verdicts'>" + "".join(pills) + "</div>" + rr.table(t)


def _reliability(bt: dict) -> str:
    rel = bt.get("reliability") or {}
    figs = [
        rr.reliability_chart(rel.get(k) or [], title)
        for k, title in (("win", "Win"), ("podium", "Podium"), ("top10", "Top ten"))
        if rel.get(k)
    ]
    return "<div class='figs'>" + "".join(figs) + "</div>" if figs else ""


def _ablation(ex: dict) -> str:
    """The feature-group ablation on the test seasons, against the full model."""
    ab = ex.get("ablation") or {}
    label = next((k for k in ab if k.startswith("test")), None)
    if not label:
        return ""
    rows = []
    for r in ab[label]:
        ci = r.get("win_logloss_ci")
        rows.append(
            {
                "features": r["variant"],
                "ndcg@5": r["ndcg5"],
                "winner %": r["winner_hit"] * 100,
                "log loss": r["win_logloss"],
                "vs full, log loss": (
                    f"{r['win_logloss_diff']:+.3f} ({ci[0]:+.3f} to {ci[1]:+.3f})" if ci else "\u2014"
                ),
            }
        )
    t = pd.DataFrame(rows).round({"ndcg@5": 3, "winner %": 1, "log loss": 3})
    return rr.table(
        t, emphasise="full model", best_cols={"ndcg@5": "max", "winner %": "max", "log loss": "min"}
    )


def _pre_quali_ablation(ex: dict) -> str:
    """The left-out groups tested again before qualifying, in both windows."""
    rows = []
    for window, table in (ex.get("pre_quali_ablation") or {}).items():
        for r in table:
            ci = r.get("win_logloss_ci")
            if ci:
                rows.append(
                    {
                        "added to the model": r["variant"]
                        .replace("full + ", "")
                        .replace(" (not in model)", ""),
                        "seasons": window,
                        "log loss change": f"{r['win_logloss_diff']:+.3f}",
                        "95% interval": f"{ci[0]:+.3f} to {ci[1]:+.3f}",
                        "verdict": _verdict(r, "win_logloss"),
                    }
                )
    if not rows:
        return ""
    return (
        "<p class='cap' style='margin-top:26px'><b>Before qualifying.</b> The groups left out, "
        "added back when the grid is the qualifying model's projection. The driver's qualifying "
        "record is already inside that projection, so in the race model it counts twice.</p>"
        + rr.table(pd.DataFrame(rows))
    )


def _verdict(row: dict, metric: str, lower_is_better: bool = True) -> str:
    ci = row.get(f"{metric}_ci")
    if not ci:
        return "reference"
    if ci[0] > 0:
        return "worse" if lower_is_better else "better"
    if ci[1] < 0:
        return "better" if lower_is_better else "worse"
    return "no clear difference"


def _choices(ex: dict) -> str:
    """The measured design choices, each with the evidence that decided it."""
    parts = []
    cal = ex.get("calibration") or []
    if cal:
        t = pd.DataFrame(cal)[
            ["variant", "win_logloss", "win_brier", "podium_brier", "ece_win", "ece_podium"]
        ]
        t = t.rename(
            columns={
                "win_logloss": "log loss",
                "win_brier": "brier",
                "podium_brier": "podium brier",
                "ece_win": "calib. error (win)",
                "ece_podium": "calib. error (podium)",
            }
        )
        parts.append(
            "<p class='cap'><b>Calibration and the simulation.</b> The same out-of-sample scores "
            "turned into probabilities five ways, on the reported races. Raw scores are badly "
            "calibrated; the temperature fixes that, and mixing in the simulation beats either "
            "half alone on log loss.</p>"
            + rr.table(t.round(4), emphasise="rolling temperature (deployed)", best_cols={"log loss": "min"})
        )
    practice = ex.get("practice") or {}
    if practice.get("qualifying"):
        rows = []
        for part, label, metric in (
            ("qualifying", "Qualifying forecast", "ndcg5"),
            ("qualifying", "Qualifying forecast", "pole_logloss"),
            ("race", "Race forecast after qualifying", "win_logloss"),
        ):
            with_ = next(
                (r for r in practice[part] if "practice" in r["variant"] and r.get(f"{metric}_ci")), None
            )
            if with_:
                lower = metric.endswith("logloss")
                rows.append(
                    {
                        "forecast": label,
                        "metric": METRIC_NAMES.get(metric, metric.replace("_", " ")),
                        "change with practice": f"{with_[f'{metric}_diff']:+.3f}",
                        "95% interval": f"{with_[f'{metric}_ci'][0]:+.3f} to {with_[f'{metric}_ci'][1]:+.3f}",
                        "verdict": _verdict(with_, metric, lower),
                    }
                )
        parts.append(
            "<p class='cap' style='margin-top:26px'><b>Practice pace.</b> FP1&ndash;FP3 aggregates, "
            f"tested on the {practice.get('weekends_with_practice')} weekends with practice data. They "
            "sharpen the qualifying forecast and add nothing once the real grid is known, so they "
            "feed the qualifying model only.</p>" + rr.table(pd.DataFrame(rows))
        )
    defs = []
    for key, label in (("form_statistic", "Median instead of mean recent form"),):
        block = ex.get(key) or {}
        for window, rows in block.items():
            alt = next((r for r in rows if r.get("win_logloss_ci")), None)
            if alt:
                defs.append(
                    {
                        "change tested": label,
                        "seasons": window,
                        "log loss change": f"{alt['win_logloss_diff']:+.3f}",
                        "95% interval": f"{alt['win_logloss_ci'][0]:+.3f} to {alt['win_logloss_ci'][1]:+.3f}",
                        "verdict": _verdict(alt, "win_logloss"),
                    }
                )
    if defs:
        kept = [
            d["change tested"] for d in defs if d["seasons"].startswith("tuning") and d["verdict"] == "better"
        ]
        outcome = (
            "It didn't, so the original definition stays."
            if not kept
            else f"Kept: {', '.join(rr.esc(k) for k in kept)}."
        )
        parts.append(
            "<p class='cap' style='margin-top:26px'><b>Feature definitions.</b> A change is kept only "
            f"if it improves the tuning seasons with an interval clear of zero. {outcome}</p>"
            + rr.table(pd.DataFrame(defs))
        )
    return "".join(parts)


def _step(n: int, title: str, body: str) -> str:
    return f"<li><div class='sn'>{n}</div><div><h3>{rr.esc(title)}</h3><p>{body}</p></div></li>"


PIPELINE = [
    (
        "Temporal features",
        (
            "One row per driver per race, every value computed from races before it: rolling form, "
            "reliability, qualifying pace, teammate head-to-head, circuit history. Tested by "
            "rebuilding from data that stops at a race and checking nothing before it changes."
        ),
    ),
    (
        "Qualifying model",
        (
            "Before qualifying: an XGBoost ranker (<code>rank:ndcg</code>, one group per session) "
            "predicts the qualifying order from history and, when available, the weekend's practice "
            "pace. Once qualifying has run, its forecast is shown for comparison and never used in "
            "place of the result."
        ),
    ),
    (
        "Race model",
        (
            "An XGBoost ranker grouped by race. After qualifying it sees the official starting grid "
            "(penalties applied, from OpenF1 until jolpica publishes it) and qualifying pace; before, "
            "the qualifying forecast stands in. Five seeds, each standardised within the race, "
            "averaged."
        ),
    ),
    (
        "One distribution",
        (
            "Plackett-Luce turns scores into a full finishing-order distribution, with the "
            "temperature refitted on the 24 races before each forecast. A Monte Carlo of the race "
            "&mdash; pace noise, safety cars, each car's retirement risk, and before qualifying an "
            "uncertain grid &mdash; gives a second. They are mixed at a weight fitted on 2022&ndash;23; "
            "win, podium, top 5, top 10 and expected finish are all read off the result, so they "
            "always agree."
        ),
    ),
    (
        "Project",
        (
            "The rest of the calendar run 10,000 times with the same noise model, sprints included, "
            "from the median of each driver's model scores over their last eight real weekends, "
            "carrying points already scored."
        ),
    ),
]

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


def build(standalone: bool = True) -> str:
    bt = _load("backtest.json")
    tb = _load("title_backtest.json")

    s: list[str] = [
        (
            "<div class='bar'><div class='inner'><b>Method and accuracy</b>"
            "<span class='sep'></span><a href='index.html'>&larr; Back to the forecast</a>"
            "</div></div>"
        ),
        "<div class='wrap'>",
        "<header class='mast'>",
        "<div class='kicker'>f1pred</div>",
        "<h1>Model specification</h1>",
        "<p class='sub'>How the forecast is made, and how well it has done.</p>",
        "</header>",
    ]

    # Plain-language summary first, for readers who don't want the metrics.
    s.append(
        "<section><div class='lab'><b>In short</b></div><div class='body'>"
        + _in_short(bt, _load("spread_calibration.json"))
        + "</div></section>"
    )

    # ---- pipeline --------------------------------------------------------
    s.append(
        "<section><div class='lab'><b>Pipeline</b></div>"
        "<div class='body'><ol class='steps'>"
        + "".join(_step(i + 1, t, b) for i, (t, b) in enumerate(PIPELINE))
        + "</ol></div></section>"
    )

    # ---- race accuracy ---------------------------------------------------
    ex = _load("experiments.json")
    if bt.get("summary"):
        body = (
            "<p class='cap'>After qualifying, each race forecast by models trained only on earlier "
            "races, against forecasts that need no model. Every baseline's probabilities are "
            "calibrated the same way as the model's. NDCG rewards the right drivers in the right "
            "order at the top; log loss and Brier grade the probabilities (lower is better).</p>"
            + _accuracy_table(bt["summary"])
            + "<p class='cap' style='margin-top:26px'><b>Against the grid, race by race.</b> "
            "The difference on each metric with a 95% bootstrap interval over races. Where the "
            "interval includes zero, the honest reading is that nobody is ahead.</p>"
            + _comparison(bt.get("comparison_vs_grid") or [], "model", "grid")
        )
        if bt.get("by_season"):
            by = pd.DataFrame(bt["by_season"])
            mine = by[by["method"] == "model"].set_index("season")
            grid = by[by["method"] == "grid"].set_index("season")
            seasons = pd.DataFrame(
                {
                    "season": mine.index,
                    "ndcg@5": mine["ndcg5"].round(3).to_numpy(),
                    "winner %": (mine["winner_hit"] * 100).round(1).to_numpy(),
                    "grid winner %": (grid["winner_hit"].reindex(mine.index) * 100).round(1).to_numpy(),
                    "log loss": mine["win_logloss"].round(3).to_numpy(),
                    "grid log loss": grid["win_logloss"].reindex(mine.index).round(3).to_numpy(),
                    "races": mine["n_races"].to_numpy(),
                }
            )
            body += (
                "<p class='cap' style='margin-top:26px'>By season. The current one is still running, "
                "and a season of 20-odd races moves the winner rate several points a race.</p>"
                + rr.table(seasons)
            )
        s.append(
            f"<section><div class='lab'><b>Race accuracy</b></div><div class='body'>{body}</div></section>"
        )

    # ---- calibration -----------------------------------------------------
    rel = _reliability(bt)
    if rel:
        ece, gece = bt.get("calibration_error") or {}, bt.get("grid_calibration_error") or {}
        s.append(
            "<section class='band'><div class='lab'><b>Calibration</b></div><div class='body'>"
            "<p class='cap'>Every driver in every race, bucketed by the probability published, "
            "against how often it happened; the line is the 95% interval. On the diagonal is "
            "calibrated, below it over-confident. Expected calibration error: "
            + ", ".join(f"{k} {v:.3f}" for k, v in ece.items())
            + (" (grid baseline: " + ", ".join(f"{k} {v:.3f}" for k, v in gece.items()) + ")" if gece else "")
            + ".</p>"
            + rel
            + "</div></section>"
        )

    # ---- before qualifying -----------------------------------------------
    pre = bt.get("pre_quali") or {}
    quali = bt.get("qualifying") or {}
    if pre.get("summary") or quali.get("summary"):
        body = (
            "<p class='cap'>Before qualifying there is no grid to lean on, so these are harder "
            "forecasts. Graded with every grid and qualifying input replaced by the qualifying "
            "model's projection &mdash; the real result never reaches them.</p>"
        )
        if pre.get("summary"):
            body += _accuracy_table(pre["summary"])
        if quali.get("summary"):
            q = pd.DataFrame(quali["summary"]).rename(
                columns={
                    "method": "qualifying forecast",
                    "pole_hit": "pole %",
                    "pole_logloss": "pole log loss",
                    "ndcg5": "ndcg@5",
                    "top10_overlap": "top 10",
                    "n_races": "sessions",
                }
            )
            q["qualifying forecast"] = q["qualifying forecast"].map(
                {"model": "Qualifying model", "recent_quali_form": "Recent qualifying form"}
            )
            q["pole %"] = q["pole %"] * 100
            body += (
                "<p class='cap' style='margin-top:26px'><b>The qualifying model</b>, against ordering "
                "the field by recent qualifying results.</p>"
                + rr.table(q.round(3), emphasise="Qualifying model")
            )
        s.append(
            f"<section><div class='lab'><b>Before qualifying</b></div><div class='body'>{body}</div></section>"
        )

    # ---- what each input adds --------------------------------------------
    ablation = _ablation(ex)
    if ablation:
        s.append(
            "<section class='band'><div class='lab'><b>What each input adds</b></div><div class='body'>"
            "<p class='cap'>The same walk-forward with feature groups added to the grid, or removed "
            "from the full model. No weights are set by hand; this is how the learned model is "
            "checked. A group whose removal costs nothing is duplicating something else. From "
            "<code>reports/experiments.json</code>, one seed per model.</p>"
            + ablation
            + _pre_quali_ablation(ex)
            + "</div></section>"
        )

    choices = _choices(ex)
    if choices:
        s.append(
            f"<section><div class='lab'><b>Choices, measured</b></div><div class='body'>{choices}</div></section>"
        )

    # ---- the championship ------------------------------------------------
    # Who wins and how close answer the same question, so one section.
    sc = _load("spread_calibration.json")
    if tb.get("checkpoints"):
        f = pd.DataFrame(tb["checkpoints"])
        cal = pd.DataFrame(tb.get("calibration") or [])
        surnames = _surnames()

        def who(d: object) -> str:
            return surnames.get(str(d), str(d).replace("_", " ").title())

        body = (
            "<p class='cap'>The projection makes two claims, and both can be graded against "
            "seasons that have finished. Every completed season was stopped at four points and "
            "the title projected from the model as it stood then.</p>"
            f"<p class='cap'><b>Who wins.</b> Across {len(f)} of those checkpoints it named the "
            f"eventual champion {f['favourite_was_right'].mean() * 100:.0f}% of the time."
        )
        # Figures come off the JSON so the prose can't drift from the table.
        if not cal.empty:
            top = cal.loc[[pd.to_numeric(cal["claimed"], errors="coerce").idxmax()]]
            n_top = int(pd.to_numeric(top["n"], errors="coerce").iloc[0])
            lo = float(pd.to_numeric(top["ci_low"], errors="coerce").iloc[0])
            body += (
                f" Where it claimed more than 95% it was right {n_top} times out of {n_top} "
                f"&mdash; which on {n_top} tries is still consistent with a true rate of {lo:.0%}, "
                "so read a published 99.9% as <em>the arithmetic says this is over</em> rather "
                "than as one chance in a thousand."
            )
        body += "</p>"

        missed = f[f["favourite_was_right"] == 0]
        if not missed.empty:
            items = "; ".join(
                f"{int(r['season'])} r{int(r['after_round'])} favoured "
                f"{rr.esc(who(r['favourite']))} at {r['p_favourite']:.1%}, "
                f"{rr.esc(who(r['champion']))} won"
                for _, r in missed.iterrows()
            )
            body += f"<p class='cap'>It was wrong {len(missed)} times: {items}.</p>"

        body += _range_block(sc)
        s.append(
            f"<section><div class='lab'><b>The championship</b></div><div class='body'>{body}</div></section>"
        )
    elif sc.get("held_out"):
        s.append(
            "<section><div class='lab'><b>The championship</b></div>"
            f"<div class='body'>{_range_block(sc)}</div></section>"
        )

    # ---- limits ----------------------------------------------------------
    s.append(
        "<section class='band'><div class='lab'><b>Not modelled</b></div>"
        "<div class='body'><p class='cap'>Sources of error the probabilities do not "
        "capture.</p>"
        "<div class='scroll'><table class='kv'>"
        + "".join(f"<tr><td>{rr.esc(k)}</td><td class='feat'>{rr.esc(v)}</td></tr>" for k, v in NOT_MODELLED)
        + "</table></div></div></section>"
    )

    s.append(
        "<footer><span>Data: jolpica-f1 &middot; FastF1</span>"
        f"<span><a href='index.html'>Forecast</a> &middot; "
        f"<a href='{rr.esc(config.REPO_URL)}'>Source</a></span></footer>"
    )
    s.append("</div>")
    return rr.document("".join(s), standalone=standalone, title="Method and accuracy")


def write(path: Path | None = None) -> Path:
    path = path or (config.REPORTS / "method.html")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build())
    log.info("Wrote %s", path)
    return path
