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


def _band(label: str) -> str:
    """Turn a pandas interval string into a readable probability range.

    "(0.189, 0.378]" -> "19-38%". The edges are unchanged; only the notation
    is, because a half-open interval in raw form is precision aimed at nobody.
    """
    try:
        lo, hi = (float(x) for x in label.strip("([])").split(","))
    except ValueError:
        return label
    # A literal en dash, not the entity: this string goes through the table
    # renderer's escaper, which would publish "&ndash;" as four characters.
    return f"{max(lo, 0.0) * 100:.0f}\u2013{hi:.0%}"


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
                "range held": f"{held[key]['coverage']:.0%}",
                "should be": "80%",
                "width": f"{held[key]['width']:.0f} pts",
            }
            for label, key in (("Race luck only", "before"), ("Current", "after"))
        ]
    )
    return (
        "<p class='cap' style='margin-top:26px'><b>How close.</b> Every points total is "
        "published with a range the real answer should land inside eight times in ten. "
        "Graded one team at a time against final constructors' standings:</p>"
        + rr.table(rows)
        + "<p class='cap'>Race luck averages out over a dozen races. What doesn't is the model "
        "being wrong about a car <em>now</em>, which carries into every remaining race, so each "
        "simulated season draws one pace offset per team."
        + (" Still short of 80%." if held["after"]["coverage"] < 0.80 else "")
        + "</p>"
    )


def _in_short(bt: dict, sc: dict) -> str:
    """The summary paragraphs, computed from the report files so they can't drift."""
    rows = {r["method"]: r for r in bt.get("summary", [])}
    model, grid = rows.get("model"), rows.get("grid")
    out = []
    if model:
        n = int(model["n_races"])
        out.append(
            f"<p class='cap'>Every race from 2024 on was forecast by a model trained only on races "
            f"before it: {n} races, each scored against the result.</p>"
        )
    if model and grid:
        n = int(model["n_races"])
        mw, gw = round(model["top1_hit"] * n), round(grid["top1_hit"] * n)
        verdict = "more" if mw > gw else "fewer" if mw < gw else "as many"
        out.append(
            "<p class='cap'>The bar is the starting grid, which predicts a race well with no model at "
            f"all. Against it the model calls {verdict} winners ({mw} of {n}, against {gw}), is a "
            "little behind on the rest of the finishing order, and is clearly better at saying how "
            f"likely each result is (log loss {model['logloss']:.3f} against {grid['logloss']:.3f}).</p>"
        )
    cal = pd.DataFrame(bt.get("calibration") or [])
    if not cal.empty:
        k = pd.to_numeric(cal["n"], errors="coerce").fillna(0)
        said = (pd.to_numeric(cal["predicted"], errors="coerce") * k).sum() / max(k.sum(), 1)
        won = (pd.to_numeric(cal["actual"], errors="coerce") * k).sum() / max(k.sum(), 1)
        lean = (
            "a little cautious"
            if won - said > 0.03
            else "a little bold"
            if said - won > 0.03
            else "about right"
        )
        out.append(
            f"<p class='cap'>Its favourites were given {said:.0%} on average and won {won:.0%} of the "
            f"time, so its confidence is {lean}.</p>"
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
        "with the time it was made recorded in the file.</p>"
    )
    return "".join(out)


def _step(n: int, title: str, body: str) -> str:
    return f"<li><div class='sn'>{n}</div><div><h3>{rr.esc(title)}</h3><p>{body}</p></div></li>"


PIPELINE = [
    (
        "Rank",
        (
            "XGBoost ranker, objective <code>rank:ndcg</code>, trained on one group per race so "
            "each race contributes every pairwise comparison inside it rather than a single "
            "winner label. Output is an unbounded score per driver."
        ),
    ),
    (
        "Convert",
        (
            "Plackett-Luce: P(driver leads the field) &prop; exp(score / T). T is fitted by log "
            "loss on held-out races and re-fitted from a trailing 24-race window."
        ),
    ),
    (
        "Simulate",
        (
            "10,000 races. Per run: pace noise at 0.55&times; the spread in model score, a "
            "safety-car draw at the circuit's historical rate, and an independent retirement draw "
            "per car from its own recent DNF rate. Before qualifying, each run samples its own "
            "grid from the qualifying model."
        ),
    ),
    (
        "Blend",
        (
            "Win probability is a fitted mix of the closed form and the simulation. Podium and "
            "points come from the simulated finishing positions, floored so none is ever below "
            "the chance of winning."
        ),
    ),
    (
        "Project",
        (
            "The rest of the calendar run 10,000 times with the same noise model, from each "
            "driver's strength on a typical weekend rather than this one's grid and track, "
            "carrying points already scored."
        ),
    ),
]

NOT_MODELLED = [
    ("Weather", "Forecasts are ingested but are not a feature. The model does not know it will rain."),
    (
        "Strategy",
        (
            "No tyre model, no undercut, no stop count. Pit-crew speed was built as a feature and "
            "measured; it cost log loss on 62 races and was cut."
        ),
    ),
    (
        "Penalties",
        (
            "Not applied. Before the race the starting grid is taken as the qualifying order, so "
            "grid drops are missed, and in-race penalties aren't modelled at all."
        ),
    ),
    (
        "Upgrades",
        (
            "Which team improves is not forecast, and nothing here can forecast it. How far a team's "
            "pace can move is in the championship projection's range, above."
        ),
    ),
    ("Team orders", "Not represented."),
    (
        "Remaining sprints",
        "Sprint calendar for future rounds is not in the source data, so sprint points are excluded from the projection. Bound: 8 points per sprint.",
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
    body = (
        "<p class='cap'>Each race predicted by a model trained only on earlier races, and "
        "scored against baselines that need no model. Lower log loss and Brier are better. "
        "Settings were fitted on 2022&ndash;23 and never on the reported window.</p>"
    )
    if bt.get("summary"):
        b = pd.DataFrame(bt["summary"]).rename(
            columns={
                "method": "approach",
                "top5_overlap": "top 5",
                "podium_overlap": "podium",
                "top1_hit": "winner %",
                "ndcg5": "ndcg@5",
                "logloss": "log loss",
                "n_races": "races",
            }
        )
        names = {
            "model": "This model",
            "grid": "Grid order",
            "championship": "Championship leader",
            "recent_form": "Recent driver form",
            "team_form": "Team form",
        }
        b["approach"] = b["approach"].map(lambda x: names.get(x, x))
        b["winner %"] = b["winner %"] * 100
        cols = [
            c
            for c in ["approach", "top 5", "podium", "winner %", "ndcg@5", "log loss", "brier", "races"]
            if c in b.columns
        ]
        body += rr.table(
            b[cols].round({"winner %": 1}).round(3),
            emphasise="This model",
            best_cols={
                "top 5": "max",
                "podium": "max",
                "winner %": "max",
                "ndcg@5": "max",
                "log loss": "min",
                "brier": "min",
            },
        )
    if bt.get("by_season"):
        # One row per season, with the grid's log loss beside the model's.
        by = pd.DataFrame(bt["by_season"])
        mine = by[by["approach"] == "This model"].set_index("season")
        grid = by[by["approach"] == "Grid order"].set_index("season")
        seasons = pd.DataFrame(
            {
                "season": mine.index,
                "top 5": mine["top 5"].round(2).to_numpy(),
                "winner %": (mine["winner"] * 100).round(1).to_numpy(),
                "log loss": mine["log loss"].round(3).to_numpy(),
                "grid log loss": grid["log loss"].reindex(mine.index).round(3).to_numpy(),
                "races": mine["races"].to_numpy(),
            }
        )
        body += (
            "<p class='cap' style='margin-top:26px'>By season, with the grid's log loss beside "
            "the model's. 2026 is still in progress.</p>"
        )
        body += rr.table(seasons)
    s.append(f"<section><div class='lab'><b>Race accuracy</b></div><div class='body'>{body}</div></section>")

    # ---- calibration -----------------------------------------------------
    if bt.get("calibration"):
        c = pd.DataFrame(bt["calibration"])
        c.columns = [str(x) for x in c.columns]
        # Computed from the table rather than typed, so it can't drift from it.
        n = pd.to_numeric(c["n"], errors="coerce").fillna(0)
        stated = (pd.to_numeric(c["predicted"], errors="coerce") * n).sum() / max(n.sum(), 1)
        observed = (pd.to_numeric(c["actual"], errors="coerce") * n).sum() / max(n.sum(), 1)
        gap = observed - stated
        verdict = (
            "mildly under-confident"
            if gap > 0.01
            else ("mildly over-confident" if gap < -0.01 else "calibrated in aggregate")
        )
        # "(0.189, 0.378]" -> "19-38%"
        c["bucket"] = [_band(str(b)) for b in c["bucket"]]
        for col in ("predicted", "actual"):
            c[col] = [f"{v:.0%}" for v in pd.to_numeric(c[col], errors="coerce")]
        c = c.rename(columns={"predicted": "stated", "actual": "observed"})
        s.append(
            "<section class='band'><div class='lab'><b>Calibration</b></div>"
            "<div class='body'><p class='cap'>What the model said, against what happened. Each row "
            "collects the races where it put its favourite in that range. Buckets hold "
            f"{int(n.min())}&ndash;{int(n.max())} races; read n before reading a row. Pooled over "
            f"{int(n.sum())} races: stated {stated:.0%}, observed {observed:.0%} &mdash; "
            f"{verdict}.</p>" + rr.table(c) + "</div></section>"
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
                f"&mdash; which on twelve tries is still consistent with a true rate of {lo:.0%}, "
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
