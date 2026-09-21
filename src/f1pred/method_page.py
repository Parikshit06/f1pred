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
    from .store import connect

    with connect(read_only=True) as con:
        return dict(con.execute("SELECT driver_id, max(family_name) FROM raw_drivers GROUP BY 1").fetchall())


def _step(n: int, title: str, body: str) -> str:
    return f"<li><div class='sn'>{n}</div><div><h3>{rr.esc(title)}</h3><p>{body}</p></div></li>"


PIPELINE = [
    (
        "Rank",
        "XGBoost ranker, objective <code>rank:ndcg</code>, trained on one group per race so "
        "each race contributes every pairwise comparison inside it rather than a single "
        "winner label. 187 training races. Output is an unbounded score per driver.",
    ),
    (
        "Convert",
        "Plackett-Luce: P(driver leads the field) &prop; exp(score / T). T is fitted by log "
        "loss on held-out races and re-fitted from a trailing 24-race window.",
    ),
    (
        "Simulate",
        "10,000 races. Per run: pace noise at 0.55&times; the spread in model score, a "
        "safety-car draw at the circuit's historical rate, and an independent retirement draw "
        "per car from its own recent DNF rate. Before qualifying, each run samples its own "
        "grid from the qualifying model.",
    ),
    (
        "Blend",
        "Published win probability is a fitted mix of the closed-form ranking and the "
        "simulation. Podium and points probabilities are read off the simulated finishing "
        "positions directly.",
    ),
    (
        "Project",
        "Remaining calendar run 10,000 times with the same noise model, carrying points "
        "already scored. Returns mean, 10th and 90th percentile, expected wins, and title "
        "probability.",
    ),
]

NOT_MODELLED = [
    ("Weather", "Forecasts are ingested but are not a feature. The model does not know it will rain."),
    ("Strategy", "No tyre model, no undercut, no stop count."),
    ("Penalties", "Pre-session grid drops arrive through the grid. In-race decisions do not."),
    ("Upgrades", "Pace is read as it has been, not as it will be after a new floor."),
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
        "<div class='bar'><div class='inner'><b>Method and accuracy</b>"
        "<span class='sep'></span><a href='index.html'>&larr; Back to the forecast</a>"
        "</div></div>",
        "<div class='wrap'>",
        "<header class='mast'>",
        "<div class='kicker'>f1pred</div>",
        "<h1>Model specification</h1>",
        "<p class='sub'>Architecture, features, and measured performance. Each section names "
        "the command that regenerates it.</p>",
        "</header><div class='kerb'></div>",
    ]

    # ---- pipeline --------------------------------------------------------
    s.append(
        "<section><div class='lab'><b>Pipeline</b><span>src/f1pred</span></div>"
        "<div class='body'><ol class='steps'>"
        + "".join(_step(i + 1, t, b) for i, (t, b) in enumerate(PIPELINE))
        + "</ol></div></section>"
    )

    # ---- features --------------------------------------------------------
    from . import features as F

    groups = [
        (
            "Driver form",
            [
                "drv_avg_finish_3",
                "drv_avg_finish_5",
                "drv_positions_gained_5",
                "drv_podium_rate_10",
                "drv_top10_rate_10",
                "drv_points_rate_5",
            ],
        ),
        (
            "Car pace",
            [
                "team_pace_gap_pct",
                "drv_pace_gap_pct",
                "team_pace_trend",
                "team_avg_quali_5",
                "drv_avg_quali_5",
                "drv_pole_rate_10",
            ],
        ),
        ("Reliability", ["drv_dnf_rate_10", "team_dnf_rate_10", "circuit_dnf_rate"]),
        (
            "This circuit",
            [
                "drv_circuit_avg_finish",
                "team_circuit_avg_finish",
                "circuit_overtaking_score",
                "circuit_pole_win_rate",
                "drv_circuit_starts",
            ],
        ),
        ("Known after qualifying", F.GRID_FEATURES),
        ("Practice, when available", F.PRACTICE_FEATURES),
    ]
    rows = "".join(
        f"<tr><td>{rr.esc(name)}</td><td class='feat'>"
        + ", ".join(f"<code>{rr.esc(f)}</code>" for f in items)
        + "</td></tr>"
        for name, items in groups
    )
    s.append(
        "<section class='band'><div class='lab'><b>Features</b>"
        f"<span>{len(set(F.RACE_FEATURES) | set(F.QUALI_FEATURES))} in total</span></div>"
        "<div class='body'><p class='cap'>All rolling windows shift by one race before "
        "aggregating, through a single shared helper. Retirements are excluded from pace "
        "averages and counted separately as unreliability.</p>"
        f"<div class='scroll'><table>{rows}</table></div></div></section>"
    )

    # ---- race accuracy ---------------------------------------------------
    body = (
        "<p class='cap'>Walk-forward. Each race predicted by a model trained only on "
        "earlier races, scored against baselines requiring no model. Lower log loss and "
        "Brier are better. Hyperparameters fitted on 2022&ndash;23, bounded at both ends, "
        "and not re-fitted on the reported window.</p>"
    )
    if bt.get("summary"):
        b = pd.DataFrame(bt["summary"]).rename(
            columns={
                "method": "approach",
                "top5_overlap": "top 5",
                "podium_overlap": "podium",
                "top1_hit": "winner",
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
        cols = [
            c
            for c in ["approach", "top 5", "podium", "winner", "ndcg@5", "log loss", "brier", "races"]
            if c in b.columns
        ]
        body += rr.table(
            b[cols].round(3),
            emphasise="This model",
            best_cols={
                "top 5": "max",
                "podium": "max",
                "winner": "max",
                "ndcg@5": "max",
                "log loss": "min",
                "brier": "min",
            },
        )
    if bt.get("by_season"):
        body += "<p class='cap' style='margin-top:26px'>By season. 2026 is live and incomplete.</p>"
        body += rr.table(pd.DataFrame(bt["by_season"]).round(3), emphasise="This model")
    s.append(
        "<section><div class='lab'><b>Race accuracy</b><span>make backtest</span></div>"
        f"<div class='body'>{body}</div></section>"
    )

    # ---- calibration -----------------------------------------------------
    if bt.get("calibration"):
        c = pd.DataFrame(bt["calibration"])
        c.columns = [str(x) for x in c.columns]
        s.append(
            "<section class='band'><div class='lab'><b>Calibration</b><span>make verify</span></div>"
            "<div class='body'><p class='cap'>Stated probability against observed frequency. "
            "Buckets hold 7&ndash;24 races; read n before reading a row. Pooled over 62 races: "
            "stated 0.538, observed 0.597 &mdash; mildly under-confident. "
            "<code>make verify</code> returns this as a warning, not a pass.</p>"
            + rr.table(c)
            + "</div></section>"
        )

    # ---- title projection, graded ---------------------------------------
    if tb.get("checkpoints"):
        f = pd.DataFrame(tb["checkpoints"])
        cal = pd.DataFrame(tb.get("calibration") or [])
        hit = f["favourite_was_right"].mean()
        body = (
            "<p class='cap'>Each completed season stopped at four checkpoints; title projected "
            "from the model as it stood at that point; compared against the eventual champion. "
            f"{len(f)} checkpoints across {f['season'].nunique()} seasons. Favourite correct "
            f"{hit * 100:.0f}% of the time, Brier {tb.get('brier', 0):.3f}.</p>"
        )
        if not cal.empty:
            body += rr.table(cal)
            body += (
                "<p class='cap' style='margin-top:20px'><b>Reading the high end.</b> 16 of 16 "
                "correct above 95%. On n=16 that is consistent with a true rate as low as 81% "
                "(Wilson, 95%). Treat a published 99.9% as <em>settled on current form and "
                "arithmetic</em>, not as a calibrated one-in-a-thousand. Recorded misses: 2020 "
                "r15 favoured Bottas at 78.4%; 2021 r20 favoured Hamilton at 62.5%; 2025 "
                "favoured Piastri at r8 and r13.</p>"
            )
        show = f[
            [
                "season",
                "after_round",
                "races_left",
                "favourite",
                "p_favourite",
                "champion",
                "favourite_was_right",
            ]
        ].copy()
        surnames = _surnames()
        for col in ("favourite", "champion"):
            show[col] = show[col].map(lambda d: surnames.get(d, d.replace("_", " ").title()))
        show["favourite_was_right"] = show["favourite_was_right"].map({1: "yes", 0: "no"})
        show = show.rename(
            columns={
                "after_round": "after round",
                "races_left": "races left",
                "favourite": "model favourite",
                "p_favourite": "claimed",
                "favourite_was_right": "correct",
            }
        )
        body += rr.table(show.round(3))
        s.append(
            "<section><div class='lab'><b>Title projection</b><span>make title-backtest</span></div>"
            f"<div class='body'>{body}</div></section>"
        )

    # ---- limits ----------------------------------------------------------
    s.append(
        "<section class='band'><div class='lab'><b>Not modelled</b><span>Known gaps</span></div>"
        "<div class='body'><p class='cap'>Sources of error the probabilities do not "
        "capture.</p>"
        "<div class='scroll'><table>"
        + "".join(f"<tr><td>{rr.esc(k)}</td><td class='feat'>{rr.esc(v)}</td></tr>" for k, v in NOT_MODELLED)
        + "</table></div></div></section>"
    )

    # ---- audit -----------------------------------------------------------
    s.append(
        "<section><div class='lab'><b>Audit trail</b><span>predictions/</span></div>"
        "<div class='body'><p class='cap'>Each forecast is written to <code>predictions/</code> "
        "as timestamped JSON and committed before the session runs. Grading happens afterwards "
        "against the classified result. Commit timestamps make the ordering verifiable by a "
        "third party.</p></div></section>"
    )

    s.append(
        "<div class='kerb'></div>"
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
