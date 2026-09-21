"""Dashboard assembly.

Two halves. This module grades the logged predictions and decides what belongs
on the page; report_render holds the design system that draws it. They are
split because the first half is about F1 and the second is about typography,
and mixing them is how a report file becomes unreadable.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import config
from . import report_render as rr
from .store import connect, database_exists

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Track record
# ---------------------------------------------------------------------------
def load_predictions() -> list[dict]:
    out = []
    for p in sorted(config.PREDICTIONS.glob("*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            log.warning("skipping unreadable prediction %s", p.name)
    return out


def scoreboard() -> pd.DataFrame:
    """Grade every logged prediction against what actually happened."""
    preds = load_predictions()
    if not preds or not database_exists():
        return pd.DataFrame()

    with connect(read_only=True) as con:
        results = con.execute(
            "SELECT season, round, driver_id, position FROM raw_results WHERE position IS NOT NULL"
        ).fetchdf()
        names = dict(con.execute("SELECT driver_id, max(family_name) FROM raw_drivers GROUP BY 1").fetchall())

    def nice(driver_id: str | None) -> str:
        return names.get(driver_id, driver_id or "—")

    rows = []
    for p in preds:
        actual = results[(results.season == p["season"]) & (results["round"] == p["round"])]
        if actual.empty:
            continue  # race not run yet
        actual_top5 = actual.sort_values("position")["driver_id"].head(5).tolist()
        # Key renamed when the published board went from five rows to ten;
        # predictions logged before that are still on disk and still gradeable.
        board = p.get("race_board") or p.get("race_top5") or []
        pred_top5 = [d["driver_id"] for d in board][:5]
        winner = actual_top5[0]
        rows.append(
            {
                "season": p["season"],
                "round": p["round"],
                "race": p["race_name"],
                "when": p["generated_at_utc"][:10],
                "stage": "post-quali" if p["grid_known"] else "pre-quali",
                "picked": nice(pred_top5[0] if pred_top5 else None),
                "actual": nice(winner),
                "winner_hit": int(bool(pred_top5) and pred_top5[0] == winner),
                "top5_overlap": len(set(pred_top5) & set(actual_top5)),
                "p_on_winner": next((d["p_win"] for d in p["field_probs"] if d["driver_id"] == winner), 0.0),
            }
        )
    # Every logged prediction may still be for a race that has not run, which
    # is the normal state right after publishing one. An empty frame has no
    # columns to sort by, so return it before trying.
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["season", "round"])


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def _section(label: str, caption: str, body: str, note: str = "", band: bool = False) -> str:
    """Label in the margin, content beside it. No cards; alternate sections sit
    on a tinted full-bleed band so the page has rhythm without panels."""
    cls = " class='band'" if band else ""
    return (
        f"<section{cls}>"
        f"<div class='lab'><b>{label}</b>{note}</div><div class='body'>"
        + (f"<p class='cap'>{caption}</p>" if caption else "")
        + body
        + "</div></section>"
    )


def build(
    prediction: dict | None = None,
    backtest_summary: pd.DataFrame | None = None,
    calibration: pd.DataFrame | None = None,
    params: dict | None = None,
    standalone: bool = True,
) -> str:
    board = scoreboard()
    generated = rr.utcnow()
    if prediction and prediction.get("generated_at_utc"):
        try:
            generated = datetime.fromisoformat(prediction["generated_at_utc"])
        except ValueError:
            pass

    s: list[str] = [rr.top_bar(prediction, generated), "<div class='wrap'>"]

    # ---- masthead --------------------------------------------------------
    s.append("<header class='mast'>")
    if prediction:
        s.append(
            f"<div class='kicker'>Round {prediction.get('round', '?')} &middot; "
            f"{rr.esc(prediction.get('season', ''))} &middot; "
            f"{rr.esc(str(prediction.get('circuit_id', '')).replace('_', ' '))}</div>"
        )
    s.append(f"<h1>{rr.esc(prediction['race_name']) if prediction else 'No race scheduled'}</h1>")
    s.append(
        "<p class='sub'>Finishing order as probabilities. Published before the session, "
        "timestamped, graded against the result.</p>"
    )
    s.append("</header>")

    if prediction:
        s.append(
            _section(
                "Race",
                "10,000 simulated races. <a href='method.html'>How these are calculated</a>.",
                rr.race_board(prediction["race_board"]),
                "<span>Top ten of twenty-two</span>",
            )
        )

        s.append(
            _section(
                "Qualifying",
                "One-lap pace. Feeds the projected grid above.",
                rr.quali_board(prediction["quali_board"]),
                band=True,
            )
        )

        # ---- championship -------------------------------------------------
        outlook = prediction.get("season_outlook") or {}
        if outlook:
            panels = (
                "<div class='pair'>"
                + rr.championship_panel("Drivers", outlook["drivers"], "name", "team")
                + rr.championship_panel("Constructors", outlook["constructors"], "team_name", None)
                + "</div>"
            )
            names = [d["name"] for d in outlook["drivers"]]
            strip = rr.title_race(outlook, names[0], names[1] if len(names) > 1 else "")
            s.append(
                _section(
                    "Championship",
                    f"{outlook['n_races']} races left, simulated from current form. Mean, with the "
                    "10th\u201390th percentile beneath it. Sprint points excluded. "
                    "<a href='method.html'>Method</a>.",
                    strip + panels,
                    f"<span>After round {outlook['last_actual_round']}</span>",
                )
            )

            if outlook.get("series"):
                s.append(
                    _section(
                        "Points",
                        "Solid where it happened, dashed where it is projected. "
                        "The band is the 10th\u201390th percentile. Hover for figures.",
                        rr.progression_chart(outlook["series"], outlook["last_actual_round"]),
                        band=True,
                    )
                )

    # ---- graded record ---------------------------------------------------
    if not board.empty:
        show = board[["race", "when", "stage", "picked", "actual", "top5_overlap"]].rename(
            columns={"when": "made", "top5_overlap": "top 5"}
        )
        s.append(
            _section(
                "Results",
                f"{board['winner_hit'].mean() * 100:.0f}% of winners called across {len(board)} "
                "graded races.",
                rr.table(show),
            )
        )

    s.append(
        "<footer>"
        "<span>Data: jolpica-f1 &middot; FastF1</span>"
        "<span><a href='method.html'>Method and accuracy</a> &middot; "
        f"<a href='{rr.esc(config.REPO_URL)}'>Source</a></span></footer>"
    )
    s.append("</div>")

    title = prediction["race_name"] if prediction else "F1 Forecast"
    return rr.document("".join(s), standalone=standalone, title=title)


def write(
    prediction: dict | None = None,
    backtest_summary: pd.DataFrame | None = None,
    calibration: pd.DataFrame | None = None,
    params: dict | None = None,
    path: Path | None = None,
) -> Path:
    path = path or (config.REPORTS / "index.html")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build(prediction, backtest_summary, calibration, params))
    log.info("Wrote %s", path)
    return path
