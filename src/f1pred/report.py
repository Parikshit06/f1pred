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


def _stage(p: dict) -> str:
    return (p.get("meta") or {}).get("stage") or ("post_quali" if p.get("grid_known") else "pre_quali")


def actual_result(season: int, rnd: int) -> dict[str, int]:
    """driver_id -> finishing position, or {} if the race hasn't run."""
    if not database_exists():
        return {}
    with connect(read_only=True) as con:
        rows = con.execute(
            "SELECT driver_id, position FROM raw_results WHERE season = ? AND round = ? AND position IS NOT NULL",
            [season, rnd],
        ).fetchall()
    return {d: int(pos) for d, pos in rows}


def _step_key(p: dict) -> tuple[str, str]:
    """(stage, label) for one logged forecast, in the reader's terms."""
    meta = p.get("meta") or {}
    if _stage(p) == "post_quali":
        grid = meta.get("grid_source")
        note = (
            "qualifying order, penalties pending" if grid == "qualifying" else "official grid" if grid else ""
        )
        return "post_quali", note
    return "pre_quali", "with this weekend's practice" if meta.get(
        "practice_data"
    ) else "before practice data"


def race_history() -> list[dict]:
    """Each completed race with every forecast logged for it, in order.

    Consecutive re-runs of the same step are merged (their range is kept), so
    a card reads as the forecast building up: before practice, after it,
    after qualifying, then the result. Nothing logged is altered.
    """
    preds = load_predictions()
    if not preds or not database_exists():
        return []
    with connect(read_only=True) as con:
        names = dict(con.execute("SELECT driver_id, max(family_name) FROM raw_drivers GROUP BY 1").fetchall())
    return history_of(preds, actual_result, names)


def history_of(preds: list[dict], result_of, names_db: dict[str, str]) -> list[dict]:
    """race_history without the database: result_of(season, round) gives
    driver_id -> finishing position, or {} for a race not yet run."""
    races: dict[tuple, list[dict]] = {}
    for p in preds:
        start = str(p.get("race_start_utc") or "").replace(" ", "T")
        if start and p.get("generated_at_utc", "") > start:
            continue  # made after the start: not a forecast
        races.setdefault((p.get("season"), p.get("round")), []).append(p)

    out = []
    for (season, rnd), group in races.items():
        result = result_of(int(season), int(rnd))
        if not result:
            continue
        winner = min(result, key=result.get)
        top5 = sorted(result, key=result.get)[:5]
        group.sort(key=lambda p: p.get("generated_at_utc", ""))
        steps: list[dict] = []
        for p in group:
            field = {f.get("driver_id"): f for f in p.get("field_probs") or []}
            board = p.get("race_board") or []
            pick = board[0].get("driver_id") if board else None
            p_pick = float(board[0].get("p_win") or 0) if board else 0.0
            stage, note = _step_key(p)
            try:
                made = datetime.fromisoformat(p["generated_at_utc"])
                hours = (
                    pd.Timestamp(p.get("race_start_utc"), tz="UTC") - pd.Timestamp(made)
                ).total_seconds() / 3600
            except (KeyError, TypeError, ValueError):
                made, hours = None, None
            step = {
                "stage": stage,
                "note": note,
                "made": made,
                "hours_before": hours,
                "pick": (field.get(pick) or {}).get("name") or names_db.get(pick, pick),
                "p_pick": p_pick,
                "p_winner": float((field.get(winner) or {}).get("p_win") or 0),
                "hit": pick == winner,
                "top5": len({d.get("driver_id") for d in board[:5]} & set(top5)),
                "runs": 1,
                "p_pick_low": p_pick,
                "p_pick_high": p_pick,
            }
            last = steps[-1] if steps else None
            if last and (last["stage"], last["note"], last["pick"]) == (stage, note, step["pick"]):
                step["runs"] = last["runs"] + 1
                step["p_pick_low"] = min(last["p_pick_low"], p_pick)
                step["p_pick_high"] = max(last["p_pick_high"], p_pick)
                steps[-1] = step
            else:
                steps.append(step)
        first = group[-1]
        win_name = next(
            (f.get("name") for f in first.get("field_probs") or [] if f.get("driver_id") == winner),
            names_db.get(winner, winner),
        )
        out.append(
            {
                "season": season,
                "round": rnd,
                "race": first.get("race_name", f"{season} round {rnd}"),
                "race_start": first.get("race_start_utc"),
                "winner": win_name,
                "steps": steps,
                "final_hit": steps[-1]["hit"],
                "final_p_winner": steps[-1]["p_winner"],
                "final_top5": steps[-1]["top5"],
            }
        )
    return sorted(out, key=lambda r: (r["season"], r["round"]), reverse=True)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
GRID_SOURCES = {
    "results": "official starting grid",
    "openf1": "official starting grid",
    "qualifying": "qualifying order (grid penalties not yet published)",
    "projected": "grid not known yet: projected from the qualifying forecast",
}


def _when(ts: str | None) -> str:
    try:
        return f"{datetime.fromisoformat(str(ts).replace(' ', 'T')):%a %d %b, %H:%M} UTC"
    except (TypeError, ValueError):
        return "\u2014"


def status_line(prediction: dict) -> str:
    """One line of what this forecast was built on, in the reader's terms."""
    meta = prediction.get("meta") or {}
    grid = meta.get("grid_source") or ("qualifying" if _stage(prediction) == "post_quali" else "projected")
    bits = [
        f"Published <b>{_when(prediction.get('generated_at_utc'))}</b>",
        f"Race starts {_when(prediction.get('race_start_utc'))}",
        f"Grid: {rr.esc(GRID_SOURCES.get(grid, str(grid)))}",
        f"{int(meta.get('n_simulations', 10000)):,} simulated races",
    ]
    if meta.get("entry_source") == "previous_race":
        bits.append("field assumed from the last race until entries are published")
    return "<p class='meta'>" + " &middot; ".join(bits) + "</p>"


def primary(rows: list[dict], prediction: dict, finished: dict[str, int]) -> str:
    """The favourite, then the leading drivers' win, podium and top-5 chances."""
    if not rows:
        return ""
    fav = rows[0]
    grid = fav.get("grid")
    grid_txt = f"P{int(grid)}" if isinstance(grid, (int, float)) else "\u2014"
    exp = fav.get("exp_position")
    cells = [
        ("Grid" if prediction.get("grid_known") else "Proj. grid", grid_txt),
        ("Podium", rr.pct(fav.get("p_podium") or 0, 0)),
        ("Exp. finish", f"{float(exp):.1f}" if isinstance(exp, (int, float)) else "\u2014"),
    ]
    if finished:
        pos = finished.get(fav.get("driver_id"))
        cells[2] = ("Finished", f"P{pos}" if pos else "DNF")
    card = (
        f"<div class='fav' style='--tc:{rr.team_colour(fav.get('team'))}'>"
        "<div class='k'>Most likely winner</div>"
        f"<div class='nm'>{rr.esc(fav.get('name', ''))}</div>"
        f"<div class='tm'>{rr.esc(rr.team_name(fav.get('team')))}</div>"
        f"<div class='pc'>{rr.pct(fav.get('p_win') or 0)}<small>to win</small></div>"
        "<dl>" + "".join(f"<div><dt>{k}</dt><dd>{v}</dd></div>" for k, v in cells) + "</dl></div>"
    )
    return f"<div class='lead-grid'>{card}{rr.prob_ladder(rows)}</div>"


def _board_rows(prediction: dict, finished: dict[str, int]) -> list[dict]:
    """Every driver, most likely winner first.

    field_probs has the whole field (with explanations in newer forecasts) but
    only surnames; race_board has full names for the top ten. Older forecasts
    may have only the board, which is then all there is to show.
    """
    listed = prediction.get("race_board") or []
    board = {r.get("driver_id"): r for r in listed if r.get("driver_id")}
    field = prediction.get("field_probs") or []
    rows = []
    for f in sorted(field, key=lambda r: -(r.get("p_win") or 0)) if field else listed:
        row = {**f, "short": f.get("short") or f.get("name")}
        if field and f.get("driver_id") in board:
            row["name"] = board[f["driver_id"]].get("name", row.get("name"))
        if finished:
            row["finished"] = finished.get(f.get("driver_id"))
        rows.append(row)
    return rows


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
    history = race_history()
    generated = rr.utcnow()
    finished: dict[str, int] = {}
    stage = None
    if prediction:
        if prediction.get("generated_at_utc"):
            try:
                generated = datetime.fromisoformat(prediction["generated_at_utc"])
            except ValueError:
                pass
        if prediction.get("season") and prediction.get("round"):
            finished = actual_result(int(prediction["season"]), int(prediction["round"]))
        stage = "result" if finished else _stage(prediction)

    s: list[str] = [rr.top_bar(prediction, generated, stage), "<main class='wrap'>"]

    # ---- masthead: race, status, the forecast itself -----------------------
    rows = _board_rows(prediction, finished) if prediction else []
    s.append("<header class='mast'>")
    if prediction:
        s.append(
            f"<div class='kicker'>{rr.esc(prediction.get('season', ''))} season &middot; "
            f"Round {prediction.get('round', '?')} &middot; "
            f"{rr.esc(str(prediction.get('circuit_id', '')).replace('_', ' '))}</div>"
        )
    s.append(f"<h1>{rr.esc(prediction['race_name']) if prediction else 'No race scheduled'}</h1>")
    s.append(
        "<p class='sub'>An F1 forecasting system. Before each race it estimates every driver's "
        "chance of winning, finishing on the podium and in the top five, publishes that, and "
        "checks itself against the result. <a href='method.html'>How it works</a>.</p>"
    )
    if prediction:
        s.append(rr.stage_track(stage or "pre_quali"))
        s.append(status_line(prediction))
        if finished:
            winner = min(finished, key=finished.get)
            said = next((r.get("p_win") for r in rows if r.get("driver_id") == winner), None)
            name = next((r.get("name") for r in rows if r.get("driver_id") == winner), winner)
            s.append(
                "<div class='notice result'><b>Race finished: "
                f"{rr.esc(name)} won</b>"
                + (f" (the forecast gave {rr.pct(said)})" if said is not None else "")
                + ". The forecast below is exactly as published before the race; the Finished "
                "column was added afterwards. The next forecast appears here in race week.</div>"
            )
        s.append(primary(rows, prediction, finished))
    s.append("</header>")

    if not prediction:
        s.append(
            _section(
                "Race",
                "",
                "<div class='notice'><b>No forecast published yet.</b> Forecasts are made in race "
                "week and committed to <code>predictions/</code> before each session; this page "
                "fills in with the first one.</div>",
            )
        )
    if prediction:
        details = {
            f.get("driver_id"): f.get("why_detail")
            for f in prediction.get("field_probs") or []
            if f.get("why_detail")
        }
        sims = (prediction.get("meta") or {}).get("n_simulations", 10000)
        s.append(
            _section(
                "Every driver",
                f"Every percentage comes from the same {sims:,} simulated races, so they always "
                "agree with each other. Select a driver to see why the model ranked them there.",
                rr.race_board(rows, details, grid_known=bool(prediction.get("grid_known"))),
                band=True,
            )
        )

        s.append(
            _section(
                "Qualifying",
                (
                    "The qualifying model's forecast, made before qualifying, beside where each "
                    "driver actually qualified. Shown for comparison only: the race forecast uses "
                    "the official starting grid."
                    if prediction.get("grid_known")
                    else "The qualifying model's forecast, from past qualifying and this weekend's "
                    "practice pace where it has run. Each simulated race draws its own grid from it; "
                    "it is a forecast, not the official grid."
                ),
                (
                    "<details class='fold'><summary>Show the qualifying forecast</summary>"
                    if prediction.get("grid_known")
                    else ""
                )
                + rr.quali_board(
                    prediction.get("quali_board") or [], qualified=bool(prediction.get("grid_known"))
                )
                + ("</details>" if prediction.get("grid_known") else ""),
                "<span>Predicted order</span>",
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
            sprints = outlook.get("sprints_left")
            s.append(
                _section(
                    "Championship",
                    f"The {outlook['n_races']} remaining races"
                    + (f" and {sprints} sprint{'s' if sprints != 1 else ''}" if sprints else "")
                    + f" simulated {int(outlook.get('n_sims') or 10000):,} times from each driver's "
                    "recent form. Mean points, with the 10th\u201390th percentile beneath. "
                    "<a href='method.html'>Method</a>.",
                    strip
                    + panels
                    + (
                        "<p class='cap' style='margin-top:30px'>Points so far (solid) and projected "
                        "(dashed, with the 10th\u201390th percentile band). Hover or tap for figures.</p>"
                        + rr.progression_chart(outlook["series"], outlook["last_actual_round"])
                        if outlook.get("series")
                        else ""
                    ),
                    f"<span>After round {outlook['last_actual_round']}</span>",
                    band=True,
                )
            )

    # ---- graded record ---------------------------------------------------
    # Shown even when empty, so it doesn't only appear once results look good.
    s.append(track_record(history))

    s.append(
        "<footer>"
        "<span>Data: jolpica-f1 &middot; OpenF1 &middot; FastF1</span>"
        "<span><a href='method.html'>Method and accuracy</a> &middot; "
        f"<a href='{rr.esc(config.REPO_URL)}'>Source</a></span></footer>"
    )
    s.append("</main>")

    title = f"{prediction['race_name']} forecast" if prediction else "F1 forecast"
    return rr.document("".join(s), standalone=standalone, title=title)


def track_record(history: list[dict]) -> str:
    """Race by race: every forecast logged before the race, then the result."""
    if not history:
        return _section(
            "Track record",
            "Nothing graded yet. Each forecast is committed to <code>predictions/</code> "
            "before its session runs and marked against the result afterwards; this fills "
            "in from the first completed race.",
            "",
            band=True,
        )
    n = len(history)
    hits = sum(r["final_hit"] for r in history)
    mean_p = sum(r["final_p_winner"] for r in history) / n
    summary = (
        "<dl class='rec-summary'>"
        f"<div><dt>Races graded</dt><dd>{n}</dd></div>"
        f"<div><dt>Winner called, final forecast</dt><dd>{hits} of {n}</dd></div>"
        f"<div><dt>Final forecast gave the winner</dt><dd>{rr.pct(mean_p, 0)}<small>on average</small></dd></div>"
        "</dl>"
    )
    return _section(
        "Track record",
        "Every forecast logged before each race, in the order it was published, then the result "
        "observed afterwards. The bar is the chance each forecast gave the driver who went on to "
        "win. Files in <code>predictions/</code> are never edited. "
        + ("Too few races yet to read anything into the hit rate." if n < 10 else ""),
        summary + rr.record_cards(history),
        f"<span>{n} race{'s' if n != 1 else ''}</span>",
        band=True,
    )


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
