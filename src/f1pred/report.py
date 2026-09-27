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


def weekend_stage(p: dict) -> str:
    """_stage, with the forecast before any practice told apart from the one
    made with it."""
    stage = _stage(p)
    if stage == "pre_quali" and not (p.get("meta") or {}).get("practice_data"):
        return "pre_practice"
    return stage


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


def page_name(season: int, rnd: int) -> str:
    """The archived page for one race's final forecast."""
    return f"race-{int(season)}-{int(rnd):02d}.html"


def final_forecasts(preds: list[dict]) -> dict[tuple[int, int], dict]:
    """(season, round) -> the last forecast logged before that race started.
    Anything made after the start is not a forecast and is ignored."""
    out: dict[tuple[int, int], dict] = {}
    for p in sorted(preds, key=lambda p: p.get("generated_at_utc", "")):
        start = str(p.get("race_start_utc") or "").replace(" ", "T")
        if start and p.get("generated_at_utc", "") > start:
            continue
        if p.get("season") and p.get("round"):
            out[(int(p["season"]), int(p["round"]))] = p
    return out


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
                "winner_team": next(
                    (f.get("team") for f in first.get("field_probs") or [] if f.get("driver_id") == winner),
                    None,
                ),
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


def status_line(prediction: dict, stage: str, rows: list[dict], finished: dict[str, int]) -> str:
    """One plain sentence of where this forecast stands."""
    meta = prediction.get("meta") or {}
    if finished:
        winner = min(finished, key=finished.get)
        said = next((r.get("p_win") for r in rows if r.get("driver_id") == winner), None)
        name = next((r.get("short") or r.get("name") for r in rows if r.get("driver_id") == winner), winner)
        lead = f"<span class='pill done'>Race finished</span> <b>{rr.esc(name)} won</b>" + (
            f". The forecast gave {rr.pct(said, 0)}." if said is not None else "."
        )
    elif stage == "post_quali":
        lead = f"<span class='pill live'>After qualifying</span> Race starts {_when(prediction.get('race_start_utc'))}."
    elif stage == "pre_practice":
        lead = (
            "<span class='pill'>Before practice</span> Made from past races only; it updates once "
            f"practice has run. Race starts {_when(prediction.get('race_start_utc'))}."
        )
    else:
        lead = (
            "<span class='pill'>After practice</span> The grid is not set yet, so it is "
            f"forecast too. Race starts {_when(prediction.get('race_start_utc'))}."
        )
    grid = meta.get("grid_source")
    note = (
        " Grid taken from the qualifying order."
        if stage in ("post_quali", "result") and grid in (None, "qualifying")
        else ""
    )
    return (
        f"<p class='status'>{lead}</p>"
        f"<p class='meta'>Forecast published {_when(prediction.get('generated_at_utc'))}, before the session, "
        f"from {int(meta.get('n_simulations', 10000)):,} simulated races.{note}</p>"
        + (
            ""
            if coherent(prediction)
            else "<p class='meta legacy'>Logged by an earlier version of the pipeline, whose win chances "
            "were not read from the same simulated finishing order as the other figures. It is shown "
            "exactly as it was published; current forecasts take every figure from one distribution.</p>"
        )
    )


def primary(rows: list[dict], prediction: dict, finished: dict[str, int]) -> str:
    """The favourite, then the top five with one bar each: the chance to win."""
    if not rows:
        return ""
    fav = rows[0]
    grid = fav.get("grid")
    where = (
        f"starts P{int(grid)}"
        if isinstance(grid, (int, float)) and prediction.get("grid_known")
        else f"forecast to start P{int(grid)}"
        if isinstance(grid, (int, float))
        else ""
    )
    card = (
        f"<div class='fav' style='--tc:{rr.team_colour(fav.get('team'))}'>"
        "<div class='k'>Most likely winner</div>"
        f"<div class='nm'>{rr.esc(fav.get('name', ''))}</div>"
        f"<div class='tm'>{rr.esc(rr.team_name(fav.get('team')))}{' &middot; ' + where if where else ''}</div>"
        f"<div class='pc'>{rr.pct(fav.get('p_win') or 0, 0)}</div>"
        "<div class='pcl'>chance to win</div></div>"
    )
    return f"<div class='hero'>{card}{rr.finish_line(rows, bool(finished))}</div>"


def _quantile(dist: list[float], q: float) -> int:
    """The first finishing position whose cumulative chance reaches q."""
    total = 0.0
    for pos, p in enumerate(dist, 1):
        total += float(p)
        if total >= q - 1e-9:
            return pos
    return len(dist)


def _spread(prediction: dict) -> dict[str, tuple[int, int, int]]:
    """driver_id -> (25th, 50th, 75th percentile finish) from the logged
    finishing-position distribution. The median is the typical finish: a
    retirement moves it by one place at most, where it drags the mean."""
    ids = (prediction.get("meta") or {}).get("matrix_driver_ids") or []
    matrix = prediction.get("position_matrix") or []
    if len(ids) != len(matrix):
        return {}
    return {
        d: (_quantile(row, 0.25), _quantile(row, 0.5), _quantile(row, 0.75)) for d, row in zip(ids, matrix)
    }


def coherent(prediction: dict, tol: float = 0.02) -> bool:
    """Whether the published win chances are the ones in the logged
    distribution. Forecasts from the current pipeline always are (tested);
    some logged by an earlier version are not, and are never rewritten."""
    ids = (prediction.get("meta") or {}).get("matrix_driver_ids") or []
    matrix = prediction.get("position_matrix") or []
    if len(ids) != len(matrix) or not matrix:
        return True
    win = {f.get("driver_id"): f.get("p_win") for f in prediction.get("field_probs") or []}
    return all(abs(float(row[0]) - float(win.get(d) or 0)) <= tol for d, row in zip(ids, matrix) if d in win)


def _board_rows(prediction: dict, finished: dict[str, int]) -> list[dict]:
    """Every driver, most likely winner first.

    field_probs has the whole field (with explanations in newer forecasts) but
    only surnames; race_board has full names for the top ten. Older forecasts
    may have only the board, which is then all there is to show.
    """
    listed = prediction.get("race_board") or []
    board = {r.get("driver_id"): r for r in listed if r.get("driver_id")}
    field = prediction.get("field_probs") or []
    spread = _spread(prediction)
    rows = []
    for f in sorted(field, key=lambda r: -(r.get("p_win") or 0)) if field else listed:
        row = {**f, "short": f.get("short") or f.get("name")}
        if f.get("driver_id") in spread:
            row["q25"], row["typical"], row["q75"] = spread[f["driver_id"]]
        if field and f.get("driver_id") in board:
            row["name"] = board[f["driver_id"]].get("name", row.get("name"))
        if finished:
            row["finished"] = finished.get(f.get("driver_id"))
        rows.append(row)
    return rows


SECTIONS = [
    ("race", "Race forecast"),
    ("qualifying", "Qualifying forecast"),
    ("championship", "Championship projection"),
    ("record", "Track record"),
]


def _section(label: str, caption: str, body: str, note: str = "", band: bool = False) -> str:
    """A ruled heading, then the content. No cards; alternate sections sit on a
    tinted full-bleed band so the page has rhythm without panels."""
    cls = " class='band'" if band else ""
    anchor = next((k for k, name in SECTIONS if name == label), "")
    return (
        f"<section{cls}{f' id={anchor!r}' if anchor else ''}>"
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
        stage = "result" if finished else weekend_stage(prediction)

    s: list[str] = [rr.top_bar(prediction, generated, stage), "<main class='wrap'>"]

    # ---- masthead: race, status, the forecast itself -----------------------
    rows = _board_rows(prediction, finished) if prediction else []
    s.append("<header class='mast'>")
    if prediction:
        s.append(
            f"<div class='kicker'>Round {prediction.get('round', '?')} &middot; "
            f"{rr.esc(prediction.get('season', ''))} season</div>"
        )
    s.append(f"<h1>{rr.esc(prediction['race_name']) if prediction else 'No race scheduled'}</h1>")
    if prediction:
        s.append(status_line(prediction, stage or "pre_quali", rows, finished))
        s.append(primary(rows, prediction, finished))
        s.append(rr.stage_track(stage or "pre_quali"))
        present = {"race", "qualifying", "record"}
        if (prediction.get("season_outlook") or {}).get("drivers"):
            present.add("championship")
        s.append(_index(present))
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
        s.append(
            _section(
                "Race forecast",
                "Every driver's chances from 10,000 simulated races. Typical is the median finish: "
                "half the simulated races end there or better.",
                rr.race_board(rows, grid_known=bool(prediction.get("grid_known"))),
                band=True,
            )
        )

        qualified = bool(prediction.get("grid_known"))
        s.append(
            _section(
                "Qualifying forecast",
                (
                    "The qualifying forecast, made before the session, with where each driver actually "
                    "starts. The race forecast above uses the real grid."
                    if qualified
                    else "Chances in qualifying, from past qualifying and this weekend's practice where it "
                    "has run. Each simulated race draws its own grid from this forecast."
                ),
                rr.quali_board(prediction.get("quali_board") or [], qualified=qualified),
                "<span>Predicted order</span>",
            )
        )

        # ---- championship -------------------------------------------------
        outlook = prediction.get("season_outlook") or {}
        if outlook and outlook.get("drivers"):
            d = outlook["drivers"]
            leader, rival = d[0]["name"], (d[1]["name"] if len(d) > 1 else "")
            k, after = outlook.get("clinch_in"), outlook.get("after_round") or 0
            sprints = outlook.get("sprints_left")
            lede = (
                f"{rr.esc(leader)} leads {rr.esc(rival)} by {outlook.get('lead', 0):.0f} points with "
                f"{outlook.get('points_available', 0)} still available over {outlook['n_races']} races"
                + (f" and {sprints} sprint{'s' if sprints != 1 else ''}" if sprints else "")
                + "."
                + (
                    f" The earliest it can be settled is round {after + k}."
                    if k and k < outlook["n_races"]
                    else ""
                )
                + " Projected from 10,000 simulated seasons; the range under each total covers 8 in 10."
            )
            body = (
                f"<p class='cap'>{lede}</p>"
                + "<div class='pair'>"
                + rr.championship_panel("Drivers", d, "name", "team")
                + rr.championship_panel("Constructors", outlook.get("constructors") or [], "team_name", None)
                + "</div>"
                + (
                    rr.progression_chart(outlook["series"], outlook["last_actual_round"])
                    if outlook.get("series")
                    else ""
                )
            )
            s.append(
                _section(
                    "Championship projection",
                    "",
                    body,
                    f"<span>after round {outlook['last_actual_round']}</span>",
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


def _tested() -> tuple[list[dict], dict]:
    """Per-race rows and the summary of the walk-forward test, if it has run."""
    path = config.REPORTS / "backtest.json"
    try:
        bt = json.loads(path.read_text()) if path.exists() else {}
    except json.JSONDecodeError:
        return [], {}
    return bt.get("races") or [], {r["method"]: r for r in bt.get("summary") or []}


def track_record(history: list[dict]) -> str:
    """The live record race by race, then every race the model was tested on."""
    races, summary = _tested()
    body = []
    if history:
        body.append(
            "<h3 class='sub'>Live forecasts</h3><p class='cap'>The chance the forecast gave the eventual "
            "winner at each step of the weekend, and whether its pick was right. Select a race to see "
            "its full forecast beside the result.</p>" + rr.record_table(history)
        )
    else:
        body.append("<p class='cap'>Live forecasts are graded here from the first completed race.</p>")
    if races and summary.get("model"):
        n, hits = len(races), sum(int(r.get("winner_hit") or 0) for r in races)
        grid = summary.get("grid", {}).get("winner_hit")
        body.append(
            f"<p class='cap tested'>Before going live, it was tested on {n} past races, each forecast "
            f"using only the races before it: its pick won {hits}"
            + (f", and backing the car on pole won {round(grid * n)}" if grid is not None else "")
            + ". <a href='method.html'>How good is it?</a></p>"
        )
    lead = "Every forecast is saved before the session and graded after the race."
    return _section("Track record", lead, "".join(body))


def _index(present: set[str]) -> str:
    """The page's sections as one line of links, for scanning."""
    links = "".join(f"<a href='#{k}'>{name}</a>" for k, name in SECTIONS if k in present)
    return f"<nav class='onpage' aria-label='On this page'>{links}</nav>"


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
