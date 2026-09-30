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
            "<span class='pill'>Before practice</span> Made from past races only. It updates once "
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
        f"<p class='meta'>Forecast published {_when(prediction.get('generated_at_utc'))}, before the race, "
        f"from {int(meta.get('n_simulations', 10000)):,} simulated races.{note}</p>"
        + (
            ""
            if coherent(prediction)
            else "<p class='legacy'>Archived forecast from an earlier version of the probability pipeline, "
            "in which some columns were calculated separately from the finishing-order distribution. "
            "It is shown unchanged for the record. Current forecasts read every figure from one "
            "consistent distribution.</p>"
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
    return f"<div class='hero'>{card}{rr.win_track(rows, bool(finished))}</div>"


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


def _quali_rows(prediction: dict) -> list[dict]:
    """The logged qualifying board, each row numbered by its place in the
    whole field's expected qualifying order.

    Forecasts logged before 2026-09-30 kept only the ten likeliest pole
    sitters, so counting rows would skip anyone expected to qualify ahead of
    a listed driver but left off the board. field_probs holds every driver's
    expected qualifying position, so the number is taken from there.
    """
    rows = prediction.get("quali_board") or []
    field = [f for f in prediction.get("field_probs") or [] if f.get("q_exp_position") is not None]
    if not field:
        return rows
    order = sorted(field, key=lambda f: f["q_exp_position"])
    rank = {f.get("driver_id"): i + 1 for i, f in enumerate(order)}
    return [{**r, "rank": rank.get(r.get("driver_id"))} for r in rows]


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


# What the site is, before anything about this race: a visitor who has never
# seen it should not have to work that out from the tables.
ABOUT = (
    "<p class='about'><b>Probabilistic forecasts for every Formula 1 Grand Prix.</b> Race, qualifying "
    "and championship chances, updated as the weekend goes on and graded after the race.</p>"
)


def _odds_note(rows: list[dict]) -> str:
    """How far to trust a title chance above 95%, from the title backtest.

    The simulation can print 99.9%, but only a handful of past projections
    were ever that sure, so the page says how many and how they did."""
    if not any(r.get("alive", True) and float(r.get("p_title") or 0) >= 0.95 for r in rows):
        return ""
    try:
        tb = json.loads((config.REPORTS / "title_backtest.json").read_text())
    except (OSError, json.JSONDecodeError):
        return ""
    top = next((b for b in tb.get("calibration") or [] if b.get("bucket") == "over 95%"), None)
    if not top:
        return ""
    n, hit = int(top["n"]), round(float(top["happened"]) * int(top["n"]))
    return (
        f"<p class='odds-note'>Read chances above 95% as very likely, not certain. When past seasons were "
        f"replayed, projections that sure came true {hit} times in {n}, too few to tell 98% from 99.9%. "
        "<a href='method.html'>How it was tested</a></p>"
    )


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


def build(prediction: dict | None = None, archived: bool = False) -> str:
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

    s: list[str] = [
        rr.top_bar(prediction, generated, stage, page="races" if archived else "forecast"),
        "<main class='wrap'>",
    ]

    # ---- masthead: race, status, the forecast itself -----------------------
    rows = _board_rows(prediction, finished) if prediction else []
    s.append("<header class='mast'>")
    if not archived:
        s.append(ABOUT)
    if archived:
        s.append(
            "<p class='archived'>Archived forecast, exactly as it was published. "
            "<a href='index.html'>Latest forecast</a> &middot; <a href='races.html'>All past races</a></p>"
        )
    if prediction:
        s.append(
            f"<div class='kicker'>Round {prediction.get('round', '?')} &middot; "
            f"{rr.esc(prediction.get('season', ''))} season</div>"
        )
    s.append(f"<h1>{rr.esc(prediction['race_name']) if prediction else 'No forecast yet'}</h1>")
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
                "<div class='notice'><b>No forecast published yet.</b> Forecasts are published in "
                "race week, before each session. This page fills in with the first one.</div>",
            )
        )
    if prediction:
        s.append(
            _section(
                "Race forecast",
                "Every driver's chances from 10,000 simulated races."
                + (
                    " Win, podium, top 5 and top 10 are all counted from the same simulated finishing "
                    "orders, so they always agree with each other."
                    if coherent(prediction)
                    else ""
                ),
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
                rr.quali_board(_quali_rows(prediction), qualified=qualified),
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
                + " Projected from 10,000 simulated seasons. The range under each total covers 8 in 10."
            )
            body = (
                f"<p class='cap'>{lede}</p>"
                + "<div class='pair'>"
                + rr.championship_panel("Drivers", d, "name", "team")
                + rr.championship_panel("Constructors", outlook.get("constructors") or [], "team_name", None)
                + "</div>"
                + _odds_note(d + (outlook.get("constructors") or []))
                + (
                    "<p class='chart-cap'>Points after each round for the top five drivers. Solid lines "
                    "are results so far, dashed lines the projection, and each shaded band the range 8 in "
                    "10 simulated seasons fall inside.</p>"
                    "<p class='chart-hint'>Swipe the chart sideways to see the projection.</p>"
                    + rr.progression_chart(outlook["series"], outlook["last_actual_round"])
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

    s.append(rr.footer(config.REPO_URL))
    s.append("</main>")

    title = f"{prediction['race_name']} forecast" if prediction else "F1 forecast"
    return rr.document("".join(s), title=title)


def _tested() -> tuple[list[dict], dict]:
    """Per-race rows and the summary of the walk-forward test, if it has run."""
    path = config.REPORTS / "backtest.json"
    try:
        bt = json.loads(path.read_text()) if path.exists() else {}
    except json.JSONDecodeError:
        return [], {}
    return bt.get("races") or [], {r["method"]: r for r in bt.get("summary") or []}


def _tested_line() -> str:
    """One sentence on the walk-forward test, or "" before it has run."""
    races, summary = _tested()
    if not (races and summary.get("model")):
        return ""
    n, hits = len(races), sum(int(r.get("winner_hit") or 0) for r in races)
    grid = summary.get("grid", {}).get("winner_hit")
    return (
        f" Before going live, it was tested on {n} past races, each forecast using only the races "
        f"before it: its pick after qualifying won {hits}"
        + (f", and backing the car on pole won {round(grid * n)}" if grid is not None else "")
        + "."
    )


def track_record(history: list[dict]) -> str:
    """The live record race by race, then every race the model was tested on."""
    body = []
    if history:
        body.append(
            "<h3 class='sub'>Live forecasts</h3><p class='cap'>The chance the forecast gave the eventual "
            "winner at each step of the weekend, and whether its pick was right. Select a race to see "
            "its full forecast beside the result.</p>" + rr.record_table(history)
        )
    else:
        body.append("<p class='cap'>Live forecasts are graded here from the first completed race.</p>")
    tested = _tested_line()
    if tested:
        body.append(f"<p class='cap tested'>{tested.strip()} <a href='method.html'>How good is it?</a></p>")
    lead = "Every forecast is saved before the session and graded after the race."
    return _section("Track record", lead, "".join(body))


def races_page() -> str:
    """Every graded race, newest first, each linking to its archived page."""
    history = race_history()
    n, hits = len(history), sum(r["final_hit"] for r in history)
    body = (
        rr.record_table(history)
        if history
        else "<div class='notice'>No race has been graded yet. Races appear here once their result is in.</div>"
    )
    s = [
        rr.top_bar(None, rr.utcnow(), page="races"),
        "<main class='wrap'><header class='mast'><h1>Past races</h1>",
        (
            "<p class='status lede'>Every race since the forecast went live: the chance it gave the eventual "
            "winner at each step of the weekend, and whether its pick was right. Select a race to see its "
            "full forecast beside the result.</p>"
        ),
        (
            f"<p class='meta'>The final forecast named the winner in {hits} of {n} race{'s' if n != 1 else ''}."
            f"{_tested_line()}</p>"
            if history
            else ""
        ),
        "</header>",
        f"<div class='racelist'>{body}</div>",
        rr.footer(config.REPO_URL),
        "</main>",
    ]
    return rr.document("".join(s), title="Past races")


def _index(present: set[str]) -> str:
    """The page's sections as one line of links, for scanning."""
    links = "".join(f"<a href='#{k}'>{name}</a>" for k, name in SECTIONS if k in present)
    return f"<nav class='onpage' aria-label='On this page'>{links}</nav>"


def write(prediction: dict | None = None, path: Path | None = None, archived: bool = False) -> Path:
    path = path or (config.REPORTS / "index.html")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build(prediction, archived=archived))
    log.info("Wrote %s", path)
    return path
