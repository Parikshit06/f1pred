"""Who is in a race, and where each car starts.

Every forecast rests on these two facts about a weekend. Each has a ladder of
sources, most official first, and each answer carries the name of the rung it
came from, so a forecast records what it was built on.

    entry list                               starting grid
    1. qualifying classification  jolpica    1. race result grid        jolpica
    2. qualifying session         OpenF1     2. official starting grid  OpenF1
    3. latest FP2/FP3/sprint      OpenF1     3. qualifying order        provisional:
    4. previous race's field      provisional                           penalties missing

A rung is used only if it passes its checks; otherwise the next one is tried.
The previous race's field is a last resort and is labelled as such: it is how
a driver who had been replaced came to be forecast for a race they weren't in.

FP1 never sets the entry list. Teams have to run rookies in it, so its driver
list is not the race's.

Qualifying position and starting grid are separate columns throughout. They
differ whenever a penalty is applied, and the race model is trained on the
grid the race actually started from.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Sessions that only race drivers take part in, latest in the weekend first.
ENTRY_SESSIONS = ("S", "SQ", "FP3", "FP2")
MIN_GRID_COVERAGE = 0.9
FIELD_SIZE = (16, 26)
MAX_CARS_PER_TEAM = 2

GRID_SOURCES = ("results", "openf1", "qualifying")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def check_grid(grid: pd.DataFrame, entrants: set[str] | None = None) -> list[str]:
    """Problems with a candidate grid of (driver_id, position); empty if usable.

    Position 0 is a pit-lane start and may repeat. Anything else must be a
    distinct slot. A driver who isn't entered in the race means the grid
    belongs to some other session - a stale grid - and it is refused rather
    than partly used.
    """
    if grid is None or grid.empty:
        return ["no grid"]
    problems = []
    if grid["driver_id"].duplicated().any():
        problems.append("a driver appears twice")
    pos = pd.to_numeric(grid["position"], errors="coerce")
    if ((pos < 0) | (pos > 30)).any():
        problems.append("a position outside 0-30")
    slots = pos[pos > 0]
    if slots.duplicated().any():
        problems.append("two cars in one grid slot")
    if entrants:
        strangers = set(grid["driver_id"]) - set(entrants)
        if strangers:
            problems.append(f"drivers not entered in this race: {sorted(strangers)}")
        placed = set(grid.loc[pos.notna(), "driver_id"]) & set(entrants)
        if len(placed) < MIN_GRID_COVERAGE * len(entrants):
            problems.append(f"covers {len(placed)} of {len(entrants)} entrants")
    return problems


def check_entries(entries: pd.DataFrame, sized: bool = True) -> list[str]:
    """Problems with a candidate entry list of (driver_id, constructor_id).

    `sized` also requires a plausible field size, which catches a session list
    fetched part-way through. The qualifying classification is complete by
    definition, so it is checked for duplicates and team counts only.
    """
    if entries is None or entries.empty:
        return ["no entries"]
    problems = []
    if entries["driver_id"].duplicated().any():
        problems.append("a driver appears twice")
    if entries["constructor_id"].isna().any():
        problems.append("a driver with no team")
    per_team = entries.groupby("constructor_id")["driver_id"].nunique()
    if (per_team > MAX_CARS_PER_TEAM).any():
        problems.append(
            f"more than {MAX_CARS_PER_TEAM} cars in {list(per_team[per_team > MAX_CARS_PER_TEAM].index)}"
        )
    if sized and not FIELD_SIZE[0] <= len(entries) <= FIELD_SIZE[1]:
        problems.append(f"{len(entries)} cars")
    return problems


# ---------------------------------------------------------------------------
# Starting grid
# ---------------------------------------------------------------------------
def _candidates(
    race: pd.DataFrame, quali: pd.DataFrame, openf1: pd.DataFrame
) -> list[tuple[str, pd.DataFrame]]:
    """Candidate grids for one race, most official first, as (driver_id, position).

    The qualifying classification is trimmed to the race's entrants, because a
    driver can qualify and then not start. The two published grids are not:
    a driver in them who isn't in the race means the grid is from somewhere else.
    """
    entrants = set(race["driver_id"])
    results = race.loc[race["grid"].notna(), ["driver_id", "grid"]].rename(columns={"grid": "position"})
    qualifying = quali.loc[quali["driver_id"].isin(entrants), ["driver_id", "quali_position"]].rename(
        columns={"quali_position": "position"}
    )
    return [
        ("results", results),
        ("openf1", openf1[["driver_id", "position"]]),
        ("qualifying", qualifying),
    ]


def resolve_grid(frame: pd.DataFrame, quali: pd.DataFrame, openf1_grid: pd.DataFrame) -> pd.DataFrame:
    """Starting grid for every race in `frame`, with the source it came from.

    frame: one row per entrant (season, round, driver_id, grid), grid from
    results and therefore empty for a race not yet run. Returns frame's keys
    with `grid` (1 = pole) and `grid_source`.

    Pit-lane starters are placed at the back of the grid - the field size -
    rather than left at 0, which every grid-based feature would read as ahead
    of pole.
    """
    keys = ["season", "round"]
    no_quali = pd.DataFrame(columns=["driver_id", "quali_position"])
    no_grid = pd.DataFrame(columns=["driver_id", "position"])
    quali_by = dict(tuple(quali.groupby(keys))) if not quali.empty else {}
    openf1_by = dict(tuple(openf1_grid.groupby(keys))) if not openf1_grid.empty else {}

    out = []
    for key, race in frame.groupby(keys, sort=False):
        entrants = set(race["driver_id"])
        chosen, source = None, None
        for name, cand in _candidates(race, quali_by.get(key, no_quali), openf1_by.get(key, no_grid)):
            problems = check_grid(cand, entrants)
            if not problems:
                chosen, source = cand, name
                break
            if not cand.empty:
                log.debug("%s grid for %s rejected: %s", name, key, "; ".join(problems))

        grid = pd.Series(np.nan, index=race.index)
        if chosen is not None:
            slot = dict(zip(chosen["driver_id"], pd.to_numeric(chosen["position"], errors="coerce")))
            grid = race["driver_id"].map(slot).astype(float)
            # A pit-lane start (0) or a listed car with no slot starts from the back.
            listed = race["driver_id"].isin(slot)
            grid = grid.where(~(listed & (grid.isna() | (grid == 0))), float(len(race)))
        out.append(
            pd.DataFrame(
                {
                    "season": race["season"],
                    "round": race["round"],
                    "driver_id": race["driver_id"],
                    "grid": grid,
                    "grid_source": pd.Series(np.where(grid.notna(), source, None), index=race.index),
                }
            )
        )
    if not out:
        return frame[["season", "round", "driver_id"]].assign(grid=np.nan, grid_source=None)
    return pd.concat(out).loc[frame.index]


# ---------------------------------------------------------------------------
# Entry list for a race not yet run
# ---------------------------------------------------------------------------
def race_entries(
    season: int,
    rnd: int,
    results: pd.DataFrame,
    quali: pd.DataFrame,
    openf1_entries: pd.DataFrame,
) -> tuple[pd.DataFrame, str | None]:
    """(driver_id, constructor_id) for a race without results, and the source.

    results/quali/openf1_entries are the full raw tables; only rows for this
    season are consulted, and only rounds before this one for the fallback.
    """
    cols = ["driver_id", "constructor_id"]

    q = quali[(quali["season"] == season) & (quali["round"] == rnd)] if not quali.empty else quali
    if not q.empty and not check_entries(q[cols], sized=False):
        return q[cols].reset_index(drop=True), "qualifying"

    weekend = (
        openf1_entries[(openf1_entries["season"] == season) & (openf1_entries["round"] == rnd)]
        if not openf1_entries.empty
        else openf1_entries
    )
    for session in ("Q", *ENTRY_SESSIONS):
        s = weekend[weekend["session"] == session] if not weekend.empty else weekend
        if s.empty:
            continue
        problems = check_entries(s[cols])
        if not problems:
            return s[cols].reset_index(drop=True), f"openf1:{session}"
        log.warning("openf1 %s entries for %d r%d rejected: %s", session, season, rnd, "; ".join(problems))

    before = results[(results["season"] == season) & (results["round"] < rnd)]
    if before.empty:
        return pd.DataFrame(columns=cols), None
    latest = before[before["round"] == before["round"].max()]
    return latest[cols].drop_duplicates("driver_id").reset_index(drop=True), "previous_race"
