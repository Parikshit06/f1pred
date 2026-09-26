"""Who is in the race, and where they start.

Regression tests for two real failures: a forecast built on the previous
race's field (a returning driver left out, the driver they replaced left in),
and a grid read from the qualifying order when penalties had moved cars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1pred import weekend
from f1pred.ingest import openf1

TEAMS = ["red", "blue", "green", "gold", "grey", "pink", "teal", "navy", "lime", "plum"]


def _field(n_teams: int = 10) -> pd.DataFrame:
    """Two drivers per team: red_a, red_b, blue_a, ..."""
    rows = [{"driver_id": f"{t}_{c}", "constructor_id": t} for t in TEAMS[:n_teams] for c in "ab"]
    return pd.DataFrame(rows)


def _results(season: int, rnd: int, field: pd.DataFrame) -> pd.DataFrame:
    return field.assign(season=season, round=rnd, grid=np.arange(1, len(field) + 1, dtype=float))


def _quali(season: int, rnd: int, field: pd.DataFrame) -> pd.DataFrame:
    return field.assign(season=season, round=rnd, quali_position=np.arange(1, len(field) + 1, dtype=float))


def _sessions(season: int, rnd: int, field: pd.DataFrame, session: str) -> pd.DataFrame:
    return field.assign(season=season, round=rnd, session=session)


NO_OPENF1 = pd.DataFrame(columns=["season", "round", "session", "driver_id", "constructor_id"])
NO_QUALI = pd.DataFrame(columns=["season", "round", "driver_id", "constructor_id", "quali_position"])


# ---------------------------------------------------------------------------
# Entry list
# ---------------------------------------------------------------------------
def test_qualifying_sets_the_field_not_the_previous_race():
    """The Baku failure, generalised: last race a stand-in drove red_b's car;
    this weekend red_b is back. The field must follow this weekend."""
    last_race = _field()
    last_race.loc[last_race.driver_id == "red_b", "driver_id"] = "stand_in"
    results = _results(2026, 14, last_race)
    quali = _quali(2026, 15, _field())

    entries, source = weekend.race_entries(2026, 15, results, quali, NO_OPENF1)

    assert source == "qualifying"
    assert "red_b" in set(entries.driver_id), "the returning driver was left out"
    assert "stand_in" not in set(entries.driver_id), "the replaced driver was kept"


def test_a_team_change_takes_the_team_from_this_weekend():
    last_race = _field()
    moved = _field()
    moved.loc[moved.driver_id == "blue_a", "constructor_id"] = "green"
    moved.loc[moved.driver_id == "green_a", "constructor_id"] = "blue"
    entries, _ = weekend.race_entries(
        2026, 15, _results(2026, 14, last_race), _quali(2026, 15, moved), NO_OPENF1
    )
    team = dict(zip(entries.driver_id, entries.constructor_id))
    assert team["blue_a"] == "green" and team["green_a"] == "blue"


def test_before_qualifying_a_later_practice_session_sets_the_field():
    """FP2 shows the real line-up from Friday; the previous race is stale."""
    last_race = _field()
    this_weekend = _field()
    this_weekend.loc[this_weekend.driver_id == "teal_b", "driver_id"] = "rookie"
    openf1_entries = _sessions(2026, 15, this_weekend, "FP2")

    entries, source = weekend.race_entries(2026, 15, _results(2026, 14, last_race), NO_QUALI, openf1_entries)

    assert source == "openf1:FP2"
    assert "rookie" in set(entries.driver_id) and "teal_b" not in set(entries.driver_id)


def test_fp1_alone_never_sets_the_field():
    """Teams must run rookies in FP1, so its list is not the race's."""
    last_race = _field()
    fp1 = _field()
    fp1.loc[fp1.driver_id == "navy_a", "driver_id"] = "fp1_rookie"
    entries, source = weekend.race_entries(
        2026, 15, _results(2026, 14, last_race), NO_QUALI, _sessions(2026, 15, fp1, "FP1")
    )
    assert source == "previous_race"
    assert "fp1_rookie" not in set(entries.driver_id)


def test_the_latest_race_driver_session_wins():
    early, late = _field(), _field()
    late.loc[late.driver_id == "gold_b", "driver_id"] = "late_change"
    sessions = pd.concat([_sessions(2026, 15, early, "FP2"), _sessions(2026, 15, late, "FP3")])
    entries, source = weekend.race_entries(2026, 15, _results(2026, 14, _field()), NO_QUALI, sessions)
    assert source == "openf1:FP3"
    assert "late_change" in set(entries.driver_id)


def test_a_session_list_with_three_cars_in_a_team_is_refused():
    bad = pd.concat([_field(), pd.DataFrame([{"driver_id": "extra", "constructor_id": "red"}])])
    _, source = weekend.race_entries(
        2026, 15, _results(2026, 14, _field()), NO_QUALI, _sessions(2026, 15, bad, "FP2")
    )
    assert source == "previous_race"


def test_with_nothing_from_the_weekend_the_fallback_is_labelled():
    entries, source = weekend.race_entries(2026, 15, _results(2026, 14, _field()), NO_QUALI, NO_OPENF1)
    assert source == "previous_race"
    assert len(entries) == 20


def test_the_fallback_never_reaches_into_a_later_round():
    results = pd.concat([_results(2026, 13, _field()), _results(2026, 16, _field(9))])
    entries, _ = weekend.race_entries(2026, 15, results, NO_QUALI, NO_OPENF1)
    assert len(entries) == 20, "used a round after the one being forecast"


def test_no_history_and_no_weekend_data_means_no_entries():
    entries, source = weekend.race_entries(2027, 1, _results(2026, 23, _field()), NO_QUALI, NO_OPENF1)
    assert entries.empty and source is None


# ---------------------------------------------------------------------------
# Starting grid
# ---------------------------------------------------------------------------
def _frame(grid: list[float] | None, n: int = 20) -> pd.DataFrame:
    field = _field()
    return field.assign(
        season=2026, round=15, grid=np.nan if grid is None else np.asarray(grid, dtype=float)
    )[["season", "round", "driver_id", "grid"]]


def _openf1_grid(order: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {"season": 2026, "round": 15, "driver_id": order, "position": np.arange(1, len(order) + 1)}
    )


def test_a_grid_penalty_moves_the_car_but_not_its_qualifying_position():
    """Qualified third, dropped five places: starts eighth, still qualified third."""
    field = _field()
    quali = _quali(2026, 15, field)
    order = list(field.driver_id)
    penalised = order.pop(2)
    order.insert(7, penalised)

    out = weekend.resolve_grid(_frame(None), quali, _openf1_grid(order))
    grid = dict(zip(out.driver_id, out.grid))
    assert grid[penalised] == 8
    assert quali.set_index("driver_id").loc[penalised, "quali_position"] == 3
    assert set(out.grid_source) == {"openf1"}


def test_the_results_grid_outranks_every_other_source():
    field = _field()
    out = weekend.resolve_grid(
        _frame(list(range(20, 0, -1))), _quali(2026, 15, field), _openf1_grid(list(field.driver_id))
    )
    assert out.grid.iloc[0] == 20
    assert set(out.grid_source) == {"results"}


def test_without_an_official_grid_qualifying_order_is_used_and_labelled():
    out = weekend.resolve_grid(_frame(None), _quali(2026, 15, _field()), _openf1_grid([]))
    assert set(out.grid_source) == {"qualifying"}
    assert out.grid.tolist() == list(range(1, 21))


def test_a_grid_from_another_race_is_refused():
    """A driver who isn't entered means the grid is stale: refuse all of it."""
    order = list(_field().driver_id)
    order[5] = "someone_from_last_week"
    out = weekend.resolve_grid(_frame(None), _quali(2026, 15, _field()), _openf1_grid(order))
    assert set(out.grid_source) == {"qualifying"}


def test_a_grid_with_a_driver_twice_is_refused():
    order = list(_field().driver_id)
    order[19] = order[0]
    assert weekend.check_grid(_openf1_grid(order), set(_field().driver_id))


def test_two_cars_in_one_slot_is_refused():
    grid = _openf1_grid(list(_field().driver_id))
    grid.loc[1, "position"] = 1
    assert "two cars in one grid slot" in weekend.check_grid(grid, set(grid.driver_id))


def test_a_partial_grid_is_refused():
    order = list(_field().driver_id)[:15]
    problems = weekend.check_grid(_openf1_grid(order), set(_field().driver_id))
    assert any("covers 15 of 20" in p for p in problems)


def test_the_results_grid_is_used_when_one_non_starter_is_missing():
    grid = list(range(1, 21))
    grid[19] = np.nan  # did not start
    out = weekend.resolve_grid(_frame(grid), NO_QUALI, _openf1_grid([]))
    assert set(out.grid_source.dropna()) == {"results"}
    assert np.isnan(out.grid.iloc[19])


def test_a_pit_lane_start_is_at_the_back_not_ahead_of_pole():
    grid = list(range(1, 21))
    grid[3] = 0  # started from the pit lane
    out = weekend.resolve_grid(_frame(grid), NO_QUALI, _openf1_grid([]))
    assert out.grid.iloc[3] == 20
    assert out.grid.min() == 1


def test_no_source_at_all_leaves_the_grid_empty():
    out = weekend.resolve_grid(_frame(None), NO_QUALI, _openf1_grid([]))
    assert out.grid.isna().all()
    assert out.grid_source.isna().all()


# ---------------------------------------------------------------------------
# OpenF1 -> jolpica identities
# ---------------------------------------------------------------------------
LOOKUP = pd.DataFrame(
    {
        "driver_id": ["perez", "max_verstappen", "norris", "old_norris_code"],
        "code": ["PER", "VER", "NOR", "NOR"],
        "permanent_number": [11, 1, 4, 99],
        "family_name": ["Pérez", "Verstappen", "Norris", "Norrisson"],
        "season": [2026, 2026, 2026, 2019],
    }
)


def test_drivers_map_by_code_in_the_current_season_first():
    # NOR exists twice across seasons; the current season decides.
    assert openf1.map_driver({"name_acronym": "NOR", "driver_number": 1}, LOOKUP, 2026) == "norris"


def test_a_champions_number_does_not_steal_the_mapping():
    """Norris runs #1 as champion; number 1 belongs to someone else in jolpica."""
    got = openf1.map_driver({"name_acronym": "NOR", "driver_number": 1, "last_name": "Norris"}, LOOKUP, 2026)
    assert got == "norris"


def test_surnames_match_without_accents():
    assert openf1.map_driver({"name_acronym": "", "last_name": "PEREZ"}, LOOKUP, 2026) == "perez"


def test_an_unknown_driver_is_left_out_not_guessed():
    assert (
        openf1.map_driver({"name_acronym": "ZZZ", "driver_number": 77, "last_name": "Nobody"}, LOOKUP, 2026)
        is None
    )


def test_team_names_map_by_majority_of_known_drivers():
    entries = pd.DataFrame(
        {
            "driver_id": ["a1", "a2", "reserve"],
            "team_name": ["Team A Racing", "Team A Racing", "Team A Racing"],
        }
    )
    teams = openf1.map_teams(entries, {"a1": "team_a", "a2": "team_a"})
    assert teams["Team A Racing"] == "team_a"


@pytest.mark.parametrize(
    "offset_hours,expected",
    [(0, 1), (3, 1), (30, 0)],
)
def test_a_weekend_is_matched_on_the_race_start(offset_hours, expected):
    start = pd.Timestamp("2026-09-26 11:00")
    sessions = pd.DataFrame(
        {
            "session_key": [1, 2],
            "code": ["Q", "R"],
            "meeting_key": [7, 7],
            "start_utc": [start - pd.Timedelta(days=1), start + pd.Timedelta(hours=offset_hours)],
        }
    )
    matched = openf1.weekend_sessions(sessions, start)
    assert (not matched.empty) == bool(expected)
