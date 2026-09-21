"""Validation checks must fire on bad data, not just pass on good data.

Each test injects one specific defect into an in-memory database and asserts
the corresponding check catches it.
"""

from __future__ import annotations

import duckdb
import pytest

from f1pred import store, validate


@pytest.fixture
def con():
    c = duckdb.connect(":memory:")
    c.execute(store.SCHEMA)
    c.execute(
        """
        INSERT INTO raw_races (season, round, race_name, circuit_id, race_date)
        VALUES (2024, 1, 'A', 'aaa', DATE '2024-03-02'),
               (2024, 2, 'B', 'bbb', DATE '2024-03-09')
        """
    )
    for rnd in (1, 2):
        for pos in range(1, 21):
            c.execute(
                """
                INSERT INTO raw_results
                  (season, round, driver_id, constructor_id, grid, position,
                   classified, position_text, points, laps, status, finished, dnf)
                VALUES (2024, ?, ?, 'team', ?, ?, TRUE, ?, ?, 58, 'Finished', TRUE, FALSE)
                """,
                [rnd, f"d{pos}", pos, pos, str(pos), max(0, 26 - pos * 2)],
            )
        for pos in range(1, 21):
            c.execute(
                "INSERT INTO raw_qualifying (season, round, driver_id, position) VALUES (2024, ?, ?, ?)",
                [rnd, f"d{pos}", pos],
            )
    for pos in range(1, 21):
        c.execute(
            "INSERT INTO raw_drivers (season, driver_id, code) VALUES (2024, ?, ?)",
            [f"d{pos}", f"D{pos:02d}"],
        )
    yield c
    c.close()


def _names(report) -> set[str]:
    return {f.check for f in report.findings}


def test_clean_data_passes(con):
    report = validate.run(con)
    assert report.ok, report.render()


def test_missing_round_is_caught(con):
    con.execute(
        "INSERT INTO raw_races (season, round, race_name, circuit_id, race_date) "
        "VALUES (2024, 4, 'D', 'ddd', DATE '2024-04-01')"
    )
    report = validate.run(con)
    assert "rounds_contiguous" in _names(report)
    assert not report.ok


def test_two_winners_is_caught(con):
    con.execute("UPDATE raw_results SET position = 1 WHERE season=2024 AND round=1 AND driver_id='d2'")
    report = validate.run(con)
    assert "one_winner_per_race" in _names(report)
    assert "finishing_positions_unique" in _names(report)
    assert not report.ok


def test_impossible_position_is_caught(con):
    con.execute("UPDATE raw_results SET position = 99 WHERE driver_id='d5' AND round=1")
    report = validate.run(con)
    assert "positions_in_range" in _names(report)


def test_negative_points_is_caught(con):
    con.execute("UPDATE raw_results SET points = -5 WHERE driver_id='d3' AND round=1")
    report = validate.run(con)
    assert "positions_in_range" in _names(report)


def test_unknown_driver_is_caught(con):
    con.execute("DELETE FROM raw_drivers WHERE driver_id='d7'")
    report = validate.run(con)
    assert "drivers_are_known" in _names(report)


def test_orphaned_result_is_caught(con):
    con.execute(
        """
        INSERT INTO raw_results
          (season, round, driver_id, constructor_id, grid, position, classified,
           position_text, points, laps, status, finished, dnf)
        VALUES (2024, 9, 'd1', 'team', 1, 1, TRUE, '1', 25, 58, 'Finished', TRUE, FALSE)
        """
    )
    report = validate.run(con)
    assert "results_reference_a_race" in _names(report)


def test_races_out_of_order_is_caught(con):
    con.execute("UPDATE raw_races SET race_date = DATE '2024-01-01' WHERE round = 2")
    report = validate.run(con)
    assert "races_chronological" in _names(report)


def test_hindsight_forecast_is_caught(con):
    con.execute("UPDATE raw_races SET race_start_utc = TIMESTAMP '2024-03-02 14:00:00' WHERE round=1")
    con.execute(
        """
        INSERT INTO raw_forecast (season, round, session, fetched_at_utc, valid_at_utc, temp_c)
        VALUES (2024, 1, 'R', TIMESTAMP '2024-03-02 18:00:00', TIMESTAMP '2024-03-02 14:00:00', 21)
        """
    )
    report = validate.run(con)
    assert "forecasts_predate_their_session" in _names(report)
    assert not report.ok, "a forecast made after the race must be a hard error"


def test_forecast_made_before_the_race_is_fine(con):
    con.execute("UPDATE raw_races SET race_start_utc = TIMESTAMP '2024-03-02 14:00:00' WHERE round=1")
    con.execute(
        """
        INSERT INTO raw_forecast (season, round, session, fetched_at_utc, valid_at_utc, temp_c)
        VALUES (2024, 1, 'R', TIMESTAMP '2024-03-01 06:00:00', TIMESTAMP '2024-03-02 14:00:00', 21)
        """
    )
    report = validate.run(con)
    assert "forecasts_predate_their_session" not in _names(report)


def test_tiny_field_is_caught(con):
    con.execute("DELETE FROM raw_results WHERE round = 2 AND CAST(substr(driver_id,2) AS INT) > 8")
    report = validate.run(con)
    assert "field_size_plausible" in _names(report)
