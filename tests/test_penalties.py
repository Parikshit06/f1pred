"""Grid penalties announced before the official grid. The failures worth
guarding are a penalised car forecast from where it qualified, a penalty
applied twice, and the announcement being used to quietly rewrite the record.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from f1pred import penalties, simulate

BACK = np.inf


def test_a_back_of_grid_penalty_sends_the_car_last_and_moves_everyone_up():
    grid = np.array([1.0, 2.0, 3.0, 4.0])
    assert penalties.apply(grid, np.array([BACK, 0, 0, 0])).tolist() == [4, 1, 2, 3]


def test_a_place_penalty_drops_the_car_behind_the_one_that_qualified_there():
    """Five places from pole is sixth: the car that qualified sixth moves up."""
    grid = np.arange(1.0, 9.0)
    out = penalties.apply(grid, np.array([5, 0, 0, 0, 0, 0, 0, 0]))
    assert out[0] == 6
    assert out[5] == 5
    assert sorted(out.tolist()) == list(range(1, 9))


def test_two_back_of_grid_starters_keep_their_qualifying_order():
    grid = np.array([3.0, 1.0, 2.0, 4.0])
    out = penalties.apply(grid, np.array([BACK, BACK, 0, 0]))
    assert out.tolist() == [4, 3, 1, 2]


def test_every_drawn_grid_is_penalised_on_its_own():
    grids = np.array([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])
    out = penalties.apply(grids, np.array([0, BACK, 0]))
    assert out.tolist() == [[1, 3, 2], [2, 3, 1]]


def test_no_penalty_leaves_the_grid_alone():
    grid = np.array([2.0, 1.0, 3.0])
    assert penalties.apply(grid, np.zeros(3)).tolist() == [2, 1, 3]


def _write(tmp_path, rows):
    path = tmp_path / "grid_penalties.json"
    path.write_text(json.dumps(rows))
    return path


def _row(**over):
    row = {
        "season": 2026,
        "round": 17,
        "driver_id": "russell",
        "penalty": "back",
        "reason": "Power unit change",
        "source": "https://example.com/news",
        "announced": "2026-10-07",
    }
    return {**row, **over}


def test_penalties_are_read_for_their_own_race_only(tmp_path):
    path = _write(tmp_path, [_row(), _row(round=18, driver_id="norris", penalty=10)])
    got = penalties.for_race(2026, 17, path)
    assert [(p.driver_id, p.places) for p in got] == [("russell", None)]
    assert penalties.for_race(2026, 16, path) == []
    assert penalties.places_for(["norris", "russell"], got).tolist() == [0, BACK]


def test_a_malformed_penalty_fails_loudly(tmp_path):
    path = _write(tmp_path, [_row(penalty=-3)])
    with pytest.raises(ValueError):
        penalties.for_race(2026, 17, path)


def test_no_file_means_no_penalties(tmp_path):
    assert penalties.for_race(2026, 17, tmp_path / "missing.json") == []


def test_the_simulation_starts_a_penalised_favourite_from_the_back():
    """Before qualifying the grid is drawn from the qualifying forecast. A
    penalised car must start every one of those races from the back."""
    n = 6
    inputs = simulate.SimInputs(
        driver_ids=[f"d{i}" for i in range(n)],
        scores=np.linspace(3.0, 0.0, n),
        dnf_prob=np.zeros(n),
        grid_scores=np.array([10.0, 0.0, -1.0, -2.0, -3.0, -4.0]),
        overtaking_score=1.0,
        safety_car_prob=0.0,
    )
    free = simulate.simulate_matrix(inputs, n_sims=4000, grid_temperature=0.1)
    inputs.grid_penalty = np.array([BACK, 0, 0, 0, 0, 0])
    held = simulate.simulate_matrix(inputs, n_sims=4000, grid_temperature=0.1)
    assert held[0, 0] < free[0, 0] - 0.2
    assert np.allclose(held.sum(axis=1), 1.0)


def _forecast(tmp_path, monkeypatch, generated, applied):
    from f1pred import config
    from f1pred.predict import Prediction

    monkeypatch.setattr(config, "PREDICTIONS", tmp_path)
    return Prediction(
        season=2026,
        round=17,
        race_name="Singapore Grand Prix",
        circuit_id="marina_bay",
        race_start_utc=(datetime.now(UTC) + timedelta(days=3)).isoformat(),
        generated_at_utc=generated.isoformat(timespec="seconds"),
        grid_known=False,
        meta={"practice_data": False, "grid_penalties": applied},
    )


def test_a_newly_announced_penalty_logs_a_new_forecast_and_keeps_the_old_one(tmp_path, monkeypatch):
    now = datetime.now(UTC)
    first = _forecast(tmp_path, monkeypatch, now - timedelta(hours=20), [])
    kept = first.save()
    before = kept.read_text()

    russell = [{"driver_id": "russell", "places": None}]
    updated = _forecast(tmp_path, monkeypatch, now, russell)
    new = updated.save()
    assert new is not None and new != kept
    assert kept.read_text() == before, "the earlier forecast was rewritten"

    again = _forecast(tmp_path, monkeypatch, now + timedelta(hours=1), russell)
    assert again.save() == kept, "the same penalty reopened the stage twice"
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_a_back_of_grid_starter_is_forecast_to_hold_back_in_qualifying_some_of_the_time():
    """Front-runners starting from the back often don't push in qualifying.
    The fastest car keeps a real pole chance, but a smaller one, and the
    distribution stays coherent."""
    ids = [f"d{i}" for i in range(20)]
    scores = np.linspace(4.0, 0.0, 20)
    usual = simulate.ranking_forecast(ids, scores, 0.5)
    held = simulate.ranking_forecast(ids, scores, 0.5, held_back=[0])
    assert held.matrix[0, 0] < usual.matrix[0, 0] * 0.6
    assert held.matrix[0, 0] > 0.05
    assert held.column("exp_position")[0] > usual.column("exp_position")[0] + 3
    assert np.allclose(held.matrix.sum(axis=0), 1.0)
    assert np.allclose(held.matrix.sum(axis=1), 1.0)


def test_a_held_back_lap_lands_around_fourteenth_of_twenty():
    scores = np.linspace(4.0, 0.0, 20)
    out = penalties.held_back_scores(scores, [0])
    assert int((out > out[0]).sum()) + 1 == 14
