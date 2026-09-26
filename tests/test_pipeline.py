"""The whole pipeline, end to end, on the synthetic championship in f1pred.demo.

No network and no real database, so this runs in CI: raw tables -> validation
-> features -> walk-forward -> live forecast before and after qualifying.
The leakage tests here are the strongest in the suite because they treat the
feature code as a black box: whatever a feature does internally, adding future
races or rewriting a result must not move any feature of an earlier race.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from f1pred import backtest, config, demo, features, predict, probability, store, validate

KEYS = ["season", "round", "driver_id"]


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    saved = {k: getattr(config, k) for k in ("DB_PATH", "PREDICTIONS", "REPORTS", "SEASON_NOW")}
    demo.use(tmp_path_factory.mktemp("demo"))
    demo.build_database()
    yield
    for k, v in saved.items():
        setattr(config, k, v)


@pytest.fixture(scope="module")
def frame(db):
    df = features.build(include_upcoming=True)
    features.save(df)
    return df


def _cols() -> list[str]:
    return sorted(set(features.RACE_FEATURES + features.QUALI_FEATURES))


def _same(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    """Per feature, how many rows differ (NaN equal to NaN)."""
    a = a.set_index(KEYS)[_cols()].sort_index()
    b = b.set_index(KEYS)[_cols()].loc[a.index]
    close = np.isclose(a.astype(float), b.astype(float), equal_nan=True)
    return pd.Series((~close).sum(axis=0), index=a.columns)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def test_the_fixture_passes_validation(db):
    report = validate.run()
    assert report.ok, report.render()


def test_one_row_per_driver_per_race(frame):
    assert not frame.duplicated(KEYS).any()


def test_a_pit_lane_start_is_placed_at_the_back(frame):
    race = frame[(frame.season == demo.SEASONS[2]) & (frame["round"] == 8)]
    with store.connect(read_only=True) as con:
        raw = con.execute(
            "SELECT driver_id FROM raw_results WHERE season = ? AND round = 8 AND grid = 0", [demo.SEASONS[2]]
        ).fetchone()[0]
    assert race.set_index("driver_id").loc[raw, "grid"] == len(race)


def test_a_grid_penalty_separates_grid_from_qualifying(frame):
    race = frame[(frame.season == demo.SEASONS[2]) & (frame["round"] == 4)]
    moved = race[race["grid"] != race["quali_position"]]
    assert not moved.empty
    assert (race["grid_source"] == "results").all()


def test_a_missing_qualifying_time_gives_a_missing_teammate_gap(frame):
    race = frame[(frame.season == demo.SEASONS[1]) & (frame["round"] == 5)]
    no_time = race[race["best_ms"].isna()]
    assert len(no_time) == 1
    mate = race[(race.constructor_id == no_time.constructor_id.iloc[0]) & race.best_ms.notna()]
    assert no_time["quali_gap_to_teammate_pct"].isna().all()
    assert mate["quali_gap_to_teammate_pct"].isna().all(), (
        "a gap was invented with nothing to compare against"
    )


def test_the_faster_teammate_sets_the_reference(frame):
    """One-sided gap: in every garage with two times, the faster car is at 0."""
    timed = frame.dropna(subset=["quali_gap_to_teammate_pct"])
    per_team = timed.groupby(["season", "round", "constructor_id"])["quali_gap_to_teammate_pct"]
    assert (per_team.min() == 0).all()
    assert (timed["quali_gap_to_teammate_pct"] >= 0).all()


# ---------------------------------------------------------------------------
# Leakage, treated as a black box
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cut", [(2023, 6), (2024, 3), (2025, 5)])
def test_features_do_not_change_when_later_races_are_added(db, cut, monkeypatch):
    """Build from data that stops at a race, and from everything: every feature
    of every race up to the cut must be identical. This caught a season-progress
    feature computed from how many races had been run so far."""
    raw = features._load()

    def truncated(r):
        def keep(t):
            return t if t.empty else t[(t["season"] * 100 + t["round"]) <= cut[0] * 100 + cut[1]]

        return replace(
            r,
            results=keep(r.results),
            quali=keep(r.quali),
            standings=keep(r.standings),
            pace=keep(r.pace),
            openf1_grid=keep(r.openf1_grid),
            openf1_entries=keep(r.openf1_entries),
        )

    monkeypatch.setattr(features, "_load", lambda: truncated(raw))
    early = features.build(include_upcoming=False)
    monkeypatch.setattr(features, "_load", lambda: raw)
    full = features.build(include_upcoming=False)
    moved = _same(early, full)
    assert moved.sum() == 0, f"features saw the future: {moved[moved > 0].to_dict()}"


def test_rewriting_a_result_moves_only_later_features(db, monkeypatch):
    raw = features._load()
    target = (demo.SEASONS[1], 6)
    tampered = raw.results.copy()
    mask = (tampered["season"] == target[0]) & (tampered["round"] == target[1])
    for col in ("position", "points"):
        tampered.loc[mask, col] = tampered.loc[mask, col].to_numpy()[::-1]
    tampered.loc[mask, "dnf"] = tampered.loc[mask, "dnf"].to_numpy()[::-1]

    monkeypatch.setattr(features, "_load", lambda: raw)
    before = features.build(include_upcoming=False)
    monkeypatch.setattr(features, "_load", lambda: replace(raw, results=tampered))
    after = features.build(include_upcoming=False)

    key = before["season"] * 100 + before["round"]
    upto = before[key <= target[0] * 100 + target[1]]
    later = before[key > target[0] * 100 + target[1]]
    assert _same(upto, after).sum() == 0, "a feature reacted to its own race's result"
    assert _same(later, after).sum() > 0, "later races should learn from the rewritten result"


# ---------------------------------------------------------------------------
# Walk-forward and the live forecast
# ---------------------------------------------------------------------------
def test_the_walk_forward_grades_every_race_coherently(frame):
    res = backtest.walk_forward(
        frame, demo.SEASONS[-1], retrain_every=3, n_seeds=1, settings=backtest.Settings(n_sims=400), warmup=4
    )
    model_rows = res.races[res.races.method == "model"]
    assert len(model_rows) == demo.ROUNDS - demo.UNRUN
    o = res.outcomes[res.outcomes.method == "model"].groupby(["season", "round"])
    assert np.allclose(o["p_win"].sum(), 1.0)
    assert np.allclose(o["p_podium"].sum(), 3.0)
    assert {"grid", "championship"} <= set(res.races.method)


@pytest.fixture(scope="module")
def prequali(frame):
    return predict.run(n_sims=500)


def test_before_qualifying_the_forecast_says_so(prequali):
    assert prequali.meta["stage"] == "pre_quali"
    assert prequali.meta["grid_source"] == "projected"
    assert prequali.meta["entry_source"] == "previous_race"
    assert not prequali.grid_known


def test_the_live_forecast_is_one_coherent_distribution(prequali):
    m = np.array(prequali.position_matrix)
    assert probability.check_distribution(m, atol=2e-3) == []
    for r in prequali.field_probs:
        assert 0 < r["p_win"] <= r["p_podium"] <= r["p_top5"] <= r["p_top10"] <= 1
        assert r["why"], "every driver gets an explanation"


def test_the_forecast_records_where_it_came_from(prequali):
    prov = prequali.meta["provenance"]
    for key in ("model_version", "feature_version", "training_cutoff", "data", "software", "git_commit"):
        assert key in prov
    assert prov["data"]["results_hash"]


@pytest.fixture(scope="module")
def postquali(frame, prequali):
    """Qualifying is in for the next race: a reserve replaces aurora_one, and the
    official grid (OpenF1) carries a penalty the qualifying order doesn't."""
    season, rnd = prequali.season, prequali.round
    with store.connect(read_only=True) as con:
        last = con.execute(
            "SELECT driver_id, constructor_id FROM raw_results WHERE season = ? AND round = ?",
            [season, rnd - 1],
        ).fetchdf()
    field = last.copy()
    field.loc[field.driver_id == "aurora_one", "driver_id"] = "reserve_x"
    field = field.reset_index(drop=True)
    quali = field.assign(
        season=season,
        round=rnd,
        position=np.arange(1, len(field) + 1),
        q1_ms=90_000 + 50 * np.arange(len(field)),
        q2_ms=None,
        q3_ms=None,
        best_ms=90_000 + 50 * np.arange(len(field)),
    )
    order = field["driver_id"].tolist()
    penalised = order.pop(1)
    order.insert(6, penalised)
    grid = pd.DataFrame(
        {
            "season": season,
            "round": rnd,
            "driver_id": order,
            "position": np.arange(1, len(order) + 1),
            "driver_number": np.arange(1, len(order) + 1),
            "session_key": 1,
            "session_start_utc": pd.Timestamp("2025-01-01"),
            "fetched_at_utc": pd.Timestamp("2025-01-01"),
        }
    )
    with store.connect() as con:
        store.upsert(con, "raw_qualifying", quali, ["season", "round", "driver_id"])
        store.upsert(con, "raw_openf1_grid", grid, ["season", "round", "driver_id"])
    try:
        yield predict.run(n_sims=500), penalised
    finally:
        with store.connect() as con:
            con.execute("DELETE FROM raw_qualifying WHERE season = ? AND round = ?", [season, rnd])
            con.execute("DELETE FROM raw_openf1_grid WHERE season = ? AND round = ?", [season, rnd])


def test_after_qualifying_the_field_follows_the_weekend(postquali):
    pred, _ = postquali
    ids = {r["driver_id"] for r in pred.field_probs}
    assert "reserve_x" in ids and "aurora_one" not in ids
    assert pred.meta["entry_source"] == "qualifying"


def test_after_qualifying_the_official_grid_is_used(postquali):
    pred, penalised = postquali
    assert pred.meta["stage"] == "post_quali"
    assert pred.meta["grid_source"] == "openf1"
    row = next(r for r in pred.field_probs if r["driver_id"] == penalised)
    assert row["grid"] == 7 and row["quali_position"] == 2


def test_a_driver_sitting_out_stays_in_the_title_race(frame):
    """dune_two misses rounds 4-6 of the last season to a stand-in. Projected
    from after round 5, dune_two isn't in the next race's field but keeps their
    points - leaving such a driver out once handed a clinched title to a teammate."""
    import numpy as np

    from f1pred import championship

    season = demo.SEASONS[-1]
    nxt = frame[(frame.season == season) & (frame["round"] == 6)]
    assert "dune_two" not in set(nxt.driver_id)
    out = championship.project(
        nxt["driver_id"].tolist(),
        nxt["constructor_id"].tolist(),
        np.zeros(len(nxt)),
        np.full(len(nxt), 0.05),
        season,
        5,
        n_sims=300,
    )
    drivers = out["drivers"].set_index("driver_id")
    assert "dune_two" in drivers.index
    row = drivers.loc["dune_two"]
    assert row["projected"] == pytest.approx(row["now"]), "a driver not racing scored points"
    assert out["drivers"]["p_title"].sum() == pytest.approx(1.0, abs=2e-3)
