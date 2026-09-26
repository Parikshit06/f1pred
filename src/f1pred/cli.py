"""Command line entry point.

python -m f1pred.cli ingest-jolpica --seasons 2018-2026
python -m f1pred.cli ingest-fastf1  --seasons 2024-2026
python -m f1pred.cli validate
python -m f1pred.cli build-features
python -m f1pred.cli backtest --start-season 2024 --tune --tune-season 2022
python -m f1pred.cli title-backtest
python -m f1pred.cli predict --next
python -m f1pred.cli dashboard
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings

from . import config, store


def _parse_seasons(text: str) -> list[int]:
    seasons: list[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            lo, hi = (int(x) for x in chunk.split("-", 1))
            seasons.extend(range(lo, hi + 1))
        elif chunk:
            seasons.append(int(chunk))
    return sorted(set(seasons))


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    warnings.filterwarnings("ignore", category=FutureWarning)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="f1pred")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    default_range = f"{config.FIRST_SEASON}-{config.CURRENT_SEASON}"

    p = sub.add_parser("ingest-jolpica", help="results, quali, sprints, pit stops, standings")
    p.add_argument("--seasons", default=default_range)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("ingest-fastf1", help="practice/quali pace and session weather (slow)")
    p.add_argument("--seasons", default=default_range)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("forecast", help="Open-Meteo forecast for upcoming sessions")
    p.add_argument("--days-ahead", type=int, default=10)

    sub.add_parser("init", help="create the database schema")
    sub.add_parser("status", help="row counts per table")

    p = sub.add_parser("validate", help="run data quality checks")
    p.add_argument("--quiet", action="store_true", help="hide example rows")

    sub.add_parser("build-features", help="rebuild the feature table")
    sub.add_parser(
        "repair",
        help="recompute finished/retired flags from stored status text (idempotent)",
    )

    p = sub.add_parser("backtest", help="walk-forward evaluation against baselines")
    p.add_argument("--start-season", type=int, default=2022)
    p.add_argument("--retrain-every", type=int, default=1)
    p.add_argument("--tune", action="store_true", help="fit temperature, blend and recency first")
    p.add_argument(
        "--tune-season",
        type=int,
        help="season to fit settings on; must precede --start-season (default: the season before)",
    )
    p.add_argument("--sims", type=int, default=3000)

    p = sub.add_parser("predict", help="forecast a race and log it")
    p.add_argument(
        "--force-log",
        action="store_true",
        help="write a new forecast file even if this race and stage already has one",
    )
    p.add_argument("--next", action="store_true", help="the next unrun race")
    p.add_argument("--season", type=int)
    p.add_argument("--round", type=int)
    p.add_argument("--sims", type=int, default=config.N_SIMULATIONS)

    p = sub.add_parser("bias", help="per-driver bias report (is it favouring stars?)")
    p.add_argument("--start-season", type=int, default=2024)
    p.add_argument("--retrain-every", type=int, default=3)

    p = sub.add_parser("verify", help="full model verification: leakage, bias, weighting, accuracy")
    p.add_argument("--start-season", type=int, default=2024)
    p.add_argument("--sims", type=int, default=3000)
    p.add_argument("--retrain-every", type=int, default=3)

    p = sub.add_parser(
        "title-backtest",
        help="grade the championship projection against completed seasons",
    )
    p.add_argument("--sims", type=int, default=4000)

    p = sub.add_parser(
        "calibrate-spread",
        help="re-fit the championship projection's range against real final standings",
    )
    p.add_argument("--sims", type=int, default=6000)

    sub.add_parser("dashboard", help="render reports/index.html and reports/method.html")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    store.init_db()

    if args.command == "init":
        print(f"Database ready at {config.DB_PATH}")
        return 0

    if args.command == "status":
        print(store.table_counts().to_string(index=False))
        return 0

    if args.command == "ingest-jolpica":
        from .ingest import jolpica

        seasons = _parse_seasons(args.seasons)
        print(f"jolpica: {seasons[0]}-{seasons[-1]} ({len(seasons)} seasons)")
        jolpica.ingest(seasons, force=args.force)
        print(store.table_counts().to_string(index=False))
        return 0

    if args.command == "ingest-fastf1":
        from .ingest import fastf1_pull

        seasons = _parse_seasons(args.seasons)
        print(f"FastF1: {seasons[0]}-{seasons[-1]}. Slow; cache goes to {config.FASTF1_CACHE}")
        fastf1_pull.ingest(seasons, force=args.force)
        print(store.table_counts().to_string(index=False))
        return 0

    if args.command == "forecast":
        from .ingest import weather

        print(f"Stored {weather.ingest_upcoming(days_ahead=args.days_ahead)} forecast rows")
        return 0

    if args.command == "validate":
        from . import validate

        report = validate.run()
        print(report.render(show_samples=not args.quiet))
        return 0 if report.ok else 1

    if args.command == "build-features":
        from . import features

        df = features.build()
        features.save(df)
        coverage = df[features.RACE_FEATURES].notna().mean().mul(100).round(1)
        print(f"\n{len(df)} rows, {df.race_seq.nunique()} races")
        print("\nfeature coverage (%):")
        print(coverage.sort_values().to_string())
        return 0

    if args.command == "repair":
        changed = store.repair_status_flags()
        if changed.empty:
            print("status flags already consistent - nothing to repair")
        else:
            print(changed.to_string(index=False))
            print("\nRebuild features next: make features")
        return 0

    if args.command == "backtest":
        from . import backtest, features

        df = features.load()
        temperature = config.DEFAULT_TEMPERATURE
        blend_weight = config.DEFAULT_BLEND_WEIGHT
        recency = config.DEFAULT_CURRENT_SEASON_WEIGHT
        tune_season = args.tune_season or (args.start_season - 1)

        if args.tune:
            # Deliberately an EARLIER season than the one being reported.
            # Fitting and reporting on the same races makes the settings part
            # of the answer, and the table stops being out-of-sample.
            if tune_season >= args.start_season:
                parser_error = (
                    f"--tune-season ({tune_season}) must be earlier than "
                    f"--start-season ({args.start_season}), or the reported "
                    f"numbers are the ones the settings were chosen on"
                )
                raise SystemExit(parser_error)
            # Bounded at both ends so the settings are never fitted on the races the
            # table then reports.
            tune_end = args.start_season - 1
            print(f"Fitting temperature, blend and recency on {tune_season}-{tune_end}...")
            temperature = backtest.tune_temperature(df, tune_season, end_season=tune_end)
            blend_weight = backtest.tune_blend(df, tune_season, temperature, end_season=tune_end)
            recency = backtest.tune_recency(df, tune_season, end_season=tune_end)
            print(
                f"  temperature={temperature:.3f}  blend={blend_weight:.2f}  recency={recency:.1f}\n"
                f"Reporting on {args.start_season}+, which the tuner never saw."
            )

        res = backtest.walk_forward(
            df,
            args.start_season,
            retrain_every=args.retrain_every,
            n_sims=args.sims,
            blend_weight=blend_weight,
            temperature=temperature,
            current_season_weight=recency,
        )
        summary = res.summary()
        print("\n" + summary.to_string())
        print("\ncalibration:")
        print(res.calibration().to_string(index=False))

        # Split by season as well as pooled: a good average can hide a bad
        # year, and the live season is the one a reader cares about most.
        names = {"model": "This model", "grid": "Grid order"}
        sub = res.races[res.races.method.isin(names)]
        by_season = (
            sub.groupby(["season", "method"])[
                ["top5_overlap", "podium_overlap", "top1_hit", "ndcg5", "logloss", "brier"]
            ]
            .mean()
            .reset_index()
        )
        by_season["races"] = sub.groupby(["season", "method"]).size().values
        by_season["method"] = by_season["method"].map(names)
        by_season = by_season.rename(
            columns={
                "method": "approach",
                "top5_overlap": "top 5",
                "podium_overlap": "podium",
                "top1_hit": "winner",
                "ndcg5": "ndcg@5",
                "logloss": "log loss",
            }
        ).round(3)

        out = config.REPORTS / "backtest.json"
        out.write_text(
            json.dumps(
                {
                    "summary": summary.reset_index().to_dict("records"),
                    "by_season": by_season.to_dict("records"),
                    "calibration": res.calibration().astype(str).to_dict("records"),
                    "params": {
                        "temperature": temperature,
                        "blend_weight": blend_weight,
                        "recency_weight": recency,
                        "start_season": args.start_season,
                        "tuned_on_season": tune_season if args.tune else None,
                    },
                },
                indent=2,
            )
        )
        print(f"\nwrote {out}")
        return 0

    if args.command == "predict":
        from . import predict

        season = args.season if not args.next else None
        rnd = args.round if not args.next else None
        p = predict.run(season=season, rnd=rnd, n_sims=args.sims)
        path = p.save(force=getattr(args, "force_log", False))
        if p.season_outlook:
            config.SEASON_NOW.write_text(json.dumps(p.season_outlook, default=str))

        stage = "grid known" if p.grid_known else "before qualifying"
        if path is None:
            ahead = p.days_out() or 0.0
            print(f"\nnot logged: {ahead:.1f} days before the race, outside the logging window")
        print(f"\n{p.race_name}  ({stage})")
        print(f"trained on {p.meta['n_training_races']} races\n")
        print(f"QUALIFYING - top {config.TOP_N}")
        for i, d in enumerate(p.quali_board, 1):
            print(
                f"  {i:2}. {d['name']:<24} pole {d['p_win'] * 100:5.1f}%   top10 {d['p_top10'] * 100:5.1f}%"
            )
        print(f"\nRACE - top {config.TOP_N}")
        for i, d in enumerate(p.race_board, 1):
            print(
                f"  {i:2}. {d['name']:<24} win {d['p_win'] * 100:5.1f}%"
                f"   podium {d['p_podium'] * 100:5.1f}%   points {d['p_top10'] * 100:5.1f}%"
            )
        print(f"\nlogged to {path}" if path else "\nnot logged (see above); the page still renders")
        return 0

    if args.command == "bias":
        from . import diagnostics

        result = diagnostics.measure(args.start_season, args.retrain_every)
        out = config.REPORTS / "bias.json"
        out.write_text(json.dumps(result, indent=2))
        print(diagnostics.report(result))
        print(f"\nwrote {out}")
        return 0

    if args.command == "verify":
        from . import verify

        result = verify.run(args.start_season, args.sims, args.retrain_every)
        print(result.render())
        return 0 if result.ok else 1

    if args.command == "title-backtest":
        import json as _json

        from . import features, title_backtest

        frame = title_backtest.run(features.load(), n_sims=args.sims)
        print("\n" + title_backtest.report(frame))
        out = config.REPORTS / "title_backtest.json"
        out.write_text(
            _json.dumps(
                {
                    "checkpoints": frame.to_dict("records"),
                    "calibration": title_backtest.calibration(frame).astype(str).to_dict("records"),
                    "brier": float(((frame["p_favourite"] - frame["favourite_was_right"]) ** 2).mean())
                    if not frame.empty
                    else None,
                },
                indent=1,
            )
        )
        print(f"\nwrote {out}")
        return 0

    if args.command == "calibrate-spread":
        import json as _json

        from . import features, spread_calibration

        result = spread_calibration.run(features.load(), n_sims=args.sims)
        print("\n" + spread_calibration.report(result))
        if result:
            out = config.REPORTS / "spread_calibration.json"
            out.write_text(
                _json.dumps(
                    {
                        "sweep": result["sweep"].to_dict("records"),
                        "best": result["best"],
                        "fit_seasons": list(spread_calibration.FIT_SEASONS),
                        "grade_seasons": list(spread_calibration.GRADE_SEASONS),
                        "held_out": {
                            label: {
                                "coverage": float(frame["inside"].mean()),
                                "width": float(frame["width"].mean()),
                                "n": len(frame),
                            }
                            for label, frame in result["graded"].items()
                        },
                        "in_use": config.SEASON_PACE_UNCERTAINTY,
                    },
                    indent=1,
                )
            )
            print(f"\nwrote {out}")
            if abs(result["best"] - config.SEASON_PACE_UNCERTAINTY) > 1e-9:
                print(
                    f"note: config.SEASON_PACE_UNCERTAINTY is {config.SEASON_PACE_UNCERTAINTY}, "
                    f"this run picked {result['best']:.1f}"
                )
        return 0

    if args.command == "dashboard":
        import pandas as pd

        from . import report

        prediction = None
        preds = report.load_predictions()
        if preds:
            prediction = max(preds, key=lambda p: p["generated_at_utc"])
        # The race and qualifying boards are the logged forecast and never
        # change. The championship panel shows the latest projection instead.
        if prediction and config.SEASON_NOW.exists():
            prediction = {**prediction, "season_outlook": json.loads(config.SEASON_NOW.read_text())}

        summary = calibration = None
        params = None
        bt = config.REPORTS / "backtest.json"
        if bt.exists():
            data = json.loads(bt.read_text())
            summary = pd.DataFrame(data["summary"]).set_index("method")
            calibration = pd.DataFrame(data["calibration"])
            params = data.get("params")

        path = report.write(prediction, summary, calibration, params)
        print(f"wrote {path}")

        from . import method_page

        print(f"wrote {method_page.write()}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
