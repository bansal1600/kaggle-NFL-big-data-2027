"""Paths and loaders for the Big Data Bowl 2027 competition files.

Raw CSVs live in data/raw/ (see scripts/download_data.sh). The first call to a
loader for a large table converts it to Parquet in data/interim/ so later loads
are fast.
"""
from __future__ import annotations

import os
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
RAW = Path(os.environ.get("BDB_RAW_DIR", ROOT / "data" / "raw"))
# BDB_WORK_DIR moves the Parquet cache and processed tables elsewhere (the Kaggle notebook uses /tmp
# so that no copy of the competition data ends up in its public output).
WORK = Path(os.environ.get("BDB_WORK_DIR", ROOT / "data"))
INTERIM = WORK / "interim"
PROCESSED = WORK / "processed"
REPORTS = ROOT / "reports"

NA = ["NA", ""]
SEASONS = (2023, 2024, 2025)

# Combine position -> broad group used throughout the analysis.
POSITION_GROUPS = ("WR", "TE", "DB", "OL", "DL")


def _read(name: str, **kw) -> pl.DataFrame:
    return pl.read_csv(RAW / name, null_values=NA, infer_schema_length=100_000, **kw)


def players() -> pl.DataFrame:
    return _read("players.csv")


def combine_results() -> pl.DataFrame:
    return _read("combine_results.csv")


def career() -> pl.DataFrame:
    return _read("player_career_successes.csv")


def games() -> pl.DataFrame:
    return _read("games.csv")


def _parquet_cached(name: str) -> Path:
    INTERIM.mkdir(parents=True, exist_ok=True)
    out = INTERIM / name.replace(".csv", ".parquet")
    if not out.exists():
        pl.scan_csv(RAW / name, null_values=NA, infer_schema_length=100_000).sink_parquet(out)
    return out


def player_play() -> pl.DataFrame:
    # Empty strings are meaningful in a few columns (e.g. pass_result on runs),
    # so only "NA" is treated as null here.
    out = INTERIM / "player_play.parquet"
    if not out.exists():
        INTERIM.mkdir(parents=True, exist_ok=True)
        pl.read_csv(RAW / "player_play.csv", null_values="NA", infer_schema_length=100_000).write_parquet(out)
    return pl.read_parquet(out)


def combine_tracking() -> pl.LazyFrame:
    return pl.scan_parquet(_parquet_cached("combine_tracking.csv"))


def game_tracking(season: int | None = None) -> pl.LazyFrame:
    seasons = SEASONS if season is None else (season,)
    return pl.concat(
        [pl.scan_parquet(_parquet_cached(f"game_tracking_{s}.csv")) for s in seasons]
    )


def player_base() -> pl.DataFrame:
    """One row per drafted player: bio, draft slot, combine results, career outcomes."""
    return (
        players()
        .join(combine_results().drop("draft_year"), on="nfl_id", how="left")
        .join(career(), on="nfl_id", how="left")
        .rename({"combine_position": "pos_group"})
    )
