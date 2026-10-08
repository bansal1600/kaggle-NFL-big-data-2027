"""Four movement traits, measured the same way at the Combine and in NFL games.

    top_speed  max speed                                   (yd/s)
    burst      max tangential acceleration (speeding up)   (yd/s^2)
    brake      max tangential deceleration (slowing down)  (yd/s^2)
    bend       max lateral (centripetal) acceleration at speed >= BEND_MIN_SPEED (yd/s^2)

Velocity and acceleration come from Savitzky-Golay derivatives of x/y, never
from the provided s/a/dis columns, so both sources share one pipeline.

Derivatives are computed over each player's concatenated frames (one
vectorised pass) and frames within EDGE frames of a segment boundary are
discarded, which is equivalent to filtering every segment separately.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy.signal import savgol_filter

DT = 0.1
WINDOW, POLY = 7, 2
EDGE = WINDOW // 2
BEND_MIN_SPEED = 2.0
TRAITS = ("top_speed", "burst", "brake", "bend")
GAME_WINDOW_S = 5.0  # seconds after the snap


def _derivatives(df: pl.DataFrame, seg_cols: list[str]) -> pl.DataFrame:
    """Add v, a_tan, a_lat to frames sorted by segment then time."""
    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    vx = savgol_filter(x, WINDOW, POLY, deriv=1, delta=DT)
    vy = savgol_filter(y, WINDOW, POLY, deriv=1, delta=DT)
    ax = savgol_filter(x, WINDOW, POLY, deriv=2, delta=DT)
    ay = savgol_filter(y, WINDOW, POLY, deriv=2, delta=DT)
    v = np.hypot(vx, vy)
    with np.errstate(invalid="ignore", divide="ignore"):
        a_tan = (ax * vx + ay * vy) / v
        a_lat = np.abs(ax * vy - ay * vx) / v
    return df.with_columns(v=v, a_tan=a_tan, a_lat=a_lat).with_columns(
        _i=pl.int_range(pl.len()).over(seg_cols), _n=pl.len().over(seg_cols)
    ).filter((pl.col("_i") >= EDGE) & (pl.col("_i") < pl.col("_n") - EDGE)).drop("_i", "_n")


def _segment_maxima(df: pl.DataFrame, seg_cols: list[str]) -> pl.DataFrame:
    return df.group_by(seg_cols).agg(
        top_speed=pl.col("v").max(),
        burst=pl.col("a_tan").max(),
        brake=(-pl.col("a_tan")).max(),
        bend=pl.col("a_lat").filter(pl.col("v") >= BEND_MIN_SPEED).max(),
        frames=pl.len(),
    )


def combine_attempt_traits(ct: pl.LazyFrame) -> pl.DataFrame:
    """Trait maxima for every Combine drill attempt (drill names canonicalised)."""
    from .combine_features import DRILL_ALIASES

    seg = ["nfl_id", "drill_name", "attempt", "event_id"]
    df = (
        ct.filter(pl.col("entity_type") == "PLAYER")
        .with_columns(pl.col("drill_name").replace(DRILL_ALIASES))
        .sort(*seg, "time")
        .select(*seg, "drill_type", "time", "x", "y")
        .collect()
    )
    df = df.filter(pl.len().over(seg) > WINDOW)
    return _segment_maxima(_derivatives(df, seg), seg)


def combine_traits(ct: pl.LazyFrame) -> pl.DataFrame:
    """Best value of each trait per player and drill."""
    return combine_attempt_traits(ct).group_by("nfl_id", "drill_name").agg(
        [pl.col(t).max() for t in TRAITS] + [pl.len().alias("attempts")]
    )


def game_play_traits(gt: pl.LazyFrame, ids: list[int] | None = None) -> pl.DataFrame:
    """Trait maxima for every player-play, using the first GAME_WINDOW_S seconds after the snap."""
    seg = ["game_id", "play_id", "nfl_id"]
    lf = gt if ids is None else gt.filter(pl.col("nfl_id").is_in(ids))
    df = (
        lf.with_columns(pl.col("time").str.to_datetime())
        .with_columns(snap=pl.col("time").filter(pl.col("event") == "ball_snap").first().over(seg))
        .filter(pl.col("snap").is_not_null())
        .filter((pl.col("time") >= pl.col("snap"))
                & (pl.col("time") <= pl.col("snap") + pl.duration(milliseconds=int(GAME_WINDOW_S * 1000))))
        .sort(*seg, "time")
        .select(*seg, "x", "y")
        .collect()
    )
    df = df.filter(pl.len().over(seg) > WINDOW)
    return _segment_maxima(_derivatives(df, seg), seg)


def game_player_traits(play_traits: pl.DataFrame, q: float = 0.9, min_plays: int = 50) -> pl.DataFrame:
    """A player's in-game trait = the q-quantile of his play-level maxima (his 'top 10% of snaps')."""
    return (
        play_traits.group_by("nfl_id")
        .agg([pl.col(t).quantile(q).alias(f"game_{t}") for t in TRAITS] + [pl.len().alias("game_plays")])
        .filter(pl.col("game_plays") >= min_plays)
    )
