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


def season_offsets(play_traits: pl.DataFrame, games: pl.DataFrame, groups: pl.DataFrame, iters: int = 200) -> pl.DataFrame:
    """Season-level measurement drift in each trait and position group, from the SAME players across seasons.

    `groups` maps nfl_id -> group (we use the Combine position group). Within
    each group: two-way fixed effects (player + season) on player-season means
    of the per-snap maxima, weighted by snaps, solved by alternating weighted
    demeaning; offsets are centred so the snap-weighted average is zero.
    The drift is not uniform: in 2023 games, WR burst reads ~0.66 yd/s^2 higher
    than in 2024-25 for the same receivers, other groups ~0.2-0.3. That is a
    tracking change, not a change in the athletes.
    """
    ps = (play_traits.join(games.select("game_id", "season"), on="game_id")
          .join(groups.select("nfl_id", "group"), on="nfl_id")
          .group_by("nfl_id", "group", "season").agg([pl.col(t).mean() for t in TRAITS] + [pl.len().alias("w")]))
    rows = []
    for (grp,), pg in ps.group_by(["group"], maintain_order=True):
        seasons = sorted(pg["season"].unique().to_list())
        for t in TRAITS:
            d = pg.drop_nulls(t)
            b = {s: 0.0 for s in seasons}
            for _ in range(iters):
                a = (d.with_columns(r=pl.col(t) - pl.col("season").replace_strict(b, return_dtype=pl.Float64))
                     .group_by("nfl_id").agg(a=(pl.col("r") * pl.col("w")).sum() / pl.col("w").sum()))
                bb = (d.join(a, on="nfl_id").with_columns(r=pl.col(t) - pl.col("a"))
                      .group_by("season").agg(b=(pl.col("r") * pl.col("w")).sum() / pl.col("w").sum()))
                b = dict(zip(bb["season"].to_list(), bb["b"].to_list()))
            wsum = d.group_by("season").agg(pl.col("w").sum())
            wmap = dict(zip(wsum["season"].to_list(), wsum["w"].to_list()))
            centre = sum(b[s] * wmap[s] for s in seasons) / sum(wmap.values())
            rows += [{"group": grp, "season": s, "trait": t, "offset": b[s] - centre} for s in seasons]
    return pl.DataFrame(rows).sort("group", "trait", "season")


def season_adjust(play_traits: pl.DataFrame, games: pl.DataFrame, groups: pl.DataFrame,
                  offsets: pl.DataFrame | None = None) -> pl.DataFrame:
    """Subtract each (position group, season) measurement offset from the per-snap trait maxima.

    Players without a group keep their raw values.
    """
    offsets = season_offsets(play_traits, games, groups) if offsets is None else offsets
    wide = offsets.pivot(on="trait", index=["group", "season"], values="offset").rename({t: f"_off_{t}" for t in TRAITS})
    out = (play_traits.join(games.select("game_id", "season"), on="game_id", how="left")
           .join(groups.select("nfl_id", "group"), on="nfl_id", how="left")
           .join(wide, on=["group", "season"], how="left")
           .with_columns([pl.col(t) - pl.col(f"_off_{t}").fill_null(0.0) for t in TRAITS]))
    assert out.height == play_traits.height
    return out.drop([f"_off_{t}" for t in TRAITS] + ["season", "group"])
