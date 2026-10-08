"""Movement features from Combine Zebra tracking (10 Hz).

Each drill attempt becomes one row of kinematic features. Attempts are then
collapsed to one value per player and drill (the best attempt for "max"
features, the mean for the rest), giving a wide player-level table.
"""
from __future__ import annotations

import numpy as np
import polars as pl

MOVING = 1.0  # yd/s; below this, heading is too noisy to trust

# The same drill is spelled differently across Combine years.
DRILL_ALIASES = {
    "LINE": "LINE_DRILL",
    "PASS_PRO-MIRROR_DRILL": "PASS_PRO_MIRROR_DRILL",
    "OVER_THE_SHOULDER_ADJUST": "OVER_SHOULDER_ADJUST",
    "SPEED_OUT_LEFT": "SPEED_OUT_ROUTE_LEFT",
}


def _wrap(deg: np.ndarray) -> np.ndarray:
    return (deg + 180.0) % 360.0 - 180.0


def _time_to_distance(t: np.ndarray, cum: np.ndarray, s: np.ndarray, target: float, max_extrap: float = 2.0) -> float:
    """Interpolated time at which cumulative distance first reaches `target`.

    Clips sometimes end a yard or so short of the finish line; within
    `max_extrap` yards we extrapolate at the final speed.
    """
    i = int(np.searchsorted(cum, target))
    if i == 0:
        return np.nan
    if i >= len(cum):
        short = target - cum[-1]
        return float(t[-1] + short / s[-1]) if short <= max_extrap and s[-1] > 1 else np.nan
    frac = (target - cum[i - 1]) / max(cum[i] - cum[i - 1], 1e-9)
    return float(t[i - 1] + frac * (t[i] - t[i - 1]))


def attempt_features(g: pl.DataFrame) -> dict:
    t = g["t"].to_numpy()
    s = g["s"].to_numpy()
    a = g["a"].to_numpy()
    # The provided `dis` column lags and under-counts in Combine data; use the x/y path instead.
    step = np.hypot(np.diff(g["x"].to_numpy()), np.diff(g["y"].to_numpy()))
    heading = g["dir"].to_numpy()

    dt = np.diff(t)
    dt[dt <= 0] = np.nan
    accel_long = np.diff(s) / dt  # signed: + speeding up, - braking

    moving = (s[1:] > MOVING) & (s[:-1] > MOVING)
    dturn = np.abs(_wrap(np.diff(heading)))
    ang_vel = np.where(moving, dturn / dt, np.nan)

    peak_i = int(np.nanargmax(s))
    start = int(np.argmax(s > 0.5)) if (s > 0.5).any() else 0

    out = {
        "duration": float(t[-1] - t[start]),
        "distance": float(np.nansum(step[start:])),
        "peak_speed": float(np.nanmax(s)),
        "mean_speed": float(np.nanmean(s[start:])),
        "peak_accel": float(np.nanmax(a)),
        "peak_long_accel": float(np.nanmax(accel_long)) if len(accel_long) else np.nan,
        "peak_long_decel": float(-np.nanmin(accel_long)) if len(accel_long) else np.nan,
        "time_to_peak_speed": float(t[peak_i] - t[start]),
        "total_turn_deg": float(np.nansum(np.where(moving, dturn, 0.0))),
        "peak_ang_vel": float(np.nanmax(ang_vel)) if np.isfinite(ang_vel).any() else np.nan,
        # Speed held through the sharpest turn: a change-of-direction efficiency signal.
        "speed_at_peak_turn": float(s[1:][np.nanargmax(ang_vel)]) if np.isfinite(ang_vel).any() else np.nan,
    }

    if g["drill_type"][0] == "FORTY_YARD_DASH":
        # Clips begin as the athlete leaves the stance, so splits run from the first frame.
        cum = np.concatenate([[0.0], np.cumsum(step)])
        tt = t - t[0]
        for yd in (2, 5, 10, 20, 40):
            out[f"split_{yd}"] = _time_to_distance(tt, cum, s, yd)
        out["flying_10_speed"] = 10.0 / (out["split_40"] - _time_to_distance(tt, cum, s, 30))
    return out


def per_attempt(ct: pl.LazyFrame) -> pl.DataFrame:
    df = (
        ct.filter(pl.col("entity_type") == "PLAYER")
        .with_columns(
            pl.col("time").str.to_datetime(),
            pl.col("drill_name").replace(DRILL_ALIASES),
        )
        .sort("nfl_id", "drill_name", "attempt", "time")
        .with_columns(
            ((pl.col("time") - pl.col("time").first()).dt.total_milliseconds() / 1000.0)
            .over("nfl_id", "drill_name", "attempt", "event_id")
            .alias("t")
        )
        .collect()
    )
    keys = ["nfl_id", "drill_type", "drill_name", "attempt", "event_id"]
    rows = []
    for key, g in df.group_by(keys, maintain_order=True):
        if g.height < 5:
            continue
        rows.append(dict(zip(keys, key)) | attempt_features(g))
    return pl.DataFrame(rows).fill_nan(None)


# Higher is better for these, so the best attempt is the max; everything else is averaged.
_BEST_MAX = ["peak_speed", "peak_accel", "peak_long_accel", "peak_long_decel", "peak_ang_vel", "flying_10_speed"]
_BEST_MIN = ["split_2", "split_5", "split_10", "split_20", "split_40"]


def per_player_drill(att: pl.DataFrame) -> pl.DataFrame:
    feats = [c for c in att.columns if c not in {"nfl_id", "drill_type", "drill_name", "attempt", "event_id"}]
    aggs = [
        pl.col(c).max() if c in _BEST_MAX else pl.col(c).min() if c in _BEST_MIN else pl.col(c).mean()
        for c in feats
    ]
    return att.group_by("nfl_id", "drill_type", "drill_name").agg(pl.len().alias("n_attempts"), *aggs)


def wide(ppd: pl.DataFrame, min_players: int = 30) -> pl.DataFrame:
    """Pivot to one row per player with `<drill>__<feature>` columns, dropping rare drills."""
    common = ppd.group_by("drill_name").agg(pl.col("nfl_id").n_unique().alias("n")).filter(pl.col("n") >= min_players)
    ppd = ppd.join(common.select("drill_name"), on="drill_name")
    feats = [c for c in ppd.columns if c not in {"nfl_id", "drill_type", "drill_name"}]
    long = ppd.unpivot(index=["nfl_id", "drill_name"], on=feats).drop_nulls("value")
    long = long.with_columns((pl.col("drill_name") + "__" + pl.col("variable")).alias("col"))
    return long.pivot(on="col", index="nfl_id", values="value")
