"""Bend: how hard a player can accelerate *sideways* while running a curve.

Acceleration is decomposed into a tangential part (speeding up / slowing down
along the path) and a lateral part (centripetal, a_lat = v^2 / r: changing
direction at speed). The same decomposition is applied to Combine drills and
to in-game pass rushes so the two are directly comparable.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy.signal import savgol_filter

DT = 0.1  # both Combine and game tracking are 10 Hz
WINDOW, POLY = 7, 2  # Savitzky-Golay smoothing (0.7 s window)
MIN_SPEED = 2.0  # yd/s; ignore near-stationary frames where heading is noise


def kinematics(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Smoothed speed, tangential and lateral acceleration (yd/s, yd/s^2) from positions."""
    w = min(WINDOW, len(x) - (1 - len(x) % 2))
    vx = savgol_filter(x, w, POLY, deriv=1, delta=DT)
    vy = savgol_filter(y, w, POLY, deriv=1, delta=DT)
    ax = savgol_filter(x, w, POLY, deriv=2, delta=DT)
    ay = savgol_filter(y, w, POLY, deriv=2, delta=DT)
    v = np.hypot(vx, vy)
    with np.errstate(invalid="ignore", divide="ignore"):
        a_tan = (ax * vx + ay * vy) / v
        a_lat = np.abs(ax * vy - ay * vx) / v
    return {"v": v, "a_tan": a_tan, "a_lat": a_lat}


def summarize(k: dict[str, np.ndarray], prefix: str = "") -> dict[str, float]:
    m = k["v"] >= MIN_SPEED
    if m.sum() < 5:
        return {}
    v, at, al = k["v"][m], k["a_tan"][m], k["a_lat"][m]
    return {
        f"{prefix}lat_p90": float(np.quantile(al, 0.9)),
        f"{prefix}lat_mean": float(al.mean()),
        f"{prefix}tan_p90": float(np.quantile(at, 0.9)),
        f"{prefix}speed_p90": float(np.quantile(v, 0.9)),
        f"{prefix}speed_mean": float(v.mean()),
        f"{prefix}total_p90": float(np.quantile(np.hypot(at, al), 0.9)),
    }


def hoop_drill(ct: pl.LazyFrame) -> pl.DataFrame:
    """Per-player Bend metrics from the DL Run-the-Hoop drill (figure-8 around two hoops, then a sprint out).

    Only the figure-8 is used; the straight-line sprint out is cut off.
    """
    df = (
        ct.filter((pl.col("entity_type") == "PLAYER") & (pl.col("drill_name") == "RUN_THE_HOOP_DRILL"))
        .with_columns(pl.col("time").str.to_datetime())
        .sort("nfl_id", "attempt", "time")
        .collect()
    )
    rows = []
    for (pid, attempt), g in df.group_by(["nfl_id", "attempt"], maintain_order=True):
        if g.height < 20:
            continue
        k = kinematics(g["x"].to_numpy(), g["y"].to_numpy())
        loop = _figure8_mask(g["x"].to_numpy(), g["y"].to_numpy())
        if loop.sum() < 15:
            continue
        loop_k = {key: val[loop] for key, val in k.items()}
        rows.append({"nfl_id": pid, "attempt": attempt, "loop_frames": int(loop.sum()),
                     "loop_time": float(loop.sum() * DT)} | summarize(loop_k, "hoop_"))
    att = pl.DataFrame(rows)
    # Best attempt = the one with the highest lateral acceleration (most players have only one).
    return att.sort("hoop_lat_p90", descending=True).group_by("nfl_id").first()


def _figure8_mask(x: np.ndarray, y: np.ndarray, pad: float = 0.3) -> np.ndarray:
    """Frames on the figure-8, from the start to the last frame inside the loop region.

    The first 60% of every path is figure-8, so its bounding box (plus `pad`
    yards) is the loop region. The sprint out is the final exit from that box.
    Orientation-agnostic, so mirrored drill set-ups work too.
    """
    head = slice(0, max(int(0.6 * len(x)), 5))
    inside = (
        (x >= x[head].min() - pad) & (x <= x[head].max() + pad)
        & (y >= y[head].min() - pad) & (y <= y[head].max() + pad)
    )
    mask = np.zeros(len(x), bool)
    if inside.any():
        mask[: np.flatnonzero(inside)[-1] + 1] = True
    return mask
