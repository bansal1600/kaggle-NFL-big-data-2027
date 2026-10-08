"""Route-break mechanics, measured the same way in Combine route drills and NFL routes.

A route is reduced to its main break:

* cut routes (out, in, slant, corner, post, dig...): the break is the frame of
  peak turning rate (lateral accel / speed) once the receiver is moving;
* stop routes (hitch / curl / comeback): the break is the frame of minimum
  speed after the stem.

Around the break we measure

    entry_speed  max speed in the 1.0 s before the break       (yd/s)
    apex_speed   speed at the break                            (yd/s)
    exit_speed   max speed in the 0.7 s after the break        (yd/s)
    retention    apex_speed / entry_speed (cut routes)         (0-1)
    brake        max deceleration in the 1.0 s before the break (yd/s^2)
    turn_deg     heading change from 0.5 s before to 0.5 s after
    stem_time    seconds from the start of the route to the break
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy.signal import savgol_filter

DT = 0.1
WINDOW, POLY = 7, 2
MOVING = 1.5  # yd/s
MIN_STEM_YDS = 3.0
POST_THROW = 10  # frames

# NFL route -> family, and Combine drill -> (family, matched NFL routes)
STOP, CUT, GO = "stop", "cut", "go"
GAME_FAMILY = {"HITCH": STOP, "OUT": CUT, "IN": CUT, "SLANT": CUT, "CORNER": CUT, "POST": CUT, "GO": GO}
DRILL_MATCH = {
    "CURL_ROUTE_RIGHT": (STOP, ["HITCH"]),
    "COMEBACK_ROUTE_RIGHT": (STOP, ["HITCH"]),
    "SPEED_OUT_ROUTE_LEFT": (CUT, ["OUT"]),
    "DAGGER_ROUTE_LEFT": (CUT, ["IN"]),
    "SLANT_ROUTE_LEFT": (CUT, ["SLANT"]),
    "POST_CORNER_ROUTE_RIGHT": (CUT, ["CORNER"]),
    "POST_CORNER_ROUTE_LEFT": (CUT, ["CORNER"]),
    "GO_ROUTE_RIGHT": (GO, ["GO"]),
}
METRICS = ("entry_speed", "apex_speed", "exit_speed", "retention", "brake", "turn_deg", "stem_time")


def _kin(x: np.ndarray, y: np.ndarray):
    vx = savgol_filter(x, WINDOW, POLY, deriv=1, delta=DT)
    vy = savgol_filter(y, WINDOW, POLY, deriv=1, delta=DT)
    ax = savgol_filter(x, WINDOW, POLY, deriv=2, delta=DT)
    ay = savgol_filter(y, WINDOW, POLY, deriv=2, delta=DT)
    v = np.hypot(vx, vy)
    with np.errstate(invalid="ignore", divide="ignore"):
        a_tan = (ax * vx + ay * vy) / v
        omega = (ax * vy - ay * vx) / v**2  # signed turning rate, rad/s
    heading = np.degrees(np.unwrap(np.arctan2(vy, vx)))
    return v, a_tan, np.abs(omega), heading


def break_metrics(x: np.ndarray, y: np.ndarray, family: str) -> dict | None:
    if len(x) < WINDOW + 4:
        return None
    v, a_tan, omega, heading = _kin(x, y)
    travelled = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    ok = (travelled >= MIN_STEM_YDS) & (np.arange(len(x)) >= WINDOW // 2) & (np.arange(len(x)) < len(x) - 3)
    if family == GO:
        return {"entry_speed": float(v.max()), "stem_time": float(len(x) * DT)}
    if family == STOP:
        cand = np.where(ok & (np.arange(len(x)) > np.argmax(v * ok)), v, np.inf)
        if not np.isfinite(cand).any():
            return None
        k = int(np.argmin(cand))
    else:
        cand = np.where(ok & (v >= MOVING), omega, -np.inf)
        if not np.isfinite(cand).any():
            return None
        k = int(np.argmax(cand))
    pre, post = slice(max(0, k - 10), k + 1), slice(k, min(len(x), k + 8))
    entry = float(v[pre].max())
    out = {
        "entry_speed": entry,
        "apex_speed": float(v[k]),
        "exit_speed": float(v[post].max()),
        "retention": float(v[k] / entry) if entry > 0 else np.nan,
        "brake": float(np.nanmax(-a_tan[pre])),
        "turn_deg": float(abs(heading[min(len(x) - 1, k + 5)] - heading[max(0, k - 5)])),
        "stem_time": float(k * DT),
    }
    return out


def combine_breaks(ct: pl.LazyFrame) -> pl.DataFrame:
    """Best-attempt break metrics per player for every matched Combine route drill."""
    df = (
        ct.filter((pl.col("entity_type") == "PLAYER") & pl.col("drill_name").is_in(list(DRILL_MATCH)))
        .with_columns(pl.col("time").str.to_datetime())
        .sort("nfl_id", "drill_name", "attempt", "event_id", "time")
        .collect()
    )
    rows = []
    for (pid, drill, att, _), g in df.group_by(["nfl_id", "drill_name", "attempt", "event_id"], maintain_order=True):
        fam = DRILL_MATCH[drill][0]
        m = break_metrics(g["x"].to_numpy(), g["y"].to_numpy(), fam)
        if m:
            rows.append({"nfl_id": pid, "drill_name": drill, "family": fam, "attempt": att} | m)
    att = pl.DataFrame(rows)
    # Best attempt = highest entry speed carried through the break (exit speed for stop routes).
    return att.sort("exit_speed", descending=True, nulls_last=True).group_by("nfl_id", "drill_name").first()


def game_breaks(gt: pl.LazyFrame, routes: pl.DataFrame) -> pl.DataFrame:
    """Break metrics for NFL routes, from the snap to 1 s after the throw.

    `routes` holds one row per (game_id, play_id, nfl_id) with route_ran.
    """
    keys = ["game_id", "play_id", "nfl_id"]
    frames = (
        gt.join(routes.lazy().select(*keys, "route_ran"), on=keys)
        .sort(*keys, "time")
        .collect()
    )
    rows = []
    for key, g in frames.group_by(keys, maintain_order=True):
        ev = g["event"].to_list()
        if "ball_snap" not in ev:
            continue
        i0 = ev.index("ball_snap")
        ends = [j for j in range(i0 + 1, len(ev)) if ev[j] in ("pass_forward", "pass_shovel", "qb_sack", "qb_strip_sack")]
        # Keep 1 s past the throw: the throw often arrives mid-break, and cutting
        # there would truncate the deceleration and the turn.
        i1 = min(len(ev), (ends[0] + 1 + POST_THROW) if ends else i0 + 40)
        fam = GAME_FAMILY.get(g["route_ran"][0])
        if fam is None:
            continue
        m = break_metrics(g["x"].to_numpy()[i0:i1], g["y"].to_numpy()[i0:i1], fam)
        if m:
            rows.append(dict(zip(keys, key)) | {"route_ran": g["route_ran"][0], "family": fam, "throw": bool(ends)} | m)
    return pl.DataFrame(rows)
