"""In-game pass-rush kinematics for defensive linemen.

For every regular-season pass-rush snap (player_play.unblocked_pressure is
recorded only for pass rushers), take the rusher's tracking from the snap until
the throw / sack / 4 s, whichever comes first, and summarize with the same
lateral/tangential decomposition used for the Combine Run-the-Hoop drill.
"""
from __future__ import annotations

import polars as pl

from . import bend

MAX_RUSH_FRAMES = 40  # 4.0 s
END_EVENTS = {"pass_forward", "qb_sack", "qb_strip_sack", "pass_shovel", "qb_spike", "run", "handoff",
              "fumble", "out_of_bounds", "tackle", "qb_slide"}


def rush_plays(pp: pl.DataFrame, games: pl.DataFrame, ids: list[int]) -> pl.DataFrame:
    """Regular-season, non-nullified pass-rush snaps by the given players, with play context."""
    return (
        pp.join(games.select("game_id", "season", "season_type"), on="game_id")
        .filter(
            (pl.col("season_type") == "REG")
            & (pl.col("play_nullified_by_penalty") == "N")
            & pl.col("unblocked_pressure").is_not_null()
            & pl.col("nfl_id").is_in(ids)
        )
        .with_columns(pressure=pl.col("time_to_pressure").is_not_null())
        .select("game_id", "play_id", "nfl_id", "season", "lined_up_position", "down", "yards_to_go",
                "quarter", "offense_formation", "pass_result", "unblocked_pressure", "blitzing",
                "player_get_off", "pressure", "time_to_pressure", "sack")
    )


def rush_kinematics(gt: pl.LazyFrame, plays: pl.DataFrame) -> pl.DataFrame:
    keys = ["game_id", "play_id", "nfl_id"]
    frames = (
        gt.join(plays.lazy().select(keys), on=keys)
        .sort(*keys, "time")
        .collect()
    )
    rows = []
    for key, g in frames.group_by(keys, maintain_order=True):
        ev = g["event"].to_list()
        if "ball_snap" not in ev:
            continue
        i0 = ev.index("ball_snap")
        i1 = min(len(ev), i0 + MAX_RUSH_FRAMES)
        for j in range(i0 + 1, i1):
            if ev[j] in END_EVENTS:
                i1 = j + 1
                break
        if i1 - i0 < 10:  # < 1 s of rush: no meaningful bend to measure
            continue
        x, y = g["x"].to_numpy()[i0:i1], g["y"].to_numpy()[i0:i1]
        k = bend.kinematics(x, y)
        summary = bend.summarize(k, "rush_")
        if summary:
            rows.append(dict(zip(keys, key)) | {"rush_frames": i1 - i0} | summary)
    return pl.DataFrame(rows)
