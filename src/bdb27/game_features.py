"""Regular-season game performance per player (and per player-season).

Two sources:
  * player_play.csv  - position-specific production / efficiency rates
  * game_tracking_*  - in-game movement ceilings (how fast a player actually plays)
"""
from __future__ import annotations

import polars as pl

MAX_PLAUSIBLE_SPEED = 12.5  # yd/s; above this is almost always a tracking glitch

c = pl.col


def _rate(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    return pl.when(den > 0).then(num / den)


def play_outcomes(pp: pl.DataFrame, games: pl.DataFrame, by_season: bool = False) -> pl.DataFrame:
    df = (
        pp.join(games.select("game_id", "season", "season_type"), on="game_id")
        .filter((c("season_type") == "REG") & (c("play_nullified_by_penalty") == "N"))
        .with_columns(
            route=c("route_ran").is_not_null(),
            pass_block=c("pass_rushers_encountered").is_not_null(),
            pass_rush=c("unblocked_pressure").is_not_null(),
            pressure_made=c("time_to_pressure").is_not_null(),
            coverage=c("coverage_assignment").is_not_null() & (c("coverage_assignment") != "NO_ASSIGNMENT"),
            yac_oe=c("yards_after_catch") - c("expected_yards_after_catch"),
        )
    )
    keys = ["nfl_id", "season"] if by_season else ["nfl_id"]
    out = df.group_by(keys).agg(
        snaps=pl.len(),
        games=c("game_id").n_unique(),
        # Receivers
        routes=c("route").sum(),
        targets=c("target").sum(),
        rec_yards=c("rec_yards").sum(),
        mean_separation=c("separation_at_pass_forward").mean(),
        mean_yac_oe=c("yac_oe").mean(),
        # Pass blockers
        pass_block_snaps=c("pass_block").sum(),
        pressures_allowed=c("pressure_allowed").sum(),
        sacks_allowed=c("sack_allowed").sum(),
        mean_peak_pressure_prob_allowed=c("peak_pressure_probability_allowed").mean(),
        # Pass rushers
        pass_rush_snaps=c("pass_rush").sum(),
        pressures=c("pressure_made").sum(),
        sacks=c("sack").sum(),
        mean_get_off=c("player_get_off").mean(),
        mean_time_to_pressure=c("time_to_pressure").mean(),
        # Coverage / run defense
        coverage_snaps=c("coverage").sum(),
        mean_cushion=c("cushion").mean(),
        tackles=c("tackle").sum(),
        tfl=c("tackle_for_loss").sum(),
    )
    return out.with_columns(
        target_rate=_rate(c("targets"), c("routes")),
        yards_per_route=_rate(c("rec_yards"), c("routes")),
        pressure_allowed_rate=_rate(c("pressures_allowed"), c("pass_block_snaps")),
        pressure_rate=_rate(c("pressures"), c("pass_rush_snaps")),
        snaps_per_game=_rate(c("snaps"), c("games")),
    ).sort(keys)


def movement_ceilings(gt: pl.LazyFrame, games: pl.DataFrame, by_season: bool = False) -> pl.DataFrame:
    keys = ["nfl_id", "season"] if by_season else ["nfl_id"]
    return (
        gt.join(games.lazy().select("game_id", "season", "season_type"), on="game_id")
        .filter((c("season_type") == "REG") & (c("s") <= MAX_PLAUSIBLE_SPEED))
        .group_by(keys)
        .agg(
            game_frames=pl.len(),
            game_top_speed=c("s").quantile(0.999),
            game_p95_speed=c("s").quantile(0.95),
            game_p99_accel=c("a").quantile(0.99),
        )
        .sort(keys)
        .collect()
    )
