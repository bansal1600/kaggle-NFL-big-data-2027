"""Build the player-level feature tables in data/processed/.

    python scripts/build_features.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bdb27 import combine_features as cf  # noqa: E402
from bdb27 import data, game_features as gf, traits  # noqa: E402


def main() -> None:
    data.PROCESSED.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    att = cf.per_attempt(data.combine_tracking())
    att.write_parquet(data.PROCESSED / "combine_attempts.parquet")
    ppd = cf.per_player_drill(att)
    ppd.write_parquet(data.PROCESSED / "combine_player_drill.parquet")
    cf.wide(ppd).write_parquet(data.PROCESSED / "combine_tracking_wide.parquet")
    print(f"combine features: {att.height} attempts, {ppd.height} player-drills  ({time.time() - t0:.0f}s)")

    games, pp = data.games(), data.player_play()
    for by_season, suffix in ((False, ""), (True, "_by_season")):
        gf.play_outcomes(pp, games, by_season).write_parquet(data.PROCESSED / f"game_outcomes{suffix}.parquet")
        gf.movement_ceilings(data.game_tracking(), games, by_season).write_parquet(
            data.PROCESSED / f"game_movement{suffix}.parquet"
        )
    print(f"game features done ({time.time() - t0:.0f}s)")

    # Movement traits measured identically at the Combine and in games (see src/bdb27/traits.py).
    traits.combine_attempt_traits(data.combine_tracking()).write_parquet(data.PROCESSED / "combine_attempt_traits.parquet")
    traits.combine_traits(data.combine_tracking()).write_parquet(data.PROCESSED / "combine_traits.parquet")
    reg = games.filter(pl.col("season_type") == "REG")["game_id"].to_list()
    pl.concat([traits.game_play_traits(data.game_tracking(s).filter(pl.col("game_id").is_in(reg))) for s in data.SEASONS]
              ).write_parquet(data.PROCESSED / "game_play_traits.parquet")
    print(f"trait tables done ({time.time() - t0:.0f}s)")

    data.player_base().write_parquet(data.PROCESSED / "player_base.parquet")
    print(f"wrote {sorted(p.name for p in data.PROCESSED.glob('*.parquet'))}")


if __name__ == "__main__":
    main()
