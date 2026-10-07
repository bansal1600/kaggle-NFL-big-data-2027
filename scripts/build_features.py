"""Build the player-level feature tables in data/processed/.

    python scripts/build_features.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bdb import combine_features as cf  # noqa: E402
from bdb import data, game_features as gf  # noqa: E402


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

    data.player_base().write_parquet(data.PROCESSED / "player_base.parquet")
    print(f"wrote {sorted(p.name for p in data.PROCESSED.glob('*.parquet'))}")


if __name__ == "__main__":
    main()
