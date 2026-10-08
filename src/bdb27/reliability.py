"""How repeatable is each trait, at the Combine and on Sundays?

Three reliabilities, all computed on residualised ranks (body weight and NFL
roster position partialled out, exactly like the translation map):

* Combine retest      - attempt 1 vs attempt 2 of the same drill
* Combine cross-drill - the same trait in two *different* drills
* NFL split-half      - odd vs even snaps, Spearman-Brown corrected

If a trait is stable in games but not across Combine reps or drills, the
problem is the Combine measurement, not the trait.
"""
from __future__ import annotations

import itertools

import polars as pl

from . import traits
from .translation import _controls, partial_spearman

MIN_PAIRS = 20


def spearman_brown(r: float, k: float) -> float:
    """Reliability of the mean of k parallel measurements with single-measurement reliability r."""
    return k * r / (1 + (k - 1) * r)


def reps_needed(r: float, target: float = 0.8) -> float:
    if r <= 0:
        return float("inf")
    return target * (1 - r) / (r * (1 - target))


def combine_retest(attempts: pl.DataFrame, base: pl.DataFrame, groups) -> pl.DataFrame:
    at = (attempts.sort("nfl_id", "drill_name", "attempt")
          .with_columns(k=pl.int_range(pl.len()).over("nfl_id", "drill_name"))
          .filter(pl.col("k") < 2)
          .join(base, on="nfl_id"))
    rows = []
    for grp in groups:
        for (drill,), g in at.filter(pl.col("pos_group") == grp).group_by("drill_name"):
            w = g.pivot(on="k", index=["nfl_id", "combine_weight", "nfl_position"], values=list(traits.TRAITS)).drop_nulls()
            if w.height < MIN_PAIRS or "top_speed_1" not in w.columns:
                continue
            for t in traits.TRAITS:
                rows.append(dict(group=grp, drill=drill, trait=t, n=w.height,
                                 r=partial_spearman(w[f"{t}_0"].to_numpy(), w[f"{t}_1"].to_numpy(), _controls(w))))
    return pl.DataFrame(rows)


def cross_drill(combine_traits: pl.DataFrame, base: pl.DataFrame, groups, min_players: int = 30) -> pl.DataFrame:
    c = combine_traits.join(base, on="nfl_id")
    rows = []
    for grp in groups:
        cg = c.filter(pl.col("pos_group") == grp)
        drills = cg.group_by("drill_name").len().filter(pl.col("len") >= min_players)["drill_name"].sort().to_list()
        for a, b in itertools.combinations(drills, 2):
            pair = (cg.filter(pl.col("drill_name") == a).select("nfl_id", "combine_weight", "nfl_position", *traits.TRAITS)
                    .join(cg.filter(pl.col("drill_name") == b).select("nfl_id", *traits.TRAITS), on="nfl_id", suffix="_b"))
            for t in traits.TRAITS:
                d = pair.drop_nulls([t, f"{t}_b"])
                if d.height >= 25:
                    rows.append(dict(group=grp, drill_a=a, drill_b=b, trait=t, n=d.height,
                                     r=partial_spearman(d[t].to_numpy(), d[f"{t}_b"].to_numpy(), _controls(d))))
    return pl.DataFrame(rows)


def game_split_half(play_traits: pl.DataFrame, base: pl.DataFrame, groups, q: float = 0.9,
                    min_plays_per_half: int = 25) -> pl.DataFrame:
    halves = (play_traits.sort("game_id", "play_id")
              .with_columns(half=pl.int_range(pl.len()).over("nfl_id") % 2)
              .group_by("nfl_id", "half")
              .agg([pl.col(t).quantile(q) for t in traits.TRAITS] + [pl.len().alias("n")]))
    w = (halves.pivot(on="half", index="nfl_id", values=list(traits.TRAITS) + ["n"])
         .filter(pl.col("n_0") >= min_plays_per_half).join(base, on="nfl_id").drop_nulls())
    rows = []
    for grp in groups:
        g = w.filter(pl.col("pos_group") == grp)
        if g.height < MIN_PAIRS:
            continue
        for t in traits.TRAITS:
            r = partial_spearman(g[f"{t}_0"].to_numpy(), g[f"{t}_1"].to_numpy(), _controls(g))
            rows.append(dict(group=grp, trait=t, n=g.height, r_half=r, r=spearman_brown(r, 2)))
    return pl.DataFrame(rows)


def summary(retest: pl.DataFrame, cross: pl.DataFrame, game: pl.DataFrame) -> pl.DataFrame:
    """One row per (group, trait): median retest, median cross-drill agreement, NFL split-half."""
    rt = retest.group_by("group", "trait").agg(combine_retest=pl.col("r").median(), retest_drills=pl.len())
    cd = cross.group_by("group", "trait").agg(combine_cross_drill=pl.col("r").median(), cross_pairs=pl.len())
    gm = game.select("group", "trait", pl.col("r").alias("nfl_split_half"))
    return gm.join(rt, on=["group", "trait"], how="left").join(cd, on=["group", "trait"], how="left")
