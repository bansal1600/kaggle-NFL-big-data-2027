"""Translation map: does a trait measured in a Combine drill show up in the same player's NFL snaps?

For each position group, trait and Combine source (a tracked drill or an
official measurement) we report the Spearman correlation with the player's
in-game trait, and the partial Spearman correlation controlling for body
weight, NFL roster position (e.g. DT vs DE vs OLB) and draft year (Combine
tracking drifts between years), so that size, role and measurement drift
cannot masquerade as translation.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import rankdata

from . import traits

# Official tests where lower is better are negated so that + always means "more athletic".
OFFICIAL = {"forty": -1, "ten_yd_split": -1, "three_cone": -1, "short_shuttle": -1, "vertical": 1, "broad_jump": 1}
MIN_N = 25


def rank(a: np.ndarray) -> np.ndarray:
    """Average ranks (ties share a rank), so results never depend on row order."""
    return rankdata(a).astype(float)


def partial_spearman(x: np.ndarray, y: np.ndarray, Z: np.ndarray) -> float:
    rx, ry = rank(x), rank(y)
    X = np.column_stack([np.ones(len(x)), Z]) if Z.size else np.ones((len(x), 1))
    res = lambda v: v - X @ np.linalg.lstsq(X, v, rcond=None)[0]  # noqa: E731
    return float(np.corrcoef(res(rx), res(ry))[0, 1])


def _controls(d: pl.DataFrame) -> np.ndarray:
    """Weight rank + roster-position dummies + draft-year dummies (Combine tracking drifts by year)."""
    cols = [rank(d["combine_weight"].to_numpy())]
    for col in ("nfl_position", "draft_year"):
        if col in d.columns:
            dummies = d[col].cast(pl.String).to_dummies()
            cols += [dummies[c].to_numpy().astype(float) for c in dummies.columns[1:]]
    Z = np.column_stack(cols)
    return Z[:, Z.std(0) > 0]


def _sources(base: pl.DataFrame, combine_traits: pl.DataFrame, trait: str) -> dict[str, pl.DataFrame]:
    """Every Combine source of `trait` as a (nfl_id, x) frame."""
    out = {
        drill: g.select("nfl_id", pl.col(trait).alias("x"))
        for (drill,), g in combine_traits.drop_nulls(trait).group_by("drill_name")
    }
    for col, sign in OFFICIAL.items():
        out[col] = base.select("nfl_id", (pl.col(col) * sign).alias("x")).drop_nulls()
    return out


def translation_map(base: pl.DataFrame, combine_traits: pl.DataFrame, game_traits: pl.DataFrame,
                    groups: tuple[str, ...]) -> pl.DataFrame:
    rows = []
    players = base.select("nfl_id", "pos_group", "nfl_position", "combine_weight", "draft_year").join(game_traits, on="nfl_id")
    for grp in groups:
        g = players.filter(pl.col("pos_group") == grp)
        for t in traits.TRAITS:
            for src, s in _sources(base, combine_traits, t).items():
                d = g.join(s, on="nfl_id").drop_nulls(["x", f"game_{t}", "combine_weight"])
                if d.height < MIN_N:
                    continue
                x, y = d["x"].to_numpy(), d[f"game_{t}"].to_numpy()
                rows.append(dict(group=grp, trait=t, source=src, official=src in OFFICIAL, n=d.height,
                                 rho=float(np.corrcoef(rank(x), rank(y))[0, 1]),
                                 partial=partial_spearman(x, y, _controls(d))))
    return pl.DataFrame(rows)


def add_fwer(tmap: pl.DataFrame, base: pl.DataFrame, combine_traits: pl.DataFrame, game_traits: pl.DataFrame,
             n_perm: int = 1000, seed: int = 0) -> pl.DataFrame:
    """Family-wise (max-|rho|) permutation p-values within each (group, trait) family.

    The in-game trait is shuffled across a group's players, every source in the
    family is re-correlated, and each observed |partial rho| is compared with
    the permutation distribution of the family maximum. This accounts for having
    searched over ~15-20 drills/tests per family.
    """
    rng = np.random.default_rng(seed)
    players = base.select("nfl_id", "pos_group", "nfl_position", "combine_weight", "draft_year").join(game_traits, on="nfl_id")
    out = []
    for (grp, t), fam in tmap.group_by(["group", "trait"], maintain_order=True):
        g = players.filter(pl.col("pos_group") == grp).drop_nulls([f"game_{t}", "combine_weight"]).sort("nfl_id")
        ids = g["nfl_id"].to_numpy()
        y = g[f"game_{t}"].to_numpy()
        pos = {pid: i for i, pid in enumerate(ids)}
        srcs = _sources(base, combine_traits, t)
        prepared = []
        for src in fam["source"].to_list():
            d = g.join(srcs[src], on="nfl_id").drop_nulls("x")
            idx = np.array([pos[p] for p in d["nfl_id"].to_list()])
            Z = _controls(d)
            X = np.column_stack([np.ones(d.height), Z])
            M = np.eye(d.height) - X @ np.linalg.pinv(X)
            rx = M @ rank(d["x"].to_numpy())
            prepared.append((idx, M, rx / np.linalg.norm(rx)))
        null_max = np.zeros(n_perm)
        for b in range(n_perm):
            yp = y[rng.permutation(len(y))]
            best = 0.0
            for idx, M, rxn in prepared:
                ry = M @ rank(yp[idx])
                best = max(best, abs(float(rxn @ ry) / np.linalg.norm(ry)))
            null_max[b] = best
        out.append(fam.with_columns(
            p_fwer=pl.col("partial").abs().map_elements(lambda v: float((null_max >= v).mean()), return_dtype=pl.Float64)))
    return pl.concat(out)
