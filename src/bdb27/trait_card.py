"""Weekly trait card: a player's in-game movement traits, updated every week, with honest uncertainty.

For every player-season and week we take all regular-season snaps to date and
compute each trait (top speed, burst, brake, bend) as the 90th percentile of
per-snap maxima, the same statistic used in the writeup.

Raw season-to-date values are noisy early in the season, so they are turned
into a *projection* of the player's rest-of-season level using a calibration
learned from the data itself:

    for a snap budget n, take each player-season's first n snaps ("early") and
    the remaining snaps ("later"); within each role, standardise both
    (mu_n, sd_n for early; mu_L, sd_L for later); rho_n = corr(early z, later z).

The best linear predictor of the later z-score is rho_n * early z. The 80%
band is conformal: the empirical 80th percentile of |later z - rho_n * early z|
in the calibration data, so it does not assume normality. The band narrows as
snaps accumulate. Calibration constants are interpolated
in log(n) between budgets, and they can be fitted on some seasons and applied
to others, which lets us check the bands out of sample.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import norm

from . import traits

TRAITS = traits.TRAITS
BUDGETS = (10, 20, 35, 50, 75, 100, 150, 200, 300)
MIN_LATER = 100  # rest-of-season snaps required to calibrate a budget
MIN_SNAPS_CARD = 10

ROLE = {"WR": "WR", "TE": "TE", "FB": "TE", "RB": "WR",
        "CB": "CB", "DB": "CB", "FS": "S", "SS": "S",
        "T": "OT", "G": "IOL", "C": "IOL",
        "DE": "EDGE", "OLB": "EDGE", "DT": "IDL", "NT": "IDL", "ILB": "EDGE", "MLB": "EDGE"}
ROLE_ORDER = ("WR", "TE", "CB", "S", "OT", "IOL", "EDGE", "IDL")


def season_snaps(play_traits: pl.DataFrame, games: pl.DataFrame, base: pl.DataFrame) -> pl.DataFrame:
    """Regular-season snaps with role, season, week and a chronological snap index within player-season."""
    reg = games.filter(pl.col("season_type") == "REG").select("game_id", "season", "week")
    return (
        play_traits.join(reg, on="game_id")
        .join(base.select("nfl_id", "display_name", "draft_year", "nfl_position", "pos_group"), on="nfl_id")
        .with_columns(role=pl.col("nfl_position").replace_strict(ROLE, default=None),
                      nfl_year=pl.col("season") - pl.col("draft_year") + 1)
        .drop_nulls("role")
        .sort("nfl_id", "season", "game_id", "play_id")  # game_id is YYYYMMDDnn, so this is chronological
        .with_columns(snap_idx=pl.int_range(1, pl.len() + 1).over("nfl_id", "season"))
    )


def _q90(col: str) -> pl.Expr:
    return pl.col(col).quantile(0.9, interpolation="linear")


def calibrate(snaps: pl.DataFrame, seasons: list[int] | None = None) -> pl.DataFrame:
    """Calibration table: one row per (budget, role, trait) with mu_n, sd_n, mu_L, sd_L, and pooled rho_n per (budget, trait)."""
    s = snaps if seasons is None else snaps.filter(pl.col("season").is_in(seasons))
    totals = s.group_by("nfl_id", "season").agg(total=pl.len())
    rows = []
    for n in BUDGETS:
        ok = totals.filter(pl.col("total") >= n + MIN_LATER).select("nfl_id", "season")
        d = s.join(ok, on=["nfl_id", "season"])
        early = d.filter(pl.col("snap_idx") <= n).group_by("nfl_id", "season", "role").agg([_q90(t).alias(t) for t in TRAITS])
        later = d.filter(pl.col("snap_idx") > n).group_by("nfl_id", "season", "role").agg([_q90(t).alias(t) for t in TRAITS])
        pair = early.join(later, on=["nfl_id", "season", "role"], suffix="_L")
        for t in TRAITS:
            stats = pair.group_by("role").agg(
                mu_n=pl.col(t).mean(), sd_n=pl.col(t).std(), mu_L=pl.col(f"{t}_L").mean(), sd_L=pl.col(f"{t}_L").std(),
                players=pl.len())
            z = (pair.join(stats, on="role")
                 .filter(pl.col("players") >= 5)
                 .with_columns(ze=(pl.col(t) - pl.col("mu_n")) / pl.col("sd_n"), zl=(pl.col(f"{t}_L") - pl.col("mu_L")) / pl.col("sd_L")))
            if z.height <= 10:
                continue
            ze, zl = z["ze"].to_numpy(), z["zl"].to_numpy()
            rho = float(np.corrcoef(ze, zl)[0, 1])
            # Conformal band: empirical 80th percentile of |later z - rho * early z|, not a normal-theory width.
            half80 = float(np.quantile(np.abs(zl - rho * ze), 0.8))
            for r in stats.iter_rows(named=True):
                rows.append({"budget": n, "trait": t, "rho": rho, "half80": half80, "pairs": z.height} | r)
    return pl.DataFrame(rows)


def _interp(cal: pl.DataFrame, role: str, trait: str, n: np.ndarray) -> dict[str, np.ndarray]:
    """Interpolate calibration constants in log(n); clamp outside the calibrated budget range."""
    c = cal.filter((pl.col("role") == role) & (pl.col("trait") == trait)).sort("budget")
    if c.is_empty():
        return {}
    x = np.log(c["budget"].to_numpy())
    ln = np.log(np.clip(n, c["budget"].min(), c["budget"].max()))
    return {k: np.interp(ln, x, c[k].to_numpy()) for k in ("mu_n", "sd_n", "mu_L", "sd_L", "rho", "half80")}


def weekly_cards(snaps: pl.DataFrame, cal: pl.DataFrame, widths: pl.DataFrame | None = None) -> pl.DataFrame:
    """One row per (player, season, week, trait): snaps to date, raw value, projected percentile and 80% band.

    `widths` (from conformal_widths) gives season-blocked conformal band widths;
    without it the in-sample conformal width from `calibrate` is used.
    """
    weeks = snaps.select("season", "week").unique()
    ps = snaps.select("nfl_id", "season", "display_name", "role", "nfl_position", "nfl_year").unique()
    cum = []
    for (season,), w in weeks.group_by("season"):
        for wk in sorted(w["week"].to_list()):
            to_date = snaps.filter((pl.col("season") == season) & (pl.col("week") <= wk))
            cum.append(to_date.group_by("nfl_id").agg(
                [_q90(t).alias(t) for t in TRAITS]
                + [pl.len().alias("snaps"), (pl.col("week") == wk).any().alias("played_this_week")])
                .with_columns(season=pl.lit(season), week=pl.lit(wk)))
    cum = pl.concat(cum).filter(pl.col("snaps") >= MIN_SNAPS_CARD).join(ps, on=["nfl_id", "season"])
    out = []
    for (role,), g in cum.group_by("role"):
        n = g["snaps"].to_numpy().astype(float)
        for t in TRAITS:
            k = _interp(cal, role, t, n)
            if not k:
                continue
            ze = (g[t].to_numpy() - k["mu_n"]) / k["sd_n"]
            zp = k["rho"] * ze
            half = k["half80"]
            out.append(g.select("nfl_id", "season", "week", "display_name", "role", "nfl_position", "nfl_year", "snaps",
                                "played_this_week").with_columns(
                trait=pl.lit(t), raw=g[t], rho=k["rho"], z_early=ze, z_proj=zp, half=half,
                proj_value=k["mu_L"] + zp * k["sd_L"]))
    cards = pl.concat(out)
    if widths is not None:
        cards = (cards.with_columns(_snap_bin(pl.col("snaps")))
                 .join(widths.select("role", "trait", "snap_bin", pl.col("half").alias("half_c")),
                       on=["role", "trait", "snap_bin"], how="left")
                 .with_columns(half=pl.coalesce("half_c", "half")).drop("half_c", "snap_bin"))
    z, h = cards["z_proj"].to_numpy(), cards["half"].to_numpy()
    return (cards.with_columns(pct=norm.cdf(z) * 100, pct_lo=norm.cdf(z - h) * 100, pct_hi=norm.cdf(z + h) * 100)
            .sort("season", "week", "role", "nfl_id", "trait"))


def coverage(cards: pl.DataFrame, snaps: pl.DataFrame, min_rest: int = MIN_LATER) -> pl.DataFrame:
    """Out-of-sample check: does the 80% band contain the player's rest-of-season percentile?

    Rest-of-season percentile uses the later-snap calibration (mu_L, sd_L) at
    the card's snap count, i.e. the same scale the projection is on.
    """
    last_week = cards.group_by("nfl_id", "season").agg(pl.col("week").max().alias("last"))
    c = cards.join(last_week, on=["nfl_id", "season"]).filter(pl.col("week") < pl.col("last"))
    rest = []
    for (season, wk), g in c.select("season", "week").unique().group_by(["season", "week"]):
        r = snaps.filter((pl.col("season") == season) & (pl.col("week") > wk)).group_by("nfl_id").agg(
            [_q90(t).alias(t) for t in TRAITS] + [pl.len().alias("rest_snaps")])
        rest.append(r.with_columns(season=pl.lit(season), week=pl.lit(wk)))
    rest = pl.concat(rest).filter(pl.col("rest_snaps") >= min_rest).unpivot(
        index=["nfl_id", "season", "week", "rest_snaps"], on=list(TRAITS), variable_name="trait", value_name="rest_value")
    return c.join(rest, on=["nfl_id", "season", "week", "trait"])


def score_coverage(joined: pl.DataFrame, cal: pl.DataFrame) -> pl.DataFrame:
    out = []
    for (role, t), g in joined.group_by(["role", "trait"]):
        k = _interp(cal, role, t, g["snaps"].to_numpy().astype(float))
        if not k:
            continue
        zl = (g["rest_value"].to_numpy() - k["mu_L"]) / k["sd_L"]
        pct_rest = norm.cdf(zl) * 100
        out.append(g.with_columns(z_rest=zl, pct_rest=pct_rest,
                                  inside=(pct_rest >= g["pct_lo"].to_numpy()) & (pct_rest <= g["pct_hi"].to_numpy())))
    return pl.concat(out)


SNAP_BINS = (25, 50, 100, 200)  # band widths are learned per snap-count bin
MIN_RESID = 150  # residuals needed for a role-specific width; fewer -> pooled across roles


def _snap_bin(n: pl.Expr) -> pl.Expr:
    return pl.lit(len(SNAP_BINS)).sub(sum((n <= b).cast(pl.Int32) for b in SNAP_BINS)).alias("snap_bin")


def blocked_residuals(snaps: pl.DataFrame, seasons: list[int]) -> pl.DataFrame:
    """Out-of-season projection errors: for each season s, calibrate on the other seasons and predict s.

    Returns |rest-of-season z - projected z| for every card row with enough
    rest-of-season snaps. Season-blocked folds make the errors honest about
    season-to-season differences (new rookies, development, drift residue).
    """
    out = []
    for s in seasons:
        train = [x for x in seasons if x != s]
        cal = calibrate(snaps, train)
        cards = weekly_cards(snaps.filter(pl.col("season") == s), cal, widths=None)
        scored = score_coverage(coverage(cards, snaps), cal)
        out.append(scored.select("role", "trait", "snaps", resid=(
            pl.col("z_rest") - pl.col("z_proj")).abs()))
    return pl.concat(out)


def conformal_widths(resid: pl.DataFrame, level: float = 0.8) -> pl.DataFrame:
    """Band half-width (z units) per (role, trait, snap bin): empirical `level` quantile of out-of-season errors."""
    r = resid.with_columns(_snap_bin(pl.col("snaps")))
    pooled = r.group_by("trait", "snap_bin").agg(half_pooled=pl.col("resid").quantile(level))
    by_role = r.group_by("role", "trait", "snap_bin").agg(half_role=pl.col("resid").quantile(level), k=pl.len())
    return (by_role.join(pooled, on=["trait", "snap_bin"])
            .with_columns(half=pl.when(pl.col("k") >= MIN_RESID).then("half_role").otherwise("half_pooled"))
            .select("role", "trait", "snap_bin", "half", "k"))
