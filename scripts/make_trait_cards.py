"""Weekly trait cards: build, validate out of sample, and render.

    python scripts/build_features.py      # once
    python scripts/make_trait_cards.py    # -> reports/trait_cards/

Outputs
    weekly_trait_cards.parquet / .csv   every player-season-week-trait: snaps to date, raw value,
                                        projected percentile within role and 80% band, flag
    validation.json                     season-held-out coverage and quartile hit rates
    trait-card-validation.png           how fast each trait firms up + hit rates
    card-<player>-<season>.png          example cards
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bdb27 import data, trait_card as tc, traits  # noqa: E402

OUT = data.REPORTS / "trait_cards"
SEASONS = [2023, 2024, 2025]
TRAIT_LABEL = {"top_speed": "Top speed", "burst": "Burst", "brake": "Brake", "bend": "Bend"}
UNITS = {"top_speed": "yd/s", "burst": "yd/s²", "brake": "yd/s²", "bend": "yd/s²"}
ROLE_LABEL = {"WR": "wide receivers", "TE": "tight ends", "CB": "cornerbacks", "S": "safeties", "OT": "offensive tackles",
              "IOL": "interior OL", "EDGE": "edge rushers", "IDL": "interior DL"}

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, BLUE_BAND, ORANGE, AQUA, YELLOW = "#2a78d6", "#cde2fb", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10.5,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
})


def header(fig, title: str, subtitle: str, top: float):
    """Title + subtitle in a reserved band above the axes (no overlap regardless of text length)."""
    fig.subplots_adjust(top=top)
    fig.text(0.01, 0.985, title, ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.985 - 0.075 * (4.4 / fig.get_figheight()), subtitle, ha="left", va="top", fontsize=9.5,
             color=INK2, linespacing=1.35)


def flag(pct: pl.Expr) -> pl.Expr:
    return pl.when(pct >= 75).then(pl.lit("top quarter")).when(pct <= 25).then(pl.lit("bottom quarter")).otherwise(pl.lit(""))


def load_snaps() -> pl.DataFrame:
    games, base = data.games(), data.player_base()
    pt = traits.season_adjust(pl.read_parquet(data.PROCESSED / "game_play_traits.parquet"), games,
                              base.select("nfl_id", pl.col("pos_group").alias("group")))
    return tc.season_snaps(pt, games, base)


def nested_validation(snaps: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Hold each season out of BOTH the calibration and the conformal widths, then score it."""
    scored = []
    for test in SEASONS:
        train = [s for s in SEASONS if s != test]
        widths = tc.conformal_widths(tc.blocked_residuals(snaps, train))
        cal = tc.calibrate(snaps, train)
        cards = tc.weekly_cards(snaps.filter(pl.col("season") == test), cal, widths)
        scored.append(tc.score_coverage(tc.coverage(cards, snaps), cal).with_columns(test_season=pl.lit(test)))
    c = pl.concat(scored)
    bins = [(10, 50, "10-50"), (50, 100, "51-100"), (100, 200, "101-200"), (200, 10**6, "200+")]
    hits = []
    for lo, hi, lab in bins:
        d = c.filter((pl.col("snaps") > lo) & (pl.col("snaps") <= hi) | ((pl.col("snaps") == lo) & (lo == 10)))
        top, bot = d.filter(pl.col("pct") >= 75), d.filter(pl.col("pct") <= 25)
        hits.append({"snaps": lab, "rows": d.height,
                     "share_flagged": (top.height + bot.height) / d.height,
                     "top_quarter_finish_above_median": float((top["pct_rest"] > 50).mean()) if top.height else None,
                     "bottom_quarter_finish_below_median": float((bot["pct_rest"] < 50).mean()) if bot.height else None,
                     "n_top": top.height, "n_bottom": bot.height,
                     "band80_coverage": float(d["inside"].mean())})
    summary = {
        "coverage_overall": float(c["inside"].mean()),
        "coverage_by_trait": {t: float(v) for t, v in c.group_by("trait").agg(pl.col("inside").mean()).iter_rows()},
        "coverage_by_role": {r: float(v) for r, v in c.group_by("role").agg(pl.col("inside").mean()).sort("role").iter_rows()},
        "coverage_by_test_season": {str(s): float(v) for s, v in c.group_by("test_season").agg(pl.col("inside").mean()).sort("test_season").iter_rows()},
        "hit_rates": hits,
        "rows": c.height,
    }
    return c, summary


def fig_validation(cal: pl.DataFrame, summary: dict):
    rho = cal.select("budget", "trait", "rho").unique().sort("budget")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    colors = {"top_speed": BLUE, "burst": ORANGE, "bend": AQUA, "brake": YELLOW}
    markers = {"top_speed": "o", "burst": "D", "bend": "s", "brake": "^"}
    ends = {}
    for t in traits.TRAITS:
        r = rho.filter(pl.col("trait") == t)
        ax.plot(r["budget"], r["rho"], color=colors[t], marker=markers[t], ms=7, mec=SURFACE, lw=2)
        ends[t] = (r["budget"][-1], r["rho"][-1])
    # Direct labels at the line ends, nudged apart so close endpoints never overlap.
    placed = []
    for t, (xe, ye) in sorted(ends.items(), key=lambda kv: -kv[1][1]):
        y = ye
        for py in placed:
            if abs(y - py) < 0.045:
                y = py - 0.045
        placed.append(y)
        ax.annotate(TRAIT_LABEL[t], (xe, ye), xytext=(xe * 1.12, y), textcoords="data", va="center", color=INK, fontsize=10,
                    arrowprops={"arrowstyle": "-", "color": GRID, "lw": 0.8} if abs(y - ye) > 0.01 else None)
    ax.set_xscale("log")
    ax.set_xticks([10, 20, 50, 100, 200, 300], ["10", "20", "50", "100", "200", "300"])
    ax.set_ylim(0, 1)
    ax.set_xlim(8, 520)
    ax.set_xlabel("snaps played so far this season")
    ax.set_ylabel("correlation with rest-of-season value")
    ax.set_title("How fast each trait firms up")

    ax = axes[1]
    h = summary["hit_rates"]
    x = np.arange(len(h))
    top = [r["top_quarter_finish_above_median"] or 0 for r in h]
    bot = [r["bottom_quarter_finish_below_median"] or 0 for r in h]
    ax.bar(x - 0.2, top, width=0.38, color=BLUE, label="flagged top quarter → finished above median")
    ax.bar(x + 0.2, bot, width=0.38, color=ORANGE, label="flagged bottom quarter → finished below median")
    for i, r in enumerate(h):
        ax.text(i - 0.2, top[i] + 0.015, f"{top[i]:.0%}", ha="center", fontsize=9, color=INK)
        ax.text(i + 0.2, bot[i] + 0.015, f"{bot[i]:.0%}", ha="center", fontsize=9, color=INK)
        ax.text(i, -0.12, f"{r['share_flagged']:.0%} flagged", ha="center", fontsize=8.5, color=INK2, transform=ax.get_xaxis_transform())
    ax.axhline(0.5, color=INK2, lw=1, ls="--")
    ax.text(len(h) - 0.5, 0.515, "coin flip", ha="right", fontsize=8.5, color=INK2)
    ax.set_xticks(x, [r["snaps"] for r in h])
    ax.set_xlabel("snaps played so far this season", labelpad=18)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("When the card flags a player, is it right?", pad=40)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1, fontsize=8.5)
    header(fig, "Weekly trait card: validation on held-out seasons",
           f"Each season is predicted by a model calibrated on the other two. 80% bands cover "
           f"{summary['coverage_overall']:.0%} of rest-of-season outcomes.", top=0.74)
    fig.savefig(OUT / "trait-card-validation.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_card(cards: pl.DataFrame, nfl_id: int, season: int) -> Path:
    c = cards.filter((pl.col("nfl_id") == nfl_id) & (pl.col("season") == season)).sort("week")
    name, role, year = c["display_name"][0], c["role"][0], c["nfl_year"][0]
    last = c.filter(pl.col("week") == c["week"].max())
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.9), sharey=True)
    for ax, t in zip(axes, traits.TRAITS):
        d = c.filter(pl.col("trait") == t)
        wk = d["week"].to_numpy()
        ax.axhspan(75, 100, color="#f0efec", zorder=0)
        ax.axhspan(0, 25, color="#f0efec", zorder=0)
        ax.fill_between(wk, d["pct_lo"], d["pct_hi"], color=BLUE_BAND, step=None, lw=0, zorder=1)
        ax.plot(wk, d["pct"], color=BLUE, lw=2, zorder=2)
        played = d["played_this_week"].to_numpy()
        ax.plot(wk[played], d["pct"].to_numpy()[played], "o", color=BLUE, ms=5, mec=SURFACE, zorder=3)
        ax.axhline(50, color=INK2, lw=0.8, ls="--", zorder=1)
        lv = last.filter(pl.col("trait") == t).row(0, named=True)
        fl = "  ▲ top quarter" if lv["pct"] >= 75 else "  ▼ bottom quarter" if lv["pct"] <= 25 else ""
        ax.set_title(f"{TRAIT_LABEL[t]}  {lv['pct']:.0f}th{fl}", fontsize=11)
        ax.text(0.02, 0.03, f"80% band {lv['pct_lo']:.0f}–{lv['pct_hi']:.0f}  ·  raw {lv['raw']:.2f} {UNITS[t]}",
                transform=ax.transAxes, fontsize=8.5, color=INK2)
        ax.set_xlim(0.5, 18.5)
        ax.set_ylim(0, 100)
        ax.set_xticks([1, 6, 12, 18])
        ax.set_xlabel("week")
    axes[0].set_ylabel(f"projected percentile among {ROLE_LABEL[role]}")
    snaps_now = int(last["snaps"][0])
    header(fig, f"{name}  ·  {role}  ·  {season} (NFL year {year})  ·  {snaps_now} tracked snaps through week {int(last['week'][0])}",
           "Line: projected rest-of-season percentile among " + ROLE_LABEL[role] + ", shrunk toward average by how much "
           "the evidence is worth.\nBand: 80% range (calibrated on held-out seasons). Dots: weeks played. Shaded zones: top and bottom quarter.",
           top=0.72)
    slug = name.lower().replace(" ", "-").replace(".", "").replace("'", "")
    path = OUT / f"card-{slug}-{season}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def pick_examples(cards: pl.DataFrame, season: int = 2025) -> list[int]:
    """A few rookies with the most snaps in different roles, preferring ones the card flags."""
    last = (cards.filter((pl.col("season") == season) & (pl.col("nfl_year") == 1))
            .filter(pl.col("week") == pl.col("week").max().over("nfl_id")))
    per = last.group_by("nfl_id", "role").agg(pl.col("snaps").first(), flags=(pl.col("flag") != "").sum())
    picks = []
    for role in ("WR", "EDGE", "CB", "OT"):
        r = per.filter(pl.col("role") == role).sort(["flags", "snaps"], descending=True)
        if r.height:
            picks.append(int(r["nfl_id"][0]))
    return picks


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    snaps = load_snaps()

    scored, summary = nested_validation(snaps)
    cal = tc.calibrate(snaps, SEASONS)
    widths = tc.conformal_widths(tc.blocked_residuals(snaps, SEASONS))
    cards = tc.weekly_cards(snaps, cal, widths).with_columns(flag=flag(pl.col("pct")))
    cards.write_parquet(OUT / "weekly_trait_cards.parquet")
    cards.select("season", "week", "nfl_id", "display_name", "role", "nfl_year", "snaps", "trait", "raw",
                 "pct", "pct_lo", "pct_hi", "flag").with_columns(pl.col(pl.Float64).round(2)).write_csv(OUT / "weekly_trait_cards.csv")
    cal.write_csv(OUT / "calibration.csv")
    widths.write_csv(OUT / "conformal_widths.csv")
    summary["rho_by_budget"] = cal.select("budget", "trait", "rho").unique().sort("trait", "budget").to_dicts()
    (OUT / "validation.json").write_text(json.dumps(summary, indent=1))

    fig_validation(cal, summary)
    paths = [fig_card(cards, pid, 2025) for pid in pick_examples(cards)]
    print(json.dumps({k: summary[k] for k in ("coverage_overall", "coverage_by_trait", "hit_rates")}, indent=1))
    print("cards:", [p.name for p in paths])


if __name__ == "__main__":
    main()
