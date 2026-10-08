"""Compute every number and figure used in the writeup.

    python scripts/build_features.py   # once
    python scripts/make_report.py      # -> reports/figures/*.png, reports/results.json
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
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bdb27 import data, reliability as rl, traits, translation as tr  # noqa: E402

FIG = data.REPORTS / "figures"
GROUPS = ("WR", "DB", "OL", "DL")  # TE: too few with >= 50 tracked NFL snaps
SHORT = {"FORTY_YARD_DASH": "40 (tracked)", "SHORT_SHUTTLE": "Shuttle (tracked)", "THREE_CONE_DRILL": "3-cone (tracked)",
         "forty": "40 time", "ten_yd_split": "10-yd split", "three_cone": "3-cone time", "short_shuttle": "Shuttle time",
         "vertical": "Vertical", "broad_jump": "Broad jump"}


def short(src: str) -> str:
    if src in SHORT:
        return SHORT[src]
    words = src.replace("_DRILL", "").replace("_ROUTE", "").replace("_LEFT", " L").replace("_RIGHT", " R").split("_")
    return " ".join(words).title()[:16]


TRAIT_LABEL = {"top_speed": "Top speed", "burst": "Burst", "brake": "Brake", "bend": "Bend"}

# Reference palette (dataviz skill): validated all-pairs for 3 slots, light surface.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
DIVERGING = LinearSegmentedColormap.from_list("bdb_div", ["#d03b3b", "#f0b5ad", "#f0efec", "#9ec5f4", "#1c5cab"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10.5,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "lines.linewidth": 2, "legend.frameon": False,
})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def load():
    P = data.PROCESSED
    base = data.player_base()
    ct = pl.read_parquet(P / "combine_traits.parquet")
    at = pl.read_parquet(P / "combine_attempt_traits.parquet")
    # Remove season-level tracking drift (estimated from the same players across seasons) before anything else.
    pt = traits.season_adjust(pl.read_parquet(P / "game_play_traits.parquet"), data.games(),
                              base.select("nfl_id", pl.col("pos_group").alias("group")))
    gt = traits.game_player_traits(pt)
    go = pl.read_parquet(P / "game_outcomes.parquet")
    return base, ct, at, pt, gt, go


# --------------------------------------------------------------------------- figure 1
def fig_concept(base, pt):
    """Same pipeline, two settings: a DL's Run-the-Hoop rep and one of his NFL rushes, coloured by lateral acceleration."""
    ct = (data.combine_tracking()
          .filter((pl.col("entity_type") == "PLAYER") & (pl.col("drill_name") == "RUN_THE_HOOP_DRILL"))
          .collect().sort("nfl_id", "attempt", "time"))
    dl_ids = set(base.filter(pl.col("pos_group") == "DL")["nfl_id"].to_list())
    rushes = pt.filter(pl.col("nfl_id").is_in(list(dl_ids)))
    counts = rushes.group_by("nfl_id").len().filter(pl.col("len") > 300)
    pid = int(ct.join(counts, on="nfl_id").sort("len", descending=True)["nfl_id"][0])
    rep = ct.filter((pl.col("nfl_id") == pid) & (pl.col("attempt") == ct.filter(pl.col("nfl_id") == pid)["attempt"].min()))
    play = rushes.filter(pl.col("nfl_id") == pid).sort("bend", descending=True).row(len(rushes.filter(pl.col("nfl_id") == pid)) // 10, named=True)
    gt = (pl.scan_csv(data.RAW / f"game_tracking_{str(play['game_id'])[:4] if str(play['game_id'])[4:6] != '01' else int(str(play['game_id'])[:4]) - 1}.csv", null_values="NA")
          .filter((pl.col("game_id") == play["game_id"]) & (pl.col("play_id") == play["play_id"]) & (pl.col("nfl_id") == pid))
          .collect().with_columns(pl.col("time").str.to_datetime()).sort("time"))
    snap = gt["time"].filter(gt["event"] == "ball_snap")[0]
    gt = gt.filter((pl.col("time") >= snap) & (pl.col("time") <= snap + pl.duration(seconds=traits.GAME_WINDOW_S)))
    name = base.filter(pl.col("nfl_id") == pid)["display_name"][0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
    vmax = 8.0
    cmap = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#6da7ec", "#2a78d6", "#104281"])
    for ax, df, title in [(axes[0], rep, "Combine: Run-the-Hoop drill"), (axes[1], gt, "NFL: one pass rush (first 5 s after snap)")]:
        x, y = df["x"].to_numpy(), df["y"].to_numpy()
        k = traits._derivatives(df.with_columns(seg=pl.lit(0)), ["seg"])
        xs, ys, lat = k["x"].to_numpy(), k["y"].to_numpy(), np.where(k["v"].to_numpy() >= traits.BEND_MIN_SPEED, k["a_lat"].to_numpy(), np.nan)
        ax.plot(x - x.mean(), y - y.mean(), color=GRID, lw=6, solid_capstyle="round", zorder=1)
        sc = ax.scatter(xs - x.mean(), ys - y.mean(), c=np.clip(lat, 0, vmax), cmap=cmap, vmin=0, vmax=vmax, s=28,
                        edgecolors=SURFACE, linewidths=0.6, zorder=2)
        ax.plot(x[0] - x.mean(), y[0] - y.mean(), marker="o", ms=9, color=INK, zorder=3)
        ax.annotate("start", (x[0] - x.mean(), y[0] - y.mean()), xytext=(6, 6), textcoords="offset points", color=INK2)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("yards")
    cb = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("lateral acceleration (yd/s², speed ≥ 2 yd/s)")
    fig.suptitle(f"Figure 1  |  One pipeline for both settings: {name}'s bend at the Combine and on Sundays", x=0.01, ha="left", fontsize=13, fontweight="bold")
    save(fig, "figure-1-one-pipeline.png")
    return {"player": name}


# --------------------------------------------------------------------------- figure 2
def fig_reliability(summ: pl.DataFrame):
    fig, axes = plt.subplots(1, 4, figsize=(12, 4.0), sharey=True)
    series = [("combine_cross_drill", "Combine: same trait, different drill", AQUA, "s"),
              ("combine_retest", "Combine: same drill, 2nd attempt", ORANGE, "D"),
              ("nfl_split_half", "NFL: odd vs even snaps", BLUE, "o")]
    ypos = {g: i for i, g in enumerate(reversed(GROUPS))}
    for ax, t in zip(axes, traits.TRAITS):
        s = summ.filter(pl.col("trait") == t)
        for g in GROUPS:
            row = s.filter(pl.col("group") == g)
            if row.is_empty():
                continue
            vals = [row[c][0] for c, *_ in series]
            # Small vertical offsets keep coincident markers visible (e.g. DL burst, DB top speed).
            for k, ((c, lab, col, mk), v) in enumerate(zip(series, vals)):
                if v is not None:
                    y = ypos[g] + (1 - k) * 0.17
                    ax.plot([0, v], [y, y], color=GRID, lw=2, zorder=1)
                    ax.plot(v, y, marker=mk, ms=8, color=col, mec=SURFACE, mew=1.2, ls="none", zorder=3)
        ax.set_title(TRAIT_LABEL[t])
        ax.set_xlim(-0.2, 1.0)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_yticks(list(ypos.values()), list(ypos.keys()))
        ax.set_xlabel("reliability (partial ρ)")
    handles = [plt.Line2D([], [], marker=mk, color=col, ls="none", ms=9, mec=SURFACE, label=lab) for _, lab, col, mk in series]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 1.03), ncol=3)
    fig.suptitle("Figure 2  |  Stable on Sundays, noisy at the Combine", x=0.01, y=1.2,
                 ha="left", fontsize=13, fontweight="bold")
    fig.text(0.01, 1.135, "Reliability by trait and position group (partial Spearman ρ; controls: weight, roster position).\n"
             "NFL traits repeat; Combine change-of-direction numbers barely repeat and don't agree across drills.",
             ha="left", va="top", fontsize=9.5, color=INK2, linespacing=1.3)
    save(fig, "figure-2-reliability.png")


# --------------------------------------------------------------------------- figure 3
def fig_translation(tmap: pl.DataFrame):
    best = (tmap.filter(pl.col("group").is_in(GROUPS))
            .sort(pl.col("partial").abs(), descending=True)
            .group_by("group", "trait", "official", maintain_order=True).first())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, official, title in [(axes[0], True, "Best stopwatch / tape-measure test"), (axes[1], False, "Best sensor-tracked drill")]:
        M = np.full((len(GROUPS), len(traits.TRAITS)), np.nan)
        for i, g in enumerate(GROUPS):
            for j, t in enumerate(traits.TRAITS):
                r = best.filter((pl.col("group") == g) & (pl.col("trait") == t) & (pl.col("official") == official))
                if r.is_empty():
                    continue
                M[i, j] = r["partial"][0]
                src = short(r["source"][0])
                star = "  ✱" if r["p_fwer"][0] < 0.05 else ""
                ink = SURFACE if abs(M[i, j]) > 0.45 else INK
                ax.text(j, i - 0.12, f"{M[i, j]:+.2f}{star}", ha="center", va="center", fontsize=11, fontweight="bold", color=ink)
                ax.text(j, i + 0.24, f"{src}\nn={r['n'][0]}", ha="center", va="center", fontsize=7.5, color=ink, linespacing=1.1)
        im = ax.imshow(M, cmap=DIVERGING, vmin=-0.7, vmax=0.7, aspect="auto")
        ax.set_xticks(range(4), [TRAIT_LABEL[t] for t in traits.TRAITS])
        ax.set_yticks(range(len(GROUPS)), GROUPS)
        ax.grid(False)
        ax.set_title(title)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(GROUPS), 1), minor=True)
        ax.grid(which="minor", color=SURFACE, lw=3)
        ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label("partial ρ with same trait in NFL games")
    fig.suptitle("Figure 3  |  Translation map: Combine trait vs. the same trait in NFL snaps", x=0.01, y=1.06, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.01, 0.985, "Strongest single source per cell; partial Spearman ρ controlling for weight and roster position.  ✱ = family-wise permutation p < 0.05",
             ha="left", fontsize=9.5, color=INK2)
    save(fig, "figure-3-translation-map.png")


# --------------------------------------------------------------------------- figure 4
def fig_speed(base, ct, gt):
    wr = base.filter(pl.col("pos_group") == "WR").join(gt, on="nfl_id")
    tracked = wr.join(ct.filter(pl.col("drill_name") == "FORTY_YARD_DASH").select("nfl_id", "top_speed"), on="nfl_id").drop_nulls(["top_speed", "combine_weight"])
    stop = wr.drop_nulls(["forty", "combine_weight"])
    r1 = tr.partial_spearman(tracked["top_speed"].to_numpy(), tracked["game_top_speed"].to_numpy(), tr._controls(tracked))
    r2 = tr.partial_spearman(-stop["forty"].to_numpy(), stop["game_top_speed"].to_numpy(), tr._controls(stop))
    to_mph = 3600 / 1760
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, x, y, xl, r in [(axes[0], tracked["top_speed"] * to_mph, tracked["game_top_speed"] * to_mph, "Combine 40: tracked peak speed (mph)", r1),
                            (axes[1], stop["forty"], stop["game_top_speed"] * to_mph, "Combine 40: official time (s)", r2)]:
        ax.scatter(x, y, s=42, color=BLUE, edgecolors=SURFACE, linewidths=1.2, alpha=0.9)
        ax.set_xlabel(xl)
        ax.set_title(f"partial ρ = {r:+.2f}  (n = {len(x)})" if ax is axes[0] else f"partial ρ = {r:+.2f} on −time  (n = {len(x)})")
    axes[1].invert_xaxis()
    axes[0].set_ylabel("NFL top speed, 90th pct of snaps (mph)")
    fig.suptitle("Figure 4  |  WR speed travels, and the sensor reads it better than the stopwatch", x=0.01, y=1.03, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.01, 0.955, "ρ = partial Spearman controlling for body weight", ha="left", fontsize=9.5, color=INK2)
    save(fig, "figure-4-wr-speed.png")
    return {"wr_speed_tracked": r1, "wr_speed_stopwatch": r2, "n_tracked": tracked.height, "n_stopwatch": stop.height}


# --------------------------------------------------------------------------- figure 5
def fig_dl(base, ct, gt, go):
    dl = base.filter(pl.col("pos_group") == "DL").join(gt, on="nfl_id").join(go, on="nfl_id", how="left")
    a = dl.join(ct.filter(pl.col("drill_name") == "SHORT_SHUTTLE").select("nfl_id", pl.col("burst").alias("shuttle_burst")), on="nfl_id").drop_nulls(["shuttle_burst", "combine_weight"])
    b = dl.filter(pl.col("pass_rush_snaps") >= 100).drop_nulls(["pressure_rate", "combine_weight"])
    ra = tr.partial_spearman(a["shuttle_burst"].to_numpy(), a["game_burst"].to_numpy(), tr._controls(a))
    rb = tr.partial_spearman(b["game_burst"].to_numpy(), b["pressure_rate"].to_numpy(), tr._controls(b))
    rc = tr.partial_spearman(b["game_burst"].to_numpy(), -b["mean_get_off"].to_numpy(), tr._controls(b.drop_nulls("mean_get_off")))
    def within(d, x, y):
        out = {}
        for role, mask in (("edge", pl.col("nfl_position").is_in(["DE", "OLB"])), ("interior", ~pl.col("nfl_position").is_in(["DE", "OLB"]))):
            g = d.filter(mask)
            out[role] = (tr.partial_spearman(g[x].to_numpy(), g[y].to_numpy(), tr._controls(g)), g.height)
        return out

    wa, wb = within(a, "shuttle_burst", "game_burst"), within(b, "game_burst", "pressure_rate")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    edge = lambda d: d["nfl_position"].is_in(["DE", "OLB"]).to_numpy()  # noqa: E731
    for ax, d, x, y, xl, yl, r, w in [
        (axes[0], a, "shuttle_burst", "game_burst", "Combine short shuttle: tracked burst (yd/s²)", "NFL burst, 90th pct of snaps (yd/s²)", ra, wa),
        (axes[1], b, "game_burst", "pressure_rate", "NFL burst, 90th pct of snaps (yd/s²)", "pressure rate per pass rush", rb, wb),
    ]:
        e = edge(d)
        ax.scatter(d[x].to_numpy()[e], d[y].to_numpy()[e], s=44, color=BLUE, marker="o", edgecolors=SURFACE, linewidths=1.2,
                   label=f"edge (DE/OLB): ρ = {w['edge'][0]:+.2f}, n = {w['edge'][1]}")
        ax.scatter(d[x].to_numpy()[~e], d[y].to_numpy()[~e], s=44, color=ORANGE, marker="D", edgecolors=SURFACE, linewidths=1.2,
                   label=f"interior (DT/NT): ρ = {w['interior'][0]:+.2f}, n = {w['interior'][1]}")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f"all: partial ρ = {r:+.2f} (n = {d.height})")
        ax.legend(loc="lower right" if ax is axes[0] else "upper left", fontsize=9)
    fig.suptitle("Figure 5  |  DL burst: shuttle rep → NFL burst → edge-rusher pressure",
                 x=0.01, y=1.03, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.01, 0.955, "ρ = partial Spearman controlling for weight and roster position, overall and within role", ha="left", fontsize=9.5, color=INK2)
    save(fig, "figure-5-dl-burst.png")
    return {"dl_shuttle_to_game_burst": ra, "n_shuttle": a.height, "dl_game_burst_to_pressure": rb, "n_pressure": b.height,
            "dl_game_burst_to_getoff": rc, "within_shuttle": wa, "within_pressure": wb}


def bootstrap_ci(x, y, Z, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        i = rng.integers(0, len(x), len(x))
        Zi = Z[i][:, Z[i].std(0) > 0] if Z.size else Z
        out.append(tr.partial_spearman(x[i], y[i], Zi))
    return [float(v) for v in np.nanpercentile(out, [2.5, 97.5])]


def main():
    base, ct, at, pt, gt, go = load()
    FIG.mkdir(parents=True, exist_ok=True)
    ctl = base.select("nfl_id", "pos_group", "nfl_position", "combine_weight", "draft_year").drop_nulls()
    res = {}

    retest = rl.combine_retest(at, ctl, GROUPS)
    cross = rl.cross_drill(ct, ctl, GROUPS)
    game = rl.game_split_half(pt, ctl, GROUPS)
    summ = rl.summary(retest, cross, game).sort("group", "trait")
    retest = retest.sort("group", "drill", "trait")
    summ.write_csv(data.REPORTS / "reliability_summary.csv")
    retest.write_csv(data.REPORTS / "combine_retest.csv")
    res["reliability"] = summ.sort("group", "trait").to_dicts()
    res["retest_by_drill"] = retest.to_dicts()

    tmap = tr.add_fwer(tr.translation_map(base, ct, gt, GROUPS), base, ct, gt)
    tmap.write_csv(data.REPORTS / "translation_map.csv")
    res["translation_survivors"] = tmap.filter(pl.col("p_fwer") < 0.05).sort("p_fwer").to_dicts()
    res["n_tests"] = tmap.height
    pos_drills = tmap.filter(~pl.col("official") & ~pl.col("source").is_in(["FORTY_YARD_DASH", "SHORT_SHUTTLE", "THREE_CONE_DRILL"]))
    res["position_drill_tests"] = pos_drills.height
    res["position_drill_survivors"] = pos_drills.filter(pl.col("p_fwer") < 0.05).height

    # Pooled reps-needed table: median single-rep retest per trait across groups and drills.
    # Table 1: position drills (coach-led) per trait, plus the tracked 40 for top speed.
    pos_rt = (retest.filter(pl.col("drill") != "FORTY_YARD_DASH").group_by("trait")
              .agg(r=pl.col("r").median(), drills=pl.col("drill").n_unique()).with_columns(source=pl.lit("position drills")))
    forty_rt = (retest.filter((pl.col("drill") == "FORTY_YARD_DASH") & (pl.col("trait") == "top_speed")).group_by("trait")
                .agg(r=pl.col("r").median(), drills=pl.col("drill").n_unique()).with_columns(source=pl.lit("tracked 40")))
    reps = (pl.concat([forty_rt, pos_rt]).sort("source", "trait")
            .with_columns(reps_for_0_8=pl.col("r").map_elements(rl.reps_needed, return_dtype=pl.Float64)))
    res["reps_needed"] = reps.to_dicts()

    res["fig1"] = fig_concept(base, pt)
    fig_reliability(summ)
    fig_translation(tmap)
    res["fig4"] = fig_speed(base, ct, gt)
    res["fig5"] = fig_dl(base, ct, gt, go)

    # Bootstrap CIs for the headline correlations.
    wr = base.filter(pl.col("pos_group") == "WR").join(gt, on="nfl_id").join(
        ct.filter(pl.col("drill_name") == "FORTY_YARD_DASH").select("nfl_id", "top_speed"), on="nfl_id").drop_nulls(["top_speed", "combine_weight"])
    wr = wr.sort("nfl_id")  # bootstrap resamples depend on row order
    res["fig4"]["ci_tracked"] = bootstrap_ci(wr["top_speed"].to_numpy(), wr["game_top_speed"].to_numpy(), tr._controls(wr))
    dl = base.filter(pl.col("pos_group") == "DL").join(gt, on="nfl_id").join(go, on="nfl_id").filter(pl.col("pass_rush_snaps") >= 100).drop_nulls(["pressure_rate", "combine_weight"])
    dl = dl.sort("nfl_id")
    res["fig5"]["ci_burst_pressure"] = bootstrap_ci(dl["game_burst"].to_numpy(), dl["pressure_rate"].to_numpy(), tr._controls(dl))

    res["coverage"] = {"players": base.height, "with_game_traits": gt.height,
                       "game_player_plays": pt.height, "combine_attempts": at.height}
    (data.REPORTS / "results.json").write_text(json.dumps(res, indent=1, default=float))
    print(summ.sort("trait", "group"))
    print(reps)
    print(json.dumps({k: res[k] for k in ("fig4", "fig5", "n_tests", "position_drill_tests", "position_drill_survivors", "coverage")}, indent=1, default=float))


if __name__ == "__main__":
    main()
