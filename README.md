# NFL Big Data Bowl 2027

Our entry for the [NFL Big Data Bowl 2027](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027) on Kaggle.

> **Prompt:** uncover non-obvious linkages between Combine sensor tracking and regular-season NFL game performance.

| | |
|---|---|
| Format | Analytics competition (judged write-up, no leaderboard) |
| Tracks | Open; Undergraduate/Graduate |
| Prize | $100,000 shared; finalists present at the 2027 NFL Scouting Combine |
| Team size | up to 5 |
| Deadline | **2027-01-06 23:59 UTC** (also the team-merger deadline) |
| Data license | CC BY-NC 4.0, so the data is **not** committed here |

## Data at a glance

510 drafted players from the 2023–2025 Combines, in five position groups: WR (107), TE (42), DB (122), OL (121), DL (118).

| file | rows | what it is |
|---|---|---|
| `players.csv` | 510 | bio, college, draft round/pick, NFL position |
| `combine_results.csv` | 510 | official measurements: height, weight, hands, arms, wingspan, 10/40, vertical, broad, 3-cone, shuttle, bench, NGS scores |
| `combine_tracking.csv` | 463k | 10 Hz Zebra tracking of every Combine drill attempt (40, 3-cone, shuttle + 55 position drills), `x, y, s, a, dis, dir` |
| `player_career_successes.csv` | 510 | career snaps (off/def/ST), games active/started, All-Pro, Pro Bowl |
| `games.csv` | 1,002 | 2023–2025 PRE/REG/POST games |
| `player_play.csv` | 314k | one row per player-play: context, EPA, routes, separation, pressure allowed/generated, get-off, tackles, coverage |
| `game_tracking_{2023,2024,2025}.csv` | 22.6M | 10 Hz in-game tracking of the same players (`x, y, s, a, dis, o, dir, event`) |

Things to know about the data:
- **Combine `dis` is unreliable.** It lags and under-counts (about 37 yd summed over a 40-yd dash). We use the x/y path length instead.
- **Combine clips start in motion**, so tracking-derived 40 times run about 0.3 s slower than the official laser times. They still correlate at r = 0.91.
- **Drill names drift across years** (`LINE` → `LINE_DRILL`, `PASS_PRO_MIRROR_DRILL` → `PASS_PRO-MIRROR_DRILL`, …). They're canonicalized in `combine_features.DRILL_ALIASES`. A drill run in only one year is confounded with draft class.
- Undrafted players have no draft pick; we treat them as pick 260.
- Exposure differs by class: 2023 draftees have three seasons of games, 2025 draftees only one.

## Entry #1: *Measure the Athlete, Not the Drill*

Writeup: [`reports/writeup.md`](reports/writeup.md) · notebook: [`notebooks/measure-the-athlete.ipynb`](notebooks/measure-the-athlete.ipynb) · figures: [`reports/figures/`](reports/figures)

Four movement traits (top speed, burst, brake and bend) are computed by **one** pipeline on Combine reps and on NFL snaps, then compared for the same player:

- **Speed travels.** Tracked 40 peak speed repeats at ρ ≈ 0.89 and predicts WR NFL top speed better than the stopwatch time (partial ρ 0.58 vs 0.49).
- **Change of direction is stable in games (split-half 0.63–0.95) but not at the Combine.** A single position-drill rep repeats at 0.24–0.31, two drills agree at 0.01–0.14, and 0 of 144 position-drill → NFL tests survive a family-wise permutation test.
- **Max-effort tests are the exception.** The tracked short shuttle predicts DL NFL burst (0.59) and brake (0.63), and NFL burst predicts edge-rusher pressure rate (0.43).

## Setup

```bash
pip install -r requirements.txt
export KAGGLE_API_TOKEN=...          # kaggle.com → Settings → API
./scripts/download_data.sh           # ~500 MB zip → data/raw/ (~2.3 GB)
python scripts/build_features.py     # → data/processed/*.parquet  (~30 s)
python scripts/make_report.py        # → reports/figures/*.png, reports/results.json (entry #1)
python scripts/make_notebook.py      # → notebooks/measure-the-athlete.ipynb (self-contained Kaggle notebook)
python scripts/first_look.py         # → reports/first_look.md (early exploratory screen)
```

## Layout

```
src/bdb27/
  data.py               paths + loaders (large CSVs cached to Parquet in data/interim/)
  traits.py             top speed / burst / brake / bend, one pipeline for Combine reps and NFL snaps
  translation.py        Combine -> NFL translation map, partial Spearman, family-wise permutation p-values
  reliability.py        Combine retest, cross-drill agreement, NFL split-half, Spearman-Brown
  combine_features.py   per-attempt kinematics from Combine tracking (drill-name harmonisation lives here)
  game_features.py      regular-season production rates (player_play) + in-game speed ceilings
  routes.py             route-break mechanics (Combine route drills vs NFL routes; exploratory)
  bend.py, pass_rush.py DL Run-the-Hoop bend and in-game pass-rush kinematics (exploratory)
scripts/
  download_data.sh      fetch competition data via the Kaggle API
  build_features.py     build all processed tables
  make_report.py        every number + figure in the writeup
  make_notebook.py      generate the Kaggle notebook from the package source
  first_look.py         early correlation screen vs. outcomes, controlling for draft slot
reports/                writeup, figures, results
notebooks/              Kaggle notebook
```

### Features so far

**Combine, per drill attempt:** peak speed; mean speed; peak acceleration; signed peak longitudinal acceleration and deceleration; time to peak speed; distance; duration; total turning; peak angular velocity; speed held through the sharpest turn. The 40 also gets 10/20/40-yd splits and flying-10 speed. Attempts are collapsed to the best value per player and drill.

**Game, regular season only, nullified plays dropped:**
- receivers: targets per route, yards per route, separation at throw, YAC over expected
- blockers: pressure-allowed rate, peak pressure probability allowed
- rushers: pressure rate, get-off, time to pressure
- defenders: cushion, tackles, TFL
- everyone: snaps per game, in-game top speed (99.9th percentile), 95th-percentile speed, 99th-percentile acceleration

## First look (exploratory)

The full tables are in [`reports/first_look.md`](reports/first_look.md). Each table ranks Spearman ρ and partial ρ, the partial controlling for draft slot, which asks: what does this measurement say *beyond* what teams already priced in?

- **Combine speed carries into games.** Tracking peak speed in the 40 correlates with in-game top speed at r ≈ 0.86 across everyone (WR partial ρ ≈ +0.55).
- **DL:** the 3-cone, shuttle and 40 relate to pressure rate (|ρ| ≈ 0.55–0.7), and broad jump, weight and 40 relate to get-off (|ρ| ≈ 0.7). Much of this is probably **interior vs. edge** role, not athleticism.
- **OL:** longer arms and wingspan go with *more* pressure allowed (ρ ≈ +0.45). This is almost certainly **tackle vs. guard**: tackles have longer arms and face better rushers. The real comparison needs a within-role control.
- **WR:** height and weight correlate negatively with separation (ρ ≈ −0.45), and change-of-direction signals from the shuttle and route drills tie to starts.

**Caveat:** about 1,900 pairs were tested on 25–120 players each, so many of these will be noise.

## Roadmap

1. **Control for confounders.** Sub-position and alignment (T/G/C; DT/DE/OLB; CB/S, slot vs. wide), draft class and exposure, team context.
2. **Better outcomes.** Per-snap value models, e.g. separation over expected given route and coverage, or pressure over expected given matchup.
3. **Better Combine features.** Drill-specific segmentation: break-point timing in route drills, plant-and-drive in DB drills, hip-flip in back-pedal drills. Also compare the same movement at the Combine and in games (e.g. a WR's route-break deceleration in both).
4. **Validation.** Out-of-class tests (fit on 2023–24, test on 2025), bootstrap CIs, multiple-testing control.
5. **Story and visuals.** Pick 1–2 non-obvious, robust findings and build the write-up and animations around them.
