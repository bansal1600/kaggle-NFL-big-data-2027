# Measure the Athlete, Not the Drill

**Subtitle:** Which Combine sensor numbers survive the trip to Sundays? A reliability-first look at speed, burst, brake and bend.

---

## The question behind every Combine number

Before a front office trusts a Combine measurement, it needs answers to three questions:

1. **Is it repeatable?** Would the prospect post the same number if he ran the drill again?
2. **Does it travel?** Does the same movement show up when he plays NFL snaps?
3. **Does it matter?** When it shows up on Sundays, does it help him win?

Stopwatch times only answer these indirectly. Sensor tracking at the Combine and in games lets us answer all three *with the same ruler*. This entry builds that ruler. It then measures every Combine drill against it for 510 prospects from the 2023–25 classes: 6,310 drill reps and 250,461 regular-season player-snaps.

**The short answer:**

- **Speed travels.** The tracked 40-yard dash is nearly perfectly repeatable (rep-to-rep ρ ≈ 0.89). For receivers it predicts NFL top speed *better than the stopwatch time from the same rep* (ρ 0.58 vs 0.49).
- **Change of direction mostly doesn't, because the Combine measures it poorly.** On Sundays, burst, brake and bend are stable player traits (split-half reliability 0.63–0.95). At the Combine, a position-drill rep barely repeats (median ρ 0.24–0.31), and two different drills almost never agree on the same trait (median ρ 0.01–0.14). Of 144 tests linking a coach-led position drill to the same trait in games, **none** survive a family-wise permutation test.
- **Max-effort, standardized tests are the exception.** The tracked short shuttle predicts a defensive lineman's NFL burst (ρ 0.59) and braking (ρ 0.63). NFL burst in turn predicts an edge rusher's pressure rate (ρ 0.43). Yet only 39% of prospects run the shuttle.

## One pipeline, four traits

Tracking rows are 10 Hz positions. We ignore the provided speed, acceleration and distance columns; over a 40, the Combine `dis` column sums to about 10% less than the x/y path. Instead we re-derive velocity and acceleration from x/y with a Savitzky–Golay filter (0.7 s window, quadratic). Acceleration is then split into the component **along** the path and the component **across** it:

| Trait | Definition (per rep or per snap) |
|---|---|
| **Top speed** | max speed |
| **Burst** | max tangential acceleration (speeding up) |
| **Brake** | max tangential deceleration (slowing down) |
| **Bend** | max lateral (centripetal) acceleration, v²/r, while moving ≥ 2 yd/s |

Exactly the same code runs on a Combine rep and on the first five seconds after an NFL snap (Figure 1). A player's **NFL trait** is the 90th percentile of his per-snap maxima, his "top 10% of snaps" (min. 50 snaps; 337 players qualify). His **Combine trait** for a drill is his best rep.

![Figure 1: one pipeline for Combine and NFL](figure-1-one-pipeline.png)
*Figure 1. Bend is lateral acceleration. Byron Young's Run-the-Hoop rep (left) and one of his NFL pass rushes (right) are coloured by the same scale and computed by the same function.*

**Controls everywhere.** A 330-lb tackle and a 190-lb corner differ in every trait for reasons no scout needs a sensor to see. All correlations are therefore **partial Spearman ρ within a position group**, controlling for body weight and roster position (e.g. DT vs DE vs OLB). The question is always whether a measurement separates players *of the same size and role*.

## 1. Sundays are stable; the Combine is not

We measure reliability three ways, using the same controls:

- **Combine retest:** a player's 1st vs 2nd attempt of the same drill (the 40 and six position drills have enough repeat attempts).
- **Combine cross-drill:** the same trait measured in two *different* drills (e.g. brake in the curl route vs brake in the gauntlet).
- **NFL split-half:** odd vs even snaps, Spearman–Brown corrected.

![Figure 2: reliability](figure-2-reliability.png)
*Figure 2. Reliability of each trait by position group. Blue (NFL) sits far right for every trait; the Combine markers do not.*

Three things stand out:

1. **NFL movement traits are real, stable player characteristics.** Split-half reliability is 0.63–0.95 for burst, brake and bend in every group. A receiver who brakes hard on his odd snaps brakes hard on his even snaps too.
2. **The 40 is the Combine's one precision instrument.** Tracked 40 top speed repeats at ρ = 0.88–0.91 across groups. The same trait in position drills repeats at only 0.51.
3. **Position drills measure the drill.** Change-of-direction traits from a single position-drill rep repeat at median ρ 0.24 (brake) to 0.31 (burst, bend). Across *different* drills the same trait barely correlates (median 0.01–0.14). A receiver's brake in the curl drill tells you almost nothing about his brake in the comeback drill. These drills are coach-paced teaching reps, not max-effort tests, and the sensors faithfully record that.

**What it would take.** With the Spearman–Brown formula you can ask how many position-drill reps you would need to average for a reliable (0.8) Combine trait:

| Trait | Single-rep reliability | Reps needed for 0.8 |
|---|---|---|
| Top speed (tracked 40) | 0.89 | 1 |
| Burst (position drills) | 0.31 | 9 |
| Bend (position drills) | 0.31 | 9 |
| Brake (position drills) | 0.24 | 13 |

*Table 1. Median retest reliability within position group, after weight/position controls.*

No prospect runs nine reps of a drill. As currently run, position drills cannot deliver a trustworthy change-of-direction number, however good the sensor.

## 2. The translation map

Next we correlate every Combine source with the *same* trait in the player's NFL snaps. The sources are the 40, shuttle, 3-cone and every position drill run by at least 25 players, plus the six stopwatch/tape tests. That is 260 tests in 16 families (position group × trait).

Searching many drills guarantees some impressive-looking correlations by chance. Within each family we therefore shuffle the NFL trait across players 1,000 times, recompute every test, and compare each observed ρ with the distribution of the *family maximum*. This gives a family-wise p-value that already accounts for picking the best drill.

![Figure 3: translation map](figure-3-translation-map.png)
*Figure 3. Strongest single source in each cell; ✱ = family-wise p < 0.05. Red cells are as informative as blue ones: when the "best" drill has the wrong sign, the family is noise.*

Five relationships survive:

| Group | Trait | Combine source | n | partial ρ | family-wise p |
|---|---|---|---|---|---|
| WR | Top speed | 40, tracked peak speed | 64 | **0.58** | 0.003 |
| WR | Top speed | 40, stopwatch time | 65 | 0.49 | 0.009 |
| DL | Brake | short shuttle, tracked | 25 | **0.63** | 0.004 |
| DL | Burst | short shuttle, tracked | 25 | **0.59** | 0.006 |
| DB | Bend | broad jump | 61 | 0.36 | 0.048 |

*Table 2. Combine → NFL translations surviving the family-wise test. **0 of 144** coach-led position-drill tests survive.*

The pattern follows Figure 2 exactly. What translates comes from **max-effort, standardized** tests (the 40, the shuttle and the broad jump). Coach-led position drills, with near-zero cross-drill agreement, do not translate. A few of them produce eye-catching single cells (WR Dagger route bend ρ 0.35), but these do not beat the permutation maximum.

## 3. Where it works

### Receivers: speed travels, and the sensor reads it better

![Figure 4: WR speed](figure-4-wr-speed.png)
*Figure 4. WR NFL top speed (90th percentile of snaps) vs the tracked 40 peak speed and the official 40 time.*

Tracked peak speed from the 40 explains NFL top speed better than the official time from the same sprint (ρ 0.58 [95% bootstrap CI 0.40–0.72] vs 0.49). The time is an *average* over 40 yards, so it mixes start, acceleration and top-end. The sensor isolates the top-end, which is the part a receiver uses on a vertical route.

### Defensive line: shuttle burst → NFL burst → pressure

![Figure 5: DL burst](figure-5-dl-burst.png)
*Figure 5. Left: tracked shuttle burst vs NFL burst (holds within edge and interior players). Right: NFL burst vs pressure rate per pass rush (unblocked pressures excluded), with partial ρ overall and within role.*

The chain has two links:

- **Combine → NFL.** The tracked shuttle's burst predicts a lineman's NFL burst (ρ 0.59, n = 25). It holds within edge (0.57) and interior players (0.85). The shuttle's stopwatch time does worse (0.34); the sensor reads the explosive change of direction that the time averages away.
- **NFL burst → production.** Linemen with more NFL burst generate more pressure (ρ 0.34 [0.10–0.53], n = 68) and get off the ball faster (ρ 0.42 with NGS get-off). The pressure link lives among **edge rushers** (ρ 0.43, n = 42) and is absent for interior linemen (0.05, n = 26). That fits football logic: interior rushers win with hands and power in a phone booth, while edge rushers must win a race.

The weak link is participation. Only **39% of DL prospects** ran the shuttle (and 36% the 3-cone), so the one Combine test that reads pass-rush burst is missing for most of the class.

## What teams and the league can do

**For scouting departments (draft season and in-season pro scouting):**

1. **Use tracked 40 peak speed, not just the time,** as the speed input in WR and DB models. It is the most reliable number the Combine produces, and it beats the stopwatch at predicting game speed.
2. **Discount single position-drill reps.** Treat sensor outputs from coach-led drills as film context, not as a trait score. A standout curl-drill "brake" number repeats at ρ ≈ 0.24.
3. **Use NFL tracking as the trait source once it exists.** Because NFL traits are so stable (0.63–0.95 split-half), a rookie's first 50–100 tracked snaps tell you more about his burst or bend than his entire Combine. Re-rank young players and practice-squad candidates on these four traits week to week, using the identical pipeline in the notebook.
4. **For edge rushers, NFL burst is a production indicator** with a clear football mechanism. Track it weekly as an early read on pass-rush development.

**For the league office (Combine design):**

5. **Make the shuttle the standard agility test for linemen,** or add a short, max-effort shuttle-like rep to every DL workout. It is the only agility measure that translated.
6. **If position drills are meant to measure athletes, standardize them:** fixed geometry, max-effort instructions and multiple reps. Table 1 puts the cost at roughly 9–13 reps per trait; one rep per drill cannot deliver it.

## Limitations

- **Small samples.** n = 25–79 per test. The surviving relationships have wide intervals, and the DL shuttle result rests on 25 players. The family-wise test protects against false discovery, but not against imprecision.
- **Selection.** Only players who reached ≥ 50 tracked NFL snaps have game traits, and better athletes earn more snaps. Weight and roster-position controls do not remove role differences *within* a position (slot vs outside WR).
- **"Max per snap" is a ceiling, not a skill.** A receiver's NFL brake reflects his route tree as well as his body. In an exploratory route-matched check (Combine curl, out, slant, dagger and corner reps vs the same NFL route types), 38 of 43 drill-metric pairs had |ρ| < 0.25, with mixed signs.
- **Clip boundaries.** Combine clips begin with the athlete already moving, so we do not use tracking-derived 10-yard splits.

## Appendix

- **Reproducible notebook:** [Measure the Athlete, Not the Drill](https://www.kaggle.com/code/guaravbansal/measure-the-athlete-not-the-drill). It rebuilds every number and figure from the raw competition files in a few minutes.
- **Methods.** Savitzky–Golay derivatives (window 7, order 2) of x/y; frames within three samples of a clip edge are dropped. NFL window: snap to snap + 5 s, regular season, non-nullified plays. Partial Spearman = Pearson correlation of rank residuals after OLS on weight rank and roster-position dummies. Family-wise p = share of 1,000 permutations whose family-maximum |ρ| ≥ observed. Bootstrap CIs use 2,000 player resamples. Drill names were harmonized across years (e.g. `LINE` → `LINE_DRILL`).
