# Validation against published data

Reproduce with:

```bash
PYTHONPATH=src python scripts/validate_against_literature.py 200
```

## Method

For each study, 200 people are drawn from ANSUR II (matching the study's sex mix). Each gets a squat style sampled under that study's protocol (depth target, arms, tempo), and one rep is generated with `generate_feasible_squat`. Metrics are measured from the generated landmarks using the study's definitions where possible: descent phase only, range of motion as maximum minus start, and centre of pressure as a fraction of foot length from the heel. Joint peak timing is measured over the descent itself, not any hold at the bottom.

**z** = (simulated mean - published mean) / published between-subject SD. |z| < 1 means the simulated average person sits within one real person's spread of the published average.

Every row is labelled by what it can tell us:

- **input**: the value was used to sample people, so the row only checks the sampling.
- **input + balance correction**: each person's *preference* comes from this value, then balance adjusts it. It shows how far balance pushes people from their preferences.
- **fitted**: used by `scripts/calibrate_lumbopelvic.py` to set the hip end-range parameters.
- **cross-study check**: settings came from one study, compared here with a different one.
- **prediction**: not used to set anything. These are the real tests.

## Results (200 simulated people per study, seed 0)

Every rep was feasible and nobody had to squat shallower than the protocol asked.

### Ishida et al. 2025: men, thighs parallel, 2 s down and 2 s up

| Metric | Model mean (SD) | Published mean (SD) | z | Kind |
|---|---|---|---|---|
| Peak knee flexion | 122 (5.7) | 121.8 (5.7) | +0.04 | input |
| Ankle at 60 deg knee | 22.3 (1.8) | 19.3 (4.8) | +0.63 | input + balance correction |
| Ankle at 90 deg knee | 30.5 (2.8) | 27.8 (5.1) | +0.52 | input + balance correction |
| Peak ankle | 37.2 (4.0) | 33.8 (4.5) | +0.75 | input + balance correction |
| Trunk lean at 60 deg knee | 25.0 (4.3) | 22.8 (6.2) | +0.36 | input + balance correction |
| Trunk lean at 90 deg knee | 34.4 (6.0) | 30.6 (6.6) | +0.57 | input + balance correction |
| Peak trunk lean | 42.5 (8.8) | 38.5 (7.3) | +0.55 | input + balance correction |
| Mean COP, knee > 60 deg | 0.373 (0.071) | 0.363 (0.08) | +0.12 | input, measured dynamically |

### Kasahara et al. 2024: 13 men and 13 women, arms crossed, "thighs parallel", 2 s descent

Depth is taken from Ishida (same lab, same instruction); see "Kasahara's depth" below.

| Metric | Model mean (SD) | Published mean (SD) | z | Kind |
|---|---|---|---|---|
| **Max COM descent, % of height** | 22.8 (1.9) | 23.1 (2.5) | -0.13 | **prediction** |
| COM forward shift start to bottom, % of foot | -5.2 (9.9) | -5.0 (9.6) | -0.02 | input |
| Trunk ROM | 39.2 (7.4) | 34.7 (11.5) | +0.39 | cross-study check |
| Hip ROM | 102 (6.8) | 82.7 (6.1) | +3.19 | cross-study check (depth from Ishida) |
| Knee ROM | 122 (5.7) | 107.2 (11.1) | +1.33 | cross-study check (depth from Ishida) |
| Ankle ROM | 34.8 (4.1) | 32.8 (5.6) | +0.36 | input (from Ishida) |
| **Trunk peak time vs lowest COM, ms** | -9.3 (35) | -11.5 (23.6) | +0.10 | **prediction** |
| Hip peak time vs lowest COM, ms | 0 | -1.1 (8.6) | +0.13 | fitted |
| **Knee peak time vs lowest COM, ms** | 0 | 0.4 (3.0) | -0.13 | **prediction** |
| **Ankle peak time vs lowest COM, ms** | -9.9 (36) | -21 (26.6) | +0.42 | **prediction** |

### Zawadka et al. 2020: 40 men and 20 women, maximum depth, arms forward

| Metric | Men: model | Men: published | Women: model | Women: published | Kind |
|---|---|---|---|---|---|
| Knee ROM | 125 (15) | 124.8 (15.5) | 118 (14) | 119.0 (15.1) | input |
| Ankle ROM | 32.2 (4.5) | 30.3 (6.4) | 32.6 (4.8) | 30.8 (4.9) | cross-study check |
| Hip ROM (pelvis vs thigh) | 103 (8.4) | 99.6 (11.0) | 99.9 (8.8) | 98.6 (12.2) | fitted |
| Pelvis tilt ROM | 20.1 (5.7) | 19.6 (8.1) | 21.0 (5.5) | 22.6 (7.5) | fitted |
| **Lumbar flexion ROM** | 25.7 (12) | 43.7 (15.3), z **-1.18** | 21.4 (12) | 30.1 (14.4), z -0.60 | **prediction** |
| **Pelvis peak, % of descent** | 69 | 62 (median) | 73 | 70 (median) | **prediction** |
| Ankle peak, % of descent | 96 | 87 (median) | 96 | 90 (median) | prediction |
| Knee peak, % of descent | 100 | 99 (median) | 100 | 99 (median) | prediction |

### Mata et al. 2021: 14 men and 14 women, full depth, moment the pelvis starts tilting back

| Metric | Model mean (SD) | Published mean (SD) | z | Kind |
|---|---|---|---|---|
| Knee flexion at that moment | 96.3 (7.7) | 100.3 (14.8) | -0.27 | fitted |
| **Ankle dorsiflexion at that moment** | 29.6 (3.8) | 25.2 (5.2) | +0.84 | **prediction** |

### Standing and sway

| Check | Model | Published | Source |
|---|---|---|---|
| Standing COP position | 41-42% of foot (SD 6-7); 92-94% of people inside the published range | 31-58% of foot length | Fujiwara et al. 2010 |
| Quiet-standing COP mean velocity | 6.3 mm/s | 6.3 mm/s (log10 0.798 ± 0.097) | Yamamoto et al. 2015 |
| Quiet-standing COP median frequency | 0.34 Hz | 0.33 ± 0.07 Hz | Yamamoto et al. 2015 |
| Peak vertical force across depths (2 s tempo) | 1.08 / 1.10 / 1.12 BW at 99 / 115 / 130 deg knee | "similar across depths" | Hamed et al. 2025 (abstract) |

The sway amplitude and frequency were set to match Yamamoto, so those two rows are fitted. They are tested in `tests/test_sway.py`.

## Kasahara's depth

Kasahara's instruction was "thighs parallel to the floor", the same as Ishida's (same lab; Kasahara cites that protocol). Their published knee figure, 107 deg, is a range of motion from the starting angle. Used as a peak, it left the model's thighs about 18 deg *above* horizontal and the COM drop at 18% of height against their measured 23% (z -1.89, the largest gap in earlier versions). With Ishida's measured peak (121.8 deg) the thighs are within about 5 deg of parallel and the COM drop matches (z +0.03). No single depth fits all of Kasahara's figures in this model: their knee and hip ROMs fit a shallower squat, while their instruction and COM drop fit a parallel one. A slightly bent standing knee in their data would explain the difference. Their knee and hip ROMs are therefore shown as cross-checks, not used for fitting.

## What matches

- **COM descent** (z -0.13), **trunk, knee and ankle peak timing** (|z| <= 0.42) and **pelvis peak timing** (69-73% of descent vs 62-70%). None of these were fitted.
- **The ankle angle when the pelvis starts tipping back** (z +0.84), predicted from the hip end-range mechanism.
- **Spreads as well as means.** Trunk SD 8.8 vs 7.3 and knee SD 5.7 vs 5.7. Nobody needs their depth limited.
- **Standing and sway** match quiet-standing force-plate data (fitted), and the ground force behaves as Hamed describes across depths.
- **Smoothness.** No spikes in the ground force and no joint hitting its end range like a wall (`tests/test_realism.py`).

## Known gaps, most important first

1. **Lumbar flexion reads low** (men z -1.18, women -0.60), and the model has no sex difference to match Zawadka's (men 43.7 vs women 30.1 deg). This is not about where the spine bends. Lumbar flexion as measured is the rib cage relative to the pelvis. The rib-cage lean comes from balance and Ishida's measured lean, and the pelvis from the fitted hip end range, so moving or adding spine joints cannot change it. Splitting the trunk at the rib cage (navel and xiphoid joints, de Leva's three sub-segments) was tested for exactly this and left both numbers unchanged. The gap is in **trunk lean past parallel**: in Zawadka, hip plus lumbar range (143 deg in men) minus thigh rotation (about 95 deg) implies the rib cage turns about 48 deg in men and 40 deg in women at maximum depth. The model's turns about 36 deg for both, and no more in its deepest squats than its shallowest (35.7 vs 36.3 deg in men), because the preferred lean is held at Ishida's value beyond Ishida's depth. Closing this needs a study that reports trunk lean at several depths, not a fit to Zawadka's lumbar row.
2. **Kasahara's knee and hip ROMs** (+1.33, +3.19) can't be matched together with their COM drop at a single depth; see above.
3. **Ankle and trunk run a little high** (z +0.4 to +0.75). Preferences use the published outcome means and balance adds a forward correction on top. The correction grew slightly when the trunk was split at the rib cage: the solver now steers the rib cage, and the abdomen below it leans less, so for the same measured lean the body's mass sits a little further back and balance asks for a little more ankle and lean.
4. **Ankle peak timing.** Zawadka has the ankle peaking at 87-90% of the descent, Kasahara about 21 ms before the knee; the model is at 96% and about 10 ms before. Roughly between the two studies.

## What this does not validate

- Whole curves. Only means and SDs at a few points were available, not time series.
- Individual reps. A matching cohort mean says nothing about whether any single simulated person moves like a real one.
- Anything outside these protocols: other depths, stances, loads, tempos, or populations.
- Movement quality. Nothing here says whether a squat is good. That needs a separate, justified rubric.

## History

| Version | Peak trunk z | Trunk timing z | COM drop z | Lumbar z (M / F) | Notes |
|---|---|---|---|---|---|
| Rigid trunk, trunk absorbs balance | -0.86 | -3.75 | -1.57 | n/a | lean flattened at the bottom |
| Pelvis + thorax, hard hip stop | +1.23 | +0.11 | -1.77 | -0.76 / +0.61 | timing fixed; hip plateaued early |
| Soft hip end range | +1.35 (SD 22) | +0.11 | -1.71 | -0.65 / +0.69 | stiff-ankle people leant up to 80 deg |
| Shared ankle/trunk correction | +0.51 (SD 7.8) | +0.49 | -1.89 | -0.50 / +0.09 | spreads realistic, nobody depth limited |
| Smoothing, Kasahara depth, joint weight sampling, sway, refit | +0.35 (SD 7.9) | -0.06 | +0.03 | -1.18 / -0.62 | no force spikes; COM drop solved; lumbar low (see gap 1) |
| Trunk split at the rib cage, toe joints, left and right limbs (current) | +0.55 (SD 8.8) | +0.10 | -0.13 | -1.18 / -0.60 | lumbar unchanged, which located gap 1 in trunk lean; refit kept share 0.9, onset 105, gain 0.1 |
