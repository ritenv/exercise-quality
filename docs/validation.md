# Validation against published squat studies

Reproduce with:

```bash
PYTHONPATH=src python scripts/validate_against_literature.py 200
```

## Method

For each study, 200 people are drawn from ANSUR II (matching the study's sex mix). Each gets a squat style sampled under that study's protocol (depth target, arms, tempo), and one rep is generated with `generate_feasible_squat`. Metrics are measured from the generated landmarks using the study's definitions where possible: descent phase only, range of motion as maximum minus start, and centre of pressure as a fraction of foot length from the heel.

**z** = (simulated mean - published mean) / published between-subject SD. |z| < 1 means the simulated average person sits within one real person's spread of the published average.

Every row is labelled by what it can tell us:

- **input**: the value was used to sample people, so the row only checks the sampling.
- **input + balance correction**: each person's *preference* comes from this value, then balance adjusts it. It shows how far balance pushes people from their preferences.
- **fitted**: used by `scripts/calibrate_lumbopelvic.py` to set the hip end-range parameters.
- **cross-study check**: settings came from one study, compared here with a different one.
- **prediction**: not used to set anything. These are the real tests.

## Results (200 simulated people per study, seed 0)

Every rep was feasible and nobody had to squat shallower than the protocol asked (0% depth-limited).

### Ishida et al. 2025: men, thighs parallel, 2 s down and 2 s up

| Metric | Model mean (SD) | Published mean (SD) | z | Kind |
|---|---|---|---|---|
| Peak knee flexion | 122 (5.5) | 121.8 (5.7) | -0.03 | input |
| Ankle at 60 deg knee | 21.7 (2.2) | 19.3 (4.8) | +0.50 | input + balance correction |
| Ankle at 90 deg knee | 29.9 (2.8) | 27.8 (5.1) | +0.42 | input + balance correction |
| Peak ankle | 37.3 (3.8) | 33.8 (4.5) | +0.77 | input + balance correction |
| Trunk lean at 60 deg knee | 24.3 (4.6) | 22.8 (6.2) | +0.24 | input + balance correction |
| Trunk lean at 90 deg knee | 33.8 (6.4) | 30.6 (6.6) | +0.49 | input + balance correction |
| Peak trunk lean | 42.2 (7.8) | 38.5 (7.3) | +0.51 | input + balance correction |
| Mean COP, knee > 60 deg | 0.377 (0.08) | 0.363 (0.08) | +0.18 | input, measured dynamically |

### Kasahara et al. 2024: 13 men and 13 women, arms crossed, thighs parallel, 2 s descent

| Metric | Model mean (SD) | Published mean (SD) | z | Kind |
|---|---|---|---|---|
| Max COM descent, % of height | 18.4 (3.1) | 23.1 (2.5) | **-1.89** | **prediction** |
| COM forward shift start to bottom, % of foot | -4.2 (8.9) | -5.0 (9.6) | +0.09 | input |
| Trunk ROM | 37.2 (7.4) | 34.7 (11.5) | +0.22 | cross-study check |
| Hip ROM (pelvis vs thigh) | 88.9 (7.1) | 82.7 (6.1) | +1.02 | fitted |
| Knee ROM | 106 (10) | 107.2 (11.1) | -0.14 | input |
| Ankle ROM | 31.1 (4.2) | 32.8 (5.6) | -0.31 | input (from Ishida) |
| Trunk peak time vs lowest COM, ms | 0 | -11.5 (23.6) | +0.49 | **prediction** |
| Hip peak time vs lowest COM, ms | 0 | -1.1 (8.6) | +0.13 | fitted |
| Knee peak time vs lowest COM, ms | 0 | 0.4 (3.0) | -0.13 | **prediction** |
| Ankle peak time vs lowest COM, ms | 0 | -21 (26.6) | +0.79 | **prediction** |

### Zawadka et al. 2020: 40 men and 20 women, maximum depth, arms forward

| Metric | Men: model | Men: published | Women: model | Women: published | Kind |
|---|---|---|---|---|---|
| Knee ROM | 125 (16) | 124.8 (15.5) | 120 (17) | 119.0 (15.1) | input |
| Ankle ROM | 32.6 (5.0) | 30.3 (6.4) | 32.3 (4.0) | 30.8 (4.9) | cross-study check |
| Hip ROM (pelvis vs thigh) | 92.7 (7.4) | 99.6 (11.0) | 92.9 (7.9) | 98.6 (12.2) | fitted |
| Pelvis tilt ROM | 24.6 (5.0) | 19.6 (8.1) | 25.0 (4.0) | 22.6 (7.5) | fitted |
| **Lumbar flexion ROM** | 36.2 (15) | 43.7 (15.3), z -0.50 | 31.3 (17) | 30.1 (14.4), z +0.09 | **prediction** |
| **Pelvis peak, % of descent (median)** | 59 | 62 | 62 | 70 | **prediction** |
| Ankle peak, % of descent (median) | 96 | 87 | 97 | 90 | prediction |
| Knee peak, % of descent (median) | 100 | 99 | 100 | 99 | prediction |

### Mata et al. 2021: 14 men and 14 women, full depth, moment the pelvis starts tilting back

| Metric | Model mean (SD) | Published mean (SD) | z | Kind |
|---|---|---|---|---|
| Knee flexion at that moment | 80.5 (9.8) | 100.3 (14.8) | **-1.33** | fitted |
| **Ankle dorsiflexion at that moment** | 25.3 (4.2) | 25.2 (5.2) | +0.02 | **prediction** |

## What matches

- **Lumbar flexion** (men z -0.50, women z +0.09). It comes entirely from the hip end-range mechanism plus balance. Nothing about the lower back was fitted.
- **When the pelvis peaks**: 59-62% of the descent vs 62-70% measured. It emerges from when each person's hip reaches end range.
- **Joint timing at the bottom.** Trunk, knee and ankle all peak at the lowest point, matching Kasahara.
- **The ankle angle when the pelvis starts tipping back** (z +0.02).
- **Spreads, not just means.** Trunk SD 7.8 vs 7.3, knee SD 5.5 vs 5.7. Before the ankle and trunk shared the balance correction, simulated trunk lean varied about three times as much as real people's and up to 20% of people couldn't reach parallel.
- **Cross-study checks.** Trunk range of motion in Kasahara (z +0.22) and ankle range of motion in Zawadka (z +0.31 to +0.36), with settings taken from Ishida.

## Known gaps, most important first

1. **Centre-of-mass drop too small (z -1.89).** At the same knee range of motion, the model's centre of mass drops 18% of height vs 23% measured. It has been low in every version. Candidate causes: Kasahara reports knee *range of motion*, and if their standing knee was slightly bent, their true peak was deeper; Japanese students (1.67 m) have different proportions from ANSUR II soldiers; and the model's foot stays rigidly flat.
2. **Pelvis tips back too early (Mata knee z -1.33) while the hip still runs past Kasahara's range (z +1.02).** The fit can't satisfy both with one onset angle and one gain past it. Real end range is probably more gradual or more individual.
3. **Ankle and trunk means a few degrees high (z +0.2 to +0.8).** Preferences use the published outcome means and balance adds a forward correction on top. Re-centring the preferences would remove this, but only by fitting those means.
4. **Ankle peak timing.** Zawadka finds the ankle peaking at 87-90% of the descent; the model is at 96-97% (Kasahara agrees with the model). The two studies disagree with each other.

## What this does not validate

- Whole curves. Only means and SDs at a few points were available, not time series.
- Individual reps. A matching cohort mean says nothing about whether any single simulated person moves like a real one.
- Anything outside these protocols: other depths, stances, loads, tempos, or populations.
- Movement quality. Nothing here says whether a squat is good. That needs a separate, justified rubric.

## History of the trunk model

| Version | Peak trunk lean z | Trunk timing z | COM drop z | Notes |
|---|---|---|---|---|
| Rigid trunk, trunk absorbs balance | -0.86 | -3.75 | -1.57 | lean flattened at the bottom |
| Pelvis + thorax, hard hip stop | +1.23 | +0.11 | -1.77 | timing fixed; hip plateaued early |
| Soft hip end range | +1.35 (SD 22) | +0.11 | -1.71 | stiff-ankle people leant up to 80 deg |
| Shared ankle/trunk correction (current) | +0.51 (SD 7.8) | +0.49 | -1.89 | spreads realistic, nobody depth-limited |
