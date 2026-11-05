# Validation against published squat studies

Reproduce with:

```bash
PYTHONPATH=src python scripts/validate_against_literature.py 200
```

## Method

For each study, 200 people are drawn from ANSUR II (matching the study's sex mix), each gets a squat style sampled under that study's protocol (depth target, arms, tempo), and one rep is generated. Metrics are measured from the generated landmarks using the study's definitions where possible: descent phase only, range of motion as maximum minus start, and centre of pressure as a fraction of foot length from the heel.

**z** = (simulated mean - published mean) / published between-subject SD. |z| < 1 means the simulated average person sits within one real person's spread of the published average.

Some inputs come from the same studies (see [model.md](model.md)). Those rows are marked **input** and only check that the sampling works. Rows marked **prediction** come out of the physics (balance, body proportions, mass distribution) and were not fitted.

## Results (200 simulated people per study, seed 0)

### Ishida et al. 2025: men, thighs parallel, 2 s down and 2 s up

| Metric | Model mean (SD) | Published mean (SD) | z | Kind |
|---|---|---|---|---|
| Peak knee flexion | 122 (5.8) | 121.8 (5.7) | +0.00 | input |
| Ankle dorsiflexion at 60 deg knee | 22.4 (2.7) | 19.3 (4.8) | +0.64 | input (shape fitted) |
| Ankle dorsiflexion at 90 deg knee | 30.5 (3.4) | 27.8 (5.1) | +0.52 | input (shape fitted) |
| Peak ankle dorsiflexion | 36.7 (4.4) | 33.8 (4.5) | +0.65 | input |
| Trunk lean at 60 deg knee | 22.2 (9.7) | 22.8 (6.2) | -0.10 | **prediction** |
| Trunk lean at 90 deg knee | 30.1 (13) | 30.6 (6.6) | -0.08 | **prediction** |
| Peak trunk lean | 32.2 (13) | 38.5 (7.3) | -0.86 | **prediction** |
| Mean COP, knee > 60 deg | 0.38 (0.08) | 0.363 (0.08) | +0.21 | input, but measured dynamically |

The ankle sits about 3 deg high everywhere because the model adds the standing lean (about 2-3 deg) on top of the excursion. Ishida's zero reference for the ankle is not stated, so this offset can't be resolved.

### Kasahara et al. 2024: 13 men and 13 women, arms crossed, thighs parallel, 2 s descent

| Metric | Model mean (SD) | Published mean (SD) | z | Kind |
|---|---|---|---|---|
| Max COM descent, % of height | 19.2 (4.0) | 23.1 (2.5) | **-1.57** | **prediction** |
| COM forward shift start to bottom, % of foot | -6.2 (9.0) | -5.0 (9.6) | -0.12 | input |
| Trunk ROM | 32.5 (13) | 34.7 (11.5) | -0.19 | **prediction** |
| Hip ROM | 109 (18) | 82.7 (6.1) | n/a | different definition (see below) |
| Knee ROM | 107 (12) | 107.2 (11.1) | -0.05 | input |
| Ankle ROM | 29.3 (4.9) | 32.8 (5.6) | -0.62 | input (from Ishida) |
| Trunk peak time vs lowest COM, ms | -100 (190) | -11.5 (23.6) | **-3.75** | **prediction** |
| Hip peak time vs lowest COM, ms | -1.4 (20) | -1.1 (8.6) | -0.03 | prediction |
| Knee peak time vs lowest COM, ms | 0 (0) | 0.4 (3.0) | -0.13 | prediction |
| Ankle peak time vs lowest COM, ms | 0 (0) | -21 (26.6) | +0.79 | prediction |

### Zawadka et al. 2020: 40 men and 20 women, maximum depth, arms forward

| Metric | Men: model | Men: published | Women: model | Women: published |
|---|---|---|---|---|
| Knee ROM | 122 (15) | 124.8 (15.5) | 119 (14) | 119.0 (15.1) |
| Ankle ROM | 33.6 (5.8) | 30.3 (6.4) | 32.9 (4.8) | 30.8 (4.9) |
| Ankle peak, % of descent (median) | 100 | 86.8 | 100 | 90.2 |
| Model hip vs study hip only (lower bound) | 108 (14) | 99.6 (11.0) | 111 (15) | 98.6 (12.2) |
| Model hip vs study hip + lumbar (upper bound) | 108 (14) | 143 (18.9) | 111 (15) | 129 (18.9) |

Knee range of motion is an input here, so it only checks the sampling. Ankle range of motion is a cross-study check: the ankle settings come from Ishida, and Zawadka is an independent sample. The model hip angle (rigid trunk vs thigh) falls between the study's hip-only and hip-plus-lumbar values, as it should.

## What matches

- **Trunk lean through most of the descent.** At 60 and 90 deg of knee flexion the predicted trunk lean is within 0.1 SD of measured. Nothing was fitted to it. It comes from body proportions, de Leva mass distribution and keeping the centre of mass over the foot.
- **Trunk range of motion** in a different study and population (Kasahara, mixed sex): z = -0.19.
- **Dynamic centre of pressure.** Computed from accelerations, not just the static target, and it lands within 0.2 SD of Ishida's measurement.
- **Ankle range of motion across studies.** It falls between Kasahara (z = -0.62) and Zawadka (z = +0.5), even though both were independent of the Ishida-based settings.
- **Joint timing at the bottom.** Hip, knee and ankle peak at the lowest point, as Kasahara measured.

## Known gaps, most important first

1. **Peak trunk lean and its timing (z = -0.86, timing z = -3.75).** In the model, trunk lean flattens once the thigh passes horizontal, because the hip then stops moving backwards and balance no longer demands more lean. Real people keep leaning until the bottom. **Likely cause: the rigid trunk.** Real squats add 20-45 deg of pelvic tilt and lumbar flexion near the bottom (Zawadka), and a rounding spine makes the thorax lean further than the straight hip-to-C7 line.
2. **Centre-of-mass drop too small (z = -1.57).** For the same knee flexion, the model's centre of mass drops 19% of height vs 23% measured. The rigid trunk is again the likely cause: spinal flexion lowers the upper body. Kasahara's subjects (Japanese students, 1.67 m) also have different proportions from ANSUR II soldiers.
3. **Trunk lean varies about twice as much as in real people** (SD 10-13 vs 6-7 deg). Variation in ankle use drives it, and body proportions alone give about 5 deg. Real people must offset ankle differences in ways the model lacks: a correlation between ankle and weight position (adding even a strong 0.9 correlation only brings the SD to about 8 deg), spinal motion, or simply measurement noise in the published ankle SDs.
4. **Ankle peak timing conflicts between studies.** Kasahara: 21 ms before the knee. Zawadka: at 87-90% of the descent. The model peaks at the bottom, which agrees with Kasahara and not Zawadka.

## What this does not validate

- Whole curves. Only means and SDs at a few points were available, not time series.
- Individual reps. A matching cohort mean says nothing about whether any single simulated person moves like a real one.
- Anything outside the protocols above: other depths, stances, loads, tempos, or populations.
- Movement quality. Nothing here says whether a squat is good. That needs a separate, justified rubric.

## Most useful next step

Split the trunk into a pelvis segment and a thorax segment, using de Leva's lower, middle and upper trunk data, with pelvic tilt and lumbar flexion timing taken from Zawadka (pelvis peaks at 62-70% of the descent). Gaps 1 and 2, and probably part of gap 3, should then shrink. If they don't, the cause is somewhere else and needs looking for.
