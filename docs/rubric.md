# Squat form rubric (side view)

`exercises/squat/rubric.py` grades bodyweight squats from a side-view pose track: real video through MediaPipe, or synthetic reps through the same tracker view. It takes the person's body proportions into account.

```python
from exercise_quality.exercises.squat import rubric
from exercise_quality.video import mediapipe_pose as mpp

result = rubric.assess(mpp.observed_track("squat_1"), sex="female")
result.score, result.deficits, result.reliable, result.warnings
result.reps[0].criteria["trunk"]       # value, threshold, passed, note
```

```bash
PYTHONPATH=src python scripts/rubric_report.py 200   # real clips, synthetic cohorts, injected faults (~15 s)
```

## Where the criteria come from

The criteria are taken from the **Back Squat Assessment** (BSA) of Myer, Kushner, Brent et al., "The back squat: a proposed assessment of functional deficits and technical factors that limit performance", *Strength Cond J* 2014;36(6):4-27 ([PMC4262933](https://pmc.ncbi.nlm.nih.gov/articles/PMC4262933)). It has ten criteria. A criterion is a deficit when it's missed "during 2 or more repetitions", and the total counts deficits. The authors present it as a proposed screen; the paper reports no reliability or validity data. It's a coaching-technique standard, not an injury predictor, and nothing here makes medical claims.

Five criteria can be measured from the side and are graded:

| Criterion (BSA no.) | BSA wording | Measured as | Standard | Tolerance |
|---|---|---|---|---|
| depth (9) | "tops of thighs are at least parallel to the ground" | hip-to-knee line at the deepest knee bend | -90 deg (parallel) or deeper | 3.1 deg |
| heels (7) | "Entire foot remains in contact with the ground" | heel rise relative to the toe point | 0 | 0.094 of foot length (about 2.5 cm) |
| trunk (3) | "Trunk is parallel to tibia" | trunk lean minus shin lean at the bottom | 0, or the **build allowance** if larger | 3.8 deg |
| ascent (10) | "Shoulders and hips rise at the same, constant speed" | extra trunk lean after the rise starts (hips leading) | 0 | 2.6 deg |
| descent (8) | "controlled, constant speed throughout descent" | largest upward hip movement while going down | 0 | 0.039 of standing hip height |

Notes on the measures:

- **Depth.** The thigh line is the joint-to-joint line. On a tapering thigh the top surface reaches horizontal a few degrees later, so this standard is slightly generous.
- **Descent.** Only "uninterrupted" is graded. A smooth human movement speeds up and slows down, so "constant speed" isn't graded literally.
- **Ascent.** It's measured from the start of the rise, so a pause at the bottom doesn't count.

Not graded, with reasons (`rubric.NOT_GRADED`):

- **Head (1).** The ear landmark is rough, and coaching advice on the head conflicts. The change in neck angle is reported.
- **Thoracic position (2) and lower-back curve (part of 3).** These need spine points that a 2D tracker doesn't give.
- **Hip level (4) and frontal knee (5).** These need a front or back view.
- **Tibial progression (6).** Knees passing the toes is reported, not graded:
  - restricting it shifts load to the hips and lower back (Fry, Smith and Schilling 2003, *J Strength Cond Res* 17(4):629-633, [PubMed](https://pubmed.ncbi.nlm.nih.gov/14636100/): hip torque about 28 N·m unrestricted vs 303 N·m restricted);
  - the tracker's toe point isn't the toe tip.
- **Tempo ratio (part of 10).** Table 1's "Ascent: Descent timing ratio is at least 2:1" is ambiguous as quoted. Durations are reported.

## Body proportions: the trunk allowance

Whether the trunk can stay parallel to the shin depends on the body. The centre of mass has to stay over the foot. With long thighs and a short trunk the hips sit further behind the feet at the same depth, and the trunk must lean further forward to balance them. So the rubric doesn't hold everyone to the same line:

1. It matches the person to the closest ANSUR II body by tracked thigh and trunk proportions. Synthetic reps use their real body.
2. At the rep's own depth (thigh angle) and arm position, it finds the trunk-minus-shin angle that puts the centre of mass over mid-foot. Mid-foot here is 36.3% of foot length, Ishida 2025's measured bottom COP. The shins are set to lean **36.8 deg**, the peak ankle angle Ishida's men reached at parallel depth when cued to shift their weight forward. That's an achievable lean people actually showed, not a maximal range.
3. If that balance angle is above 0, it becomes the person's allowance.

Across ANSUR II at parallel depth with arms crossed:

| | Trunk minus shin needed (median, 5-95%) | People with an allowance | Longest-thigh 5% | Shortest-thigh 5% | Correlation with thigh/trunk ratio |
|---|---|---|---|---|---|
| Women | -5.0 (-13.5 to 7.5) | 23% | +12.8 | -14.2 | 0.80 |
| Men | -9.6 (-17.0 to 3.9) | 9% | +5.7 | -17.6 | 0.79 |

So most people *can* keep the trunk parallel to the shin at parallel depth with a typical cued ankle. About a quarter of women and a tenth of men can't, and they aren't marked down for it.

Each rep also reports **trunk needed at own shin**: the balance angle at the shin lean the person actually used.

- If the measured trunk-minus-shin is close to it, the lean is what their shin angle forces, and the useful cue is about the knees and ankles.
- If it's well above, their weight is ahead of mid-foot.

This separates two kinds of BSA deficit, mobility or strategy versus balance, from video alone.

## Tolerances: only tracker error

A criterion fails only when the measurement is past its standard by more than the tracker could plausibly be wrong. Each tolerance is the 95th percentile of |error| when synthetic reps (both protocols, varied tempos) pass through the jitter and drift of the tracker-noise model, with its assumed spread between videos (`rubric.measurement_tolerances()`, mean of two seeds of 300 reps; a test recomputes them).

The noise model's knees-out projection isn't counted as error. When the knees travel out over the toes, a side view genuinely shows the shin about 5 deg more upright and the thigh about 3 deg deeper than the same squat in a single plane. The BSA judges what the side view shows, so the rubric does too.

## Score and reliability

- `deficits` follows the BSA rule: a criterion counts once if it fails in 2 or more reps (or in the only rep). `deficit_total` runs 0-5, and lower is better.
- `score` is 100 x the mean share of reps passing each criterion, with equal weights. That's a choice, not evidence; the BSA also counts criteria equally.
- `reliable` is False and `warnings` explains why in two cases:
  - the camera is more than 26 deg from side-on (from MediaPipe's 3D hip axis);
  - the tracked standing pose puts the centre of mass outside 0.16-0.73 of foot length. That's Fujiwara 2010's 31-58% quiet-standing range, widened by the tracker COM error. Such a pose can't be stood in, so the angles are distorted.

## How it behaves

**Injected faults**, 100 Ishida-style synthetic reps each, graded through the tracker-noise model:

| Fault | Caught |
|---|---|
| Too shallow (85 deg knee) | 100% |
| Heels lift 25% / 10% of foot length | 100% / 88% |
| Hips rise first, trunk +10 / +5 deg | 99% / 88% |
| Hitch on the way down, 8% / 4% of hip height | 100% / 75% |
| Forward lean (strong trunk preference, little ankle) | 100% |

The fault-specific tests (`tests/test_rubric.py`) also check that each fault fails only its own criterion.

**Unmodified synthetic reps** (200 per protocol): how often each criterion passes, graded from exact positions and through the tracker view, and how often the two agree.

| | Ishida parallel: exact / tracked / agree | Zawadka max depth: exact / tracked / agree |
|---|---|---|
| depth | 0.40 / 0.56 / 0.82 | 0.56 / 0.60 / 0.95 |
| heels | 1.00 / 0.95 / 0.95 | 1.00 / 0.94 / 0.94 |
| trunk | 0.93 / 0.72 / 0.79 | 0.90 / 0.72 / 0.81 |
| ascent | 0.96 / 0.90 / 0.92 | 0.71 / 0.67 / 0.90 |
| descent | 1.00 / 0.95 / 0.95 | 1.00 / 0.97 / 0.97 |

- **Depth.** About half of people asked for "thighs parallel" don't reach a strict parallel. That matches Ishida's knee spread around a mean that sits right at parallel.
- **Trunk, tracked view.** The knees-out projection fails more people (72% pass vs 93% exact).
- **Agreement of 0.8-0.97 is the label noise** a model trained on tracked-view grades would face. Labels for training should be computed on the same tracked view the model sees.
- **Ascent in Zawadka reps.** The generator makes 29% of them lean further forward while rising out of the 3 s hold. That comes from how the generator handles holds, not from data. Whether real people do it is unknown.

**Real clips:**

| Clip | Reliable | Score | Deficits |
|---|---|---|---|
| squat_1 (side-on) | yes | 60 | depth (thigh line 0.3-10 deg short of parallel in all 6 reps), trunk (12-20 deg ahead of the shin, 36 in the "sit back" rep, allowance 0) |
| squat_2 (48 deg oblique) | **no**: camera angle, and an impossible standing balance (0.77) | 55 | not to be trusted |

squat_1's trunk lean in reps 1-4 is within 4 deg of what balance needs at her own shin lean (28 deg, below the 36.8 deg cued value). By this rubric's logic her extra lean comes from shins that stay back, not from her weight sitting forward.

## Limits

- The BSA is a proposed, unvalidated screen. Its standards (parallel depth, trunk parallel to the shin) suit a general back-squat pattern and aren't the only valid squat. A front squat, a deep squat for mobility or a box squat would need other standards.
- Only side-view criteria. Knee valgus, hip shift and other frontal-plane faults are invisible here.
- The build allowance rests on the balance model, de Leva mass fractions and Ishida's numbers from 16 young men. It's applied to everyone.
- Tolerances come from a noise model calibrated on one side-on clip.
- Not validated against expert raters. Comparing these grades with coaches scoring the same videos is the next real test.
