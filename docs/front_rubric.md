# Squat form rubric (front view)

`exercises/squat/front_rubric.py` grades bodyweight squats from a front-view MediaPipe track. It covers the two frontal-plane criteria of the Back Squat Assessment, which the side-view rubric ([rubric.md](rubric.md)) can't see. Real clips and 3D synthetic reps filmed by the virtual camera ([model3d.md](model3d.md)) go through the same code.

```python
from exercise_quality.core import model3d as M
from exercise_quality.exercises.squat import front_rubric as FR
from exercise_quality.video import mediapipe_pose as mpp

result = FR.assess(mpp.load_raw("front_squat_5"), sex="male")
result.score, result.deficits, result.reliable, result.camera_angle
result.reps[0].criteria["knees_over_feet"]      # value, threshold, passed, note
```

```bash
PYTHONPATH=src python scripts/front_rubric_report.py 200   # real clips, synthetic cohorts, injected faults (~20 s)
```

## Criteria

Wording is from Table 1 of Myer et al. 2014, *Strength Cond J* 36(6):4-27 ([PMC4262933](https://pmc.ncbi.nlm.nih.gov/articles/PMC4262933)).

| Criterion (BSA no.) | BSA wording | Measured as | Standard | Tolerance |
|---|---|---|---|---|
| hips_level (4) | "Line of hips is parallel to ground in frontal plane throughout squat." | largest tilt of the line between MediaPipe's two hip points, from the start of the descent to the end of the ascent | 0 deg | 4.9 deg |
| knees_over_feet (5) | "Lateral aspect of knee does not cross medial malleolus for either leg." | how far the knee centre moves towards the midline past the ankle centre, minus half the knee's width and half the ankle's width; worse leg, whole rep. Above 0, the outside of the knee is inside the inner ankle bone. | 0 cm | 3.1 cm |

- **Widths.**
  - Ankle: ANSUR II bimalleolar breadth.
  - Knee: ANSUR II has no knee breadth. It's estimated as femoral bicondylar (bone) breadth, 8.86 cm for men and 7.85 cm for women (dried femora of 360 Greek adults, [med.auth.gr](https://med.auth.gr/node/3488)), scaled by the person's lower thigh circumference and plus 1.5 cm of soft tissue. That's an assumption; no published skin-to-skin knee breadth norm was found. Median: men 10.4 cm, women 9.4 cm.
  - Without a matched body, the median ANSUR II body of the given sex is used.
- **Image scale.** Pixels are converted to cm with the standing leg's image height against the body's thigh + shank length.
- **Deficits** follow the BSA rule: a criterion counts once if it fails in 2 or more reps (or in the only rep). `score` is 100 × the mean share of reps passing each criterion.

**Reported, not graded:**
- knee-to-ankle separation at the bottom (KASR) and each knee's crossing value;
- hip shift and standing tilt;
- stance width;
- the camera angle.

"Knees in line with the toes" isn't graded: MediaPipe's 3D foot directions are lopsided from the front ([front_view_check.md](front_view_check.md)).

## Tolerances

Each tolerance is the 95th percentile of the signed error, the graded value minus the true value from the 3D landmarks. Only errors that make a rep look worse than it is can fail a good one. The errors come from 3D synthetic reps (both protocols, sampled stances) filmed by phones placed as people place them (`sample_front_camera`, an assumption: within 8 deg of square, 3-5 m away, lens 0.4-1.3 m up), with tracker jitter and drift. `measurement_tolerances()` recomputes them, and a test checks the stored values.

**Left-right pairs share their tracker noise.** The tracker-noise model was measured on one side-on clip, and on its own it moves each landmark independently. From the front, that made the hip line wobble far more than real clips show:

| Hip-line tilt while standing still, SD | Median | Range |
|---|---|---|
| Real front-view clips (5) | 0.81 deg | 0.46-1.17 |
| Synthetic, independent noise | 1.97 | 1.04-2.87 (10-90%) |
| Synthetic, pairs share 80% (`camera.PAIR_SHARE`) | 0.87 | 0.46-1.30 |

MediaPipe seems to move a pair such as the two hips together, as one pelvis. The 80% share is fitted to that median. With it, the hips-level error has median +2.2 deg and 95th percentile +4.9. The largest tilt over a whole rep picks up the peaks of the slow drift, so a single-frame value would be less noisy but would miss a brief drop.

**Camera angle.** MediaPipe's 3D hips read every front-view clip 11-16 deg (mean 13) further round than image symmetry says ([front_view_check.md](front_view_check.md)). For real clips, `camera_angle` removes that 13 deg (`MEDIAPIPE_FRONT_YAW_BIAS`); synthetic clips have exact 3D landmarks. Beyond 10 deg from square the result is marked unreliable:
- a knee ahead of the ankle shows up sideways in the image, one knee towards the midline and the other away;
- the hip line tilts by about 0.14 deg per degree off square.

## How it behaves

**Synthetic cohorts** (200 reps each, true values against the filmed view):

| | Ishida parallel: true / filmed / agree | Zawadka max depth: true / filmed / agree |
|---|---|---|
| hips_level | 1.00 / 0.96 / 0.96 | 1.00 / 0.95 / 0.95 |
| knees_over_feet | 1.00 / 0.97 / 0.97 | 0.98 / 0.97 / 0.98 |

The model's knees track over the toes, so the outside of the knee stays a median 4 cm short of the inner ankle bone. False alarms run at 3-5%, as the 95th-percentile tolerances intend.

**Injected faults** (100 Ishida-style reps each):

| Fault | Caught |
|---|---|
| Both knees cave in 15 / 25 / 35 deg (`Stance.knee_track`) | 16% / 56% / 90% |
| Right knee caves in 25 / 35 deg | 45% / 76% |
| Right hip drops 6 / 10 deg at the bottom (image edit, `drop_hip`) | 92% / 100% |

The BSA's knee standard is gross: it fails only when the outside of the knee passes the inner ankle bone. A knee turning 15 deg in from the toe line usually doesn't get that far, so it passes. That's the published standard working as written, not a measurement problem. A more sensitive check would need a reference for "too far in" that the BSA doesn't give. The 3D model has no pelvic drop of its own, so the hip fault is an edit of the image points.

**Real clips** (median body of the apparent sex):

| Clip | Camera (from square) | Reliable | Hips level, worst rep | Knee crossing, worse leg | Deficits |
|---|---|---|---|---|---|
| front_squat_1 | +2 deg | yes | 2.4 deg | -2.1 cm | none |
| front_squat_2 | 0 deg | yes | 3.7 deg | -2.5 cm | none |
| front_squat_3 | -19 deg | **no** | 6.3 deg | +0.2 cm | hips (from the view, see below) |
| front_squat_4 | -13 deg | **no** | 4.0 deg | -2.8 cm | none |
| front_squat_5 (the repository owner) | +2 deg | yes | 2.7 deg | -0.5 cm (median: right -4.6, left -2.1) | none |

- **The three square clips pass both checks.**
- **front_squat_5:** the left knee comes closest, a median 2 cm short of the inner ankle bone. The toes are out of frame, which doesn't matter here: the check uses knees and ankles only.
- **front_squat_3 is filmed 19 deg off square.** Its hip-line tilt of 5-6 deg includes about 3 deg from the view. It's marked unreliable, and the check can't tell whether the rest is real.

## Limits

- **Tracker noise:** only partly measured from the front. The pair share is fitted to standing wobble in five clips, two of them probably the same person.
- **Knee width** is estimated from bone breadth plus an assumed soft-tissue allowance.
- **The camera-angle correction** comes from five clips. It assumes MediaPipe's front-view yaw bias is the same for everyone.
- **No reference measurement:** MediaPipe's hip and knee points aren't checked against motion capture.
- **Front views only.** Back views (left and right swapped) aren't handled, and the person must face the camera.
- **Not validated against expert raters,** like the side-view rubric.
