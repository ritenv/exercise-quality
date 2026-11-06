# Squat form rubric (side view)

`exercises/squat/rubric.py` grades bodyweight squats from a side-view pose track: real video through MediaPipe, or synthetic reps through the same tracker view. It takes the person's body proportions into account. It grades the five side-view criteria of a published screen plus three further checks: balance, lockout and consistency.

```python
from exercise_quality.exercises.squat import rubric
from exercise_quality.video import mediapipe_pose as mpp

result = rubric.assess(mpp.observed_track("squat_1"), sex="female")
result.score, result.bsa_score, result.deficits, result.reliable, result.warnings
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
| depth (9) | "tops of thighs are at least parallel to the ground" | hip-to-knee line at the deepest knee bend | -90 deg (parallel) or deeper | 6.6 deg (mostly the phone's height, below) |
| heels (7) | "Entire foot remains in contact with the ground" | heel rise relative to the toe point, against a straight baseline from the start to the end of the rep | 0 | 0.088 of foot length (about 2.3 cm) |
| trunk (3) | "Trunk is parallel to tibia" | trunk lean minus shin lean at the bottom | 0, or the **build allowance** if larger | 4.6 deg |
| ascent (10) | "Shoulders and hips rise at the same, constant speed" | trunk lean on the way up minus on the way down, at the same hip height (lower 60% of the rep) | 0 | 4.3 deg |
| descent (8) | "controlled, constant speed throughout descent" | largest upward hip movement while going down | 0 | 0.042 of standing hip height |

Notes on the measures:

- **Depth.** The thigh line is the joint-to-joint line. On a tapering thigh the top surface reaches horizontal a few degrees later, so this standard is slightly generous.
- **Descent.** Only "uninterrupted" is graded. A smooth human movement speeds up and slows down, so "constant speed" isn't graded literally.
- **Ascent.** Hips leading tips the trunk further forward on the way up than it was at the same height on the way down. A symmetric rep scores 0, even when the trunk leans most a little above the bottom on both passes, which is common. An earlier version measured extra lean after the bottom, which flagged symmetric deep squats.
- **Heels.** The straight baseline keeps slow heel-toe drift, such as the camera rolling as it pans, from counting as a lift.

## Further side-view checks

Three more checks are graded the same way, but they're not in the BSA and have no published pass mark. Each standard says where it comes from.

| Criterion | Measured as | Standard | Where the standard comes from | Tolerance |
|---|---|---|---|---|
| balance | centre of mass over the foot at the bottom (0 heel, 1 toe), from the tracked pose and the matched body | 0.20-0.52 | Ishida 2025's bottom centre of pressure, 0.363 ± 2 between-person SD | 0.094 of foot length |
| lockout | most extended knee at the top after the rep | 5 deg of flexion or less | Convention. Standing knees in healthy young adults read about 1-2 deg on radiographs. | 0.28 deg |
| consistency | each rep's depth (thigh angle) and trunk lean at the bottom, against the median of the other reps; the larger difference | 10 deg | Convention: no study reports normal rep-to-rep spread within a person | 4.5 deg |

- **Balance.** Below the range, the weight sits on the heels; above it, on the toes. The estimate is quasi-static, giving where the mass is rather than where the floor pushes. That's close at the slow bottom of a rep. When the knees travel out over the toes, the thigh looks shorter from the side. So the thigh and shank take their seen length in the estimate (`observed_balance.projected_body`). Rebuilding the pose with full-length segments would put the hip, and the centre of mass, about 0.1 foot lengths too far back.
- **Lockout.** The tracker's hip angle at the top is reported, not graded. It depends on pelvic tilt and standing posture, and there's no norm for it. Tracker noise makes the most extended frame look slightly straighter (about 2 deg), so this check errs lenient.
- **Consistency.** It needs at least three reps; with fewer, it isn't graded.

Not graded, with reasons (`rubric.NOT_GRADED`):

- **Head (1).** The ear landmark is rough, and coaching advice on the head conflicts. The change in neck angle is reported.
- **Thoracic position (2) and lower-back curve (part of 3).** These need spine points that a 2D tracker doesn't give.
- **Hip level (4) and frontal knee (5).** These need a front view: see the front-view rubric ([front_rubric.md](front_rubric.md)).
- **Tibial progression (6).** Knees passing the toes is reported, not graded:
  - restricting it shifts load to the hips and lower back (Fry, Smith and Schilling 2003, *J Strength Cond Res* 17(4):629-633, [PubMed](https://pubmed.ncbi.nlm.nih.gov/14636100/): hip torque about 28 N·m unrestricted vs 303 N·m restricted);
  - the tracker's toe point isn't the toe tip.
- **Bottom control (bouncing out of the bottom).** There's no published pass mark. At 30 fps, tracker noise also roughly doubles the measured turnaround acceleration: in synthetic reps the median is 0.23 g exact vs 0.42 g tracked, with a correlation of 0.7-0.8. The peak upward hip acceleration at the turnaround is reported in g. squat_1 reads 0.39 g, inside the tracked synthetic spread.
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

## Tolerances: only tracker and camera error

A criterion fails only when the measurement is past its standard by more than the tracker and the phone could plausibly be wrong. Each tolerance is the 95th percentile of |error| over 3D synthetic reps (`rubric.measurement_tolerances()`, mean of two seeds of 300 reps; a test recomputes them):

- **The reps:** both protocols, varied tempos, and a sampled stance with the knees travelling out over turned-out toes ([model3d.md](model3d.md)).
- **Filmed** by a virtual phone placed as people place one (`camera.sample_side_camera`, an assumption): within 10 deg of side-on, 3-5 m away, lens 0.4-1.3 m above the floor.
- **Read by the real-video pipeline,** with MediaPipe-like jitter and drift and their assumed spread between videos.
- **Compared with a perfect side view** of the same rep. The BSA judges what a side view shows, and a side view of knees travelling out genuinely shows a shorter thigh and a more upright shin. Balance is compared with the true centre of mass instead.

Some criteria are handled differently:
- **Lockout:** the tolerance is the 95th percentile of the signed error. Noise mostly makes the knee look straighter, and only errors towards "bent" can fail a locked knee.
- **Consistency:** two noise draws with the same phone and noise settings stand in for two reps of one video.

**The phone's height sets the depth tolerance.** At the bottom the knees are out to the side, nearer the camera. Seen from a lens above them, they project lower in the image, so the thigh looks higher than it is. The error grows when the phone is turned towards the person's front. Thigh angle error at the bottom, median (95th percentile), phone 4 m away, 60 reps:

| Lens height | 10 deg towards the back | side-on | 10 deg towards the front |
|---|---|---|---|
| 0.4 m (knee height) | -0.1 (+0.6) | -0.1 (+0.2) | 0.0 (+0.8) |
| 0.7 m | +0.4 (+1.5) | +1.4 (+2.3) | +2.4 (+4.0) |
| 1.0 m | +0.8 (+2.6) | +2.9 (+4.8) | +4.9 (+7.2) |
| 1.3 m | +1.4 (+4.2) | +4.2 (+7.2) | +7.0 (+10.4) |

So the old 3.1 deg tolerance, from tracker noise alone, was too tight for a phone at chest height. A squat filmed that way would have been marked shallow when it wasn't. **Filming advice now includes: phone at knee height.**

The error is a bias, not noise. A lens height known from the video would let the rubric correct for it and keep a tight tolerance. `mediapipe_pose.lens_height_estimate` reads the height from floor perspective, since the far foot looks higher in the image the higher the lens. It's calibrated only on virtual-camera reps and gives about 1.5 m for squat_1, so it isn't used for grading yet ([real_video_check.md](real_video_check.md)).

## Score and reliability

- `deficits` follows the BSA rule for every criterion: a criterion counts once if it fails in 2 or more reps (or in the only rep). `deficit_total` covers all of them and lower is better. `bsa_deficit_total` (0-5) counts only the BSA criteria.
- `score` is 100 x the mean share of reps passing each criterion, with equal weights. That's a choice, not evidence; the BSA also counts criteria equally. `bsa_score` is the same over the BSA criteria only.
- `reliable` is False and `warnings` explains why in two cases:
  - the camera is more than 26 deg from side-on (from MediaPipe's 3D hip axis);
  - the tracked standing pose puts the centre of mass outside 0.16-0.73 of foot length. That's Fujiwara 2010's 31-58% quiet-standing range, widened by the tracker COM error. Such a pose can't be stood in, so the angles are distorted.

## How it behaves

**Injected faults**, 100 Ishida-style 3D reps each, filmed by a sampled phone, edited, then given tracker noise:

| Fault | Caught |
|---|---|
| Too shallow (85 deg knee) | 100% |
| Heels lift 25% / 10% of foot length | 100% / 91% |
| Hips rise first, trunk +10 / +5 deg | 100% / 72% |
| Hitch on the way down, 8% / 4% of hip height | 100% / 70% |
| Forward lean (strong trunk preference, little ankle) | 99% |
| Knees 15 / 8 deg short of straight at the top | 97% / 59% |
| Sit back, trunk 35 / 20 deg further back at the bottom | 98% / 41% |

The fault-specific tests (`tests/test_rubric.py`) also check that each fault fails only its own criterion.

**Unmodified synthetic reps** (200 per protocol): how often each criterion passes, graded from a perfect side view and from the filmed, tracked view, and how often the two agree.

| | Ishida parallel: exact / tracked / agree | Zawadka max depth: exact / tracked / agree |
|---|---|---|
| depth | 0.74 / 0.59 / 0.85 | 0.68 / 0.63 / 0.93 |
| heels | 1.00 / 0.96 / 0.96 | 1.00 / 0.92 / 0.92 |
| trunk | 0.95 / 0.94 / 0.96 | 0.87 / 0.90 / 0.96 |
| ascent | 1.00 / 0.95 / 0.95 | 1.00 / 0.96 / 0.96 |
| descent | 1.00 / 0.95 / 0.95 | 1.00 / 0.98 / 0.98 |
| balance | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| lockout | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |

- **Depth.** The wider tolerance passes more people from a perfect side view (74% vs 40% before). The filmed view passes fewer (59%), because a phone above knee height makes the thigh look higher.
- **Trunk.** Exact and tracked views now agree (96%). Before, the knees-out noise term was added on top of a single-plane rep, so it counted as tracker error. Now both views show the same 3D squat.
- **Agreement of 0.85-0.98 is the label noise** a model trained on tracked-view grades would face. Labels for training should be computed on the same tracked view the model sees.
- **False alarms run at about 5%** on heels, ascent and descent in the tracked view, as the 95th-percentile tolerances intend. Balance and lockout almost never false-alarm. The generator always balances and stands up straight, so these cohorts can't show how often real people fail them.
- Consistency isn't in this table: each synthetic track has one rep.

**Real clips:**

| Clip | Reliable | Score (BSA only) | Deficits |
|---|---|---|---|
| squat_1 (side-on) | yes | 73 (57) | depth (4 of 6 reps fail; the thigh line is 7-10 deg short of parallel in those, 3-6 in the other two); trunk (12-20 deg ahead of the shin, 36 in the "sit back" rep, allowance 0); ascent (3 of 6 reps, by 0.4-2.4 deg over the tolerance). Balance, lockout and consistency all pass. |
| squat_2 (48 deg oblique) | **no**: camera angle, and an impossible standing balance (0.77) | 46 (51) | not to be trusted |

On squat_1:

- **Trunk.** In reps 1-4 her shin lean is 28 deg, below the 36.8 deg cued value. At that shin lean, balance alone needs her trunk 5-12 deg ahead of the shin. She leans 3-8 deg more than that, and her centre of mass sits at 0.40-0.46 of the foot. That's ahead of Ishida's 0.363 average but inside the normal range. So by this rubric's logic, most of her extra lean comes from shins that stay back, and the rest from weight a little forward of average.
- **Sit-back rep.** The sixth rep (a deliberate "sit back" squat) puts her centre of mass at 0.17, towards the heels. That's just inside the tolerance (0.20 - 0.094 = 0.11), so it passes balance.
- **Ascent.** Her ascent deficit is marginal. If MediaPipe's temporal smoothing lags some landmarks more than others, it would produce the same up-down difference, and the noise model has no lag.

squat_2 fails lockout in every rep (knees 7-19 deg bent at the top, continuous fast reps) and balance (centre of mass 0.61-0.87). But the clip is unreliable, and its standing pose already reads off-balance.

## Limits

- The BSA is a proposed, unvalidated screen. Its standards (parallel depth, trunk parallel to the shin) suit a general back-squat pattern and aren't the only valid squat. A front squat, a deep squat for mobility or a box squat would need other standards.
- Only side-view criteria. Knee valgus, hip shift and other frontal-plane faults are invisible here; the front-view rubric grades BSA 4 and 5 ([front_rubric.md](front_rubric.md)).
- The build allowance rests on the balance model, de Leva mass fractions and Ishida's numbers from 16 young men. It's applied to everyone.
- Tolerances come from a noise model calibrated on one side-on clip, without tracker lag, and from assumed phone placements.
- Missing frames (MediaPipe dropouts) are interpolated; ten random dropouts per point plus a 0.5 s gap changed no grade on the side-on clip (`tests/test_rubric.py`). A person facing -x is mirrored with a warning, and poorly visible legs (MediaPipe visibility below 0.5) mark the result unreliable.
- The balance range comes from 16 young men in one study. The lockout and consistency standards are conventions.
- Not validated against expert raters. Comparing these grades with coaches scoring the same videos is the next real test.
