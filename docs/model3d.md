# 3D model and virtual camera

The side-view model ([model.md](model.md)) only knows what happens in one plane. To see a squat from the front, or from wherever someone props their phone, and to make front-view faults such as knees caving in, the body now has width.

- `core/model3d.py` is the 3D body and pose.
- `video/camera.py` is a virtual phone camera that reports MediaPipe-style landmarks.

```python
from exercise_quality.core import model3d as M
from exercise_quality.core.sagittal_body import BodyPopulation
from exercise_quality.exercises.squat.balanced import SquatStyle, generate_squat
from exercise_quality.video import camera as cam, mediapipe_pose as mpp

body = M.make_body3d(BodyPopulation.load(), "female", 10)
rep = generate_squat(body, SquatStyle(stance=M.Stance(toe_out=20, knee_track_right=15)))
rep.landmarks3d                     # 3D pose; rep.landmarks is the side view
raw = cam.synthetic_raw(rep, cam.Camera(azimuth=0, distance=3.5))   # a front-on phone
track = mpp.observed_track(raw)     # from here on, the same code as real video
```

## How it works

The validated side-view model stays the core. Every joint angle keeps its meaning: the knee, ankle and leg-segment angles are measured within each leg's own plane, and the trunk within the body's mid-plane. 3D adds where that geometry sits in space.

1. **Feet.** Both feet are flat on the floor. The distance between them is `Stance.width_ratio` × hip breadth, measured centre to centre, and each is turned out by `Stance.toe_out`.
2. **Each leg moves in its own plane, which contains the foot's long axis.** That's the "knees track over the toes" standard. In 3D studies of healthy squats the knee itself barely bends sideways: peak knee valgus 0.3 ± 3.7° and varus 2.2 ± 4.2° (Han et al. 2013, below), and Sinclair 2022's peak knee adduction is about 0 ± 8°. Sideways knee movement comes from the hip.
3. **Closed chain.** Each leg's plane tilts sideways about the foot axis until the two hip joint centres land their measured distance apart. With feet wider than the hips, the legs lean in; that's hip abduction.
4. **Balance.** The side-view balance solver now uses the forward position of the 3D centre of mass.
5. **Trunk, head and arms** stay in the body's mid-plane at the side-view angles. The shoulders sit `shoulder_width` apart.

With the ankles directly under the hip joint centres, no toe-out and no faults, every side-view coordinate equals the old model's exactly (`tests/test_model3d.py`).

**Faults** (`Stance`), all growing from nothing at the start of the rep to full size at the bottom:
- `knee_track_right` / `knee_track_left` turn that leg's plane inward about the vertical, so the knee caves in.
- `lateral_shift` moves the pelvis sideways.

## Where the numbers come from

| What | Value | Source |
|---|---|---|
| Distance between hip joint centres | 16.0 ± 0.8 cm per person, correlated 0.35 with bicristal (iliac crest) breadth | Mullaji et al. 2010, *Skeletal Radiol* 39:363-368, CT of 200 adults (abstract). Not influenced by height or BMI. The sex difference was significant but isn't given, so one distribution covers both (assumption). Indian adults. |
| Shoulder, hip, foot and head breadths | per person | ANSUR II biacromial, hip, foot and head breadths, now in `data/ansur_ii_sagittal.csv`. The shoulder joint is placed below the acromion (assumption). |
| Stance width | 1.23 ± 0.16 × hip breadth, centre to centre | Sinclair et al. 2022 (PMC9503729): 70 male lifters, self-selected, barbell at 70% 1RM, pooled from tertile groups. Measured relative to greater-trochanter marker width; ANSUR hip breadth stands in for it (assumption). |
| Toe-out | 23.9 ± 6.9° | Sinclair et al. 2022 (PMC9501107), same 70 lifters, pooled. |
| Knee frontal-plane joint angles in a healthy squat | near zero | Han, Ge, Liu and Liu 2013, *J Hum Kinet* 39:59-66 (PMC3916928): 15 adults, neutral squat to parallel. |

Bodyweight squats by people who aren't lifters may use different feet. Drawing width and toe-out independently is an assumption, since no correlation is reported. `protocols.sample_stance` draws a stance; `cohort.simulate_cohort(..., three_d=True)` runs the same people and styles as a 2D cohort, with a stance each.

## Validation

`PYTHONPATH=src python scripts/validate_against_literature.py 200 --3d` uses the same cohorts as [validation.md](validation.md).

**Frontal plane** (a prediction: only the stance comes from the same study). Sinclair 2022, men; simulated with the Kasahara parallel protocol, whose depth is closest to Sinclair's peak hip flexion of about 98°:

| | Model | Sinclair 2022 | z |
|---|---|---|---|
| Peak hip abduction | 24.6 (5.6) | 28.5 (8.2) | -0.48 |
| Hip abduction range during the squat | 17.2 (6.1) | 18.2 (7.7) | -0.13 |

Standing hip abduction from the stance alone is 8.2 ± 2.5°. Sinclair's peak minus range implies about 10°.

**Side view, 2D vs 3D.** Nothing was refitted. Turned-out, wider feet shorten the legs' reach front to back, so balance needs less ankle bend and less trunk lean. Demers et al. 2018 described this effect for wide stances: hip abduction "effectively shortens the thigh in the sagittal plane". Most side-view rows move closer to the data:

| | 2D z | 3D z |
|---|---|---|
| Ishida: ankle at 60 / 90 deg knee, peak ankle | +0.63 / +0.52 / +0.70 | +0.16 / +0.08 / +0.11 |
| Ishida: trunk at 60 / 90 deg knee, peak trunk | +0.36 / +0.59 / +0.70 | -0.01 / +0.23 / +0.19 |
| Mata: ankle when the pelvis tilts back | +0.85 | +0.35 |
| Kasahara: trunk range | +0.49 | +0.39 |
| Zawadka lumbar range, men / women | -1.15 / -0.57 | -1.18 / -0.61 |

Everything else moves by 0.1 or less. Kasahara's hip range stays at z +3.2, a known protocol issue ([validation.md](validation.md)).

**What a side camera sees** (a prediction). With the knees going out over turned-out toes, the side view sees the thigh shortened.

| Thigh length in the side view at 100° of knee bend / standing | Value |
|---|---|
| Model, Ishida-protocol cohort | 0.909 ± 0.050 |
| squat_1, real MediaPipe video | 0.90 |

The tracker-noise model needed a fitted "knees out 24°" term to produce this ([real_video_check.md](real_video_check.md)). The 3D geometry now gives it from the stance alone.

**Knee frontal-plane projection angle** (FPPA: from the front, how far the knee bends off the hip-to-ankle line). At 60° of knee bend it's -28 ± 8°: knees outside the line, as knees over turned-out toes must be. This hasn't been checked against front-view data.

## Camera placement

`PYTHONPATH=src python scripts/camera_views.py 120` films 120 3D reps with phones at several angles and distances, adds tracker noise, and runs the real-video pipeline and the rubric. The table shows the error at the deepest knee bend, side-view angle minus the true angle, as median (5-95%). Knee and thigh are compared within the leg's plane; the trunk is the hip-to-shoulder line the tracker measures.

| Camera | Knee | Trunk | Thigh | Depth grade agrees with a perfect side view | Marked reliable |
|---|---|---|---|---|---|
| side-on, 2 m | -3.9 (-12.1, -0.9) | +1.5 | +7.3 | 68% | 100% |
| side-on, 3.5 m | -2.2 (-8.5, +1.5) | +0.9 | +4.2 | 84% | 100% |
| side-on, 6 m | -1.0 (-6.7, +3.8) | +0.3 | +2.4 | 87% | 100% |
| 15° off side-on, 3.5 m | -5.5 (-17.4, +0.6) | -2.6 | +7.5 | 72% | 100% |
| 30° off, 3.5 m | -10.9 (-30.6, +0.2) | -7.1 | +12.3 | 65% | 0% |
| 45° off, 3.5 m | -19.1 (-56.4, -0.7) | -14.3 | +18.5 | 60% | 0% |

- **Even a perfect side camera reads the knee 1-2° less bent and the thigh 2-4° deeper than the true angles.** That's the knees travelling out of the plane, which a side view can't see. The rubric deliberately grades what the side view shows ([rubric.md](rubric.md)).
- **A phone 2 m away adds perspective error.** The thigh reads 7° deeper and depth grades agree only 68% of the time. Iizuka et al. 2026 (PMC13347614) found a 2 m webcam underestimating hip and knee angles by about 11°, mostly from projection.
- **Filming advice: 3.5 m or more, within about 15° of side-on.** The rubric's camera-angle check (26°) passes 15° off but rejects 30°, and errors grow quickly past 15°. Depth agreement stays below 90% even at 6 m, because about half of people asked to squat to parallel finish within a few degrees of it.

## Limits

- **Unchecked from the front.** The front-view predictions, knee FPPA and how the hips and knees look in the image, haven't been compared with any front-view data. That needs front-view videos.
- **Landmark positions:**
  - MediaPipe's hip and shoulder points are learned from human labels, not exact joint centres. Using joint centres matters more from the front.
  - Face and finger points are approximated.
  - Crossed arms are drawn in the shoulders' planes, not across the chest.
- **Tracker noise from the front is unmeasured.** Jitter and drift come from one side-on clip and are assumed the same from every angle.
- **Lateral balance isn't solved.** A sideways shift moves the centre of mass sideways but nothing checks it stays over the feet; small shifts (a few cm) stay well inside.
- **Sources:**
  - The stance comes from loaded male lifters.
  - The hip joint width comes from one CT study, with sexes pooled.
- **Not yet ported.** The real-video check and the rubric's synthetic tests still use the side-view model with the knees-out noise term. Moving them onto the 3D model and camera is the next step, along with a front-view rubric (knees caving in, hip shift, stance) once there are front-view videos.
