# exercise-quality

Synthetic, physically grounded exercise movement, generated in the open, as the basis for small models that describe exercise form. The first exercise is the bodyweight squat, seen from the side.

The current focus is making the generated movement true to life before any machine learning. A first side-view form rubric now exists, built on a published screening standard and adjusted for body proportions.

## What it does now

- Builds bodies from **real people** (ANSUR II, 6068 adults), not random limb lengths.
- Gives them **published mass distribution** (de Leva 1996) and computes the whole-body centre of mass.
- Generates squats in **real seconds** with minimum-jerk motion. **Balance is solved every frame**, with the correction shared between ankle and trunk, so long thighs or stiff ankles change the squat without being programmed in.
- Splits the trunk into **pelvis, abdomen and rib cage** (de Leva's three trunk parts) joined by two spine joints, so the pelvis tips back and the lower back rounds once the hip runs out of range, as measured in real squats. People who can't reach the requested depth squat as deep as their body allows.
- Models **left and right limbs** separately (the right side faces the camera) and a **toe joint** at the ball of the foot, ready for one-legged and heel-raise movements.
- Adds **natural postural sway** matched to quiet-standing force-plate data, so nobody stands unnaturally still.
- Checks the **dynamic centre of pressure** to confirm each rep would stay on its feet.
- Compares simulated cohorts with **published squat and standing studies**. See [docs/validation.md](docs/validation.md), including where it falls short.
- Tracks **real videos** with MediaPipe Pose and compares them with synthetic reps of matched people, seen through a **tracker-noise model** measured on real video. See [docs/real_video_check.md](docs/real_video_check.md).
- Grades squats with a **side-view form rubric**: five criteria from the Back Squat Assessment, plus balance over the foot, lockout at the top and rep-to-rep consistency. The trunk criterion allows for build: someone whose proportions need more forward lean to balance isn't marked down for it. See [docs/rubric.md](docs/rubric.md).

## Quick start

Use a native Python (arm64 on Apple silicon): MediaPipe has no wheels for Intel macOS, including Python running under Rosetta.

```bash
uv venv .venv --python 3.13 && source .venv/bin/activate   # or python3 -m venv .venv
uv pip install -r requirements.txt
python -m pytest                                              # all tests
PYTHONPATH=src python scripts/animate_balanced_squat.py 3     # one rep, seed 3
PYTHONPATH=src python scripts/plot_squat_gallery.py           # 12 people at the bottom
PYTHONPATH=src python scripts/plot_squat_curves.py            # angles, COM, COP over time
PYTHONPATH=src python scripts/render_filmstrip.py 5 zawadka2020_max_depth out.png  # one rep as stills
PYTHONPATH=src python scripts/validate_against_literature.py  # tables in docs/validation.md
PYTHONPATH=src python scripts/calibrate_lumbopelvic.py        # refit hip end range (parallel, ~1 min)
PYTHONPATH=src python scripts/stress_test.py 1500              # check 1500 reps for spikes, falls, outliers
PYTHONPATH=src python scripts/extract_video_landmarks.py VIDEO # track a video with MediaPipe (model downloads to models/)
PYTHONPATH=src python scripts/real_video_check.py              # real vs matched synthetic reps (docs/real_video_check.md)
PYTHONPATH=src python scripts/rubric_report.py                 # rubric on real clips, cohorts and injected faults
```

```python
import numpy as np
from exercise_quality.core.sagittal_body import BodyPopulation
from exercise_quality.core.dynamics import analyse_dynamics
from exercise_quality.exercises.squat.balanced import generate_feasible_squat
from exercise_quality.exercises.squat.protocols import ISHIDA_PARALLEL, sample_style

rng = np.random.default_rng(0)
body = BodyPopulation.load().sample(rng)
rep = generate_feasible_squat(body, sample_style(rng, body, ISHIDA_PARALLEL))
rep.landmarks["right_knee"]    # (T, 2) metres
rep.measured()["trunk_inclination"]
analyse_dynamics(rep).cop_fraction

from exercise_quality.core.observed import from_model_landmarks
from exercise_quality.exercises.squat import rubric
result = rubric.assess(from_model_landmarks(rep.landmarks, rep.fps), body=body)
result.score, result.bsa_score, result.deficits
```

## Layout

```
src/exercise_quality/
  core/
    deleva.py          segment mass and centre-of-mass tables (de Leva 1996)
    sagittal_body.py   bodies from ANSUR II measurements
    sagittal_model.py  foot-rooted side-view kinematics, joint angles
    balance.py         centre of mass, heel-to-toe support
    dynamics.py        dynamic centre of pressure
    timing.py          minimum-jerk rep timing
    sway.py            postural sway noise
    geometry.py        shared 2D helpers
    observed.py        what a pose tracker sees, and angles from it
    observed_balance.py centre of mass over the foot from a tracked pose
    tracker_noise.py   MediaPipe-like noise for synthetic tracks
    signal.py          zero-lag low-pass filter
    (skeleton.py, kinematics.py, body.py, body_generator.py, biomechanics.py: first prototype)
  exercises/squat/
    balanced.py        balance-constrained squat generator
    protocols.py       per-person styles sampled from published studies
    literature.py      the published numbers, with citations
    cohort.py          simulate a study cohort and compare
    metrics.py         descriptors: depth category, trunk-tibia angle, pelvic tilt
    reps.py            find and measure reps in a tracked or synthetic track
    rubric.py          side-view form rubric (Back Squat Assessment, build-adjusted)
    track_faults.py    inject known faults into a track, for testing the rubric
    drawing.py         plotting helpers
    (movement.py, movement_v2.py, validation.py, squat_animation.py: first prototype)
  video/
    mediapipe_pose.py  stored MediaPipe tracks -> tracker view, camera-angle correction
    compare.py         body matching and matched synthetic reps
  data/                ANSUR II subset, real-video landmark tracks, provenance
docs/
  conventions.md       axes, angle signs, landmarks
  model.md             every source and assumption
  validation.md        comparison with published data and known gaps
  real_video_check.md  two tracked real videos vs matched synthetic reps
  rubric.md            the form rubric, its sources and how it behaves
```

## Limits

Side view only, two spine joints, quasi-static planning, young and fit reference populations. Two real videos so far. The rubric is a technique standard, not validated against expert raters. No muscle forces, no medical or injury claims. See [docs/model.md](docs/model.md).
