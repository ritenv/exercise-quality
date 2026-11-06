# exercise-quality

Synthetic, physically grounded exercise movement, generated in the open, as the basis for small models that describe exercise form. The first exercise is the bodyweight squat, seen from the side.

The current focus is making the generated movement true to life before any machine learning. Nothing here scores movement quality yet.

## What it does now

- Builds bodies from **real people** (ANSUR II, 6068 adults), not random limb lengths.
- Gives them **published mass distribution** (de Leva 1996) and computes the whole-body centre of mass.
- Generates squats in **real seconds** with minimum-jerk motion. **Balance is solved every frame**, with the correction shared between ankle and trunk, so long thighs or stiff ankles change the squat without being programmed in.
- Splits the trunk into **pelvis and thorax**, so the pelvis tips back and the lower back rounds once the hip runs out of range, as measured in real squats. People who can't reach the requested depth squat as deep as their body allows.
- Checks the **dynamic centre of pressure** to confirm each rep would stay on its feet.
- Compares simulated cohorts with **four published squat studies**. See [docs/validation.md](docs/validation.md), including where it falls short.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest                                              # all tests
PYTHONPATH=src python scripts/animate_balanced_squat.py 3     # one rep, seed 3
PYTHONPATH=src python scripts/plot_squat_gallery.py           # 12 people at the bottom
PYTHONPATH=src python scripts/plot_squat_curves.py            # angles, COM, COP over time
PYTHONPATH=src python scripts/validate_against_literature.py  # tables in docs/validation.md
PYTHONPATH=src python scripts/calibrate_lumbopelvic.py        # refit hip end range (slow)
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
rep.landmarks["knee"]          # (T, 2) metres
rep.measured()["trunk_inclination"]
analyse_dynamics(rep).cop_fraction
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
    geometry.py        shared 2D helpers
    (skeleton.py, kinematics.py, body.py, body_generator.py, biomechanics.py: first prototype)
  exercises/squat/
    balanced.py        balance-constrained squat generator
    protocols.py       per-person styles sampled from published studies
    literature.py      the published numbers, with citations
    cohort.py          simulate a study cohort and compare
    metrics.py         descriptors: depth category, trunk-tibia angle, pelvic tilt
    drawing.py         plotting helpers
    (movement.py, movement_v2.py, validation.py, squat_animation.py: first prototype)
  data/                ANSUR II subset and its provenance
docs/
  conventions.md       axes, angle signs, landmarks
  model.md             every source and assumption
  validation.md        comparison with published data and known gaps
```

## Limits

Side view only, one lumbar joint, quasi-static planning, young and fit reference populations. No muscle forces, no medical or injury claims. See [docs/model.md](docs/model.md).
