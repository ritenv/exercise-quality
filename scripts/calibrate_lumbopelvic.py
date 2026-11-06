"""Fit the hip end-range parameters that no study reports directly.

    PYTHONPATH=src python scripts/calibrate_lumbopelvic.py [n_per_cohort]

Parameters (population means):
  hip_limit     hip flexion, relative to standing, at which posterior
                pelvic tilt begins (end range onset)
  gain          share of further hip flexion demand the hip still takes
                past end range (0 = hard stop)

The lumbopelvic share before end range is held at protocols.PELVIS_SHARE.
An earlier grid over (share, hip_limit) with a hard stop put the share at
the top of its range (1.0): cost 2.1, versus 4.7 at 0.8 and 18.7 at 0.6.

Fitting targets, chosen because they measure the pelvis and hip directly:
  Zawadka 2020 pelvis tilt ROM (men, women)
  Zawadka 2020 hip ROM (men, women)
  Kasahara 2024 hip ROM
  Kasahara 2024 hip peak time relative to the lowest COM point
  Mata 2021 knee angle when posterior pelvic tilt begins

The range-of-motion targets alone cannot separate a hard stop at 90 deg
from a softer end range that starts lower (a first grid over gains 0-0.5
found a flat ridge: gain 0 at 90 deg cost 2.16, gain 0.4 at 72.5 deg cost
2.91). The hip timing target separates them: with a hard stop the hip
plateaus and peaks long before the bottom, while Kasahara measured it
peaking at the bottom (-1.1 +- 8.6 ms).

After the ankle and trunk started sharing the balance correction, the
refit slid along that ridge to the edge of the grid (onset 70 deg, gain
0.4). An onset that low would mean about 40 deg of hip flexion in
standing against Mata's absolute 110 deg, which is not plausible. Mata's
knee angle at the start of posterior tilt measures where end range
begins, so it is added as a target to pin the onset down.

The objective is the sum of squared z scores against these. Everything
else (lumbar flexion, trunk lean, COM descent, timing, Mata's knee angle
at pelvic tilt) is left out of the fit so it stays an honest prediction.

Every grid point reuses the same random seeds (common random numbers),
so differences between grid points come from the parameters, not noise.
"""

import itertools
import sys

import numpy as np

from exercise_quality.core.sagittal_body import BodyPopulation
from exercise_quality.exercises.squat import literature as lit
from exercise_quality.exercises.squat import protocols as p
from exercise_quality.exercises.squat.cohort import simulate_cohort

GAINS = np.round(np.arange(0.1, 0.501, 0.1), 2)
HIP_LIMITS = np.arange(75.0, 95.01, 2.5)
FPS = 30.0


def targets():
    z = lit.ZAWADKA_2020
    return [
        ("zawadka", "male", "pelvis_rom", z["male"]["pelvis_rom"]),
        ("zawadka", "female", "pelvis_rom", z["female"]["pelvis_rom"]),
        ("zawadka", "male", "hip_rom", z["male"]["hip_rom"]),
        ("zawadka", "female", "hip_rom", z["female"]["hip_rom"]),
        ("kasahara", None, "hip_rom", lit.KASAHARA_2024["hip_rom"]),
        ("kasahara", None, "hip_peak_ms", lit.KASAHARA_2024["hip_peak_ms"]),
        # Mata used full-depth squats, mixed sexes: the 50/50 maximal-depth
        # cohort stands in for it
        ("zawadka", None, "knee_at_ppt", lit.MATA_2021["knee_at_ppt"]),
    ]


def evaluate(population, gain, hip_limit, n):
    kwargs = dict(hip_limit=(hip_limit, p.HIP_LIMIT[1]), hip_gain=gain, fps=FPS)
    cohorts = {
        "zawadka": simulate_cohort(population, p.ZAWADKA_MAX_DEPTH, n, seed=101,
                                   female_share=0.5, **kwargs),
        "kasahara": simulate_cohort(population, p.KASAHARA_PARALLEL, n, seed=202, **kwargs),
    }
    zs = []
    for study, sex, metric, ref in targets():
        values = cohorts[study].values(metric, sex)
        zs.append((values.mean() - ref.mean) / ref.sd)
    return np.array(zs)


def main(n):
    population = BodyPopulation.load()
    names = [f"{s}:{x or 'all'}:{m}" for s, x, m, _ in targets()]
    print("gain   hip    cost   " + "  ".join(f"{nm:>24}" for nm in names))
    best = None
    for gain, hip in itertools.product(GAINS, HIP_LIMITS):
        zs = evaluate(population, gain, hip, n)
        cost = float(np.sum(zs**2))
        print(f"{gain:4.1f}  {hip:5.1f}  {cost:6.2f}  "
              + "  ".join(f"{z:+24.2f}" for z in zs))
        if best is None or cost < best[0]:
            best = (cost, gain, hip, zs)
    cost, gain, hip, zs = best
    print(f"\nBest: gain = {gain:.1f}, hip_limit = {hip:.1f} deg, cost = {cost:.2f}")
    for nm, z in zip(names, zs):
        print(f"  {nm:<28} z = {z:+.2f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 80)
