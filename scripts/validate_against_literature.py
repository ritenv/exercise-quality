"""Compare simulated cohorts with published squat studies.

    PYTHONPATH=src python scripts/validate_against_literature.py [n_per_cohort] [--3d]

Prints one markdown table per study. z is (model mean - published mean)
divided by the published between-subject SD.
"""

import sys

import numpy as np

from exercise_quality.core.sagittal_body import BodyPopulation
from exercise_quality.exercises.squat import literature as lit
from exercise_quality.exercises.squat.cohort import KIND, validation_report


def main(n, three_d=False):
    report = validation_report(BodyPopulation.load(), n=n, three_d=three_d)
    for title, (cohort, rows, study) in report.items():
        print(f"\n### {title}\n")
        print(f"Simulated people: {len(cohort.metrics)}, feasible reps: "
              f"{100 * cohort.feasible_share:.0f}%, depth limited by mobility: "
              f"{100 * cohort.depth_limited_share:.0f}%\n")
        print("| Metric | Model mean (SD) | Published mean (SD) | z | Kind |")
        print("|---|---|---|---|---|")
        for r in rows:
            pub_sd = "n/a" if not np.isfinite(r.published_sd) else f"{r.published_sd:.3g}"
            z = "n/a" if not np.isfinite(r.z) else f"{r.z:+.2f}"
            kind = KIND.get((study, r.metric), "")
            print(f"| {r.what} | {r.model_mean:.3g} ({r.model_sd:.2g}) | "
                  f"{r.published_mean:.3g} ({pub_sd}) | {z} | {kind} |")
        standing = cohort.values("standing_cop_fraction")
        lo, hi = lit.QUIET_STANDING_COP_RANGE
        inside = 100 * np.mean((standing >= lo) & (standing <= hi))
        print(f"\nStanding COP {100 * standing.mean():.0f}% of foot (SD {100 * standing.std():.0f}); "
              f"{inside:.0f}% of people inside Fujiwara 2010's quiet-standing range "
              f"({100 * lo:.0f}-{100 * hi:.0f}%).")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--3d"]
    main(int(args[0]) if args else 200, three_d="--3d" in sys.argv)
