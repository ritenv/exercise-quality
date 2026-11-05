"""Compare simulated cohorts with published squat studies.

    PYTHONPATH=src python scripts/validate_against_literature.py [n_per_cohort]

Prints one markdown table per study. z is (model mean - published mean)
divided by the published between-subject SD.
"""

import sys

import numpy as np

from exercise_quality.core.sagittal_body import BodyPopulation
from exercise_quality.exercises.squat.cohort import validation_report


def main(n):
    report = validation_report(BodyPopulation.load(), n=n)
    for title, (cohort, rows) in report.items():
        print(f"\n### {title}\n")
        print(f"Simulated people: {len(cohort.metrics)}, feasible reps: "
              f"{100 * cohort.feasible_share:.0f}%\n")
        print("| Metric | Model mean (SD) | Published mean (SD) | z |")
        print("|---|---|---|---|")
        for r in rows:
            pub_sd = "n/a" if not np.isfinite(r.published_sd) else f"{r.published_sd:.3g}"
            z = "n/a" if not np.isfinite(r.z) else f"{r.z:+.2f}"
            print(f"| {r.what} | {r.model_mean:.3g} ({r.model_sd:.2g}) | "
                  f"{r.published_mean:.3g} ({pub_sd}) | {z} |")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
