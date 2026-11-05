"""Bottom position of 12 real ANSUR II people doing a balanced parallel squat.

    PYTHONPATH=src python scripts/plot_squat_gallery.py [output.png]

Red dot: whole-body centre of mass. Grey bar: heel to toe. Each title
gives the person's thigh/shank ratio and the trunk lean the balance
solver needed, which is what makes different bodies squat differently.
"""

import sys

import matplotlib

if len(sys.argv) > 1:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from exercise_quality.core.sagittal_body import BodyPopulation
from exercise_quality.exercises.squat.balanced import generate_feasible_squat
from exercise_quality.exercises.squat.drawing import draw_pose, style_axes
from exercise_quality.exercises.squat.protocols import ISHIDA_PARALLEL, sample_style

population = BodyPopulation.load()
rng = np.random.default_rng(42)

fig, axes = plt.subplots(3, 4, figsize=(14, 13))
for ax in axes.flat:
    body = population.sample(rng)
    seq = generate_feasible_squat(body, sample_style(rng, body, ISHIDA_PARALLEL), fps=50)
    i = int(np.argmax(seq.angles.knee_flexion))
    m = seq.measured()
    draw_pose(ax, seq.frame(i), com=seq.com[i], support=seq.support)
    style_axes(ax)
    status = "" if seq.is_feasible else "  [INFEASIBLE: " + ", ".join(seq.flags) + "]"
    if "depth_limited_by_mobility" in seq.notes:
        status += "  [depth limited]"
    ax.set_title(
        f"{body.sex}, {body.stature:.2f} m, thigh/shank {body.thigh / body.shank:.2f}\n"
        f"knee {m['knee_flexion'][i]:.0f}, ankle {m['ankle_dorsiflexion'][i]:.0f}, "
        f"trunk {m['trunk_inclination'][i]:.0f} deg\n"
        f"COM at {100 * seq.support.fraction(seq.com[i, 0]):.0f}% of foot{status}",
        fontsize=9,
    )

fig.suptitle("Balanced parallel squats, real body proportions (ANSUR II)")
fig.tight_layout()
if len(sys.argv) > 1:
    fig.savefig(sys.argv[1], dpi=110)
    print(f"saved {sys.argv[1]}")
else:
    plt.show()
