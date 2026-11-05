"""Animate one balanced squat rep with its centre of mass and pressure.

    PYTHONPATH=src python scripts/animate_balanced_squat.py [seed]

Red dot: centre of mass. Black triangle: dynamic centre of pressure.
Grey bar: heel to toe.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from exercise_quality.core.dynamics import analyse_dynamics
from exercise_quality.core.sagittal_body import BodyPopulation
from exercise_quality.exercises.squat.balanced import generate_squat
from exercise_quality.exercises.squat.drawing import draw_pose, style_axes
from exercise_quality.exercises.squat.protocols import ISHIDA_PARALLEL, sample_style

FPS = 30

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
rng = np.random.default_rng(seed)
body = BodyPopulation.load().sample(rng)
seq = generate_squat(body, sample_style(rng, body, ISHIDA_PARALLEL), fps=FPS)
dyn = analyse_dynamics(seq)
m = seq.measured()

fig, ax = plt.subplots(figsize=(5, 7))


def animate(i):
    ax.clear()
    draw_pose(ax, seq.frame(i), com=seq.com[i], support=seq.support, cop_x=dyn.cop_x[i])
    style_axes(ax)
    ax.set_title(
        f"{body.sex}, {body.stature:.2f} m   t = {seq.times[i]:.2f} s\n"
        f"knee {m['knee_flexion'][i]:.0f}  ankle {m['ankle_dorsiflexion'][i]:.0f}  "
        f"trunk {m['trunk_inclination'][i]:.0f} deg",
        fontsize=9,
    )


animation = FuncAnimation(fig, animate, frames=len(seq.times), interval=1000 / FPS, repeat=True)
plt.show()
