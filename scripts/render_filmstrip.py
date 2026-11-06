"""Render one rep as a filmstrip of frames plus its angle curves.

    PYTHONPATH=src python scripts/render_filmstrip.py SEED [PROTOCOL] [OUT.png]

PROTOCOL is one of: ishida2025_parallel (default), kasahara2024_parallel,
zawadka2020_max_depth. Useful for checking a generated rep by eye without
running an interactive animation.
"""

import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from exercise_quality.core.dynamics import analyse_dynamics
from exercise_quality.core.sagittal_body import BodyPopulation
from exercise_quality.exercises.squat.balanced import generate_feasible_squat
from exercise_quality.exercises.squat.drawing import draw_pose, style_axes
from exercise_quality.exercises.squat.protocols import PROTOCOLS, sample_style

FRAMES = 12


def main(seed, protocol_name, out):
    protocol = PROTOCOLS[protocol_name]
    rng = np.random.default_rng(seed)
    body = BodyPopulation.load().sample(rng)
    seq = generate_feasible_squat(body, sample_style(rng, body, protocol), fps=50)
    dyn = analyse_dynamics(seq)
    m = seq.measured()
    t = seq.style.timing
    moving = (seq.times >= t.descent_start - 0.2) & (seq.times <= t.ascent_start + t.ascent + 0.2)
    idx = np.linspace(np.argmax(moving), len(moving) - 1 - np.argmax(moving[::-1]), FRAMES).astype(int)

    fig = plt.figure(figsize=(22, 10))
    grid = fig.add_gridspec(2, FRAMES // 2 + 2)
    for k, i in enumerate(idx):
        ax = fig.add_subplot(grid[k // (FRAMES // 2), k % (FRAMES // 2)])
        draw_pose(ax, seq.frame(i), com=seq.com[i], support=seq.support, cop_x=dyn.cop_x[i])
        style_axes(ax, 1.9)
        ax.set_title(f"t={seq.times[i]:.2f}  knee {m['knee_flexion'][i]:.0f}\n"
                     f"trunk {m['trunk_inclination'][i]:.0f} pelvis {m['pelvis_inclination'][i]:.0f} "
                     f"lumbar {m['lumbar_flexion'][i]:.0f}", fontsize=8)
        ax.tick_params(labelsize=6)

    ax = fig.add_subplot(grid[0, FRAMES // 2:])
    for name in ("knee_flexion", "hip_flexion", "ankle_dorsiflexion", "trunk_inclination",
                 "pelvis_inclination", "lumbar_flexion"):
        ax.plot(seq.times, m[name], label=name.replace("_", " "))
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.2)
    ax.set_title("angles (deg)")
    ax = fig.add_subplot(grid[1, FRAMES // 2:])
    ax.plot(seq.times, 100 * dyn.cop_fraction, label="COP % foot")
    ax.plot(seq.times, 100 * seq.support.fraction(seq.com[:, 0]), label="COM % foot")
    ax.plot(seq.times, 100 * dyn.vertical_force, label="vertical force % BW")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)
    ax.set_xlabel("time (s)")

    fig.suptitle(f"seed {seed}, {protocol_name}: {body.sex} {body.stature:.2f} m, "
                 f"thigh/shank {body.thigh / body.shank:.2f}  flags={seq.flags} notes={seq.notes}")
    fig.tight_layout()
    fig.savefig(out, dpi=70)
    print(f"saved {out}")


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    protocol = sys.argv[2] if len(sys.argv) > 2 else "ishida2025_parallel"
    out = sys.argv[3] if len(sys.argv) > 3 else f"filmstrip_{seed}.png"
    main(seed, protocol, out)
