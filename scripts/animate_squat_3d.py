"""Animate one 3D squat rep: front view, side view and a 3D view you can rotate.

    PYTHONPATH=src python scripts/animate_squat_3d.py [seed] [options]

Options:
    --cave DEG       both knees cave in by DEG at the bottom (Stance.knee_track)
    --cave-right DEG only the right knee
    --shift M        pelvis shifts M metres to the person's right at the bottom
    --protocol NAME  ishida (parallel, default) or zawadka (as deep as possible)
    --save out.gif   save instead of showing (mp4 needs ffmpeg)

The person and their foot stance (width, toe-out) are sampled from the seed.
Drag the 3D panel to rotate it. Red dot: centre of mass. Grey: the feet's
support area. The right side is drawn darker.
"""

import argparse
from dataclasses import replace

import matplotlib

parser = argparse.ArgumentParser()
parser.add_argument("seed", nargs="?", type=int, default=0)
parser.add_argument("--cave", type=float, default=0.0)
parser.add_argument("--cave-right", type=float, default=0.0)
parser.add_argument("--shift", type=float, default=0.0)
parser.add_argument("--protocol", choices=("ishida", "zawadka"), default="ishida")
parser.add_argument("--save")
args = parser.parse_args()
if args.save:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from exercise_quality.core import model3d as M
from exercise_quality.core.sagittal_body import BodyPopulation
from exercise_quality.core.sagittal_model import CONNECTIONS
from exercise_quality.exercises.squat import protocols as P
from exercise_quality.exercises.squat.balanced import generate_feasible_squat

FPS = 30
BONES = CONNECTIONS + (("left_hip", "right_hip"), ("neck", "right_shoulder"),
                       ("neck", "left_shoulder"))
# right side last, so it's drawn on top
BONES = sorted(BONES, key=lambda bone: any("right" in end for end in bone))
PROTOCOL = {"ishida": P.ISHIDA_PARALLEL, "zawadka": P.ZAWADKA_MAX_DEPTH}[args.protocol]

rng = np.random.default_rng(args.seed)
body = M.sample_body3d(BodyPopulation.load(), rng)
stance = replace(P.sample_stance(rng), knee_track_right=args.cave + args.cave_right,
                 knee_track_left=args.cave, lateral_shift=args.shift)
style = replace(P.sample_style(rng, body, PROTOCOL), stance=stance)
seq = generate_feasible_squat(body, style, fps=FPS)
lm, com = seq.landmarks3d, seq.com3d
m = seq.measured()
abduction = M.joint_angles_3d(lm)

title = (f"seed {args.seed}: {body.sex}, {body.stature:.2f} m, feet {stance.width_ratio:.2f}x "
         f"hip width, toes out {stance.toe_out:.0f} deg")
if args.cave or args.cave_right or args.shift:
    title += f"\nfaults: cave L {stance.knee_track_left:.0f} / R {stance.knee_track_right:.0f} deg, shift {args.shift:.2f} m"
if seq.notes:
    title += "\n" + ", ".join(n.replace("_", " ") for n in seq.notes)
if not seq.is_feasible:
    title += "\nINFEASIBLE: " + ", ".join(seq.flags)

fig = plt.figure(figsize=(14, 6.5))
grid = fig.add_gridspec(1, 3, width_ratios=(1, 1, 1.6))
fig.suptitle(title, fontsize=9, color="red" if not seq.is_feasible else "black")
ax_front = fig.add_subplot(grid[0])
ax_side = fig.add_subplot(grid[1])
ax_3d = fig.add_subplot(grid[2], projection="3d")
ax_3d.view_init(elev=15, azim=-50)


def colour(a, b):
    return "navy" if "right" in a or "right" in b else "cornflowerblue"


def foot_outline(side):
    return np.array([lm[f"{side}_{p}"][0] for p in ("heel", "ball", "toe", "heel")])


def draw_2d(ax, i, h, label):
    """``h`` maps a (3,) point to its image x."""
    ax.clear()
    for side in ("left", "right"):
        f = foot_outline(side)
        ax.plot([h(p) for p in f], f[:, 1], color="0.7", lw=4, zorder=0)
    for a, b in BONES:
        p, q = lm[a][i], lm[b][i]
        ax.plot([h(p), h(q)], [p[1], q[1]], lw=3, color=colour(a, b), solid_capstyle="round")
    c = com[i]
    ax.scatter([h(c)], [c[1]], s=60, color="crimson", zorder=4, edgecolor="white")
    ax.plot([h(c), h(c)], [0, c[1]], ls=":", lw=1, color="crimson")
    ax.set_xlim(-0.75, 0.75)
    ax.set_ylim(-0.05, 1.95)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.set_title(label, fontsize=9)


def animate(i):
    # Facing the person: their right is on the viewer's left.
    draw_2d(ax_front, i, lambda p: -p[2], "front (camera in front)")
    draw_2d(ax_side, i, lambda p: p[0], "side (from their right)")
    elev, azim = ax_3d.elev, ax_3d.azim
    ax_3d.clear()
    for side in ("left", "right"):
        f = foot_outline(side)
        ax_3d.plot(f[:, 0], f[:, 2], f[:, 1], color="0.7", lw=4)
    for a, b in BONES:
        p, q = lm[a][i], lm[b][i]
        ax_3d.plot([p[0], q[0]], [p[2], q[2]], [p[1], q[1]], lw=3, color=colour(a, b))
    ax_3d.scatter([com[i][0]], [com[i][2]], [com[i][1]], s=40, color="crimson")
    ax_3d.set_xlim(-0.6, 0.6)
    ax_3d.set_ylim(-0.6, 0.6)
    ax_3d.set_zlim(0, 1.9)
    ax_3d.set_box_aspect((1.2, 1.2, 1.9))
    ax_3d.set_xlabel("forward")
    ax_3d.set_ylabel("right")
    ax_3d.view_init(elev=elev, azim=azim)
    ax_3d.set_title(
        f"t = {seq.times[i]:.2f} s   knee {m['knee_flexion'][i]:.0f}  "
        f"trunk {m['trunk_inclination'][i]:.0f}  hip abd {abduction['right_hip_abduction'][i]:.0f} deg",
        fontsize=9)


animation = FuncAnimation(fig, animate, frames=len(seq.times), interval=1000 / FPS, repeat=True)
if args.save:
    animation.save(args.save, fps=FPS, dpi=90)
    print(f"saved {args.save}")
else:
    plt.show()
