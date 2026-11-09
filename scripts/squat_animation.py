"""Shared code for animate_squat_3d.py, animate_squat_side.py and animate_squat_front.py.

Each script animates one 3D squat rep from one viewpoint. The person and
their foot stance (width, toe-out) are sampled from the seed, so the same
seed and options give the same rep in every view.

Options:
    --cave DEG       both knees cave in by DEG at the bottom (Stance.knee_track)
    --cave-right DEG only the right knee
    --shift M        pelvis shifts M metres to the person's right at the bottom
    --protocol NAME  ishida (parallel, default) or zawadka (as deep as possible)
    --save out.gif   save instead of showing (mp4 needs ffmpeg)

Red dot: centre of mass. Black triangle (side view): dynamic centre of
pressure. Grey: the feet. The right side is drawn darker.
"""

import argparse
import sys
from dataclasses import replace

import matplotlib

FPS = 30


def parse_args(view):
    parser = argparse.ArgumentParser(
        prog=f"animate_squat_{view}.py",
        description=f"Animate one 3D squat rep, {view} view.")
    parser.add_argument("seed", nargs="?", type=int, default=0)
    parser.add_argument("--cave", type=float, default=0.0,
                        help="both knees cave in by this many degrees at the bottom")
    parser.add_argument("--cave-right", type=float, default=0.0,
                        help="extra caving for the right knee only (deg)")
    parser.add_argument("--shift", type=float, default=0.0,
                        help="pelvis shift to the person's right at the bottom (m)")
    parser.add_argument("--protocol", choices=("ishida", "zawadka"), default="ishida")
    parser.add_argument("--save", help="save to this .gif or .mp4 instead of showing")
    return parser.parse_args()


def run(view):
    """``view``: "3d", "side" or "front"."""
    args = parse_args(view)
    if args.save:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.animation import FuncAnimation

    from exercise_quality.core import model3d as M
    from exercise_quality.core.dynamics import analyse_dynamics
    from exercise_quality.core.sagittal_body import BodyPopulation
    from exercise_quality.core.sagittal_model import CONNECTIONS
    from exercise_quality.exercises.squat import protocols as P
    from exercise_quality.exercises.squat.balanced import generate_feasible_squat

    bones = CONNECTIONS + (("left_hip", "right_hip"), ("neck", "right_shoulder"),
                           ("neck", "left_shoulder"))
    # right side last, so it's drawn on top
    bones = sorted(bones, key=lambda bone: any("right" in end for end in bone))
    protocol = {"ishida": P.ISHIDA_PARALLEL, "zawadka": P.ZAWADKA_MAX_DEPTH}[args.protocol]

    rng = np.random.default_rng(args.seed)
    body = M.sample_body3d(BodyPopulation.load(), rng)
    stance = replace(P.sample_stance(rng), knee_track_right=args.cave + args.cave_right,
                     knee_track_left=args.cave, lateral_shift=args.shift)
    style = replace(P.sample_style(rng, body, protocol), stance=stance)
    seq = generate_feasible_squat(body, style, fps=FPS)
    lm, com = seq.landmarks3d, seq.com3d
    m = seq.measured()
    angles3d = M.joint_angles_3d(lm)
    cop_x = analyse_dynamics(seq).cop_x if view == "side" else None

    title = (f"seed {args.seed}: {body.sex}, {body.stature:.2f} m, feet {stance.width_ratio:.2f}x "
             f"hip width, toes out {stance.toe_out:.0f} deg")
    if args.cave or args.cave_right or args.shift:
        title += (f"\nfaults: cave L {stance.knee_track_left:.0f} / R {stance.knee_track_right:.0f}"
                  f" deg, shift {args.shift:.2f} m")
    if seq.notes:
        title += "\n" + ", ".join(n.replace("_", " ") for n in seq.notes)
    if not seq.is_feasible:
        title += "\nINFEASIBLE: " + ", ".join(seq.flags)

    def colour(a, b):
        return "navy" if "right" in a or "right" in b else "cornflowerblue"

    def foot_outline(side):
        return np.array([lm[f"{side}_{p}"][0] for p in ("heel", "ball", "toe", "heel")])

    def frame_label(i):
        if view == "front":
            extra = f"hip abd {angles3d['hip_abduction'][i]:.0f}"
        elif view == "side":
            extra = f"ankle {m['ankle_dorsiflexion'][i]:.0f}  trunk {m['trunk_inclination'][i]:.0f}"
        else:
            extra = (f"trunk {m['trunk_inclination'][i]:.0f}  "
                     f"hip abd {angles3d['hip_abduction'][i]:.0f}")
        return f"t = {seq.times[i]:.2f} s   knee {m['knee_flexion'][i]:.0f}  {extra} deg"

    if view == "3d":
        fig = plt.figure(figsize=(7, 7.5))
        ax = fig.add_subplot(projection="3d")
        ax.view_init(elev=15, azim=-50)
        title += "\ndrag to rotate"
    else:
        fig, ax = plt.subplots(figsize=(5.5, 7.5))
    fig.suptitle(title, fontsize=9, color="red" if not seq.is_feasible else "black")

    def draw_2d(i):
        # front: facing the person, their right is on the viewer's left
        h = (lambda p: -p[2]) if view == "front" else (lambda p: p[0])
        ax.clear()
        for side in ("left", "right"):
            f = foot_outline(side)
            ax.plot([h(p) for p in f], f[:, 1], color="0.7", lw=4, zorder=0)
        for a, b in bones:
            p, q = lm[a][i], lm[b][i]
            ax.plot([h(p), h(q)], [p[1], q[1]], lw=3, color=colour(a, b), solid_capstyle="round")
        c = com[i]
        ax.scatter([h(c)], [c[1]], s=60, color="crimson", zorder=4, edgecolor="white")
        ax.plot([h(c), h(c)], [0, c[1]], ls=":", lw=1, color="crimson")
        if cop_x is not None:
            ax.scatter([cop_x[i]], [0], s=60, marker="^", color="black", zorder=5)
        ax.set_xlim(-0.75, 0.75)
        ax.set_ylim(-0.05, 1.95)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        ax.set_xlabel("person's right  <-   (m)   ->  person's left" if view == "front"
                      else "back  <-   (m)   ->  forward", fontsize=8)

    def draw_3d(i):
        elev, azim = ax.elev, ax.azim
        ax.clear()
        for side in ("left", "right"):
            f = foot_outline(side)
            ax.plot(f[:, 0], f[:, 2], f[:, 1], color="0.7", lw=4)
        for a, b in bones:
            p, q = lm[a][i], lm[b][i]
            ax.plot([p[0], q[0]], [p[2], q[2]], [p[1], q[1]], lw=3, color=colour(a, b))
        ax.scatter([com[i][0]], [com[i][2]], [com[i][1]], s=40, color="crimson")
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)
        ax.set_zlim(0, 1.9)
        ax.set_box_aspect((1.2, 1.2, 1.9))
        ax.set_xlabel("forward")
        ax.set_ylabel("right")
        ax.view_init(elev=elev, azim=azim)

    def animate(i):
        (draw_3d if view == "3d" else draw_2d)(i)
        ax.set_title(frame_label(i), fontsize=9)

    animation = FuncAnimation(fig, animate, frames=len(seq.times), interval=1000 / FPS,
                              repeat=True)
    if args.save:
        animation.save(args.save, fps=FPS, dpi=90)
        print(f"saved {args.save}")
    else:
        plt.show()
    return animation


if __name__ == "__main__":
    sys.exit("Run animate_squat_3d.py, animate_squat_side.py or animate_squat_front.py.")
