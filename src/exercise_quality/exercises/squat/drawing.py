"""Matplotlib drawing helpers for the sagittal model (kept out of the model code)."""

import numpy as np

from exercise_quality.core.sagittal_model import CONNECTIONS

BODY_COLOUR = "steelblue"
JOINT_COLOUR = "darkorange"
COM_COLOUR = "crimson"
COP_COLOUR = "black"


def draw_pose(ax, landmarks, com=None, support=None, cop_x=None, alpha=1.0):
    """Draw one frame. ``landmarks`` maps names to (2,) arrays."""
    artists = []
    for a, b in CONNECTIONS:
        p, q = landmarks[a], landmarks[b]
        artists += ax.plot([p[0], q[0]], [p[1], q[1]], lw=3, color=BODY_COLOUR,
                           alpha=alpha, solid_capstyle="round")
    pts = np.array(list(landmarks.values()))
    artists.append(ax.scatter(pts[:, 0], pts[:, 1], s=14, color=JOINT_COLOUR,
                              zorder=3, alpha=alpha))
    if support is not None:
        artists += ax.plot([support.heel_x, support.toe_x], [0, 0], lw=7,
                           color="0.75", zorder=0, solid_capstyle="butt")
    if com is not None:
        artists.append(ax.scatter([com[0]], [com[1]], s=70, color=COM_COLOUR,
                                  marker="o", zorder=4, edgecolor="white"))
        artists += ax.plot([com[0], com[0]], [0, com[1]], ls=":", lw=1, color=COM_COLOUR)
    if cop_x is not None:
        artists.append(ax.scatter([cop_x], [0], s=60, marker="^", color=COP_COLOUR, zorder=5))
    return artists


def style_axes(ax, height=2.0):
    ax.set_xlim(-0.75, 0.75)
    ax.set_ylim(-0.05, height)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
