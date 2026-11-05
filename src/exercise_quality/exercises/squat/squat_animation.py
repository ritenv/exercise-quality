import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from exercise_quality.exercises.squat.movement import (
    calculate_squat_pose,
)
from exercise_quality.core.skeleton import (
    calculate_joints,
    connections,
)
from exercise_quality.core.body import BodyProportions
from exercise_quality.core.body_generator import (
    generate_random_body,
)

# Animation settings
FPS = 30
DURATION = 4
FRAMES = FPS * DURATION

body = generate_random_body(seed=42)

def squat_progress(t):
    """Smoothly transition standing -> squat -> standing."""
    return np.sin(np.pi * t) ** 2


def animate(frame):
    ax.clear()

    t = frame / (FRAMES - 1)
    progress = squat_progress(t)

    joints = calculate_squat_pose(
        progress,
        depth=1.0,
        hip_shift=0.25,
        torso_lean=35,
        body=body,
    )

    for start, end in connections:
        p1, p2 = joints[start], joints[end]
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            linewidth=4,
            color="steelblue",
        )

    for name, position in joints.items():
        ax.scatter(*position, s=80, color="darkorange")

    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.1, 1.8)
    ax.set_aspect("equal")
    ax.set_title("Exercise Quality — Squat Animation")
    ax.grid(True, alpha=0.2)


fig, ax = plt.subplots(figsize=(5, 7))

animation = FuncAnimation(
    fig,
    animate,
    frames=FRAMES,
    interval=1000 / FPS,
    repeat=True,
)

plt.show()
