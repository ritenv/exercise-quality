
import numpy as np
import matplotlib.pyplot as plt

from exercise_quality.core.body_generator import generate_random_body
from exercise_quality.exercises.squat.movement import calculate_squat_pose
from exercise_quality.core.skeleton import connections

rng = np.random.default_rng(42)

fig, axes = plt.subplots(3, 4, figsize=(14, 12))

for i, ax in enumerate(axes.flat):
    body = generate_random_body(seed=i)

    depth = rng.uniform(0.4, 1.0)
    hip_shift = rng.uniform(0.10, 0.35)
    torso_lean = rng.uniform(15, 60)

    joints = calculate_squat_pose(
        progress=1.0,
        depth=depth,
        hip_shift=hip_shift,
        torso_lean=torso_lean,
        body=body,
    )

    for start, end in connections:
        p1, p2 = joints[start], joints[end]

        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            linewidth=3,
            color="steelblue",
        )

    for position in joints.values():
        ax.scatter(*position, s=25, color="darkorange")

    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.1, 1.8)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)

    ax.set_title(
        f"Depth: {depth:.2f}\n"
        f"Hip shift: {hip_shift:.2f}m\n"
        f"Torso: {torso_lean:.0f}°",
        fontsize=9,
    )

plt.suptitle("Exercise Quality — Squat Variations")
plt.tight_layout()
plt.show()
