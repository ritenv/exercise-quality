
import numpy as np
import matplotlib.pyplot as plt

from exercise_quality.core.body import BodyProportions
from exercise_quality.core.biomechanics import measure_pose
from exercise_quality.exercises.squat.movement import (
    calculate_squat_pose,
)
from exercise_quality.exercises.squat.movement_v2 import (
    calculate_squat_pose_v2,
)

body = BodyProportions()
progress_values = np.linspace(0, 1, 101)

metrics = [
    "knee_flexion_deg",
    "shin_inclination_deg",
    "torso_inclination_deg",
    "hip_height_m",
]

old_data = {metric: [] for metric in metrics}
new_data = {metric: [] for metric in metrics}

for progress in progress_values:
    old_pose = calculate_squat_pose(progress, body=body)
    new_pose = calculate_squat_pose_v2(progress, body=body)

    old = measure_pose(old_pose, body)
    new = measure_pose(new_pose, body)

    for metric in metrics:
        old_data[metric].append(old[metric])
        new_data[metric].append(new[metric])

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax, metric in zip(axes.flat, metrics):
    ax.plot(
        progress_values,
        old_data[metric],
        label="Original generator",
    )
    ax.plot(
        progress_values,
        new_data[metric],
        label="Joint-coordinated generator",
    )

    ax.set_title(metric)
    ax.set_xlabel("Descent progress")
    ax.grid(alpha=0.2)
    ax.legend()

plt.tight_layout()
plt.show()
