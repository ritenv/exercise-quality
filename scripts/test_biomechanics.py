
from exercise_quality.core.body import BodyProportions
from exercise_quality.core.biomechanics import measure_pose
from exercise_quality.exercises.squat.movement import (
    calculate_squat_pose,
)

body = BodyProportions()

for progress in [0.0, 0.25, 0.5, 0.75, 1.0]:
    joints = calculate_squat_pose(progress, body=body)
    measurements = measure_pose(joints, body)

    print(f"\nSquat progress: {progress:.0%}")

    for name, value in measurements.items():
        print(f"  {name}: {value:.3f}")
