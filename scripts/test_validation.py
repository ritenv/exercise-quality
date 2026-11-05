import numpy as np

from exercise_quality.core.body_generator import (
    generate_random_body,
)
from exercise_quality.exercises.squat.movement import (
    calculate_squat_pose,
)
from exercise_quality.exercises.squat.validation import (
    validate_squat_pose,
)

total = 100
valid = 0
invalid = 0

for seed in range(total):
    body = generate_random_body(seed)

    try:
        # Check the entire movement, not just the bottom
        for progress in np.linspace(0, 1, 30):
            joints = calculate_squat_pose(
                progress,
                body=body,
            )

            result = validate_squat_pose(joints)

            if not result["valid"]:
                raise ValueError(result["issues"])

        valid += 1

    except ValueError as error:
        invalid += 1
        print(f"Body {seed}: {error}")

print(f"\nValid movements: {valid}/{total}")
print(f"Invalid movements: {invalid}/{total}")
