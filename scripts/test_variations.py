import numpy as np
from collections import Counter

from exercise_quality.core.body_generator import generate_random_body
from exercise_quality.exercises.squat.movement import calculate_squat_pose
from exercise_quality.exercises.squat.validation import validate_squat_pose

rng = np.random.default_rng(42)

total = 1000
valid = 0
failures = Counter()

for i in range(total):
    body = generate_random_body(seed=i)

    depth = rng.uniform(0.4, 1.0)
    hip_shift = rng.uniform(0.10, 0.35)
    torso_lean = rng.uniform(15, 60)

    try:
        for progress in np.linspace(0, 1, 30):
            joints = calculate_squat_pose(
                progress,
                depth=depth,
                hip_shift=hip_shift,
                torso_lean=torso_lean,
                body=body,
            )

            result = validate_squat_pose(joints)

            if not result["valid"]:
                failures.update(result["issues"])
                raise ValueError("Invalid pose")

        valid += 1

    except ValueError as error:
        if str(error) != "Invalid pose":
            failures["ik_unreachable"] += 1

print(f"Valid movements: {valid}/{total}")
print(f"Invalid movements: {total - valid}/{total}")
print(f"Acceptance rate: {valid / total:.1%}")
print(f"Failure reasons: {dict(failures)}")
