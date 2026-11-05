
import numpy as np


def solve_two_bone_ik(
    start,
    target,
    length1,
    length2,
    bend_direction=1,
):
    """
    Solve a 2D two-bone chain.

    start: ankle position
    target: desired hip position
    length1: shin length
    length2: thigh length
    bend_direction: -1 or 1

    Returns the knee position.
    """
    start = np.asarray(start, dtype=float)
    target = np.asarray(target, dtype=float)

    delta = target - start
    distance = np.linalg.norm(delta)

    min_reach = abs(length1 - length2)
    max_reach = length1 + length2

    if not min_reach < distance < max_reach:
        raise ValueError(
            f"Target distance {distance:.3f}m "
            "is outside the supported IK range"
        )

    direction = delta / distance

    # Distance along the ankle-to-hip line
    a = (
        length1**2 - length2**2 + distance**2
    ) / (2 * distance)

    # Perpendicular distance to knee
    h = np.sqrt(max(0.0, length1**2 - a**2))

    midpoint = start + a * direction

    perpendicular = np.array([
        direction[1],
        -direction[0],
    ])

    knee = midpoint + bend_direction * h * perpendicular

    return knee
