import numpy as np

from exercise_quality.core.body import BodyProportions


def generate_random_body(seed=None):
    rng = np.random.default_rng(seed)

    # Overall body-size variation
    scale = rng.uniform(0.85, 1.15)

    # Relative proportions
    leg_ratio = rng.uniform(0.90, 1.10)
    thigh_ratio = rng.uniform(0.90, 1.10)
    torso_ratio = rng.uniform(0.90, 1.10)

    return BodyProportions(
        shin=0.43 * scale * leg_ratio,
        thigh=0.45 * scale * leg_ratio * thigh_ratio,
        torso=0.55 * scale * torso_ratio,
        neck=0.12 * scale,
        foot=0.25 * scale,
    )
