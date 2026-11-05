"""Small 2D geometry helpers shared across the package.

Coordinate frame (see docs/conventions.md): x points forward (towards the
toes), y points up, units are metres, angles are in degrees.
"""

import numpy as np


def up_vector(inclination_deg):
    """Unit vector tilted forward from vertical-up by ``inclination_deg``.

    Works on scalars or arrays; the last axis of the result is (x, y).
    """
    a = np.radians(inclination_deg)
    return np.stack([np.sin(a), np.cos(a)], axis=-1)


def down_vector(angle_deg):
    """Unit vector swung forward from vertical-down by ``angle_deg``.

    0 hangs straight down, 90 points straight forward, 180 points up.
    """
    a = np.radians(angle_deg)
    return np.stack([np.sin(a), -np.cos(a)], axis=-1)


def inclination_from_vertical(vector):
    """Signed angle of ``vector`` from vertical-up, positive when it leans forward."""
    v = np.asarray(vector, dtype=float)
    return np.degrees(np.arctan2(v[..., 0], v[..., 1]))


def cross2d(a, b):
    """z component of the cross product of 2D vectors (numpy 2 dropped 2D np.cross)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def wrap_degrees(angle):
    """Wrap angles into (-180, 180]."""
    return 180.0 - np.mod(180.0 - np.asarray(angle, dtype=float), 360.0)


def angle_between(v1, v2):
    """Smallest angle between two 2D vectors, in degrees.

    Raises ValueError for a zero-length vector, because the angle is
    undefined and silently returning a number hides bugs.
    """
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    denominator = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denominator < 1e-12:
        raise ValueError("Cannot measure an angle with a zero-length vector")
    cosine = np.clip(np.dot(v1, v2) / denominator, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))
