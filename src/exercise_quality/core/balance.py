"""Whole-body centre of mass and the base of support in the side view.

Segment masses and centre-of-mass positions come from de Leva (1996),
see core/deleva.py. Each segment's centre of mass lies on the line
between its two joint-centre landmarks, at de Leva's fraction from the
origin landmark. Left and right limbs are separate segments.

Static balance: if the body is still, the ground reaction force acts
straight up through the centre of mass, so the centre of pressure (COP)
under the feet equals the horizontal position of the centre of mass. For
a person to stay on flat feet it must lie between heel and toe. During
movement the two separate; see core/dynamics.py.
"""

from dataclasses import dataclass

import numpy as np

from exercise_quality.core.sagittal_model import SEGMENTS


@dataclass(frozen=True)
class ModelSegment:
    """Inertia of one model segment for a particular body."""

    mass: float               # kg
    origin: str               # landmark
    other: str                # landmark
    com_fraction: float       # along origin -> other
    gyration_fraction: float  # radius of gyration / origin-other distance


def model_segments(body):
    """Every model segment with this body's masses (one per side for limbs).

    The foot is de Leva's single heel-to-toe-tip segment. In the model it
    is split at the ball of the foot, and all its mass is carried on the
    rearfoot (heel to ball): the toes are a small fraction of foot mass
    and de Leva gives no split. The centre of mass keeps de Leva's
    distance from the heel, so it is re-expressed as a fraction of the
    heel-to-ball length; likewise the radius of gyration.
    """
    out = {}
    for name, (deleva_name, origin, other) in SEGMENTS.items():
        p = body.segments[deleva_name]
        com, gyration = p.com_fraction, p.sagittal_gyration_fraction
        if deleva_name == "foot":
            scale = body.foot_length / body.ball_length
            com, gyration = com * scale, gyration * scale
        out[name] = ModelSegment(body.mass * p.mass_fraction, origin, other, com, gyration)
    return out


def segment_masses(body):
    """Mass in kg of each model segment."""
    return {name: s.mass for name, s in model_segments(body).items()}


def segment_centres(body, landmarks):
    """Centre of mass position of each segment, shape (..., 2)."""
    centres = {}
    for name, s in model_segments(body).items():
        start = landmarks[s.origin]
        centres[name] = start + s.com_fraction * (landmarks[s.other] - start)
    return centres


def centre_of_mass(body, landmarks):
    """Whole-body centre of mass, shape (..., 2).

    Normalised by the sum of segment masses, so the 0.01% rounding gap in
    de Leva's female table does not bias the result.
    """
    masses = segment_masses(body)
    centres = segment_centres(body, landmarks)
    total = sum(masses.values())
    return sum(masses[n] * centres[n] for n in masses) / total


@dataclass(frozen=True)
class SupportRegion:
    """Heel-to-toe interval on the floor, in the body frame (metres)."""

    heel_x: float
    toe_x: float

    @property
    def length(self):
        return self.toe_x - self.heel_x

    def fraction(self, x):
        """Position as a fraction of foot length from the heel (0 heel, 1 toe)."""
        return (np.asarray(x, dtype=float) - self.heel_x) / self.length

    def x_at(self, fraction):
        return self.heel_x + np.asarray(fraction, dtype=float) * self.length

    def margin(self, x):
        """Distance to the nearer edge; negative when outside the foot."""
        x = np.asarray(x, dtype=float)
        return np.minimum(x - self.heel_x, self.toe_x - x)

    def contains(self, x, margin=0.0):
        return self.margin(x) >= margin


def support_region(body):
    return SupportRegion(-body.heel_offset, body.toe_offset)
