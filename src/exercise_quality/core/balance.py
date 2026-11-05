"""Whole-body centre of mass and the base of support in the side view.

Segment masses and centre-of-mass positions come from de Leva (1996),
see core/deleva.py. Each segment's centre of mass lies on the line
between its two joint-centre landmarks, at de Leva's fraction from the
origin landmark. Bilateral segments (arms, legs, feet) count twice.

Static balance: if the body is still, the ground reaction force acts
straight up through the centre of mass, so the centre of pressure (COP)
under the feet equals the horizontal position of the centre of mass. For
a person to stay on flat feet it must lie between heel and toe. During
movement the two separate; see core/dynamics.py.
"""

from dataclasses import dataclass

import numpy as np

from exercise_quality.core import deleva
from exercise_quality.core.sagittal_model import SEGMENT_ENDPOINTS


def segment_masses(body):
    """Mass in kg of each modelled segment, both sides combined."""
    return {
        name: body.mass * p.mass_fraction * (2 if name in deleva.BILATERAL else 1)
        for name, p in body.segments.items()
    }


def segment_centres(body, landmarks):
    """Centre of mass position of each segment, shape (..., 2)."""
    centres = {}
    for name, p in body.segments.items():
        origin, other = SEGMENT_ENDPOINTS[name]
        start = landmarks[origin]
        centres[name] = start + p.com_fraction * (landmarks[other] - start)
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
