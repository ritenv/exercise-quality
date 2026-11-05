"""Movement timing in real seconds.

Joint motion follows the minimum-jerk profile, the classic model of
smooth voluntary point-to-point movement (Flash T, Hogan N. "The
coordination of arm movements: an experimentally confirmed mathematical
model." Journal of Neuroscience 5(7):1688-1703, 1985). Position,
velocity and acceleration are all zero at both ends, unlike smoothstep,
whose acceleration jumps from 0 to 6 at the start.

Whether squats follow minimum jerk exactly has not been checked against
data in this project; it is a principled default, not a measured fact.
"""

from dataclasses import dataclass

import numpy as np


def minimum_jerk(s):
    """0 -> 1 as s goes 0 -> 1 (clipped outside)."""
    s = np.clip(np.asarray(s, dtype=float), 0.0, 1.0)
    return s**3 * (10.0 - 15.0 * s + 6.0 * s**2)


def minimum_jerk_velocity(s):
    """d/ds of ``minimum_jerk`` (zero outside [0, 1])."""
    s = np.asarray(s, dtype=float)
    inside = (s >= 0.0) & (s <= 1.0)
    return np.where(inside, 30.0 * s**2 * (1.0 - s) ** 2, 0.0)


def minimum_jerk_acceleration(s):
    """d2/ds2 of ``minimum_jerk`` (zero outside [0, 1])."""
    s = np.asarray(s, dtype=float)
    inside = (s >= 0.0) & (s <= 1.0)
    return np.where(inside, 60.0 * s - 180.0 * s**2 + 120.0 * s**3, 0.0)


@dataclass(frozen=True)
class RepTiming:
    """Phase durations of one repetition, in seconds.

    The defaults (2 s down, 2 s up) match the metronome used by Ishida et
    al. (2025) and the 2 s descent in Kasahara et al. (2024).
    """

    stand_before: float = 0.5
    descent: float = 2.0
    bottom_hold: float = 0.0
    ascent: float = 2.0
    stand_after: float = 0.5

    def __post_init__(self):
        for name in ("stand_before", "bottom_hold", "stand_after"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        for name in ("descent", "ascent"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def total(self):
        return (self.stand_before + self.descent + self.bottom_hold
                + self.ascent + self.stand_after)

    @property
    def descent_start(self):
        return self.stand_before

    @property
    def ascent_start(self):
        return self.stand_before + self.descent + self.bottom_hold

    def times(self, fps):
        n = int(round(self.total * fps)) + 1
        return np.arange(n) / fps

    def profile(self, t, lead=1.0):
        """Normalised joint excursion over the rep, 0 standing, 1 at the bottom.

        ``lead`` is the fraction of the descent a joint needs to reach its
        peak. With lead < 1 the joint gets there early and holds. On the
        way up the motion is the time reverse: the joint stays at its peak
        and starts returning later, finishing with the others. There is no
        ascent timing data behind that choice; it is an assumption.
        """
        if not 0.0 < lead <= 1.0:
            raise ValueError("lead must be in (0, 1]")
        t = np.asarray(t, dtype=float)
        down = (t - self.descent_start) / self.descent
        up = (t - self.ascent_start) / self.ascent
        going_down = minimum_jerk(down / lead)
        coming_up = minimum_jerk((1.0 - up) / lead)
        return np.where(t < self.ascent_start, going_down, coming_up)
