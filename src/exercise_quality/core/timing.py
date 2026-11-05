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

    def profile(self, t):
        """Main movement profile over the rep, 0 standing, 1 at the bottom.

        With a pause at the bottom, descent and ascent are separate
        minimum-jerk movements that each start and end at rest.

        Without a pause the rep is one movement that turns around at the
        bottom: velocity is zero there but acceleration is not. Stopping
        acceleration at the turnaround would make the ground force dip
        back to body weight at the bottom, which is not how a continuous
        squat feels. The two halves are quintics that share position,
        zero velocity and the same acceleration at the bottom, with that
        acceleration chosen to minimise total jerk (a minimum-jerk path
        through a via point, after Flash and Hogan 1985).
        """
        t = np.asarray(t, dtype=float)
        if self.bottom_hold > 0:
            return self.lagged_profile(t, 1.0)
        down_coef, up_coef = self._through_bottom_coefficients()
        down = np.clip(t - self.descent_start, 0.0, self.descent)
        up = np.clip(t - self.ascent_start, 0.0, self.ascent)
        going_down = np.polyval(down_coef[::-1], down)
        coming_up = np.polyval(up_coef[::-1], up)
        return np.where(t < self.ascent_start, going_down, coming_up)

    def lagged_profile(self, t, lead):
        """Profile for a joint that reaches its peak early and holds it.

        ``lead`` is the fraction of the descent the joint needs to reach
        its peak. On the way up the motion is the time reverse: the joint
        stays at its peak and starts returning later, finishing with the
        others. There is no ascent timing data behind that choice; it is
        an assumption.
        """
        if not 0.0 < lead <= 1.0:
            raise ValueError("lead must be in (0, 1]")
        t = np.asarray(t, dtype=float)
        down = (t - self.descent_start) / self.descent
        up = (t - self.ascent_start) / self.ascent
        going_down = minimum_jerk(down / lead)
        coming_up = minimum_jerk((1.0 - up) / lead)
        return np.where(t < self.ascent_start, going_down, coming_up)

    def bottom_acceleration(self):
        """Acceleration of the main profile at the bottom (1/s^2)."""
        if self.bottom_hold > 0:
            return 0.0
        return _best_bottom_acceleration(self.descent, self.ascent)

    def _through_bottom_coefficients(self):
        acc = self.bottom_acceleration()
        down = _quintic(0.0, 0.0, 0.0, 1.0, 0.0, acc, self.descent)
        up = _quintic(1.0, 0.0, acc, 0.0, 0.0, 0.0, self.ascent)
        return down, up


def _quintic(x0, v0, a0, x1, v1, a1, duration):
    """Coefficients c0..c5 of the quintic meeting both end states."""
    T = duration
    A = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0],
        [1, T, T**2, T**3, T**4, T**5],
        [0, 1, 2 * T, 3 * T**2, 4 * T**3, 5 * T**4],
        [0, 0, 2, 6 * T, 12 * T**2, 20 * T**3],
    ], dtype=float)
    return np.linalg.solve(A, np.array([x0, v0, a0, x1, v1, a1], dtype=float))


def _jerk_cost(coef, duration):
    """Integral of squared jerk of a quintic over [0, duration]."""
    _, _, _, c3, c4, c5 = coef
    T = duration
    # jerk = 6 c3 + 24 c4 t + 60 c5 t^2, squared and integrated exactly
    return (36 * c3**2 * T + 144 * c3 * c4 * T**2 + (192 * c4**2 + 240 * c3 * c5) * T**3
            + 720 * c4 * c5 * T**4 + 720 * c5**2 * T**5)


def _best_bottom_acceleration(descent, ascent):
    """Bottom acceleration minimising total jerk; the cost is quadratic in it."""
    def cost(acc):
        return (_jerk_cost(_quintic(0, 0, 0, 1, 0, acc, descent), descent)
                + _jerk_cost(_quintic(1, 0, acc, 0, 0, 0, ascent), ascent))
    c0, c1, c2 = cost(0.0), cost(1.0), cost(-1.0)
    curvature = (c1 + c2 - 2 * c0) / 2
    slope = (c1 - c2) / 2
    return -slope / (2 * curvature)
