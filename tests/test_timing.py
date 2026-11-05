import numpy as np
import pytest

from exercise_quality.core.timing import (
    RepTiming,
    minimum_jerk,
    minimum_jerk_acceleration,
    minimum_jerk_velocity,
)


def test_minimum_jerk_boundary_conditions():
    assert minimum_jerk(0.0) == 0.0
    assert minimum_jerk(1.0) == 1.0
    for fn in (minimum_jerk_velocity, minimum_jerk_acceleration):
        assert fn(0.0) == pytest.approx(0.0)
        assert fn(1.0) == pytest.approx(0.0)
    # the property smoothstep lacks: no acceleration jump at the start
    assert minimum_jerk_acceleration(1e-6) == pytest.approx(0.0, abs=1e-3)


def test_minimum_jerk_is_symmetric_and_monotonic():
    s = np.linspace(0, 1, 1001)
    np.testing.assert_allclose(minimum_jerk(1 - s), 1 - minimum_jerk(s), atol=1e-12)
    assert np.all(np.diff(minimum_jerk(s)) >= 0)
    assert minimum_jerk(0.5) == pytest.approx(0.5)
    assert minimum_jerk_velocity(0.5) == pytest.approx(1.875)  # peak speed


def test_minimum_jerk_derivatives_match_finite_differences():
    s = np.linspace(0.01, 0.99, 99)
    h = 1e-6
    numeric_v = (minimum_jerk(s + h) - minimum_jerk(s - h)) / (2 * h)
    numeric_a = (minimum_jerk_velocity(s + h) - minimum_jerk_velocity(s - h)) / (2 * h)
    np.testing.assert_allclose(minimum_jerk_velocity(s), numeric_v, atol=1e-5)
    np.testing.assert_allclose(minimum_jerk_acceleration(s), numeric_a, atol=1e-4)


def test_minimum_jerk_clips_outside_unit_interval():
    assert minimum_jerk(-0.5) == 0.0
    assert minimum_jerk(1.5) == 1.0
    assert minimum_jerk_velocity(1.5) == 0.0


def test_rep_phases():
    timing = RepTiming(stand_before=0.5, descent=2.0, bottom_hold=0.3,
                       ascent=1.5, stand_after=0.4)
    assert timing.total == pytest.approx(4.7)
    p = timing.profile
    assert p(0.0) == 0.0 and p(0.5) == 0.0          # standing
    assert p(1.5) == pytest.approx(0.5)              # halfway down
    assert p(2.5) == 1.0 and p(2.7) == 1.0           # bottom hold
    assert p(2.8 + 0.75) == pytest.approx(0.5)       # halfway up
    assert p(4.3) == 0.0 and p(4.7) == 0.0           # standing again


def test_lead_reaches_peak_early_and_mirrors_on_ascent():
    timing = RepTiming(stand_before=0.0, descent=2.0, ascent=2.0, stand_after=0.0)
    lead = 0.8
    assert timing.lagged_profile(1.6, lead) == pytest.approx(1.0)
    assert timing.lagged_profile(1.6, 1.0) < 1.0
    # time reverse: still at the peak until 1 - lead of the ascent
    assert timing.lagged_profile(2.0 + 0.4, lead) == pytest.approx(1.0)
    t = np.linspace(0, 4, 401)
    np.testing.assert_allclose(timing.lagged_profile(t, lead),
                               timing.lagged_profile(4 - t, lead), atol=1e-12)


def _derivatives(timing, t, h=1e-5):
    p = timing.profile
    v = (p(t + h) - p(t - h)) / (2 * h)
    a = (p(t + h) - 2 * p(t) + p(t - h)) / h**2
    return v, a


def test_continuous_rep_turns_around_without_stopping_acceleration():
    timing = RepTiming(stand_before=0.5, descent=2.0, bottom_hold=0.0, ascent=1.6)
    bottom = timing.ascent_start
    assert timing.profile(bottom) == pytest.approx(1.0)
    v, a = _derivatives(timing, np.array([bottom - 1e-3, bottom + 1e-3]))
    np.testing.assert_allclose(v, 0.0, atol=1e-2)
    # acceleration at the bottom is negative (pushing back up) and the
    # same from both sides
    assert a[0] < -0.5 and a[0] == pytest.approx(a[1], rel=0.01)
    assert timing.bottom_acceleration() == pytest.approx(a[0], rel=0.01)
    # still starts and ends at rest with zero acceleration
    for edge in (timing.descent_start + 1e-3, timing.ascent_start + timing.ascent - 1e-3):
        v, a = _derivatives(timing, np.array([edge]))
        assert abs(v[0]) < 1e-2 and abs(a[0]) < 0.1


def test_continuous_profile_stays_between_standing_and_bottom():
    timing = RepTiming(descent=1.3, ascent=2.1)
    p = timing.profile(timing.times(500))
    assert p.min() >= -1e-9 and p.max() <= 1 + 1e-9
    down = p[(timing.times(500) <= timing.ascent_start)]
    assert np.all(np.diff(down) >= -1e-12)


def test_bottom_acceleration_minimises_jerk():
    from exercise_quality.core.timing import _jerk_cost, _quintic

    timing = RepTiming(descent=2.0, ascent=1.5)
    best = timing.bottom_acceleration()

    def cost(acc):
        return (_jerk_cost(_quintic(0, 0, 0, 1, 0, acc, 2.0), 2.0)
                + _jerk_cost(_quintic(1, 0, acc, 0, 0, 0, 1.5), 1.5))

    assert cost(best) < cost(best + 0.05) and cost(best) < cost(best - 0.05)


def test_jerk_cost_matches_numeric_integral():
    from exercise_quality.core.timing import _jerk_cost, _quintic

    coef = _quintic(0, 0, 0, 1, 0, -2.0, 1.7)
    t = np.linspace(0, 1.7, 20001)
    jerk = 6 * coef[3] + 24 * coef[4] * t + 60 * coef[5] * t**2
    assert _jerk_cost(coef, 1.7) == pytest.approx(np.trapezoid(jerk**2, t), rel=1e-6)


def test_pause_at_bottom_uses_separate_movements():
    timing = RepTiming(bottom_hold=1.0)
    assert timing.bottom_acceleration() == 0.0
    t = timing.times(100)
    np.testing.assert_allclose(timing.profile(t), timing.lagged_profile(t, 1.0))


def test_times_cover_the_rep():
    timing = RepTiming()
    t = timing.times(100)
    assert t[0] == 0.0
    assert t[-1] == pytest.approx(timing.total)
    assert np.allclose(np.diff(t), 0.01)


@pytest.mark.parametrize("kwargs", [{"descent": 0}, {"ascent": -1}, {"bottom_hold": -0.1}])
def test_invalid_timing_rejected(kwargs):
    with pytest.raises(ValueError):
        RepTiming(**kwargs)


def test_invalid_lead_rejected():
    with pytest.raises(ValueError):
        RepTiming().lagged_profile(1.0, lead=0.0)
