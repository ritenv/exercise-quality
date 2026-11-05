"""Dynamic centre of pressure."""

import numpy as np
import pytest

from exercise_quality.core.dynamics import GRAVITY, analyse_dynamics, centre_of_pressure
from exercise_quality.core.sagittal_body import BodyPopulation
from exercise_quality.core.timing import RepTiming
from exercise_quality.exercises.squat.balanced import SquatStyle, generate_squat

DT = 0.01
T = np.arange(0, 2.0 + 1e-9, DT)


@pytest.fixture(scope="module")
def body():
    return BodyPopulation.load().median_body("male")


def single(centre, angle=None, inertia=0.0):
    centres = centre[:, None, :]
    angles = np.zeros((len(T), 1)) if angle is None else angle[:, None]
    return centre_of_pressure([70.0], centres, angles, np.array([inertia]), DT)


def test_still_mass_presses_straight_down():
    centre = np.tile([0.05, 1.0], (len(T), 1))
    np.testing.assert_allclose(single(centre), 0.05, atol=1e-12)


def test_horizontal_acceleration_shifts_cop_backwards():
    """Accelerating forward at a needs a push behind the COM: p = x - h a / g."""
    a, h = 0.8, 0.9
    x = 0.5 * a * T**2
    centre = np.stack([x, np.full_like(T, h)], axis=1)
    cop = single(centre)
    np.testing.assert_allclose(cop[2:-2], x[2:-2] - h * a / GRAVITY, atol=1e-9)


def test_vertical_acceleration_does_not_move_a_centred_cop():
    y = 1.0 + 0.1 * np.sin(2 * np.pi * T)
    centre = np.stack([np.full_like(T, 0.03), y], axis=1)
    np.testing.assert_allclose(single(centre), 0.03, atol=1e-9)


def test_spinning_segment_shifts_cop_by_i_alpha_over_mg():
    """A segment spinning up counter-clockwise about a fixed centre needs
    the ground force in front of it: p = I alpha / (m g)."""
    alpha, inertia = 2.0, 1.5
    angle = 0.5 * alpha * T**2
    centre = np.tile([0.0, 1.0], (len(T), 1))
    cop = single(centre, angle, inertia)
    np.testing.assert_allclose(cop[2:-2], inertia * alpha / (70.0 * GRAVITY), atol=1e-9)


def test_cop_equals_com_when_standing_still(body):
    seq = generate_squat(body, SquatStyle())
    dyn = analyse_dynamics(seq)
    np.testing.assert_allclose(dyn.cop_x[:20], dyn.com_x[:20], atol=1e-9)
    np.testing.assert_allclose(dyn.vertical_force[:20], 1.0, atol=1e-9)


def test_normal_tempo_squat_stays_on_the_foot(body):
    dyn = analyse_dynamics(generate_squat(body, SquatStyle()))
    assert dyn.stays_on_foot
    # at 2 s per phase the quasi-static plan is close to the real COP
    assert dyn.max_cop_offset < 0.02
    # the floor never has to pull; vertical force stays near body weight
    assert 0.7 < dyn.vertical_force.min() and dyn.vertical_force.max() < 1.3


def test_ground_force_peaks_through_a_continuous_bottom(body):
    """Without a pause, the push back up starts before the body stops, so
    the vertical ground force is above body weight at the bottom instead
    of dipping back to 1."""
    seq = generate_squat(body, SquatStyle(timing=RepTiming(bottom_hold=0.0)))
    dyn = analyse_dynamics(seq)
    bottom = int(np.argmax(seq.angles.knee_flexion))
    assert dyn.vertical_force[bottom] > 1.05
    window = dyn.vertical_force[bottom - 30: bottom + 31]
    assert dyn.vertical_force[bottom] >= window.max() - 0.01


def test_paused_bottom_returns_to_body_weight(body):
    seq = generate_squat(body, SquatStyle(timing=RepTiming(bottom_hold=1.0)))
    dyn = analyse_dynamics(seq)
    t = seq.style.timing
    holding = (seq.times > t.ascent_start - 0.8) & (seq.times < t.ascent_start - 0.2)
    np.testing.assert_allclose(dyn.vertical_force[holding], 1.0, atol=1e-6)


def test_faster_reps_move_the_cop_further(body):
    offsets = []
    for duration in (3.0, 2.0, 1.0, 0.6):
        timing = RepTiming(descent=duration, ascent=duration)
        dyn = analyse_dynamics(generate_squat(body, SquatStyle(timing=timing)))
        offsets.append(dyn.max_cop_offset)
    assert offsets == sorted(offsets)


def test_slow_reps_scale_with_inverse_square_of_duration(body):
    """Halving the duration quadruples every acceleration. When the
    accelerations are small next to g the COP offset is linear in them,
    so it should quadruple too. (At fast tempos the vertical acceleration
    in the denominator g + ydd breaks this, which is expected.)"""
    def offset(duration):
        timing = RepTiming(descent=duration, ascent=duration)
        return analyse_dynamics(generate_squat(body, SquatStyle(timing=timing))).max_cop_offset

    assert offset(4.0) / offset(8.0) == pytest.approx(4.0, rel=0.1)


def test_cop_is_independent_of_sampling_rate(body):
    a = analyse_dynamics(generate_squat(body, SquatStyle(), fps=100))
    b = analyse_dynamics(generate_squat(body, SquatStyle(), fps=200))
    assert a.max_cop_offset == pytest.approx(b.max_cop_offset, rel=0.02)
