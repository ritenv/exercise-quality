"""The balance-constrained squat generator."""

import dataclasses

import numpy as np
import pytest

from exercise_quality.core import deleva
from exercise_quality.core.sagittal_body import BodyPopulation
from exercise_quality.core.sagittal_model import SEGMENT_ENDPOINTS, segment_lengths
from exercise_quality.core.timing import RepTiming
from exercise_quality.exercises.squat.balanced import (
    ARMS_FORWARD,
    ARMS_HANGING,
    JointLimits,
    SquatStyle,
    generate_squat,
)


@pytest.fixture(scope="module")
def population():
    return BodyPopulation.load()


@pytest.fixture(scope="module", params=deleva.SEXES)
def body(request, population):
    return population.median_body(request.param)


@pytest.fixture(scope="module")
def rep(body):
    return generate_squat(body, SquatStyle())


def peak_trunk(seq):
    return float(np.max(seq.trunk_inclination))


def test_default_rep_is_feasible(rep):
    assert rep.is_feasible, rep.flags


def test_com_stays_on_target_every_frame(rep):
    assert np.max(np.abs(rep.balance_error)) < 1e-9
    assert np.all(rep.support.contains(rep.com[:, 0]))


def test_starts_and_ends_standing(rep):
    for i in (0, -1):
        a = {k: v[i] for k, v in rep.measured().items()}
        assert a["knee_flexion"] == pytest.approx(0.0, abs=1e-9)
        assert a["hip_flexion"] == pytest.approx(0.0, abs=1e-6)
        assert abs(a["ankle_dorsiflexion"]) < 6.0  # small standing lean


def test_reaches_requested_depth(rep):
    assert np.max(rep.angles.knee_flexion) == pytest.approx(rep.style.peak_knee_flexion)


def test_segment_lengths_preserved(rep):
    lengths = segment_lengths(rep.body)
    for seg, (a, c) in SEGMENT_ENDPOINTS.items():
        d = np.linalg.norm(rep.landmarks[c] - rep.landmarks[a], axis=-1)
        np.testing.assert_allclose(d, lengths[seg], atol=1e-9)


def test_motion_is_continuous(rep):
    # at 100 fps nothing should jump; peak joint speeds in a 2 s squat are
    # well under 150 deg/s, i.e. 1.5 deg per frame
    for name, values in rep.measured().items():
        assert np.max(np.abs(np.diff(values))) < 1.5, name


def test_rep_is_time_symmetric_with_equal_phases(rep):
    t = rep.style.timing
    assert t.descent == t.ascent and t.stand_before == t.stand_after
    np.testing.assert_allclose(rep.trunk_inclination, rep.trunk_inclination[::-1], atol=1e-7)


def test_deterministic(body):
    a = generate_squat(body, SquatStyle())
    b = generate_squat(body, SquatStyle())
    np.testing.assert_array_equal(a.com, b.com)


def test_longer_thigh_needs_more_trunk_lean(body):
    """The central fairness point of the project: longer femurs push the
    hips further back, so balance demands more forward lean."""
    base = generate_squat(body, SquatStyle())
    long_thigh = dataclasses.replace(body, thigh=body.thigh * 1.12)
    longer = generate_squat(long_thigh, SquatStyle())
    assert peak_trunk(longer) > peak_trunk(base) + 2.0


def test_limited_ankle_needs_more_trunk_lean(body):
    free = generate_squat(body, SquatStyle())
    stiff = generate_squat(body, SquatStyle(limits=JointLimits(ankle_dorsiflexion=22.0)))
    # running out of ankle is a note, not a failure in itself
    assert "ankle_at_limit" in stiff.notes
    assert "ankle_at_limit" not in stiff.flags
    assert np.max(stiff.angles.ankle_dorsiflexion) == pytest.approx(22.0)
    assert peak_trunk(stiff) > peak_trunk(free) + 5.0


def test_arms_forward_act_as_counterweight(body):
    hanging = generate_squat(body, SquatStyle(arms=ARMS_HANGING))
    forward = generate_squat(body, SquatStyle(arms=ARMS_FORWARD))
    assert peak_trunk(forward) < peak_trunk(hanging)
    # world-fixed arms stay parallel to the floor while the trunk moves
    arm = forward.landmarks["elbow"] - forward.landmarks["shoulder"]
    np.testing.assert_allclose(arm[:, 1], 0.0, atol=1e-9)


def test_com_further_forward_needs_more_lean(body):
    back = generate_squat(body, SquatStyle(com_target_bottom=0.30))
    front = generate_squat(body, SquatStyle(com_target_bottom=0.45))
    assert peak_trunk(front) > peak_trunk(back)


def test_knee_limit_is_flagged(body):
    seq = generate_squat(body, SquatStyle(peak_knee_flexion=165.0))
    assert "knee_beyond_limit" in seq.flags


def test_hip_limit_is_flagged(body):
    seq = generate_squat(body, SquatStyle(limits=JointLimits(hip_flexion=80.0)))
    assert "hip_beyond_limit" in seq.flags


def test_target_off_the_foot_is_flagged(body):
    seq = generate_squat(body, SquatStyle(com_target_bottom=1.2, com_target_shift=0.0))
    assert "com_target_outside_support" in seq.flags


def test_unreachable_balance_is_flagged(body):
    """Deep squat with no ankle bend while keeping the COM near the toes:
    the trunk would have to fold past horizontal and still not get there."""
    seq = generate_squat(body, SquatStyle(
        peak_knee_flexion=150.0, ankle_knee_ratio=0.0,
        com_target_bottom=0.95, com_target_shift=0.0))
    assert "balance_unreachable" in seq.flags or "hip_beyond_limit" in seq.flags
    assert not seq.is_feasible


def test_tempo_does_not_change_quasi_static_path(body):
    """Balance is solved per frame from geometry, so a slower rep traces
    the same poses, only more slowly."""
    fast = generate_squat(body, SquatStyle(timing=RepTiming(descent=1.0, ascent=1.0)))
    slow = generate_squat(body, SquatStyle(timing=RepTiming(descent=3.0, ascent=3.0)))
    def at_bottom(seq):
        return seq.trunk_inclination[np.argmax(seq.angles.knee_flexion)]

    assert at_bottom(fast) == pytest.approx(at_bottom(slow), abs=1e-6)
    assert peak_trunk(fast) == pytest.approx(peak_trunk(slow), abs=0.01)


def test_many_real_bodies_are_feasible(population):
    rng = np.random.default_rng(11)
    failures = []
    for _ in range(60):
        b = population.sample(rng)
        seq = generate_squat(b, SquatStyle(), fps=30)
        if not seq.is_feasible:
            failures.append((b.subject_id, seq.flags))
    assert not failures
