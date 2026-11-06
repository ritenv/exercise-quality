"""Cohort-level realism guards.

These catch artefacts that individual unit tests miss but that show up
when you watch the animations: joints hitting their limits like a wall,
spikes in the ground force, people leaning implausibly far. Bounds are
set from what the current model achieves with some headroom, and each
comment says what the bound protects against.
"""

import numpy as np
import pytest

from exercise_quality.core.dynamics import analyse_dynamics
from exercise_quality.core.sagittal_body import BodyPopulation
from exercise_quality.exercises.squat import protocols as p
from exercise_quality.exercises.squat.balanced import generate_feasible_squat

DT = 0.01


@pytest.fixture(scope="module")
def reps():
    population = BodyPopulation.load()
    rng = np.random.default_rng(21)
    out = []
    for i in range(24):
        body = population.sample(rng)
        protocol = p.ZAWADKA_MAX_DEPTH if i % 2 else p.ISHIDA_PARALLEL
        seq = generate_feasible_squat(body, p.sample_style(rng, body, protocol), fps=1 / DT)
        out.append((seq, seq.measured(), analyse_dynamics(seq)))
    return out


def peak_acceleration(values):
    return np.abs(np.gradient(np.gradient(values, DT), DT)).max()


def test_hip_does_not_hit_end_range_like_a_wall(reps):
    """With an abrupt end range the hip's angular acceleration was ~4x the
    knee's (median). A gradual end range keeps them comparable."""
    ratios = [peak_acceleration(m["hip_flexion"]) / peak_acceleration(m["knee_flexion"])
              for _, m, _ in reps]
    assert np.median(ratios) < 2.2


def test_ground_force_has_no_spikes(reps):
    """Force jerk at these tempos stayed under ~0.8 body weights per second
    for 95% of reps once the end range and trunk curve were smoothed; it
    was 2.9 before."""
    jerks = [np.abs(np.gradient(d.vertical_force, DT)).max() for _, _, d in reps]
    assert np.percentile(jerks, 95) < 1.2


def test_ground_force_stays_near_body_weight_at_normal_tempo(reps):
    for seq, _, d in reps:
        assert 0.75 < d.vertical_force.min() and d.vertical_force.max() < 1.35


def test_everyone_stays_on_their_feet(reps):
    for seq, _, d in reps:
        assert seq.is_feasible, seq.flags
        assert d.stays_on_foot


def test_no_implausible_trunk_lean(reps):
    """Before ankle and trunk shared the balance correction, some people
    leant 70-80 deg at parallel depth."""
    peaks = [m["trunk_inclination"].max() for _, m, _ in reps]
    assert max(peaks) < 70.0
    assert 30.0 < np.median(peaks) < 50.0


def test_trunk_does_not_get_stuck(reps):
    """The bug the first video showed: unsolvable frames were filled with a
    constant trunk angle. A moving squat never holds the trunk exactly
    still while the knee is bending."""
    for seq, m, _ in reps:
        t = seq.style.timing
        descending = (seq.times > t.descent_start + 0.2) & (seq.times < t.descent_start + 0.8 * t.descent)
        assert np.all(np.abs(np.diff(m["trunk_inclination"][descending])) > 1e-6)
