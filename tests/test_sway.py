"""Postural sway."""

import dataclasses

import numpy as np
import pytest

from exercise_quality.core.dynamics import analyse_dynamics
from exercise_quality.core.sagittal_body import BodyPopulation
from exercise_quality.core.sway import (
    COP_MEAN_VELOCITY_MM_S,
    COP_MEDIAN_FREQUENCY_HZ,
    cop_mean_velocity,
    median_frequency,
    sway_noise,
)
from exercise_quality.core.timing import RepTiming
from exercise_quality.exercises.squat import protocols as p
from exercise_quality.exercises.squat.balanced import SquatStyle, generate_squat

FPS = 60.0


def test_noise_has_requested_median_frequency_and_unit_sd():
    rng = np.random.default_rng(0)
    medians = []
    for _ in range(20):
        x = sway_noise(int(60 * FPS), FPS, rng, 0.33)
        assert x.std() == pytest.approx(1.0)
        assert abs(x.mean()) < 1e-9
        medians.append(median_frequency(x, FPS))
    assert np.mean(medians) == pytest.approx(0.33, abs=0.04)


def test_pendulum_filter_smooths_the_com():
    rng_a, rng_b = np.random.default_rng(1), np.random.default_rng(1)
    cop = sway_noise(2000, FPS, rng_a, 0.33)
    com = sway_noise(2000, FPS, rng_b, 0.33, pendulum_height=0.95)
    assert com.std() < cop.std()
    assert median_frequency(com, FPS) < median_frequency(cop, FPS)


@pytest.fixture(scope="module")
def quiet_stance():
    """30 s of standing before a small knee bend, for several people."""
    population = BodyPopulation.load()
    rng = np.random.default_rng(2)
    out = []
    for i in range(8):
        body = population.sample(rng)
        style = SquatStyle(
            peak_knee_flexion=60.0,
            timing=RepTiming(stand_before=30.0, descent=1.0, ascent=1.0, stand_after=0.5),
            sway_amplitude=p.SWAY_AMPLITUDE_M, sway_median_hz=COP_MEDIAN_FREQUENCY_HZ, sway_seed=i)
        seq = generate_squat(body, style, fps=FPS)
        dyn = analyse_dynamics(seq)
        window = (seq.times > 1.0) & (seq.times < 29.0)
        out.append((seq, dyn.cop_x[window]))
    return out


def test_quiet_stance_matches_yamamoto(quiet_stance):
    """Dynamic COP during quiet standing has Yamamoto et al.'s mean
    velocity (6.3 mm/s) and median frequency (0.33 Hz)."""
    velocity = np.mean([cop_mean_velocity(c, FPS) for _, c in quiet_stance])
    median = np.mean([median_frequency(c, FPS) for _, c in quiet_stance])
    assert velocity == pytest.approx(COP_MEAN_VELOCITY_MM_S, rel=0.15)
    assert median == pytest.approx(COP_MEDIAN_FREQUENCY_HZ, abs=0.07)


def test_sway_is_small_and_people_stay_balanced(quiet_stance):
    for seq, cop in quiet_stance:
        assert cop.std() < 0.006            # a few millimetres
        assert seq.is_feasible
        assert np.max(np.abs(seq.balance_error)) < 1e-9


def test_no_sway_by_default_and_reproducible_with_seed():
    body = BodyPopulation.load().median_body("female")
    still = generate_squat(body, SquatStyle())
    np.testing.assert_allclose(still.measured()["knee_flexion"][:30], 0.0, atol=1e-12)
    a = generate_squat(body, SquatStyle(sway_amplitude=0.002, sway_seed=4))
    b = generate_squat(body, SquatStyle(sway_amplitude=0.002, sway_seed=4))
    c = generate_squat(body, SquatStyle(sway_amplitude=0.002, sway_seed=5))
    np.testing.assert_array_equal(a.com, b.com)
    assert not np.allclose(a.com, c.com)
    # standing frames now move a little
    assert np.ptp(a.com[:40, 0]) > 1e-4


def test_sampled_styles_include_sway():
    body = BodyPopulation.load().median_body("male")
    rng = np.random.default_rng(3)
    styles = [p.sample_style(rng, body) for _ in range(500)]
    amps = np.array([s.sway_amplitude for s in styles])
    assert np.median(amps) == pytest.approx(p.SWAY_AMPLITUDE_M, rel=0.05)
    assert len({s.sway_seed for s in styles}) == 500
