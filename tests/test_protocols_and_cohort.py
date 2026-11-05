"""Literature-based sampling and the cohort comparison."""

import numpy as np
import pytest

from exercise_quality.core.sagittal_body import BodyPopulation
from exercise_quality.exercises.squat import literature as lit
from exercise_quality.exercises.squat import protocols as p
from exercise_quality.exercises.squat.balanced import ARMS_FORWARD, generate_squat
from exercise_quality.exercises.squat.cohort import compare, rep_metrics, simulate_cohort


@pytest.fixture(scope="module")
def population():
    return BodyPopulation.load()


@pytest.fixture(scope="module")
def ishida_cohort(population):
    return simulate_cohort(population, p.ISHIDA_PARALLEL, n=60, seed=3, sex="male")


def test_sampling_is_reproducible(population):
    body = population.median_body("male")
    a = p.sample_style(np.random.default_rng(1), body)
    b = p.sample_style(np.random.default_rng(1), body)
    assert a == b


def test_sampled_styles_stay_in_range(population):
    rng = np.random.default_rng(2)
    for _ in range(300):
        body = population.sample(rng)
        s = p.sample_style(rng, body, p.ZAWADKA_MAX_DEPTH)
        assert 60 <= s.peak_knee_flexion <= s.limits.knee_flexion
        assert 0.2 <= s.com_target_bottom <= 0.6
        assert 0.2 <= s.com_target_start <= 0.65
        assert 1.0 <= s.ankle_coupling <= 1.6
        assert s.arms == ARMS_FORWARD
        assert s.timing.bottom_hold == 3.0


def test_ankle_and_com_shift_are_correlated_like_kasahara(population):
    rng = np.random.default_rng(4)
    body = population.median_body("female")
    styles = [p.sample_style(rng, body) for _ in range(4000)]
    ratio = np.array([s.ankle_knee_ratio for s in styles])
    shift = np.array([s.com_target_shift for s in styles])
    r = np.corrcoef(ratio, shift)[0, 1]
    assert r == pytest.approx(p.ANKLE_SHIFT_CORRELATION, abs=0.05)


def test_rep_metrics_on_a_known_rep(population):
    seq = generate_squat(population.median_body("male"))
    m = rep_metrics(seq)
    assert m["knee_peak"] == pytest.approx(seq.style.peak_knee_flexion, abs=0.01)
    assert m["knee_rom"] == pytest.approx(m["knee_peak"], abs=0.01)
    assert m["ankle_at_knee60"] < m["ankle_at_knee90"] < m["ankle_peak"]
    assert 0 < m["com_descent_pct_height"] < 40
    assert m["feasible"]


def test_most_simulated_reps_are_feasible(ishida_cohort):
    assert ishida_cohort.feasible_share >= 0.9


def test_ishida_cohort_matches_published_means(ishida_cohort):
    """Regression guard on the Ishida comparison. Rows that match are held
    to |z| < 1. Known gaps (docs/validation.md) get explicit bounds so
    they cannot get worse unnoticed:
      * peak trunk lean overshoots (people with stiff ankles lean too far)
      * mean depth is low because those same people are depth limited
    """
    refs = {k: v for k, v in lit.ISHIDA_2025.items() if k != "ankle_peak_cued_forward"}
    rows = {r.metric: r for r in compare(ishida_cohort, refs)}
    for name in ("ankle_at_knee60", "ankle_at_knee90", "ankle_peak",
                 "trunk_at_knee60", "trunk_at_knee90", "cop_fraction"):
        assert abs(rows[name].z) < 1.0, (name, rows[name])
    assert 0.0 < rows["trunk_peak"].z < 2.0
    assert -2.0 < rows["knee_peak"].z < 0.5
    assert ishida_cohort.depth_limited_share < 0.35


def test_incomparable_metrics_have_no_z(ishida_cohort):
    refs = {"knee_peak": lit.ISHIDA_2025["knee_peak"]}
    row = compare(ishida_cohort, refs, incomparable={"knee_peak": "test"})[0]
    assert not row.comparable and np.isnan(row.z)
