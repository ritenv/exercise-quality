"""Checks on the de Leva tables and the ANSUR II based bodies."""

import numpy as np
import pytest

from exercise_quality.core import deleva
from exercise_quality.core.sagittal_body import (
    BodyPopulation,
    SagittalBody,
    body_from_measurements,
)


@pytest.fixture(scope="module")
def population():
    return BodyPopulation.load()


# --- de Leva tables ---------------------------------------------------------

@pytest.mark.parametrize("sex", deleva.SEXES)
def test_mass_fractions_sum_to_one(sex):
    # de Leva's own table sums to 99.99% (female) and 100.00% (male)
    assert deleva.total_mass_fraction(sex) == pytest.approx(1.0, abs=2e-4)


@pytest.mark.parametrize("sex", deleva.SEXES)
def test_segment_parameters_are_physical(sex):
    for p in deleva.segment_parameters(sex).values():
        assert 0 < p.mass_fraction < 0.5
        assert 0 < p.com_fraction < 1
        assert 0 < p.sagittal_gyration_fraction < 1
        assert 0.05 < p.reference_length_m < 0.7


@pytest.mark.parametrize("sex", deleva.SEXES)
def test_table2_offsets_reproduce_table4_forearm(sex):
    """Table 1 radiale-stylion plus the Table 2 offsets should give the
    Table 4 forearm length (EJC-WJC) to within a millimetre or so.
    This guards against sign mistakes in the offsets."""
    radiale_stylion = {"female": 0.2471, "male": 0.2513}[sex]
    off = deleva.JOINT_CENTRE_OFFSETS_M[sex]
    derived = (radiale_stylion + off["elbow_from_radiale_forearm"]
               - off["wrist_from_stylion"])
    table4 = deleva.segment_parameters(sex)["forearm"].reference_length_m
    assert derived == pytest.approx(table4, abs=0.0015)


@pytest.mark.parametrize("sex", deleva.SEXES)
def test_table2_offsets_reproduce_table4_upper_arm(sex):
    acromion_radiale = {"female": 0.3227, "male": 0.3305}[sex]
    off = deleva.JOINT_CENTRE_OFFSETS_M[sex]
    derived = (acromion_radiale + off["shoulder_from_acromion"]
               - off["elbow_from_radiale_upper_arm"])
    table4 = deleva.segment_parameters(sex)["upper_arm"].reference_length_m
    assert derived == pytest.approx(table4, abs=0.0005)


@pytest.mark.parametrize("sex", deleva.SEXES)
def test_trunk_split_conserves_mass(sex):
    split = deleva.split_trunk_parameters(sex)
    whole = deleva.segment_parameters(sex)["trunk"]
    total = split["pelvis"].mass_fraction + split["thorax"].mass_fraction
    assert total == pytest.approx(whole.mass_fraction, abs=1e-9)
    total_len = split["pelvis"].reference_length_m + split["thorax"].reference_length_m
    assert total_len == pytest.approx(whole.reference_length_m)


@pytest.mark.parametrize("sex", deleva.SEXES)
def test_trunk_split_reproduces_whole_trunk_centre_of_mass(sex):
    """The three de Leva sub-segments were measured separately from the
    whole trunk. Stacking them should land the combined centre of mass
    close to the whole-trunk row (CERV-MIDH). This is an independent check
    of the transcription and of the stacking assumption."""
    split = deleva.split_trunk_parameters(sex)
    whole = deleva.segment_parameters(sex)["trunk"]
    p, t = split["pelvis"], split["thorax"]
    # positions below CERV, metres
    thorax_com = t.com_fraction * t.reference_length_m
    # de Leva measures the pelvis centre from its origin, the navel (OMPH)
    pelvis_com = t.reference_length_m + p.com_fraction * p.reference_length_m
    combined = ((t.mass_fraction * thorax_com + p.mass_fraction * pelvis_com)
                / (t.mass_fraction + p.mass_fraction))
    expected = whole.com_fraction * whole.reference_length_m
    assert combined == pytest.approx(expected, abs=0.004)  # within 4 mm


@pytest.mark.parametrize("sex", deleva.SEXES)
def test_thorax_parameters_are_physical(sex):
    t = deleva.split_trunk_parameters(sex)["thorax"]
    assert 0.3 < t.com_fraction < 0.6
    assert 0.2 < t.sagittal_gyration_fraction < 0.5


def test_unknown_sex_rejected():
    with pytest.raises(ValueError):
        deleva.segment_parameters("other")


# --- ANSUR II bodies ---------------------------------------------------------

def test_population_size(population):
    assert population.count("male") == 4082
    assert population.count("female") == 1986


@pytest.mark.parametrize("sex", deleva.SEXES)
def test_every_body_is_valid_and_closes_to_stature(population, sex):
    for body in population.bodies(sex):
        stacked = body.ankle_height + body.shank + body.thigh + body.trunk + body.head
        assert stacked == pytest.approx(body.stature, abs=1e-9)
        assert body.shoulder_drop < body.trunk
        assert body.heel_offset < body.toe_offset


@pytest.mark.parametrize("sex", deleva.SEXES)
def test_lengths_agree_with_de_leva_after_scaling(population, sex):
    """Mean ANSUR-derived lengths should be close to de Leva's means scaled
    by stature. The two samples are different populations measured with
    different methods, so only broad agreement is expected. de Leva's
    female thigh is unusually short (15 subjects), hence the wider bound."""
    bodies = population.bodies(sex)
    stature = np.mean([b.stature for b in bodies])
    scale = stature / deleva.REFERENCE_SUBJECT[sex]["stature_m"]
    tolerance = {"shank": 0.05, "thigh": 0.12, "trunk": 0.06, "head": 0.05,
                 "upper_arm": 0.05, "forearm": 0.07, "foot": 0.16}
    for name, tol in tolerance.items():
        mean_length = np.mean([b.segment_length(
            "foot_length" if name == "foot" else name) for b in bodies])
        expected = deleva.segment_parameters(sex)[name].reference_length_m * scale
        assert abs(mean_length / expected - 1) < tol, (name, mean_length, expected)


@pytest.mark.parametrize("sex", deleva.SEXES)
def test_proportions_are_plausible(population, sex):
    for body in population.bodies(sex):
        h = body.stature
        assert 0.025 < body.ankle_height / h < 0.06
        assert 0.40 < body.standing_hip_height / h < 0.60
        assert 0.12 < body.foot_length / h < 0.19
        assert 0.10 < body.upper_arm / h < 0.22


@pytest.mark.parametrize("sex", deleva.SEXES)
def test_pelvis_and_thorax_lengths(population, sex):
    bodies = population.bodies(sex)
    for b in bodies:
        assert b.trunk == pytest.approx(b.pelvis + b.thorax)
        assert 0.03 < b.pelvis < 0.30
        assert b.shoulder_drop < b.thorax
    # male navel split agrees with de Leva's lower trunk once scaled; the
    # female one is shorter than de Leva's 15-woman sample suggests
    stature = np.mean([b.stature for b in bodies])
    scale = stature / deleva.REFERENCE_SUBJECT[sex]["stature_m"]
    expected = deleva.split_trunk_parameters(sex)["pelvis"].reference_length_m * scale
    tolerance = {"male": 0.06, "female": 0.25}[sex]
    assert abs(np.mean([b.pelvis for b in bodies]) / expected - 1) < tolerance


def test_sampling_is_reproducible(population):
    a = [population.sample(np.random.default_rng(7)) for _ in range(1)]
    b = [population.sample(np.random.default_rng(7)) for _ in range(1)]
    assert a == b
    rng = np.random.default_rng(0)
    sexes = {population.sample(rng).sex for _ in range(200)}
    assert sexes == {"male", "female"}


def test_sampled_bodies_are_real_people(population):
    rng = np.random.default_rng(1)
    for _ in range(20):
        body = population.sample(rng, sex="female")
        assert body.sex == "female"
        assert body.subject_id


def test_median_body(population):
    body = population.median_body("male")
    assert 1.70 < body.stature < 1.82
    assert body.subject_id == "median-male"


def test_invalid_bodies_rejected(population):
    good = population.median_measurements("male")
    with pytest.raises(ValueError):
        body_from_measurements({**good, "foot_length_m": -0.2}, "male")
    with pytest.raises(ValueError):
        # hip below the knee gives a negative thigh
        body_from_measurements({**good, "trochanterion_height_m": 0.3}, "male")
    with pytest.raises(ValueError):
        SagittalBody(**{**vars(population.median_body("male")), "sex": "robot"})
