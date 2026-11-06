"""Segment inertia parameters from de Leva (1996).

Source
------
de Leva P. "Adjustments to Zatsiorsky-Seluyanov's segment inertia
parameters." Journal of Biomechanics 29(9):1223-1230, 1996.

Values below are from Table 4 (joint-centre endpoints) and Table 2
(joint-centre offsets from palpable landmarks). They were transcribed
from two independent secondary copies that agree with each other:

* BMClab/BMC repository, data/BSP_ZdeLeva.txt (Duarte, Biomechanics and
  Motor Control Lab), https://github.com/BMClab/BMC
* samchon/automovie, packages/human/docs/sources/de-leva-1996.md, which
  was read from a scan of the original paper.

The underlying sample is 100 men and 15 women, young Russian physical
education students, scanned with gamma rays by Zatsiorsky et al. Mass
fractions do not follow age, adiposity or other populations.

Conventions
-----------
* ``mass_fraction``: segment mass / whole body mass, for ONE segment
  (one thigh, one foot...). Bilateral segments appear twice in a body.
* ``com_fraction``: position of the segment centre of mass along the
  line from ``origin`` to ``other``, as a fraction of that distance.
  0 means at the origin landmark.
* ``sagittal_gyration_fraction``: radius of gyration about the
  medio-lateral axis (rotation in the side view), as a fraction of the
  segment length.
* ``reference_length_m``: mean segment length in the de Leva sample.

Landmarks: VERT vertex, CERV cervicale (C7), MIDH mid point between the
hip joint centres, SJC/EJC/WJC/HJC/KJC shoulder, elbow, wrist, hip and
knee joint centres, MET3 third metacarpale, LMAL lateral malleolus,
HEEL pternion, TTIP tip of the longest toe.
"""

from dataclasses import dataclass

SEXES = ("female", "male")


@dataclass(frozen=True)
class SegmentParameters:
    name: str
    origin: str
    other: str
    mass_fraction: float
    com_fraction: float
    sagittal_gyration_fraction: float
    reference_length_m: float


def _segments(rows):
    return {
        name: SegmentParameters(name, origin, other, mass / 100.0,
                                com / 100.0, gyration / 100.0, length / 1000.0)
        for name, origin, other, length, mass, com, gyration in rows
    }


# name, origin, other, length (mm), mass (%), CM from origin (%), sagittal r (%)
# Variants chosen: head VERT-CERV and trunk CERV-MIDH (the "alternative
# endpoint" rows) so the head and trunk chain end to end at C7, and shank
# KJC-LMAL so the ankle point matches the lateral malleolus height
# measured in ANSUR II. The whole trunk row is kept for reference; the
# body model uses it split into thorax, abdomen and pelvis (see SEGMENT_TABLE).
TABLE4 = {
    "female": _segments([
        ("head", "VERT", "CERV", 243.7, 6.68, 48.41, 27.1),
        ("trunk", "CERV", "MIDH", 614.8, 42.57, 49.64, 30.7),
        ("upper_arm", "SJC", "EJC", 275.1, 2.55, 57.54, 27.8),
        ("forearm", "EJC", "WJC", 264.3, 1.38, 45.59, 26.1),
        ("hand", "WJC", "MET3", 78.0, 0.56, 74.74, 53.1),
        ("thigh", "HJC", "KJC", 368.5, 14.78, 36.12, 36.9),
        ("shank", "KJC", "LMAL", 432.3, 4.81, 44.16, 27.1),
        ("foot", "HEEL", "TTIP", 228.3, 1.29, 40.14, 29.9),
    ]),
    "male": _segments([
        ("head", "VERT", "CERV", 242.9, 6.94, 50.02, 30.3),
        ("trunk", "CERV", "MIDH", 603.3, 43.46, 51.38, 32.8),
        ("upper_arm", "SJC", "EJC", 281.7, 2.71, 57.72, 28.5),
        ("forearm", "EJC", "WJC", 268.9, 1.62, 45.74, 27.6),
        ("hand", "WJC", "MET3", 86.2, 0.61, 79.00, 62.8),
        ("thigh", "HJC", "KJC", 422.2, 14.16, 40.95, 32.9),
        ("shank", "KJC", "LMAL", 434.0, 4.33, 44.59, 25.5),
        ("foot", "HEEL", "TTIP", 258.1, 1.37, 44.15, 25.7),
    ]),
}

# Segments that exist on both sides of the body. The body model has a
# left and a right copy of each.
BILATERAL = frozenset({"upper_arm", "forearm", "hand", "thigh", "shank", "foot"})

REFERENCE_SUBJECT = {
    "female": {"stature_m": 1.735, "mass_kg": 61.9},
    "male": {"stature_m": 1.741, "mass_kg": 73.0},
}

# Table 2: distance from a palpable landmark to the joint centre along the
# segment, in metres, positive meaning the joint centre is proximal
# (towards the head) of the landmark.
JOINT_CENTRE_OFFSETS_M = {
    "female": {
        "shoulder_from_acromion": -0.0337,
        "elbow_from_radiale_upper_arm": 0.0139,
        "elbow_from_radiale_forearm": 0.0166,
        "wrist_from_stylion": -0.0015,
        "hip_from_trochanterion": 0.0028,
        "knee_from_tibiale": 0.0294,  # measured on trochanterion-tibiale
    },
    "male": {
        "shoulder_from_acromion": -0.0345,
        "elbow_from_radiale_upper_arm": 0.0143,
        "elbow_from_radiale_forearm": 0.0168,
        "wrist_from_stylion": -0.0015,
        "hip_from_trochanterion": 0.0032,
        "knee_from_tibiale": 0.0335,
    },
}

# Table 4 also gives the hand to the tip of the middle finger (WJC-DAC3:
# 170.1 mm female, 187.9 mm male). Used to convert a measured hand length
# (wrist to fingertip) to the WJC-MET3 segment length.
HAND_MET3_TO_DAC3_RATIO = {
    "female": 78.0 / 170.1,
    "male": 86.2 / 187.9,
}


# Table 4 trunk sub-segments, used to split the trunk into a pelvis, an
# abdomen and a thorax. (origin, other, length mm, mass %, CM % from origin,
# sagittal r %). The CERV-XYPH length is not tabulated; it is derived
# below as the whole trunk (CERV-MIDH) minus the middle and lower trunk,
# i.e. the three are treated as stacked vertically, matching how the
# body model treats upright standing.
TRUNK_SUBSEGMENTS = {
    "female": {
        "upper": ("CERV", "XYPH", None, 15.45, 50.50, 46.6),
        "middle": ("XYPH", "OMPH", 205.3, 14.65, 45.12, 43.3),
        "lower": ("OMPH", "MIDH", 181.5, 12.47, 49.20, 43.3),
    },
    "male": {
        "upper": ("CERV", "XYPH", None, 15.96, 50.66, 50.5),
        "middle": ("XYPH", "OMPH", 215.5, 16.33, 45.02, 48.2),
        "lower": ("OMPH", "MIDH", 145.7, 11.17, 61.15, 61.5),
    },
}


def _trunk_lengths_mm(sex):
    sub = TRUNK_SUBSEGMENTS[sex]
    whole = TABLE4[sex]["trunk"].reference_length_m * 1000.0
    middle, lower = sub["middle"][2], sub["lower"][2]
    return whole - middle - lower, middle, lower


def split_trunk_parameters(sex):
    """de Leva's three trunk sub-segments as model segments.

    thorax: upper trunk, C7 to the xiphoid (the rib cage).
    abdomen: middle trunk, xiphoid to the navel (omphalion).
    pelvis: lower trunk, navel to the hip joint centres.

    Returns {"thorax", "abdomen", "pelvis"} -> SegmentParameters, with de
    Leva's sample mean lengths as reference lengths.
    """
    sub = TRUNK_SUBSEGMENTS[sex]
    lengths = dict(zip(("upper", "middle", "lower"), _trunk_lengths_mm(sex)))
    names = {"upper": "thorax", "middle": "abdomen", "lower": "pelvis"}
    out = {}
    for key, name in names.items():
        _, _, _, mass, cm, r = sub[key]
        origin, other = sub[key][0], sub[key][1]
        out[name] = SegmentParameters(name, origin, other, mass / 100.0, cm / 100.0,
                                      r / 100.0, lengths[key] / 1000.0)
    return out


def xiphoid_fraction(sex):
    """Height of the xiphoid between the navel (0) and C7 (1) in de Leva's
    sample. ANSUR II has no xiphoid height, so each person's is placed at
    this fraction of their own navel-to-C7 distance."""
    upper, middle, _ = _trunk_lengths_mm(sex)
    return middle / (upper + middle)


def _model_segments(sex):
    table = {k: v for k, v in TABLE4[sex].items() if k != "trunk"}
    table.update(split_trunk_parameters(sex))
    return table


# The segments the body model uses: Table 4 with the trunk split into
# de Leva's three sub-segments (thorax, abdomen, pelvis).
SEGMENT_TABLE = {sex: _model_segments(sex) for sex in SEXES}


def segment_parameters(sex):
    try:
        return SEGMENT_TABLE[sex]
    except KeyError:
        raise ValueError(f"sex must be one of {SEXES}, got {sex!r}") from None


def whole_trunk_parameters(sex):
    """de Leva's single trunk segment (CERV-MIDH), for reference and checks."""
    segment_parameters(sex)
    return TABLE4[sex]["trunk"]


def total_mass_fraction(sex):
    """Sum of mass fractions counting bilateral segments twice (about 1)."""
    return sum(
        p.mass_fraction * (2 if name in BILATERAL else 1)
        for name, p in segment_parameters(sex).items()
    )
