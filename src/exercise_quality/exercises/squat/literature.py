"""Published bodyweight squat measurements used for sampling and validation.

Every number here was read from the paper's own text or tables (open
access full text), except Hemmerich 2006, where only the abstract was
available. Means are between-subject means; ``sd`` is the between-subject
standard deviation the paper reports.

Angle definitions differ between studies and from this model; each entry
says what was measured so comparisons stay honest.
"""

from dataclasses import dataclass

SOURCES = {
    "ishida2025": (
        "Ishida T, Ueno R, Kitamura Y, Yamakawa Y, Samukawa M, Tohyama H. "
        "The effect of real-time feedback regarding the center-of-pressure "
        "position on patellofemoral joint loading during double-leg squatting. "
        "Orthop J Sports Med. 2025;13(3). doi:10.1177/23259671251319526"
    ),
    "kasahara2024": (
        "Kasahara S, Ishida T, Linjing J, Chiba A, Samukawa M, Tohyama H. "
        "Relationship among the COM motion, the lower extremity and the trunk "
        "during the squat. J Hum Kinet. 2024;93:29-39. doi:10.5114/jhk/183066"
    ),
    "zawadka2020": (
        "Zawadka M, Smolka J, Skublewska-Paszkowska M, Lukasik E, Gawda P. "
        "How are squat timing and kinematics in the sagittal plane related to "
        "squat depth? J Sports Sci Med. 2020;19:500-507."
    ),
    "hemmerich2006": (
        "Hemmerich A, Brown H, Smith S, Marthandam SSK, Wyss UP. Hip, knee, and "
        "ankle kinematics of high range of motion activities of daily living. "
        "J Orthop Res. 2006;24:770-781. doi:10.1002/jor.20114 (abstract only)"
    ),
}


@dataclass(frozen=True)
class Measurement:
    mean: float
    sd: float
    source: str
    what: str


# Ishida et al. 2025, "natural" condition. 16 men, 22.9 +- 1.5 y,
# 173.6 +- 5.2 cm. Thighs parallel to the floor, metronome 2 s down,
# 2 s up, 2 s standing. Arm position not stated. Values are for the
# descent phase.
ISHIDA_2025 = {
    "knee_peak": Measurement(121.8, 5.7, "ishida2025", "peak knee flexion"),
    "ankle_at_knee60": Measurement(19.3, 4.8, "ishida2025", "ankle dorsiflexion at 60 deg knee flexion"),
    "ankle_at_knee90": Measurement(27.8, 5.1, "ishida2025", "ankle dorsiflexion at 90 deg knee flexion"),
    "ankle_peak": Measurement(33.8, 4.5, "ishida2025", "peak ankle dorsiflexion"),
    "trunk_at_knee60": Measurement(22.8, 6.2, "ishida2025", "forward trunk inclination at 60 deg knee flexion"),
    "trunk_at_knee90": Measurement(30.6, 6.6, "ishida2025", "forward trunk inclination at 90 deg knee flexion"),
    "trunk_peak": Measurement(38.5, 7.3, "ishida2025", "peak forward trunk inclination"),
    "cop_fraction": Measurement(0.363, 0.08, "ishida2025",
                                "mean COP, fraction of foot length from heel, descent with knee > 60 deg"),
    # anterior COP condition: what the same men reached when cued forward
    "ankle_peak_cued_forward": Measurement(36.8, 3.9, "ishida2025", "peak ankle dorsiflexion, anterior COP cue"),
}

# Kasahara et al. 2024. 26 adults (13 men, 13 women), 1.67 +- 0.08 m.
# Feet shoulder width, arms crossed over the chest, heels down, thighs
# parallel, 2 s descent. ROM = maximum minus onset angle.
KASAHARA_2024 = {
    "com_descent_pct_height": Measurement(23.1, 2.5, "kasahara2024", "maximal COM descent, % of height"),
    "com_ap_shift_pct_foot": Measurement(-5.0, 9.6, "kasahara2024",
                                         "COM forward displacement start to bottom, % of foot length"),
    "trunk_rom": Measurement(34.7, 11.5, "kasahara2024", "trunk ROM"),
    "hip_rom": Measurement(82.7, 6.1, "kasahara2024", "hip ROM"),
    "knee_rom": Measurement(107.2, 11.1, "kasahara2024", "knee ROM"),
    "ankle_rom": Measurement(32.8, 5.6, "kasahara2024", "ankle ROM"),
    # time of each joint's peak relative to the lowest COM point, ms
    "trunk_peak_ms": Measurement(-11.5, 23.6, "kasahara2024", "trunk peak time vs max COM descent"),
    "hip_peak_ms": Measurement(-1.1, 8.6, "kasahara2024", "hip peak time vs max COM descent"),
    "knee_peak_ms": Measurement(0.4, 3.0, "kasahara2024", "knee peak time vs max COM descent"),
    "ankle_peak_ms": Measurement(-21.0, 26.6, "kasahara2024", "ankle peak time vs max COM descent"),
}

# Zawadka et al. 2020. 60 recreationally active adults (20 women, 40 men).
# Maximum depth they could hold for 3 s, arms straight forward parallel
# to the floor, heels down, preferred pace. "Hip" is pelvis vs thigh and
# lumbar is thorax vs pelvis, so this model's hip_flexion (rigid trunk vs
# thigh) corresponds roughly to hip + lumbar.
ZAWADKA_2020 = {
    "female": {
        "knee_rom": Measurement(119.00, 15.10, "zawadka2020", "knee ROM"),
        "ankle_rom": Measurement(30.81, 4.86, "zawadka2020", "ankle ROM"),
        "hip_rom": Measurement(98.61, 12.22, "zawadka2020", "hip ROM (pelvis-thigh)"),
        "lumbar_rom": Measurement(30.08, 14.44, "zawadka2020", "lumbar ROM"),
        "pelvis_rom": Measurement(22.61, 7.54, "zawadka2020", "pelvis tilt ROM"),
        "ankle_peak_time_pct": Measurement(90.17, float("nan"), "zawadka2020",
                                           "median time of max ankle angle, % of descent"),
        "knee_peak_time_pct": Measurement(98.83, float("nan"), "zawadka2020",
                                          "median time of max knee angle, % of descent"),
    },
    "male": {
        "knee_rom": Measurement(124.80, 15.50, "zawadka2020", "knee ROM"),
        "ankle_rom": Measurement(30.27, 6.36, "zawadka2020", "ankle ROM"),
        "hip_rom": Measurement(99.58, 11.03, "zawadka2020", "hip ROM (pelvis-thigh)"),
        "lumbar_rom": Measurement(43.74, 15.31, "zawadka2020", "lumbar ROM"),
        "pelvis_rom": Measurement(19.61, 8.05, "zawadka2020", "pelvis tilt ROM"),
        "ankle_peak_time_pct": Measurement(86.75, float("nan"), "zawadka2020",
                                           "median time of max ankle angle, % of descent"),
        "knee_peak_time_pct": Measurement(98.92, float("nan"), "zawadka2020",
                                          "median time of max knee angle, % of descent"),
    },
}

# Hemmerich et al. 2006 abstract: squatting with heels up needed a mean
# maximum knee flexion of 157 +- 6 deg (30 healthy Indian adults, 48 y).
KNEE_FLEXION_MAX = Measurement(157.0, 6.0, "hemmerich2006", "max knee flexion, squat with heels up")
