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
    "sinclair2022": (
        "Sinclair J, Taylor PJ, et al. 2022. Habitual and manipulated foot placement "
        "angles (PMC9501107) and stance widths (PMC9503729) in the barbell back squat, "
        "70 male lifters at 70% 1RM. Sports (MDPI)."
    ),
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
    "mata2021": (
        "Mata AJ, Hayashi H, Moreno PA, Dudley RI, Sorenson EA. Hip flexion "
        "angles during supine range of motion and bodyweight squats. Int J "
        "Exerc Sci. 2021;14(1):912-918 (volume and pages as cited by Straub and "
        "Powers 2024). doi:10.70252/LMEG8374"
    ),
    "straub2024": (
        "Straub RK, Powers CM. A biomechanical review of the squat exercise: "
        "implications for clinical practice. Int J Sports Phys Ther. "
        "2024;19(4):490-501. doi:10.26603/001c.94600 (narrative review, used "
        "for definitions and the hip end-range mechanism, not for numbers)"
    ),
    "fujiwara2010": (
        "Fujiwara K, Asai H, Kiyota N, Mammadova A. Relationship between quiet "
        "standing position and perceptibility of standing position in the "
        "anteroposterior direction. J Physiol Anthropol. 2010;29(5):197-203. "
        "doi:10.2114/jpa2.29.197 (abstract)"
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
        "pelvis_peak_time_pct": Measurement(70.25, float("nan"), "zawadka2020",
                                            "median time of max pelvic tilt, % of descent"),
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
        "pelvis_peak_time_pct": Measurement(62.33, float("nan"), "zawadka2020",
                                            "median time of max pelvic tilt, % of descent"),
        "ankle_peak_time_pct": Measurement(86.75, float("nan"), "zawadka2020",
                                           "median time of max ankle angle, % of descent"),
        "knee_peak_time_pct": Measurement(98.92, float("nan"), "zawadka2020",
                                          "median time of max knee angle, % of descent"),
    },
}

# Hemmerich et al. 2006 abstract: squatting with heels up needed a mean
# maximum knee flexion of 157 +- 6 deg (30 healthy Indian adults, 48 y).
KNEE_FLEXION_MAX = Measurement(157.0, 6.0, "hemmerich2006", "max knee flexion, squat with heels up")

# Mata et al. 2021. 28 adults (14 men, 14 women), 1.70 +- 0.11 m.
# Bodyweight squats, feet flat. Angles at the moment posterior pelvic tilt
# began. The paper gives means with 95% confidence intervals for left and
# right sides; values here average the two sides, and the SD is recovered
# from the CI half-width as half_width / t(0.975, 27) * sqrt(28).
# The hip angle is absolute (anatomical zero), not relative to standing.
MATA_2021 = {
    "knee_at_ppt": Measurement(100.27, 14.8, "mata2021", "knee flexion when the pelvis starts tilting back"),
    "ankle_at_ppt": Measurement(25.21, 5.2, "mata2021", "ankle dorsiflexion when the pelvis starts tilting back"),
    "hip_at_ppt_absolute": Measurement(109.88, 8.7, "mata2021",
                                       "hip flexion when the pelvis starts tilting back (absolute)"),
}

# Fujiwara et al. 2010 (abstract): centre of pressure in quiet standing
# lay between 31% and 58% of foot length from the heel. Only the range is
# reported.
QUIET_STANDING_COP_RANGE = (0.31, 0.58)


# Sinclair J, Taylor PJ, Shadwell G, Stone M, Booth N, Jones B, Finlay S,
# Ali AM et al. 2022. Two papers on the same 70 male recreational lifters
# squatting a barbell at 70% of their 1RM with self-selected feet,
# Visual3D Cardan angles:
#   "Two-Experiment Examination of Habitual and Manipulated Foot Placement
#   Angles..." Sports (Basel)/MDPI 2022 (PMC9501107), and
#   "A Multi-Experiment Investigation of the Effects Stance Width on the
#   Biomechanics of the Barbell Squat" Sports 10(9):136 (PMC9503729).
# Each paper reports tertile groups; the values below are pooled (mean of
# group means weighted by n, SD from within- plus between-group
# variance). Stance width: distance between the feet's segment centres
# of mass over greater trochanter marker width. Hip abduction: peak and
# range in the foot-angle paper (signs flipped to abduction positive).
SINCLAIR_2022 = {
    "stance_width_ratio": Measurement(1.23, 0.16, "sinclair2022", "stance width / greater trochanter width"),
    "toe_out": Measurement(23.9, 6.9, "sinclair2022", "foot placement angle, deg"),
    "hip_abduction_peak": Measurement(28.5, 8.2, "sinclair2022", "peak hip abduction, deg"),
    "hip_abduction_rom": Measurement(18.2, 7.7, "sinclair2022", "hip abduction range during the squat, deg"),
}
