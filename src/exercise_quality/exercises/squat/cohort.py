"""Simulate a cohort under a study's protocol and compare with its results.

Metrics are taken from the generated landmarks with the same definitions
the studies used where the model allows it (descent phase only, ROM as
maximum minus starting angle, COP as a fraction of foot length from the
heel). The comparison reports a z score:

    z = (model cohort mean - published mean) / published between-subject SD

|z| < 1 means the simulated average person is within one real person's
spread of the published average. It does not mean individual reps are
right, and it cannot validate anything the study did not measure.
"""

from dataclasses import dataclass

import numpy as np

from exercise_quality.core.dynamics import analyse_dynamics
from exercise_quality.exercises.squat import literature as lit
from exercise_quality.exercises.squat.balanced import generate_feasible_squat
from exercise_quality.exercises.squat.protocols import sample_style


def _descent_window(seq):
    t = seq.style.timing
    return (seq.times >= t.descent_start) & (seq.times <= t.ascent_start)


def _at_knee(knee, values, target):
    """Value when the knee first reaches ``target`` degrees on the way down."""
    up_to_peak = int(np.argmax(knee)) + 1
    k, v = knee[:up_to_peak], values[:up_to_peak]
    if target > k.max():
        return np.nan
    i = int(np.argmax(k >= target))
    if i == 0:
        return float(v[0])
    frac = (target - k[i - 1]) / (k[i] - k[i - 1])
    return float(v[i - 1] + frac * (v[i] - v[i - 1]))


def rep_metrics(seq, dyn=None):
    dyn = dyn or analyse_dynamics(seq)
    m = seq.measured()
    w = _descent_window(seq)
    t = seq.times[w]
    knee = m["knee_flexion"][w]
    ankle = m["ankle_dorsiflexion"][w]
    trunk = m["trunk_inclination"][w]
    hip = m["hip_flexion"][w]
    pelvis = m["pelvis_inclination"][w]
    lumbar = m["lumbar_flexion"][w]
    com_y = seq.com[w, 1]
    cop = dyn.cop_fraction[w]

    timing = seq.style.timing
    # Peak timing is measured over the descent itself (Zawadka's 0-100%),
    # not the bottom hold: during a hold the angles sit on a plateau with
    # sway and float noise, so an argmax there lands on an arbitrary frame.
    # The window is closed so the last descent frame (the bottom) counts.
    descent_only = t <= timing.descent_start + timing.descent + 1e-9
    td = t[descent_only]
    lowest = td[int(np.argmin(com_y[descent_only]))]

    def peak_time(values):
        return td[int(np.argmax(values[descent_only]))]

    def peak_ms(values):
        return 1000.0 * (peak_time(values) - lowest)

    def peak_pct(values):
        return 100.0 * (peak_time(values) - timing.descent_start) / timing.descent

    return {
        "knee_peak": float(knee.max()),
        "ankle_peak": float(ankle.max()),
        "trunk_peak": float(trunk.max()),
        "ankle_at_knee60": _at_knee(knee, ankle, 60.0),
        "ankle_at_knee90": _at_knee(knee, ankle, 90.0),
        "trunk_at_knee60": _at_knee(knee, trunk, 60.0),
        "trunk_at_knee90": _at_knee(knee, trunk, 90.0),
        "cop_fraction": float(np.mean(cop[knee > 60.0])) if np.any(knee > 60) else np.nan,
        "com_descent_pct_height": 100.0 * float(com_y[0] - com_y.min()) / seq.body.stature,
        "com_ap_shift_pct_foot": 100.0 * float(seq.com[w, 0][np.argmin(com_y)] - seq.com[w, 0][0])
        / seq.body.foot_length,
        "trunk_rom": float(trunk.max() - trunk[0]),
        "hip_rom": float(hip.max() - hip[0]),
        "knee_rom": float(knee.max() - knee[0]),
        "ankle_rom": float(ankle.max() - ankle[0]),
        "pelvis_rom": float(pelvis.max() - pelvis[0]),
        "lumbar_rom": float(lumbar.max() - lumbar[0]),
        "knee_at_ppt": float(knee[int(np.argmax(pelvis))]),
        "ankle_at_ppt": float(ankle[int(np.argmax(pelvis))]),
        "pelvis_tilts_back": bool(pelvis.max() - pelvis[-1] > 1.0),
        "trunk_peak_ms": peak_ms(trunk),
        "hip_peak_ms": peak_ms(hip),
        "knee_peak_ms": peak_ms(knee),
        "ankle_peak_ms": peak_ms(ankle),
        "ankle_peak_time_pct": peak_pct(ankle),
        "knee_peak_time_pct": peak_pct(knee),
        "pelvis_peak_time_pct": peak_pct(pelvis),
        "cop_max_offset_m": dyn.max_cop_offset,
        "standing_cop_fraction": float(dyn.cop_fraction[0]),
        "feasible": seq.is_feasible and dyn.stays_on_foot,
        "depth_limited": "depth_limited_by_mobility" in seq.notes,
        **(rep_metrics_3d(seq) if seq.is_3d else {}),
    }


def rep_metrics_3d(seq):
    """Frontal-plane measures of a 3D rep, and what a side-view tracker would see of it."""
    from exercise_quality.core.model3d import joint_angles_3d
    from exercise_quality.core.observed import from_model_landmarks, observed_angles, segment_lengths

    j = joint_angles_3d(seq.landmarks3d)
    b = int(np.argmax(seq.angles.knee_flexion))
    abduction = j["hip_abduction"]
    track = from_model_landmarks(seq.landmarks, seq.fps)
    seen_knee = observed_angles(track)["knee_flexion"]
    thigh = segment_lengths(track)["thigh"]
    at100 = int(np.argmax(seen_knee >= 100.0)) if seen_knee.max() >= 100.0 else None
    return {
        "stance_width_ratio": seq.style.stance.width_ratio,
        "toe_out": seq.style.stance.toe_out,
        "hip_abduction_standing": float(abduction[0]),
        "hip_abduction_peak": float(abduction.max()),
        "hip_abduction_rom": float(abduction.max() - abduction[0]),
        "hip_rotation_at_bottom": float(j["hip_rotation"][b]),
        "hip_flexion_3d_peak": float(j["hip_flexion"].max()),
        "knee_fppa_at_knee60": float(j["knee_fppa"][int(np.argmax(seq.angles.knee_flexion >= 60.0))]),
        # the side view sees the thigh foreshortened when the knees go out over the toes
        "side_view_thigh_ratio_at_knee100": float(thigh[at100] / thigh[0]) if at100 is not None else np.nan,
    }


@dataclass
class Cohort:
    protocol: object
    sexes: list
    metrics: list

    def values(self, key, sex=None):
        return np.array([m[key] for m, s in zip(self.metrics, self.sexes)
                         if sex is None or s == sex], dtype=float)

    @property
    def feasible_share(self):
        return float(np.mean([m["feasible"] for m in self.metrics]))

    @property
    def depth_limited_share(self):
        return float(np.mean([m["depth_limited"] for m in self.metrics]))


def simulate_cohort(population, protocol, n, seed=0, sex=None, female_share=0.5,
                    fps=100.0, three_d=False, **style_kwargs):
    """``three_d``: the same people and styles (same random draws), each
    with a sampled stance (protocols.sample_stance, from a separate random
    stream), generated with the 3D model."""
    from dataclasses import replace

    from exercise_quality.core.model3d import make_body3d
    from exercise_quality.exercises.squat.protocols import sample_stance

    rng = np.random.default_rng(seed)
    stance_rng = np.random.default_rng([seed, 3])
    sexes, metrics = [], []
    for _ in range(n):
        body = population.sample(rng, sex=sex, female_share=female_share)
        style = sample_style(rng, body, protocol, **style_kwargs)
        if three_d:
            index = next(i for i, r in enumerate(population._by_sex[body.sex])
                         if r["subject_id"] == body.subject_id)
            body = make_body3d(population, body.sex, index)
            style = replace(style, stance=sample_stance(stance_rng))
        seq = generate_feasible_squat(body, style, fps=fps)
        sexes.append(body.sex)
        metrics.append(rep_metrics(seq))
    return Cohort(protocol, sexes, metrics)


@dataclass(frozen=True)
class ComparisonRow:
    metric: str
    what: str
    model_mean: float
    model_sd: float
    published_mean: float
    published_sd: float
    comparable: bool = True
    note: str = ""

    @property
    def z(self):
        if (not self.comparable or not np.isfinite(self.published_sd)
                or self.published_sd == 0):
            return float("nan")
        return (self.model_mean - self.published_mean) / self.published_sd


def compare(cohort, references, sex=None, incomparable=None):
    """Rows comparing cohort metrics with ``references`` (name -> Measurement).

    ``incomparable`` maps metric names to a reason; those rows are shown
    without a z score because the study measured something different.
    """
    incomparable = incomparable or {}
    rows = []
    for name, ref in references.items():
        values = cohort.values(name, sex)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        rows.append(ComparisonRow(name, ref.what, float(values.mean()),
                                  float(values.std(ddof=1)), ref.mean, ref.sd,
                                  comparable=name not in incomparable,
                                  note=incomparable.get(name, "")))
    return rows


# What each compared metric is for the model: "input" rows check sampling,
# "fitted" rows were used by scripts/calibrate_lumbopelvic.py, and
# "prediction" rows were not used to set anything.
KIND = {
    ("sinclair", "hip_abduction_peak"): "prediction (stance sampled from the same study)",
    ("sinclair", "hip_abduction_rom"): "prediction (stance sampled from the same study)",
    ("ishida", "knee_peak"): "input",
    ("ishida", "ankle_at_knee60"): "input + balance correction",
    ("ishida", "ankle_at_knee90"): "input + balance correction",
    ("ishida", "ankle_peak"): "input + balance correction",
    ("ishida", "trunk_at_knee60"): "input + balance correction",
    ("ishida", "trunk_at_knee90"): "input + balance correction",
    ("ishida", "trunk_peak"): "input + balance correction",
    ("ishida", "cop_fraction"): "input (measured dynamically)",
    ("kasahara", "com_descent_pct_height"): "prediction",
    ("kasahara", "com_ap_shift_pct_foot"): "input",
    ("kasahara", "trunk_rom"): "cross-study check",
    ("kasahara", "hip_rom"): "cross-study check (depth from Ishida)",
    ("kasahara", "knee_rom"): "cross-study check (depth from Ishida)",
    ("kasahara", "ankle_rom"): "input (from Ishida)",
    ("kasahara", "trunk_peak_ms"): "prediction",
    ("kasahara", "hip_peak_ms"): "fitted",
    ("kasahara", "knee_peak_ms"): "prediction",
    ("kasahara", "ankle_peak_ms"): "prediction",
    ("zawadka", "knee_rom"): "input",
    ("zawadka", "ankle_rom"): "cross-study check",
    ("zawadka", "hip_rom"): "fitted",
    ("zawadka", "pelvis_rom"): "fitted",
    ("zawadka", "lumbar_rom"): "prediction",
    ("zawadka", "pelvis_peak_time_pct"): "prediction",
    ("zawadka", "ankle_peak_time_pct"): "prediction",
    ("zawadka", "knee_peak_time_pct"): "prediction",
    ("mata", "knee_at_ppt"): "fitted",
    ("mata", "ankle_at_ppt"): "prediction",
}


def validation_report(population, n=200, seed=0, three_d=False):
    """All comparisons used in docs/validation.md, as {title: (cohort, rows, study)}.

    ``three_d`` runs the same cohorts with the 3D model and adds the
    frontal-plane comparison with Sinclair 2022.
    """
    from exercise_quality.exercises.squat import protocols as p

    def simulate_cohort(*args, **kwargs):
        return _simulate(*args, three_d=three_d, **kwargs)

    report = {}
    ishida = simulate_cohort(population, p.ISHIDA_PARALLEL, n, seed, sex="male")
    refs = {k: v for k, v in lit.ISHIDA_2025.items() if k != "ankle_peak_cued_forward"}
    report["Ishida 2025, men, thighs parallel"] = (ishida, compare(ishida, refs), "ishida")

    kasahara = simulate_cohort(population, p.KASAHARA_PARALLEL, n, seed + 1)
    report["Kasahara 2024, mixed, arms crossed, parallel"] = (
        kasahara, compare(kasahara, lit.KASAHARA_2024), "kasahara")

    zawadka = simulate_cohort(population, p.ZAWADKA_MAX_DEPTH, n, seed + 2, female_share=1 / 3)
    for sex in ("male", "female"):
        report[f"Zawadka 2020, {sex}, max depth, arms forward"] = (
            zawadka, compare(zawadka, lit.ZAWADKA_2020[sex], sex=sex), "zawadka")

    # Mata 2021: mixed sexes, bodyweight squats to full depth with heels
    # down. Simulated with the maximal-depth protocol, 50/50 sexes.
    mata = simulate_cohort(population, p.ZAWADKA_MAX_DEPTH, n, seed + 3, female_share=0.5)
    mata_refs = {k: v for k, v in lit.MATA_2021.items() if k != "hip_at_ppt_absolute"}
    rows = compare(mata, mata_refs)
    report["Mata 2021, mixed, full depth, moment the pelvis tilts back"] = (mata, rows, "mata")
    if three_d:
        # Sinclair 2022: 70 men, barbell at 70% 1RM, self-selected feet. Hip
        # flexion there peaked near 98 deg, so the Kasahara parallel protocol
        # (men) is the closest bodyweight match; the load is not modelled.
        sinclair = simulate_cohort(population, p.KASAHARA_PARALLEL, n, seed + 4, sex="male")
        refs = {k: lit.SINCLAIR_2022[k] for k in ("hip_abduction_peak", "hip_abduction_rom")}
        report["Sinclair 2022, men, self-selected feet (frontal plane)"] = (
            sinclair, compare(sinclair, refs), "sinclair")
    return report


_simulate = simulate_cohort
