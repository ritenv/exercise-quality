"""Side-view squat form rubric, adjusted for the person's build.

Source of the criteria
----------------------
Myer GD, Kushner AM, Brent JL, et al. "The back squat: a proposed
assessment of functional deficits and technical factors that limit
performance." Strength Cond J 2014;36(6):4-27 (the Back Squat Assessment,
BSA). Ten criteria; a criterion is marked as a deficit when it is not met
"during 2 or more repetitions"; the total counts deficits (0 = perfect).
Five of the ten can be measured from a side-view 2D pose track and are
graded here. The others are reported as not measurable or not graded,
with the reason (``NOT_GRADED``). This is a coaching-technique screen,
not an injury-risk or medical assessment.

Graded criteria (BSA number, wording quoted from Table 1)
----------------------------------------------------------
* depth (9): "At apex of depth, the tops of thighs are at least parallel
  to the ground." Measured as the hip-to-knee joint line. The top surface
  of the thigh is not the joint line; on a tapering thigh the surface
  reaches horizontal a few degrees later, so this reads slightly
  generous.
* heels (7): "Entire foot remains in contact with the ground." Heel rise
  relative to the toe point, as a fraction of foot length.
* trunk (3): "Trunk is parallel to tibia..." Trunk (hip-to-shoulder line)
  lean minus shin lean at the bottom. **Build adjustment**: whether a
  parallel trunk is physically possible depends on segment proportions,
  depth, arm position and how far the shins go forward, because the
  centre of mass has to stay over the foot. For the person's matched
  body, at the depth and arm position actually used, the balance model
  finds the trunk-minus-shin angle that puts the centre of mass over
  mid-foot (``COM_TARGET``, Ishida 2025's measured bottom COP) with the
  shins at ``CUED_SHIN_LEAN``. That shin lean is what Ishida 2025's men
  reached at parallel depth when cued to shift their weight forward
  (peak ankle dorsiflexion 36.8 +- 3.9 deg), so it is a measured,
  achievable value rather than a maximal range. If balance needs more
  than 0 deg there, that amount is allowed: a long-thighed, short-trunked
  person is not marked down for geometry they cannot change.
* ascent (10): "Shoulders and hips rise at the same, constant speed."
  Hips rising first shows as the trunk leaning further forward after the
  bottom; measured as the largest increase in trunk lean during the
  ascent.
* descent (8), the part that can be measured: "...controlled, constant
  speed throughout descent." Graded as an uninterrupted descent: the
  largest upward hip movement while going down, as a fraction of the
  standing hip height. Myer's "constant speed" is not graded: a
  smooth human movement speeds up and slows down (minimum-jerk profile).

Thresholds and measurement tolerance
------------------------------------
Each criterion has a standard (parallel, no heel lift, trunk within its
build allowance of the shin, no extra trunk lean on the way up, no
upward movement on the way down) and a tolerance equal to the 95th
percentile of the tracker measurement error for that quantity
(``TOLERANCE``). The errors come from passing synthetic reps through the
MediaPipe noise model (core/tracker_noise.py: jitter and drift, with its
assumed spread between videos) and comparing with the exact values. The
model's knees-out projection is not counted as error: it changes the
side-view shin and thigh angles by a few degrees (trunk minus shin about
+5, thigh about 3 deeper at 24 deg), but that is what a side view of a
real squat shows, and the BSA criteria are judged from that view; ``measurement_tolerances()`` recomputes them and
``tests/test_rubric.py`` checks the stored values. A deficit is only
marked when the measurement is past the standard by more than a tracker
could plausibly be wrong. The tolerances are about tracking error, not a
coaching judgement of what is "close enough".

Score
-----
``deficits``: BSA total over the graded criteria (each criterion counts
once if it fails in at least 2 reps, or in the only rep). ``score``:
100 x the mean share of reps passing each graded criterion, equal
weights. Equal weights are a choice, not evidence: the BSA itself counts
criteria equally.

Data checks
-----------
Results are flagged unreliable when the camera is far from side-on
(yaw scale below ``MIN_YAW_SCALE``) or when the tracked standing pose
puts the centre of mass outside where people stand (Fujiwara 2010:
31-58% of foot length, widened by the tracker COM error), which means
the angles are geometrically inconsistent.
"""

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from exercise_quality.core.observed import observed_angles, segment_lengths
from exercise_quality.core.observed_balance import com_fraction
from exercise_quality.core.sagittal_body import BodyPopulation
from exercise_quality.exercises.squat import literature as lit
from exercise_quality.exercises.squat import reps as R

COM_TARGET = lit.ISHIDA_2025["cop_fraction"].mean
CUED_SHIN_LEAN = lit.ISHIDA_2025["ankle_peak_cued_forward"].mean
PARALLEL_THIGH = -90.0

# 95th percentile of |tracker error| for each graded quantity: mean of
# measurement_tolerances(n=300) with seeds 0 and 1 (see module docstring).
TOLERANCE = {
    "depth": 3.1,          # deg of thigh angle
    "heels": 0.094,        # fraction of foot length (about 2.5 cm)
    "trunk": 3.8,          # deg of trunk minus shin
    "ascent": 2.6,         # deg of extra trunk lean
    "descent": 0.039,      # fraction of standing hip height
}

# Data checks
MIN_YAW_SCALE = 0.9                              # cos(26 deg)
STANDING_COM_RANGE = (0.31 - 0.15, 0.58 + 0.15)  # Fujiwara 2010 range +- tracker COM error p95
MIN_REPS_FOR_RULE = 2

NOT_GRADED = {
    "head (1)": "neck line vs vertical is measurable only roughly (ear landmark), and coaching advice on "
                "head position conflicts; reported, not graded",
    "thoracic (2)": "chest and shoulder-blade position need spine and scapula points a 2D tracker does not give",
    "lumbar (part of 3)": "a slightly lordotic lower back cannot be seen without spine points",
    "hip level (4)": "frontal plane, needs a front or back view",
    "frontal knee (5)": "frontal plane, needs a front or back view",
    "tibial progression (6)": "knees passing the toes is reported, not graded: restricting it moves load to "
                              "the hips and lower back (Fry et al. 2003), and the tracker toe point is not "
                              "the toe tip",
    "tempo (part of 10)": "Table 1's 'Ascent: Descent timing ratio is at least 2:1' is reported, not graded: "
                          "the direction is ambiguous as quoted",
}


@dataclass
class CriterionResult:
    name: str
    value: float
    threshold: float
    passed: bool
    note: str = ""


@dataclass
class RepResult:
    index: int
    features: dict
    criteria: dict      # name -> CriterionResult
    trunk_allowance: float = 0.0
    # Trunk minus shin that balance needs at this rep's OWN shin lean. If
    # the measured value is close to it, the lean is what the shin angle
    # forces (let the knees travel further forward); if it is well above,
    # the weight is ahead of mid-foot.
    trunk_needed_at_own_shin: float = float("nan")


@dataclass
class RubricResult:
    reps: list
    deficits: dict              # criterion -> bool
    pass_share: dict            # criterion -> share of reps passing
    score: float
    reliable: bool
    warnings: list = field(default_factory=list)
    body_id: str = ""
    reported: dict = field(default_factory=dict)

    @property
    def deficit_total(self):
        return int(sum(self.deficits.values()))


@lru_cache(maxsize=1)
def _population():
    return BodyPopulation.load()


def _bottom(angles, rep):
    return rep.start + int(np.argmax(angles["knee_flexion"][rep.start:rep.lowest + 1]))


def descent_interruption(track, rep):
    """Largest upward hip movement during the descent, fraction of standing hip height."""
    h = R.hip_height(track)[rep.start:rep.lowest + 1]
    running_min = np.minimum.accumulate(h)
    return float(np.max(h - running_min))


def balanced_trunk_minus_shin(body, thigh, shin, shoulder, elbow, neck_relative, target=COM_TARGET):
    """Trunk lean minus shin lean that puts the COM at ``target`` (fraction of foot)."""
    def offset(trunk):
        angles = {"trunk_inclination": trunk, "shank_inclination": shin,
                  "knee_flexion": shin - thigh, "thigh_inclination": thigh,
                  "neck_inclination": trunk + neck_relative,
                  "shoulder_flexion": shoulder, "elbow_flexion": elbow}
        return float(com_fraction(body, angles)) - target

    lo, hi = -40.0, 110.0
    if offset(lo) > 0:
        return lo - shin
    if offset(hi) < 0:
        return hi - shin
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if offset(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi) - shin


def trunk_allowance(body, features):
    """Build allowance for trunk-minus-shin at this rep's depth and arm position (deg, >= 0)."""
    needed = balanced_trunk_minus_shin(
        body, features["thigh_at_bottom"], CUED_SHIN_LEAN,
        features["shoulder_flexion_at_bottom"], features["elbow_flexion_at_bottom"],
        features["neck_at_bottom"] - features["trunk_at_bottom"])
    return max(0.0, needed)


def match_body(track, angles, sex=None, population=None):
    """Closest ANSUR II person by thigh and trunk proportions (either sex if None)."""
    from exercise_quality.video import compare as C

    population = population or _population()
    ratios = C.standing_proportions(track, angles)
    weights = {"thigh": 1.0, "trunk": 1.0, "foot": 0.0, "neck": 0.0}
    best = None
    for s in ([sex] if sex else ["female", "male"]):
        bodies, _, dist = C.matched_bodies(population, ratios, s, k=1, weights=weights)
        if best is None or dist < best[1]:
            best = (bodies[0], dist)
    return best[0]


def trunk_needed_at_own_shin(body, features):
    return balanced_trunk_minus_shin(
        body, features["thigh_at_bottom"], features["shank_at_bottom"],
        features["shoulder_flexion_at_bottom"], features["elbow_flexion_at_bottom"],
        features["neck_at_bottom"] - features["trunk_at_bottom"])


def grade_rep(features, track, rep, body):
    allowance = trunk_allowance(body, features)
    t = TOLERANCE
    criteria = {}

    def add(name, value, threshold, note=""):
        criteria[name] = CriterionResult(name, float(value), float(threshold), bool(value <= threshold), note)

    add("depth", features["thigh_at_bottom"], PARALLEL_THIGH + t["depth"],
        "thigh angle at the bottom, -90 = parallel; must reach parallel")
    add("heels", features["heel_rise_max"], t["heels"], "heel rise, fraction of foot length")
    add("trunk", features["trunk_minus_shank_at_bottom"], allowance + t["trunk"],
        f"trunk lean minus shin lean at the bottom; build allowance {allowance:.1f} deg")
    add("ascent", features["trunk_rise_after_bottom"], t["ascent"],
        "extra trunk lean after the bottom (hips rising first)")
    add("descent", descent_interruption(track, rep), t["descent"],
        "largest upward hip movement during the descent")
    return criteria, allowance


def assess(track, body=None, sex=None, population=None):
    """Grade every rep in an ObservedTrack (any units, person facing +x, y up)."""
    smooth = R.smoothed(track)
    angles = observed_angles(smooth)
    found = R.find_reps(smooth, angles)
    warnings = []
    if body is None:
        body = match_body(smooth, angles, sex, population)

    reliable = True
    if track.yaw_scale < MIN_YAW_SCALE:
        reliable = False
        warnings.append(f"camera about {np.degrees(np.arccos(track.yaw_scale)):.0f} deg from side-on; "
                        "angles were corrected for it but out-of-plane movement still leaks in")
    upright = angles["knee_flexion"] < 10
    if upright.sum() >= 5:
        standing = float(np.median(com_fraction(body, {k: v[upright] for k, v in angles.items()})))
        lo, hi = STANDING_COM_RANGE
        if not lo <= standing <= hi:
            reliable = False
            warnings.append(f"tracked standing pose puts the centre of mass at {standing:.2f} of the foot, "
                            f"outside where people stand ({lo:.2f}-{hi:.2f}); the angles are likely distorted")
    if not found:
        warnings.append("no squat reps found")
        return RubricResult([], {}, {}, float("nan"), False, warnings, getattr(body, "subject_id", ""))
    if len(found) < MIN_REPS_FOR_RULE:
        warnings.append("only one rep: a single failed rep counts as a deficit")

    reps = []
    for i, rep in enumerate(found):
        f = R.rep_features(smooth, rep, angles)
        criteria, allowance = grade_rep(f, smooth, rep, body)
        reps.append(RepResult(i, f, criteria, allowance, trunk_needed_at_own_shin(body, f)))

    names = list(reps[0].criteria)
    fails = {n: sum(not r.criteria[n].passed for r in reps) for n in names}
    need = min(MIN_REPS_FOR_RULE, len(reps))
    deficits = {n: fails[n] >= need for n in names}
    pass_share = {n: 1.0 - fails[n] / len(reps) for n in names}
    score = 100.0 * float(np.mean(list(pass_share.values())))

    lengths = segment_lengths(smooth)
    reported = {
        "knee_past_toe_at_bottom": float(np.median([r.features["knee_past_toe_at_bottom"] for r in reps])),
        "neck_at_bottom_minus_standing": float(
            np.median([r.features["neck_at_bottom"] for r in reps])
            - np.median(angles["neck_inclination"][upright] if upright.sum() else angles["neck_inclination"])),
        "descent_s": float(np.median([r.features["descent_s"] for r in reps])),
        "ascent_s": float(np.median([r.features["ascent_s"] for r in reps])),
        "trunk_allowance": float(np.median([r.trunk_allowance for r in reps])),
        "leg_length_units": float(np.median(lengths["thigh"] + lengths["shank"])),
    }
    return RubricResult(reps, deficits, pass_share, score, reliable, warnings,
                        getattr(body, "subject_id", ""), reported)


def measurement_tolerances(n=300, seed=0, quantile=95.0):
    """Recompute ``TOLERANCE`` from the tracker-noise model on synthetic reps."""
    from dataclasses import replace

    from exercise_quality.core import tracker_noise as TN
    from exercise_quality.core.observed import from_model_landmarks
    from exercise_quality.core.timing import RepTiming
    from exercise_quality.exercises.squat.balanced import generate_feasible_squat
    from exercise_quality.exercises.squat.protocols import ISHIDA_PARALLEL, ZAWADKA_MAX_DEPTH, sample_style

    population = _population()
    rng = np.random.default_rng(seed)
    errors = {k: [] for k in TOLERANCE}
    while len(errors["depth"]) < n:
        body = population.sample(rng)
        protocol = ISHIDA_PARALLEL if rng.random() < 0.5 else ZAWADKA_MAX_DEPTH
        style = sample_style(rng, body, protocol)
        tempo = RepTiming(stand_before=0.6, descent=rng.uniform(0.8, 2.5), ascent=rng.uniform(0.6, 2.5),
                          stand_after=0.6)
        style = replace(style, timing=tempo)
        seq = generate_feasible_squat(body, style, fps=30.0)
        if not seq.is_feasible:
            continue
        clean = R.smoothed(from_model_landmarks(seq.landmarks, 30.0))
        # Tracker error only: jitter and drift. The knees-out projection is
        # left out because it is what a side view really shows, which is
        # what the BSA's lateral-view criteria are judged on.
        noise = replace(TN.TrackerNoise.sample(rng), knee_out_deg=0.0)
        noisy = R.smoothed(TN.apply(from_model_landmarks(seq.landmarks, 30.0), noise, rng))
        reps_c, reps_n = R.find_reps(clean), R.find_reps(noisy)
        if len(reps_c) != 1 or len(reps_n) != 1:
            continue
        fc, fn = R.rep_features(clean, reps_c[0]), R.rep_features(noisy, reps_n[0])
        errors["depth"].append(fn["thigh_at_bottom"] - fc["thigh_at_bottom"])
        errors["heels"].append(fn["heel_rise_max"] - fc["heel_rise_max"])
        errors["trunk"].append(fn["trunk_minus_shank_at_bottom"] - fc["trunk_minus_shank_at_bottom"])
        errors["ascent"].append(fn["trunk_rise_after_bottom"] - fc["trunk_rise_after_bottom"])
        errors["descent"].append(descent_interruption(noisy, reps_n[0]) - descent_interruption(clean, reps_c[0]))
    return {k: float(np.percentile(np.abs(v), quantile)) for k, v in errors.items()}
