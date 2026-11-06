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
  relative to the toe point, as a fraction of foot length, against a
  straight baseline from the start to the end of the rep so slow drift
  (camera roll) does not count.
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
  Hips rising first tips the trunk further forward early in the ascent
  than it was at the same height on the way down. Measured as the largest
  trunk lean difference, up minus down, at equal hip height in the lower
  60% of the rep. A symmetric rep scores 0 even if the trunk leans most a
  little above the bottom on both passes.
* descent (8), the part that can be measured: "...controlled, constant
  speed throughout descent." Graded as an uninterrupted descent: the
  largest upward hip movement while going down, as a fraction of the
  standing hip height. Myer's "constant speed" is not graded: a
  smooth human movement speeds up and slows down (minimum-jerk profile).

Further side-view checks (not in the BSA)
-----------------------------------------
These are not BSA criteria and have no published pass mark, so each
standard below says where it comes from. They count towards ``score`` and
``deficits`` like the BSA criteria; ``bsa_score`` and
``bsa_deficit_total`` keep the BSA alone.

* balance: the centre of mass over the foot at the bottom, estimated
  from the tracked pose with the matched body, its thigh and shank
  shortened to their seen length so knees travelling out over the toes do
  not push the estimate back (core/observed_balance.py).
  Standard: within Ishida 2025's mean bottom centre of pressure +- 2
  between-person SD (``BALANCE_RANGE``, 0.20-0.52 of foot length from the
  heel). Below means the weight sits on the heels, above on the toes. The
  estimate is quasi-static (where the mass is, not where the floor
  pushes), which is close at the slow bottom of a rep.
* lockout: the knee straightens at the top after each rep. Measured as
  the most extended knee between the ascent and the next rep (or the
  following second). Standard: ``KNEE_LOCKED`` = 5 deg of flexion, a
  convention; standing knees in healthy young adults read about 1-2 deg
  on radiographs. The hip at the top is reported, not graded: the
  tracker's hip angle (hip-shoulder line vs thigh) in standing depends on
  pelvic tilt and posture, and there is no norm for it.
* consistency: each rep's depth (thigh angle) and trunk lean at the
  bottom, against the median of the other reps. Standard:
  ``CONSISTENCY_DEG`` = 10 deg, a convention: no study reports normal
  within-person rep-to-rep spread for this. Needs at least
  ``MIN_REPS_FOR_CONSISTENCY`` reps, otherwise it is not graded.

Bottom control (bouncing out of the bottom) is reported, not graded:
see ``NOT_GRADED``.

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
``deficits``: the BSA rule applied to every graded criterion (each
counts once if it fails in at least 2 reps, or in the only rep);
``bsa_deficit_total`` counts the BSA criteria only. ``score``: 100 x the
mean share of reps passing each graded criterion, equal weights;
``bsa_score`` over the BSA criteria only. Equal weights are a choice, not evidence: the BSA itself counts
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
from exercise_quality.core.observed_balance import com_fraction, projected_body
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
    "heels": 0.088,        # fraction of foot length (about 2.3 cm)
    "trunk": 3.8,          # deg of trunk minus shin
    "ascent": 4.8,         # deg of trunk lean, up minus down
    "descent": 0.039,      # fraction of standing hip height
    "balance": 0.080,      # fraction of foot length (full noise model, see measurement_tolerances)
    "lockout": 0.27,       # deg of knee flexion at the top (signed: only "looks bent" errors)
    "consistency": 4.5,    # deg, one rep vs another of the same video
}

BSA_CRITERIA = ("depth", "heels", "trunk", "ascent", "descent")
EXTRA_CRITERIA = ("balance", "lockout", "consistency")

_cop = lit.ISHIDA_2025["cop_fraction"]
BALANCE_RANGE = (_cop.mean - 2 * _cop.sd, _cop.mean + 2 * _cop.sd)
KNEE_LOCKED = 5.0
CONSISTENCY_DEG = 10.0
MIN_REPS_FOR_CONSISTENCY = 3
TOP_WINDOW_S = 1.0

# Data checks
MIN_YAW_SCALE = 0.9                              # cos(26 deg)
MIN_VISIBILITY = 0.5                             # MediaPipe visibility of hip, knee, ankle
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
    "bottom control": "bouncing out of the bottom is reported, not graded: there is no published pass "
                      "mark, and at 30 fps tracker noise roughly doubles the measured turnaround acceleration "
                      "(synthetic reps: median 0.23 g exact vs 0.42 g tracked, correlation 0.7-0.8); the peak "
                      "upward hip acceleration at the turnaround is reported in g",
    "hip lockout": "the tracker hip angle in standing depends on posture and has no norm; reported, not graded",
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
    bsa_score: float = float("nan")

    @property
    def deficit_total(self):
        return int(sum(self.deficits.values()))

    @property
    def bsa_deficit_total(self):
        return int(sum(v for k, v in self.deficits.items() if k in BSA_CRITERIA))


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


def top_after(track, angles, reps, i):
    """Frames from the start of rep ``i``'s ascent to the next rep's descent (or ``TOP_WINDOW_S`` on)."""
    rep = reps[i]
    end = reps[i + 1].start if i + 1 < len(reps) else rep.end + int(TOP_WINDOW_S * track.fps)
    return slice(rep.ascent_start, min(end, len(track) - 1) + 1)


def lockout(angles, window):
    """Knee and hip flexion when the knee is most extended in ``window``.

    Tracker noise biases the smallest angle towards straighter (about 2
    deg on synthetic reps), so the check errs lenient. The highest hip
    would be no better: near a straight leg the hip height hardly changes
    with knee bend (1% of hip height is 10-15 deg of knee).
    """
    knee = angles["knee_flexion"][window]
    j = int(np.nanargmin(knee))
    return float(knee[j]), float(angles["hip_flexion"][window][j])


def turnaround_acceleration(track, rep, body, window_s=0.25):
    """Peak upward hip acceleration around the bottom, in g (image scale from the matched leg length)."""
    lengths = segment_lengths(track)
    metres_per_unit = (body.thigh + body.shank) / float(np.median(lengths["thigh"] + lengths["shank"]))
    y = (track["hip"][:, 1] - track["ankle"][:, 1]) * metres_per_unit
    acc = np.gradient(np.gradient(y, 1 / track.fps), 1 / track.fps)
    w = int(window_s * track.fps)
    return float(np.max(acc[max(0, rep.lowest - w):rep.lowest + w + 1]) / 9.81)


def balance_offset(com):
    """How far the COM is outside ``BALANCE_RANGE`` (0 inside), fraction of foot length."""
    lo, hi = BALANCE_RANGE
    return float(max(lo - com, com - hi, 0.0))


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
    """Balance angle at the rep's own shin lean, with the segments as long as they looked."""
    from dataclasses import replace

    if "thigh_seen_scale" in features:
        body = replace(body, thigh=body.thigh * features["thigh_seen_scale"],
                       shank=body.shank * features["shank_seen_scale"])
    return balanced_trunk_minus_shin(
        body, features["thigh_at_bottom"], features["shank_at_bottom"],
        features["shoulder_flexion_at_bottom"], features["elbow_flexion_at_bottom"],
        features["neck_at_bottom"] - features["trunk_at_bottom"])


def rep_measures(track, angles, reps, i, body):
    """``reps.rep_features`` plus what the rubric needs beyond it."""
    rep = reps[i]
    f = R.rep_features(track, rep, angles)
    b = _bottom(angles, rep)
    seen = projected_body(body, track, b, angles)
    f["thigh_seen_scale"], f["shank_seen_scale"] = seen.thigh / body.thigh, seen.shank / body.shank
    f["com_at_bottom"] = float(com_fraction(seen, {k: np.asarray(v)[b] for k, v in angles.items()}))
    f["knee_at_top"], f["hip_at_top"] = lockout(angles, top_after(track, angles, reps, i))
    f["turnaround_acceleration_g"] = turnaround_acceleration(track, rep, body)
    return f


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
    add("ascent", features["trunk_extra_on_ascent"], t["ascent"],
        "trunk lean on the way up minus on the way down at the same hip height (hips rising first)")
    add("descent", descent_interruption(track, rep), t["descent"],
        "largest upward hip movement during the descent")
    com = features["com_at_bottom"]
    side = "on the heels" if com < BALANCE_RANGE[0] else "on the toes" if com > BALANCE_RANGE[1] else "mid-foot"
    add("balance", balance_offset(com), t["balance"],
        f"centre of mass at {com:.2f} of the foot at the bottom ({side}); distance outside "
        f"{BALANCE_RANGE[0]:.2f}-{BALANCE_RANGE[1]:.2f}")
    add("lockout", features["knee_at_top"], KNEE_LOCKED + t["lockout"],
        f"knee flexion at the top after the rep (hip {features['hip_at_top']:.1f} deg, not graded)")
    return criteria, allowance


def grade_consistency(reps):
    """Add the consistency criterion: each rep's depth and trunk lean against the other reps' median."""
    if len(reps) < MIN_REPS_FOR_CONSISTENCY:
        return
    for r in reps:
        others = [o for o in reps if o is not r]
        d_thigh = r.features["thigh_at_bottom"] - np.median([o.features["thigh_at_bottom"] for o in others])
        d_trunk = r.features["trunk_at_bottom"] - np.median([o.features["trunk_at_bottom"] for o in others])
        value = max(abs(d_thigh), abs(d_trunk))
        threshold = CONSISTENCY_DEG + TOLERANCE["consistency"]
        r.criteria["consistency"] = CriterionResult(
            "consistency", float(value), float(threshold), bool(value <= threshold),
            f"vs the other reps: depth {d_thigh:+.1f} deg, trunk lean {d_trunk:+.1f} deg")


def assess(track, body=None, sex=None, population=None):
    """Grade every rep in an ObservedTrack (any units, person facing +x, y up)."""
    warnings = []
    if np.nanmedian(track["toe"][:, 0] - track["heel"][:, 0]) < 0:
        track = track.map_points(lambda v: v * np.array([-1.0, 1.0]))
        warnings.append("person faced -x; the track was mirrored")
    smooth = R.smoothed(track)
    angles = observed_angles(smooth)
    found = R.find_reps(smooth, angles)
    if body is None:
        body = match_body(smooth, angles, sex, population)

    reliable = True
    if track.visibility:
        legs = np.nanmedian([np.nanmedian(track.visibility[k]) for k in ("hip", "knee", "ankle")])
        if legs < MIN_VISIBILITY:
            reliable = False
            warnings.append(f"near-side leg landmarks are poorly visible (median {legs:.2f}); "
                            "the tracker is guessing")
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
        f = rep_measures(smooth, angles, found, i, body)
        criteria, allowance = grade_rep(f, smooth, rep, body)
        reps.append(RepResult(i, f, criteria, allowance, trunk_needed_at_own_shin(body, f)))
    grade_consistency(reps)

    names = list(reps[0].criteria)
    fails = {n: sum(not r.criteria[n].passed for r in reps) for n in names}
    need = min(MIN_REPS_FOR_RULE, len(reps))
    deficits = {n: fails[n] >= need for n in names}
    pass_share = {n: 1.0 - fails[n] / len(reps) for n in names}
    score = 100.0 * float(np.mean(list(pass_share.values())))
    bsa_score = 100.0 * float(np.mean([pass_share[n] for n in names if n in BSA_CRITERIA]))

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
        "hip_at_top": float(np.median([r.features["hip_at_top"] for r in reps])),
        "turnaround_acceleration_g": float(np.median([r.features["turnaround_acceleration_g"] for r in reps])),
    }
    return RubricResult(reps, deficits, pass_share, score, reliable, warnings,
                        getattr(body, "subject_id", ""), reported, bsa_score)


def measurement_tolerances(n=300, seed=0, quantile=95.0):
    """Recompute ``TOLERANCE`` from the tracker-noise model on synthetic reps.

    BSA criteria and lockout: jitter and drift only (see module docstring).
    Balance: the full noise model including the knees-out projection,
    because the centre of mass is a physical quantity and the projection
    distorts its estimate. Consistency: two noise draws with the same
    video-level settings stand in for two reps of one video; the error is
    the difference between their errors.
    """
    from dataclasses import replace

    from exercise_quality.core import tracker_noise as TN
    from exercise_quality.core.observed import from_model_landmarks
    from exercise_quality.core.timing import RepTiming
    from exercise_quality.exercises.squat.balanced import generate_feasible_squat
    from exercise_quality.exercises.squat.protocols import ISHIDA_PARALLEL, ZAWADKA_MAX_DEPTH, sample_style

    population = _population()
    rng = np.random.default_rng(seed)
    extra = np.random.default_rng(seed + 10_000)   # separate stream: the BSA tolerances stay as they were
    errors = {k: [] for k in TOLERANCE}

    def measured(track, body):
        smooth = R.smoothed(track)
        angles = observed_angles(smooth)
        found = R.find_reps(smooth, angles)
        if len(found) != 1:
            return None
        return smooth, found[0], rep_measures(smooth, angles, found, 0, body)

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
        errors["ascent"].append(fn["trunk_extra_on_ascent"] - fc["trunk_extra_on_ascent"])
        errors["descent"].append(descent_interruption(noisy, reps_n[0]) - descent_interruption(clean, reps_c[0]))

        exact = measured(from_model_landmarks(seq.landmarks, 30.0), body)
        if exact is None:
            continue
        mc = exact[2]
        mn = measured(TN.apply(from_model_landmarks(seq.landmarks, 30.0), noise, extra), body)
        full = TN.TrackerNoise.sample(extra)
        m1 = measured(TN.apply(from_model_landmarks(seq.landmarks, 30.0), full, extra), body)
        m2 = measured(TN.apply(from_model_landmarks(seq.landmarks, 30.0), full, extra), body)
        if mn is not None:
            errors["lockout"].append(mn[2]["knee_at_top"] - mc["knee_at_top"])
        if m1 is not None:
            errors["balance"].append(m1[2]["com_at_bottom"] - mc["com_at_bottom"])
        if m1 is not None and m2 is not None:
            errors["consistency"].append(max(
                abs(m1[2]["thigh_at_bottom"] - m2[2]["thigh_at_bottom"]),
                abs(m1[2]["trunk_at_bottom"] - m2[2]["trunk_at_bottom"])))
    # Lockout takes the most extended frame, so noise mostly makes the knee
    # look straighter; only errors towards "bent" can fail a locked knee.
    return {k: float(np.percentile(v, quantile) if k == "lockout" else np.percentile(np.abs(v), quantile))
            for k, v in errors.items()}
