"""Body dimensions for the side-view (sagittal) model.

A ``SagittalBody`` holds the segment lengths between joint centres plus
the foot geometry needed for balance. Bodies are built from real people
in the ANSUR II survey: each measured landmark height is converted to a
joint-centre height with the de Leva (1996) Table 2 offsets, so the
proportions of every body come from one measured human, not from
independently randomised lengths.

ANSUR II
--------
Gordon CC et al. 2015, "2012 Anthropometric Survey of U.S. Army
Personnel: Methods and Summary Statistics", NATICK/TR-15/007. Public
domain data, 4082 men and 1986 women, distributed by the Penn State OPEN
Design Lab. US Army personnel are younger, fitter and more often male
than the general population, so the sample is not representative of
everyone. See ``exercise_quality/data/README.md``.

Assumptions (also listed in docs/model.md)
------------------------------------------
* Standing heights are treated as vertical distances between joint
  centres in a straight, upright posture. Real standing posture is not
  perfectly stacked (C7 sits a little behind the hips, the head a little
  in front), so the trunk and head lengths here are their vertical
  extents in upright standing, not true 3D segment lengths.
* The ankle joint centre is placed at the lateral malleolus height.
* The xiphoid (where the rib cage meets the abdomen) is not measured in
  ANSUR II. It is placed at de Leva's sample proportion of each person's
  navel-to-C7 distance (47%).
* The ball of the foot (the toe joint) is at ANSUR II's measured
  heel-to-ball length. That is measured to the first metatarsal head on
  the inside of the foot; the side view uses it as the toe joint axis.
* The heel extends 21% of foot length behind the ankle. This ratio is
  taken from the running model of a 2024 Scientific Reports paper on
  heel height ("Effects of shoe heel height on ankle dynamics in
  running"), which cites radiographic data. It has not been checked
  against the original radiographs, and it is the same for everyone.
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from exercise_quality.core import deleva

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "ansur_ii_sagittal.csv"

HEEL_FRACTION_OF_FOOT = 0.21

LENGTH_FIELDS = (
    "ankle_height", "heel_offset", "ball_length", "foot_length", "shank", "thigh",
    "pelvis", "abdomen", "thorax", "head", "shoulder_drop", "upper_arm", "forearm", "hand",
)


@dataclass(frozen=True)
class SagittalBody:
    """All lengths in metres, mass in kilograms.

    pelvis: hip joint centres up to the navel (omphalion) level, de Leva's
    lower trunk.
    abdomen: navel up to the xiphoid, de Leva's middle trunk.
    thorax: xiphoid up to C7 (the rib cage), de Leva's upper trunk.
    shoulder_drop: distance from C7 down the thorax line to the shoulder
    joint centre.
    heel_offset: horizontal distance the heel extends behind the ankle.
    ball_length: heel to the ball of the foot (the toe joint), along the
    sole.
    """

    sex: str
    stature: float
    mass: float
    ankle_height: float
    heel_offset: float
    ball_length: float
    foot_length: float
    shank: float
    thigh: float
    pelvis: float
    abdomen: float
    thorax: float
    head: float
    shoulder_drop: float
    upper_arm: float
    forearm: float
    hand: float
    subject_id: str = field(default="", compare=False)

    def __post_init__(self):
        deleva.segment_parameters(self.sex)  # validates sex
        for name in ("stature", "mass", *LENGTH_FIELDS):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite, got {value}")
        if self.shoulder_drop >= self.thorax:
            raise ValueError("shoulder joint must lie on the thorax, below C7")
        if not self.heel_offset < self.ball_length < self.foot_length:
            raise ValueError("the ball of the foot must lie between the ankle and the toe tip")

    @property
    def toe_offset(self):
        """Horizontal distance from the ankle forward to the toe tip."""
        return self.foot_length - self.heel_offset

    @property
    def ball_offset(self):
        """Horizontal distance from the ankle forward to the ball of the foot."""
        return self.ball_length - self.heel_offset

    @property
    def toe_length(self):
        """Ball of the foot to the toe tip."""
        return self.foot_length - self.ball_length

    @property
    def trunk(self):
        """Hip joint centres to C7 (pelvis, abdomen and thorax)."""
        return self.pelvis + self.abdomen + self.thorax

    @property
    def segments(self):
        return deleva.segment_parameters(self.sex)

    def segment_length(self, name):
        return getattr(self, name)

    @property
    def standing_hip_height(self):
        return self.ankle_height + self.shank + self.thigh


def body_from_measurements(m, sex, subject_id=""):
    """Build a body from one person's standing measurements (metres, kg).

    ``m`` needs: stature_m, lateral_malleolus_height_m, tibiale_height_m,
    trochanterion_height_m, omphalion_height_m, cervicale_height_m, acromion_height_m,
    acromion_radiale_length_m, radiale_stylion_length_m, hand_length_m,
    ball_of_foot_length_m, foot_length_m, mass_kg.
    """
    off = deleva.JOINT_CENTRE_OFFSETS_M[sex]

    ankle = m["lateral_malleolus_height_m"]
    knee = m["tibiale_height_m"] + off["knee_from_tibiale"]
    hip = m["trochanterion_height_m"] + off["hip_from_trochanterion"]
    # de Leva splits the trunk at the navel (omphalion); ANSUR II measures
    # its standing height directly
    navel = m["omphalion_height_m"]
    cervicale = m["cervicale_height_m"]
    xiphoid = navel + deleva.xiphoid_fraction(sex) * (cervicale - navel)
    shoulder = m["acromion_height_m"] + off["shoulder_from_acromion"]

    # Acromion-radiale is measured with the arm hanging. The shoulder
    # centre sits below the acromion and the elbow centre above the
    # radiale, so both offsets shorten the segment.
    upper_arm = (
        m["acromion_radiale_length_m"]
        + off["shoulder_from_acromion"]
        - off["elbow_from_radiale_upper_arm"]
    )
    # The elbow centre is above the radiale and the wrist centre just
    # below the stylion, so both offsets lengthen the forearm.
    forearm = (
        m["radiale_stylion_length_m"]
        + off["elbow_from_radiale_forearm"]
        - off["wrist_from_stylion"]
    )
    hand = m["hand_length_m"] * deleva.HAND_MET3_TO_DAC3_RATIO[sex]
    foot = m["foot_length_m"]

    return SagittalBody(
        sex=sex,
        stature=m["stature_m"],
        mass=m["mass_kg"],
        ankle_height=ankle,
        heel_offset=HEEL_FRACTION_OF_FOOT * foot,
        ball_length=m["ball_of_foot_length_m"],
        foot_length=foot,
        shank=knee - ankle,
        thigh=hip - knee,
        pelvis=navel - hip,
        abdomen=xiphoid - navel,
        thorax=cervicale - xiphoid,
        head=m["stature_m"] - cervicale,
        shoulder_drop=cervicale - shoulder,
        upper_arm=upper_arm,
        forearm=forearm,
        hand=hand,
        subject_id=str(subject_id),
    )


class BodyPopulation:
    """Real ANSUR II people, converted to sagittal bodies.

    Sampling draws whole people, so every combination of proportions is
    one that was actually measured.
    """

    def __init__(self, rows):
        self._rows = list(rows)
        if not self._rows:
            raise ValueError("population is empty")
        self._by_sex = {
            sex: [r for r in self._rows if r["sex"] == sex] for sex in deleva.SEXES
        }

    @classmethod
    def load(cls, path=DATA_FILE):
        with open(path, newline="") as f:
            rows = []
            for raw in csv.DictReader(f):
                row = {k: (v if k in ("subject_id", "sex") else float(v))
                       for k, v in raw.items()}
                rows.append(row)
        return cls(rows)

    def __len__(self):
        return len(self._rows)

    def count(self, sex):
        return len(self._by_sex[sex])

    def body(self, sex, index):
        row = self._by_sex[sex][index]
        return body_from_measurements(row, sex, row["subject_id"])

    def bodies(self, sex):
        return [self.body(sex, i) for i in range(self.count(sex))]

    def sample(self, rng, sex=None, female_share=0.5):
        """Draw one real person. ``sex=None`` picks a sex at random first."""
        if sex is None:
            sex = "female" if rng.random() < female_share else "male"
        return self.body(sex, int(rng.integers(self.count(sex))))

    def median_measurements(self, sex):
        rows = self._by_sex[sex]
        keys = [k for k in rows[0] if k not in ("subject_id", "sex")]
        return {k: float(np.median([r[k] for r in rows])) for k in keys}

    def median_body(self, sex):
        """A typical body from the per-measure medians (not a real person)."""
        return body_from_measurements(
            self.median_measurements(sex), sex, f"median-{sex}"
        )
