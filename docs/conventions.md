# Conventions

These apply to the sagittal model (`core/sagittal_model.py` and everything built on it). The older V1/V2 generators use a simpler frame with the ankle on the floor and are kept only for reference.

## Frame and units

- 2D side view of a person facing **+x**.
- **x** points forward (heel to toe), **y** points up.
- Origin: on the floor directly below the ankle joint centre. So the ankle is at `(0, ankle_height)`, the heel at `(-heel_offset, 0)` and the toe tip at `(toe_offset, 0)`.
- Lengths in metres, mass in kilograms, time in seconds, angles in **degrees** in every public API.

## Landmarks

| Name | Meaning | de Leva label |
|---|---|---|
| `heel` | back of the heel on the floor | HEEL (pternion) |
| `toe` | tip of the longest toe on the floor | TTIP |
| `ankle` | ankle joint centre, at lateral malleolus height | LMAL |
| `knee` | knee joint centre | KJC |
| `hip` | midpoint of the two hip joint centres | MIDH |
| `lumbar` | lumbar joint, at navel (omphalion) height on the trunk | OMPH level |
| `neck` | 7th cervical vertebra | CERV |
| `vertex` | top of the head | VERT |
| `shoulder` | shoulder joint centre, on the thorax line below C7 | SJC |
| `elbow`, `wrist` | joint centres | EJC, WJC |
| `hand` | head of the 3rd metacarpal (knuckle) | MET3 |

Left and right sides overlap in this view, so there is one leg and one arm. Each carries the mass of both sides.

## Segment angles (absolute)

- Leg, pelvis, thorax and head: **inclination from vertical-up**, positive when the top of the segment is forward of its bottom. A shank leaning over the toes is positive. A thigh in a squat points back from the knee to the hip, so it is negative (around -90 when horizontal).
- Arm segments: **angle from vertical-down**, positive when swung forward. 0 hangs straight down, 90 points forward, 180 points straight up.

## Joint angles (anatomical, zero in upright standing)

| Angle | Positive direction | Relation |
|---|---|---|
| `ankle_dorsiflexion` | shank tips forward over a flat foot | shank = ankle |
| `knee_flexion` | knee bends | thigh = shank - knee |
| `hip_flexion` | pelvis tips forward relative to the thigh (true hip flexion) | pelvis = thigh + hip |
| `lumbar_flexion` | thorax bends forward relative to the pelvis (lower back rounds) | thorax = pelvis + lumbar |
| `neck_flexion` | head tips forward relative to the thorax | head = thorax + neck |
| `shoulder_flexion` | arm raised forward relative to the thorax | upper arm = shoulder - thorax |
| `elbow_flexion` | forearm bends forward/up | forearm = upper arm + elbow |
| `wrist_flexion` | hand bends relative to the forearm | hand = forearm + wrist |

The trunk is two rigid segments split at navel height, following de Leva's lower trunk (pelvis) and middle plus upper trunk (thorax). Real lumbar flexion is spread over five vertebrae; one joint is the simplest split that matches the mass data.

**Trunk inclination** (`trunk_inclination` from `measure_joint_angles`) is the forward lean of the **thorax** (navel level to C7) from vertical, the quantity squat studies report as "trunk lean" or "forward trunk inclination". **Pelvis inclination** is the forward tilt of the hip-to-navel line; standing is 0, anterior tilt is positive, and a drop after its peak is posterior pelvic tilt.

## Foot and support

The support region in the side view runs from the heel (`x = -heel_offset`) to the toe tip (`x = toe_offset`). Positions along it are reported as a **fraction of foot length from the heel** (0 = heel, 1 = toe tip). That's the same convention as the centre-of-pressure data in Ishida et al. (2025).
