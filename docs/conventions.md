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
| `neck` | 7th cervical vertebra | CERV |
| `vertex` | top of the head | VERT |
| `shoulder` | shoulder joint centre, on the trunk line below C7 | SJC |
| `elbow`, `wrist` | joint centres | EJC, WJC |
| `hand` | head of the 3rd metacarpal (knuckle) | MET3 |

Left and right sides overlap in this view, so there is one leg and one arm. Each carries the mass of both sides.

## Segment angles (absolute)

- Leg, trunk and head: **inclination from vertical-up**, positive when the top of the segment is forward of its bottom. A shank leaning over the toes is positive. A thigh in a squat points back from the knee to the hip, so it is negative (around -90 when horizontal).
- Arm segments: **angle from vertical-down**, positive when swung forward. 0 hangs straight down, 90 points forward, 180 points straight up.

## Joint angles (anatomical, zero in upright standing)

| Angle | Positive direction | Relation |
|---|---|---|
| `ankle_dorsiflexion` | shank tips forward over a flat foot | shank = ankle |
| `knee_flexion` | knee bends | thigh = shank - knee |
| `hip_flexion` | trunk folds towards the thigh | trunk = thigh + hip |
| `neck_flexion` | head tips forward relative to the trunk | head = trunk + neck |
| `shoulder_flexion` | arm raised forward relative to the trunk | upper arm = shoulder - trunk |
| `elbow_flexion` | forearm bends forward/up | forearm = upper arm + elbow |
| `wrist_flexion` | hand bends relative to the forearm | hand = forearm + wrist |

`hip_flexion` is the angle between the thigh and a rigid trunk that runs from the hips to C7. In real people that motion is shared between the hip joint, the pelvis and the lumbar spine. Studies that report "hip flexion" (pelvis vs thigh) give smaller numbers than this model's `hip_flexion`, so comparisons with them have to add lumbar and pelvic motion.

**Trunk inclination**, the forward lean of the hip-to-C7 line from vertical, is the quantity most squat studies report as "trunk lean" or "forward trunk inclination".

## Foot and support

The support region in the side view runs from the heel (`x = -heel_offset`) to the toe tip (`x = toe_offset`). Positions along it are reported as a **fraction of foot length from the heel** (0 = heel, 1 = toe tip). That's the same convention as the centre-of-pressure data in Ishida et al. (2025).
