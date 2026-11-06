# Conventions

These apply to the sagittal model (`core/sagittal_model.py` and everything built on it). The older V1/V2 generators use a simpler frame with the ankle on the floor and are kept only for reference.

## Frame and units

- 2D side view of a person facing **+x**.
- **x** points forward (heel to toe), **y** points up.
- Origin: on the floor directly below the stance ankle joint centre when that foot is flat. So the ankle is at `(0, ankle_height)`, the heel at `(-heel_offset, 0)`, the ball of the foot at `(ball_offset, 0)` and the toe tip at `(toe_offset, 0)`. The stance ball of the foot never moves.
- Lengths in metres, mass in kilograms, time in seconds, angles in **degrees** in every public API.

## Sides

The person faces +x and the camera looks at their **right** side, so `right_*` is the near side and `left_*` the far side. Limb landmarks and limb joint angles exist once per side (`right_knee`, `left_knee_flexion`). The two hip joint centres project onto the same point in this view, and so do the two shoulders. One foot is the **stance** foot, the root of the chain; the other foot goes wherever its angles put it. In a symmetric pose both sides land on top of each other.

## Landmarks

| Name | Meaning | de Leva label |
|---|---|---|
| `{side}_heel` | back of the heel | HEEL (pternion) |
| `{side}_ball` | ball of the foot, the toe joint (ANSUR II heel-to-ball length) | |
| `{side}_toe` | tip of the longest toe | TTIP |
| `{side}_ankle` | ankle joint centre, at lateral malleolus height | LMAL |
| `{side}_knee` | knee joint centre | KJC |
| `{side}_hip` | hip joint centre (both sides coincide; MIDH for the pelvis) | HJC / MIDH |
| `lumbar` | lower spine joint, at navel (omphalion) height | OMPH level |
| `thoracolumbar` | upper spine joint, at xiphoid height, where the rib cage starts | XYPH level |
| `neck` | 7th cervical vertebra | CERV |
| `vertex` | top of the head | VERT |
| `{side}_shoulder` | shoulder joint centre, on the thorax line below C7 | SJC |
| `{side}_elbow`, `{side}_wrist` | joint centres | EJC, WJC |
| `{side}_hand` | head of the 3rd metacarpal (knuckle) | MET3 |

## Segment angles (absolute)

- Leg, pelvis, abdomen, thorax and head: **inclination from vertical-up**, positive when the top of the segment is forward of its bottom. A shank leaning over the toes is positive. A thigh in a squat points back from the knee to the hip, so it is negative (around -90 when horizontal).
- Arm segments: **angle from vertical-down**, positive when swung forward. 0 hangs straight down, 90 points forward, 180 points straight up.
- Feet (rearfoot heel to ball, toes ball to tip): **pitch from horizontal**, positive when the back end is higher (heel up).

## Joint angles (anatomical, zero in upright standing)

Limb angles belong to a side (`JointAngles.right_leg.knee_flexion`; `right_knee_flexion` in `measure_joint_angles`). Without a side prefix they are the mean of both sides. `JointAngles.symmetric(...)` builds a pose with both sides equal.

| Angle | Positive direction | Relation (stance side) |
|---|---|---|
| `toe_extension` | toes bend up relative to the rearfoot; with toes flat, the heel lifts | rearfoot pitch = toe extension |
| `ankle_dorsiflexion` | shank tips forward over the rearfoot | shank = rearfoot + ankle |
| `knee_flexion` | knee bends | thigh = shank - knee |
| `hip_flexion` | pelvis tips forward relative to the thigh (true hip flexion) | pelvis = thigh + hip |
| `lower_lumbar_flexion` | abdomen bends forward on the pelvis (at the navel) | abdomen = pelvis + lower lumbar |
| `upper_lumbar_flexion` | rib cage bends forward on the abdomen (at the xiphoid) | thorax = abdomen + upper lumbar |
| `lumbar_flexion` | total of the two: thorax relative to pelvis (lower back rounds) | thorax = pelvis + lumbar |
| `neck_flexion` | head tips forward relative to the thorax | head = thorax + neck |
| `shoulder_flexion` | arm raised forward relative to the thorax | upper arm = shoulder - thorax |
| `elbow_flexion` | forearm bends forward/up | forearm = upper arm + elbow |
| `wrist_flexion` | hand bends relative to the forearm | hand = forearm + wrist |

The trunk is three rigid segments, de Leva's lower trunk (pelvis, hips to navel), middle trunk (abdomen, navel to xiphoid) and upper trunk (thorax, xiphoid to C7). When a pose gives only a total lumbar flexion, 35% goes to the upper joint (`UPPER_LUMBAR_SHARE`, from per-level lumbar motion ranges).

**Trunk inclination** (`trunk_inclination` from `measure_joint_angles`) is the forward lean of the **thorax** (the rib cage, xiphoid to C7) from vertical. That's the thoracic segment that squat studies report as "trunk lean" or "forward trunk inclination", and the segment marker sets such as Plug-in Gait call the thorax. **Lumbar flexion** is that thorax relative to the pelvis, the "lumbar spine angle" of those studies. **Pelvis inclination** is the forward tilt of the hip-to-navel line; standing is 0, anterior tilt is positive, and a drop after its peak is posterior pelvic tilt.

## Foot and support

The support region of a flat foot runs from the heel (`x = -heel_offset`) to the toe tip (`x = toe_offset`). Positions along it are reported as a **fraction of foot length from the heel** (0 = heel, 1 = toe tip). That's the same convention as the centre-of-pressure data in Ishida et al. (2025). With the heel lifted, only the ball and toes are on the floor; the squat generator keeps heels down, as every study it uses instructed.
