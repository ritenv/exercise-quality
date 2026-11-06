# Squat model: how it works and where every number comes from

This describes the balance-constrained squat generator (`exercises/squat/balanced.py`) and everything it depends on. Conventions (axes, angle signs) are in [conventions.md](conventions.md). Results against published data are in [validation.md](validation.md).

## Pipeline

1. **Body.** A real person is drawn from ANSUR II (`core/sagittal_body.py`). Their measured standing heights (lateral malleolus, tibiale, trochanterion, navel, C7, acromion), heel-to-ball length and limb lengths are converted to joint-centre positions with de Leva's landmark-to-joint offsets. Proportions are never randomised independently, so every body shape is one that was actually measured.
2. **Mass.** Each segment gets de Leva's sex-specific mass fraction and centre-of-mass location (`core/deleva.py`, `core/balance.py`). The trunk is de Leva's three sub-segments: pelvis (hips to navel), abdomen (navel to xiphoid) and thorax (the rib cage, xiphoid to C7). Left and right limbs are separate segments, each with its own mass; in a symmetric squat they overlap in the side view. The foot is split at the ball into rearfoot and toes, with all of de Leva's foot mass on the rearfoot.
3. **Style.** Each person gets a squat strategy drawn from published between-subject spreads (`exercises/squat/protocols.py`): depth, preferred ankle and trunk patterns, where they keep their weight on the foot, hip end range, arm position and tempo.
4. **Motion.** The knee follows a minimum-jerk curve in real seconds. The preferred ankle and trunk angles follow the knee through couplings fitted to measured curves.
5. **Balance.** For every frame, one correction is **solved** so the whole-body centre of mass sits over the person's target point on the foot. The correction is shared between the ankle and the trunk, so body proportions change both: longer thighs mean more ankle bend *and* more forward lean.
6. **Hip end range and spine.** The balance solver steers the rib cage, the segment squat studies measure as the trunk. The pelvis follows it until the hip nears its end range. Past that point the hip only creeps further, the thigh carries the pelvis into posterior tilt, and the lower back rounds (Mata et al. 2021). The rounding is shared between the two spine joints.
7. **Feasibility.** If someone can't reach the requested depth without falling backwards or rounding their back past its limit, they squat as deep as they can (`generate_feasible_squat`), as a real person would. The rep is noted `depth_limited_by_mobility`.
8. **Sway.** A slow random wobble of the balance target (shaped like quiet-standing centre-of-pressure data and passed through the inverted-pendulum relation to the centre of mass) keeps people from standing unnaturally still. The solver turns it into small ankle and trunk motion.
9. **Check.** The true centre of pressure, including accelerations and segment rotation, is computed (`core/dynamics.py`) to confirm the person would actually stay on their feet.

## Sources

| What | Source | How it was obtained |
|---|---|---|
| Segment mass fractions, centre-of-mass locations, radii of gyration, joint-centre offsets, trunk sub-segments | de Leva P. 1996, J Biomech 29(9):1223-1230, Tables 2 and 4 | Two independent secondary transcriptions that agree: BMClab/BMC `data/BSP_ZdeLeva.txt`, and samchon/automovie notes read from a scan of the paper. Mass fractions sum to 100.00% (men) and 99.99% (women). The stacked trunk sub-segments reproduce de Leva's whole-trunk centre of mass to 3 mm (men) and 0 mm (women). |
| Body dimensions, including navel height and heel-to-ball length | ANSUR II (Gordon et al. 2015, NATICK/TR-15/007), 4082 men and 1986 women | Public-domain CSVs from the Penn State OPEN Design Lab. Subset in `src/exercise_quality/data/`. |
| Movement smoothness | Flash and Hogan 1985, J Neurosci 5(7):1688-1703 | Minimum-jerk profile, plus a via-point version for reps with no pause at the bottom. |
| Depth, ankle and trunk patterns and their spreads, weight position on the foot, tempo | Ishida et al. 2025, Orthop J Sports Med 13(3) | Full text via Europe PMC. Natural condition, 16 men. |
| Forward shift of the COM during descent, its correlation with ankle motion (R = 0.398), hip range and timing | Kasahara et al. 2024, J Hum Kinet 93:29-39 | Full text via Europe PMC. |
| Maximal-depth squat ranges including pelvis and lumbar spine, arms-forward protocol | Zawadka et al. 2020, J Sports Sci Med 19:500-507 | Journal PDF. |
| Hip angle when posterior pelvic tilt begins, and its spread | Mata et al. 2021, Int J Exerc Sci 14(1):912-918 | Full text. SD recovered from the 95% CI. |
| Hip end range causes posterior pelvic tilt and lumbar flexion; depth and trunk-tibia definitions | Straub and Powers 2024, Int J Sports Phys Ther 19(4):490-501 | Narrative review, used for mechanism and definitions only. |
| Quiet-standing centre-of-pressure position (31-58% of foot length) | Fujiwara et al. 2010, J Physiol Anthropol 29(5):197-203 | Abstract (range only). |
| Quiet-standing sway: centre-of-pressure mean velocity and median frequency | Yamamoto et al. 2015, Physiol Rep 3(3):e12329 | Full text via Europe PMC. |
| Peak vertical ground force similar across squat depths; sharper bottom transition when fast | Hamed et al. 2025, ISBS Proceedings 43(1) art. 29 | Abstract only (qualitative check). |
| Maximum knee flexion in squatting | Hemmerich et al. 2006, J Orthop Res 24:770-781 | **Abstract only.** |
| Heel position behind the ankle (21% of foot length) | 2024 Scientific Reports paper on shoe heel height, citing radiographic data | **Secondary only**, not checked against the original radiographs. |

## Assumptions and their status

"Data" means taken directly from a source above. "Fitted" means derived from published numbers by a stated calculation. "Assumption" means no data was used.

| Item | Value | Status |
|---|---|---|
| Joint centre heights from landmark heights | de Leva Table 2 offsets | Data |
| Ankle joint centre at lateral malleolus height | | Assumption (common convention) |
| Heel behind ankle | 21% of foot length, same for everyone | Secondary source |
| Upright standing is perfectly stacked (C7 above the navel above the hips) | | Assumption. Real posture has spinal curves. |
| Trunk split at the navel and xiphoid, two spine joints | | de Leva's sub-segment boundaries. Real flexion is spread over many vertebrae. |
| Xiphoid height | 47% of navel-to-C7 | de Leva's sample proportion; ANSUR II has no xiphoid height |
| Share of lumbar flexion at the upper (xiphoid) joint | 0.35 | Fitted from per-level lumbar motion ranges (Cheng et al. 2013, Table 3, with Pearcy 1984, Dvorak 1991, White and Panjabi 1990): 34-38% at L1-L3. L5-S1 taken as equal to L4-L5. Using motion ranges as the squat split is an assumption. |
| Toe joint position | ANSUR II heel-to-ball length (74% of foot length) | Data. Measured to the first metatarsal head; used as the side-view toe axis. |
| Foot mass all on the rearfoot | | Assumption. de Leva gives no toe split; toes are a small part of foot mass. |
| Camera on the person's right | | Convention. Right limbs are the near side. |
| Foot stays flat (no heel lift) | | Matches the "heels down" instruction in all the studies |
| Preferred ankle excursion (ankle / peak knee) | 0.278 ± 0.037 | Fitted (Ishida peak ankle / peak knee) |
| Ankle-knee coupling exponent | 1.27 ± 0.1 | Fitted mean (Ishida 60 and 90 deg points). The SD is an assumption. |
| Preferred trunk lean at Ishida's peak knee | 38.5 ± 7.3 deg | Data (Ishida peak trunk inclination) |
| Trunk-knee coupling exponent | 1.25 | Fitted (Ishida 60 and 90 deg points). The curve is spread over each person's own knee range and sized from Ishida's curve at their depth; shaping it over the absolute knee angle gave infinite curvature at 121.8 deg and spikes in the ground force. |
| Share of the balance correction taken at the ankle | 0.5 | Assumption. Anything from 0.25 to 0.75 gives realistic spreads; 0 (trunk only) does not. |
| Weight position on the foot at the bottom | 36.3% ± 8% of foot length from heel | Data (Ishida natural COP) |
| COM shift from start to bottom | -5.0% ± 9.6% of foot length, correlated 0.398 with ankle | Data (Kasahara) |
| Standing weight position | 41.3% ± 6.75% of foot length | Mean from Ishida minus Kasahara's shift; SD from Fujiwara's 31-58% range read as ±2 SD (assumption) |
| Correlations between standing position, shift and ankle use | standing-shift -0.57, ankle-standing -0.23 | Derived so the Ishida, Kasahara and Fujiwara statistics all hold at once; ankle-standing assumes the ankle acts only through the shift |
| Depth for the Kasahara protocol | Ishida's 121.8 ± 5.7 deg knee | Same lab and "thighs parallel" instruction. Kasahara's own 107 deg is a range of motion from the start angle; as a peak it left thighs 18 deg above parallel and the COM drop 1.9 SD low. |
| Lumbopelvic share before hip end range | 0.9 ± 0.05 | Fitted together with the end-range onset and gain (`scripts/calibrate_lumbopelvic.py`). The lower back takes about 10% of the flexion before end range. |
| Hip end range onset (relative to standing) | see `protocols.HIP_LIMIT` | Fitted mean, Mata's SD. `scripts/calibrate_lumbopelvic.py` |
| Hip flexion gain past end range | see `protocols.HIP_END_RANGE_GAIN` | Fitted. Onset and gain are constrained together along a ridge, not separately. |
| Width over which the hip stiffens into end range | 25 deg | Assumption. 8 deg made the hip accelerate about 4x faster than the knee and put bumps in the ground force. |
| Width over which the ankle stiffens into its limit | 20 deg | Assumption. 4 deg acted like a wall: in a 1500-rep stress test the trunk took over abruptly, force jerk reached 19 BW/s and 3 people's centre of pressure left the foot. |
| Style draws truncated | ±2.5 SD | Assumption. Published SDs come from small samples and say nothing about the far tails. |
| Postural sway | 2 mm COP-like wobble, 0.33 Hz median, per-person spread from Yamamoto | Fitted: reproduces Yamamoto's quiet-standing COP mean velocity (6.3 mm/s) and median frequency (0.33 Hz). Applying the same sway during movement is an assumption. |
| Lumbar flexion cap | 70 deg | Assumption (flags extreme rounding only) |
| Standing posture | lean at the ankle until balanced, hips straight | Assumption (quiet standing is mostly ankle-controlled) |
| Head follows trunk lean | 30-70% | Assumption |
| Arms forward | parallel to the floor, elbows straight | Protocol (Zawadka) |
| Arms crossed | 20 deg shoulder, 150 deg elbow | Assumption. A side-view projection of a 3D posture, chosen to put arm mass against the chest. |
| Knee limit | 157 ± 6 deg | Data (Hemmerich abstract) |
| Ankle cap | 45 deg | Assumption |
| Tempo | 2 s down, 2 s up | Protocol (Ishida, Kasahara). The ±25% jitter on the Zawadka protocol is an assumption. |
| Ascent mirrors descent | | Assumption, no ascent data used |
| Depth reduction when infeasible | 2 deg steps, down to 60 deg | Assumption about behaviour (people stop where they can) |

## Known limitations

- **Preferences come from outcome data.** Each person's preferred ankle and trunk patterns use the published *outcome* means and spreads. Balance then adds a correction on top, so simulated ankle and trunk means come out a few degrees above the published ones (see validation.md). Re-centring the preferences would remove this, but only by fitting those means away.
- **Two spine joints.** The real back bends over many vertebrae. The model bends at the navel and the xiphoid, de Leva's mass boundaries, and nothing below the navel (L4-S1 sits inside the "pelvis"). Lumbar flexion as studies measure it (rib cage relative to pelvis) is unaffected by where the joints are, because the rib-cage lean comes from balance and the pelvis from the hip end range; see validation.md, gap 1.
- **Trunk lean past parallel.** The preferred trunk lean is held at Ishida's value for squats deeper than Ishida's (121.8 deg knee). No study used here reports how lean grows past parallel, so deep squats lean no further than parallel ones except through balance.
- **Side view only.** No knee valgus, stance width, foot turn-out or left-right asymmetry.
- **Quasi-static balance.** Each frame is balanced as if still. Dynamics are checked afterwards, not used to plan the motion. This is fine at normal tempos (centre of pressure within about 7 mm of the plan at 2 s per phase) but not for fast or bouncy reps.
- **Populations.** ANSUR II is US Army personnel. de Leva is young Russian PE students. The squat studies are young, healthy, mostly Japanese, Polish and US adults. Older, heavier, injured or very mobile people are not represented.
- **No muscles or joint torques**, so the model says nothing about effort, strength limits or loading.
