# Squat model: how it works and where every number comes from

This describes the balance-constrained squat generator (`exercises/squat/balanced.py`) and everything it depends on. Conventions (axes, angle signs) are in [conventions.md](conventions.md). Results against published data are in [validation.md](validation.md).

## Pipeline

1. **Body.** A real person is drawn from ANSUR II (`core/sagittal_body.py`). Their measured standing heights (lateral malleolus, tibiale, trochanterion, navel, C7, acromion) and limb lengths are converted to joint-centre positions with de Leva's landmark-to-joint offsets. Proportions are never randomised independently, so every body shape is one that was actually measured.
2. **Mass.** Each segment gets de Leva's sex-specific mass fraction and centre-of-mass location (`core/deleva.py`, `core/balance.py`). The trunk is split at the navel into a pelvis (de Leva's lower trunk) and a thorax (middle plus upper trunk combined). Left and right limbs overlap in the side view and are modelled as one limb with double mass.
3. **Style.** Each person gets a squat strategy drawn from published between-subject spreads (`exercises/squat/protocols.py`): depth, preferred ankle and trunk patterns, where they keep their weight on the foot, hip end range, arm position and tempo.
4. **Motion.** The knee follows a minimum-jerk curve in real seconds. The preferred ankle and trunk angles follow the knee through couplings fitted to measured curves.
5. **Balance.** For every frame, one correction is **solved** so the whole-body centre of mass sits over the person's target point on the foot. The correction is shared between the ankle and the trunk, so body proportions change both: longer thighs mean more ankle bend *and* more forward lean.
6. **Hip end range and spine.** The pelvis follows the thorax until the hip nears its end range. Past that point the hip only creeps further, the thigh carries the pelvis into posterior tilt, and the lower back rounds (Mata et al. 2021).
7. **Feasibility.** If someone can't reach the requested depth without falling backwards or rounding their back past its limit, they squat as deep as they can (`generate_feasible_squat`), as a real person would. The rep is noted `depth_limited_by_mobility`.
8. **Check.** The true centre of pressure, including accelerations and segment rotation, is computed (`core/dynamics.py`) to confirm the person would actually stay on their feet.

## Sources

| What | Source | How it was obtained |
|---|---|---|
| Segment mass fractions, centre-of-mass locations, radii of gyration, joint-centre offsets, trunk sub-segments | de Leva P. 1996, J Biomech 29(9):1223-1230, Tables 2 and 4 | Two independent secondary transcriptions that agree: BMClab/BMC `data/BSP_ZdeLeva.txt`, and samchon/automovie notes read from a scan of the paper. Mass fractions sum to 100.00% (men) and 99.99% (women). The stacked trunk sub-segments reproduce de Leva's whole-trunk centre of mass to 3 mm (men) and 0 mm (women). |
| Body dimensions, including navel height | ANSUR II (Gordon et al. 2015, NATICK/TR-15/007), 4082 men and 1986 women | Public-domain CSVs from the Penn State OPEN Design Lab. Subset in `src/exercise_quality/data/`. |
| Movement smoothness | Flash and Hogan 1985, J Neurosci 5(7):1688-1703 | Minimum-jerk profile, plus a via-point version for reps with no pause at the bottom. |
| Depth, ankle and trunk patterns and their spreads, weight position on the foot, tempo | Ishida et al. 2025, Orthop J Sports Med 13(3) | Full text via Europe PMC. Natural condition, 16 men. |
| Forward shift of the COM during descent, its correlation with ankle motion (R = 0.398), hip range and timing | Kasahara et al. 2024, J Hum Kinet 93:29-39 | Full text via Europe PMC. |
| Maximal-depth squat ranges including pelvis and lumbar spine, arms-forward protocol | Zawadka et al. 2020, J Sports Sci Med 19:500-507 | Journal PDF. |
| Hip angle when posterior pelvic tilt begins, and its spread | Mata et al. 2021, Int J Exerc Sci 14(1):912-918 | Full text. SD recovered from the 95% CI. |
| Hip end range causes posterior pelvic tilt and lumbar flexion; depth and trunk-tibia definitions | Straub and Powers 2024, Int J Sports Phys Ther 19(4):490-501 | Narrative review, used for mechanism and definitions only. |
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
| Trunk split at the navel, one lumbar joint | | Assumption. Real lumbar flexion is spread over five vertebrae. |
| Foot stays flat (no heel lift) | | Matches the "heels down" instruction in all the studies |
| Preferred ankle excursion (ankle / peak knee) | 0.278 ± 0.037 | Fitted (Ishida peak ankle / peak knee) |
| Ankle-knee coupling exponent | 1.27 ± 0.1 | Fitted mean (Ishida 60 and 90 deg points). The SD is an assumption. |
| Preferred trunk lean at Ishida's peak knee | 38.5 ± 7.3 deg | Data (Ishida peak trunk inclination) |
| Trunk-knee coupling exponent | 1.25 | Fitted (Ishida 60 and 90 deg points). Cross-check: predicts 35.7 deg at Kasahara's 107 deg knee vs 34.7 measured. |
| Share of the balance correction taken at the ankle | 0.5 | Assumption. Anything from 0.25 to 0.75 gives realistic spreads; 0 (trunk only) does not. |
| Weight position on the foot at the bottom | 36.3% ± 8% of foot length from heel | Data (Ishida natural COP) |
| COM shift from start to bottom | -5.0% ± 9.6% of foot length, correlated 0.398 with ankle | Data (Kasahara) |
| Lumbopelvic share before hip end range | 1.0 ± 0.05 | Fitted (first calibration). The data favour a neutral lower back until the hip runs out. |
| Hip end range onset (relative to standing) | see `protocols.HIP_LIMIT` | Fitted mean, Mata's SD. `scripts/calibrate_lumbopelvic.py` |
| Hip flexion gain past end range | see `protocols.HIP_END_RANGE_GAIN` | Fitted. Onset and gain are constrained together along a ridge, not separately. |
| Width of the smooth hand-over at hip and ankle limits | 8 deg and 4 deg | Assumption (keeps accelerations continuous) |
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
- **One lumbar joint.** The real lower back bends over five vertebrae, and its curvature changes a little even before the hip runs out. The model keeps it neutral until end range.
- **Side view only.** No knee valgus, stance width, foot turn-out or left-right asymmetry.
- **Quasi-static balance.** Each frame is balanced as if still. Dynamics are checked afterwards, not used to plan the motion. This is fine at normal tempos (centre of pressure within about 7 mm of the plan at 2 s per phase) but not for fast or bouncy reps.
- **Populations.** ANSUR II is US Army personnel. de Leva is young Russian PE students. The squat studies are young, healthy, mostly Japanese, Polish and US adults. Older, heavier, injured or very mobile people are not represented.
- **No muscles or joint torques**, so the model says nothing about effort, strength limits or loading.
