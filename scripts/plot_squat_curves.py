"""Joint angles, centre of mass and centre of pressure over one rep.

    PYTHONPATH=src python scripts/plot_squat_curves.py [output.png]
"""

import sys

import matplotlib

if len(sys.argv) > 1:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from exercise_quality.core.dynamics import analyse_dynamics
from exercise_quality.core.sagittal_body import BodyPopulation
from exercise_quality.exercises.squat import literature as lit
from exercise_quality.exercises.squat.balanced import SquatStyle, generate_squat

body = BodyPopulation.load().median_body("male")
seq = generate_squat(body, SquatStyle())
dyn = analyse_dynamics(seq)
m = seq.measured()
t = seq.times

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

ax = axes[0, 0]
for name, label in [("knee_flexion", "knee flexion"), ("hip_flexion", "hip flexion (trunk vs thigh)"),
                    ("ankle_dorsiflexion", "ankle dorsiflexion"), ("trunk_inclination", "trunk lean")]:
    ax.plot(t, m[name], label=label)
ax.set_title("Joint angles (deg)")
ax.legend(fontsize=8)

ax = axes[0, 1]
knee = m["knee_flexion"]
bottom = int(knee.argmax())
ax.plot(knee[:bottom], m["trunk_inclination"][:bottom], label="model trunk lean")
ax.plot(knee[:bottom], m["ankle_dorsiflexion"][:bottom], label="model ankle")
ishida = lit.ISHIDA_2025
for key, knee_at, colour in [("trunk_at_knee60", 60, "C0"), ("trunk_at_knee90", 90, "C0"),
                             ("trunk_peak", ishida["knee_peak"].mean, "C0"),
                             ("ankle_at_knee60", 60, "C1"), ("ankle_at_knee90", 90, "C1"),
                             ("ankle_peak", ishida["knee_peak"].mean, "C1")]:
    ax.errorbar(knee_at, ishida[key].mean, yerr=ishida[key].sd, fmt="o", color=colour,
                capsize=3)
ax.set_xlabel("knee flexion (deg), descent")
ax.set_title("Model (lines) vs Ishida 2025 mean +- SD (dots)")
ax.legend(fontsize=8)

ax = axes[1, 0]
ax.plot(t, 100 * seq.support.fraction(seq.com[:, 0]), label="COM over foot")
ax.plot(t, 100 * dyn.cop_fraction, "--", label="dynamic COP")
ax.axhline(100 * ishida["cop_fraction"].mean, color="0.6", lw=1, label="Ishida mean COP")
ax.set_ylim(0, 100)
ax.set_ylabel("% of foot length from heel")
ax.set_title("Balance")
ax.legend(fontsize=8)

ax = axes[1, 1]
ax.plot(t, seq.com[:, 1], label="COM height (m)")
ax.plot(t, dyn.vertical_force, label="vertical ground force (body weights)")
ax.set_title("Vertical")
ax.legend(fontsize=8)

for a in axes.flat:
    a.grid(alpha=0.2)
for a in (axes[0, 0], axes[1, 0], axes[1, 1]):
    a.set_xlabel("time (s)")

fig.suptitle(f"Median ANSUR II man ({body.stature:.2f} m), default parallel squat")
fig.tight_layout()
if len(sys.argv) > 1:
    fig.savefig(sys.argv[1], dpi=110)
    print(f"saved {sys.argv[1]}")
else:
    plt.show()
