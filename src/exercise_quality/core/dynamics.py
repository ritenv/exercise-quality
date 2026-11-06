"""Centre of pressure during movement (planar inverse dynamics of the whole body).

When the body accelerates, the ground has to push in a different place
than straight under the centre of mass. For a body made of rigid
segments on flat ground, taking moments about the centre of pressure
``p`` (all forces in the sagittal plane, ground at y = 0):

    p = ( sum m_i x_i (ydd_i + g) - sum m_i y_i xdd_i + sum I_i wd_i )
        / sum m_i (ydd_i + g)

with segment mass m_i, centre of mass (x_i, y_i), its accelerations
(xdd_i, ydd_i), sagittal moment of inertia I_i and angular acceleration
wd_i (counter-clockwise positive, x forward, y up). This is the standard
zero-moment-point expression. With no accelerations it reduces to the
centre-of-mass x, which is what the quasi-static generator enforces.

Moments of inertia use de Leva's sagittal radii of gyration:
I_i = m_i (r_i L_i)^2.

Accelerations are taken by finite differences of the generated
trajectory, so they are only as smooth as the sampling (100 fps is the
generator default).
"""

from dataclasses import dataclass

import numpy as np

from exercise_quality.core.balance import model_segments, segment_centres

GRAVITY = 9.81


def centre_of_pressure(masses, centres, orientations, inertias, dt):
    """Low-level COP from segment trajectories.

    masses: (S,), centres: (T, S, 2), orientations: (T, S) radians
    counter-clockwise, inertias: (S,), dt: sample period in seconds.
    """
    masses = np.asarray(masses, float)
    acc = np.gradient(np.gradient(centres, dt, axis=0, edge_order=2),
                      dt, axis=0, edge_order=2)
    omega_dot = np.gradient(np.gradient(np.unwrap(orientations, axis=0), dt,
                                        axis=0, edge_order=2),
                            dt, axis=0, edge_order=2)
    x, y = centres[..., 0], centres[..., 1]
    xdd, ydd = acc[..., 0], acc[..., 1]
    vertical = masses * (ydd + GRAVITY)
    numerator = (np.sum(vertical * x, axis=1)
                 - np.sum(masses * y * xdd, axis=1)
                 + np.sum(inertias * omega_dot, axis=1))
    return numerator / np.sum(vertical, axis=1)


@dataclass
class DynamicBalance:
    times: np.ndarray
    cop_x: np.ndarray          # (T,) metres
    com_x: np.ndarray          # (T,)
    cop_fraction: np.ndarray   # (T,) fraction of foot length from heel
    vertical_force: np.ndarray  # (T,) in body weights

    @property
    def max_cop_offset(self):
        """Largest |COP - COM| in metres."""
        return float(np.max(np.abs(self.cop_x - self.com_x)))

    @property
    def stays_on_foot(self):
        return bool(np.all((self.cop_fraction >= 0) & (self.cop_fraction <= 1)))


def analyse_dynamics(sequence):
    """Dynamic centre of pressure for a generated SquatSequence."""
    body = sequence.body
    segments = model_segments(body)
    names = list(segments)
    masses = np.array([segments[n].mass for n in names])
    centres_dict = segment_centres(body, sequence.landmarks)
    centres = np.stack([centres_dict[n] for n in names], axis=1)

    orientations, inertias = [], []
    for n in names:
        s = segments[n]
        v = sequence.landmarks[s.other] - sequence.landmarks[s.origin]
        orientations.append(np.arctan2(v[:, 1], v[:, 0]))
        length = np.linalg.norm(v[0])
        inertias.append(s.mass * (s.gyration_fraction * length) ** 2)
    orientations = np.stack(orientations, axis=1)

    dt = sequence.times[1] - sequence.times[0]
    cop = centre_of_pressure(masses, centres, orientations, np.array(inertias), dt)

    com = np.sum(masses[None, :, None] * centres, axis=1) / masses.sum()
    com_ydd = np.gradient(np.gradient(com[:, 1], dt, edge_order=2), dt, edge_order=2)
    return DynamicBalance(
        times=sequence.times,
        cop_x=cop,
        com_x=com[:, 0],
        cop_fraction=sequence.support.fraction(cop),
        vertical_force=1.0 + com_ydd / GRAVITY,
    )
