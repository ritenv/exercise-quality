"""Filtering for tracked landmark data.

Marker and video kinematics of whole-body movements are conventionally
low-pass filtered with a zero-lag (forward-backward) Butterworth filter
at about 6 Hz (Winter, Biomechanics and Motor Control of Human Movement,
4th ed. 2009, ch. 2). A squat's own content sits well below that (one
rep takes 2-4 s), so the filter removes tracker jitter without bending
the movement.
"""

import numpy as np
from scipy.signal import butter, filtfilt

DEFAULT_CUTOFF_HZ = 6.0


def lowpass(values, fps, cutoff_hz=DEFAULT_CUTOFF_HZ, order=2):
    """Zero-lag Butterworth low-pass along axis 0. NaNs are interpolated first."""
    x = np.asarray(values, dtype=float)
    if x.shape[0] < 3 * (order + 1) * 2:
        return x.copy()
    x = fill_nan(x)
    b, a = butter(order, cutoff_hz / (fps / 2.0))
    return filtfilt(b, a, x, axis=0)


def fill_nan(values):
    """Linear interpolation over NaN runs along axis 0 (edges held)."""
    x = np.array(values, dtype=float)
    flat = x.reshape(x.shape[0], -1)
    t = np.arange(x.shape[0])
    for j in range(flat.shape[1]):
        col = flat[:, j]
        bad = np.isnan(col)
        if bad.all() or not bad.any():
            continue
        col[bad] = np.interp(t[bad], t[~bad], col[~bad])
    return flat.reshape(x.shape)
