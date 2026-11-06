"""Natural postural sway as slow coloured noise.

People never stand perfectly still: the centre of pressure wanders a few
millimetres, mostly below 1 Hz. In the generator this is added as a small
random wobble of the point on the foot the body balances over; the
balance solver then moves the ankle and trunk to follow it, so the
visible sway comes out of the same physics as everything else.

Reference values (healthy young adults, quiet standing, eyes open):
Yamamoto T, Smith CE, Suzuki Y, et al. Universal and individual
characteristics of postural sway during quiet standing in healthy young
adults. Physiol Rep. 2015;3(3):e12329. doi:10.14814/phy2.12329
  * anteroposterior COP mean velocity: log10 0.798 +- 0.097 mm/s
    (about 6.3 mm/s)
  * anteroposterior median frequency (PF50): 0.329 +- 0.071 Hz
"""

import numpy as np

COP_MEAN_VELOCITY_MM_S = 10 ** 0.798
COP_MEDIAN_FREQUENCY_HZ = 0.329


def _lowpass_median_ratio():
    """Median of a PSD shaped 1 / (1 + (f/fc)^4), as a multiple of fc."""
    x = np.linspace(0, 50, 200001)
    cdf = np.cumsum(1.0 / (1.0 + x**4))
    return float(x[np.searchsorted(cdf, cdf[-1] / 2)])


_MEDIAN_RATIO = _lowpass_median_ratio()


GRAVITY = 9.81


def sway_noise(n, fps, rng, median_hz, pendulum_height=None):
    """Sway signal for the balance target.

    First a COP-like signal: zero-mean, unit-SD noise with a second-order
    low-pass spectrum whose median frequency is ``median_hz``. If
    ``pendulum_height`` (COM height, metres) is given, it is then passed
    through the inverted-pendulum relation between centre of pressure and
    centre of mass, COM = COP / (1 + (2 pi f)^2 h / g), because the body's
    COM is a smoothed response to the COP. The generator balances the COM
    over the returned signal, so the dynamic COP it produces comes back
    close to the COP-like signal. Without this step the COM wobble was too
    jerky and its accelerations swamped the COP (mean velocity ~100x too
    high).
    """
    if n < 4:
        return np.zeros(n)
    fc = median_hz / _MEDIAN_RATIO
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    gain = 1.0 / np.sqrt(1.0 + (freqs / fc) ** 4)
    gain[0] = 0.0  # no drift of the mean
    spectrum = (rng.normal(size=freqs.size) + 1j * rng.normal(size=freqs.size)) * gain
    cop_like = np.fft.irfft(spectrum, n=n)
    sd = cop_like.std()
    if sd == 0:
        return cop_like
    spectrum = spectrum / sd
    if pendulum_height is not None:
        spectrum = spectrum / (1.0 + (2 * np.pi * freqs) ** 2 * pendulum_height / GRAVITY)
    return np.fft.irfft(spectrum, n=n)


def cop_mean_velocity(cop_x, fps):
    """Mean absolute velocity of a COP trace, mm/s (input in metres)."""
    return float(np.mean(np.abs(np.diff(cop_x))) * fps * 1000.0)


def median_frequency(signal, fps):
    """Frequency below which half the (detrended) signal power lies, Hz."""
    x = np.asarray(signal, float) - np.mean(signal)
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fps)
    cdf = np.cumsum(power[1:])
    return float(freqs[1:][np.searchsorted(cdf, cdf[-1] / 2)])
