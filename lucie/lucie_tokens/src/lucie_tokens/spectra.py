"""Sphere-aware spectral diagnostics for the dynamics model.

The 48x96 grid is periodic and uniform in longitude but a Gaussian
latitude-longitude grid in latitude (poles, non-uniform cell area), so a plain
2D FFT is wrong in latitude. We instead take the FFT along longitude only
(exact there), drop the Nyquist mode, and average the resulting zonal power
over latitude with the Gaussian quadrature weights. This directly exposes how
much small-scale (high zonal-wavenumber) power the emulator retains vs truth.

Pure numpy so it can be unit-tested without the model.
"""

from __future__ import annotations

import numpy as np

# STATE_CHANNELS layout: SH 0-7, Temperature 8-15, U-wind 16-23, V-wind 24-31, logp 32.
U_WIND_CHANNELS = slice(16, 24)
V_WIND_CHANNELS = slice(24, 32)


def zonal_power_spectrum(
    field: np.ndarray, latitude_weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Zonal (longitude) power spectrum, Nyquist removed, latitude-weighted.

    field: (..., H, W); leading dims (time, level, ...) are averaged. Returns
    (wavenumbers, power) where wavenumbers = 0..W//2-1 (Nyquist dropped for even W).
    """
    values = np.asarray(field, dtype=np.float64)
    height, width = values.shape[-2:]
    transform = np.fft.rfft(values, axis=-1)
    power = np.abs(transform) ** 2
    modes = power.shape[-1]
    # One-sided normalization: double all but DC and Nyquist so the summed power
    # equals the spatial variance; the pred/truth ratio is insensitive to it, but
    # keep it correct so absolute spectra are physically meaningful.
    onesided = np.full(modes, 2.0)
    onesided[0] = 1.0
    if width % 2 == 0:
        onesided[-1] = 1.0
    power = power * onesided / (width * width)
    weights = np.asarray(latitude_weights, dtype=np.float64)
    if weights.shape[0] != height:
        raise ValueError(f"latitude_weights length {weights.shape[0]} != height {height}")
    weights = weights / weights.sum()
    # Weighted average over the latitude axis (-2), then mean over leading dims.
    latitude_averaged = np.tensordot(power, weights, axes=([-2], [0]))
    spectrum = latitude_averaged.reshape(-1, modes).mean(axis=0)
    # Drop the Nyquist mode (aliasing-prone, not a meaningful resolved scale).
    if width % 2 == 0:
        return np.arange(modes - 1), spectrum[:-1]
    return np.arange(modes), spectrum


def kinetic_energy_spectrum(
    state: np.ndarray, latitude_weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """0.5*(E_u + E_v) zonal spectrum averaged over the 8 wind levels.

    state: (..., C, H, W) with the STATE_CHANNELS layout.
    """
    u = np.asarray(state)[..., U_WIND_CHANNELS, :, :]
    v = np.asarray(state)[..., V_WIND_CHANNELS, :, :]
    wavenumbers, energy_u = zonal_power_spectrum(u, latitude_weights)
    _, energy_v = zonal_power_spectrum(v, latitude_weights)
    return wavenumbers, 0.5 * (energy_u + energy_v)


def spectrum_metrics(
    predicted: np.ndarray, truth: np.ndarray, high_fraction: float = 0.5
) -> dict:
    """Compare two spectra: full ratio curve, relative L2, and the fraction of
    high-wavenumber (small-scale) power the prediction retains."""
    predicted = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    ratio = np.divide(
        predicted, truth, out=np.full_like(predicted, np.nan), where=truth > 0.0
    )
    relative_l2 = float(
        np.linalg.norm(predicted - truth) / np.linalg.norm(truth)
    ) if np.linalg.norm(truth) > 0 else float("nan")
    cut = max(1, int(round(len(truth) * (1.0 - high_fraction))))
    high_predicted = predicted[cut:].sum()
    high_truth = truth[cut:].sum()
    retention = float(high_predicted / high_truth) if high_truth > 0 else float("nan")
    return {
        "relative_l2": relative_l2,
        "high_wavenumber_retention": retention,
        "high_wavenumber_cut": int(cut),
        "ratio_by_wavenumber": [float(x) for x in ratio],
    }


class SpectrumAccumulator:
    """Online mean of per-frame spectra so a long rollout never has to be held
    in memory. Feed frames (C,H,W) or batches (N,C,H,W)."""

    def __init__(self, latitude_weights: np.ndarray) -> None:
        self.weights = np.asarray(latitude_weights, dtype=np.float64)
        self._per_channel_sum: np.ndarray | None = None
        self._ke_sum: np.ndarray | None = None
        self.count = 0

    def update(self, frames: np.ndarray) -> None:
        frames = np.asarray(frames, dtype=np.float64)
        if frames.ndim == 3:
            frames = frames[None]
        n, channels = frames.shape[0], frames.shape[1]
        # Per-channel zonal spectrum, averaged over this batch of frames.
        _, ke = kinetic_energy_spectrum(frames, self.weights)
        per_channel = np.stack(
            [
                zonal_power_spectrum(frames[:, c], self.weights)[1]
                for c in range(channels)
            ]
        )
        if self._per_channel_sum is None:
            self._per_channel_sum = per_channel * n
            self._ke_sum = ke * n
        else:
            self._per_channel_sum += per_channel * n
            self._ke_sum += ke * n
        self.count += n

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count == 0:
            raise ValueError("no frames accumulated")
        return self._per_channel_sum / self.count, self._ke_sum / self.count


def area_weighted_moments(
    frames: np.ndarray, latitude_weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel area-weighted global mean and std over frames (..., C, H, W).
    Used for stability (blow-up / variance-collapse) and climatology checks."""
    values = np.asarray(frames, dtype=np.float64)
    weights = np.asarray(latitude_weights, dtype=np.float64)
    weights = weights / weights.sum()
    channels = values.shape[-3]
    flat = values.reshape(-1, channels, values.shape[-2], values.shape[-1])
    # weight per (lat, lon): spread latitude weight uniformly over longitude
    cell = np.broadcast_to(
        weights[:, None] / values.shape[-1], (values.shape[-2], values.shape[-1])
    )
    cell = cell / cell.sum()
    mean = (flat * cell).sum(axis=(-2, -1)).mean(axis=0)
    var = (((flat - mean[None, :, None, None]) ** 2) * cell).sum(axis=(-2, -1)).mean(axis=0)
    return mean, np.sqrt(var)
