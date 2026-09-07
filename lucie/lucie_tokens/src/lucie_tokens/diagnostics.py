"""Every diagnostic the project computes on a trained tokenizer + dynamics model.

Field space
  zonal spectra on the sphere (longitude FFT, Nyquist dropped, Gaussian-latitude
  weighted), pixel-value histograms + EMD, area-weighted moments, and
  ``climate_diagnostics`` -- the standard long-rollout report (rollout vs
  tokenizer reconstruction vs truth) the Bridges diag chains write.

Token space
  per-scale marginal entropy, the rollout MECHANISM measurement (teacher-forced
  vs rollout predictive entropy, P(stay), unseen parent->child co-occurrence),
  the SCALE ORACLE (closed-loop rollouts with the coarsest k scales forced to
  truth), coarse-token Markov chain statistics, code-reuse statistics, and
  gust2-style marginal-entropy temperature calibration.

Support
  token-file loading, the forcing driver for conditioned models, one shared
  rollout iterator, de-normalisation metadata, relative vorticity, ERA5 truth
  references, checkpoint autopsy. Presentation (videos, montages, curve plots)
  lives in ``scripts/analyze.py``, which is the CLI over this module.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterator, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import SourceClimateFrameDataset, load_frame_grid
from .dynamics import sample_next
from .runtime import (
    atomic_json,
    autocast_context,
    resolve_device,
    resolve_precision,
)
from .schema import STATE_CHANNELS, scale_offsets, split_flat_tokens
from .tokens import load_tokenizer_checkpoint
from .workflows import load_dynamics_checkpoint

LOG2 = float(np.log(2.0))

# ---------------------------------------------------------------------------
# Spectra on the sphere
# ---------------------------------------------------------------------------
# The 48x96 grid is periodic and uniform in longitude but Gaussian in latitude
# (poles, non-uniform cell area), so a plain 2D FFT is wrong in latitude. Take
# the FFT along longitude only (exact there), drop the Nyquist mode, and average
# the zonal power over latitude with the quadrature weights.

# STATE_CHANNELS layout: SH 0-7, Temperature 8-15, U-wind 16-23, V-wind 24-31,
# logp 32. Defaults for the 33-channel Bridges source; other datasets (the
# 5-channel ERA5 T30 set) pass explicit slices.
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
    latitude_averaged = np.tensordot(power, weights, axes=([-2], [0]))
    spectrum = latitude_averaged.reshape(-1, modes).mean(axis=0)
    if width % 2 == 0:
        return np.arange(modes - 1), spectrum[:-1]
    return np.arange(modes), spectrum


def kinetic_energy_spectrum(
    state: np.ndarray,
    latitude_weights: np.ndarray,
    u_channels: slice = U_WIND_CHANNELS,
    v_channels: slice = V_WIND_CHANNELS,
) -> tuple[np.ndarray, np.ndarray]:
    """0.5*(E_u + E_v) zonal spectrum averaged over the wind levels of (..., C, H, W)."""
    u = np.asarray(state)[..., u_channels, :, :]
    v = np.asarray(state)[..., v_channels, :, :]
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

    def __init__(
        self,
        latitude_weights: np.ndarray,
        u_channels: slice = U_WIND_CHANNELS,
        v_channels: slice = V_WIND_CHANNELS,
    ) -> None:
        self.weights = np.asarray(latitude_weights, dtype=np.float64)
        self.u_channels = u_channels
        self.v_channels = v_channels
        self._per_channel_sum: np.ndarray | None = None
        self._ke_sum: np.ndarray | None = None
        self.count = 0

    def update(self, frames: np.ndarray) -> None:
        frames = np.asarray(frames, dtype=np.float64)
        if frames.ndim == 3:
            frames = frames[None]
        n, channels = frames.shape[0], frames.shape[1]
        _, ke = kinetic_energy_spectrum(
            frames, self.weights, self.u_channels, self.v_channels
        )
        per_channel = np.stack(
            [zonal_power_spectrum(frames[:, c], self.weights)[1] for c in range(channels)]
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
    """Per-channel area-weighted global mean and std over frames (..., C, H, W)."""
    values = np.asarray(frames, dtype=np.float64)
    weights = np.asarray(latitude_weights, dtype=np.float64)
    weights = weights / weights.sum()
    channels = values.shape[-3]
    flat = values.reshape(-1, channels, values.shape[-2], values.shape[-1])
    cell = np.broadcast_to(
        weights[:, None] / values.shape[-1], (values.shape[-2], values.shape[-1])
    )
    cell = cell / cell.sum()
    mean = (flat * cell).sum(axis=(-2, -1)).mean(axis=0)
    var = (((flat - mean[None, :, None, None]) ** 2) * cell).sum(axis=(-2, -1)).mean(axis=0)
    return mean, np.sqrt(var)


# ---------------------------------------------------------------------------
# Pixel histograms
# ---------------------------------------------------------------------------
# Fixed bins in normalized (per-channel sigma) units so truth/tokenizer/inference
# are directly comparable and precomputed truth histograms stay compatible.
HIST_EDGES = np.linspace(-6.0, 6.0, 121)


class PixelHistogram:
    """Per-channel histogram of normalized pixel values over many frames.
    Values outside +-6 sigma are clipped into the edge bins."""

    def __init__(self, channels: int) -> None:
        self.counts = np.zeros((channels, len(HIST_EDGES) - 1), dtype=np.int64)

    def update(self, frames: np.ndarray) -> None:
        values = np.asarray(frames, dtype=np.float64)
        if values.ndim == 3:
            values = values[None]
        values = np.clip(values, HIST_EDGES[0], HIST_EDGES[-1] - 1e-9)
        for c in range(self.counts.shape[0]):
            self.counts[c] += np.histogram(values[:, c], bins=HIST_EDGES)[0]

    def density(self) -> np.ndarray:
        widths = np.diff(HIST_EDGES)
        totals = self.counts.sum(axis=1, keepdims=True).clip(min=1)
        return self.counts / totals / widths


def histogram_emd(a_counts: np.ndarray, b_counts: np.ndarray) -> float:
    """Earth-mover distance between two single-channel histograms, in sigma
    units (integral of |CDF difference|)."""
    widths = np.diff(HIST_EDGES)
    a = a_counts / max(a_counts.sum(), 1)
    b = b_counts / max(b_counts.sum(), 1)
    return float(np.sum(np.abs(np.cumsum(a) - np.cumsum(b)) * widths))


# ---------------------------------------------------------------------------
# Per-scale token entropy (gust2-style marginal-entropy readout)
# ---------------------------------------------------------------------------
# The emitted-token marginal aggregated over the rollout horizon, per scale,
# against the same statistic on the data tokens. Cold sampling drains entropy
# while every token stays individually legal, so this catches quiet collapse.


def entropy_bits(counts: np.ndarray) -> np.ndarray:
    """Plugin (empirical) entropy in bits along the last axis of a count array."""
    counts = np.asarray(counts, dtype=np.float64)
    totals = counts.sum(axis=-1, keepdims=True).clip(min=1.0)
    p = counts / totals
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log2(p), 0.0)
    return -terms.sum(axis=-1)


def scale_token_counts(
    flat_tokens: np.ndarray,
    scales: tuple[tuple[int, int], ...],
    codebook_size: int,
) -> np.ndarray:
    """Per-scale code histograms (num_scales, codebook_size) of (N, total) flat tokens."""
    offsets = scale_offsets(scales)
    counts = np.zeros((len(scales), codebook_size), dtype=np.int64)
    for s in range(len(scales)):
        values = np.asarray(flat_tokens[..., offsets[s] : offsets[s + 1]]).reshape(-1)
        counts[s] = np.bincount(values, minlength=codebook_size)
    return counts


class TokenEntropyAccumulator:
    """Accumulates emitted-token code counts per scale, total and windowed."""

    def __init__(self, num_scales: int, codebook_size: int, window_steps: int) -> None:
        self.counts = np.zeros((num_scales, codebook_size), dtype=np.int64)
        self.window_counts = np.zeros_like(self.counts)
        self.window_steps = int(window_steps)
        self.windowed_entropy: list[np.ndarray] = []
        self._steps_in_window = 0

    def update(self, pyramid) -> None:
        for s, grid in enumerate(pyramid):
            values = np.bincount(
                grid.reshape(-1).cpu().numpy(), minlength=self.counts.shape[1]
            )
            self.counts[s] += values
            self.window_counts[s] += values
        self._steps_in_window += 1
        if self._steps_in_window >= self.window_steps:
            self._flush_window()

    def _flush_window(self) -> None:
        if self._steps_in_window:
            self.windowed_entropy.append(entropy_bits(self.window_counts))
            self.window_counts[:] = 0
            self._steps_in_window = 0

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        """(per-scale horizon entropy, (windows, scales) windowed entropy)."""
        self._flush_window()
        windowed = (
            np.stack(self.windowed_entropy)
            if self.windowed_entropy
            else np.zeros((0, self.counts.shape[0]))
        )
        return entropy_bits(self.counts), windowed


# ---------------------------------------------------------------------------
# Shared plumbing: token files, forcing driver, rollout iterator, metadata
# ---------------------------------------------------------------------------


def load_token_file(path: str | Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """(tokens, transitions, timestamps) from a tokenized split; the latter two
    are None when the file lacks them."""
    with h5py.File(path, "r") as handle:
        tokens = np.asarray(handle["tokens"], dtype=np.int64)
        transitions = (
            np.asarray(handle["transitions"], dtype=np.int64)
            if "transitions" in handle
            else None
        )
        timestamps = (
            np.asarray(handle["timestamp"], dtype=np.int64)
            if "timestamp" in handle
            else None
        )
    return tokens, transitions, timestamps


def load_token_splits(
    root: str | Path, splits: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate token files; transitions are re-indexed into the union and a
    missing timestamp dataset becomes zeros."""
    root = Path(root)
    tokens, transitions, timestamps = [], [], []
    offset = 0
    for split in splits:
        t, tr, ts = load_token_file(root / split / "source.h5")
        tokens.append(t)
        transitions.append(
            tr + offset if tr is not None else np.zeros((0, 2), np.int64)
        )
        timestamps.append(ts if ts is not None else np.zeros(len(t), np.int64))
        offset += len(t)
    return np.concatenate(tokens), np.concatenate(transitions), np.concatenate(timestamps)


def make_conditioner(dynamics, dataset_root):
    """Forcing driver for a conditioned model, or None for an unconditioned one.

    A conditioned model must be fed its forcing vector every step or the FiLM
    bias silently goes to zero; the checkpoint's width says which forcing set it
    was trained on (4 = cyclic only, 5 = cyclic + CO2)."""
    if getattr(dynamics, "conditioning_dim", 0) <= 0:
        return None
    from .conditioning import ForcingConditioner

    conditioner = ForcingConditioner(
        dataset_root, include_co2=dynamics.conditioning_dim >= 5
    )
    if conditioner.dimension != dynamics.conditioning_dim:
        raise ValueError(
            f"model expects conditioning_dim {dynamics.conditioning_dim}, "
            f"conditioner supplies {conditioner.dimension}"
        )
    return conditioner


def conditioning_at(
    conditioner, start_timestamp: int, model_call: int, cadence_hours: int, device
):
    """Forcing vector for the frame produced by ``model_call`` (0-based): it sits
    ``model_call + 1`` native cadences after the start frame."""
    if conditioner is None:
        return None
    stamp = int(start_timestamp) + (model_call + 1) * cadence_hours * 3600
    return (
        torch.from_numpy(conditioner.vector(float(stamp)))
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
    )


def decode_frames(tokenizer, pyramid) -> np.ndarray:
    """(N, C, H, W) float32 decode of a token pyramid (final scale only)."""
    return tokenizer.decode_indices(pyramid, final_only=True)[0].float().cpu().numpy()


def initial_pyramid(flat_tokens: np.ndarray, scales, device) -> tuple[torch.Tensor, ...]:
    return split_flat_tokens(
        torch.from_numpy(np.asarray(flat_tokens, dtype=np.int64)).unsqueeze(0).to(device),
        scales,
    )


def iterate_rollout(
    dynamics,
    current: tuple[torch.Tensor, ...],
    model_calls: int,
    *,
    temperature: float | Sequence[float] = 1.0,
    top_k: int | None = None,
    conditioner=None,
    start_timestamp: int = 0,
    cadence_hours: int = 6,
    device: torch.device | None = None,
    precision: str = "fp32",
) -> Iterator[tuple[int, tuple[torch.Tensor, ...]]]:
    """Yield ``(model_call, pyramid)`` after each sampled step of a free rollout.

    One place for the loop every rollout diagnostic used to hand-roll: the
    forcing clock advances one native cadence per model call, sampling runs
    under autocast, and callers decide which calls to decode."""
    device = device or current[0].device
    for call in range(model_calls):
        conditioning = conditioning_at(
            conditioner, start_timestamp, call, cadence_hours, device
        )
        with autocast_context(device, precision):
            current = sample_next(
                dynamics, current, temperature=temperature, top_k=top_k,
                conditioning=conditioning,
            )
        yield call, current


def relative_vorticity(u: np.ndarray, v: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """Relative vorticity zeta = (1/cos-phi) dv/d-lambda - du/d-phi + tan-phi*u
    from wind stacks (T, H, W), up to the constant 1/a Earth-radius factor."""
    phi = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    cos_phi = np.cos(phi)[None, :, None]
    tan_phi = np.tan(phi)[None, :, None]
    dlam = 2.0 * np.pi / u.shape[-1]
    dv_dlam = (np.roll(v, -1, axis=-1) - np.roll(v, 1, axis=-1)) / (2.0 * dlam)
    du_dphi = np.gradient(u, phi, axis=1)
    return dv_dlam / np.clip(cos_phi, 1e-3, None) - du_dphi + tan_phi * u


class GridMetadata:
    """Per-channel de-normalisation and latitudes from a dataset/refs root.

    Missing files degrade gracefully: fields stay normalized and latitudes fall
    back to a uniform 87N..87S grid (what the videos always assumed)."""

    def __init__(self, dataset_root: str | Path | None, height: int) -> None:
        self.mean = self.std = None
        self.lat = np.linspace(87.0, -87.0, height)
        if dataset_root is None:
            return
        root = Path(dataset_root)
        try:
            self.mean = np.load(root / "normalize_mean.npz")
            self.std = np.load(root / "normalize_std.npz")
            self.lat = np.load(root / "lat.npy").astype(np.float64)
        except Exception:
            pass

    def physical(self, array: np.ndarray, name: str) -> np.ndarray:
        if self.mean is not None and name in self.mean.files:
            scale = float(np.asarray(self.std[name]).reshape(-1)[0])
            offset = float(np.asarray(self.mean[name]).reshape(-1)[0])
            return array * scale + offset
        return array


# ---------------------------------------------------------------------------
# The standard rollout report
# ---------------------------------------------------------------------------


@torch.inference_mode()
def climate_diagnostics(
    dynamics_checkpoint: str | Path,
    token_file: str | Path,
    dataset_root: str | Path,
    output_dir: str | Path,
    split: str = "val",
    start_index: int = 0,
    steps: int = 2000,
    temperature: float | tuple[float, ...] = 1.0,
    top_k: int | None = None,
    max_reference_frames: int | None = None,
    cadence_hours: int = 6,
    record_every: int = 1,
    precision: str = "auto",
    device_name: str = "auto",
    data_spectrum_path: str | Path | None = None,
    data_histograms_path: str | Path | None = None,
    tokenizer_checkpoint: str | Path | None = None,
    channel_names: list[str] | None = None,
    latitude_weights: np.ndarray | None = None,
    wind_channels: tuple[slice, slice] | None = None,
) -> dict:
    """Long sampled rollout scored against tokenizer reconstruction (dynamics
    loss) and truth (total loss): spectra, pixel EMD, token entropy, stability."""
    device = resolve_device(device_name)
    precision = resolve_precision(precision, device)
    dynamics, payload = load_dynamics_checkpoint(dynamics_checkpoint, device)
    tokenizer, _ = load_tokenizer_checkpoint(
        tokenizer_checkpoint or payload["tokenizer_checkpoint"], device
    )
    # The 33-channel Bridges source takes grid + channels from the dataset root
    # and STATE_CHANNELS; other datasets (ERA5 T30 on a legendre-gauss grid)
    # pass them explicitly.
    if latitude_weights is None:
        _, _, latitude_weights = load_frame_grid(dataset_root)
    if channel_names is None:
        channel_names = list(STATE_CHANNELS)
    spectrum_kwargs = {} if wind_channels is None else {
        "u_channels": wind_channels[0],
        "v_channels": wind_channels[1],
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_tokens, _, token_timestamps = load_token_file(token_file)
    if not 0 <= start_index < len(all_tokens):
        raise IndexError(f"start_index {start_index} out of range for {len(all_tokens)} frames")
    conditioner = make_conditioner(dynamics, dataset_root)
    if conditioner is not None and token_timestamps is None:
        raise ValueError("conditioned model needs frame timestamps; token file has none")
    start_timestamp = int(token_timestamps[start_index]) if conditioner else 0

    # --- sampled autoregressive rollout (climatology + stability) ---
    # record_every model steps per recorded frame: a model whose native lead is
    # shorter than the scoring cadence (a 1-hour model against 6-hourly refs)
    # is only comparable through its COMPOSED state, so intermediate frames are
    # never decoded or scored.
    rollout_spectra = SpectrumAccumulator(latitude_weights, **spectrum_kwargs)
    rollout_hist = PixelHistogram(len(channel_names))
    codebook_size = dynamics.scale_masks.shape[1]
    entropy_window = max(50, min(250, steps // 8))
    rollout_entropy = TokenEntropyAccumulator(
        len(dynamics.scales), codebook_size, entropy_window
    )
    step_std = []
    rollout_token_sequence: list[np.ndarray] = []
    completed = 0
    blew_up = False
    for call, current in iterate_rollout(
        dynamics, initial_pyramid(all_tokens[start_index], dynamics.scales, device),
        steps * record_every, temperature=temperature, top_k=top_k,
        conditioner=conditioner, start_timestamp=start_timestamp,
        cadence_hours=cadence_hours, device=device, precision=precision,
    ):
        if (call + 1) % record_every:
            continue
        with autocast_context(device, precision):
            frame = decode_frames(tokenizer, current)[0]  # (C, H, W)
        rollout_entropy.update(current)
        rollout_token_sequence.append(
            torch.cat([g.reshape(g.shape[0], -1) for g in current], dim=1)[0].cpu().numpy()
        )
        if not np.isfinite(frame).all():
            blew_up = True
            break
        rollout_spectra.update(frame)
        rollout_hist.update(frame)
        _, std = area_weighted_moments(frame, latitude_weights)
        step_std.append(std)
        completed += 1
    rollout_channel, rollout_ke = rollout_spectra.result()
    rollout_scale_entropy, windowed_scale_entropy = rollout_entropy.result()

    # entropy target: the same marginal on the full token file at the cadence
    # the rollout was RECORDED at (chain statistics move with cadence)
    reference_tokens = all_tokens[::record_every]
    data_scale_counts = scale_token_counts(reference_tokens, dynamics.scales, codebook_size)
    data_scale_entropy = entropy_bits(data_scale_counts)
    scale_labels = [f"{h}x{w}" for h, w in dynamics.scales]

    # --- reference 1: tokenizer reconstruction of the real frames ---
    reference = (
        reference_tokens
        if max_reference_frames is None
        else reference_tokens[:max_reference_frames]
    )
    recon_spectra = SpectrumAccumulator(latitude_weights, **spectrum_kwargs)
    recon_hist = PixelHistogram(len(channel_names))
    for i in range(0, len(reference), 16):
        pyramid = split_flat_tokens(
            torch.from_numpy(reference[i : i + 16]).to(device), dynamics.scales
        )
        with autocast_context(device, precision):
            decoded = decode_frames(tokenizer, pyramid)
        recon_spectra.update(decoded)
        recon_hist.update(decoded)
    recon_channel, recon_ke = recon_spectra.result()

    # --- reference 2: true source-data climatology (computed or precomputed) ---
    data_hist_counts = None
    if data_histograms_path is not None:
        with np.load(data_histograms_path) as bundle:
            if not np.allclose(bundle["edges"], HIST_EDGES):
                raise ValueError("precomputed truth histogram bins do not match HIST_EDGES")
            data_hist_counts = np.asarray(bundle["counts"], dtype=np.int64)
    if data_spectrum_path is not None:
        with np.load(data_spectrum_path) as bundle:
            data_channel = bundle["per_channel"]
            data_ke = bundle["kinetic_energy"]
    else:
        data_spectra = SpectrumAccumulator(latitude_weights, **spectrum_kwargs)
        data_hist = PixelHistogram(len(channel_names))
        loader = DataLoader(
            SourceClimateFrameDataset(dataset_root, split, cadence_hours * record_every),
            batch_size=16, shuffle=False, num_workers=2,
        )
        seen = 0
        for state, _ in loader:
            data_spectra.update(state.numpy())
            data_hist.update(state.numpy())
            seen += len(state)
            if max_reference_frames is not None and seen >= max_reference_frames:
                break
        data_channel, data_ke = data_spectra.result()
        if data_hist_counts is None:
            data_hist_counts = data_hist.counts

    # --- stability ---
    step_std = np.asarray(step_std)  # (T, C)
    scalar_temperature = isinstance(temperature, (int, float))
    temperature_label = (
        f"{temperature:g}" if scalar_temperature
        else ",".join(f"{t:g}" for t in temperature)
    )
    stability = {
        "requested_steps": int(steps),
        "completed_steps": int(completed),
        "record_every": int(record_every),
        "model_steps": int(completed * record_every),
        "native_cadence_hours": int(cadence_hours),
        "effective_cadence_hours": int(cadence_hours * record_every),
        "blew_up": bool(blew_up),
        "temperature": (
            float(temperature) if scalar_temperature else [float(t) for t in temperature]
        ),
    }
    if completed:
        early = step_std[: max(1, completed // 10)].mean(axis=0)
        late = step_std[-max(1, completed // 10) :].mean(axis=0)
        stability["mean_channel_std_early"] = float(early.mean())
        stability["mean_channel_std_late"] = float(late.mean())
        stability["late_over_early_std"] = float(late.mean() / max(early.mean(), 1e-9))
        stability["variance_collapse"] = bool(late.mean() < 0.3)
        stability["variance_explosion"] = bool(late.mean() > 5.0)

    def compare(pred_channel, pred_ke, ref_channel, ref_ke):
        return {
            "kinetic_energy": spectrum_metrics(pred_ke, ref_ke),
            "per_channel": {
                name: spectrum_metrics(pred_channel[c], ref_channel[c])
                for c, name in enumerate(channel_names)
            },
        }

    report = {
        "split": split,
        "checkpoint": str(Path(dynamics_checkpoint).resolve()),
        "checkpoint_epoch": payload.get("epoch"),
        "reference_frames": int(len(reference)),
        "stability": stability,
        "token_entropy": {
            "units": "bits",
            "window_steps": int(entropy_window),
            "scales": scale_labels,
            "data_per_scale": [float(x) for x in data_scale_entropy],
            "rollout_per_scale": [float(x) for x in rollout_scale_entropy],
            "delta_per_scale": [float(x) for x in rollout_scale_entropy - data_scale_entropy],
        },
        "rollout_vs_reconstruction": compare(rollout_channel, rollout_ke, recon_channel, recon_ke),
        "rollout_vs_data": compare(rollout_channel, rollout_ke, data_channel, data_ke),
        "tokenizer_loss_reconstruction_vs_data": compare(recon_channel, recon_ke, data_channel, data_ke),
        "spectra": {
            "kinetic_energy": {
                "rollout": [float(x) for x in rollout_ke],
                "reconstruction": [float(x) for x in recon_ke],
                "data": [float(x) for x in data_ke],
            }
        },
    }
    if data_hist_counts is not None:
        report["pixel_distributions"] = {
            name: {
                "inference_vs_truth_emd": histogram_emd(rollout_hist.counts[c], data_hist_counts[c]),
                "tokenizer_vs_truth_emd": histogram_emd(recon_hist.counts[c], data_hist_counts[c]),
            }
            for c, name in enumerate(channel_names)
        }
    np.savez_compressed(
        output_dir / "histograms.npz",
        edges=HIST_EDGES,
        inference=rollout_hist.counts,
        tokenizer=recon_hist.counts,
        **({"truth": data_hist_counts} if data_hist_counts is not None else {}),
    )
    # the emitted token sequence itself, for token-space analyses that need
    # order rather than counts (coarse-scale Markov statistics)
    if rollout_token_sequence:
        with h5py.File(output_dir / "rollout_tokens.h5", "w") as handle:
            handle.create_dataset(
                "tokens", data=np.stack(rollout_token_sequence).astype(np.uint16)
            )
            handle.attrs["scales"] = json.dumps([list(s) for s in dynamics.scales])
            handle.attrs["temperature"] = temperature_label
    np.savez_compressed(
        output_dir / "token_entropy.npz",
        scales=np.asarray(scale_labels),
        data_counts=data_scale_counts,
        rollout_counts=rollout_entropy.counts,
        windowed_rollout_entropy=windowed_scale_entropy,
        window_steps=entropy_window,
        temperature=np.atleast_1d(np.asarray(temperature, dtype=np.float64)),
    )
    atomic_json(output_dir / "diagnostics.json", report)
    plt = _pyplot()
    if plt is not None:
        _save_entropy_plot(
            plt, output_dir, scale_labels, data_scale_entropy, rollout_scale_entropy,
            windowed_scale_entropy, entropy_window, temperature_label,
        )
        _save_spectrum_plots(plt, output_dir, rollout_ke, recon_ke, data_ke, step_std)
        _save_histogram_plot(
            plt, output_dir, rollout_hist, recon_hist, data_hist_counts, channel_names
        )
    return report


def _pyplot():
    """Headless matplotlib, or None when plotting is unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    return plt


def _save_entropy_plot(
    plt, output_dir, scale_labels, data_scale_entropy, rollout_scale_entropy,
    windowed_scale_entropy, window_steps, temperature_label,
):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(scale_labels))
    axes[0].plot(x, data_scale_entropy, "o-", color="k", label="data tokens")
    axes[0].plot(x, rollout_scale_entropy, "o-", color="tab:red", label="rollout emissions")
    axes[0].set(
        xticks=x, xticklabels=scale_labels, xlabel="scale",
        ylabel="marginal token entropy (bits)",
        title=f"Per-scale token entropy, horizon-aggregated (T={temperature_label})",
    )
    axes[0].legend()
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(scale_labels)))
    steps_axis = (np.arange(len(windowed_scale_entropy)) + 1) * window_steps
    for s, label in enumerate(scale_labels):
        axes[1].plot(steps_axis, windowed_scale_entropy[:, s], color=colors[s], label=label)
        axes[1].axhline(data_scale_entropy[s], color=colors[s], ls="--", lw=0.8)
    axes[1].set(
        xlabel="rollout step",
        ylabel=f"entropy per {window_steps}-step window (bits)",
        title="Emitted-token entropy vs data reference (dashed)",
    )
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "token_entropy.png", dpi=150)
    plt.close(fig)


def _save_spectrum_plots(plt, output_dir, rollout_ke, recon_ke, data_ke, step_std):
    k = np.arange(len(data_ke))
    m = k > 0
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].loglog(k[m], data_ke[m], label="truth", color="k")
    axes[0].loglog(k[m], recon_ke[m], label="tokenizer", color="tab:blue")
    axes[0].loglog(k[m], rollout_ke[m], label="inference", color="tab:red")
    axes[0].set(xlabel="zonal wavenumber m", ylabel="KE power", title="Kinetic-energy zonal spectrum")
    axes[0].legend()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_data = rollout_ke / data_ke
        ratio_recon = rollout_ke / recon_ke
    axes[1].semilogx(k[m], ratio_data[m], label="inference / truth", color="tab:red")
    axes[1].semilogx(k[m], ratio_recon[m], label="inference / tokenizer", color="tab:blue")
    axes[1].axhline(1.0, color="k", ls="--", lw=1)
    axes[1].set(xlabel="zonal wavenumber m", ylabel="prediction / truth", title="High-frequency retention", ylim=(0, 1.5))
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "ke_spectrum.png", dpi=150)
    plt.close(fig)
    if len(step_std):
        fig2, ax = plt.subplots(figsize=(8, 5))
        ax.plot(step_std.mean(axis=1))
        ax.set(xlabel="rollout step", ylabel="mean area-weighted std (all channels)", title="Rollout stability")
        fig2.tight_layout()
        fig2.savefig(output_dir / "stability.png", dpi=150)
        plt.close(fig2)


def _save_histogram_plot(plt, output_dir, rollout_hist, recon_hist, data_hist_counts, channel_names):
    centers = 0.5 * (HIST_EDGES[:-1] + HIST_EDGES[1:])
    widths = np.diff(HIST_EDGES)
    inference = rollout_hist.density()
    tokenizer = recon_hist.density()
    truth = None
    if data_hist_counts is not None:
        totals = data_hist_counts.sum(axis=1, keepdims=True).clip(min=1)
        truth = data_hist_counts / totals / widths
    # 33-channel source: 8 wide (one variable family per row); other channel
    # counts adapt: up to 8 columns, each cell ~3.2x2.9 in.
    n = len(channel_names)
    cols = min(8, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.9 * rows), sharex=True, squeeze=False)
    for c, name in enumerate(channel_names):
        ax = axes.flat[c]
        if truth is not None:
            ax.semilogy(centers, truth[c], color="k", lw=1.2, label="truth")
        ax.semilogy(centers, tokenizer[c], color="tab:blue", lw=1.0, label="tokenizer")
        ax.semilogy(centers, inference[c], color="tab:red", lw=1.0, label="inference")
        ax.set_title(name, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_ylim(bottom=1e-6)
    for ax in axes.flat[len(channel_names) :]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=12)
    fig.suptitle("Pixel-value distributions (normalized units)", fontsize=14)
    fig.supxlabel("value (sigma)")
    fig.supylabel("density")
    fig.tight_layout(rect=(0.01, 0.01, 1, 0.98))
    fig.savefig(output_dir / "pixel_histograms.png", dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Rollout mechanism, in token space
# ---------------------------------------------------------------------------
# Three questions the spectra/EMD dashboards can only hint at: exposure bias
# (teacher-forced vs rollout predictive entropy), off-support structure (unseen
# parent->child co-occurrences) and persistence (data stay rate vs predicted and
# realised P(stay)). Token-only except the checkpoint load.


def parent_index(parent_shape, child_shape, device) -> torch.Tensor:
    """For each child cell (flattened), the flattened parent cell covering it."""
    ph, pw = parent_shape
    ch, cw = child_shape
    rows = (torch.arange(ch, device=device) * ph) // ch
    cols = (torch.arange(cw, device=device) * pw) // cw
    return (rows[:, None] * pw + cols[None, :]).reshape(-1)


class CooccurrenceTable:
    """Seen (parent_token, child_token) pairs per scale transition, from train."""

    def __init__(self, train_tokens: np.ndarray, scales, codebook: int, device):
        self.scales = scales
        self.codebook = codebook
        self.offsets = scale_offsets(scales)
        self.parents = [
            parent_index(scales[k], scales[k + 1], device) for k in range(len(scales) - 1)
        ]
        tokens = torch.from_numpy(train_tokens).to(device)
        self.seen = []
        for k in range(len(scales) - 1):
            keys = self.pair_keys(tokens, k)
            table = torch.zeros(codebook * codebook, dtype=torch.bool, device=device)
            table[keys.unique()] = True
            self.seen.append(table)

    def pair_keys(self, flat_tokens: torch.Tensor, k: int) -> torch.Tensor:
        parent = flat_tokens[:, self.offsets[k] : self.offsets[k + 1]]
        child = flat_tokens[:, self.offsets[k + 1] : self.offsets[k + 2]]
        return (parent[:, self.parents[k]] * self.codebook + child).reshape(-1)

    def unseen_rate(self, flat_tokens: torch.Tensor) -> list[float]:
        return [
            float((~self.seen[k][self.pair_keys(flat_tokens, k)]).float().mean())
            for k in range(len(self.scales) - 1)
        ]


def scale_stats(logits: torch.Tensor, current_scale: torch.Tensor) -> tuple[float, float]:
    """Per-position entropy (bits) and P(stay) from one scale's logits."""
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs.clamp(min=-1e4)).sum(-1) / LOG2
    stay = probs.gather(-1, current_scale.reshape(*probs.shape[:-1], 1).long())
    return entropy.mean().item(), stay.mean().item()


def teacher_forced_stats(
    model, tokens, transitions, timestamps, conditioner, device, precision, batch_size, label
) -> dict:
    """Predictive entropy, P(stay), data stay rate and top-1 per scale on
    truth pairs."""
    scales = model.scales
    n_scales = len(scales)
    entropy = np.zeros(n_scales)
    p_stay = np.zeros(n_scales)
    stay_data = np.zeros(n_scales)
    top1 = np.zeros(n_scales)
    count = 0
    for start in range(0, len(transitions), batch_size):
        pairs = transitions[start : start + batch_size]
        current = split_flat_tokens(torch.from_numpy(tokens[pairs[:, 0]]).to(device), scales)
        target = split_flat_tokens(torch.from_numpy(tokens[pairs[:, 1]]).to(device), scales)
        conditioning = None
        if conditioner is not None:
            conditioning = torch.from_numpy(
                np.stack([conditioner.vector(float(timestamps[j])) for j in pairs[:, 1]])
            ).to(device=device, dtype=torch.float32)
        with torch.no_grad(), autocast_context(device, precision):
            logits = model(current, target, conditioning)
        for k in range(n_scales):
            h, s = scale_stats(logits[k], current[k])
            entropy[k] += h * len(pairs)
            p_stay[k] += s * len(pairs)
            stay_data[k] += (target[k] == current[k]).float().mean().item() * len(pairs)
            top1[k] += (
                logits[k].argmax(-1).reshape(target[k].shape) == target[k]
            ).float().mean().item() * len(pairs)
        count += len(pairs)
    return {
        "label": label,
        "pairs": int(count),
        "predictive_entropy_bits": (entropy / count).tolist(),
        "predicted_p_stay": (p_stay / count).tolist(),
        "data_stay_rate": (stay_data / count).tolist(),
        "top1_accuracy": (top1 / count).tolist(),
    }


def mechanism_rollout(
    model, tokens, timestamps, start, steps, temperature, conditioner, device,
    precision, cadence_hours, table: CooccurrenceTable,
) -> dict:
    """Free rollout that records per-scale predictive entropy, predicted and
    realised P(stay), and the unseen parent->child rate at every step."""
    scales = model.scales
    n_scales = len(scales)
    current = initial_pyramid(tokens[start], scales, device)
    entropy = np.zeros((steps, n_scales))
    p_stay = np.zeros((steps, n_scales))
    realized_stay = np.zeros((steps, n_scales))
    unseen = np.zeros((steps, n_scales - 1))
    for step in range(steps):
        conditioning = conditioning_at(
            conditioner, int(timestamps[start]), step, cadence_hours, device
        )
        known = []
        with torch.no_grad(), autocast_context(device, precision):
            for k, scale in enumerate(scales):
                logits = model.predict_next_scale(current, tuple(known), conditioning)
                entropy[step, k], p_stay[step, k] = scale_stats(logits, current[k])
                probabilities = (logits.float() / temperature).softmax(-1)
                flat = torch.multinomial(probabilities.reshape(-1, probabilities.shape[-1]), 1)
                grid = flat.reshape(1, scale[0], scale[1])
                realized_stay[step, k] = (grid == current[k]).float().mean().item()
                known.append(grid)
        current = tuple(known)
        unseen[step] = table.unseen_rate(torch.cat([g.reshape(1, -1) for g in current], dim=1))
    tenth = max(1, steps // 10)
    return {
        "steps": int(steps),
        "temperature": float(temperature),
        "predictive_entropy_bits": entropy.mean(0).tolist(),
        "predictive_entropy_first_tenth": entropy[:tenth].mean(0).tolist(),
        "predictive_entropy_last_tenth": entropy[-tenth:].mean(0).tolist(),
        "predicted_p_stay": p_stay.mean(0).tolist(),
        "realized_stay_rate": realized_stay.mean(0).tolist(),
        "unseen_parent_child_rate": unseen.mean(0).tolist(),
    }


def rollout_mechanism(
    dynamics_checkpoint: str | Path,
    token_root: str | Path,
    held_out: Sequence[str] = ("val", "test"),
    train_pairs: int = 2000,
    steps: int = 2000,
    temperature: float = 1.0,
    batch_size: int = 32,
    cadence_hours: int = 6,
    dataset_root: str | Path | None = None,
    device_name: str = "auto",
    precision: str = "auto",
) -> dict:
    device = resolve_device(device_name)
    precision = resolve_precision(precision, device)
    model, payload = load_dynamics_checkpoint(dynamics_checkpoint, device)
    root = Path(token_root)
    scales = model.scales
    conditioner = make_conditioner(model, dataset_root)
    train_tokens, train_transitions, train_ts = load_token_file(root / "train" / "source.h5")
    if train_ts is None:
        train_ts = np.zeros(len(train_tokens), np.int64)
    held_tokens, held_transitions, held_ts = load_token_splits(root, held_out)
    table = CooccurrenceTable(train_tokens, scales, int(model.codebook_size), device)
    return {
        "checkpoint": str(Path(dynamics_checkpoint).resolve()),
        "checkpoint_epoch": payload.get("epoch"),
        "scales": [f"{h}x{w}" for h, w in scales],
        "held_out_splits": ",".join(held_out),
        "teacher_forced_held_out": teacher_forced_stats(
            model, held_tokens, held_transitions, held_ts, conditioner, device,
            precision, batch_size, "held_out"),
        "teacher_forced_train_tail": teacher_forced_stats(
            model, train_tokens, train_transitions[-train_pairs:], train_ts,
            conditioner, device, precision, batch_size, "train_tail"),
        "unseen_parent_child_rate_held_out_truth": table.unseen_rate(
            torch.from_numpy(held_tokens).to(device)),
        "rollout": mechanism_rollout(
            model, held_tokens, held_ts, 0, steps, temperature, conditioner,
            device, precision, cadence_hours, table),
    }


def format_mechanism(result: dict) -> str:
    tf = result["teacher_forced_held_out"]
    ro = result["rollout"]
    lines = [
        f"== {result['checkpoint']} (epoch {result.get('checkpoint_epoch')}) ==",
        f"{'scale':>7} {'H_tf':>6} {'H_roll':>7} {'dH':>6} | {'stay_data':>9} {'Pstay_tf':>8} "
        f"{'Pstay_ro':>8} {'stay_ro':>7} | {'unseen_truth':>12} {'unseen_roll':>11}",
    ]
    for k, label in enumerate(result["scales"]):
        unseen_t = result["unseen_parent_child_rate_held_out_truth"][k - 1] if k > 0 else float("nan")
        unseen_r = ro["unseen_parent_child_rate"][k - 1] if k > 0 else float("nan")
        lines.append(
            f"{label:>7} {tf['predictive_entropy_bits'][k]:6.2f} {ro['predictive_entropy_bits'][k]:7.2f} "
            f"{ro['predictive_entropy_bits'][k] - tf['predictive_entropy_bits'][k]:6.2f} | "
            f"{tf['data_stay_rate'][k]:9.3f} {tf['predicted_p_stay'][k]:8.3f} {ro['predicted_p_stay'][k]:8.3f} "
            f"{ro['realized_stay_rate'][k]:7.3f} | {unseen_t:12.4f} {unseen_r:11.4f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scale oracle: which token scale owns the low-wavenumber variance deficit?
# ---------------------------------------------------------------------------
# Closed-loop rollouts over the held-out sequence with the coarsest k scales of
# each NEW frame forced to truth: oracle_0 = free rollout, oracle_k = truth for
# scales < k, recon = truth everywhere (tokenizer floor), onestep = every step
# restarts from the TRUE previous frame (no compounding).


def sample_scales(model, current, forced_known, temperature, conditioning):
    """sample_next with the first len(forced_known) scales fixed."""
    known = list(forced_known)
    with torch.no_grad():
        for scale in model.scales[len(known):]:
            logits = model.predict_next_scale(current, tuple(known), conditioning)
            probabilities = (logits.float() / temperature).softmax(-1)
            flat = torch.multinomial(probabilities.reshape(-1, probabilities.shape[-1]), 1)
            known.append(flat.reshape(1, scale[0], scale[1]))
    return tuple(known)


class _OracleAccumulator:
    def __init__(self, weights, u_slice, v_slice, n_scales, codebook):
        self.spectra = SpectrumAccumulator(weights, u_slice, v_slice)
        self.weights = weights
        self.std = []
        self.mean = []
        self.counts = np.zeros((n_scales, codebook), dtype=np.int64)
        self.codebook = codebook

    def update(self, frame: np.ndarray, flat_tokens: np.ndarray, scales):
        self.spectra.update(frame)
        mean, std = area_weighted_moments(frame, self.weights)
        self.std.append(std)
        self.mean.append(mean)
        self.counts += scale_token_counts(flat_tokens[None], scales, self.codebook)

    def summary(self):
        per_channel, ke = self.spectra.result()
        return {
            "per_channel": per_channel,
            "ke": ke,
            "channel_std": np.asarray(self.std).mean(0),
            "channel_mean_std_over_time": np.asarray(self.mean).std(0),
            "entropy_bits": entropy_bits(self.counts),
        }


def scale_oracle(
    dynamics_checkpoint: str | Path,
    token_root: str | Path,
    splits: Sequence[str] = ("val", "test"),
    tokenizer_checkpoint: str | Path | None = None,
    max_steps: int | None = None,
    temperature: float = 1.0,
    channels: Sequence[str] = ("temperature", "humidity", "u_wind", "v_wind", "surface_pressure"),
    u_channel: str = "u_wind",
    v_channel: str = "v_wind",
    low_m: int = 5,
    cadence_hours: int = 6,
    dataset_root: str | Path | None = None,
    device_name: str = "auto",
    precision: str = "auto",
    progress=None,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Returns (report, {mode: ke_spectrum})."""
    from .era5_npz import legendre_gauss_area_weights

    device = resolve_device(device_name)
    precision = resolve_precision(precision, device)
    model, payload = load_dynamics_checkpoint(dynamics_checkpoint, device)
    tokenizer, _ = load_tokenizer_checkpoint(
        tokenizer_checkpoint or payload["tokenizer_checkpoint"], device
    )
    scales = model.scales
    n_scales = len(scales)
    codebook = int(model.codebook_size)
    channels = list(channels)
    ui, vi = channels.index(u_channel), channels.index(v_channel)
    weights = legendre_gauss_area_weights(tokenizer.grid_shape[0])
    conditioner = make_conditioner(model, dataset_root)
    tokens, _, timestamps = load_token_splits(token_root, splits)
    steps = len(tokens) - 1 if max_steps is None else min(max_steps, len(tokens) - 1)
    truth = [initial_pyramid(tokens[t], scales, device) for t in range(steps + 1)]
    modes = ["recon", "onestep"] + [f"oracle_{k}" for k in range(n_scales)]
    results = {}
    for mode in modes:
        acc = _OracleAccumulator(weights, slice(ui, ui + 1), slice(vi, vi + 1), n_scales, codebook)
        current = truth[0]
        for step in range(steps):
            target_truth = truth[step + 1]
            if mode == "recon":
                new = target_truth
            else:
                if mode == "onestep":
                    current, forced = truth[step], ()
                else:
                    forced = target_truth[: int(mode.split("_")[1])]
                conditioning = conditioning_at(
                    conditioner, int(timestamps[0]), step, cadence_hours, device
                )
                with autocast_context(device, precision):
                    new = sample_scales(model, current, forced, temperature, conditioning)
            with torch.no_grad(), autocast_context(device, precision):
                frame = decode_frames(tokenizer, new)[0]
            flat = torch.cat([g.reshape(-1) for g in new]).cpu().numpy()
            acc.update(frame, flat, scales)
            current = new
        results[mode] = acc.summary()
        if progress is not None:
            progress(f"  {mode:>9} done ({steps} steps)")

    ref = results["recon"]
    low = slice(1, low_m + 1)
    report = {
        "checkpoint": str(Path(dynamics_checkpoint).resolve()),
        "checkpoint_epoch": payload.get("epoch"),
        "scales": [f"{h}x{w}" for h, w in scales],
        "steps": int(steps),
        "splits": ",".join(splits),
        "temperature": temperature,
        "low_m_wavenumbers": f"1..{low_m}",
        "runs": {},
    }
    for mode in modes:
        r = results[mode]
        metrics = spectrum_metrics(r["ke"], ref["ke"])
        ratio = np.asarray(metrics["ratio_by_wavenumber"])
        std_ratio = r["channel_std"] / ref["channel_std"]
        report["runs"][mode] = {
            "low_m_ke_ratio": float(np.nanmean(ratio[low])),
            "ke_relative_l2": metrics["relative_l2"],
            "high_wavenumber_retention": metrics["high_wavenumber_retention"],
            "ke_ratio_by_wavenumber": ratio.tolist(),
            "channel_std_ratio": float(np.mean(std_ratio)),
            "channel_std_ratio_per_channel": dict(zip(channels, std_ratio.tolist())),
            "entropy_delta_bits": (r["entropy_bits"] - ref["entropy_bits"]).tolist(),
            "temporal_std_of_mean_ratio": dict(zip(channels, (
                r["channel_mean_std_over_time"]
                / np.maximum(ref["channel_mean_std_over_time"], 1e-9)
            ).tolist())),
        }
    return report, {mode: results[mode]["ke"] for mode in modes}


def format_oracle(report: dict) -> str:
    lines = [
        f"== {report['checkpoint']} (epoch {report.get('checkpoint_epoch')}), "
        f"{report['steps']} steps, vs recon of truth ==",
        f"{'run':>9} {'lowm_KE':>8} {'relL2':>6} {'HFret':>6} {'std/ref':>7} | entropy delta per scale (bits)",
    ]
    for mode, r in report["runs"].items():
        lines.append(
            f"{mode:>9} {r['low_m_ke_ratio']:8.3f} {r['ke_relative_l2']:6.3f} "
            f"{r['high_wavenumber_retention']:6.3f} {r['channel_std_ratio']:7.3f} | "
            + " ".join(f"{d:+.2f}" for d in r["entropy_delta_bits"])
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Coarse-token Markov chain and code reuse (CPU, token-only)
# ---------------------------------------------------------------------------


def coarse_sequence(flat_tokens: np.ndarray, scales, scale_index: int) -> np.ndarray:
    """(N, tokens_at_scale) slice of the token pyramid for one scale."""
    offsets = scale_offsets(scales)
    return np.asarray(
        flat_tokens[..., offsets[scale_index] : offsets[scale_index + 1]], dtype=np.int64
    )


def chain_stats(sequence: np.ndarray, codebook_size: int) -> dict:
    """Markov statistics of a per-position token sequence over time: states
    visited, perplexity, change rate, dwell, decorrelation lag."""
    steps, positions = sequence.shape
    stats: dict[str, object] = {}
    counts = np.zeros(codebook_size, dtype=np.int64)
    for p in range(positions):
        counts += np.bincount(sequence[:, p], minlength=codebook_size)
    probability = counts / counts.sum()
    nonzero = probability[probability > 0]
    stats["states_visited"] = int((counts > 0).sum())
    stats["marginal_entropy_bits"] = float(entropy_bits(counts))
    stats["perplexity"] = float(np.exp(-(nonzero * np.log(nonzero)).sum()))
    stats["top1_share"] = float(probability.max())
    stats["top10_share"] = float(np.sort(probability)[-10:].sum())
    changed = sequence[1:] != sequence[:-1]
    stats["change_rate"] = float(changed.mean())
    dwell = []
    for p in range(positions):
        column = sequence[:, p]
        breaks = np.flatnonzero(np.diff(column)) + 1
        runs = np.diff(np.concatenate(([0], breaks, [len(column)])))
        dwell.append(runs.mean())
    stats["mean_dwell_steps"] = float(np.mean(dwell))
    stats["self_transition"] = float(1.0 - stats["change_rate"])
    # token-identity autocorrelation crossing 1/e above chance
    chance = float((probability**2).sum())
    agreement = {}
    for lag in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        if lag >= steps:
            break
        agreement[lag] = float((sequence[lag:] == sequence[:-lag]).mean())
    stats["chance_agreement"] = chance
    stats["agreement_by_lag"] = agreement
    decorrelation = None
    for lag, same in agreement.items():
        if (same - chance) / max(1e-9, 1.0 - chance) < 1 / np.e:
            decorrelation = lag
            break
    stats["decorrelation_lag_steps"] = decorrelation
    return stats


def code_reuse_table(
    train: np.ndarray, held: np.ndarray, scales, codebook_size: int
) -> list[dict]:
    """Per scale: distinct train codes, tokens per code, held-out fraction on
    rare/unseen codes, per-position unseen rate, unique train tuples."""
    offsets = scale_offsets(scales)
    rows = []
    for k, (h, w) in enumerate(scales):
        a = train[:, offsets[k] : offsets[k + 1]]
        b = held[:, offsets[k] : offsets[k + 1]]
        counts = np.bincount(a.reshape(-1), minlength=codebook_size)
        distinct = int((counts > 0).sum())
        held_counts = counts[b.reshape(-1)]
        _, tuple_counts = np.unique(a, axis=0, return_counts=True)
        rows.append({
            "scale": f"{h}x{w}",
            "positions": h * w,
            "distinct_train_codes": distinct,
            "tokens_per_code": a.size / max(distinct, 1),
            "held_out_unseen": float((held_counts == 0).mean()),
            "held_out_code_lt5": float((held_counts < 5).mean()),
            "held_out_code_lt20": float((held_counts < 20).mean()),
            "held_out_unseen_per_position": float(np.mean(
                [np.isin(b[:, p], a[:, p], invert=True).mean() for p in range(a.shape[1])]
            )),
            "train_unique_tuple_fraction": float((tuple_counts == 1).sum() / len(a)),
        })
    return rows


# ---------------------------------------------------------------------------
# Marginal-entropy temperature calibration (gust2-style)
# ---------------------------------------------------------------------------
# Solves H_roll_k(T_k) = H_data_k per scale from token-only rollouts at a grid
# of temperatures. On climate data this trades marginal spread for temporal
# persistence and T=1 has won every rollout comparison; kept for the paper
# comparison with gust2.


def solve_temperature(
    temperatures: np.ndarray, entropies: np.ndarray, target: float
) -> tuple[float, bool]:
    """Invert the monotone entropy response; returns (T, clamped)."""
    order = np.argsort(entropies)
    h, t = entropies[order], temperatures[order]
    if target <= h[0]:
        return float(t[0]), True
    if target >= h[-1]:
        return float(t[-1]), True
    return float(np.interp(target, h, t)), False


def calibrate_temperature(
    dynamics_checkpoint: str | Path,
    token_file: str | Path,
    temperatures: Sequence[float] = (0.7, 1.0, 1.3, 1.6, 2.0, 2.5),
    steps: int = 1000,
    start: int = 0,
    window_steps: int = 0,
    cadence_hours: int = 6,
    dataset_root: str | Path | None = None,
    device_name: str = "auto",
    precision: str = "auto",
    progress=None,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Returns (result, data_entropy per scale, responses (T, scales)).

    window_steps > 0 matches mean WINDOWED entropies: for seasonal systems the
    horizon marginal folds the annual cycle into the target, which an
    unconditioned model can only meet by over-dispersing within windows."""
    device = resolve_device(device_name)
    precision = resolve_precision(precision, device)
    temperatures = np.asarray([float(t) for t in temperatures])
    dynamics, _ = load_dynamics_checkpoint(dynamics_checkpoint, device)
    codebook_size = dynamics.scale_masks.shape[1]
    scale_labels = [f"{h}x{w}" for h, w in dynamics.scales]
    all_tokens, _, token_timestamps = load_token_file(token_file)
    conditioner = make_conditioner(dynamics, dataset_root)
    if conditioner is not None and token_timestamps is None:
        raise ValueError("conditioned model needs timestamps in the token file")
    start_timestamp = int(token_timestamps[start]) if conditioner else 0
    if window_steps > 0:
        data_entropy = np.stack([
            entropy_bits(scale_token_counts(
                all_tokens[i : i + window_steps], dynamics.scales, codebook_size))
            for i in range(0, len(all_tokens) - window_steps + 1, window_steps)
        ]).mean(axis=0)
    else:
        data_entropy = entropy_bits(scale_token_counts(all_tokens, dynamics.scales, codebook_size))
    initial = initial_pyramid(all_tokens[start], dynamics.scales, device)
    responses = []
    with torch.inference_mode():
        for temperature in temperatures:
            accumulator = TokenEntropyAccumulator(
                len(dynamics.scales), codebook_size, window_steps=window_steps or steps
            )
            for _, current in iterate_rollout(
                dynamics, initial, steps, temperature=float(temperature),
                conditioner=conditioner, start_timestamp=start_timestamp,
                cadence_hours=cadence_hours, device=device, precision=precision,
            ):
                accumulator.update(current)
            aggregated, windowed = accumulator.result()
            rollout_entropy = windowed.mean(axis=0) if window_steps > 0 else aggregated
            responses.append(rollout_entropy)
            if progress is not None:
                progress(f"T={temperature:g}: {dict(zip(scale_labels, rollout_entropy.round(3)))}")
    responses = np.stack(responses)
    solved = [
        solve_temperature(temperatures, responses[:, s], data_entropy[s])
        for s in range(len(scale_labels))
    ]
    per_scale = [t for t, _ in solved]
    # global read: the finest two scales carry the high-frequency content that
    # cold sampling drains first, and dominate the token budget
    result = {
        "dynamics_checkpoint": str(Path(dynamics_checkpoint).resolve()),
        "token_file": str(Path(token_file).resolve()),
        "steps": steps,
        "window_steps": window_steps,
        "units": "bits",
        "scales": scale_labels,
        "reference_temperatures": [float(t) for t in temperatures],
        "data_entropy_per_scale": [float(x) for x in data_entropy],
        "rollout_entropy_per_scale": {
            f"{t:g}": [float(x) for x in row] for t, row in zip(temperatures, responses)
        },
        "solved_temperature_per_scale": [float(t) for t in per_scale],
        "solved_clamped_per_scale": [c for _, c in solved],
        "global_temperature_finest_two": float(np.mean(per_scale[-2:])),
    }
    return result, data_entropy, responses


def save_calibration_plot(out_dir: Path, result: dict, data_entropy, responses) -> None:
    plt = _pyplot()
    if plt is None:
        return
    temperatures = np.asarray(result["reference_temperatures"])
    labels = result["scales"]
    per_scale = result["solved_temperature_per_scale"]
    global_t = result["global_temperature_finest_two"]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(labels)))
    for s, label in enumerate(labels):
        ax.plot(temperatures, responses[:, s], "o-", color=colors[s], label=label)
        ax.axhline(data_entropy[s], color=colors[s], ls="--", lw=0.8)
        ax.axvline(per_scale[s], color=colors[s], ls=":", lw=0.8)
    ax.axvline(global_t, color="k", lw=1.2, label=f"global T={global_t:.2f}")
    ax.set(
        xlabel="sampling temperature",
        ylabel=f"emitted-token entropy over {result['steps']} steps (bits)",
        title="Marginal-entropy calibration: response vs data target (dashed)",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "calibration.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# ERA5 truth references and checkpoint autopsy
# ---------------------------------------------------------------------------


def build_era5_references(
    config: dict,
    output: str | Path,
    split: str = "val",
    panels_stride: int = 4,
    panels_fields: Sequence[str] = ("temperature", "humidity", "u_wind", "v_wind"),
    batch: int = 64,
) -> dict:
    """Truth references for the ERA5 T30 set, computed with the SAME
    accumulators the rollout report uses: truth_spectrum.npz, truth_hist.npz,
    truth_panels.npz (compare video; stride MUST match the render stride),
    normalize_{mean,std}.npz and lat.npy (gauss latitudes, north first)."""
    from .era5_npz import load_era5_npz

    # fp32 for the truth reference: the climatology should be exact, not the
    # bf16 the tokenizer happens to ingest
    data_config = dict(config["data"])
    data_config["dtype"] = "fp32"
    data = load_era5_npz(data_config, torch.device("cpu"), precision="fp32")
    channels = list(data.channels)
    if split not in data.frames:
        raise ValueError(f"split {split!r} not in {list(data.frames)}")
    frames = data.frames[split].cpu().numpy().astype(np.float32)
    weights = data.latitude_weights
    ui, vi = channels.index("u_wind"), channels.index("v_wind")
    accum = SpectrumAccumulator(weights, slice(ui, ui + 1), slice(vi, vi + 1))
    hist = PixelHistogram(len(channels))
    for i in range(0, len(frames), batch):
        chunk = frames[i : i + batch]
        accum.update(chunk)
        hist.update(chunk)
    per_channel, kinetic_energy = accum.result()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "truth_spectrum.npz", per_channel=per_channel, kinetic_energy=kinetic_energy)
    np.savez(out / "truth_hist.npz", edges=HIST_EDGES, counts=hist.counts)
    collect = sorted({channels.index(n) for n in panels_fields})
    raw = frames[::panels_stride][:, collect]
    np.savez(out / "truth_panels.npz", raw=raw.astype(np.float32),
             collect=np.asarray(collect, dtype=np.int64))
    np.savez(out / "normalize_mean.npz", **{n: np.float32(data.mean[i]) for i, n in enumerate(channels)})
    np.savez(out / "normalize_std.npz", **{n: np.float32(data.std[i]) for i, n in enumerate(channels)})
    nodes, _ = np.polynomial.legendre.leggauss(frames.shape[-2])
    lat = np.sort(np.degrees(np.arcsin(np.asarray(nodes))))[::-1]
    np.save(out / "lat.npy", lat.astype(np.float64))
    return {"split": split, "frames": int(len(frames)), "channels": channels,
            "wind": (ui, vi), "panels_raw_shape": list(raw.shape), "collect": collect,
            "stride": panels_stride, "output": str(out)}


def _tensor_quantiles(t: torch.Tensor) -> dict[str, float]:
    t = t.float()
    q = torch.quantile(t, torch.tensor([0.0, 0.05, 0.5, 0.95, 1.0]))
    return {"min": q[0].item(), "p05": q[1].item(), "median": q[2].item(),
            "p95": q[3].item(), "max": q[4].item(), "mean": t.mean().item()}


def codebook_autopsy(path: str | Path, dead_threshold: float = 0.5) -> dict:
    """Static forensics on a tokenizer checkpoint: codebook health (perplexity,
    dead fraction, top-k mass, row norms), parameter-group norms, Adam
    second-moment extremes. No data or GPU needed."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ckpt["model"]
    codebook = model["quantizer.codebook"].float()
    ema_count = model["quantizer.ema_count"].float()
    probs = ema_count / ema_count.sum()
    entropy = -(probs * (probs + 1e-12).log()).sum().item()
    top = probs.sort(descending=True).values
    norms = codebook.norm(dim=1)
    median_norm = norms.median().item()
    groups: dict[str, dict[str, float]] = {}
    for prefix in ("encoder.", "decoder.", "quantizer.phi."):
        sq, mx = 0.0, 0.0
        for key, value in model.items():
            if key.startswith(prefix) and value.is_floating_point():
                sq += value.float().square().sum().item()
                mx = max(mx, value.float().abs().max().item())
        groups[prefix.rstrip(".")] = {"global_norm": math.sqrt(sq), "abs_max": mx}
    adam_v_max = 0.0
    for state in ckpt.get("optimizer", {}).get("state", {}).values():
        v = state.get("exp_avg_sq")
        if v is not None:
            adam_v_max = max(adam_v_max, v.float().max().item())
    phi_gain = {}
    for key, value in model.items():
        if key.startswith("quantizer.phi.") and key.endswith("weight"):
            phi_gain[key.split(".")[2]] = float(
                torch.linalg.svdvals(value.float().sum(dim=(2, 3)))[0]
            )
    return {
        "checkpoint": str(path),
        "epoch": ckpt.get("epoch"),
        "codebook": {
            "n_codes": int(codebook.shape[0]),
            "perplexity": math.exp(entropy),
            "dead_fraction": (ema_count < dead_threshold).float().mean().item(),
            "top16_mass": top[:16].sum().item(),
            "top256_mass": top[:256].sum().item(),
            "ema_count": _tensor_quantiles(ema_count),
            "row_norms": _tensor_quantiles(norms),
            "near_zero_rows": (norms < 0.05 * median_norm).float().mean().item(),
        },
        "weight_groups": groups,
        "phi_dc_gain_by_scale": phi_gain,
        "amp_scale": ckpt.get("scaler", {}).get("scale"),
        "adam_exp_avg_sq_max": adam_v_max,
    }
