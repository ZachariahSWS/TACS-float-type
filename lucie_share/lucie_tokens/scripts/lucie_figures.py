#!/usr/bin/env python3
"""LUCIE-paper (Guan et al. 2024, arXiv:2405.16297) Figures 4, 5 and 10 for
ERA5 T30 and token-model ensembles.

  Fig. 4  mean diurnal range of T0.95 (max - min within each 00Z day)
  Fig. 5  mean annual range of T0.95 (max - min within each 365-day block)
  Fig. 10 zonal power spectra of T0.95 and V0.34, full state and 6-h tendency,
          full globe and extratropics (LUCIE_train.py rows 7:15 and 32:40)

  lucie_figures.py truth   --npz era5_T30_regridded.npz --out era5_stats.npz
  lucie_figures.py rollout --dynamics ep50.pt --tokenizer best.pt
                           --token-file val/source.h5 --refs era5_refs/y1
                           --members 10 --years 10 --out model_stats.npz
  lucie_figures.py plot    --truth era5_stats.npz --model LABEL=model_stats.npz
                           --out-dir figs
  lucie_figures.py calibrate --dynamics ep50.pt --train-tokens train/source.h5
                           --token-file val/source.h5 --out calib.json

``rollout --score`` also scores the ensemble the way the ladder diagnostics do
(pixel EMD vs the refs' truth_hist.npz in sigma units, KE-spectrum relative L2
vs truth_spectrum.npz, emitted-token entropy) and writes <out>.score.json.
``calibrate`` is the GUST paper's marginal-entropy read (Sec. 5.2): target =
per-scale token entropy of the TRAINING record (nats); one global T for every
scale; start at T=1 and step 0.2 against the sign of the finest-two mean gap
until it changes sign; running maximum over T; linear interpolation. Each trial
rollout pools members x steps frames (default 10 x 1460 = the 14,600-frame
training record) so the plug-in entropies are compared at matched counts.

Both passes feed the same FigureStats accumulator with PHYSICAL fields (K,
m/s), so the panels differ only in their source. Grids are south-first.
Ensemble members are independent samples from one initial condition (the paper
perturbs weights instead; a sampled token model is already stochastic).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

FIELDS = ("temperature", "v_wind")  # T0.95 and V0.34 in the LUCIE 2D set
STEPS_PER_DAY = 4  # 6-hourly
STEPS_PER_YEAR = 1460  # 365-day blocks
EXTRATROPICS = np.r_[7:15, 32:40]  # LUCIE_train.py:88 (south-first rows)


def _zonal_power(x):
    """One-sided zonal power |rfft|^2 / n^2 (interior k doubled; DC and
    Nyquist single), the convention of diagnostics.zonal_power_spectrum."""
    import torch

    power = torch.fft.rfft(x.float(), dim=-1).abs().square() / x.shape[-1] ** 2
    power[..., 1 : (x.shape[-1] + 1) // 2] *= 2
    return power


class FigureStats:
    """Streaming statistics over frames (B, len(FIELDS), H, W) that start on a
    00Z boundary: daily/annual ranges of the first field and zonal power
    spectra of every field and its one-step tendency. Torch tensors on the
    frames' device; nothing leaves it until save()."""

    def __init__(self, members: int, height: int, width: int, device="cpu") -> None:
        import torch

        from lucie_tokens.era5_npz import legendre_gauss_area_weights

        shape = (members, height, width)
        full = lambda value: torch.full(shape, value, device=device)  # noqa: E731
        self.day_max, self.day_min = full(-torch.inf), full(torch.inf)
        self.year_max, self.year_min = full(-torch.inf), full(torch.inf)
        zeros = lambda *s: torch.zeros(s, dtype=torch.float64, device=device)  # noqa: E731
        self.diurnal_sum = zeros(height, width)
        self.annual_sum = zeros(height, width)
        self.days = self.years = 0
        k = width // 2 + 1
        # per-row power sums; rows are averaged into regions at plot time
        self.state_power = zeros(len(FIELDS), height, k)
        self.tendency_power = zeros(len(FIELDS), height, k)
        self.state_count = self.tendency_count = 0
        self.global_mean_t = []  # (B,) per step, area-weighted
        # mean-1 latitude weights: the grid mean of t * w is the area mean
        self.weights = torch.as_tensor(
            legendre_gauss_area_weights(height), device=device)[:, None]
        self.previous = None
        self.step = 0

    def update(self, frames) -> None:
        import torch

        frames = torch.as_tensor(frames, device=self.day_max.device).float()
        t = frames[:, 0]
        torch.maximum(self.day_max, t, out=self.day_max)
        torch.minimum(self.day_min, t, out=self.day_min)
        torch.maximum(self.year_max, t, out=self.year_max)
        torch.minimum(self.year_min, t, out=self.year_min)
        self.step += 1
        if self.step % STEPS_PER_DAY == 0:
            self.diurnal_sum += (self.day_max - self.day_min).sum(dim=0)
            self.days += len(t)
            self.day_max.fill_(-torch.inf)
            self.day_min.fill_(torch.inf)
        if self.step % STEPS_PER_YEAR == 0:
            self.annual_sum += (self.year_max - self.year_min).sum(dim=0)
            self.years += len(t)
            self.year_max.fill_(-torch.inf)
            self.year_min.fill_(torch.inf)
        self.state_power += _zonal_power(frames).sum(dim=0)
        self.state_count += len(frames)
        if self.previous is not None:
            self.tendency_power += _zonal_power(frames - self.previous).sum(dim=0)
            self.tendency_count += len(frames)
        self.previous = frames
        self.global_mean_t.append((t * self.weights).mean(dim=(-2, -1)))

    def save(self, path: str | Path, **meta) -> None:
        import torch

        cpu = lambda x: x.double().cpu().numpy()  # noqa: E731
        np.savez(
            path,
            diurnal_range=cpu(self.diurnal_sum) / max(self.days, 1), days=self.days,
            annual_range=cpu(self.annual_sum) / max(self.years, 1), years=self.years,
            state_power=cpu(self.state_power) / max(self.state_count, 1),
            tendency_power=cpu(self.tendency_power) / max(self.tendency_count, 1),
            global_mean_t=torch.stack(self.global_mean_t, dim=1).cpu().numpy().astype(np.float32),
            fields=np.array(FIELDS), **meta,
        )


# variable order of the Zenodo record (dataset_generator.py) shared by
# era5_T30_regridded.npz and the normalized preprocessed/data_inp.npy
ZENODO_VARS = ("temperature", "humidity", "u_wind", "v_wind", "surface_pressure")


def _truth_frames(args) -> np.ndarray:
    """(T, len(FIELDS), H, W) physical frames from either Zenodo layout."""
    if args.npz:
        with np.load(args.npz) as bundle:
            return np.stack([bundle[name] for name in FIELDS], axis=1)
    root = Path(args.preprocessed)
    inputs = np.load(root / "data_inp.npy", mmap_mode="r")  # normalized by raw_*
    mean, std = np.load(root / "raw_means.npy"), np.load(root / "raw_stds.npy")
    index = [ZENODO_VARS.index(name) for name in FIELDS]
    return np.stack([inputs[:, i] * std[i] + mean[i] for i in index], axis=1)


def cmd_truth(args) -> None:
    data = _truth_frames(args)  # (T, 2, H, W)
    stop = len(data) if args.frames is None else args.start + args.frames
    data = data[args.start:stop]
    stats = FigureStats(1, *data.shape[-2:])
    for frame in data:
        stats.update(frame[None])
    stats.save(args.out, source=str(args.npz or args.preprocessed), start=args.start,
               frames=len(data))
    print(f"wrote {args.out}: {len(data)} frames, {stats.days} days, {stats.years} years")


class RolloutScore:
    """Ensemble scores of normalized frames (B, C, H, W) on their own device:
    per-member and pooled pixel histograms, KE spectrum, and emitted-token
    entropy per scale, the definitions climate-diagnostics uses."""

    def __init__(self, members, channels, weights, wind, scales, codebook_size, device="cpu"):
        import torch

        from lucie_tokens import diagnostics as D

        self.D = D
        self.edges = torch.as_tensor(D.HIST_EDGES, dtype=torch.float64, device=device)
        self.bins = len(D.HIST_EDGES) - 1
        self.counts = torch.zeros((members, channels, self.bins), dtype=torch.int64, device=device)
        weights = torch.as_tensor(weights, dtype=torch.float64, device=device)
        self.weights = (weights / weights.sum()).float()
        self.wind = wind
        self.ke_sum = None
        self.frames = 0
        self.token_counts = torch.zeros((len(scales), codebook_size), dtype=torch.int64, device=device)

    def update(self, frames, pyramid) -> None:
        import torch

        b, c = frames.shape[:2]
        values = frames.double().clamp(self.edges[0], self.edges[-1] - 1e-9)
        index = (torch.bucketize(values, self.edges, right=True) - 1).clamp(0, self.bins - 1)
        index += (torch.arange(b * c, device=index.device).view(b, c, 1, 1) * self.bins)
        self.counts += torch.bincount(index.flatten(), minlength=b * c * self.bins).view(b, c, self.bins)
        # KE spectrum: latitude-weighted zonal power, levels and batch averaged
        u, v = (frames[:, w] for w in self.wind)
        power = 0.5 * (_zonal_power(u) + _zonal_power(v))
        ke = torch.einsum("blhk,h->k", power.double(), self.weights.double()) / power.shape[1]
        self.ke_sum = ke if self.ke_sum is None else self.ke_sum + ke
        self.frames += b
        for s, grid in enumerate(pyramid):
            self.token_counts[s] += torch.bincount(grid.flatten(), minlength=self.token_counts.shape[1])

    def result(self, refs: Path, channels) -> dict:
        D = self.D
        with np.load(refs / "truth_hist.npz") as bundle:
            if not np.allclose(bundle["edges"], D.HIST_EDGES):
                raise ValueError("truth histogram bins do not match HIST_EDGES")
            truth = bundle["counts"]
        with np.load(refs / "truth_spectrum.npz") as bundle:
            truth_ke = bundle["kinetic_energy"]
        counts = self.counts.cpu().numpy()
        pooled = counts.sum(axis=0)
        emd = {n: D.histogram_emd(pooled[i], truth[i]) for i, n in enumerate(channels)}
        member = np.array([[D.histogram_emd(counts[m, i], truth[i]) for i in range(len(channels))]
                           for m in range(len(counts))]).mean(axis=1)
        # Nyquist dropped, as in diagnostics.zonal_power_spectrum (even width)
        ke = (self.ke_sum / self.frames).cpu().numpy()[:-1]
        entropy = D.entropy_bits(self.token_counts.cpu().numpy()) * np.log(2)
        return {
            "ensemble_mean_pixel_emd": float(np.mean(list(emd.values()))),
            "ensemble_pixel_emd": emd,
            "member_mean_pixel_emd": {"median": float(np.median(member)), "p05": float(np.percentile(member, 5)),
                                      "p95": float(np.percentile(member, 95)), "all": member.tolist()},
            "ke_rel_l2_vs_data": D.spectrum_metrics(ke, truth_ke)["relative_l2"],
            "token_entropy_nats": entropy.tolist(),
            "ke_spectrum": ke.tolist(),
            "pixel_counts": pooled.tolist(),
        }


def _entropy_response(D, dynamics, initial, members, steps, temperature, device, precision):
    """Per-scale horizon entropy (nats) of the tokens one batched rollout emits,
    counted on the device (one host transfer at the end)."""
    import torch

    codebook = dynamics.scale_masks.shape[1]
    counts = torch.zeros((len(dynamics.scales), codebook), dtype=torch.int64, device=device)
    current = tuple(g.expand(members, *g.shape[1:]).contiguous() for g in initial)
    for _, current in D.iterate_rollout(dynamics, current, steps, temperature=temperature,
                                        device=device, precision=precision):
        for s, grid in enumerate(current):
            counts[s] += torch.bincount(grid.flatten(), minlength=codebook)
    return D.entropy_bits(counts.cpu().numpy()) * np.log(2)


def cmd_calibrate(args) -> None:
    import json

    import torch

    from lucie_tokens import diagnostics as D
    from lucie_tokens.runtime import resolve_device, resolve_precision
    from lucie_tokens.workflows import load_dynamics_checkpoint

    device = resolve_device(args.device)
    precision = resolve_precision(args.precision, device)
    torch.manual_seed(args.seed)
    dynamics, _ = load_dynamics_checkpoint(args.dynamics, device)
    codebook = dynamics.scale_masks.shape[1]
    train_tokens, _, _ = D.load_token_file(args.train_tokens)
    target = D.entropy_bits(D.scale_token_counts(train_tokens, dynamics.scales, codebook)) * np.log(2)
    tokens, _, _ = D.load_token_file(args.token_file)
    initial = D.initial_pyramid(tokens[args.start], dynamics.scales, device)
    finest = slice(-2, None)
    tried: dict[float, np.ndarray] = {}

    def measure(t):
        began = time.time()
        with torch.inference_mode():
            tried[t] = _entropy_response(D, dynamics, initial, args.members, args.steps, t,
                                         device, precision)
        gap = float((tried[t][finest] - target[finest]).mean())
        print(f"T={t:.2f} gap={gap:+.4f} nats  per-scale delta "
              f"{np.round(tried[t] - target, 3).tolist()}  ({time.time() - began:.0f}s)", flush=True)
        return gap

    t = 1.0
    gap = measure(t)
    direction = -1.0 if gap > 0 else 1.0
    clamped = False
    while True:
        nxt = round(t + direction * args.step, 4)
        if not args.t_min <= nxt <= args.t_max:
            clamped = True
            break
        t = nxt
        if np.sign(measure(t)) != np.sign(gap):
            break
    # running maximum over increasing T makes the response monotone (Sec. 5.2)
    temps = np.array(sorted(tried))
    response = np.maximum.accumulate(np.stack([tried[x] for x in temps]), axis=0)
    gaps = (response[:, finest] - target[finest]).mean(axis=1)
    if clamped:
        solved = float(temps[np.argmin(np.abs(gaps))])
    else:
        # the running maximum makes the gap non-decreasing in T
        i = int(np.flatnonzero(np.diff(np.sign(gaps)) != 0)[0])
        solved = float(np.interp(0.0, gaps[i:i + 2], temps[i:i + 2]))
    result = {
        "dynamics": str(args.dynamics), "members": args.members, "steps": args.steps,
        "scales": [f"{h}x{w}" for h, w in dynamics.scales], "units": "nats",
        "data_entropy": target.tolist(),
        "temperatures": temps.tolist(),
        "response_running_max": response.tolist(),
        "finest_two_gap": gaps.tolist(),
        "solved_temperature": solved, "clamped": clamped,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"T* = {solved:.3f}{' (clamped)' if clamped else ''} -> {args.out}")


def cmd_rollout(args) -> None:
    import json

    import torch

    from lucie_tokens import diagnostics as D
    from lucie_tokens.runtime import autocast_context, resolve_device, resolve_precision
    from lucie_tokens.tokens import load_tokenizer_checkpoint
    from lucie_tokens.workflows import load_dynamics_checkpoint

    device = resolve_device(args.device)
    precision = resolve_precision(args.precision, device)
    torch.manual_seed(args.seed)
    dynamics, _ = load_dynamics_checkpoint(args.dynamics, device)
    tokenizer, _ = load_tokenizer_checkpoint(args.tokenizer, device)
    if getattr(dynamics, "conditioning_dim", 0) > 0:
        raise SystemExit("conditioned models need the forcing clock; not wired here")
    tokens, _, stamps = D.load_token_file(args.token_file)
    if stamps is not None and int(stamps[args.start]) % 86400:
        raise SystemExit("start frame is not on a 00Z boundary; diurnal days would be shifted")
    channels = args.channels.split(",")
    collect = [channels.index(name) for name in FIELDS]
    refs = Path(args.refs)
    with np.load(refs / "normalize_mean.npz") as m, np.load(refs / "normalize_std.npz") as s:
        mean = np.array([float(m[n]) for n in FIELDS])[None, :, None, None]
        std = np.array([float(s[n]) for n in FIELDS])[None, :, None, None]
    steps = args.years * STEPS_PER_YEAR if args.steps is None else args.steps
    start = D.initial_pyramid(tokens[args.start], dynamics.scales, device)
    current = tuple(g.expand(args.members, *g.shape[1:]).contiguous() for g in start)
    stats = None
    score = None
    if args.score:
        from lucie_tokens.era5_npz import legendre_gauss_area_weights

        wind = tuple(slice(channels.index(n), channels.index(n) + 1) for n in ("u_wind", "v_wind"))
        score = RolloutScore(args.members, len(channels), legendre_gauss_area_weights(48), wind,
                             dynamics.scales, dynamics.scale_masks.shape[1], device)
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device)
    std_t = torch.as_tensor(std, dtype=torch.float32, device=device)
    snapshot_steps = sorted(int(x) for x in args.snapshot_steps.split(",")) if args.snapshot_steps else []
    snapshots = {}
    began = time.time()
    with torch.inference_mode():
        for call, current in D.iterate_rollout(
            dynamics, current, steps, temperature=args.temperature,
            device=device, precision=precision,
        ):
            with autocast_context(device, precision):
                normalized = tokenizer.decode_indices(current, final_only=True)[0].float()
            if score is not None:
                score.update(normalized, current)
            if call + 1 in snapshot_steps:  # member 0, normalized, every channel
                snapshots[f"step_{call + 1}"] = normalized[0].cpu().numpy()
            frames = normalized[:, collect] * std_t + mean_t
            if stats is None:
                stats = FigureStats(args.members, *frames.shape[-2:], device=device)
            stats.update(frames)
            if (call + 1) % args.log_every == 0:
                # one host sync per log line instead of a finiteness check per step
                if not bool(torch.isfinite(normalized).all()):
                    raise SystemExit(f"non-finite frame by step {call + 1}")
                rate = (call + 1) / (time.time() - began)
                print(f"step {call + 1}/{steps}  {rate:.1f} steps/s  "
                      f"eta {(steps - call - 1) / rate / 60:.1f} min", flush=True)
            if (call + 1) % STEPS_PER_YEAR == 0 or call + 1 == steps:
                stats.save(args.out, dynamics=str(args.dynamics), members=args.members,
                           steps_done=call + 1, temperature=args.temperature)
    if snapshots:
        np.savez(str(args.out).removesuffix(".npz") + ".snapshots.npz", channels=np.array(channels), **snapshots)
    print(f"wrote {args.out}: {args.members} members x {steps} steps, "
          f"{stats.days} member-days, {stats.years} member-years, {time.time() - began:.0f}s")
    if score is not None:
        result = score.result(refs, channels)
        result.update(dynamics=str(args.dynamics), members=args.members, steps=steps,
                      temperature=args.temperature, seconds=round(time.time() - began))
        path = Path(str(args.out).removesuffix(".npz") + ".score.json")
        path.write_text(json.dumps(result, indent=2))
        print(f"ensemble EMD {result['ensemble_mean_pixel_emd']:.4f}  member median "
              f"{result['member_mean_pixel_emd']['median']:.4f}  KE relL2 "
              f"{result['ke_rel_l2_vs_data']:.3f} -> {path}")


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _maps(plt, panels, key, title, vmax, path):
    from lucie_tokens.coastline import draw_coastlines

    roll = 48  # Greenwich in the middle, as in the paper
    dlon = 3.75
    extent = [-180 - dlon / 2, 180 - dlon / 2, -90, 90]
    # one column like the paper for up to three panels, otherwise a 2-wide grid
    ncols = 1 if len(panels) <= 3 else 2
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 3.3 * nrows + 0.6), squeeze=False,
                             gridspec_kw={"hspace": 0.5})
    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)
    for ax, (label, bundle) in zip(axes.ravel(), panels):
        field = np.roll(bundle[key], roll, axis=-1)
        image = ax.imshow(field, origin="lower", extent=extent, cmap="viridis",
                          vmin=0, vmax=vmax, aspect="auto", interpolation="nearest")
        draw_coastlines(image, roll=roll, color="k", linewidth=0.7, alpha=1.0)
        ax.set_title(f"{label}\n{title}", fontsize=12, fontweight="bold")
        ax.set_xticks([-180, -90, 0, 90, 180], ["180°", "90°W", "0°", "90°E", "180°"])
        ax.set_yticks([-45, 0, 45], ["45°S", "0°", "45°N"])
    fig.colorbar(image, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.04 / nrows ** 0.5,
                 pad=0.09, extend="max", label="Kelvin")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def _spectra(plt, curves, key, kind, path):
    names = {"temperature": "$T_{0.95}$", "v_wind": "$V_{0.34}$"}
    regions = {"Full Globe": slice(None), "Extra Tropic": EXTRATROPICS}
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.4))
    for r, field in enumerate(FIELDS):
        for c, (region, rows) in enumerate(regions.items()):
            ax = axes[r, c]
            for label, bundle, color, width in curves:
                fi = list(bundle["fields"]).index(field)
                power = bundle[key][fi][rows].mean(axis=0)
                k = np.arange(len(power))
                ax.loglog(k[1:], power[1:], color=color, lw=width, label=label)
            ax.set_title(f"{names[field]} {region} {kind} Power Spectra", fontsize=10)
            ax.set_xticks([1, 2, 4, 8, 16, 32, 48], ["1", "2", "4", "8", "16", "32", "48"])
            ax.set_xlim(1, 48)
            ax.set_ylabel("Power")
            if r == 1:
                ax.set_xlabel("Wave Number")
    axes[0, 0].legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def cmd_plot(args) -> None:
    plt = _plt()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    truth = np.load(args.truth)
    models = [(spec.split("=", 1)[0], np.load(spec.split("=", 1)[1])) for spec in args.model]
    # paper order (models above ERA5) for one model; ERA5 leads a ladder grid
    panels = models + [("ERA5", truth)] if len(models) <= 2 else [("ERA5", truth)] + models
    _maps(plt, panels, "diurnal_range", "Average Diurnal Range", 8, out / "fig4_diurnal_range.png")
    _maps(plt, panels, "annual_range", "Average Annual Temperature Range", 70,
          out / "fig5_annual_range.png")
    blues = plt.get_cmap("Blues")(np.linspace(0.45, 0.95, max(len(models), 1)))
    curves = [("ERA5", truth, "red", 1.8)] + [
        (label, bundle, blues[i], 1.4) for i, (label, bundle) in enumerate(models)
    ]
    _spectra(plt, curves, "state_power", "", out / "fig10a_state_spectra.png")
    _spectra(plt, curves, "tendency_power", "Tendency", out / "fig10b_tendency_spectra.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("truth", help="ERA5 statistics (CPU)")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--npz", help="era5_T30_regridded.npz (physical units)")
    source.add_argument("--preprocessed", help="Zenodo preprocessed/ dir (data_inp.npy + raw_*.npy)")
    p.add_argument("--start", type=int, default=0, help="first frame (must be 00Z)")
    p.add_argument("--frames", type=int, help="default: all frames from --start")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_truth)

    p = sub.add_parser("rollout", help="ensemble rollout statistics (GPU)")
    p.add_argument("--dynamics", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--token-file", required=True)
    p.add_argument("--refs", required=True, help="dir with normalize_{mean,std}.npz")
    p.add_argument("--channels", default="temperature,humidity,u_wind,v_wind,surface_pressure")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--members", type=int, default=10)
    p.add_argument("--years", type=int, default=10)
    p.add_argument("--steps", type=int, help="override --years (smoke tests)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--precision", default="auto")
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--score", action="store_true",
                   help="also score vs the refs' truth_hist/truth_spectrum -> <out>.score.json")
    p.add_argument("--snapshot-steps", help="comma list of steps: save member 0's normalized frame "
                   "-> <out>.snapshots.npz")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_rollout)

    p = sub.add_parser("calibrate", help="GUST marginal-entropy temperature read")
    p.add_argument("--dynamics", required=True)
    p.add_argument("--train-tokens", required=True, help="target: training-record tokens")
    p.add_argument("--token-file", required=True, help="initial condition (val)")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--members", type=int, default=10)
    p.add_argument("--steps", type=int, default=STEPS_PER_YEAR)
    p.add_argument("--step", type=float, default=0.2)
    p.add_argument("--t-min", type=float, default=0.2)
    p.add_argument("--t-max", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--precision", default="auto")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("plot", help="Fig. 4/5 maps and Fig. 10 spectra")
    p.add_argument("--truth", required=True)
    p.add_argument("--model", action="append", default=[], help="LABEL=stats.npz (repeatable)")
    p.add_argument("--out-dir", required=True)
    p.set_defaults(func=cmd_plot)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
