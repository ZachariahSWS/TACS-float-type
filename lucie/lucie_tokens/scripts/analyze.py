#!/usr/bin/env python3
"""One command for every analysis of a trained LUCIE model.

The numbers live in ``lucie_tokens.diagnostics``; this file is the argparse
front end plus the presentation-only renderers (videos, montages, curve plots).
The standard rollout report itself is ``python -m lucie_tokens
climate-diagnostics`` (unchanged; the Bridges diag chains call it).

  analyze.py mechanism       --dynamics <ckpt> --token-root <root> --out m.json
  analyze.py oracle          --dynamics <ckpt> --token-root <root> --out <dir>
  analyze.py chain           --data-tokens val/source.h5 --rollout-tokens lbl=<h5>
  analyze.py code-reuse      --token-root <root> --scales 1x2,2x4,3x6,6x12,12x24
  analyze.py calibrate       --dynamics <ckpt> --token-file val/source.h5 --out <dir>
  analyze.py refs            --config <era5 toml> --output <refs dir>
  analyze.py animate-channel --dynamics .. --tokenizer .. --token-file .. --out t.mp4
  analyze.py animate-panels  --dynamics .. --tokenizer .. --token-file .. --out p.mp4
  analyze.py animate-compare --dynamics .. --tokenizer .. --token-file .. --truth-npz ..
  analyze.py animate-compare --pred-npz frames.npz --truth-npz .. --dataset-root ..
  analyze.py decode-rollout  --tokenizer .. --rollout-tokens diag_*/rollout_tokens.h5 --out frames.npz
  analyze.py snapshots       --dynamics .. --tokenizer .. --token-file .. --out-dir ..
  analyze.py curves          out.png LABEL=metrics.jsonl ...
  analyze.py autopsy         <ckpt> [<ckpt> ...] [--out <dir>]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from lucie_tokens import diagnostics as D
from lucie_tokens.coastline import draw_coastlines
from lucie_tokens.config import load_config
from lucie_tokens.runtime import atomic_json
from lucie_tokens.schema import STATE_CHANNELS, split_flat_tokens
from lucie_tokens.tokens import load_tokenizer_checkpoint
from lucie_tokens.workflows import load_dynamics_checkpoint


def _scales(text: str):
    return tuple(tuple(int(v) for v in part.split("x")) for part in text.split(","))


def _temperature(text: str):
    values = [float(v) for v in text.split(",")]
    return values[0] if len(values) == 1 else tuple(values)


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# token-space analyses
# ---------------------------------------------------------------------------


def cmd_mechanism(args) -> None:
    result = D.rollout_mechanism(
        args.dynamics, args.token_root, held_out=args.held_out.split(","),
        train_pairs=args.train_pairs, steps=args.steps, temperature=args.temperature,
        batch_size=args.batch, cadence_hours=args.cadence_hours,
        dataset_root=args.dataset_root, device_name=args.device, precision=args.precision,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(D.format_mechanism(result))
    print(f"-> {out}")


def cmd_oracle(args) -> None:
    report, spectra = D.scale_oracle(
        args.dynamics, args.token_root, splits=args.splits.split(","),
        tokenizer_checkpoint=args.tokenizer, max_steps=args.max_steps,
        temperature=args.temperature, channels=args.channels.split(","),
        u_channel=args.u_channel, v_channel=args.v_channel, low_m=args.low_m,
        cadence_hours=args.cadence_hours, dataset_root=args.dataset_root,
        device_name=args.device, precision=args.precision,
        progress=lambda line: print(line, flush=True),
    )
    print(D.format_oracle(report))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "scale_oracle.json").write_text(json.dumps(report, indent=2))
    np.savez(out / "scale_oracle_spectra.npz", **{f"{m}_ke": ke for m, ke in spectra.items()})
    print(f"-> {out}")


def cmd_chain(args) -> None:
    scales = _scales(args.scales)
    label = args.scales.split(",")[args.scale]
    data_flat, _, _ = D.load_token_file(args.data_tokens)
    series = {"data": D.coarse_sequence(data_flat, scales, args.scale)}
    for spec in args.rollout_tokens:
        name, path = spec.split("=", 1)
        tokens, _, _ = D.load_token_file(path)
        series[name] = D.coarse_sequence(tokens, scales, args.scale)
    report = {}
    print(f"coarse-chain statistics at scale {label} ({series['data'].shape[1]} tokens/frame)\n")
    print(f"{'series':16s} {'states':>7s} {'perplex':>8s} {'H(bits)':>8s} "
          f"{'top1':>6s} {'change':>7s} {'dwell':>7s} {'decorr':>7s}")
    for name, sequence in series.items():
        s = D.chain_stats(sequence, args.codebook_size)
        report[name] = {**s, "steps": int(sequence.shape[0])}
        print(f"{name:16s} {s['states_visited']:7d} {s['perplexity']:8.1f} "
              f"{s['marginal_entropy_bits']:8.2f} {s['top1_share']:6.3f} "
              f"{s['change_rate']:7.3f} {s['mean_dwell_steps']:7.2f} "
              f"{str(s['decorrelation_lag_steps']):>7s}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")


def cmd_code_reuse(args) -> None:
    root = Path(args.token_root)
    train, _, _ = D.load_token_file(root / "train" / "source.h5")
    held, _, _ = D.load_token_splits(root, args.held_out.split(","))
    rows = D.code_reuse_table(train, held, _scales(args.scales), args.codebook_size)
    print(f"== {root.name}: train frames {len(train)}, held-out frames {len(held)}")
    print(f"{'scale':>6} {'pos':>5} {'distinct_tr':>11} {'tok/code':>8} {'ho_unseen':>9} "
          f"{'ho_code<5':>9} {'ho_code<20':>10} {'ho_unseen_pos':>13} {'tr_uniq_tuple':>13}")
    for r in rows:
        print(f"{r['scale']:>6} {r['positions']:5d} {r['distinct_train_codes']:11d} "
              f"{r['tokens_per_code']:8.1f} {r['held_out_unseen']:9.3f} {r['held_out_code_lt5']:9.3f} "
              f"{r['held_out_code_lt20']:10.3f} {r['held_out_unseen_per_position']:13.3f} "
              f"{r['train_unique_tuple_fraction']:13.3f}")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))


def cmd_calibrate(args) -> None:
    result, data_entropy, responses = D.calibrate_temperature(
        args.dynamics, args.token_file,
        temperatures=[float(v) for v in args.temperatures.split(",")],
        steps=args.steps, start=args.start, window_steps=args.window_steps,
        cadence_hours=args.cadence_hours, dataset_root=args.dataset_root,
        progress=lambda line: print(line, flush=True),
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "calibration.json", result)
    D.save_calibration_plot(out, result, data_entropy, responses)
    print(json.dumps({k: result[k] for k in (
        "solved_temperature_per_scale", "global_temperature_finest_two")}, indent=2))
    print(f"-> {out}")


def cmd_refs(args) -> None:
    summary = D.build_era5_references(
        load_config(args.config), args.output, split=args.split,
        panels_stride=args.panels_stride, panels_fields=args.panels_fields.split(","),
        batch=args.batch,
    )
    print(json.dumps(summary, indent=2))


def cmd_autopsy(args) -> None:
    reports = [D.codebook_autopsy(path) for path in args.checkpoints]
    print(json.dumps(reports, indent=2))
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "autopsy.json").write_text(json.dumps(reports, indent=2))


# ---------------------------------------------------------------------------
# rollout visuals (matplotlib / imageio only here)
# ---------------------------------------------------------------------------


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _load_models(args):
    device = _device()
    dynamics, _ = load_dynamics_checkpoint(args.dynamics, device)
    tokenizer, _ = load_tokenizer_checkpoint(args.tokenizer, device)
    tokens, _, stamps = D.load_token_file(args.token_file)
    conditioner = D.make_conditioner(dynamics, args.dataset_root)
    if conditioner is not None and stamps is None:
        raise SystemExit("conditioned model needs timestamps in the token file")
    return device, dynamics, tokenizer, tokens, stamps, conditioner


def _rolled_frames(args, start, model_calls, keep, collect):
    """Decode channel subset ``collect`` at the model calls where ``keep(call)``
    is true; returns (T, len(collect), H, W)."""
    device, dynamics, tokenizer, tokens, stamps, conditioner = _load_models(args)
    frames = []
    with torch.inference_mode():
        for call, current in D.iterate_rollout(
            dynamics, D.initial_pyramid(tokens[start], dynamics.scales, device),
            model_calls, temperature=_temperature(str(args.temperature)),
            conditioner=conditioner,
            start_timestamp=int(stamps[start]) if stamps is not None else 0,
            cadence_hours=args.cadence_hours, device=device,
        ):
            if keep(call):
                decoded = tokenizer.decode_indices(current, final_only=True)[0]
                frames.append(decoded[0, collect].float().cpu().numpy())
    return np.stack(frames).astype(np.float32)


def _channel_names(args):
    return tuple(args.channels.split(",")) if args.channels else STATE_CHANNELS


def _write_video(fig, images_setter, n_frames, title_setter, out, fps):
    import imageio.v2 as imageio

    writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=8)
    for frame in range(n_frames):
        images_setter(frame)
        title_setter(frame)
        fig.canvas.draw()
        writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    writer.close()
    print(f"wrote {out}: {n_frames} frames @ {fps}fps ({n_frames / fps:.1f}s)")


def cmd_animate_channel(args) -> None:
    """One channel (or relative vorticity from a wind pair) of a sampled rollout."""
    plt = _plt()
    names = _channel_names(args)
    vort = args.vorticity_level is not None or args.vort_u_channel
    if vort:
        u_name = args.vort_u_channel or f"U-wind_{args.vorticity_level}"
        v_name = args.vort_v_channel or f"V-wind_{args.vorticity_level}"
        collect = [names.index(u_name), names.index(v_name)]
    else:
        channel = names.index(args.channel) if args.channel in names else int(args.channel)
        collect = [channel]
    stacked = _rolled_frames(args, args.start, args.steps, lambda c: c % args.stride == 0, collect)
    meta = D.GridMetadata(args.dataset_root, stacked.shape[-2])
    if vort:
        frames = D.relative_vorticity(
            meta.physical(stacked[:, 0], u_name), meta.physical(stacked[:, 1], v_name), meta.lat
        )
        label, cmap = "relative vorticity", "RdBu_r"
        limit = float(np.percentile(np.abs(frames), 98))
        vmin, vmax = -limit, limit
    else:
        name = names[collect[0]]
        frames = meta.physical(stacked[:, 0], name)
        label = name if meta.mean is not None and name in meta.mean.files else f"{name} (normalized)"
        cmap = args.cmap
        vmin, vmax = np.percentile(frames, [2, 98])
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    im = ax.imshow(frames[0], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[0, 360, -90, 90], aspect="auto")
    draw_coastlines(im)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    title = ax.set_title("")
    fig.tight_layout()
    _write_video(
        fig, lambda i: im.set_data(frames[i]), len(frames),
        lambda i: title.set_text(
            f"{label}   step {i * args.stride}   (day {i * args.stride * args.cadence_hours / 24:.1f})"),
        args.out, args.fps,
    )
    plt.close(fig)


def _four_fields(args, names):
    """Channel names of the four panel fields (T, Q, U, V + vorticity winds)."""
    return {
        "T": args.temp_channel or f"Temperature_{args.temp_level}",
        "Q": args.humidity_channel or f"Specific_Humidity_{args.humidity_level}",
        "U": args.u_channel or f"U-wind_{args.wind_level}",
        "V": args.v_channel or f"V-wind_{args.wind_level}",
        "Uv": args.vort_u_channel or args.u_channel or f"U-wind_{args.vorticity_level}",
        "Vv": args.vort_v_channel or args.v_channel or f"V-wind_{args.vorticity_level}",
    }


def cmd_animate_panels(args) -> None:
    """2x2 rollout video: vorticity, temperature, wind speed, humidity."""
    plt = _plt()
    names = _channel_names(args)
    needed = _four_fields(args, names)
    if args.frames_npz:
        bundle = np.load(args.frames_npz)
        raw = bundle["raw"].astype(np.float32)
        collect = [int(c) for c in bundle["collect"]]
    else:
        collect = sorted({names.index(n) for n in needed.values()})
        raw = _rolled_frames(args, args.start, args.steps, lambda c: c % args.stride == 0, collect)
    slot = {channel: i for i, channel in enumerate(collect)}
    meta = D.GridMetadata(args.dataset_root, raw.shape[-2])

    def field(key):
        name = needed[key]
        return meta.physical(raw[:, slot[names.index(name)]], name)

    temperature = field("T")
    humidity = field("Q")
    wind_speed = np.sqrt(field("U") ** 2 + field("V") ** 2)
    vorticity = D.relative_vorticity(field("Uv"), field("Vv"), meta.lat)
    zlim = float(np.percentile(np.abs(vorticity), 98))
    panels = [
        (vorticity, f"relative vorticity ({needed['Uv']})", "RdBu_r", -zlim, zlim),
        (temperature, f"temperature ({needed['T']})", "turbo", *np.percentile(temperature, [2, 98])),
        (wind_speed, f"wind speed ({needed['U']})", "viridis", *np.percentile(wind_speed, [2, 98])),
        (humidity, f"specific humidity ({needed['Q']})", "Blues", *np.percentile(humidity, [2, 98])),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6))
    images = []
    for ax, (data, title, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        image = ax.imshow(data[0], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                          extent=[0, 360, -90, 90], aspect="auto")
        draw_coastlines(image)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, shrink=0.82, pad=0.02)
        images.append(image)
    suptitle = fig.suptitle("")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    def set_images(i):
        for image, (data, *_rest) in zip(images, panels):
            image.set_data(data[i])

    _write_video(
        fig, set_images, raw.shape[0],
        lambda i: suptitle.set_text(
            f"{args.title_prefix}   step {i * args.stride}   "
            f"(day {i * args.stride * args.cadence_hours / 24:.1f})"),
        args.out, args.fps,
    )
    plt.close(fig)


def cmd_animate_compare(args) -> None:
    """2x3 truth-vs-emulator video (wind speed, temperature, humidity), rollout
    started at the first frame so the two rows are time-aligned."""
    plt = _plt()
    names = _channel_names(args)
    needed = _four_fields(args, names)
    ti, qi, ui, vi = (names.index(needed[k]) for k in ("T", "Q", "U", "V"))
    collect = sorted({ti, qi, ui, vi})
    slot = {c: i for i, c in enumerate(collect)}
    truth = np.load(args.truth_npz)
    truth_raw = truth["raw"].astype(np.float32)
    truth_slot = {int(c): i for i, c in enumerate(truth["collect"])}
    n_truth = truth_raw.shape[0]
    if args.pred_npz:
        # frames from decode-rollout: frame k is the state after model call
        # steps[k] (a multiple of --stride), i.e. truth panel steps[k] // stride
        bundle = np.load(args.pred_npz)
        slot = {int(c): i for i, c in enumerate(bundle["collect"])}
        if missing := [names[c] for c in (ti, qi, ui, vi) if c not in slot]:
            raise SystemExit(f"{args.pred_npz} lacks channels {missing}")
        panel = bundle["steps"] // args.stride
        keep = panel < n_truth
        pred_raw = bundle["raw"][keep].astype(np.float32)
        truth_raw = truth_raw[panel[keep]]
        days = bundle["steps"][keep] * args.cadence_hours / 24
    else:
        if not (args.dynamics and args.tokenizer and args.token_file):
            raise SystemExit("--dynamics, --tokenizer and --token-file are required without --pred-npz")
        # keep the initial condition, then every stride
        device, dynamics, tokenizer, tokens, stamps, conditioner = _load_models(args)
        with torch.inference_mode():
            current = D.initial_pyramid(tokens[0], dynamics.scales, device)
            pred = [tokenizer.decode_indices(current, final_only=True)[0][0, collect].float().cpu().numpy()]
            for call, current in D.iterate_rollout(
                dynamics, current, (n_truth - 1) * args.stride,
                temperature=_temperature(str(args.temperature)), conditioner=conditioner,
                start_timestamp=int(stamps[0]) if stamps is not None else 0,
                cadence_hours=args.cadence_hours, device=device,
            ):
                if (call + 1) % args.stride == 0:
                    pred.append(tokenizer.decode_indices(current, final_only=True)[0][0, collect].float().cpu().numpy())
        pred_raw = np.stack(pred).astype(np.float32)
        days = np.arange(len(pred_raw)) * args.stride * args.cadence_hours / 24
    n = min(len(pred_raw), len(truth_raw))
    pred_raw, truth_raw = pred_raw[:n], truth_raw[:n]
    meta = D.GridMetadata(args.dataset_root, pred_raw.shape[-2])

    def fields(raw, smap):
        u = meta.physical(raw[:, smap[ui]], needed["U"])
        v = meta.physical(raw[:, smap[vi]], needed["V"])
        return (np.sqrt(u ** 2 + v ** 2), meta.physical(raw[:, smap[ti]], needed["T"]),
                meta.physical(raw[:, smap[qi]], needed["Q"]))

    pw, pt, ph = fields(pred_raw, slot)
    tw, tt, th = fields(truth_raw, truth_slot)

    def limits(a, b):
        return tuple(np.percentile(np.concatenate([a.ravel(), b.ravel()]), [2, 98]))

    cols = [
        (f"wind speed ({needed['U']})", "viridis", limits(pw, tw), tw, pw),
        (f"temperature ({needed['T']})", "turbo", limits(pt, tt), tt, pt),
        (f"specific humidity ({needed['Q']})", "Blues", limits(ph, th), th, ph),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 6))
    images = [[None] * 3, [None] * 3]
    for c, (title, cmap, (vmin, vmax), tdat, pdat) in enumerate(cols):
        for r, (data, label) in enumerate([(tdat, "ERA5 truth"), (pdat, "emulator")]):
            ax = axes[r, c]
            images[r][c] = ax.imshow(data[0], origin=args.origin, cmap=cmap, vmin=vmin, vmax=vmax,
                                     extent=[0, 360, -90, 90], aspect="auto")
            draw_coastlines(images[r][c])
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=11)
            if c == 0:
                ax.set_ylabel(label, fontsize=12)
        fig.colorbar(images[0][c], ax=[axes[0, c], axes[1, c]], shrink=0.7, pad=0.02)
    suptitle = fig.suptitle("")

    def set_images(i):
        for c, (_t, _cm, _l, tdat, pdat) in enumerate(cols):
            images[0][c].set_data(tdat[i])
            images[1][c].set_data(pdat[i])

    _write_video(
        fig, set_images, n,
        lambda i: suptitle.set_text(f"rollout from val start   day {days[i]:.0f}"),
        args.out, args.fps,
    )
    plt.close(fig)


def cmd_decode_rollout(args) -> None:
    """Decode a stored rollout (climate-diagnostics' rollout_tokens.h5) at every
    stride-th model call into normalized frames, so a video of exactly the
    scored rollout needs no second (possibly GPU) rollout."""
    import h5py

    device = _device()
    tokenizer, _ = load_tokenizer_checkpoint(args.tokenizer, device)
    with h5py.File(args.rollout_tokens, "r") as handle:
        tokens = handle["tokens"][...]
        scales = [tuple(s) for s in json.loads(handle.attrs["scales"])]
    collect = [int(c) for c in args.collect.split(",")]
    # stored frame i is the state after model call i + 1
    calls = np.arange(args.stride, len(tokens) + 1, args.stride)
    # resumable: the login node kills long CPU jobs, so the output is rewritten
    # (atomically) every --save-every frames and a rerun continues from it
    frames = []
    if os.path.exists(args.out):
        prev = np.load(args.out)
        if (list(prev["collect"]) == collect
                and np.array_equal(prev["steps"], calls[: len(prev["steps"])])):
            frames = list(prev["raw"])
            print(f"resuming at {len(frames)}/{len(calls)}", flush=True)

    def save():
        tmp = f"{args.out}.tmp.npz"
        np.savez(tmp, raw=np.stack(frames).astype(np.float32),
                 collect=np.asarray(collect, dtype=np.int64), steps=calls[: len(frames)].astype(np.int64))
        os.replace(tmp, args.out)

    with torch.inference_mode():
        for start in range(len(frames), len(calls), args.batch):
            chunk = calls[start : start + args.batch]
            pyramid = split_flat_tokens(torch.from_numpy(tokens[chunk - 1].astype(np.int64)).to(device), scales)
            frames.extend(tokenizer.decode_indices(pyramid, final_only=True)[0][:, collect].float().cpu().numpy())
            if len(frames) % args.save_every < args.batch or len(frames) == len(calls):
                save()
                print(f"decoded {len(frames)}/{len(calls)}", flush=True)
    print(f"wrote {args.out}: {len(frames)} frames (model calls {calls[0]}..{calls[-1]})")


def cmd_snapshots(args) -> None:
    """Still frames of one rollout at chosen steps: a prediction-only montage,
    or truth | prediction | difference per field when --truth-npz is given."""
    plt = _plt()
    want = [int(s) for s in args.steps.split(",")]
    names = list(_channel_names(args))
    pred = _rolled_frames(args, args.start, max(want), lambda c: (c + 1) in want,
                          list(range(len(names))))
    meta = D.GridMetadata(args.dataset_root, pred.shape[-2])

    def fields(stack):
        u = meta.physical(stack[:, names.index(args.u_channel)], args.u_channel)
        v = meta.physical(stack[:, names.index(args.v_channel)], args.v_channel)
        return {
            "vorticity": (D.relative_vorticity(u, v, meta.lat), "relative vorticity", "RdBu_r"),
            "temperature": (meta.physical(stack[:, names.index(args.temp_channel)], args.temp_channel),
                            "temperature", "turbo"),
            "wind_speed": (np.sqrt(u ** 2 + v ** 2), "wind speed", "viridis"),
            "humidity": (meta.physical(stack[:, names.index(args.humidity_channel)], args.humidity_channel),
                         "specific humidity", "Blues"),
        }

    os.makedirs(args.out_dir, exist_ok=True)
    pred_fields = fields(pred)

    def limits(data, key):
        if key == "vorticity":
            lim = float(np.percentile(np.abs(data), 98)) or 1e-9
            return -lim, lim
        return tuple(float(x) for x in np.percentile(data, [2, 98]))

    def panel(ax, data, cmap, vmin, vmax, fig):
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower",
                       aspect="auto", interpolation="nearest")
        draw_coastlines(im)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.032, pad=0.02)

    if args.truth_npz:
        bundle = np.load(args.truth_npz)
        truth_all = np.asarray(bundle["frames"], dtype=np.float32)
        truth_steps = [int(s) for s in bundle["steps"]]
        missing = [s for s in want if s not in truth_steps]
        if missing:
            raise SystemExit(f"truth npz lacks steps {missing}; has {truth_steps}")
        truth_fields = fields(np.stack([truth_all[truth_steps.index(s)] for s in want]))
        for key, (t_data, title, cmap) in truth_fields.items():
            p_data = pred_fields[key][0]
            vmin, vmax = limits(t_data, key)
            dlim = float(np.percentile(np.abs(p_data - t_data), 98)) or 1e-9
            fig, axes = plt.subplots(len(want), 3, figsize=(11.5, 2.05 * len(want) + 0.9), squeeze=False)
            for r, step in enumerate(want):
                for c, (data, sub, cm, lo, hi) in enumerate((
                    (t_data[r], "ground truth", cmap, vmin, vmax),
                    (p_data[r], "prediction", cmap, vmin, vmax),
                    (p_data[r] - t_data[r], "difference", "RdBu_r", -dlim, dlim),
                )):
                    panel(axes[r][c], data, cm, lo, hi, fig)
                    if r == 0:
                        axes[r][c].set_title(sub, fontsize=11)
                    if c == 0:
                        axes[r][c].set_ylabel(f"step {step}\n{step * args.cadence_hours / 24:.0f} d", fontsize=9)
            fig.suptitle(f"{args.label} — {title} — truth vs prediction (start {args.start})", fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            path = os.path.join(args.out_dir, f"{args.label}_{key}.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"wrote {path}", flush=True)
    else:
        keys = list(pred_fields)
        fig, axes = plt.subplots(len(want), len(keys),
                                 figsize=(2.7 * len(keys) + 0.6, 2.05 * len(want) + 0.9), squeeze=False)
        for r, step in enumerate(want):
            for c, key in enumerate(keys):
                data, title, cmap = pred_fields[key]
                panel(axes[r][c], data[r], cmap, *limits(data, key), fig)
                if r == 0:
                    axes[r][c].set_title(title, fontsize=11)
                if c == 0:
                    axes[r][c].set_ylabel(f"step {step}\n{step * args.cadence_hours / 24:.0f} d", fontsize=9)
        temp_str = str(args.temperature) if "," not in str(args.temperature) else "per-scale"
        fig.suptitle(f"{args.label} — predicted rollout state at steps {','.join(map(str, want))} "
                     f"(T={temp_str}, start {args.start})", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        path = os.path.join(args.out_dir, f"{args.label}_montage.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"wrote {path}", flush=True)
    np.savez_compressed(os.path.join(args.out_dir, f"{args.label}_snapshots.npz"),
                        prediction=pred, steps=np.array(want), channels=np.array(names))


def cmd_curves(args) -> None:
    """val/train loss per epoch for several runs' metrics.jsonl, plus lr."""
    plt = _plt()
    runs = []
    for spec in args.runs:
        label, _, path = spec.partition("=")
        rows = sorted((json.loads(l) for l in Path(path).read_text().splitlines() if l),
                      key=lambda m: m["epoch"])
        runs.append((label, rows))
    has_lr = any("lr" in rows[0] for _, rows in runs)
    figure, axes = plt.subplots(1 + int(has_lr), 1, figsize=(9, 8 if has_lr else 5.5),
                                sharex=True, height_ratios=[3, 1] if has_lr else None)
    loss_axis = axes[0] if has_lr else axes
    for label, rows in runs:
        key = "val_loss" if "val_loss" in rows[0] else "val_final_weighted_mse"
        epochs = [m["epoch"] for m in rows]
        line = loss_axis.plot(epochs, [m[key] for m in rows], label=f"{label} val", linewidth=2)[0]
        loss_axis.plot(epochs, [m["train_loss"] for m in rows], label=f"{label} train",
                       color=line.get_color(), linestyle="--", alpha=0.6)
        best = min(rows, key=lambda m: m[key])
        loss_axis.scatter([best["epoch"]], [best[key]], color=line.get_color(), zorder=5)
        loss_axis.annotate(f"ep{best['epoch']}: {best[key]:.3f}", (best["epoch"], best[key]),
                           textcoords="offset points", xytext=(6, -12), color=line.get_color(), fontsize=9)
    loss_axis.set_ylabel("loss")
    loss_axis.legend()
    loss_axis.grid(alpha=0.3)
    if has_lr:
        for label, rows in runs:
            if "lr" in rows[0]:
                axes[1].plot([m["epoch"] for m in rows], [m["lr"] for m in rows], label=label)
        axes[1].set_ylabel("lr")
        axes[1].set_xlabel("epoch")
        axes[1].grid(alpha=0.3)
    else:
        loss_axis.set_xlabel("epoch")
    figure.tight_layout()
    figure.savefig(args.out, dpi=150)
    print(args.out)


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def _add_model_args(p, tokenizer=True, required=True):
    p.add_argument("--dynamics", required=required)
    if tokenizer:
        p.add_argument("--tokenizer", required=required)
    p.add_argument("--token-file", required=required)
    p.add_argument("--dataset-root", help="normalize_{mean,std}.npz + lat.npy, and forcings for conditioned models")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--temperature", default="1.0", help="scalar or per-scale comma list")
    p.add_argument("--cadence-hours", type=int, default=6)
    p.add_argument("--channels", help="comma-separated frame channel names (default: STATE_CHANNELS)")


def _add_field_args(p, era5_defaults=False):
    d = (lambda v: v) if era5_defaults else (lambda v: None)
    p.add_argument("--temp-level", type=int, default=7)
    p.add_argument("--humidity-level", type=int, default=7)
    p.add_argument("--wind-level", type=int, default=3)
    p.add_argument("--vorticity-level", type=int, default=3)
    p.add_argument("--temp-channel", default=d("temperature"))
    p.add_argument("--humidity-channel", default=d("humidity"))
    p.add_argument("--u-channel", default=d("u_wind"))
    p.add_argument("--v-channel", default=d("v_wind"))
    p.add_argument("--vort-u-channel")
    p.add_argument("--vort-v-channel")


def _add_video_args(p):
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--stride", type=int, default=8, help="keep every Nth step")
    p.add_argument("--fps", type=int, default=25)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="analyze.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("mechanism", help="teacher-forced vs rollout entropy, P(stay), unseen co-occurrence")
    p.add_argument("--dynamics", required=True)
    p.add_argument("--token-root", required=True)
    p.add_argument("--held-out", default="val,test")
    p.add_argument("--train-pairs", type=int, default=2000)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--cadence-hours", type=int, default=6)
    p.add_argument("--dataset-root")
    p.add_argument("--device", default="auto")
    p.add_argument("--precision", default="auto")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_mechanism)

    p = sub.add_parser("oracle", help="closed-loop rollouts with coarse scales forced to truth")
    p.add_argument("--dynamics", required=True)
    p.add_argument("--tokenizer")
    p.add_argument("--token-root", required=True)
    p.add_argument("--splits", default="val,test")
    p.add_argument("--max-steps", type=int)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--channels", default="temperature,humidity,u_wind,v_wind,surface_pressure")
    p.add_argument("--u-channel", default="u_wind")
    p.add_argument("--v-channel", default="v_wind")
    p.add_argument("--low-m", type=int, default=5)
    p.add_argument("--cadence-hours", type=int, default=6)
    p.add_argument("--dataset-root")
    p.add_argument("--device", default="auto")
    p.add_argument("--precision", default="auto")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_oracle)

    p = sub.add_parser("chain", help="coarse-scale token Markov statistics, data vs rollouts")
    p.add_argument("--data-tokens", required=True)
    p.add_argument("--rollout-tokens", action="append", default=[], help="label=path to a rollout token h5")
    p.add_argument("--scale", type=int, default=0)
    p.add_argument("--scales", default="1x2,2x4,3x6,6x12,12x24")
    p.add_argument("--codebook-size", type=int, default=4096)
    p.add_argument("--out")
    p.set_defaults(func=cmd_chain)

    p = sub.add_parser("code-reuse", help="per-scale code reuse between train and held-out")
    p.add_argument("--token-root", required=True)
    p.add_argument("--scales", required=True)
    p.add_argument("--held-out", default="val,test")
    p.add_argument("--codebook-size", type=int, default=4096)
    p.add_argument("--out")
    p.set_defaults(func=cmd_code_reuse)

    p = sub.add_parser("calibrate", help="gust2-style marginal-entropy temperature solve")
    p.add_argument("--dynamics", required=True)
    p.add_argument("--token-file", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dataset-root")
    p.add_argument("--cadence-hours", type=int, default=6)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--temperatures", default="0.7,1.0,1.3,1.6,2.0,2.5")
    p.add_argument("--window-steps", type=int, default=0)
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("refs", help="ERA5 truth references (spectrum, histogram, panels, norm, lat)")
    p.add_argument("--config", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--output", required=True)
    p.add_argument("--panels-stride", type=int, default=4)
    p.add_argument("--panels-fields", default="temperature,humidity,u_wind,v_wind")
    p.add_argument("--batch", type=int, default=64)
    p.set_defaults(func=cmd_refs)

    p = sub.add_parser("animate-channel", help="video of one channel or relative vorticity")
    _add_model_args(p)
    _add_video_args(p)
    p.add_argument("--out", default="rollout.mp4")
    p.add_argument("--channel", default="Temperature_7", help="channel name or index")
    p.add_argument("--vorticity-level", type=int)
    p.add_argument("--vort-u-channel")
    p.add_argument("--vort-v-channel")
    p.add_argument("--cmap", default="turbo")
    p.set_defaults(func=cmd_animate_channel)

    p = sub.add_parser("animate-panels", help="2x2 video: vorticity, temperature, wind speed, humidity")
    _add_model_args(p)
    _add_video_args(p)
    _add_field_args(p)
    p.add_argument("--out", default="panels.mp4")
    p.add_argument("--frames-npz", help="precomputed frames (raw, collect) instead of a rollout")
    p.add_argument("--title-prefix", default="LUCIE emulator rollout")
    p.set_defaults(func=cmd_animate_panels)

    p = sub.add_parser("animate-compare", help="2x3 truth-vs-emulator video from the first frame")
    _add_model_args(p, required=False)
    _add_field_args(p)
    p.add_argument("--truth-npz", required=True, help="raw (T,C,H,W) + collect")
    p.add_argument("--pred-npz", help="decode-rollout output: render it instead of rolling the model out "
                   "(then --dynamics/--tokenizer/--token-file are not needed)")
    p.add_argument("--origin", choices=["upper", "lower"], default="lower",
                   help="row 0 at the bottom (lower, north up) or top (upper); both the ERA5 T30 "
                        "npz and the LUCIE-3D source are stored south-first")
    p.add_argument("--out", default="compare.mp4")
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--fps", type=int, default=10)
    p.set_defaults(func=cmd_animate_compare)

    p = sub.add_parser("decode-rollout", help="decode a stored rollout_tokens.h5 to frames (for animate-compare --pred-npz)")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--rollout-tokens", required=True, help="climate-diagnostics diag_*/rollout_tokens.h5")
    p.add_argument("--collect", default="0,1,2,3", help="channel indices to keep (ERA5: T, q, u, v)")
    p.add_argument("--stride", type=int, default=4, help="keep every Nth model call (match the truth panels)")
    p.add_argument("--batch", type=int, default=4, help="frames decoded per tokenizer call")
    p.add_argument("--save-every", type=int, default=40, help="rewrite the (resumable) output every N frames")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_decode_rollout)

    p = sub.add_parser("snapshots", help="still frames at chosen rollout steps")
    _add_model_args(p)
    _add_field_args(p, era5_defaults=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--label", default="model")
    p.add_argument("--steps", default="5,10,50,100,500,1000")
    p.add_argument("--truth-npz", help="frames (S,C,H,W) + steps (S,) of real frames")
    p.set_defaults(func=cmd_snapshots)

    p = sub.add_parser("curves", help="training curves from metrics.jsonl files")
    p.add_argument("out")
    p.add_argument("runs", nargs="+", help="LABEL=path/to/metrics.jsonl")
    p.set_defaults(func=cmd_curves)

    p = sub.add_parser("autopsy", help="codebook / weight-norm forensics on tokenizer checkpoints")
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--out")
    p.set_defaults(func=cmd_autopsy)
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
