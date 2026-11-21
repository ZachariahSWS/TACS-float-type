from __future__ import annotations

import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .config import config_hash
from .data import (
    BlockShuffleBatchSampler,
    FramePairDataset,
    SourceClimateFrameDataset,
    load_frame_dataset_manifest,
    load_frame_grid,
)
from .dynamics import NextScaleDynamics, dynamics_loss
from .noised_epochs import NoisedEpochPipeline
from .token_noise import TokenSubstitutionPipeline
from .runtime import (
    append_jsonl,
    atomic_json,
    atomic_torch_save,
    autocast_context,
    configure_torch,
    make_adamw,
    make_grad_scaler,
    make_lr_schedule,
    peak_memory_gb,
    resolve_device,
    resolve_precision,
    set_seed,
    software_manifest,
)
from .tokenizer import ClimateTokenizer, tokenizer_loss
from .conditioning import (
    ConditionedTokenPairDataset,
    ForcingConditioner,
    collate_conditioned_pairs,
)
from .schema import DEFAULT_SCALES
from .tokens import _move_state, TokenPairDataset, collate_token_pairs, load_tokenizer_checkpoint


def _loader_kwargs(runtime: dict[str, Any], device: torch.device) -> dict:
    workers = int(runtime.get("num_workers", 2))
    kwargs = {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = int(runtime.get("prefetch_factor", 2))
    return kwargs


def _profiler_context(
    runtime: dict[str, Any],
    output_dir: Path,
    device: torch.device,
    enabled: bool,
):
    active_steps = int(runtime.get("profile_active_steps", 0))
    if not enabled or active_steps <= 0:
        return nullcontext(None)
    trace_dir = output_dir / "profiler"
    trace_dir.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=2, warmup=2, active=active_steps),
        on_trace_ready=lambda profile: profile.export_chrome_trace(
            str(trace_dir / "trace.json")
        ),
        record_shapes=True,
        profile_memory=True,
    )


def _limit_dataset(dataset, maximum: int | None):
    if maximum is None or maximum <= 0 or len(dataset) <= maximum:
        return dataset
    return Subset(dataset, range(maximum))


def _limit_frames(frames, timestamps, maximum: int | None):
    if maximum is None or maximum <= 0 or frames.shape[0] <= maximum:
        return frames, timestamps
    return frames[: int(maximum)], timestamps[: int(maximum)]


def _spanning_subset(dataset, maximum: int | None):
    """Evenly spaced subset covering the whole dataset.

    Splits are chronological, so a contiguous prefix is a single season: the
    old validation_batches=16 evaluated 256 of 1459 val pairs, i.e. January
    to early March only. Spacing the subset across the split makes a limited
    evaluation representative of the full year instead of one season."""
    if maximum is None or maximum <= 0 or len(dataset) <= maximum:
        return dataset
    positions = np.linspace(0, len(dataset) - 1, maximum).round().astype(np.int64)
    return Subset(dataset, sorted({int(p) for p in positions}))


def _resume(
    latest_path: Path, enabled: bool, model, optimizer, scaler, lr_schedule
) -> tuple[int, float]:
    """Restore a run from latest.pt; returns (start_epoch, best_val).

    Older checkpoints carry no schedule state, so the schedule is fast-forwarded
    from the epoch count (the LR drops to where it would have been)."""
    if not (enabled and latest_path.exists()):
        return 0, float("inf")
    checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    start_epoch = int(checkpoint["epoch"]) + 1
    if "lr_schedule" in checkpoint:
        lr_schedule.load_state_dict(checkpoint["lr_schedule"])
    else:
        lr_schedule.fast_forward(start_epoch)
    return start_epoch, float(checkpoint["best_val"])


def _save_checkpoint(
    output_dir: Path, model, optimizer, scaler, lr_schedule, epoch: int,
    val: float, best_val: float, config: dict, extra: dict,
) -> tuple[dict, float]:
    """Write latest.pt (and best.pt on a new val minimum); returns
    (payload, best_val) so callers can add milestone copies."""
    payload = {
        "format_version": 1,
        "architecture": model.architecture(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "lr_schedule": lr_schedule.state_dict(),
        "epoch": epoch,
        "best_val": min(best_val, val),
        "config": config,
        **extra,
    }
    atomic_torch_save(output_dir / "latest.pt", payload)
    if val < best_val:
        best_val = val
        atomic_torch_save(output_dir / "best.pt", payload)
    return payload, best_val


def _optimizer_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    gradient_clip: float,
    lr_schedule,
) -> None:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    lr_schedule.apply()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


@torch.inference_mode()
def _evaluate_tokenizer(
    model: ClimateTokenizer,
    loader: DataLoader,
    device: torch.device,
    precision: str,
    latitude_weights: torch.Tensor,
    maximum_batches: int,
) -> float:
    model.eval()
    total = 0.0
    examples = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= maximum_batches:
            break
        state = _move_state(batch, device, precision)
        with autocast_context(device, precision):
            pyramid = model.encode(state, update_ema=False)
            reconstruction = model.decode_partials((pyramid.partial_latents[-1],))[0]
        error = (
            (reconstruction.float() - state.float()).square()
            * latitude_weights.view(1, 1, -1, 1)
        ).mean()
        total += float(error) * state.shape[0]
        examples += state.shape[0]
    return total / max(examples, 1)


def train_tokenizer(config: dict[str, Any]) -> Path:
    runtime = dict(config.get("runtime", {}))
    model_config = dict(config["tokenizer"])
    optimization = dict(config["optimization"])
    paths = dict(config["paths"])
    data_config = dict(config.get("data", {}))
    cadence_hours = int(data_config.get("cadence_hours", 6))
    device = resolve_device(runtime.get("device", "auto"))
    precision = resolve_precision(runtime.get("precision", "auto"), device)
    configure_torch(bool(runtime.get("performance_mode", True)))
    set_seed(int(runtime.get("seed", 0)))
    output_dir = Path(paths["tokenizer_run"])
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_size = int(optimization["batch_size"])
    seed = int(runtime.get("seed", 0))
    source_kind = str(data_config.get("source_kind", "hdf5"))
    if source_kind == "era5_npz":
        from .era5_npz import InVramFrameLoader, load_era5_npz

        npz_data = load_era5_npz(data_config, device, precision)
        dataset_manifest = npz_data.manifest
        latitude_weights = torch.from_numpy(npz_data.latitude_weights).to(device)
        train_frames, train_timestamps = _limit_frames(
            npz_data.frames["train"],
            npz_data.timestamps["train"],
            runtime.get("max_train_frames"),
        )
        val_frames, val_timestamps = _limit_frames(
            npz_data.frames["val"],
            npz_data.timestamps["val"],
            runtime.get("max_val_frames"),
        )
        train_loader = InVramFrameLoader(
            train_frames, train_timestamps, batch_size, shuffle=True, seed=seed
        )
        val_loader = InVramFrameLoader(
            val_frames, val_timestamps, batch_size, shuffle=False, drop_last=False
        )
        # assigned to `sampler` so the epoch loop's set_epoch call reshuffles it
        sampler = train_loader
    else:
        dataset_root = Path(paths["dataset_root"])
        dataset_manifest = load_frame_dataset_manifest(dataset_root)
        _, _, latitude_weight_values = load_frame_grid(dataset_root)
        latitude_weights = torch.from_numpy(latitude_weight_values).to(device)
        train_dataset = _limit_dataset(
            SourceClimateFrameDataset(dataset_root, "train", cadence_hours),
            runtime.get("max_train_frames"),
        )
        val_dataset = _limit_dataset(
            SourceClimateFrameDataset(dataset_root, "val", cadence_hours),
            runtime.get("max_val_frames"),
        )
        if isinstance(train_dataset, SourceClimateFrameDataset):
            sampler = BlockShuffleBatchSampler(
                len(train_dataset),
                batch_size,
                block_size=int(runtime.get("shuffle_block", 256)),
                seed=seed,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=sampler,
                **_loader_kwargs(runtime, device),
            )
        else:
            sampler = None
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                **_loader_kwargs(runtime, device),
            )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            **_loader_kwargs(runtime, device),
        )
    model = ClimateTokenizer(
        width=int(model_config["width"]),
        depth=int(model_config["depth"]),
        heads=int(model_config["heads"]),
        code_dim=int(model_config["code_dim"]),
        codebook_size=int(model_config["codebook_size"]),
        activation_checkpointing=bool(
            model_config.get("activation_checkpointing", False)
        ),
        state_channels=len(dataset_manifest["state_channels"]),
        grid_shape=tuple(dataset_manifest["grid_shape"]),
        ema_decay=float(model_config.get("ema_decay", 0.99)),
        norm_guard_factor=float(model_config.get("norm_guard_factor", 8.0)),
        dead_code_threshold=float(model_config.get("dead_code_threshold", 0.5)),
        dead_code_relative=bool(model_config.get("dead_code_relative", False)),
        dead_code_reinit=str(model_config.get("dead_code_reinit", "batch")),
        codebook_sizes=model_config.get("codebook_sizes"),
        phi_gain_limit=float(model_config.get("phi_gain_limit", 0.0)),
        ema_anchor=str(model_config.get("ema_anchor", "median")),
        scales=tuple(
            tuple(pair) for pair in model_config.get("scales", DEFAULT_SCALES)
        ),
    ).to(device)
    optimizer = make_adamw(
        model.parameters(),
        optimization["lr"],
        optimization["weight_decay"],
        device,
    )
    # warmup + cosine decay per optimizer step (gust2's recipe); a constant LR
    # made val turn back up after ~5-20 epochs once the model neared a minimum
    lr_schedule = make_lr_schedule(optimizer, optimization, len(train_loader))
    scaler = make_grad_scaler(device, precision)
    start_epoch, best_val = _resume(
        output_dir / "latest.pt", bool(runtime.get("resume", True)),
        model, optimizer, scaler, lr_schedule,
    )
    train_model = model
    if bool(runtime.get("compile", False)):
        train_model = torch.compile(model, mode=runtime.get("compile_mode", "default"))
    accumulation = int(optimization.get("gradient_accumulation", 1))
    gradient_clip = float(optimization.get("gradient_clip", 1.0))
    scale_batch_size = int(model_config.get("decode_scale_batch_size", 1))
    decoder_activation_checkpointing = bool(
        model_config.get("decoder_activation_checkpointing", False)
    )
    scale_weights = tuple(
        float(value)
        for value in model_config.get(
            "scale_loss_weights", [0.05, 0.10, 0.15, 0.25, 0.45]
        )
    )
    run_manifest = {
        "config": config,
        "config_hash": config_hash(config),
        "dataset_hash": dataset_manifest["dataset_hash"],
        "software": software_manifest(),
        "precision": precision,
        "device": str(device),
        "architecture": model.architecture(),
    }
    atomic_json(output_dir / "run_manifest.json", run_manifest)
    for epoch in range(start_epoch, int(optimization["epochs"])):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        start = time.perf_counter()
        total_loss = 0.0
        examples = 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with _profiler_context(
            runtime,
            output_dir,
            device,
            enabled=epoch == start_epoch,
        ) as step_profiler:
            final_window = len(train_loader) % accumulation
            for step, batch in enumerate(train_loader):
                state = _move_state(batch, device, precision)
                window_size = (
                    final_window
                    if final_window and step >= len(train_loader) - final_window
                    else accumulation
                )
                with autocast_context(device, precision):
                    output = train_model(
                        state,
                        scale_batch_size=scale_batch_size,
                        checkpoint_decoder=(
                            decoder_activation_checkpointing
                        ),
                    )
                    loss, _ = tokenizer_loss(
                        output,
                        state,
                        latitude_weights,
                        scale_weights=scale_weights,
                        commitment_weight=float(
                            model_config.get("commitment_weight", 0.25)
                        ),
                    )
                    scaled_loss = loss / window_size
                scaler.scale(scaled_loss).backward()
                if (step + 1) % accumulation == 0:
                    _optimizer_step(
                        model, optimizer, scaler, gradient_clip, lr_schedule
                    )
                total_loss += float(loss.detach()) * state.shape[0]
                examples += state.shape[0]
                if step_profiler is not None:
                    step_profiler.step()
        if len(train_loader) % accumulation:
            _optimizer_step(model, optimizer, scaler, gradient_clip, lr_schedule)
        # validation_batches <= 0 scores the whole split (matches the
        # dynamics trainer's convention); a positive value caps the batches
        validation_batches = int(runtime.get("validation_batches", 16))
        val = _evaluate_tokenizer(
            model,
            val_loader,
            device,
            precision,
            latitude_weights,
            validation_batches if validation_batches > 0 else len(val_loader),
        )
        elapsed = time.perf_counter() - start
        metrics = {
            "epoch": epoch,
            "train_loss": total_loss / max(examples, 1),
            "val_final_weighted_mse": val,
            "lr": lr_schedule.last_lr,
            "seconds": elapsed,
            "frames_per_second": examples / max(elapsed, 1e-9),
            "peak_memory_gb": peak_memory_gb(device),
        }
        append_jsonl(output_dir / "metrics.jsonl", metrics)
        _, best_val = _save_checkpoint(
            output_dir, model, optimizer, scaler, lr_schedule, epoch, val, best_val,
            config, {"dataset_hash": dataset_manifest["dataset_hash"]},
        )
        print(json.dumps(metrics, sort_keys=True), flush=True)
    return output_dir / "best.pt"


def _move_token_pyramid(
    pyramid: tuple[torch.Tensor, ...], device: torch.device
) -> tuple[torch.Tensor, ...]:
    return tuple(value.to(device, non_blocking=True) for value in pyramid)


def _unpack_dynamics_batch(batch, device: torch.device):
    """Accept both plain (current, target) and conditioned
    (current, target, conditioning) batches so the conditioned and
    unconditioned training paths share one loop."""
    if len(batch) == 3:
        current, target, conditioning = batch
        conditioning = conditioning.to(device, non_blocking=True)
    else:
        current, target = batch
        conditioning = None
    return (
        _move_token_pyramid(current, device),
        _move_token_pyramid(target, device),
        conditioning,
    )


@torch.inference_mode()
def _evaluate_dynamics(
    model: NextScaleDynamics,
    loader: DataLoader,
    device: torch.device,
    precision: str,
    maximum_batches: int,
    loss_options: dict[str, Any] | None = None,
) -> float:
    loss_options = loss_options or {}
    model.eval()
    total = 0.0
    examples = 0
    # top-1 teacher-forced accuracy per scale, alongside the CE. Cross-entropy
    # scores the whole predicted distribution, so it can worsen (over-confident
    # tails) while the argmax prediction keeps improving; tracking both makes
    # that divergence visible instead of hidden.
    correct = [0] * len(model.scales)
    counted = [0] * len(model.scales)
    for batch_index, batch in enumerate(loader):
        # <= 0 means evaluate everything the loader holds; the loader's
        # dataset is already the spanning subset when a cap was requested
        if 0 < maximum_batches <= batch_index:
            break
        current, target, conditioning = _unpack_dynamics_batch(batch, device)
        with autocast_context(device, precision):
            logits = model(current, target, conditioning)
            loss, _ = dynamics_loss(logits, target, **loss_options)
        batch = current[0].shape[0]
        total += float(loss) * batch
        examples += batch
        for index, (scale_logits, scale_target) in enumerate(zip(logits, target)):
            predicted = scale_logits.argmax(dim=-1).reshape(-1)
            truth = scale_target.reshape(-1)
            correct[index] += int((predicted == truth).sum())
            counted[index] += int(truth.numel())
    accuracy = {
        f"val_acc_{h}x{w}": correct[i] / max(counted[i], 1)
        for i, (h, w) in enumerate(model.scales)
    }
    accuracy["val_acc_mean"] = sum(accuracy.values()) / max(len(accuracy), 1)
    return total / max(examples, 1), accuracy


def train_dynamics(config: dict[str, Any]) -> Path:
    runtime = dict(config.get("runtime", {}))
    model_config = dict(config["dynamics"])
    optimization = dict(config["dynamics_optimization"])
    paths = dict(config["paths"])
    device = resolve_device(runtime.get("device", "auto"))
    precision = resolve_precision(runtime.get("precision", "auto"), device)
    configure_torch(bool(runtime.get("performance_mode", True)))
    set_seed(int(runtime.get("seed", 0)))
    tokenizer, tokenizer_checkpoint = load_tokenizer_checkpoint(
        paths["tokenizer_checkpoint"], device
    )
    token_root = Path(paths["token_root"])
    cadence_hours = int(config.get("data", {}).get("cadence_hours", 6))
    # Optional forcing conditioning (CO2 + solar/seasonal phase). When enabled
    # the datasets yield a per-pair conditioning vector and the model gains a
    # conditioning projection; when disabled this path is byte-for-byte the old
    # unconditioned behaviour, so existing chains are unaffected.
    conditioned = bool(model_config.get("conditioning", False))
    conditioner = (
        ForcingConditioner(
            paths["dataset_root"],
            include_co2=bool(model_config.get("conditioning_co2", True)),
        )
        if conditioned
        else None
    )
    conditioning_dim = conditioner.dimension if conditioned else 0
    token_collate = collate_conditioned_pairs if conditioned else collate_token_pairs

    # Prediction lead, decoupled from frame spacing. cadence_hours picks WHICH
    # frames exist; pair_lead_hours picks how far ahead the target sits. Hourly
    # tokens with lead 6 give 6x the pairs at the same 6-hour task, from every
    # start phase instead of only multiples of 6.
    pair_lead_hours = config.get("data", {}).get("pair_lead_hours")
    pair_lead_hours = int(pair_lead_hours) if pair_lead_hours else None

    def make_token_dataset(split: str):
        if conditioned:
            return ConditionedTokenPairDataset(
                token_root, split, conditioner, pair_lead_hours=pair_lead_hours
            )
        return TokenPairDataset(token_root, split, pair_lead_hours=pair_lead_hours)

    # Input noising (float-style): the SAME Gaussian noise is added to the x0/x1
    # frames of each pair and both are re-encoded with the frozen tokenizer, so
    # the NSP learns to predict from a perturbed input frame -- the
    # off-distribution condition it faces during rollout. Applied to TRAINING
    # only; validation always uses clean tokens. The noise-encode runs as a
    # batched per-epoch pre-pass into device-resident token buffers
    # (noised_epochs.py), overlapped with the previous epoch's training, rather
    # than inside each training step.
    noise_std = float(optimization.get("input_noise_std", 0.0))
    # Two mutually exclusive input-noise schemes. Float noise perturbs frames
    # and re-encodes them every epoch (needs the tokenizer and the frames);
    # token substitution resamples a fraction of positions from each
    # position's training vocabulary, which needs neither and is ~free.
    substitution_rate = float(optimization.get("token_substitution_rate", 0.0))
    # Self-generated corruption: replace a fraction of input frames with the
    # model's own one-step sample from the preceding frame (self_corrupt.py).
    self_corrupt_rate = float(optimization.get("self_corrupt_rate", 0.0))
    if sum(v > 0.0 for v in (noise_std, substitution_rate, self_corrupt_rate)) > 1:
        raise ValueError(
            "set only one of input_noise_std / token_substitution_rate / "
            "self_corrupt_rate"
        )
    substituted = substitution_rate > 0.0
    self_corrupted = self_corrupt_rate > 0.0
    noised = noise_std > 0.0 or substituted or self_corrupted
    val_dataset = _limit_dataset(
        make_token_dataset("val"), runtime.get("max_val_pairs")
    )
    # validation_batches <= 0 evaluates the whole split; a positive value now
    # samples evenly ACROSS the split rather than taking a seasonal prefix
    validation_batches = int(runtime.get("validation_batches", 16))
    if validation_batches > 0:
        val_dataset = _spanning_subset(
            val_dataset, validation_batches * int(optimization["batch_size"])
        )
    loader_kwargs = _loader_kwargs(runtime, device)
    if self_corrupted:
        from .self_corrupt import SelfCorruptPipeline

        train_loader = None
        pipeline = SelfCorruptPipeline(
            token_root,
            "train",
            tokenizer.scales,
            device,
            seed=int(runtime.get("seed", 0)),
            train_batch_size=int(optimization["batch_size"]),
            rate=self_corrupt_rate,
            temperature=float(optimization.get("self_corrupt_temperature", 1.0)),
            conditioner=conditioner,
            max_pairs=runtime.get("max_train_pairs"),
            autocast=lambda: autocast_context(device, precision),
        )
    elif substituted:
        train_loader = None
        pipeline = TokenSubstitutionPipeline(
            token_root,
            "train",
            tokenizer.scales,
            device,
            seed=int(runtime.get("seed", 0)),
            train_batch_size=int(optimization["batch_size"]),
            substitution_rate=substitution_rate,
            conditioner=conditioner,
            max_pairs=runtime.get("max_train_pairs"),
        )
    elif noised:
        train_loader = None
        pipeline = NoisedEpochPipeline(
            tokenizer,
            _limit_dataset(
                FramePairDataset(
                    paths["dataset_root"],
                    token_root,
                    "train",
                    cadence_hours,
                    conditioner,
                    pair_lead_hours=pair_lead_hours,
                ),
                runtime.get("max_train_pairs"),
            ),
            noise_std=noise_std,
            device=device,
            precision=precision,
            seed=int(runtime.get("seed", 0)),
            train_batch_size=int(optimization["batch_size"]),
            total_epochs=int(optimization["epochs"]),
            prepass_batch_size=int(optimization.get("noise_prepass_batch_size", 64)),
            loader_kwargs=loader_kwargs,
            # rolling 1/K noise refresh: re-encode only 1/K of pairs per epoch
            refresh_k=int(optimization.get("noise_refresh_k", 1)),
            # page-cache prewarm budget, hidden behind compile/model setup
            warm_cache_bytes=int(
                float(optimization.get("noise_warm_cache_gb", 0.0)) * (1 << 30)
            ),
            prepass_workers=optimization.get("noise_prepass_workers"),
        )
    else:
        pipeline = None
        train_loader = DataLoader(
            _limit_dataset(
                make_token_dataset("train"), runtime.get("max_train_pairs")
            ),
            batch_size=int(optimization["batch_size"]),
            shuffle=True,
            drop_last=True,
            collate_fn=token_collate,
            **loader_kwargs,
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(optimization["batch_size"]),
        shuffle=False,
        collate_fn=token_collate,
        **loader_kwargs,
    )
    scale_masks = torch.from_numpy(np.load(token_root / "scale_masks.npy"))
    # Optional tighter support: one allowed set per POSITION rather than per
    # scale, derived from the tokenized training split.
    position_masks = None
    if str(model_config.get("mask_granularity", "scale")) == "position":
        import h5py

        from .token_noise import position_vocabularies

        with h5py.File(token_root / "train" / "source.h5", "r") as handle:
            train_tokens = np.asarray(handle["tokens"], dtype=np.int64)
        table, sizes = position_vocabularies(train_tokens)
        position_masks = torch.zeros(
            table.shape[0], scale_masks.shape[1], dtype=torch.bool
        )
        for index in range(table.shape[0]):
            position_masks[index, torch.from_numpy(table[index, : sizes[index]])] = True
    model = NextScaleDynamics(
        tokenizer.quantizer.codebook,
        tokenizer.scales,
        width=int(model_config["width"]),
        depth=int(model_config["depth"]),
        heads=int(model_config["heads"]),
        activation_checkpointing=bool(
            model_config.get("activation_checkpointing", False)
        ),
        scale_masks=scale_masks,
        conditioning_dim=conditioning_dim,
        dropout=float(model_config.get("dropout", 0.0)),
        position_masks=position_masks,
        refine_layers=int(model_config.get("refine_layers", 0)),
    ).to(device)
    # per-scale CE weighting + label smoothing, shared by train and eval so
    # val_loss stays comparable to the training objective
    loss_options = {
        "weighting": str(optimization.get("scale_loss_weighting", "inv_sqrt")),
        "label_smoothing": float(optimization.get("label_smoothing", 0.0)),
    }
    if noise_std > 0.0:
        tokenizer.eval()  # frozen encoder used to re-tokenize noised frames
    else:
        # token substitution / self-corruption never touch the tokenizer, so
        # free it once the model has taken its codebook and scales
        del tokenizer
    optimizer = make_adamw(
        model.parameters(),
        optimization["lr"],
        optimization["weight_decay"],
        device,
    )
    steps_per_epoch = pipeline.batches_per_epoch if noised else len(train_loader)
    # warmup + cosine decay per optimizer step (gust2's recipe); a constant LR
    # made val turn back up after ~5-20 epochs once the model neared a minimum
    lr_schedule = make_lr_schedule(optimizer, optimization, steps_per_epoch)
    scaler = make_grad_scaler(device, precision)
    output_dir = Path(paths["dynamics_run"])
    output_dir.mkdir(parents=True, exist_ok=True)
    start_epoch, best_val = _resume(
        output_dir / "latest.pt", bool(runtime.get("resume", True)),
        model, optimizer, scaler, lr_schedule,
    )
    train_model = model
    if bool(runtime.get("compile", False)):
        train_model = torch.compile(model, mode=runtime.get("compile_mode", "default"))
    accumulation = int(optimization.get("gradient_accumulation", 1))
    gradient_clip = float(optimization.get("gradient_clip", 1.0))
    checkpoint_epochs = {int(e) for e in optimization.get("checkpoint_epochs", [])}
    atomic_json(
        output_dir / "run_manifest.json",
        {
            "config": config,
            "config_hash": config_hash(config),
            "software": software_manifest(),
            "precision": precision,
            "architecture": model.architecture(),
            "tokenizer_dataset_hash": tokenizer_checkpoint.get("dataset_hash"),
        },
    )
    for epoch in range(start_epoch, int(optimization["epochs"])):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        start = time.perf_counter()
        total = 0.0
        examples = 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if self_corrupted:
            batches = pipeline.epoch_batches(epoch, model)
        elif noised:
            batches = pipeline.epoch_batches(epoch)
        else:
            batches = train_loader
        final_window = steps_per_epoch % accumulation
        for step, batch in enumerate(batches):
            # the substitution pipeline additionally yields a NOISED copy of
            # the t1 pyramid for the teacher-forced input path; the loss still
            # scores against the clean `target`
            target_context = None
            if substituted:
                current, target, conditioning, target_context = batch
            elif noised:
                current, target, conditioning = batch
            else:
                current, target, conditioning = _unpack_dynamics_batch(batch, device)
            window_size = (
                final_window
                if final_window and step >= steps_per_epoch - final_window
                else accumulation
            )
            with autocast_context(device, precision):
                logits = train_model(
                    current, target, conditioning, target_context=target_context
                )
                loss, _ = dynamics_loss(logits, target, **loss_options)
                scaled_loss = loss / window_size
            scaler.scale(scaled_loss).backward()
            if (step + 1) % accumulation == 0:
                _optimizer_step(model, optimizer, scaler, gradient_clip, lr_schedule)
            batch = current[0].shape[0]
            total += float(loss.detach()) * batch
            examples += batch
        if steps_per_epoch % accumulation:
            _optimizer_step(model, optimizer, scaler, gradient_clip, lr_schedule)
        val, val_accuracy = _evaluate_dynamics(
            model,
            val_loader,
            device,
            precision,
            int(runtime.get("validation_batches", 16)),
            loss_options,
        )
        elapsed = time.perf_counter() - start
        metrics = {
            "epoch": epoch,
            "train_loss": total / max(examples, 1),
            "val_loss": val,
            "lr": lr_schedule.last_lr,
            "seconds": elapsed,
            "pairs_per_second": examples / max(elapsed, 1e-9),
            "peak_memory_gb": peak_memory_gb(device),
            **val_accuracy,
        }
        append_jsonl(output_dir / "metrics.jsonl", metrics)
        payload, best_val = _save_checkpoint(
            output_dir, model, optimizer, scaler, lr_schedule, epoch, val, best_val,
            config, {"tokenizer_checkpoint": str(Path(paths["tokenizer_checkpoint"]).resolve())},
        )
        # Milestone checkpoints, named by EPOCH COUNT COMPLETED (1-based), so
        # `checkpoint_epochs = [15, 20, 25]` writes epoch15/20/25.pt. Needed
        # because best.pt is chosen by val CE, which this project has measured
        # selecting far too early: rollout quality keeps improving past the CE
        # minimum, so the checkpoints worth comparing must be kept explicitly
        # rather than recovered from a latest.pt that keeps being overwritten.
        if (epoch + 1) in checkpoint_epochs:
            atomic_torch_save(output_dir / f"epoch{epoch + 1}.pt", payload)
        print(json.dumps(metrics, sort_keys=True), flush=True)
    if pipeline is not None:
        pipeline.close()
    return output_dir / "best.pt"
