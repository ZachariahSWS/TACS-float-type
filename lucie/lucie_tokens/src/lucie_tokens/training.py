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
from .runtime import (
    append_jsonl,
    atomic_json,
    atomic_torch_save,
    autocast_context,
    configure_torch,
    make_adamw,
    make_grad_scaler,
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
from .tokens import TokenPairDataset, collate_token_pairs, load_tokenizer_checkpoint


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


def _move_state(
    batch,
    device: torch.device,
    precision: str,
) -> torch.Tensor:
    state = batch[0] if isinstance(batch, (tuple, list)) else batch
    if device.type == "cuda" and precision != "fp32":
        return state.to(device=device, non_blocking=True)
    return state.to(device=device, dtype=torch.float32, non_blocking=True)


def _optimizer_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    gradient_clip: float,
) -> None:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
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
    dataset_root = Path(paths["dataset_root"])
    output_dir = Path(paths["tokenizer_run"])
    output_dir.mkdir(parents=True, exist_ok=True)
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
    batch_size = int(optimization["batch_size"])
    if isinstance(train_dataset, SourceClimateFrameDataset):
        sampler = BlockShuffleBatchSampler(
            len(train_dataset),
            batch_size,
            block_size=int(runtime.get("shuffle_block", 256)),
            seed=int(runtime.get("seed", 0)),
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
    scaler = make_grad_scaler(device, precision)
    start_epoch = 0
    best_val = float("inf")
    latest_path = output_dir / "latest.pt"
    if bool(runtime.get("resume", True)) and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint["best_val"])
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
                    _optimizer_step(model, optimizer, scaler, gradient_clip)
                total_loss += float(loss.detach()) * state.shape[0]
                examples += state.shape[0]
                if step_profiler is not None:
                    step_profiler.step()
        if len(train_loader) % accumulation:
            _optimizer_step(model, optimizer, scaler, gradient_clip)
        val = _evaluate_tokenizer(
            model,
            val_loader,
            device,
            precision,
            latitude_weights,
            int(runtime.get("validation_batches", 16)),
        )
        elapsed = time.perf_counter() - start
        metrics = {
            "epoch": epoch,
            "train_loss": total_loss / max(examples, 1),
            "val_final_weighted_mse": val,
            "seconds": elapsed,
            "frames_per_second": examples / max(elapsed, 1e-9),
            "peak_memory_gb": peak_memory_gb(device),
        }
        append_jsonl(output_dir / "metrics.jsonl", metrics)
        payload = {
            "format_version": 1,
            "architecture": model.architecture(),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_val": min(best_val, val),
            "dataset_hash": dataset_manifest["dataset_hash"],
            "config": config,
        }
        atomic_torch_save(latest_path, payload)
        if val < best_val:
            best_val = val
            atomic_torch_save(output_dir / "best.pt", payload)
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
) -> float:
    model.eval()
    total = 0.0
    examples = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= maximum_batches:
            break
        current, target, conditioning = _unpack_dynamics_batch(batch, device)
        with autocast_context(device, precision):
            logits = model(current, target, conditioning)
            loss, _ = dynamics_loss(logits, target)
        batch = current[0].shape[0]
        total += float(loss) * batch
        examples += batch
    return total / max(examples, 1)


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
    conditioner = ForcingConditioner(paths["dataset_root"]) if conditioned else None
    conditioning_dim = conditioner.dimension if conditioned else 0
    token_collate = collate_conditioned_pairs if conditioned else collate_token_pairs

    def make_token_dataset(split: str):
        if conditioned:
            return ConditionedTokenPairDataset(token_root, split, conditioner)
        return TokenPairDataset(token_root, split)

    # Input noising (float-style): the SAME Gaussian noise is added to the x0/x1
    # frames of each pair and both are re-encoded with the frozen tokenizer, so
    # the NSP learns to predict from a perturbed input frame -- the
    # off-distribution condition it faces during rollout. Applied to TRAINING
    # only; validation always uses clean tokens. The noise-encode runs as a
    # batched per-epoch pre-pass into device-resident token buffers
    # (noised_epochs.py), overlapped with the previous epoch's training, rather
    # than inside each training step.
    noise_std = float(optimization.get("input_noise_std", 0.0))
    noised = noise_std > 0.0
    val_dataset = _limit_dataset(
        make_token_dataset("val"), runtime.get("max_val_pairs")
    )
    loader_kwargs = _loader_kwargs(runtime, device)
    if noised:
        train_loader = None
        pipeline = NoisedEpochPipeline(
            tokenizer,
            _limit_dataset(
                FramePairDataset(
                    paths["dataset_root"], token_root, "train", cadence_hours, conditioner
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
    ).to(device)
    if noised:
        tokenizer.eval()  # frozen encoder used to re-tokenize noised frames
    else:
        del tokenizer
    optimizer = make_adamw(
        model.parameters(),
        optimization["lr"],
        optimization["weight_decay"],
        device,
    )
    scaler = make_grad_scaler(device, precision)
    output_dir = Path(paths["dynamics_run"])
    output_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    best_val = float("inf")
    latest_path = output_dir / "latest.pt"
    if bool(runtime.get("resume", True)) and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint["best_val"])
    train_model = model
    if bool(runtime.get("compile", False)):
        train_model = torch.compile(model, mode=runtime.get("compile_mode", "default"))
    accumulation = int(optimization.get("gradient_accumulation", 1))
    gradient_clip = float(optimization.get("gradient_clip", 1.0))
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
        steps_per_epoch = pipeline.batches_per_epoch if noised else len(train_loader)
        batches = pipeline.epoch_batches(epoch) if noised else train_loader
        final_window = steps_per_epoch % accumulation
        for step, batch in enumerate(batches):
            if noised:
                current, target, conditioning = batch
            else:
                current, target, conditioning = _unpack_dynamics_batch(batch, device)
            window_size = (
                final_window
                if final_window and step >= steps_per_epoch - final_window
                else accumulation
            )
            with autocast_context(device, precision):
                logits = train_model(current, target, conditioning)
                loss, _ = dynamics_loss(logits, target)
                scaled_loss = loss / window_size
            scaler.scale(scaled_loss).backward()
            if (step + 1) % accumulation == 0:
                _optimizer_step(model, optimizer, scaler, gradient_clip)
            batch = current[0].shape[0]
            total += float(loss.detach()) * batch
            examples += batch
        if steps_per_epoch % accumulation:
            _optimizer_step(model, optimizer, scaler, gradient_clip)
        val = _evaluate_dynamics(
            model,
            val_loader,
            device,
            precision,
            int(runtime.get("validation_batches", 16)),
        )
        elapsed = time.perf_counter() - start
        metrics = {
            "epoch": epoch,
            "train_loss": total / max(examples, 1),
            "val_loss": val,
            "seconds": elapsed,
            "pairs_per_second": examples / max(elapsed, 1e-9),
            "peak_memory_gb": peak_memory_gb(device),
        }
        append_jsonl(output_dir / "metrics.jsonl", metrics)
        payload = {
            "format_version": 1,
            "architecture": model.architecture(),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_val": min(best_val, val),
            "config": config,
            "tokenizer_checkpoint": str(Path(paths["tokenizer_checkpoint"]).resolve()),
        }
        atomic_torch_save(latest_path, payload)
        if val < best_val:
            best_val = val
            atomic_torch_save(output_dir / "best.pt", payload)
        print(json.dumps(metrics, sort_keys=True), flush=True)
    if pipeline is not None:
        pipeline.close()
    return output_dir / "best.pt"
