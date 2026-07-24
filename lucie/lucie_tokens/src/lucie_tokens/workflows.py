from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from .dynamics import NextScaleDynamics, sample_next
from .runtime import autocast_context, resolve_device, resolve_precision
from .schema import split_flat_tokens
from .tokens import load_tokenizer_checkpoint


def load_dynamics_checkpoint(
    checkpoint_path: str | Path, device: torch.device
) -> tuple[NextScaleDynamics, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    architecture = checkpoint["architecture"]
    state = checkpoint["model"]
    model = NextScaleDynamics(
        state["codebook"],
        scales=tuple(tuple(value) for value in architecture["scales"]),
        width=int(architecture["width"]),
        depth=int(architecture["depth"]),
        heads=int(architecture["heads"]),
        activation_checkpointing=False,
        scale_masks=state["scale_masks"],
    )
    model.load_state_dict(state)
    model.to(device).eval()
    return model, checkpoint


@torch.inference_mode()
def rollout(
    dynamics_checkpoint: str | Path,
    token_file: str | Path,
    output_path: str | Path,
    start_index: int = 0,
    steps: int = 28,
    temperature: float = 0.0,
    top_k: int | None = None,
    precision: str = "auto",
    device_name: str = "auto",
) -> Path:
    device = resolve_device(device_name)
    precision = resolve_precision(precision, device)
    dynamics, dynamics_payload = load_dynamics_checkpoint(
        dynamics_checkpoint, device
    )
    tokenizer_path = dynamics_payload["tokenizer_checkpoint"]
    tokenizer, _ = load_tokenizer_checkpoint(tokenizer_path, device)
    with h5py.File(token_file, "r") as source:
        all_tokens = np.asarray(source["tokens"], dtype=np.int64)
        timestamps = np.asarray(source["timestamp"], dtype=np.int64)
    if start_index < 0 or start_index + steps >= len(all_tokens):
        raise IndexError("Requested rollout extends beyond the token file")
    current_flat = torch.from_numpy(all_tokens[start_index]).unsqueeze(0).to(device)
    current = split_flat_tokens(current_flat, dynamics.scales)
    prediction_frames = []
    generated_tokens = []
    for _ in range(steps):
        with autocast_context(device, precision):
            current = sample_next(
                dynamics,
                current,
                temperature=temperature,
                top_k=top_k,
            )
            decoded = tokenizer.decode_indices(current, final_only=True)[0]
        prediction_frames.append(decoded[0].float().cpu().numpy())
        generated_tokens.append(
            torch.cat([value.flatten(1) for value in current], dim=1)[0]
            .cpu()
            .numpy()
            .astype(np.uint16)
        )
    truth_flat = torch.from_numpy(
        all_tokens[start_index + 1 : start_index + steps + 1]
    ).to(device)
    truth_frames = []
    decode_batch = 16
    for start in range(0, len(truth_flat), decode_batch):
        batch_tokens = split_flat_tokens(
            truth_flat[start : start + decode_batch], dynamics.scales
        )
        with autocast_context(device, precision):
            decoded = tokenizer.decode_indices(batch_tokens, final_only=True)[0]
        truth_frames.append(decoded.float().cpu().numpy())
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as target:
        target.create_dataset(
            "prediction_norm", data=np.stack(prediction_frames), dtype="f4"
        )
        target.create_dataset(
            "truth_norm", data=np.concatenate(truth_frames), dtype="f4"
        )
        target.create_dataset(
            "generated_tokens", data=np.stack(generated_tokens), dtype="u2"
        )
        target.create_dataset(
            "timestamp", data=timestamps[start_index + 1 : start_index + steps + 1]
        )
        target.attrs["dynamics_checkpoint"] = str(
            Path(dynamics_checkpoint).resolve()
        )
        target.attrs["token_file"] = str(Path(token_file).resolve())
        target.attrs["temperature"] = float(temperature)
        target.attrs["top_k"] = -1 if top_k is None else int(top_k)
    return output_path
