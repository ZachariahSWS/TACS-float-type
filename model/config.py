from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional
import tomllib


CONFIG_VERSION = 3
VALID_PRECISIONS = {"fp32", "bf16"}


def _optional_float(value: object) -> Optional[float]:
    return None if value is None else float(value)


def _grad_clip_norm(value: object) -> Optional[float]:
    if value is None:
        return None
    out = float(value)
    return None if out < 0.0 else out


def _optional_int(raw: dict, key: str) -> Optional[int]:
    return None if key not in raw else int(raw[key])


def _optional_str(raw: dict, key: str) -> Optional[str]:
    return None if key not in raw else str(raw[key])


@dataclass(frozen=True)
class DataConfig:
    data_path: str
    field: str
    train_start: int
    train_stop: int
    val_start: int
    val_stop: int


@dataclass(frozen=True)
class RunConfig:
    output_dir: str
    seed: int = 0
    resume: bool = False


@dataclass(frozen=True)
class ModelDefaultsConfig:
    image_size: int = 256
    patch_size: int = 8
    width: int = 128
    n_heads: int = 8
    mlp_dim: int = 512
    cleanup_hidden_channels: int = 32
    precision: str = "bf16"


@dataclass(frozen=True)
class StageConfig:
    stage: int
    in_channels: int
    out_channels: int
    image_size: int
    patch_size: int
    width: int
    depth: int
    n_heads: int
    mlp_dim: int
    cleanup_hidden_channels: int
    precision: str


@dataclass(frozen=True)
class OptimizationConfig:
    batch_size: int = 8
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip_norm: Optional[float] = None


@dataclass(frozen=True)
class LoaderConfig:
    num_workers: int = 4
    prefetch_factor: int = 4
    persistent_workers: bool = True
    pin_memory: bool = True


@dataclass(frozen=True)
class NoiseConfig:
    noise_scale: float = 0.0
    initial_noise_mse: Optional[float] = None
    mask_seed: int = 0


@dataclass(frozen=True)
class TrainConfig:
    config_version: int
    data: DataConfig
    run: RunConfig
    model: ModelDefaultsConfig
    stages: tuple[StageConfig, ...]
    optimization: OptimizationConfig
    loader: LoaderConfig
    noise: NoiseConfig

    @classmethod
    def from_toml(cls, path: str | Path) -> "TrainConfig":
        with open(path, "rb") as f:
            raw = tomllib.load(f)

        version = int(raw.get("config_version", 0))
        if version != CONFIG_VERSION:
            raise ValueError(f"Expected config_version = {CONFIG_VERSION}, got {version}")

        data_raw = raw["data"]
        run_raw = raw["run"]
        model_raw = raw.get("model", {})
        optimization_raw = raw.get("optimization", {})
        loader_raw = raw.get("loader", {})
        noise_raw = raw.get("noise", {})

        model = ModelDefaultsConfig(
            image_size=int(model_raw.get("image_size", 256)),
            patch_size=int(model_raw.get("patch_size", 8)),
            width=int(model_raw.get("width", 128)),
            n_heads=int(model_raw.get("n_heads", 8)),
            mlp_dim=int(model_raw.get("mlp_dim", 512)),
            cleanup_hidden_channels=int(model_raw.get("cleanup_hidden_channels", 32)),
            precision=str(model_raw.get("precision", "bf16")),
        )
        _validate_precision(model.precision, "[model].precision")

        return cls(
            config_version=version,
            data=DataConfig(
                data_path=str(data_raw["data_path"]),
                field=str(data_raw["field"]),
                train_start=int(data_raw["train_start"]),
                train_stop=int(data_raw["train_stop"]),
                val_start=int(data_raw["val_start"]),
                val_stop=int(data_raw["val_stop"]),
            ),
            run=RunConfig(
                output_dir=str(run_raw["output_dir"]),
                seed=int(run_raw.get("seed", 0)),
                resume=bool(run_raw.get("resume", False)),
            ),
            model=model,
            stages=_parse_stages(raw, model),
            optimization=OptimizationConfig(
                batch_size=int(optimization_raw.get("batch_size", 8)),
                epochs=int(optimization_raw.get("epochs", 50)),
                lr=float(optimization_raw.get("lr", 1e-4)),
                weight_decay=float(optimization_raw.get("weight_decay", 1e-4)),
                grad_clip_norm=_grad_clip_norm(optimization_raw.get("grad_clip_norm", -1.0)),
            ),
            loader=LoaderConfig(
                num_workers=int(loader_raw.get("num_workers", 4)),
                prefetch_factor=int(loader_raw.get("prefetch_factor", 4)),
                persistent_workers=bool(loader_raw.get("persistent_workers", True)),
                pin_memory=bool(loader_raw.get("pin_memory", True)),
            ),
            noise=NoiseConfig(
                noise_scale=float(noise_raw.get("noise_scale", 0.0)),
                initial_noise_mse=_optional_float(noise_raw.get("initial_noise_mse")),
                mask_seed=int(noise_raw.get("mask_seed", 0)),
            ),
        )

    def serializable(self) -> dict:
        return asdict(self)


def _parse_stages(raw: dict, model: ModelDefaultsConfig) -> tuple[StageConfig, ...]:
    stage_raw = raw.get("stage")
    if not isinstance(stage_raw, dict) or "1" not in stage_raw:
        raise ValueError("Config must define at least [stage.1].")

    stage_ids = sorted(int(key) for key in stage_raw.keys())
    expected = list(range(1, len(stage_ids) + 1))
    if stage_ids != expected:
        raise ValueError(f"Stages must be sequential starting at 1; got {stage_ids}")

    stages = []
    for stage in stage_ids:
        item = stage_raw[str(stage)]
        if "depth" not in item:
            raise ValueError(f"[stage.{stage}] must define depth")

        precision = _optional_str(item, "precision") or model.precision
        _validate_precision(precision, f"[stage.{stage}].precision")

        stages.append(
            StageConfig(
                stage=stage,
                in_channels=stage,
                out_channels=1,
                image_size=_optional_int(item, "image_size") or model.image_size,
                patch_size=_optional_int(item, "patch_size") or model.patch_size,
                width=_optional_int(item, "width") or model.width,
                depth=int(item["depth"]),
                n_heads=_optional_int(item, "n_heads") or model.n_heads,
                mlp_dim=_optional_int(item, "mlp_dim") or model.mlp_dim,
                cleanup_hidden_channels=_optional_int(item, "cleanup_hidden_channels") or model.cleanup_hidden_channels,
                precision=precision,
            )
        )

    return tuple(stages)


def _validate_precision(precision: str, where: str) -> None:
    if precision not in VALID_PRECISIONS:
        valid = ", ".join(sorted(VALID_PRECISIONS))
        raise ValueError(f"{where} must be one of {{{valid}}}, got {precision!r}")
