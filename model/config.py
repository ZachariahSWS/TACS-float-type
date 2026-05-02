from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional
import tomllib


def _optional_float(value: object) -> Optional[float]:
    return None if value is None else float(value)


@dataclass(frozen=True)
class DataConfig:
    data_path: str
    field: str
    train_start: int
    train_stop: int
    val_start: int
    val_stop: int


@dataclass(frozen=True)
class ExperimentRunConfig:
    experiment: str
    output_dir: str
    seed: int = 0
    resume: bool = False


@dataclass(frozen=True)
class OptimizationConfig:
    batch_size: int = 8
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip_norm: Optional[float] = 1.0


@dataclass(frozen=True)
class LoaderConfig:
    num_workers: int = 4
    prefetch_factor: int = 4
    persistent_workers: bool = True
    pin_memory: bool = True


@dataclass(frozen=True)
class NoiseConfig:
    stage1_noise_scale: float = 0.0
    stage2_noise_scale: float = 0.0
    stage3_noise_scale: float = 0.0
    initial_noise_mse: Optional[float] = None
    mask_seed: int = 0


@dataclass(frozen=True)
class TrainConfig:
    data: DataConfig
    run: ExperimentRunConfig
    optimization: OptimizationConfig
    loader: LoaderConfig
    noise: NoiseConfig

    @classmethod
    def from_toml(cls, path: str | Path) -> "TrainConfig":
        with open(path, "rb") as f:
            raw = tomllib.load(f)

        data_raw = raw["data"]
        run_raw = raw["run"]
        optimization_raw = raw.get("optimization", {})
        loader_raw = raw.get("loader", {})
        noise_raw = raw.get("noise", {})

        return cls(
            data=DataConfig(
                data_path=str(data_raw["data_path"]),
                field=str(data_raw["field"]),
                train_start=int(data_raw["train_start"]),
                train_stop=int(data_raw["train_stop"]),
                val_start=int(data_raw["val_start"]),
                val_stop=int(data_raw["val_stop"]),
            ),
            run=ExperimentRunConfig(
                experiment=str(run_raw["experiment"]),
                output_dir=str(run_raw["output_dir"]),
                seed=int(run_raw.get("seed", 0)),
                resume=bool(run_raw.get("resume", False)),
            ),
            optimization=OptimizationConfig(
                batch_size=int(optimization_raw.get("batch_size", 8)),
                epochs=int(optimization_raw.get("epochs", 50)),
                lr=float(optimization_raw.get("lr", 1e-4)),
                weight_decay=float(optimization_raw.get("weight_decay", 1e-4)),
                grad_clip_norm=_optional_float(optimization_raw.get("grad_clip_norm", 1.0)),
            ),
            loader=LoaderConfig(
                num_workers=int(loader_raw.get("num_workers", 4)),
                prefetch_factor=int(loader_raw.get("prefetch_factor", 4)),
                persistent_workers=bool(loader_raw.get("persistent_workers", True)),
                pin_memory=bool(loader_raw.get("pin_memory", True)),
            ),
            noise=NoiseConfig(
                stage1_noise_scale=float(noise_raw.get("stage1_noise_scale", 0.0)),
                stage2_noise_scale=float(noise_raw.get("stage2_noise_scale", 0.0)),
                stage3_noise_scale=float(noise_raw.get("stage3_noise_scale", 0.0)),
                initial_noise_mse=_optional_float(noise_raw.get("initial_noise_mse")),
                mask_seed=int(noise_raw.get("mask_seed", 0)),
            ),
        )

    def serializable(self) -> dict:
        return asdict(self)
