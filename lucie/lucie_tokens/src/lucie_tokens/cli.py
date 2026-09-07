from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .config import load_config


def _split_values(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lucie-tokens")
    commands = parser.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser("inspect-source")
    inspect.add_argument("source")
    inspect.add_argument("--splits", default="train,val,test")
    inspect.add_argument("--samples", type=int, default=16)
    inspect.add_argument("--output")

    train_tok = commands.add_parser("train-tokenizer")
    train_tok.add_argument("config")

    tokenize = commands.add_parser("tokenize")
    tokenize.add_argument("config")
    tokenize.add_argument("--splits", default="train,val,test")
    tokenize.add_argument("--overwrite", action="store_true")

    evaluate_tok = commands.add_parser("evaluate-tokenizer")
    evaluate_tok.add_argument("config")
    evaluate_tok.add_argument("--split", default="val")
    evaluate_tok.add_argument("--output")

    train_dyn = commands.add_parser("train-dynamics")
    train_dyn.add_argument("config")

    rollout_parser = commands.add_parser("rollout")
    rollout_parser.add_argument("dynamics_checkpoint")
    rollout_parser.add_argument("token_file")
    rollout_parser.add_argument("output")
    rollout_parser.add_argument("--start", type=int, default=0)
    rollout_parser.add_argument("--steps", type=int, default=28)
    rollout_parser.add_argument("--temperature", type=float, default=0.0)
    rollout_parser.add_argument("--top-k", type=int)

    eval_rollout = commands.add_parser("evaluate-rollout")
    eval_rollout.add_argument("rollout")
    eval_rollout.add_argument("output")

    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("config")
    benchmark.add_argument("--iterations", type=int, default=5)
    benchmark.add_argument("--output")

    diagnostics = commands.add_parser("climate-diagnostics")
    diagnostics.add_argument("config")
    diagnostics.add_argument("--split", default="val")
    diagnostics.add_argument("--steps", type=int, default=2000)
    diagnostics.add_argument("--start", type=int, default=0)
    diagnostics.add_argument(
        "--temperature",
        default="1.0",
        help="sampling temperature; a comma list gives one per scale",
    )
    diagnostics.add_argument("--top-k", type=int)
    diagnostics.add_argument(
        "--record-every",
        type=int,
        default=1,
        help="model steps per recorded frame; use 6 to score a 1-hour model "
        "at the 6-hourly cadence of the references",
    )
    diagnostics.add_argument("--max-reference-frames", type=int)
    diagnostics.add_argument("--checkpoint", help="override paths.dynamics_checkpoint")
    diagnostics.add_argument(
        "--data-spectrum", help="npz of precomputed truth spectrum (skips source read)"
    )
    diagnostics.add_argument(
        "--data-histograms",
        help="npz of precomputed truth pixel histograms (see precompute_truth_hist.py)",
    )
    diagnostics.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "inspect-source":
        from .data import inspect_source
        from .runtime import atomic_json

        result = inspect_source(
            args.source, _split_values(args.splits), args.samples
        )
        if args.output:
            atomic_json(args.output, result)
    elif args.command == "train-tokenizer":
        from .training import train_tokenizer

        result = {"checkpoint": str(train_tokenizer(load_config(args.config)))}
    elif args.command == "tokenize":
        from .tokens import tokenize_dataset

        config = load_config(args.config)
        result = tokenize_dataset(
            config["paths"]["dataset_root"],
            config["paths"]["tokenizer_checkpoint"],
            config["paths"]["token_root"],
            _split_values(args.splits),
            batch_size=int(config.get("tokenization", {}).get("batch_size", 32)),
            num_workers=int(config.get("runtime", {}).get("num_workers", 2)),
            precision=config.get("runtime", {}).get("precision", "auto"),
            device_name=config.get("runtime", {}).get("device", "auto"),
            overwrite=args.overwrite,
            cadence_hours=int(config.get("data", {}).get("cadence_hours", 6)),
            data_config=config.get("data", {}),
        )
    elif args.command == "evaluate-tokenizer":
        from .analysis import evaluate_tokenizer

        config = load_config(args.config)
        output = args.output or str(
            Path(config["paths"]["tokenizer_run"]) / "evaluation.json"
        )
        result = evaluate_tokenizer(
            config["paths"]["dataset_root"],
            config["paths"]["tokenizer_checkpoint"],
            output,
            split=args.split,
            batch_size=int(config.get("evaluation", {}).get("batch_size", 8)),
            maximum_batches=int(
                config.get("evaluation", {}).get("maximum_batches", 32)
            ),
            precision=config.get("runtime", {}).get("precision", "auto"),
            device_name=config.get("runtime", {}).get("device", "auto"),
            cadence_hours=int(config.get("data", {}).get("cadence_hours", 6)),
        )
    elif args.command == "train-dynamics":
        from .training import train_dynamics

        result = {"checkpoint": str(train_dynamics(load_config(args.config)))}
    elif args.command == "rollout":
        from .workflows import rollout

        result = {
            "output": str(
                rollout(
                    args.dynamics_checkpoint,
                    args.token_file,
                    args.output,
                    args.start,
                    args.steps,
                    args.temperature,
                    args.top_k,
                )
            )
        }
    elif args.command == "evaluate-rollout":
        from .analysis import evaluate_rollout_file

        result = evaluate_rollout_file(args.rollout, args.output)
    elif args.command == "benchmark":
        from .benchmark import benchmark_tokenizer
        from .runtime import atomic_json

        result = benchmark_tokenizer(
            load_config(args.config), args.iterations
        )
        if args.output:
            atomic_json(args.output, result)
    elif args.command == "climate-diagnostics":
        from .diagnostics import climate_diagnostics

        config = load_config(args.config)
        temperature_values = [float(v) for v in str(args.temperature).split(",")]
        temperature = (
            temperature_values[0]
            if len(temperature_values) == 1
            else tuple(temperature_values)
        )
        token_file = str(
            Path(config["paths"]["token_root"]) / args.split / "source.h5"
        )
        output = args.output or str(
            Path(config["paths"]["dynamics_run"]) / f"diagnostics_{args.split}"
        )
        # The 33-channel Bridges source infers its grid/channel layout from the
        # dataset root + STATE_CHANNELS. The in-VRAM ERA5 T30 set has a different
        # channel list on a legendre-gauss grid and no per-frame source tree, so
        # its layout is derived here from the config and passed in explicitly.
        channel_names = latitude_weights = wind_channels = None
        data_config = config.get("data", {})
        if data_config.get("source_kind") == "era5_npz":
            from .era5_npz import PROGNOSTIC_CHANNELS, legendre_gauss_area_weights

            channel_names = list(data_config.get("channels", PROGNOSTIC_CHANNELS))
            grid_shape = data_config.get("grid_shape", (48, 96))
            latitude_weights = legendre_gauss_area_weights(int(grid_shape[0]))
            wind_channels = (
                slice(channel_names.index("u_wind"), channel_names.index("u_wind") + 1),
                slice(channel_names.index("v_wind"), channel_names.index("v_wind") + 1),
            )
        result = climate_diagnostics(
            args.checkpoint or config["paths"]["dynamics_checkpoint"],
            token_file,
            config["paths"]["dataset_root"],
            output,
            split=args.split,
            start_index=args.start,
            steps=args.steps,
            temperature=temperature,
            top_k=args.top_k,
            max_reference_frames=args.max_reference_frames,
            cadence_hours=int(config.get("data", {}).get("cadence_hours", 6)),
            record_every=args.record_every,
            precision=config.get("runtime", {}).get("precision", "auto"),
            device_name=config.get("runtime", {}).get("device", "auto"),
            data_spectrum_path=args.data_spectrum,
            data_histograms_path=args.data_histograms,
            tokenizer_checkpoint=config["paths"]["tokenizer_checkpoint"],
            channel_names=channel_names,
            latitude_weights=latitude_weights,
            wind_channels=wind_channels,
        )
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    # Exit without running interpreter finalizers. Training completes and this
    # result prints, but CPython teardown then unwinds CUDA/NVRTC state in an
    # order that segfaults (`__cxa_finalize` -> libnvrtc), which Slurm records
    # as FAILED on a job that actually succeeded — corrupting accounting and
    # breaking any afterok dependency built on it. Everything durable is
    # already on disk by here (checkpoints and metrics are written through
    # atomic_json/atomic_torch_save inside the epoch loop), so the only thing
    # skipped is the teardown that crashes.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
