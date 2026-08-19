from __future__ import annotations

import argparse
import json
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
    diagnostics.add_argument("--temperature", type=float, default=1.0)
    diagnostics.add_argument("--top-k", type=int)
    diagnostics.add_argument("--max-reference-frames", type=int)
    diagnostics.add_argument("--checkpoint", help="override paths.dynamics_checkpoint")
    diagnostics.add_argument(
        "--data-spectrum", help="npz of precomputed truth spectrum (skips source read)"
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
        token_file = str(
            Path(config["paths"]["token_root"]) / args.split / "source.h5"
        )
        output = args.output or str(
            Path(config["paths"]["dynamics_run"]) / f"diagnostics_{args.split}"
        )
        result = climate_diagnostics(
            args.checkpoint or config["paths"]["dynamics_checkpoint"],
            token_file,
            config["paths"]["dataset_root"],
            output,
            split=args.split,
            start_index=args.start,
            steps=args.steps,
            temperature=args.temperature,
            top_k=args.top_k,
            max_reference_frames=args.max_reference_frames,
            cadence_hours=int(config.get("data", {}).get("cadence_hours", 6)),
            precision=config.get("runtime", {}).get("precision", "auto"),
            device_name=config.get("runtime", {}).get("device", "auto"),
            data_spectrum_path=args.data_spectrum,
            tokenizer_checkpoint=config["paths"]["tokenizer_checkpoint"],
        )
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
