from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import evaluate_rollout_file, evaluate_tokenizer
from .benchmark import benchmark_tokenizer
from .config import load_config
from .data import (
    build_training_cache,
    convert_hourly,
    inspect_source,
    validate_dataset,
)
from .runtime import atomic_json
from .imports import import_lucie2d_preprocessed, import_lucie3d_test
from .tokens import tokenize_dataset
from .training import train_dynamics, train_tokenizer
from .workflows import rollout


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

    convert = commands.add_parser("convert")
    convert.add_argument("source")
    convert.add_argument("output")
    convert.add_argument("--splits", default="train,val,test")
    convert.add_argument("--compression", choices=("lzf", "gzip"))
    convert.add_argument("--overwrite", action="store_true")
    convert.add_argument(
        "--years",
        help="Comma-separated source filename years, for example 1981,2010",
    )
    convert.add_argument(
        "--max-files-per-year",
        type=int,
        help="Keep only the first N hourly files from each selected year",
    )

    cache = commands.add_parser("build-cache")
    cache.add_argument("dataset")
    cache.add_argument("--splits", default="train,val,test")
    cache.add_argument("--overwrite", action="store_true")
    cache.add_argument("--cadence-hours", type=int, default=6)

    validate = commands.add_parser("validate-data")
    validate.add_argument("dataset")
    validate.add_argument("--cadence-hours", type=int, default=6)

    import_3d = commands.add_parser("import-lucie3d-test")
    import_3d.add_argument("source")
    import_3d.add_argument("output")
    import_3d.add_argument("--train-years", default="2011")
    import_3d.add_argument("--val-years", default="2012")
    import_3d.add_argument("--test-years", default="2013")
    import_3d.add_argument("--overwrite", action="store_true")

    import_2d = commands.add_parser("import-lucie2d")
    import_2d.add_argument("source")
    import_2d.add_argument("output")
    import_2d.add_argument("--train-stop", type=int, default=16000)
    import_2d.add_argument("--val-stop", type=int, default=16100)
    import_2d.add_argument("--overwrite", action="store_true")

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
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "inspect-source":
        result = inspect_source(
            args.source, _split_values(args.splits), args.samples
        )
        if args.output:
            atomic_json(args.output, result)
    elif args.command == "convert":
        result = convert_hourly(
            source_root=args.source,
            output_root=args.output,
            splits=_split_values(args.splits),
            compression=args.compression,
            overwrite=args.overwrite,
            years=(
                tuple(int(value) for value in _split_values(args.years))
                if args.years
                else None
            ),
            max_files_per_year=args.max_files_per_year,
        )
    elif args.command == "build-cache":
        result = build_training_cache(
            args.dataset,
            _split_values(args.splits),
            args.overwrite,
            args.cadence_hours,
        )
    elif args.command == "validate-data":
        result = validate_dataset(args.dataset, args.cadence_hours)
    elif args.command == "import-lucie3d-test":
        result = import_lucie3d_test(
            args.source,
            args.output,
            train_years=tuple(
                int(value) for value in _split_values(args.train_years)
            ),
            val_years=tuple(
                int(value) for value in _split_values(args.val_years)
            ),
            test_years=tuple(
                int(value) for value in _split_values(args.test_years)
            ),
            overwrite=args.overwrite,
        )
    elif args.command == "import-lucie2d":
        result = import_lucie2d_preprocessed(
            args.source,
            args.output,
            train_stop=args.train_stop,
            val_stop=args.val_stop,
            overwrite=args.overwrite,
        )
    elif args.command == "train-tokenizer":
        result = {"checkpoint": str(train_tokenizer(load_config(args.config)))}
    elif args.command == "tokenize":
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
        result = {"checkpoint": str(train_dynamics(load_config(args.config)))}
    elif args.command == "rollout":
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
        result = evaluate_rollout_file(args.rollout, args.output)
    elif args.command == "benchmark":
        result = benchmark_tokenizer(
            load_config(args.config), args.iterations
        )
        if args.output:
            atomic_json(args.output, result)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
