import json

import h5py
import numpy as np
import torch

from lucie_tokens.config import load_config
from lucie_tokens.data import (
    BlockShuffleBatchSampler,
    FramePairDataset,
    SourceClimateFrameDataset,
    load_frame_dataset_manifest,
    load_frame_grid,
    load_normalization,
    parse_source_filename,
    source_files,
)
from lucie_tokens.geometry import gaussian_latitude_weights
from lucie_tokens.noised_epochs import NoisedEpochPipeline
from lucie_tokens.schema import DEFAULT_SCALES, STATE_CHANNELS, scale_offsets
from lucie_tokens.tokenizer import ClimateTokenizer
from lucie_tokens.tokens import load_tokenizer_checkpoint, tokenize_dataset
from lucie_tokens.training import train_dynamics


def make_source_fixture(root):
    np.savez(
        root / "normalize_mean.npz",
        **{name: np.asarray([0.0]) for name in STATE_CHANNELS},
    )
    np.savez(
        root / "normalize_std.npz",
        **{name: np.asarray([2.0]) for name in STATE_CHANNELS},
    )
    np.save(root / "lat.npy", np.linspace(-87.0, 87.0, 48))
    np.save(root / "lon.npy", np.linspace(0.0, 360.0, 96, endpoint=False))
    split = root / "train"
    split.mkdir()
    for hour in (0, 1):
        with h5py.File(split / f"1981_{hour:04d}.h5", "w") as handle:
            group = handle.create_group("input")
            for name in STATE_CHANNELS:
                group.create_dataset(
                    name, data=np.full((48, 96), 2.0, dtype=np.float32)
                )
    (split / "1981_0000_backup.h5").touch()


def test_static_contract():
    assert len(STATE_CHANNELS) == 33
    assert scale_offsets(DEFAULT_SCALES)[-1] == 388
    assert parse_source_filename("1981_0006.h5")[1] == 6


def test_source_files_ignore_non_timestep_hdf5(tmp_path):
    split = tmp_path / "train"
    split.mkdir()
    (split / "1981_0000.h5").touch()
    (split / "1981_0000_backup.h5").touch()

    assert [path.name for path in source_files(tmp_path, "train")] == [
        "1981_0000.h5"
    ]


def test_direct_source_dataset_and_contract(tmp_path):
    make_source_fixture(tmp_path)

    dataset = SourceClimateFrameDataset(tmp_path, "train", cadence_hours=6)
    state, timestamp = dataset[0]
    manifest = load_frame_dataset_manifest(tmp_path)
    mean, std, weights = load_frame_grid(tmp_path)

    assert len(dataset) == 1
    assert state.shape == (len(STATE_CHANNELS), 48, 96)
    assert np.all(state.numpy() == 1.0)
    assert timestamp.ndim == 0
    assert manifest["source_kind"] == "direct_hourly_h5"
    assert len(manifest["dataset_hash"]) == 64
    assert np.all(mean == 0.0)
    assert np.all(std == 2.0)
    assert np.isclose(weights.mean(), 1.0)


def test_direct_source_token_export(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    make_source_fixture(source_root)
    model = ClimateTokenizer(
        width=64,
        depth=1,
        heads=4,
        code_dim=32,
        codebook_size=128,
    )
    checkpoint_path = tmp_path / "tokenizer.pt"
    torch.save(
        {
            "architecture": model.architecture(),
            "model": model.state_dict(),
            "dataset_hash": "fixture",
        },
        checkpoint_path,
    )

    output_root = tmp_path / "tokens"
    result = tokenize_dataset(
        source_root,
        checkpoint_path,
        output_root,
        splits=("train",),
        batch_size=1,
        num_workers=0,
        device_name="cpu",
    )

    with h5py.File(output_root / "train" / "source.h5", "r") as handle:
        assert handle["tokens"].shape == (1, 388)
        assert handle["transitions"].shape == (0, 2)
    assert result["source_kind"] == "direct_hourly_h5"
    assert (output_root / "scale_masks.npy").exists()


def test_latitude_weights_are_symmetric():
    latitude = np.linspace(-87.0, 87.0, 48)
    weights = gaussian_latitude_weights(latitude)
    np.testing.assert_allclose(weights, weights[::-1])
    assert np.isclose(weights.mean(), 1.0)


def test_load_normalization_hashes_files(tmp_path):
    mean_path = tmp_path / "normalize_mean.npz"
    std_path = tmp_path / "normalize_std.npz"
    np.savez(mean_path, **{name: np.asarray([0.0]) for name in STATE_CHANNELS})
    np.savez(std_path, **{name: np.asarray([1.0]) for name in STATE_CHANNELS})

    mean, std, metadata = load_normalization(tmp_path)

    assert mean.shape == (len(STATE_CHANNELS),)
    assert std.shape == (len(STATE_CHANNELS),)
    assert len(metadata["mean_sha256"]) == 64
    assert len(metadata["std_sha256"]) == 64


def test_load_config_expands_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("LUCIE_TEST_ROOT", str(tmp_path))
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        'config_version = 1\n[paths]\ndataset_root = "${LUCIE_TEST_ROOT}/data"\n'
    )

    config = load_config(config_path)

    assert config["paths"]["dataset_root"] == str(tmp_path / "data")


def test_block_sampler_length_matches_iteration():
    sampler = BlockShuffleBatchSampler(
        length=20,
        batch_size=4,
        block_size=6,
    )

    assert len(sampler) == len(list(sampler)) == 3


def make_sequence_fixture(root, split="train", hours=(0, 6, 12, 18)):
    """Source tree with distinct random frames so transitions exist and
    token sequences differ per frame (make_source_fixture is constant)."""
    rng = np.random.default_rng(abs(hash(split)) % 2**32)
    for name, value in (("normalize_mean.npz", 0.0), ("normalize_std.npz", 1.0)):
        if not (root / name).exists():
            np.savez(root / name, **{c: np.asarray([value]) for c in STATE_CHANNELS})
    if not (root / "lat.npy").exists():
        np.save(root / "lat.npy", np.linspace(-87.0, 87.0, 48))
        np.save(root / "lon.npy", np.linspace(0.0, 360.0, 96, endpoint=False))
    split_dir = root / split
    split_dir.mkdir()
    for hour in hours:
        with h5py.File(split_dir / f"1981_{hour:04d}.h5", "w") as handle:
            group = handle.create_group("input")
            for name in STATE_CHANNELS:
                group.create_dataset(
                    name, data=rng.normal(size=(48, 96)).astype(np.float32)
                )


def tokenize_sequence_fixture(tmp_path, splits=("train",)):
    source_root = tmp_path / "source"
    source_root.mkdir()
    for split in splits:
        make_sequence_fixture(source_root, split)
    model = ClimateTokenizer(
        width=64, depth=1, heads=4, code_dim=32, codebook_size=128
    )
    checkpoint_path = tmp_path / "tokenizer.pt"
    torch.save(
        {
            "architecture": model.architecture(),
            "model": model.state_dict(),
            "dataset_hash": "fixture",
        },
        checkpoint_path,
    )
    output_root = tmp_path / "tokens"
    tokenize_dataset(
        source_root,
        checkpoint_path,
        output_root,
        splits=splits,
        batch_size=2,
        num_workers=0,
        device_name="cpu",
    )
    return source_root, checkpoint_path, output_root


def test_noised_epoch_buffer_matches_clean_tokens_at_zero_std(tmp_path):
    source_root, checkpoint_path, output_root = tokenize_sequence_fixture(tmp_path)
    device = torch.device("cpu")
    tokenizer, _ = load_tokenizer_checkpoint(checkpoint_path, device)
    pairs = FramePairDataset(source_root, output_root, "train", cadence_hours=6)
    pipeline = NoisedEpochPipeline(
        tokenizer,
        pairs,
        noise_std=0.0,
        device=device,
        precision="float32",
        seed=0,
        train_batch_size=1,
        total_epochs=1,
        prepass_batch_size=2,
    )
    source, target, conditioning = pipeline._build(0)
    pipeline.close()
    with h5py.File(output_root / "train" / "source.h5", "r") as handle:
        stored = np.asarray(handle["tokens"])
        transitions = np.asarray(handle["transitions"])

    assert len(pairs) == len(transitions) == 3
    assert conditioning is None
    np.testing.assert_array_equal(source.numpy(), stored[transitions[:, 0]])
    np.testing.assert_array_equal(target.numpy(), stored[transitions[:, 1]])


def test_noised_epoch_buffers_seeded_per_epoch(tmp_path):
    source_root, checkpoint_path, output_root = tokenize_sequence_fixture(tmp_path)
    device = torch.device("cpu")
    tokenizer, _ = load_tokenizer_checkpoint(checkpoint_path, device)
    pairs = FramePairDataset(source_root, output_root, "train", cadence_hours=6)
    pipeline = NoisedEpochPipeline(
        tokenizer,
        pairs,
        noise_std=1.0,
        device=device,
        precision="float32",
        seed=0,
        train_batch_size=1,
        total_epochs=1,
        prepass_batch_size=2,
    )
    source_a, _, _ = pipeline._build(0)
    source_b, _, _ = pipeline._build(0)
    source_next, _, _ = pipeline._build(1)
    batches = list(pipeline.epoch_batches(0))
    pipeline.close()

    assert torch.equal(source_a, source_b)  # same epoch -> same noise draw
    assert not torch.equal(source_a, source_next)  # fresh noise per epoch
    assert len(batches) == pipeline.batches_per_epoch == 3
    current, target, conditioning = batches[0]
    assert len(current) == len(target) == len(DEFAULT_SCALES)
    assert current[0].shape[0] == 1
    assert conditioning is None


def test_train_dynamics_noised_end_to_end(tmp_path):
    source_root, checkpoint_path, output_root = tokenize_sequence_fixture(
        tmp_path, splits=("train", "val")
    )
    run_dir = tmp_path / "dynamics"
    config = {
        "config_version": 1,
        "data": {"cadence_hours": 6},
        "paths": {
            "dataset_root": str(source_root),
            "tokenizer_checkpoint": str(checkpoint_path),
            "token_root": str(output_root),
            "dynamics_run": str(run_dir),
        },
        "runtime": {
            "device": "cpu",
            "precision": "auto",
            "seed": 0,
            "resume": False,
            "num_workers": 0,
            "validation_batches": 1,
            "performance_mode": False,
        },
        "dynamics": {"width": 32, "depth": 1, "heads": 4},
        "dynamics_optimization": {
            "batch_size": 1,
            "gradient_accumulation": 2,
            "epochs": 2,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "input_noise_std": 0.1,
            "noise_prepass_batch_size": 2,
        },
    }

    best = train_dynamics(config)

    assert best.exists()
    metrics = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
    ]
    assert [m["epoch"] for m in metrics] == [0, 1]
    assert all(np.isfinite(m["train_loss"]) for m in metrics)
    assert all(np.isfinite(m["val_loss"]) for m in metrics)
