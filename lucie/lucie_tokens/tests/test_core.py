import json

import h5py
import numpy as np
import pytest
import torch
import torch.nn.functional as F

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
from lucie_tokens.training import train_dynamics, train_tokenizer
from lucie_tokens.era5_npz import (
    PROGNOSTIC_CHANNELS,
    InVramFrameLoader,
    legendre_gauss_area_weights,
    load_era5_npz,
)


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


def test_rolling_refresh_updates_one_slice_and_matches_materialize(tmp_path):
    source_root, checkpoint_path, output_root = tokenize_sequence_fixture(tmp_path)
    device = torch.device("cpu")
    tokenizer, _ = load_tokenizer_checkpoint(checkpoint_path, device)
    pairs = FramePairDataset(source_root, output_root, "train", cadence_hours=6)
    pipeline = NoisedEpochPipeline(
        tokenizer,
        pairs,
        noise_std=2.0,
        device=device,
        precision="float32",
        seed=0,
        train_batch_size=1,
        total_epochs=8,
        prepass_batch_size=2,
        refresh_k=2,
    )
    b0 = pipeline._build(0)
    b1_mat = pipeline._build(1)
    # rolling advance from epoch 0 must equal a from-scratch materialize of
    # epoch 1 — the resume-correctness invariant
    b1_adv = pipeline._advance(b0, 1)
    assert torch.equal(b1_adv[0], b1_mat[0])
    assert torch.equal(b1_adv[1], b1_mat[1])
    # epoch 0 -> 1 refreshes slice 1 (rows 1, 3, ...) and leaves slice 0 alone
    rows0 = pipeline._slice_rows[0]
    rows1 = pipeline._slice_rows[1]
    assert torch.equal(b0[0][rows0], b1_mat[0][rows0])
    assert not torch.equal(b0[0][rows1], b1_mat[0][rows1])
    # state is a pure function of the epoch (path-independent)
    assert torch.equal(pipeline._build(1)[0], b1_mat[0])
    pipeline.close()


def test_rolling_refresh_zero_std_stays_clean(tmp_path):
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
        total_epochs=8,
        prepass_batch_size=2,
        refresh_k=4,
    )
    with h5py.File(output_root / "train" / "source.h5", "r") as handle:
        stored = np.asarray(handle["tokens"])
        transitions = np.asarray(handle["transitions"])
    for epoch in (0, 3, 7):
        source, target, _ = pipeline._build(epoch)
        np.testing.assert_array_equal(source.numpy(), stored[transitions[:, 0]])
        np.testing.assert_array_equal(target.numpy(), stored[transitions[:, 1]])
    pipeline.close()


def test_rolling_refresh_k1_matches_legacy_schedule(tmp_path):
    source_root, checkpoint_path, output_root = tokenize_sequence_fixture(tmp_path)
    device = torch.device("cpu")
    tokenizer, _ = load_tokenizer_checkpoint(checkpoint_path, device)
    pairs = FramePairDataset(source_root, output_root, "train", cadence_hours=6)

    def build(refresh_k, epoch):
        pipeline = NoisedEpochPipeline(
            tokenizer, pairs, noise_std=2.0, device=device, precision="float32",
            seed=0, train_batch_size=1, total_epochs=4, prepass_batch_size=2,
            refresh_k=refresh_k,
        )
        buffers = pipeline._build(epoch)
        pipeline.close()
        return buffers

    # refresh_k=1 keys noise as (seed, epoch), one full pass — the historical
    # schedule; distinct epochs must still draw distinct noise
    a0, a1 = build(1, 0), build(1, 1)
    assert not torch.equal(a0[0], a1[0])
    assert torch.equal(build(1, 1)[0], a1[0])


def test_noised_pipeline_spawn_workers_match_inline(tmp_path):
    """Slice loaders must survive a spawn boundary and give identical tokens.

    Guards the fork-after-CUDA fix: workers are spawned (not forked), which
    requires the frame dataset to pickle — its h5py handles must be dropped."""
    import pickle

    source_root, checkpoint_path, output_root = tokenize_sequence_fixture(tmp_path)
    device = torch.device("cpu")
    tokenizer, _ = load_tokenizer_checkpoint(checkpoint_path, device)
    pairs = FramePairDataset(source_root, output_root, "train", cadence_hours=6)
    pairs[0]  # populate any lazy handle cache before pickling
    pickle.loads(pickle.dumps(pairs))  # must not raise

    def build(loader_kwargs):
        pipeline = NoisedEpochPipeline(
            tokenizer, pairs, noise_std=1.0, device=device, precision="float32",
            seed=0, train_batch_size=1, total_epochs=4, prepass_batch_size=2,
            refresh_k=2, loader_kwargs=loader_kwargs,
        )
        buffers = pipeline._build(1)
        kwargs = dict(pipeline._loader_kwargs)
        pipeline.close()
        return buffers, kwargs

    inline, inline_kwargs = build({"num_workers": 0})
    spawned, spawn_kwargs = build({"num_workers": 2, "persistent_workers": True})

    assert "multiprocessing_context" not in inline_kwargs
    assert spawn_kwargs["multiprocessing_context"] == "spawn"
    assert "persistent_workers" not in spawn_kwargs  # would leak per-refresh
    assert torch.equal(inline[0], spawned[0])
    assert torch.equal(inline[1], spawned[1])


def test_warm_page_cache_respects_budget(tmp_path):
    from lucie_tokens.data import warm_page_cache

    payload = b"x" * (1 << 20)
    paths = []
    for i in range(3):
        p = tmp_path / f"shard_{i}.bin"
        p.write_bytes(payload)
        paths.append(p)
    # budget caps total bytes; a generous budget reads everything
    assert warm_page_cache(paths, budget_bytes=1 << 19, threads=1, block_bytes=1 << 16) <= 3 << 20
    assert warm_page_cache(paths, budget_bytes=1 << 30, threads=2) == 3 << 20
    assert warm_page_cache([], budget_bytes=1 << 30) == 0
    # unreadable paths are skipped, never raised
    assert warm_page_cache([tmp_path / "missing.bin"], budget_bytes=1 << 20) == 0


def test_prepass_workers_override_and_contiguous_slices(tmp_path):
    source_root, checkpoint_path, output_root = tokenize_sequence_fixture(tmp_path)
    device = torch.device("cpu")
    tokenizer, _ = load_tokenizer_checkpoint(checkpoint_path, device)
    pairs = FramePairDataset(source_root, output_root, "train", cadence_hours=6)
    pipeline = NoisedEpochPipeline(
        tokenizer, pairs, noise_std=0.0, device=device, precision="float32",
        seed=0, train_batch_size=1, total_epochs=4, prepass_batch_size=2,
        refresh_k=3, prepass_workers=0,
        loader_kwargs={"num_workers": 2, "prefetch_factor": 2},
    )
    # explicit override wins, and prefetch_factor is dropped for in-thread mode
    assert pipeline._loader_kwargs["num_workers"] == 0
    assert "prefetch_factor" not in pipeline._loader_kwargs
    # slices partition the pairs contiguously and completely
    rows = [r.tolist() for r in pipeline._slice_rows]
    assert sorted(sum(rows, [])) == list(range(len(pairs)))
    for chunk in rows:
        assert chunk == list(range(chunk[0], chunk[-1] + 1))  # contiguous
    pipeline.close()


def test_sample_next_requires_conditioning_when_model_is_conditioned():
    """A conditioned model rolled out without its forcing vector would
    silently lose the FiLM bias, so it must refuse instead."""
    from lucie_tokens.dynamics import NextScaleDynamics, sample_next

    scales = ((1, 2), (2, 4))
    codebook = torch.randn(16, 8)
    model = NextScaleDynamics(
        codebook, scales=scales, width=32, depth=1, heads=4, conditioning_dim=5
    )
    model.eval()
    current = tuple(torch.zeros(1, h, w, dtype=torch.long) for h, w in scales)
    with pytest.raises(ValueError, match="conditioning"):
        sample_next(model, current, temperature=1.0)
    out = sample_next(
        model, current, temperature=1.0, conditioning=torch.zeros(1, 5)
    )
    assert [tuple(t.shape) for t in out] == [(1, h, w) for h, w in scales]

    # an unconditioned model is unaffected
    plain = NextScaleDynamics(codebook, scales=scales, width=32, depth=1, heads=4)
    plain.eval()
    assert len(sample_next(plain, current, temperature=1.0)) == len(scales)


def test_conditioner_prefers_reduced_co2_and_matches_full_field(tmp_path):
    """The pre-reduced global-mean series must be preferred when present and
    give vectors identical to reducing the full spatial field."""
    from lucie_tokens.conditioning import ForcingConditioner

    steps = 40
    rng = np.random.default_rng(0)
    field = rng.normal(400.0, 5.0, size=(steps, 4, 6))
    series = field.reshape(steps, -1).mean(axis=1)

    full_dir = tmp_path / "full"
    full_dir.mkdir()
    np.savez(full_dir / "co2_6hr_1981-2024.npz", data=field)

    both_dir = tmp_path / "both"
    both_dir.mkdir()
    np.savez(both_dir / "co2_6hr_1981-2024.npz", data=field)
    np.savez(both_dir / "co2_global_mean_6hr.npz", data=series)

    full = ForcingConditioner(full_dir)
    reduced = ForcingConditioner(both_dir)  # must pick the reduced one
    assert np.allclose(full._co2, reduced._co2)
    assert full._co2_mean == pytest.approx(reduced._co2_mean)
    for offset_steps in (0, 7, steps - 1, steps + 50):
        ts = 347155200.0 + offset_steps * 6 * 3600  # 1981-01-01Z + n*6h
        assert np.array_equal(full.vector(ts), reduced.vector(ts))

    # explicit filename overrides the preference order
    assert np.allclose(
        ForcingConditioner(both_dir, "co2_6hr_1981-2024.npz")._co2, series
    )
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="no CO2 series"):
        ForcingConditioner(empty_dir)


def test_load_dynamics_checkpoint_restores_conditioning(tmp_path):
    """A conditioned checkpoint must round-trip: without conditioning_dim the
    rebuilt module has no FiLM projection and load_state_dict fails."""
    from lucie_tokens.dynamics import NextScaleDynamics
    from lucie_tokens.workflows import load_dynamics_checkpoint

    scales = ((1, 2), (2, 4))
    codebook = torch.randn(16, 8)
    model = NextScaleDynamics(
        codebook, scales=scales, width=32, depth=1, heads=4, conditioning_dim=5
    )
    path = tmp_path / "dyn.pt"
    torch.save(
        {"architecture": model.architecture(), "model": model.state_dict()}, path
    )
    loaded, _ = load_dynamics_checkpoint(path, torch.device("cpu"))
    assert loaded.conditioning_dim == 5
    assert loaded.conditioning_projection is not None


def test_scale_loss_weighting_schemes_and_conditioner_without_co2(tmp_path):
    """inv_log must flatten the coarse-scale dominance of inv_sqrt, and
    dropping CO2 must leave only the cyclic (year-agnostic) features."""
    import math
    from lucie_tokens.conditioning import ForcingConditioner
    from lucie_tokens.dynamics import dynamics_loss

    ladder = ((1, 2), (2, 4), (4, 8), (8, 16), (16, 32))
    target = tuple(torch.zeros(2, h, w, dtype=torch.long) for h, w in ladder)
    logits = tuple(torch.randn(2, h * w, 32) for h, w in ladder)
    for mode in ("inv_sqrt", "inv_log", "uniform"):
        loss, metrics = dynamics_loss(logits, target, weighting=mode)
        # per-scale losses + total + off-support rate
        assert torch.isfinite(loss) and len(metrics) == len(ladder) + 2
    with pytest.raises(ValueError, match="unknown scale weighting"):
        dynamics_loss(logits, target, weighting="nope")

    counts = [h * w for h, w in ladder]
    def share(fn):
        raw = [fn(n) for n in counts]
        return raw[0] / sum(raw)
    # coarsest scale's share of the loss drops under inv_log
    assert share(lambda n: 1 / math.sqrt(n)) > share(lambda n: 1 / math.log(n + 1))

    # CO2-free conditioner needs no CO2 file at all and yields 4 cyclic dims
    plain = ForcingConditioner(tmp_path, include_co2=False)
    assert plain.dimension == 4
    v = plain.vector(347155200.0)
    assert v.shape == (4,) and np.all(np.abs(v) <= 1.0)
    # same day-of-year in different years must map to the same vector (no
    # year identity leaks in), which is the point of dropping CO2
    year = 365.25 * 24 * 3600
    assert np.allclose(plain.vector(347155200.0 + 4 * year),
                       plain.vector(347155200.0), atol=2e-2)


def test_spanning_subset_covers_whole_split_not_a_prefix():
    """A capped validation set must sample across the chronological split.
    Taking a prefix evaluated Jan-early-Mar only, which silently made every
    val number a winter-only measurement."""
    from lucie_tokens.training import _limit_dataset, _spanning_subset

    data = list(range(1459))
    capped = _spanning_subset(data, 256)
    assert len(capped) == 256
    covered = [data[i] for i in capped.indices]
    assert covered == sorted(covered)
    assert covered[0] == 0 and covered[-1] == 1458      # spans both ends
    assert max(covered) - min(covered) == 1458
    # the old behaviour stopped at 17% of the year; spanning reaches the end
    assert max(_limit_dataset(data, 256).indices) == 255

    # no-op cases pass the dataset straight through
    assert _spanning_subset(data, None) is data
    assert _spanning_subset(data, 0) is data
    assert _spanning_subset(data, 5000) is data


def test_token_substitution_noise(tmp_path):
    """Substituted tokens must be legal at their position, hit the requested
    rate, be seeded per epoch, and leave the target frame untouched."""
    from lucie_tokens.token_noise import (
        TokenSubstitutionPipeline,
        position_vocabularies,
    )

    source_root, checkpoint_path, output_root = tokenize_sequence_fixture(
        tmp_path, splits=("train",)
    )
    scales = DEFAULT_SCALES
    device = torch.device("cpu")

    with h5py.File(output_root / "train" / "source.h5", "r") as handle:
        stored = np.asarray(handle["tokens"], dtype=np.int64)
        transitions = np.asarray(handle["transitions"], dtype=np.int64)
    table, sizes = position_vocabularies(stored)
    assert table.shape[0] == stored.shape[1] == sizes.shape[0]
    # padding never introduces a token outside the position's own vocabulary
    for p in range(stored.shape[1]):
        assert set(table[p].tolist()) <= set(np.unique(stored[:, p]).tolist())

    def build(rate):
        return TokenSubstitutionPipeline(
            output_root, "train", scales, device, seed=0, train_batch_size=1,
            substitution_rate=rate, vocabulary_split="train",
        )

    clean = build(0.0)
    assert torch.equal(clean.substituted_source(0), clean.source)
    assert torch.equal(clean.substituted_source(3), clean.source)

    noisy = build(1.0)  # every position replaced -> check legality everywhere
    swapped = noisy.substituted_source(0).cpu().numpy()
    for p in range(stored.shape[1]):
        legal = set(np.unique(stored[:, p]).tolist())
        assert set(swapped[:, p].tolist()) <= legal

    half = build(0.5)
    a0, a0_again, a1 = (
        half.substituted_source(0),
        half.substituted_source(0),
        half.substituted_source(1),
    )
    assert torch.equal(a0, a0_again)          # same epoch -> same draw
    assert not torch.equal(a0, a1)            # rerolled each epoch
    # target frames are never perturbed
    expected_target = torch.from_numpy(stored[transitions[:, 1]]).int()
    assert torch.equal(half.target, expected_target)
    # realized substitution rate is close to requested (some draws re-pick the
    # original token, so this is a lower bound on positions actually selected)
    changed = (a0 != half.source).float().mean().item()
    assert 0.2 < changed < 0.5

    # context noise: independent draw from the source noise, labels clean
    context0 = half.substituted_context(0)
    assert torch.equal(context0, half.substituted_context(0))
    assert not torch.equal(context0, half.target)  # perturbed
    assert not torch.equal(  # independent of the source draw
        context0 != half.target, a0 != half.source
    )
    assert torch.equal(clean.substituted_context(0), clean.target)  # rate 0

    batches = list(half.epoch_batches(0))
    assert len(batches) == half.batches_per_epoch == len(transitions)
    current, target, conditioning, target_context = batches[0]
    assert len(current) == len(target) == len(target_context) == len(scales)
    assert conditioning is None


def test_train_dynamics_token_substitution_end_to_end(tmp_path):
    """Substitution noise must drive training without the tokenizer or any
    frame reads, and must refuse to run alongside the float noiser."""
    source_root, checkpoint_path, output_root = tokenize_sequence_fixture(
        tmp_path, splits=("train", "val")
    )
    run_dir = tmp_path / "dyn_sub"
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
            "device": "cpu", "precision": "fp32", "seed": 0, "resume": False,
            "num_workers": 0, "validation_batches": 1, "performance_mode": False,
        },
        "dynamics": {"width": 32, "depth": 1, "heads": 4},
        "dynamics_optimization": {
            "batch_size": 1, "gradient_accumulation": 1, "epochs": 2,
            "lr": 1e-3, "weight_decay": 0.0,
            "token_substitution_rate": 0.2,
        },
    }
    best = train_dynamics(config)
    assert best.exists()
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) == 2 and all("val_loss" in r for r in rows)

    config["dynamics_optimization"]["input_noise_std"] = 0.1
    with pytest.raises(ValueError, match="only one of"):
        train_dynamics(config)


def test_position_masking_is_tighter_and_round_trips(tmp_path):
    """Per-position masking must restrict generation to each position's own
    training support, and survive a checkpoint round-trip."""
    from lucie_tokens.dynamics import NextScaleDynamics, sample_next
    from lucie_tokens.workflows import load_dynamics_checkpoint

    scales = ((1, 2), (2, 4))
    total = sum(h * w for h, w in scales)
    codebook = torch.randn(16, 8)
    # position p may only emit tokens {p, p+1}
    masks = torch.zeros(total, 16, dtype=torch.bool)
    for p in range(total):
        masks[p, p % 16] = True
        masks[p, (p + 1) % 16] = True

    model = NextScaleDynamics(
        codebook, scales=scales, width=32, depth=1, heads=4, position_masks=masks
    )
    model.eval()
    current = tuple(torch.zeros(1, h, w, dtype=torch.long) for h, w in scales)
    emitted = sample_next(model, current, temperature=1.0)
    flat = torch.cat([g.reshape(-1) for g in emitted])
    for p, token in enumerate(flat.tolist()):
        assert token in {p % 16, (p + 1) % 16}, (p, token)

    path = tmp_path / "pos.pt"
    torch.save(
        {"architecture": model.architecture(), "model": model.state_dict()}, path
    )
    loaded, _ = load_dynamics_checkpoint(path, torch.device("cpu"))
    assert loaded.position_masks is not None
    assert torch.equal(loaded.position_masks, masks)
    assert loaded.architecture()["position_masked"] is True

    # unmasked models still work and report no position masking
    plain = NextScaleDynamics(codebook, scales=scales, width=32, depth=1, heads=4)
    assert plain.position_masks is None
    assert plain.architecture()["position_masked"] is False


def test_forward_masks_support_and_loss_drops_off_support_targets():
    """forward() must apply the same support mask as generation, and targets
    outside that support must be dropped from the loss rather than producing
    the infinite cross-entropy that once made val_loss a constant."""
    from lucie_tokens.dynamics import NextScaleDynamics, dynamics_loss

    scales = ((1, 2), (2, 4))
    codebook = torch.randn(16, 8)
    masks = torch.zeros(2, 16, dtype=torch.bool)
    masks[:, :8] = True                      # only codes 0-7 are in support
    model = NextScaleDynamics(
        codebook, scales=scales, width=32, depth=1, heads=4, scale_masks=masks
    )
    model.eval()
    current = tuple(torch.zeros(1, h, w, dtype=torch.long) for h, w in scales)

    legal = tuple(torch.full((1, h, w), 3, dtype=torch.long) for h, w in scales)
    with torch.no_grad():
        logits = model(current, legal)
    # forward output is masked: forbidden codes are driven to the floor
    for scale_logits in logits:
        assert bool((scale_logits[..., 8:] <= torch.finfo(scale_logits.dtype).min / 2).all())
        assert bool((scale_logits[..., :8] > torch.finfo(scale_logits.dtype).min / 2).all())

    loss, metrics = dynamics_loss(logits, legal)
    assert torch.isfinite(loss) and float(metrics["off_support_target_rate"]) == 0.0

    # a half in- / half out-of-support target stays finite, and the rate is right
    mixed = (
        torch.tensor([[[3, 12]]]),
        torch.tensor([[[3, 12, 3, 12], [3, 3, 12, 12]]]),
    )
    with torch.no_grad():
        loss, metrics = dynamics_loss(model(current, mixed), mixed)
    assert torch.isfinite(loss)
    assert float(metrics["off_support_target_rate"]) == pytest.approx(0.5)

    # even an entirely off-support scale stays finite and backpropagates
    bad = tuple(torch.full((1, h, w), 12, dtype=torch.long) for h, w in scales)
    loss, metrics = dynamics_loss(model(current, bad), bad)
    loss.backward()
    assert torch.isfinite(loss)
    assert float(metrics["off_support_target_rate"]) == 1.0



def test_pair_lead_decouples_prediction_lead_from_frame_spacing(tmp_path):
    """Hourly frames paired at a 6-hour lead must yield a pair for EVERY start
    phase, not just multiples of 6 — that is the 6x-data change."""
    from lucie_tokens.tokens import TokenPairDataset, transitions_at_lead

    hour = 3600
    stamps = np.arange(0, 24 * hour, hour, dtype=np.int64)  # 24 hourly frames

    lead6 = transitions_at_lead(stamps, 6)
    assert len(lead6) == 18                      # every t with t+6h present
    assert (stamps[lead6[:, 1]] - stamps[lead6[:, 0]] == 6 * hour).all()
    assert sorted(lead6[:, 0].tolist()) == list(range(18))   # all phases
    assert len({int(s) % 6 for s in lead6[:, 0]}) == 6       # not just 0 mod 6

    lead1 = transitions_at_lead(stamps, 1)
    assert len(lead1) == 23
    assert (stamps[lead1[:, 1]] - stamps[lead1[:, 0]] == hour).all()

    # gaps drop out rather than pairing across them
    gapped = np.array([0, hour, 2 * hour, 10 * hour, 11 * hour], dtype=np.int64)
    assert len(transitions_at_lead(gapped, 6)) == 0

    # end-to-end through the dataset: stored transitions vs recomputed lead
    source_root, checkpoint_path, output_root = tokenize_sequence_fixture(tmp_path)
    stored = TokenPairDataset(output_root, "train")
    recomputed = TokenPairDataset(output_root, "train", pair_lead_hours=6)
    assert len(stored) == len(recomputed)
    assert [r[1:] for r in stored.records] == [r[1:] for r in recomputed.records]


# --- ERA5 T30 in-VRAM npz source -------------------------------------------

_ERA5_VARS = (
    "temperature",
    "humidity",
    "u_wind",
    "v_wind",
    "surface_pressure",
    "precipitation",
    "tisr",
    "orography",
)


def make_era5_npz(path, n=24):
    rng = np.random.default_rng(0)
    np.savez(
        path,
        **{
            name: rng.standard_normal((n, 48, 96)).astype(np.float32)
            for name in _ERA5_VARS
        },
    )
    return path


def era5_data_config(path, dtype="fp32"):
    return {
        "source_kind": "era5_npz",
        "npz_path": str(path),
        "channels": list(PROGNOSTIC_CHANNELS),
        "cadence_hours": 6,
        "dtype": dtype,
        "split_index": {"train": [0, 16], "val": [16, 20], "test": [20, 24]},
    }


def test_era5_npz_load_and_manifest(tmp_path):
    path = make_era5_npz(tmp_path / "era5.npz")
    data = load_era5_npz(era5_data_config(path), torch.device("cpu"))
    assert data.frames["train"].shape == (16, 5, 48, 96)
    assert data.frames["val"].shape == (4, 5, 48, 96)
    assert data.frames["test"].shape == (4, 5, 48, 96)
    # standardization uses TRAIN statistics: train channels are ~0 mean, ~1 std
    train = data.frames["train"].float()
    assert torch.allclose(train.mean(dim=(0, 2, 3)), torch.zeros(5), atol=1e-4)
    assert torch.allclose(
        train.std(dim=(0, 2, 3), unbiased=False), torch.ones(5), atol=1e-4
    )
    # synthesized timestamps are 6-hourly and contiguous across the splits
    stamps = torch.cat([data.timestamps[s] for s in ("train", "val", "test")])
    assert torch.all(stamps[1:] - stamps[:-1] == 6 * 3600)
    assert data.manifest["state_channels"] == list(PROGNOSTIC_CHANNELS)
    assert data.manifest["grid_shape"] == [48, 96]
    assert data.manifest["source_kind"] == "era5_npz"
    assert len(data.latitude_weights) == 48
    assert np.isclose(data.latitude_weights.mean(), 1.0)


def test_legendre_gauss_weights_symmetric():
    weights = legendre_gauss_area_weights(48)
    np.testing.assert_allclose(weights, weights[::-1], rtol=1e-6)
    assert np.isclose(weights.mean(), 1.0)


def test_invram_loader_covers_and_drops():
    frames = torch.arange(24 * 5 * 48 * 96, dtype=torch.float32).reshape(24, 5, 48, 96)
    stamps = torch.arange(24, dtype=torch.int64)
    loader = InVramFrameLoader(frames, stamps, batch_size=4, shuffle=True, seed=0)
    assert len(loader) == 6
    batches = list(loader)
    assert len(batches) == 6
    state0, stamp0 = batches[0]
    assert state0.shape == (4, 5, 48, 96)
    assert stamp0.shape == (4,)
    # every frame appears exactly once across an unshuffled epoch
    ordered = InVramFrameLoader(frames, stamps, batch_size=4, shuffle=False)
    seen = torch.cat([stamp for _, stamp in ordered])
    assert sorted(seen.tolist()) == list(range(24))
    # drop_last drops the trailing partial batch
    odd = InVramFrameLoader(frames, stamps, batch_size=5, shuffle=False)
    assert len(odd) == 4


def test_era5_train_tokenizer_and_tokenize(tmp_path):
    path = make_era5_npz(tmp_path / "era5.npz")
    data_config = era5_data_config(path)
    run_dir = tmp_path / "run"
    config = {
        "config_version": 1,
        "data": data_config,
        "paths": {
            "dataset_root": str(tmp_path),
            "tokenizer_run": str(run_dir),
            "tokenizer_checkpoint": str(run_dir / "best.pt"),
            "token_root": str(tmp_path / "tokens"),
        },
        "runtime": {
            "device": "cpu",
            "precision": "fp32",
            "seed": 0,
            "resume": False,
            "performance_mode": False,
            "validation_batches": 2,
            "num_workers": 0,
        },
        "tokenizer": {
            "width": 32,
            "depth": 1,
            "heads": 4,
            "code_dim": 32,
            "codebook_size": 128,
        },
        "optimization": {
            "batch_size": 4,
            "gradient_accumulation": 1,
            "epochs": 1,
            "lr": 2e-4,
            "weight_decay": 0.0,
            "gradient_clip": 1.0,
        },
    }
    checkpoint = train_tokenizer(config)
    assert checkpoint.exists()
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["architecture"]["state_channels"] == 5

    output_root = tmp_path / "tokens"
    result = tokenize_dataset(
        config["paths"]["dataset_root"],
        checkpoint,
        output_root,
        splits=("train", "val"),
        batch_size=4,
        num_workers=0,
        device_name="cpu",
        data_config=data_config,
    )
    assert result["source_kind"] == "era5_npz"
    with h5py.File(output_root / "train" / "source.h5", "r") as handle:
        assert handle["tokens"].shape == (16, 388)
        # consecutive 6-hourly frames pair i -> i+1: 15 transitions over 16 frames
        assert handle["transitions"].shape == (15, 2)
        assert list(handle["transitions"][0]) == [0, 1]


def test_lr_schedule_warmup_cosine_and_resume():
    from lucie_tokens.runtime import make_lr_schedule

    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    optimization = {
        "lr": 2e-4,
        "epochs": 10,
        "gradient_accumulation": 2,
        "warmup_steps": 10,
    }
    # 25 batches / accum 2 -> 13 optimizer steps/epoch, 130 total
    schedule = make_lr_schedule(optimizer, optimization, 25)
    assert schedule.steps_per_epoch == 13
    assert schedule.total_steps == 130
    # warmup: 0 at step 0, peak reached at step == warmup_steps
    assert schedule.apply() == 0.0
    for _ in range(10):
        lr = schedule.apply()
    assert lr == pytest.approx(2e-4)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(lr)
    # decay: monotone down to the 1% floor at total_steps
    lrs = [schedule.apply() for _ in range(130 - schedule.step)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))
    assert lrs[-1] == pytest.approx(2e-4 * 0.01, rel=2e-2)
    # past the end it stays at the floor
    assert schedule.apply() == pytest.approx(2e-4 * 0.01)

    # resume: saved state restores the exact step; old checkpoints without
    # saved state fast-forward from the epoch count
    fresh = make_lr_schedule(optimizer, optimization, 25)
    fresh.load_state_dict(schedule.state_dict())
    assert fresh.step == schedule.step
    fresh.fast_forward(4)
    assert fresh.step == 4 * 13

    # constant schedule preserves the historical behaviour
    constant = make_lr_schedule(
        optimizer, {**optimization, "lr_schedule": "constant"}, 25
    )
    for _ in range(50):
        assert constant.apply() == pytest.approx(2e-4)


def test_self_corrupt_pipeline_triples_and_rates(tmp_path):
    """Self-generated corruption: triples chain stored transitions, rate 0 is
    byte-identical to clean training, rate 1 replaces every input row with a
    legal (in-support) sample while labels stay clean."""
    from lucie_tokens.dynamics import NextScaleDynamics
    from lucie_tokens.self_corrupt import SelfCorruptPipeline, consecutive_triples

    source_root, checkpoint_path, output_root = tokenize_sequence_fixture(
        tmp_path, splits=("train",)
    )
    device = torch.device("cpu")
    tokenizer, _ = load_tokenizer_checkpoint(checkpoint_path, device)
    with h5py.File(output_root / "train" / "source.h5", "r") as handle:
        stored = np.asarray(handle["tokens"], dtype=np.int64)
        transitions = np.asarray(handle["transitions"], dtype=np.int64)
    triples = consecutive_triples(transitions)
    following = {int(a): int(b) for a, b in transitions}
    assert len(triples) == sum(int(b) in following for _, b in transitions)
    assert all(following[i] == j and following[j] == k for i, j, k in triples)

    masks = torch.from_numpy(np.load(output_root / "scale_masks.npy"))
    model = NextScaleDynamics(
        tokenizer.quantizer.codebook, tokenizer.scales, width=32, depth=1,
        heads=4, scale_masks=masks,
    )

    def build(rate):
        return SelfCorruptPipeline(
            output_root, "train", tokenizer.scales, device, seed=0,
            train_batch_size=1, rate=rate,
        )

    clean = build(0.0)
    batches = list(clean.epoch_batches(0, model))
    assert len(batches) == clean.batches_per_epoch == len(triples)
    for current, target, conditioning in batches:
        assert conditioning is None
        flat = torch.cat([g.flatten(1) for g in current], dim=1)
        row = triples[
            [int(np.flatnonzero((stored[triples[:, 1]] == flat.numpy()).all(1))[0])]
        ]
        assert row.shape == (1, 3)

    noisy = build(1.0)
    saw_change = False
    for current, target, _ in noisy.epoch_batches(0, model):
        flat = torch.cat([g.flatten(1) for g in current], dim=1).numpy()
        labels = torch.cat([g.flatten(1) for g in target], dim=1).numpy()
        # labels are always a stored frame; inputs are sampled in-support
        assert any((labels == stored[k]).all() for k in triples[:, 2])
        for scale_index, grid in enumerate(current):
            assert masks[scale_index][grid.flatten().long()].all()
        saw_change |= not any((flat == stored[j]).all() for j in triples[:, 1])
    assert saw_change  # an untrained model does not reproduce the truth
    assert not model.training or True  # mode restored (model built in train)
    assert model.training

    # stochastic rounding of the fractional row count averages to the rate
    half = build(0.3)
    rows = [half.corrupted_rows(16, 0, step) for step in range(400)]
    assert 4.0 < float(np.mean(rows)) < 5.6
    assert half.corrupted_rows(16, 0, 7) == half.corrupted_rows(16, 0, 7)


def test_train_dynamics_self_corrupt_end_to_end(tmp_path):
    source_root, checkpoint_path, output_root = tokenize_sequence_fixture(
        tmp_path, splits=("train", "val")
    )
    run_dir = tmp_path / "dyn_self"
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
            "device": "cpu", "precision": "fp32", "seed": 0, "resume": False,
            "num_workers": 0, "validation_batches": 1, "performance_mode": False,
        },
        "dynamics": {"width": 32, "depth": 1, "heads": 4},
        "dynamics_optimization": {
            "batch_size": 1, "gradient_accumulation": 1, "epochs": 2,
            "lr": 1e-3, "weight_decay": 0.0, "self_corrupt_rate": 0.5,
        },
    }
    best = train_dynamics(config)
    assert best.exists()
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) == 2 and all("val_loss" in r for r in rows)
    config["dynamics_optimization"]["token_substitution_rate"] = 0.2
    with pytest.raises(ValueError, match="only one of"):
        train_dynamics(config)


def test_gust2_codebook_semantics_and_defaults_unchanged():
    """dead_code_relative + dead_code_reinit='normal' reproduce gust2/vq.py's
    ema_update: dead codes become random unit normals credited with the
    batch's uniform usage; the default path still reseeds from batch values."""
    from lucie_tokens.tokenizer import ResidualMultiScaleVQ

    torch.manual_seed(0)
    scales = ((1, 2), (2, 4))
    values = [torch.randn(4, 8, h, w) * 5 for h, w in scales]     # (B, D, h, w)
    indices = [torch.zeros(4 * h * w, dtype=torch.long) for h, w in scales]  # everything -> code 0
    total_vectors = sum(i.numel() for i in indices)

    g2 = ResidualMultiScaleVQ(8, 16, scales, ema_decay=0.85, dead_code_threshold=0.25,
                              dead_code_relative=True, dead_code_reinit="normal",
                              norm_guard_factor=0.0)
    g2.train()
    g2.ema_count.zero_()   # fresh codes start with count 1 and survive one step; force the dead branch
    g2._ema_update(values, indices)
    uniform = total_vectors / 16
    assert torch.allclose(g2.codebook[1:].norm(dim=1), torch.ones(15), atol=1e-5)   # unit normals
    assert torch.allclose(g2.ema_count[1:], torch.full((15,), uniform))            # uniform credit
    assert float(g2.ema_count[0]) > uniform                                           # live code kept
    assert not torch.allclose(g2.codebook[0].norm(), torch.tensor(1.0))

    lucie = ResidualMultiScaleVQ(8, 16, scales)   # defaults: absolute 0.5, batch reseed, guard on
    lucie.train()
    lucie.ema_count.zero_()
    lucie._ema_update(values, indices)
    flat = torch.cat([v.permute(0, 2, 3, 1).reshape(-1, 8) for v in values])
    # reseeded from (possibly norm-guarded) batch values, count 1
    assert torch.allclose(lucie.ema_count[1:], torch.ones(15))
    assert not torch.allclose(lucie.codebook[1:].norm(dim=1), torch.ones(15), atol=1e-3)
    with pytest.raises(ValueError):
        ResidualMultiScaleVQ(8, 16, scales, dead_code_reinit="bogus")
    arch = ClimateTokenizer(width=16, depth=1, heads=4, code_dim=8, codebook_size=16,
                            dead_code_reinit="normal", dead_code_relative=True).architecture()
    assert arch["dead_code_reinit"] == "normal" and arch["dead_code_relative"] is True


def test_per_scale_codebooks_slice_ids_ema_and_round_trip(tmp_path):
    """codebook_sizes gives each scale a private id range of one concatenated
    book: ids are global, EMA/reseed touch only the scale's own slice and draw
    from the scale's own values, and the checkpoint round-trips through
    load_tokenizer_checkpoint / tokenize with a total-vocabulary mask."""
    from lucie_tokens.tokenizer import ResidualMultiScaleVQ

    torch.manual_seed(0)
    scales = ((1, 2), (2, 4), (3, 6))
    sizes = (4, 8, 16)
    vq = ResidualMultiScaleVQ(8, 4096, scales, codebook_sizes=sizes,
                              dead_code_relative=True, dead_code_threshold=0.25)
    assert vq.codebook_size == 28 and vq.codebook_offsets == (0, 4, 12)
    assert vq.codebook.shape == (28, 8)
    before = vq.codebook.clone()
    vq.train()
    pyramid = vq(torch.randn(3, 8, 3, 6) * 3)
    for scale_index, (offset, size) in enumerate(zip(vq.codebook_offsets, sizes)):
        ids = pyramid.indices[scale_index]
        assert ids.shape == (3, *scales[scale_index])
        assert int(ids.min()) >= offset and int(ids.max()) < offset + size
    # every slice moved (EMA), and only via its own scale: feed one scale a
    # huge value and check the other slices are untouched
    vq2 = ResidualMultiScaleVQ(8, 4096, scales, codebook_sizes=sizes,
                               dead_code_relative=True, dead_code_threshold=0.0,
                               norm_guard_factor=0.0)
    vq2.train()
    snapshot = vq2.codebook.clone()
    values = [torch.zeros(1, 8, h, w) for h, w in scales]
    values[1] = torch.full((1, 8, 2, 4), 50.0)
    indices = [torch.full((h * w,), o, dtype=torch.long)
               for (h, w), o in zip(scales, vq2.codebook_offsets)]
    vq2._ema_update(values, indices)
    # slice 1's code 4 is pulled toward 50; slices 0 and 2 (fed zeros) never
    # see the 50s, so nothing there can grow
    assert float(vq2.codebook[4].mean()) > 1.0
    assert float(vq2.codebook[0:4].abs().max()) < 1.5
    assert float(vq2.codebook[12:28].abs().max()) < 1.5
    # reseeds draw from the scale's own values: kill every code of scale 1 and
    # every replacement must be a scale-1 value (all 50s), never a 0 from
    # another scale
    vq3 = ResidualMultiScaleVQ(8, 4096, scales, codebook_sizes=sizes,
                               dead_code_relative=True, dead_code_threshold=0.5,
                               norm_guard_factor=0.0)
    vq3.train()
    vq3.ema_count.zero_()
    vq3._ema_update(values, indices)
    assert torch.allclose(vq3.codebook[5:12], torch.full((7, 8), 50.0))
    assert torch.allclose(vq3.codebook[1:4], torch.zeros(3, 8))
    with pytest.raises(ValueError):
        ResidualMultiScaleVQ(8, 4096, scales, codebook_sizes=(4, 8))
    with pytest.raises(ValueError):   # absolute threshold + private slices
        ResidualMultiScaleVQ(8, 4096, scales, codebook_sizes=sizes)
    # the shared default is untouched: single (K, D) book, ids < K
    shared = ResidualMultiScaleVQ(8, 32, scales)
    assert shared.codebook_sizes is None and shared.codebook.shape == (32, 8)
    assert shared.codebook_offsets == (0, 0, 0)

    tokenizer = ClimateTokenizer(width=16, depth=1, heads=4, code_dim=8,
                                 codebook_size=4096, scales=scales,
                                 codebook_sizes=sizes, dead_code_relative=True,
                                 state_channels=5, grid_shape=(6, 12))
    assert tokenizer.codebook_size == 28
    arch = tokenizer.architecture()
    assert arch["codebook_sizes"] == [4, 8, 16] and arch["codebook_size"] == 28
    path = tmp_path / "tok.pt"
    torch.save({"architecture": arch, "model": tokenizer.state_dict()}, path)
    loaded, _ = load_tokenizer_checkpoint(path, torch.device("cpu"))
    assert loaded.codebook_sizes == sizes and loaded.codebook_size == 28
    state = torch.randn(2, 5, 6, 12)
    assert all(torch.equal(a, b) for a, b in zip(
        loaded.encode(state).indices, tokenizer.eval().encode(state).indices))


def test_cascade_stabilisation_gain_clamp_and_absolute_anchor(tmp_path):
    """phi_gain_limit bounds each stage's per-sample rms gain (encode and
    decode paths agree); ema_anchor='encoder' turns the norm guard into an
    absolute bound that does not ratchet with the codebook. Defaults off."""
    from lucie_tokens.tokenizer import ResidualMultiScaleVQ

    torch.manual_seed(0)
    scales = ((1, 2), (2, 4), (3, 6))
    vq = ResidualMultiScaleVQ(8, 32, scales, phi_gain_limit=1.0)
    for phi in vq.phi:                       # force gains far above 1
        with torch.no_grad():
            phi.weight.mul_(50.0)
    vq.eval()
    z = torch.randn(4, 8, 3, 6)
    pyramid = vq(z, update_ema=False)
    prev = torch.zeros_like(z)
    with torch.no_grad():
        for k, partial in enumerate(pyramid.partial_latents):
            update = partial - prev
            values = F.embedding(pyramid.indices[k].flatten(), vq.codebook)
            values = values.reshape(4, *scales[k], 8).permute(0, 3, 1, 2)
            contribution = F.interpolate(values, size=(3, 6), mode="bilinear", align_corners=False)
            gain = update.square().mean(dim=(1, 2, 3)).sqrt() / contribution.square().mean(dim=(1, 2, 3)).sqrt()
            assert torch.all(gain <= 1.0 + 1e-4), gain
            prev = partial
    decoded = vq.indices_to_partials(pyramid.indices)
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(decoded, pyramid.partial_latents))
    off = ResidualMultiScaleVQ(8, 32, scales)
    off.load_state_dict(vq.state_dict())     # same parameters, no clamp
    off.eval()
    loud = off(z, update_ema=False).partial_latents[-1]
    assert loud.abs().max() > pyramid.partial_latents[-1].abs().max() * 5

    values = [torch.ones(1, 8, h, w) * 1000.0 for h, w in scales]   # runaway residuals
    indices = [torch.zeros(h * w, dtype=torch.long) for h, w in scales]
    anchored = ResidualMultiScaleVQ(8, 32, scales, ema_anchor="encoder")
    ratchet = ResidualMultiScaleVQ(8, 32, scales)
    anchored.train(); ratchet.train()
    # both clamp on step 1 (min of medians = 1 -> limit 8); the historical
    # guard then ratchets: idle codes die, get reseeded with clamped (norm 8)
    # values, the codebook median rises and so does the limit. The anchored
    # guard is pinned to the encoder norm and never lets a code past 8.
    for _ in range(400):
        anchored._ema_update(values, indices, anchor=torch.tensor(1.0))
        ratchet._ema_update(values, indices)
    assert float(anchored.codebook.norm(dim=1).max()) <= 8.0 + 1e-3
    assert float(ratchet.codebook[0].norm()) > 16.0, float(ratchet.codebook[0].norm())
    with pytest.raises(ValueError):
        ResidualMultiScaleVQ(8, 32, scales, ema_anchor="bogus")
    tok = ClimateTokenizer(width=16, depth=1, heads=4, code_dim=8, codebook_size=16,
                           scales=scales, state_channels=5, grid_shape=(6, 12),
                           phi_gain_limit=1.0, ema_anchor="encoder")
    arch = tok.architecture()
    assert arch["phi_gain_limit"] == 1.0 and arch["ema_anchor"] == "encoder"
    path = tmp_path / "tok.pt"
    torch.save({"architecture": arch, "model": tok.state_dict()}, path)
    loaded, _ = load_tokenizer_checkpoint(path, torch.device("cpu"))
    assert loaded.quantizer.phi_gain_limit == 1.0 and loaded.quantizer.ema_anchor == "encoder"
    tok.train(); tok(torch.randn(2, 5, 6, 12))    # anchored EMA path runs end to end


def _tiny_dynamics_run(tmp_path):
    """Synthetic 33-ch source -> tokens -> 2-epoch dynamics; returns paths."""
    source_root, tokenizer_path, token_root = tokenize_sequence_fixture(
        tmp_path, splits=("train", "val")
    )
    run_dir = tmp_path / "dynamics"
    config = {
        "config_version": 1,
        "data": {"cadence_hours": 6},
        "paths": {
            "dataset_root": str(source_root),
            "tokenizer_checkpoint": str(tokenizer_path),
            "token_root": str(token_root),
            "dynamics_run": str(run_dir),
        },
        "runtime": {"device": "cpu", "precision": "fp32", "seed": 0, "resume": False,
                    "num_workers": 0, "validation_batches": 1, "performance_mode": False},
        "dynamics": {"width": 32, "depth": 1, "heads": 4},
        "dynamics_optimization": {"batch_size": 1, "gradient_accumulation": 1,
                                  "epochs": 2, "lr": 1e-3, "weight_decay": 0.0},
    }
    train_dynamics(config)
    return source_root, tokenizer_path, token_root, run_dir


def test_climate_diagnostics_end_to_end(tmp_path):
    """Characterisation: the standard rollout report runs on the synthetic
    source and writes the artifacts the Bridges diag chain reads."""
    from lucie_tokens.diagnostics import climate_diagnostics

    source_root, _, token_root, run_dir = _tiny_dynamics_run(tmp_path)
    out = tmp_path / "diag"
    report = climate_diagnostics(
        run_dir / "latest.pt", token_root / "val" / "source.h5", source_root, out,
        split="val", steps=4, temperature=1.0, max_reference_frames=4,
        device_name="cpu", precision="fp32",
    )
    assert report["stability"]["completed_steps"] == 4
    assert not report["stability"]["blew_up"]
    ke = report["rollout_vs_data"]["kinetic_energy"]
    assert np.isfinite(ke["relative_l2"]) and len(ke["ratio_by_wavenumber"]) == 48
    assert set(report["pixel_distributions"]) == set(STATE_CHANNELS)
    assert len(report["token_entropy"]["delta_per_scale"]) == len(DEFAULT_SCALES)
    for name in ("diagnostics.json", "histograms.npz", "token_entropy.npz",
                 "rollout_tokens.h5", "ke_spectrum.png", "pixel_histograms.png"):
        assert (out / name).exists(), name
    with h5py.File(out / "rollout_tokens.h5", "r") as handle:
        assert handle["tokens"].shape == (4, 388)
    # precomputed truth references reproduce the from-source path
    from lucie_tokens.diagnostics import HIST_EDGES
    refs = tmp_path / "refs"
    refs.mkdir()
    np.savez(refs / "truth_spectrum.npz",
             per_channel=np.ones((33, 48)), kinetic_energy=np.ones(48))
    np.savez(refs / "truth_hist.npz", edges=HIST_EDGES, counts=np.ones((33, 120)))
    report2 = climate_diagnostics(
        run_dir / "latest.pt", token_root / "val" / "source.h5", source_root,
        tmp_path / "diag2", split="val", steps=2, device_name="cpu", precision="fp32",
        data_spectrum_path=refs / "truth_spectrum.npz",
        data_histograms_path=refs / "truth_hist.npz",
    )
    assert report2["spectra"]["kinetic_energy"]["data"] == [1.0] * 48


def test_token_space_diagnostics_on_fixture(tmp_path):
    """The merged token-space analyses run end to end on the synthetic run:
    mechanism, scale oracle, coarse chain, code reuse, calibration, autopsy."""
    from lucie_tokens import diagnostics as D

    source_root, tokenizer_path, token_root, run_dir = _tiny_dynamics_run(tmp_path)
    ckpt = run_dir / "latest.pt"
    mech = D.rollout_mechanism(ckpt, token_root, held_out=("val",), train_pairs=4,
                               steps=3, batch_size=2, device_name="cpu", precision="fp32")
    assert len(mech["teacher_forced_held_out"]["top1_accuracy"]) == len(DEFAULT_SCALES)
    assert mech["rollout"]["steps"] == 3
    assert all(0.0 <= r <= 1.0 for r in mech["rollout"]["unseen_parent_child_rate"])
    assert "H_tf" in D.format_mechanism(mech)

    report, spectra = D.scale_oracle(
        ckpt, token_root, splits=("val",), max_steps=2, channels=list(STATE_CHANNELS),
        u_channel="U-wind_3", v_channel="V-wind_3", device_name="cpu", precision="fp32",
    )
    assert set(report["runs"]) == {"recon", "onestep"} | {f"oracle_{k}" for k in range(len(DEFAULT_SCALES))}
    assert np.isclose(report["runs"]["recon"]["low_m_ke_ratio"], 1.0)
    assert set(spectra) == set(report["runs"])
    assert "lowm_KE" in D.format_oracle(report)

    tokens, transitions, _ = D.load_token_file(token_root / "val" / "source.h5")
    assert transitions is not None
    stats = D.chain_stats(D.coarse_sequence(tokens, DEFAULT_SCALES, 0), 128)
    assert stats["states_visited"] >= 1 and 0.0 <= stats["change_rate"] <= 1.0
    train, _, _ = D.load_token_file(token_root / "train" / "source.h5")
    rows = D.code_reuse_table(train, tokens, DEFAULT_SCALES, 128)
    assert [r["scale"] for r in rows] == ["1x2", "2x4", "3x6", "6x12", "12x24"]
    assert all(0.0 <= r["held_out_unseen"] <= 1.0 for r in rows)

    result, data_entropy, responses = D.calibrate_temperature(
        ckpt, token_root / "val" / "source.h5", temperatures=(0.8, 1.2), steps=3,
        device_name="cpu", precision="fp32",
    )
    assert responses.shape == (2, len(DEFAULT_SCALES)) and len(result["solved_temperature_per_scale"]) == 5
    D.save_calibration_plot(tmp_path, result, data_entropy, responses)

    autopsy = D.codebook_autopsy(tokenizer_path)
    assert autopsy["codebook"]["n_codes"] == 128 and len(autopsy["phi_dc_gain_by_scale"]) == 5

    # pure-numpy pieces
    weights = np.ones(48)
    k, power = D.zonal_power_spectrum(np.random.default_rng(0).standard_normal((3, 48, 96)), weights)
    assert len(k) == 48 and np.all(power >= 0)
    metrics = D.spectrum_metrics(power, power)
    assert np.isclose(metrics["relative_l2"], 0.0) and np.isclose(metrics["high_wavenumber_retention"], 1.0)
    assert D.histogram_emd(np.ones(120), np.ones(120)) == 0.0
