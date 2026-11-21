"""Forcing conditioning for the next-scale dynamics model (Tier A).

lucie.md specifies the emulator's forcings (CO2, solar insolation, SST, ...)
as conditioning inputs rather than tokenized channels. This module supplies a
lightweight, per-frame conditioning vector derived entirely from each frame's
timestamp plus the CO2 series -- no re-tokenize and no per-frame source reads:

    [ co2_norm, sin(doy), cos(doy), sin(hod), cos(hod) ]

where doy is the fractional day-of-year (season -> insolation envelope) and hod
is the fractional hour-of-day (diurnal cycle; at 6-hour steps it cycles through
0/6/12/18). CO2 is the well-mixed global mean, indexed by timestamp. Spatial
forcing fields (Tier B) are a deliberate follow-up.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .tokens import TokenPairDataset, collate_token_pairs

# 6-hourly CO2 series is anchored at 1981-01-01 00Z (see lucie.md / filename).
_CO2_EPOCH = datetime(1981, 1, 1, tzinfo=timezone.utc).timestamp()
_CO2_STEP_SECONDS = 6 * 3600


class ForcingConditioner:
    """Maps a UTC timestamp (seconds) to a fixed-width conditioning vector.

    include_co2=False drops the CO2 channel, leaving only the cyclic
    [sin/cos(doy), sin/cos(hod)] features. CO2 rises monotonically over the
    record, so it nearly uniquely identifies the year a sample came from —
    a memorization key rather than a physical driver — and the held-out year
    sits in a sparsely covered part of that axis. The cyclic features repeat
    across all years and carry no year identity.
    """

    # A pre-reduced global-mean series loads in milliseconds instead of
    # pulling the 1.16 GB spatial field through the reduction on every run.
    # It is preferred when present; the full field stays as the fallback.
    # Both yield identical vectors: reducing an already-1D series is a no-op.
    CO2_FILENAMES = ("co2_global_mean_6hr.npz", "co2_6hr_1981-2024.npz")

    def __init__(
        self,
        dataset_root: str | Path,
        co2_filename: str | None = None,
        include_co2: bool = True,
    ) -> None:
        self.include_co2 = bool(include_co2)
        self.dimension = 5 if self.include_co2 else 4
        if not self.include_co2:
            self._co2 = None
            return
        root = Path(dataset_root)
        candidates = (co2_filename,) if co2_filename else self.CO2_FILENAMES
        for name in candidates:
            co2_path = root / name
            if co2_path.exists():
                break
        else:
            raise FileNotFoundError(
                f"no CO2 series in {root}; looked for {list(candidates)}"
            )
        with np.load(co2_path) as bundle:
            # The CO2 field is the largest array in the archive; reduce any
            # spatial dims to a global-mean series indexed by time.
            key = max(bundle.keys(), key=lambda name: bundle[name].size)
            field = np.asarray(bundle[key], dtype=np.float64)
        series = field.reshape(field.shape[0], -1).mean(axis=1) if field.ndim > 1 else field
        self._co2 = series
        self._co2_mean = float(series.mean())
        self._co2_std = float(series.std()) or 1.0

    def _co2_at(self, timestamp: float) -> float:
        index = int(round((timestamp - _CO2_EPOCH) / _CO2_STEP_SECONDS))
        index = min(max(index, 0), len(self._co2) - 1)
        return (self._co2[index] - self._co2_mean) / self._co2_std

    def vector(self, timestamp: float) -> np.ndarray:
        moment = datetime.fromtimestamp(int(timestamp), timezone.utc)
        year_start = datetime(moment.year, 1, 1, tzinfo=timezone.utc).timestamp()
        year_end = datetime(moment.year + 1, 1, 1, tzinfo=timezone.utc).timestamp()
        day_of_year = (timestamp - year_start) / (year_end - year_start)
        hour_of_day = moment.hour / 24.0
        tau = 2.0 * math.pi
        cyclic = [
            math.sin(tau * day_of_year),
            math.cos(tau * day_of_year),
            math.sin(tau * hour_of_day),
            math.cos(tau * hour_of_day),
        ]
        if not self.include_co2:
            return np.array(cyclic, dtype=np.float32)
        return np.array([self._co2_at(timestamp)] + cyclic, dtype=np.float32)


class ConditionedTokenPairDataset(TokenPairDataset):
    """TokenPairDataset that also yields the target frame's conditioning."""

    def __init__(
        self,
        token_root: str | Path,
        split: str,
        conditioner: ForcingConditioner,
        pair_lead_hours: int | None = None,
    ) -> None:
        super().__init__(token_root, split, pair_lead_hours=pair_lead_hours)
        self.conditioner = conditioner
        timestamps: dict[Path, np.ndarray] = {}
        for path in self.files:
            with h5py.File(path, "r") as handle:
                timestamps[path] = np.asarray(handle["timestamp"])
        if self.records:
            self.conditioning = np.stack(
                [
                    conditioner.vector(float(timestamps[path][target]))
                    for path, _source, target in self.records
                ]
            )
        else:
            self.conditioning = np.zeros(
                (0, conditioner.dimension), dtype=np.float32
            )

    def __getitem__(self, index: int):
        current, target = super().__getitem__(index)
        conditioning = torch.from_numpy(self.conditioning[index])
        return current, target, conditioning


def collate_conditioned_pairs(batch):
    current_items, target_items, conditioning_items = zip(*batch)
    current, target = collate_token_pairs(
        list(zip(current_items, target_items))
    )
    conditioning = torch.stack(list(conditioning_items))
    return current, target, conditioning
