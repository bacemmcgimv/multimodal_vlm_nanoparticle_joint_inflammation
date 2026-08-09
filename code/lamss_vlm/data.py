from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler


@dataclass(frozen=True)
class StudyRecord:
    participant_id: str
    volume_path: Path
    site_id: str
    grades: tuple[int, int, int]
    phrase: str


@dataclass(frozen=True)
class Batch:
    volumes: Tensor
    sites: Tensor
    grades: Tensor
    phrase_indices: Tensor
    participant_ids: tuple[str, ...]


class VolumeTransform:
    def __init__(
        self,
        output_shape: tuple[int, int, int],
        clip_percentiles: tuple[float, float] = (0.5, 99.5),
        flip_probability: float = 0.5,
        noise_std: float = 0.01,
        training: bool = False,
    ) -> None:
        self.output_shape = output_shape
        self.clip_percentiles = clip_percentiles
        self.flip_probability = flip_probability
        self.noise_std = noise_std
        self.training = training

    def _clip(self, volume: Tensor) -> Tensor:
        flat = volume.flatten(1)
        lower = torch.quantile(flat, self.clip_percentiles[0] / 100.0, dim=1)
        upper = torch.quantile(flat, self.clip_percentiles[1] / 100.0, dim=1)
        lower = lower[:, None, None, None]
        upper = upper[:, None, None, None]
        return volume.clamp(lower, upper)

    def _standardize(self, volume: Tensor) -> Tensor:
        mean = volume.mean(dim=(1, 2, 3), keepdim=True)
        std = volume.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
        return (volume - mean) / std

    def _fit_shape(self, volume: Tensor) -> Tensor:
        target_d, target_h, target_w = self.output_shape
        _, depth, height, width = volume.shape
        pad_d = max(0, target_d - depth)
        pad_h = max(0, target_h - height)
        pad_w = max(0, target_w - width)
        volume = torch.nn.functional.pad(
            volume,
            (
                pad_w // 2,
                pad_w - pad_w // 2,
                pad_h // 2,
                pad_h - pad_h // 2,
                pad_d // 2,
                pad_d - pad_d // 2,
            ),
        )
        _, depth, height, width = volume.shape
        start_d = (depth - target_d) // 2
        start_h = (height - target_h) // 2
        start_w = (width - target_w) // 2
        return volume[
            :,
            start_d : start_d + target_d,
            start_h : start_h + target_h,
            start_w : start_w + target_w,
        ]

    def _augment(self, volume: Tensor) -> Tensor:
        for axis in (1, 2, 3):
            if torch.rand(()) < self.flip_probability:
                volume = volume.flip(axis)
        if self.noise_std > 0:
            volume = volume + torch.randn_like(volume) * self.noise_std
        return volume

    def __call__(self, volume: Tensor) -> Tensor:
        if volume.ndim != 4:
            raise ValueError("volume must have shape C,D,H,W")
        volume = self._fit_shape(volume.float())
        volume = self._standardize(self._clip(volume))
        if self.training:
            volume = self._augment(volume)
        return volume.contiguous()


class KneeVolumeDataset(Dataset[Mapping[str, object]]):
    required_columns = {
        "participant_id",
        "volume_path",
        "site_id",
        "hoffa_grade",
        "effusion_grade",
        "bml_grade",
        "phrase",
    }

    def __init__(
        self,
        manifest: str | Path,
        site_index: Mapping[str, int],
        phrase_index: Mapping[str, int],
        transform: Callable[[Tensor], Tensor],
        allowed_sites: Sequence[str] | None = None,
    ) -> None:
        frame = pd.read_csv(manifest)
        missing = self.required_columns.difference(frame.columns)
        if missing:
            raise ValueError(f"manifest is missing columns: {sorted(missing)}")
        if allowed_sites is not None:
            frame = frame[frame["site_id"].isin(allowed_sites)]
        self.records = [self._record(row) for _, row in frame.iterrows()]
        self.site_index = dict(site_index)
        self.phrase_index = dict(phrase_index)
        self.transform = transform
        if not self.records:
            raise ValueError("manifest selection is empty")

    def _record(self, row: pd.Series) -> StudyRecord:
        grades = (int(row["hoffa_grade"]), int(row["effusion_grade"]), int(row["bml_grade"]))
        if min(grades) < 0:
            raise ValueError("grades must be nonnegative")
        return StudyRecord(
            participant_id=str(row["participant_id"]),
            volume_path=Path(str(row["volume_path"])),
            site_id=str(row["site_id"]),
            grades=grades,
            phrase=str(row["phrase"]),
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Mapping[str, object]:
        record = self.records[index]
        if record.site_id not in self.site_index:
            raise KeyError(f"unknown site: {record.site_id}")
        if record.phrase not in self.phrase_index:
            raise KeyError(f"unknown phrase: {record.phrase}")
        array = np.load(record.volume_path, allow_pickle=False)
        volume = self.transform(torch.from_numpy(array))
        return {
            "volume": volume,
            "site": self.site_index[record.site_id],
            "grades": torch.tensor(record.grades, dtype=torch.long),
            "phrase_index": self.phrase_index[record.phrase],
            "participant_id": record.participant_id,
        }


def collate_studies(items: Sequence[Mapping[str, object]]) -> Batch:
    volumes = torch.stack([item["volume"] for item in items if isinstance(item["volume"], Tensor)])
    grades = torch.stack([item["grades"] for item in items if isinstance(item["grades"], Tensor)])
    sites = torch.tensor([int(item["site"]) for item in items], dtype=torch.long)
    phrases = torch.tensor([int(item["phrase_index"]) for item in items], dtype=torch.long)
    identifiers = tuple(str(item["participant_id"]) for item in items)
    if len(volumes) != len(items) or len(grades) != len(items):
        raise TypeError("batch contains invalid tensors")
    return Batch(volumes, sites, grades, phrases, identifiers)


class SiteBalancedSampler(Sampler[int]):
    def __init__(self, site_labels: Sequence[int], seed: int) -> None:
        super().__init__()
        self.site_labels = tuple(site_labels)
        self.seed = seed
        self.epoch = 0
        self.groups: dict[int, list[int]] = {}
        for index, label in enumerate(self.site_labels):
            self.groups.setdefault(label, []).append(index)
        if not self.groups:
            raise ValueError("site labels must not be empty")

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        largest = max(len(indices) for indices in self.groups.values())
        selected: list[int] = []
        for indices in self.groups.values():
            source = torch.tensor(indices, dtype=torch.long)
            choices = torch.randint(len(indices), (largest,), generator=generator)
            selected.extend(source[choices].tolist())
        order = torch.randperm(len(selected), generator=generator).tolist()
        return iter(selected[index] for index in order)

    def __len__(self) -> int:
        return max(len(indices) for indices in self.groups.values()) * len(self.groups)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
