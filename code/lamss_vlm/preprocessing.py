from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as functional


@dataclass(frozen=True)
class SpatialGeometry:
    spacing: tuple[float, float, float]
    shape: tuple[int, int, int]


@dataclass(frozen=True)
class AugmentationState:
    flipped_axes: tuple[int, ...]
    polarity_channel: int | None
    bias_strength: float
    elastic_strength: float
    relaxivity: float


@dataclass(frozen=True)
class VolumeStatistics:
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    means: tuple[float, ...]
    deviations: tuple[float, ...]


def validate_volume(volume: Tensor) -> None:
    if volume.ndim != 4:
        raise ValueError("volume must have channel, depth, height and width dimensions")
    if any(size <= 0 for size in volume.shape):
        raise ValueError("volume dimensions must be positive")
    if not volume.is_floating_point():
        raise ValueError("volume must use a floating dtype")
    if not torch.isfinite(volume).all():
        raise ValueError("volume contains non-finite values")


def channel_percentiles(volume: Tensor, low: float = 1.0, high: float = 99.0) -> tuple[Tensor, Tensor]:
    validate_volume(volume)
    if not 0 <= low < high <= 100:
        raise ValueError("percentiles must be ordered inside zero and one hundred")
    flattened = volume.flatten(start_dim=1)
    lower = torch.quantile(flattened, low / 100.0, dim=1)
    upper = torch.quantile(flattened, high / 100.0, dim=1)
    return lower, upper


def percentile_clip(volume: Tensor, low: float = 1.0, high: float = 99.0) -> Tensor:
    lower, upper = channel_percentiles(volume, low, high)
    shape = (volume.shape[0], 1, 1, 1)
    return torch.maximum(torch.minimum(volume, upper.view(shape)), lower.view(shape))


def channel_zscore(volume: Tensor, epsilon: float = 1e-6) -> Tensor:
    validate_volume(volume)
    dimensions = (1, 2, 3)
    means = volume.mean(dim=dimensions, keepdim=True)
    deviations = volume.std(dim=dimensions, keepdim=True, unbiased=False).clamp_min(epsilon)
    return (volume - means) / deviations


def normalize_volume(volume: Tensor, low: float = 1.0, high: float = 99.0) -> Tensor:
    return channel_zscore(percentile_clip(volume.float(), low, high))


def target_grid(
    source_shape: Sequence[int],
    source_spacing: Sequence[float],
    target_spacing: Sequence[float],
) -> tuple[int, int, int]:
    if not (len(source_shape) == len(source_spacing) == len(target_spacing) == 3):
        raise ValueError("spatial inputs must each contain three values")
    result = []
    for size, old, new in zip(source_shape, source_spacing, target_spacing, strict=True):
        if size <= 0 or old <= 0 or new <= 0:
            raise ValueError("shape and spacing values must be positive")
        result.append(max(1, int(round(size * old / new))))
    return result[0], result[1], result[2]


def resample_volume(
    volume: Tensor,
    source_spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float] = (0.6, 0.6, 0.6),
) -> Tensor:
    validate_volume(volume)
    shape = target_grid(volume.shape[-3:], source_spacing, target_spacing)
    sampled = functional.interpolate(
        volume.unsqueeze(0), size=shape, mode="trilinear", align_corners=False
    )
    return sampled.squeeze(0)


def center_crop_or_pad(volume: Tensor, shape: tuple[int, int, int]) -> Tensor:
    validate_volume(volume)
    result = volume
    slices: list[slice] = [slice(None)]
    for current, desired in zip(result.shape[-3:], shape, strict=True):
        start = max(0, (current - desired) // 2)
        slices.append(slice(start, start + min(current, desired)))
    result = result[tuple(slices)]
    padding: list[int] = []
    for current, desired in reversed(list(zip(result.shape[-3:], shape, strict=True))):
        difference = max(0, desired - current)
        before = difference // 2
        padding.extend((before, difference - before))
    return functional.pad(result, padding)


def preprocess_volume(
    volume: Tensor,
    source_spacing: tuple[float, float, float],
    target: SpatialGeometry = SpatialGeometry((0.6, 0.6, 0.6), (160, 160, 160)),
) -> Tensor:
    sampled = resample_volume(volume, source_spacing, target.spacing)
    shaped = center_crop_or_pad(sampled, target.shape)
    return normalize_volume(shaped)


def random_flips(volume: Tensor, generator: torch.Generator, probability: float = 0.5) -> tuple[Tensor, tuple[int, ...]]:
    validate_volume(volume)
    axes: list[int] = []
    result = volume
    for axis in (1, 2, 3):
        if float(torch.rand((), generator=generator)) < probability:
            result = result.flip(axis)
            axes.append(axis)
    return result, tuple(axes)


def polarity_flip(
    volume: Tensor, generator: torch.Generator, probability: float = 0.25
) -> tuple[Tensor, int | None]:
    validate_volume(volume)
    if float(torch.rand((), generator=generator)) >= probability:
        return volume, None
    channel = int(torch.randint(volume.shape[0], (), generator=generator))
    result = volume.clone()
    result[channel] = -result[channel]
    return result, channel


def bias_field(
    volume: Tensor, generator: torch.Generator, maximum_strength: float = 0.3
) -> tuple[Tensor, float]:
    validate_volume(volume)
    strength = float(torch.rand((), generator=generator)) * maximum_strength
    depth, height, width = volume.shape[-3:]
    z = torch.linspace(-1.0, 1.0, depth, device=volume.device)
    y = torch.linspace(-1.0, 1.0, height, device=volume.device)
    x = torch.linspace(-1.0, 1.0, width, device=volume.device)
    coefficients = torch.randn(6, generator=generator, device=volume.device) * strength
    field = coefficients[0] * z[:, None, None]
    field = field + coefficients[1] * y[None, :, None]
    field = field + coefficients[2] * x[None, None, :]
    field = field + coefficients[3] * z[:, None, None] ** 2
    field = field + coefficients[4] * y[None, :, None] ** 2
    field = field + coefficients[5] * x[None, None, :] ** 2
    return volume * field.exp().unsqueeze(0), strength


def elastic_grid(
    shape: tuple[int, int, int],
    generator: torch.Generator,
    strength: float,
    device: torch.device,
) -> Tensor:
    depth, height, width = shape
    noise = torch.randn((1, 3, 5, 5, 5), generator=generator, device=device)
    displacement = functional.interpolate(
        noise, size=shape, mode="trilinear", align_corners=True
    ).permute(0, 2, 3, 4, 1)
    z, y, x = torch.meshgrid(
        torch.linspace(-1, 1, depth, device=device),
        torch.linspace(-1, 1, height, device=device),
        torch.linspace(-1, 1, width, device=device),
        indexing="ij",
    )
    base = torch.stack((x, y, z), dim=-1).unsqueeze(0)
    return (base + displacement * strength).clamp(-1.0, 1.0)


def elastic_deformation(
    volume: Tensor, generator: torch.Generator, maximum_strength: float = 0.08
) -> tuple[Tensor, float]:
    validate_volume(volume)
    strength = float(torch.rand((), generator=generator)) * maximum_strength
    grid = elastic_grid(volume.shape[-3:], generator, strength, volume.device)
    result = functional.grid_sample(
        volume.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    return result.squeeze(0), strength


def relaxivity_rescale(volume: Tensor, relaxivity: float, reference: float = 3.8) -> Tensor:
    validate_volume(volume)
    if relaxivity <= 0 or reference <= 0:
        raise ValueError("relaxivity values must be positive")
    factor = relaxivity / reference
    return torch.sign(volume) * torch.log1p(volume.abs() * factor) / np.log1p(factor)


def sample_relaxivity(generator: torch.Generator) -> float:
    choices = torch.tensor([3.8, 5.6, 9.2, 14.1, 19.0])
    index = int(torch.randint(choices.numel(), (), generator=generator))
    return float(choices[index])


def augment_volume(volume: Tensor, seed: int) -> tuple[Tensor, AugmentationState]:
    generator = torch.Generator(device=volume.device)
    generator.manual_seed(seed)
    result, axes = random_flips(volume, generator)
    result, channel = polarity_flip(result, generator)
    result, bias = bias_field(result, generator)
    result, elastic = elastic_deformation(result, generator)
    relaxivity = sample_relaxivity(generator)
    result = relaxivity_rescale(result, relaxivity)
    return normalize_volume(result), AugmentationState(axes, channel, bias, elastic, relaxivity)


def volume_statistics(volume: Tensor, low: float = 1.0, high: float = 99.0) -> VolumeStatistics:
    lower, upper = channel_percentiles(volume, low, high)
    flattened = volume.flatten(start_dim=1)
    means = flattened.mean(dim=1)
    deviations = flattened.std(dim=1, unbiased=False)
    return VolumeStatistics(
        tuple(float(value) for value in lower),
        tuple(float(value) for value in upper),
        tuple(float(value) for value in means),
        tuple(float(value) for value in deviations),
    )


def load_numpy_volume(path: str | Path) -> Tensor:
    array = np.load(Path(path), allow_pickle=False)
    if not isinstance(array, np.ndarray):
        raise ValueError("volume file does not contain an array")
    result = torch.from_numpy(np.asarray(array, dtype=np.float32))
    validate_volume(result)
    return result


def save_numpy_volume(path: str | Path, volume: Tensor) -> None:
    validate_volume(volume)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, volume.detach().cpu().numpy(), allow_pickle=False)
    temporary.replace(destination)
