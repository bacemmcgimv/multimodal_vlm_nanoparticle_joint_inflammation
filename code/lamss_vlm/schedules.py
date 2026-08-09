from __future__ import annotations

from dataclasses import dataclass
from math import cos, exp, pi
from typing import Iterator, Sequence

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


@dataclass(frozen=True)
class SchedulePoint:
    epoch: int
    step: int
    learning_rate: float
    beta: float
    gamma: float
    vocabulary_size: int


@dataclass(frozen=True)
class TrainingDimensions:
    cases: int
    batch_size: int
    accumulation: int
    world_size: int
    epochs: int

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.accumulation * self.world_size

    @property
    def steps_per_epoch(self) -> int:
        return max(1, (self.cases + self.effective_batch_size - 1) // self.effective_batch_size)

    @property
    def optimizer_steps(self) -> int:
        return self.steps_per_epoch * self.epochs


def validate_dimensions(dimensions: TrainingDimensions) -> None:
    values = (
        dimensions.cases,
        dimensions.batch_size,
        dimensions.accumulation,
        dimensions.world_size,
        dimensions.epochs,
    )
    if any(value <= 0 for value in values):
        raise ValueError("training dimensions must be positive")


def cosine_learning_rate(
    step: int,
    total_steps: int,
    warmup_steps: int,
    maximum: float,
    minimum: float = 0.0,
) -> float:
    if total_steps <= 0 or maximum <= 0 or minimum < 0:
        raise ValueError("schedule values are invalid")
    if not 0 <= step <= total_steps:
        raise ValueError("step lies outside schedule")
    if warmup_steps < 0 or warmup_steps >= total_steps:
        raise ValueError("warmup must lie inside schedule")
    if step < warmup_steps:
        return maximum * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return minimum + 0.5 * (maximum - minimum) * (1.0 + cos(pi * progress))


def beta_weight(epoch: int) -> float:
    if epoch < 0:
        raise ValueError("epoch must be nonnegative")
    return 0.5 * (1.0 - exp(-epoch / 10.0))


def gamma_weight(epoch: int) -> float:
    if epoch < 0:
        raise ValueError("epoch must be nonnegative")
    return min(1.0, epoch / 20.0)


def vocabulary_size(epoch: int) -> int:
    if epoch < 0:
        raise ValueError("epoch must be nonnegative")
    if epoch < 30:
        return 10
    if epoch < 60:
        return 25
    return 50


def schedule_points(
    dimensions: TrainingDimensions,
    maximum_learning_rate: float = 3e-4,
    warmup_epochs: int = 2,
) -> tuple[SchedulePoint, ...]:
    validate_dimensions(dimensions)
    total_steps = dimensions.optimizer_steps
    warmup_steps = dimensions.steps_per_epoch * warmup_epochs
    points: list[SchedulePoint] = []
    for epoch in range(dimensions.epochs):
        for local_step in range(dimensions.steps_per_epoch):
            step = epoch * dimensions.steps_per_epoch + local_step
            points.append(
                SchedulePoint(
                    epoch,
                    step,
                    cosine_learning_rate(step, total_steps, warmup_steps, maximum_learning_rate),
                    beta_weight(epoch),
                    gamma_weight(epoch),
                    vocabulary_size(epoch),
                )
            )
    return tuple(points)


class WarmupCosineScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int,
        minimum_ratio: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if total_steps <= 0:
            raise ValueError("total steps must be positive")
        if warmup_steps < 0 or warmup_steps >= total_steps:
            raise ValueError("warmup steps must lie inside total steps")
        if not 0 <= minimum_ratio <= 1:
            raise ValueError("minimum ratio must lie between zero and one")
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.minimum_ratio = minimum_ratio
        super().__init__(optimizer, last_epoch)

    def ratio(self, step: int) -> float:
        if step < self.warmup_steps:
            return (step + 1) / max(1, self.warmup_steps)
        progress = min(1.0, (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps))
        return self.minimum_ratio + 0.5 * (1.0 - self.minimum_ratio) * (
            1.0 + cos(pi * progress)
        )

    def get_lr(self) -> list[float]:
        ratio = self.ratio(self.last_epoch)
        return [base_learning_rate * ratio for base_learning_rate in self.base_lrs]


def optimizer_groups(
    module: torch.nn.Module,
    weight_decay: float,
) -> list[dict[str, object]]:
    if weight_decay < 0:
        raise ValueError("weight decay must be nonnegative")
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1 or name.endswith("bias") or "norm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_adamw(
    module: torch.nn.Module,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-2,
) -> torch.optim.AdamW:
    if learning_rate <= 0:
        raise ValueError("learning rate must be positive")
    groups = optimizer_groups(module, weight_decay)
    return torch.optim.AdamW(groups, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)


def gradient_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    squares = []
    for parameter in parameters:
        if parameter.grad is not None:
            squares.append(parameter.grad.detach().float().square().sum())
    if not squares:
        return 0.0
    return float(torch.stack(squares).sum().sqrt())


def clip_gradients(module: torch.nn.Module, maximum_norm: float) -> float:
    if maximum_norm <= 0:
        raise ValueError("maximum norm must be positive")
    parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    return float(torch.nn.utils.clip_grad_norm_(parameters, maximum_norm))


def schedule_iterator(
    dimensions: TrainingDimensions,
    maximum_learning_rate: float = 3e-4,
    warmup_epochs: int = 2,
) -> Iterator[SchedulePoint]:
    yield from schedule_points(dimensions, maximum_learning_rate, warmup_epochs)


def apply_learning_rate(optimizer: Optimizer, learning_rate: float) -> None:
    if learning_rate < 0:
        raise ValueError("learning rate must be nonnegative")
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def current_learning_rates(optimizer: Optimizer) -> tuple[float, ...]:
    return tuple(float(group["lr"]) for group in optimizer.param_groups)


def seed_sequence(count: int = 20) -> tuple[int, ...]:
    if count <= 0:
        raise ValueError("seed count must be positive")
    return tuple(range(count))


def effective_batch_size(batch_size: int, accumulation: int, world_size: int) -> int:
    if batch_size <= 0 or accumulation <= 0 or world_size <= 0:
        raise ValueError("batch parameters must be positive")
    return batch_size * accumulation * world_size
