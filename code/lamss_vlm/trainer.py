from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import torch
from torch import Tensor, nn

from lamss_vlm.config import ExperimentConfig
from lamss_vlm.data import Batch
from lamss_vlm.information import StableInformationEstimator
from lamss_vlm.objectives import LamssObjective, LossBreakdown
from lamss_vlm.runtime import append_jsonl, atomic_torch_save, checkpoint_payload, primary_process


logger = logging.getLogger(__name__)


@dataclass
class RunningLosses:
    total: float = 0.0
    ordinal: float = 0.0
    kl: float = 0.0
    language: float = 0.0
    site: float = 0.0
    count: int = 0

    def update(self, losses: LossBreakdown, batch_size: int) -> None:
        self.total += float(losses.total.detach()) * batch_size
        self.ordinal += float(losses.ordinal.detach()) * batch_size
        self.kl += float(losses.kl.detach()) * batch_size
        self.language += float(losses.language.detach()) * batch_size
        self.site += float(losses.site.detach()) * batch_size
        self.count += batch_size

    def averages(self) -> dict[str, float]:
        divisor = max(1, self.count)
        return {
            "loss": self.total / divisor,
            "ordinal_loss": self.ordinal / divisor,
            "kl_loss": self.kl / divisor,
            "language_loss": self.language / divisor,
            "site_information": self.site / divisor,
        }


class WarmupCosineScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        step = max(0, self.last_epoch)
        if self.warmup_steps > 0 and step < self.warmup_steps:
            factor = (step + 1) / self.warmup_steps
        else:
            elapsed = step - self.warmup_steps
            duration = max(1, self.total_steps - self.warmup_steps)
            progress = min(1.0, elapsed / duration)
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [base_lr * factor for base_lr in self.base_lrs]


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        site_estimator: StableInformationEstimator,
        objective: LamssObjective,
        optimizer: torch.optim.Optimizer,
        site_optimizer: torch.optim.Optimizer,
        scheduler: WarmupCosineScheduler,
        device: torch.device,
        config: ExperimentConfig,
    ) -> None:
        self.model = model
        self.site_estimator = site_estimator
        self.objective = objective
        self.optimizer = optimizer
        self.site_optimizer = site_optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.global_step = 0
        precision = config.training.precision
        self.autocast_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[precision]
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=device.type == "cuda" and precision == "float16"
        )

    def _move(self, batch: Batch) -> Batch:
        return Batch(
            volumes=batch.volumes.to(self.device, non_blocking=True),
            sites=batch.sites.to(self.device, non_blocking=True),
            grades=batch.grades.to(self.device, non_blocking=True),
            phrase_indices=batch.phrase_indices.to(self.device, non_blocking=True),
            participant_ids=batch.participant_ids,
        )

    def _site_updates(self, latent: Tensor, sites: Tensor) -> None:
        for _ in range(self.config.training.site_estimator_steps):
            self.site_optimizer.zero_grad(set_to_none=True)
            estimator_loss = self.site_estimator.learning_loss(latent, sites)
            estimator_loss.backward()
            self.site_optimizer.step()

    def _forward(self, batch: Batch) -> LossBreakdown:
        output = self.model(batch.volumes, batch.phrase_indices)
        self._site_updates(output.latent.detach(), batch.sites)
        for parameter in self.site_estimator.parameters():
            parameter.requires_grad_(False)
        site_information = self.site_estimator(output.latent, batch.sites)
        losses = self.objective(output, batch.grades, site_information)
        for parameter in self.site_estimator.parameters():
            parameter.requires_grad_(True)
        return losses

    def train_epoch(self, batches: Iterable[Batch], epoch: int) -> Mapping[str, float]:
        self.model.train()
        self.site_estimator.train()
        self.objective.beta_text = 0.5 * (1.0 - math.exp(-epoch / 10.0))
        self.objective.gamma_site = min(1.0, epoch / 20.0)
        accumulated = self.config.training.gradient_accumulation
        running = RunningLosses()
        self.optimizer.zero_grad(set_to_none=True)
        for batch_index, raw_batch in enumerate(batches):
            batch = self._move(raw_batch)
            use_autocast = self.device.type == "cuda" and self.autocast_dtype != torch.float32
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.autocast_dtype,
                enabled=use_autocast,
            ):
                losses = self._forward(batch)
                scaled_loss = losses.total / accumulated
            self.scaler.scale(scaled_loss).backward()
            should_step = (batch_index + 1) % accumulated == 0
            if should_step:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.gradient_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.global_step += 1
            running.update(losses, batch.volumes.shape[0])
        metrics = running.averages()
        metrics["epoch"] = float(epoch)
        metrics["learning_rate"] = self.optimizer.param_groups[0]["lr"]
        return metrics

    @torch.no_grad()
    def validate(self, batches: Iterable[Batch]) -> Mapping[str, float]:
        self.model.eval()
        self.site_estimator.eval()
        running = RunningLosses()
        for raw_batch in batches:
            batch = self._move(raw_batch)
            output = self.model(batch.volumes, batch.phrase_indices)
            information = self.site_estimator(output.latent, batch.sites)
            losses = self.objective(output, batch.grades, information)
            running.update(losses, batch.volumes.shape[0])
        return {f"validation_{key}": value for key, value in running.averages().items()}

    def save(self, epoch: int, destination: Path) -> None:
        payload = checkpoint_payload(
            self.model,
            self.optimizer,
            self.site_estimator,
            self.site_optimizer,
            self.scheduler,
            epoch,
            self.global_step,
            self.config.training.seed,
            self.config,
        )
        atomic_torch_save(payload, destination)

    def fit(self, train_batches: Iterable[Batch], validation_batches: Iterable[Batch]) -> None:
        output = self.config.training.output_dir
        for epoch in range(self.config.training.epochs):
            training_metrics = self.train_epoch(train_batches, epoch)
            validation_metrics = self.validate(validation_batches)
            metrics = {**training_metrics, **validation_metrics}
            if primary_process():
                append_jsonl(output / "metrics.jsonl", metrics)
                self.save(epoch, output / "latest.pt")
                logger.info("epoch=%d metrics=%s", epoch, metrics)
