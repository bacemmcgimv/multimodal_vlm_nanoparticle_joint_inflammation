from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from lamss_vlm.config import ExperimentConfig, load_config
from lamss_vlm.data import Batch, KneeVolumeDataset, VolumeTransform, collate_studies
from lamss_vlm.information import StableInformationEstimator
from lamss_vlm.model import LamssVLM
from lamss_vlm.objectives import LamssObjective
from lamss_vlm.runtime import (
    close_distributed,
    configure_logging,
    distributed_active,
    initialize_distributed,
    set_seed,
)
from lamss_vlm.trainer import Trainer, WarmupCosineScheduler


logger = logging.getLogger(__name__)


def _load_phrase_bank(path: Path) -> tuple[tuple[str, ...], Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("phrase bank must contain a mapping")
    phrases_raw = payload.get("phrases")
    embeddings = payload.get("embeddings")
    if not isinstance(phrases_raw, Sequence) or isinstance(phrases_raw, (str, bytes)):
        raise ValueError("phrase bank phrases are invalid")
    if not isinstance(embeddings, Tensor):
        raise ValueError("phrase bank embeddings are invalid")
    phrases = tuple(str(value) for value in phrases_raw)
    return phrases, embeddings


def _site_index(config: ExperimentConfig) -> dict[str, int]:
    sites = sorted(set(config.data.train_sites) | set(config.data.validation_sites))
    return {site: index for index, site in enumerate(sites)}


def _loader(
    config: ExperimentConfig,
    phrases: Sequence[str],
    sites: Mapping[str, int],
    training: bool,
) -> DataLoader[Batch]:
    transform = VolumeTransform(config.data.volume_shape, training=training)
    allowed = config.data.train_sites if training else config.data.validation_sites
    phrase_index = {phrase: index for index, phrase in enumerate(phrases)}
    dataset = KneeVolumeDataset(
        config.data.manifest,
        sites,
        phrase_index,
        transform,
        allowed,
    )
    sampler: torch.utils.data.Sampler[int] | None = None
    if distributed_active():
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            shuffle=training,
            drop_last=training,
        )
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=training and sampler is None,
        sampler=sampler,
        num_workers=config.data.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=training,
        persistent_workers=config.data.workers > 0,
        collate_fn=collate_studies,
    )


def _model(
    config: ExperimentConfig,
    phrases: Sequence[str],
    embeddings: Tensor,
) -> LamssVLM:
    if embeddings.shape[1] != config.model.phrase_dim:
        raise ValueError("configured phrase dimension does not match phrase bank")
    return LamssVLM(
        input_channels=config.data.channels,
        embed_dim=config.model.embed_dim,
        latent_dim=config.model.latent_dim,
        depths=config.model.depths,
        heads=config.model.heads,
        window=config.model.window_size,
        phrase_embeddings=embeddings,
        phrases=phrases,
        grade_counts=config.model.grade_counts,
    )


def _parameter_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1 or name.endswith("bias") or "relative_bias" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def run(config: ExperimentConfig) -> None:
    device = initialize_distributed()
    set_seed(config.training.seed)
    phrases, embeddings = _load_phrase_bank(config.model.phrase_bank)
    sites = _site_index(config)
    train_loader = _loader(config, phrases, sites, True)
    validation_loader = _loader(config, phrases, sites, False)
    model: nn.Module = _model(config, phrases, embeddings).to(device)
    site_estimator: nn.Module = StableInformationEstimator(
        config.model.latent_dim, len(sites)
    ).to(device)
    if distributed_active():
        device_ids = [device.index] if device.type == "cuda" else None
        model = nn.parallel.DistributedDataParallel(model, device_ids=device_ids)
        site_estimator = nn.parallel.DistributedDataParallel(site_estimator, device_ids=device_ids)
    optimizer = torch.optim.AdamW(
        _parameter_groups(model, config.training.weight_decay),
        lr=config.training.learning_rate,
    )
    site_optimizer = torch.optim.AdamW(
        site_estimator.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    updates_per_epoch = max(
        1,
        len(train_loader) // config.training.gradient_accumulation,
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        config.training.warmup_steps,
        config.training.epochs * updates_per_epoch,
    )
    feature_order = tuple(config.model.grade_counts)
    objective = LamssObjective(
        feature_order,
        config.model.grade_counts,
        config.training.beta_kl,
        config.training.beta_text,
        config.training.gamma_site,
    )
    trainer = Trainer(
        model,
        cast(StableInformationEstimator, site_estimator),
        objective,
        optimizer,
        site_optimizer,
        scheduler,
        device,
        config,
    )
    try:
        trainer.fit(train_loader, validation_loader)
    finally:
        close_distributed()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lamss-train")
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main() -> None:
    configure_logging()
    arguments = _parser().parse_args()
    config = load_config(arguments.config)
    logger.info("configuration=%s", json.dumps(str(config)))
    run(config)


if __name__ == "__main__":
    main()
