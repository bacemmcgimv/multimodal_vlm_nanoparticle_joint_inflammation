from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import yaml


@dataclass(frozen=True)
class DataConfig:
    manifest: Path
    volume_shape: tuple[int, int, int]
    channels: int
    workers: int
    train_sites: tuple[str, ...]
    validation_sites: tuple[str, ...]


@dataclass(frozen=True)
class ModelConfig:
    embed_dim: int
    latent_dim: int
    depths: tuple[int, ...]
    heads: tuple[int, ...]
    window_size: tuple[int, int, int]
    phrase_dim: int
    phrase_bank: Path
    grade_counts: dict[str, int]


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: Path
    batch_size: int
    gradient_accumulation: int
    epochs: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    gradient_clip: float
    precision: str
    seed: int
    beta_kl: float
    beta_text: float
    gamma_site: float
    site_estimator_steps: int


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return cast(Mapping[str, Any], value)


def _required(mapping: Mapping[str, Any], key: str, section: str) -> Any:
    if key not in mapping or mapping[key] is None:
        raise ValueError(f"{section}.{key} is unresolved")
    return mapping[key]


def _positive_int(value: Any, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _positive_float(value: Any, name: str) -> float:
    result = float(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    result = float(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _int_tuple(value: Any, name: str, length: int | None = None) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    result = tuple(_positive_int(item, name) for item in value)
    if length is not None and len(result) != length:
        raise ValueError(f"{name} must contain {length} values")
    return result


def _str_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    result = tuple(str(item) for item in value)
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _parse_data(raw: Mapping[str, Any]) -> DataConfig:
    return DataConfig(
        manifest=Path(str(_required(raw, "manifest", "data"))),
        volume_shape=cast(
            tuple[int, int, int],
            _int_tuple(_required(raw, "volume_shape", "data"), "data.volume_shape", 3),
        ),
        channels=_positive_int(_required(raw, "channels", "data"), "data.channels"),
        workers=_nonnegative_int(_required(raw, "workers", "data"), "data.workers"),
        train_sites=_str_tuple(_required(raw, "train_sites", "data"), "data.train_sites"),
        validation_sites=_str_tuple(
            _required(raw, "validation_sites", "data"), "data.validation_sites"
        ),
    )


def _parse_model(raw: Mapping[str, Any]) -> ModelConfig:
    grade_raw = _mapping(_required(raw, "grade_counts", "model"), "model.grade_counts")
    grade_counts = {
        str(key): _positive_int(value, f"model.grade_counts.{key}")
        for key, value in grade_raw.items()
    }
    depths = _int_tuple(_required(raw, "depths", "model"), "model.depths")
    heads = _int_tuple(_required(raw, "heads", "model"), "model.heads")
    if len(depths) != len(heads):
        raise ValueError("model.depths and model.heads must have equal lengths")
    return ModelConfig(
        embed_dim=_positive_int(_required(raw, "embed_dim", "model"), "model.embed_dim"),
        latent_dim=_positive_int(_required(raw, "latent_dim", "model"), "model.latent_dim"),
        depths=depths,
        heads=heads,
        window_size=cast(
            tuple[int, int, int],
            _int_tuple(_required(raw, "window_size", "model"), "model.window_size", 3),
        ),
        phrase_dim=_positive_int(_required(raw, "phrase_dim", "model"), "model.phrase_dim"),
        phrase_bank=Path(str(_required(raw, "phrase_bank", "model"))),
        grade_counts=grade_counts,
    )


def _parse_training(raw: Mapping[str, Any]) -> TrainingConfig:
    precision = str(_required(raw, "precision", "training"))
    if precision not in {"float32", "float16", "bfloat16"}:
        raise ValueError("training.precision must be float32, float16, or bfloat16")
    return TrainingConfig(
        output_dir=Path(str(_required(raw, "output_dir", "training"))),
        batch_size=_positive_int(
            _required(raw, "batch_size", "training"), "training.batch_size"
        ),
        gradient_accumulation=_positive_int(
            _required(raw, "gradient_accumulation", "training"),
            "training.gradient_accumulation",
        ),
        epochs=_positive_int(_required(raw, "epochs", "training"), "training.epochs"),
        learning_rate=_positive_float(
            _required(raw, "learning_rate", "training"), "training.learning_rate"
        ),
        weight_decay=_nonnegative_float(
            _required(raw, "weight_decay", "training"), "training.weight_decay"
        ),
        warmup_steps=_nonnegative_int(
            _required(raw, "warmup_steps", "training"), "training.warmup_steps"
        ),
        gradient_clip=_positive_float(
            _required(raw, "gradient_clip", "training"), "training.gradient_clip"
        ),
        precision=precision,
        seed=int(_required(raw, "seed", "training")),
        beta_kl=_nonnegative_float(_required(raw, "beta_kl", "training"), "training.beta_kl"),
        beta_text=_nonnegative_float(
            _required(raw, "beta_text", "training"), "training.beta_text"
        ),
        gamma_site=_nonnegative_float(
            _required(raw, "gamma_site", "training"), "training.gamma_site"
        ),
        site_estimator_steps=_positive_int(
            _required(raw, "site_estimator_steps", "training"),
            "training.site_estimator_steps",
        ),
    )


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    root = _mapping(document, "configuration")
    return ExperimentConfig(
        data=_parse_data(_mapping(_required(root, "data", "configuration"), "data")),
        model=_parse_model(_mapping(_required(root, "model", "configuration"), "model")),
        training=_parse_training(
            _mapping(_required(root, "training", "configuration"), "training")
        ),
    )
