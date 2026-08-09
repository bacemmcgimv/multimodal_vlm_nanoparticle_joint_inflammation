from __future__ import annotations

import json
import logging
import os
import random
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def distributed_active() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def rank() -> int:
    return torch.distributed.get_rank() if distributed_active() else 0


def world_size() -> int:
    return torch.distributed.get_world_size() if distributed_active() else 1


def primary_process() -> bool:
    return rank() == 0


def initialize_distributed() -> torch.device:
    if "RANK" not in os.environ:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    torch.distributed.init_process_group(backend=backend)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def close_distributed() -> None:
    if distributed_active():
        torch.distributed.destroy_process_group()


def reduce_mean(value: Tensor) -> Tensor:
    if not distributed_active():
        return value
    output = value.detach().clone()
    torch.distributed.all_reduce(output, op=torch.distributed.ReduceOp.SUM)
    return output / world_size()


def atomic_torch_save(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def unwrap(module: nn.Module) -> nn.Module:
    if isinstance(module, nn.parallel.DistributedDataParallel):
        return module.module
    return module


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    site_estimator: nn.Module,
    site_optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
    seed: int,
    config: object,
) -> dict[str, Any]:
    config_value = asdict(config) if is_dataclass(config) else config
    return {
        "model": unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "site_estimator": unwrap(site_estimator).state_dict(),
        "site_optimizer": site_optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "seed": seed,
        "config": config_value,
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }


def append_jsonl(destination: Path, values: Mapping[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(values), sort_keys=True) + "\n")
