from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, cast

import torch
from torch import Tensor

from lamss_vlm.phrase_bank import PHRASES, validate_phrase_bank


MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def load_text_components() -> tuple[torch.nn.Module, Callable[[list[str]], Tensor]]:
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(MODEL)
    tokenizer = open_clip.get_tokenizer(MODEL)
    return cast(torch.nn.Module, model), cast(Callable[[list[str]], Tensor], tokenizer)


def encode_phrases(device: torch.device, batch_size: int) -> tuple[tuple[str, ...], Tensor]:
    validate_phrase_bank(PHRASES)
    model, tokenizer = load_text_components()
    model = model.to(device)
    model.eval()
    phrases = tuple(item.text for item in PHRASES)
    outputs: list[Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(phrases), batch_size):
            tokens = tokenizer(list(phrases[start : start + batch_size])).to(device)
            encoder = cast(Any, model)
            values = encoder.encode_text(tokens)
            outputs.append(torch.nn.functional.normalize(values.float(), dim=-1).cpu())
    return phrases, torch.cat(outputs, dim=0)


def atomic_save(payload: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def run(destination: Path, batch_size: int) -> None:
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phrases, embeddings = encode_phrases(device, batch_size)
    atomic_save(
        {
            "model": MODEL,
            "phrases": phrases,
            "embeddings": embeddings,
        },
        destination,
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="prepare-phrase-bank")
    value.add_argument("--output", type=Path, default=Path("data/moaks_phrases.pt"))
    value.add_argument("--batch-size", type=int, default=32)
    return value


def main() -> None:
    arguments = parser().parse_args()
    run(arguments.output, arguments.batch_size)


if __name__ == "__main__":
    main()
