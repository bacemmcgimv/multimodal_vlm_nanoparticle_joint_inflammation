from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(source: Path) -> Iterator[dict[str, str]]:
    with source.open("r", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            yield {str(key): str(value) for key, value in row.items()}


def validate(source: Path, destination: Path) -> None:
    required = (
        "participant_id",
        "volume_path",
        "site_id",
        "hoffa_grade",
        "effusion_grade",
        "bml_grade",
        "phrase",
    )
    accepted: list[dict[str, str]] = []
    identities: set[str] = set()
    for row in _rows(source):
        missing = [name for name in required if not row.get(name)]
        if missing:
            raise ValueError(f"row has empty fields: {missing}")
        identity = row["participant_id"]
        if identity in identities:
            raise ValueError(f"duplicate participant: {identity}")
        identities.add(identity)
        volume_path = Path(row["volume_path"])
        array = np.load(volume_path, mmap_mode="r", allow_pickle=False)
        if array.ndim != 4:
            raise ValueError(f"invalid shape for {volume_path}: {array.shape}")
        for field in ("hoffa_grade", "effusion_grade", "bml_grade"):
            if int(row[field]) < 0:
                raise ValueError(f"negative grade for {identity}")
        row["volume_sha256"] = _digest(volume_path)
        row["volume_shape"] = json.dumps(list(array.shape), separators=(",", ":"))
        accepted.append(row)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(accepted[0]))
        writer.writeheader()
        writer.writerows(accepted)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    validate(arguments.source, arguments.destination)


if __name__ == "__main__":
    main()
