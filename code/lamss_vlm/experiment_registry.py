from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class RunSpec:
    identifier: str
    experiment: str
    seed: int
    configuration: str
    devices: int
    precision: str


RUNS: tuple[RunSpec, ...] = (
    RunSpec(
        "main_seed_00", "main", 0, "configs/main.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "main_seed_01", "main", 1, "configs/main.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "main_seed_02", "main", 2, "configs/main.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "main_seed_03", "main", 3, "configs/main.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "main_seed_04", "main", 4, "configs/main.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "main_seed_05", "main", 5, "configs/main.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "main_seed_06", "main", 6, "configs/main.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "main_seed_07", "main", 7, "configs/main.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "main_seed_08", "main", 8, "configs/main.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "main_seed_09", "main", 9, "configs/main.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "main_seed_10", "main", 10, "configs/main.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "main_seed_11", "main", 11, "configs/main.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "main_seed_12", "main", 12, "configs/main.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "main_seed_13", "main", 13, "configs/main.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "main_seed_14", "main", 14, "configs/main.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "main_seed_15", "main", 15, "configs/main.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "main_seed_16", "main", 16, "configs/main.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "main_seed_17", "main", 17, "configs/main.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "main_seed_18", "main", 18, "configs/main.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "main_seed_19", "main", 19, "configs/main.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_curriculum_seed_00", "ablation_no_curriculum", 0, "configs/ablation_no_curriculum.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_curriculum_seed_01", "ablation_no_curriculum", 1, "configs/ablation_no_curriculum.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_curriculum_seed_02", "ablation_no_curriculum", 2, "configs/ablation_no_curriculum.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_curriculum_seed_03", "ablation_no_curriculum", 3, "configs/ablation_no_curriculum.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_curriculum_seed_04", "ablation_no_curriculum", 4, "configs/ablation_no_curriculum.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_curriculum_seed_05", "ablation_no_curriculum", 5, "configs/ablation_no_curriculum.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_curriculum_seed_06", "ablation_no_curriculum", 6, "configs/ablation_no_curriculum.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_curriculum_seed_07", "ablation_no_curriculum", 7, "configs/ablation_no_curriculum.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_curriculum_seed_08", "ablation_no_curriculum", 8, "configs/ablation_no_curriculum.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_curriculum_seed_09", "ablation_no_curriculum", 9, "configs/ablation_no_curriculum.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_curriculum_seed_10", "ablation_no_curriculum", 10, "configs/ablation_no_curriculum.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_curriculum_seed_11", "ablation_no_curriculum", 11, "configs/ablation_no_curriculum.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_curriculum_seed_12", "ablation_no_curriculum", 12, "configs/ablation_no_curriculum.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_curriculum_seed_13", "ablation_no_curriculum", 13, "configs/ablation_no_curriculum.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_curriculum_seed_14", "ablation_no_curriculum", 14, "configs/ablation_no_curriculum.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_curriculum_seed_15", "ablation_no_curriculum", 15, "configs/ablation_no_curriculum.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_curriculum_seed_16", "ablation_no_curriculum", 16, "configs/ablation_no_curriculum.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_curriculum_seed_17", "ablation_no_curriculum", 17, "configs/ablation_no_curriculum.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_curriculum_seed_18", "ablation_no_curriculum", 18, "configs/ablation_no_curriculum.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_curriculum_seed_19", "ablation_no_curriculum", 19, "configs/ablation_no_curriculum.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_language_seed_00", "ablation_no_language", 0, "configs/ablation_no_language.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_language_seed_01", "ablation_no_language", 1, "configs/ablation_no_language.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_language_seed_02", "ablation_no_language", 2, "configs/ablation_no_language.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_language_seed_03", "ablation_no_language", 3, "configs/ablation_no_language.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_language_seed_04", "ablation_no_language", 4, "configs/ablation_no_language.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_language_seed_05", "ablation_no_language", 5, "configs/ablation_no_language.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_language_seed_06", "ablation_no_language", 6, "configs/ablation_no_language.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_language_seed_07", "ablation_no_language", 7, "configs/ablation_no_language.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_language_seed_08", "ablation_no_language", 8, "configs/ablation_no_language.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_language_seed_09", "ablation_no_language", 9, "configs/ablation_no_language.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_language_seed_10", "ablation_no_language", 10, "configs/ablation_no_language.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_language_seed_11", "ablation_no_language", 11, "configs/ablation_no_language.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_language_seed_12", "ablation_no_language", 12, "configs/ablation_no_language.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_language_seed_13", "ablation_no_language", 13, "configs/ablation_no_language.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_language_seed_14", "ablation_no_language", 14, "configs/ablation_no_language.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_language_seed_15", "ablation_no_language", 15, "configs/ablation_no_language.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_language_seed_16", "ablation_no_language", 16, "configs/ablation_no_language.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_language_seed_17", "ablation_no_language", 17, "configs/ablation_no_language.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_language_seed_18", "ablation_no_language", 18, "configs/ablation_no_language.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_language_seed_19", "ablation_no_language", 19, "configs/ablation_no_language.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_language_no_site_seed_00", "ablation_no_language_no_site", 0, "configs/ablation_no_language_no_site.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_language_no_site_seed_01", "ablation_no_language_no_site", 1, "configs/ablation_no_language_no_site.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_language_no_site_seed_02", "ablation_no_language_no_site", 2, "configs/ablation_no_language_no_site.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_language_no_site_seed_03", "ablation_no_language_no_site", 3, "configs/ablation_no_language_no_site.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_language_no_site_seed_04", "ablation_no_language_no_site", 4, "configs/ablation_no_language_no_site.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_language_no_site_seed_05", "ablation_no_language_no_site", 5, "configs/ablation_no_language_no_site.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_language_no_site_seed_06", "ablation_no_language_no_site", 6, "configs/ablation_no_language_no_site.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_language_no_site_seed_07", "ablation_no_language_no_site", 7, "configs/ablation_no_language_no_site.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_language_no_site_seed_08", "ablation_no_language_no_site", 8, "configs/ablation_no_language_no_site.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_language_no_site_seed_09", "ablation_no_language_no_site", 9, "configs/ablation_no_language_no_site.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_language_no_site_seed_10", "ablation_no_language_no_site", 10, "configs/ablation_no_language_no_site.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_language_no_site_seed_11", "ablation_no_language_no_site", 11, "configs/ablation_no_language_no_site.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_language_no_site_seed_12", "ablation_no_language_no_site", 12, "configs/ablation_no_language_no_site.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_language_no_site_seed_13", "ablation_no_language_no_site", 13, "configs/ablation_no_language_no_site.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_language_no_site_seed_14", "ablation_no_language_no_site", 14, "configs/ablation_no_language_no_site.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_language_no_site_seed_15", "ablation_no_language_no_site", 15, "configs/ablation_no_language_no_site.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_language_no_site_seed_16", "ablation_no_language_no_site", 16, "configs/ablation_no_language_no_site.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_language_no_site_seed_17", "ablation_no_language_no_site", 17, "configs/ablation_no_language_no_site.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_language_no_site_seed_18", "ablation_no_language_no_site", 18, "configs/ablation_no_language_no_site.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_language_no_site_seed_19", "ablation_no_language_no_site", 19, "configs/ablation_no_language_no_site.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_site_seed_00", "ablation_no_site", 0, "configs/ablation_no_site.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_site_seed_01", "ablation_no_site", 1, "configs/ablation_no_site.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_site_seed_02", "ablation_no_site", 2, "configs/ablation_no_site.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_site_seed_03", "ablation_no_site", 3, "configs/ablation_no_site.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_site_seed_04", "ablation_no_site", 4, "configs/ablation_no_site.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_site_seed_05", "ablation_no_site", 5, "configs/ablation_no_site.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_site_seed_06", "ablation_no_site", 6, "configs/ablation_no_site.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_site_seed_07", "ablation_no_site", 7, "configs/ablation_no_site.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_site_seed_08", "ablation_no_site", 8, "configs/ablation_no_site.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_site_seed_09", "ablation_no_site", 9, "configs/ablation_no_site.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_site_seed_10", "ablation_no_site", 10, "configs/ablation_no_site.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_site_seed_11", "ablation_no_site", 11, "configs/ablation_no_site.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_site_seed_12", "ablation_no_site", 12, "configs/ablation_no_site.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_site_seed_13", "ablation_no_site", 13, "configs/ablation_no_site.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_site_seed_14", "ablation_no_site", 14, "configs/ablation_no_site.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_site_seed_15", "ablation_no_site", 15, "configs/ablation_no_site.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_site_seed_16", "ablation_no_site", 16, "configs/ablation_no_site.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_site_seed_17", "ablation_no_site", 17, "configs/ablation_no_site.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_site_seed_18", "ablation_no_site", 18, "configs/ablation_no_site.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_no_site_seed_19", "ablation_no_site", 19, "configs/ablation_no_site.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_softmax_seed_00", "ablation_softmax", 0, "configs/ablation_softmax.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_softmax_seed_01", "ablation_softmax", 1, "configs/ablation_softmax.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_softmax_seed_02", "ablation_softmax", 2, "configs/ablation_softmax.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_softmax_seed_03", "ablation_softmax", 3, "configs/ablation_softmax.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_softmax_seed_04", "ablation_softmax", 4, "configs/ablation_softmax.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_softmax_seed_05", "ablation_softmax", 5, "configs/ablation_softmax.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_softmax_seed_06", "ablation_softmax", 6, "configs/ablation_softmax.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_softmax_seed_07", "ablation_softmax", 7, "configs/ablation_softmax.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_softmax_seed_08", "ablation_softmax", 8, "configs/ablation_softmax.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_softmax_seed_09", "ablation_softmax", 9, "configs/ablation_softmax.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_softmax_seed_10", "ablation_softmax", 10, "configs/ablation_softmax.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_softmax_seed_11", "ablation_softmax", 11, "configs/ablation_softmax.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_softmax_seed_12", "ablation_softmax", 12, "configs/ablation_softmax.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_softmax_seed_13", "ablation_softmax", 13, "configs/ablation_softmax.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_softmax_seed_14", "ablation_softmax", 14, "configs/ablation_softmax.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_softmax_seed_15", "ablation_softmax", 15, "configs/ablation_softmax.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_softmax_seed_16", "ablation_softmax", 16, "configs/ablation_softmax.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_softmax_seed_17", "ablation_softmax", 17, "configs/ablation_softmax.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_softmax_seed_18", "ablation_softmax", 18, "configs/ablation_softmax.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_softmax_seed_19", "ablation_softmax", 19, "configs/ablation_softmax.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_10_seed_00", "ablation_vocab_10", 0, "configs/ablation_vocab_10.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_10_seed_01", "ablation_vocab_10", 1, "configs/ablation_vocab_10.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_10_seed_02", "ablation_vocab_10", 2, "configs/ablation_vocab_10.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_10_seed_03", "ablation_vocab_10", 3, "configs/ablation_vocab_10.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_10_seed_04", "ablation_vocab_10", 4, "configs/ablation_vocab_10.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_10_seed_05", "ablation_vocab_10", 5, "configs/ablation_vocab_10.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_10_seed_06", "ablation_vocab_10", 6, "configs/ablation_vocab_10.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_10_seed_07", "ablation_vocab_10", 7, "configs/ablation_vocab_10.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_10_seed_08", "ablation_vocab_10", 8, "configs/ablation_vocab_10.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_10_seed_09", "ablation_vocab_10", 9, "configs/ablation_vocab_10.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_10_seed_10", "ablation_vocab_10", 10, "configs/ablation_vocab_10.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_10_seed_11", "ablation_vocab_10", 11, "configs/ablation_vocab_10.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_10_seed_12", "ablation_vocab_10", 12, "configs/ablation_vocab_10.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_10_seed_13", "ablation_vocab_10", 13, "configs/ablation_vocab_10.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_10_seed_14", "ablation_vocab_10", 14, "configs/ablation_vocab_10.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_10_seed_15", "ablation_vocab_10", 15, "configs/ablation_vocab_10.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_10_seed_16", "ablation_vocab_10", 16, "configs/ablation_vocab_10.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_10_seed_17", "ablation_vocab_10", 17, "configs/ablation_vocab_10.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_10_seed_18", "ablation_vocab_10", 18, "configs/ablation_vocab_10.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_10_seed_19", "ablation_vocab_10", 19, "configs/ablation_vocab_10.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_25_seed_00", "ablation_vocab_25", 0, "configs/ablation_vocab_25.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_25_seed_01", "ablation_vocab_25", 1, "configs/ablation_vocab_25.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_25_seed_02", "ablation_vocab_25", 2, "configs/ablation_vocab_25.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_25_seed_03", "ablation_vocab_25", 3, "configs/ablation_vocab_25.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_25_seed_04", "ablation_vocab_25", 4, "configs/ablation_vocab_25.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_25_seed_05", "ablation_vocab_25", 5, "configs/ablation_vocab_25.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_25_seed_06", "ablation_vocab_25", 6, "configs/ablation_vocab_25.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_25_seed_07", "ablation_vocab_25", 7, "configs/ablation_vocab_25.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_25_seed_08", "ablation_vocab_25", 8, "configs/ablation_vocab_25.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_25_seed_09", "ablation_vocab_25", 9, "configs/ablation_vocab_25.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_25_seed_10", "ablation_vocab_25", 10, "configs/ablation_vocab_25.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_25_seed_11", "ablation_vocab_25", 11, "configs/ablation_vocab_25.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_25_seed_12", "ablation_vocab_25", 12, "configs/ablation_vocab_25.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_25_seed_13", "ablation_vocab_25", 13, "configs/ablation_vocab_25.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_25_seed_14", "ablation_vocab_25", 14, "configs/ablation_vocab_25.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_25_seed_15", "ablation_vocab_25", 15, "configs/ablation_vocab_25.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_25_seed_16", "ablation_vocab_25", 16, "configs/ablation_vocab_25.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_25_seed_17", "ablation_vocab_25", 17, "configs/ablation_vocab_25.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_25_seed_18", "ablation_vocab_25", 18, "configs/ablation_vocab_25.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_25_seed_19", "ablation_vocab_25", 19, "configs/ablation_vocab_25.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_50_seed_00", "ablation_vocab_50", 0, "configs/ablation_vocab_50.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_50_seed_01", "ablation_vocab_50", 1, "configs/ablation_vocab_50.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_50_seed_02", "ablation_vocab_50", 2, "configs/ablation_vocab_50.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_50_seed_03", "ablation_vocab_50", 3, "configs/ablation_vocab_50.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_50_seed_04", "ablation_vocab_50", 4, "configs/ablation_vocab_50.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_50_seed_05", "ablation_vocab_50", 5, "configs/ablation_vocab_50.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_50_seed_06", "ablation_vocab_50", 6, "configs/ablation_vocab_50.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_50_seed_07", "ablation_vocab_50", 7, "configs/ablation_vocab_50.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_50_seed_08", "ablation_vocab_50", 8, "configs/ablation_vocab_50.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_50_seed_09", "ablation_vocab_50", 9, "configs/ablation_vocab_50.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_50_seed_10", "ablation_vocab_50", 10, "configs/ablation_vocab_50.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_50_seed_11", "ablation_vocab_50", 11, "configs/ablation_vocab_50.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_50_seed_12", "ablation_vocab_50", 12, "configs/ablation_vocab_50.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_50_seed_13", "ablation_vocab_50", 13, "configs/ablation_vocab_50.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_50_seed_14", "ablation_vocab_50", 14, "configs/ablation_vocab_50.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_50_seed_15", "ablation_vocab_50", 15, "configs/ablation_vocab_50.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_50_seed_16", "ablation_vocab_50", 16, "configs/ablation_vocab_50.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_50_seed_17", "ablation_vocab_50", 17, "configs/ablation_vocab_50.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_50_seed_18", "ablation_vocab_50", 18, "configs/ablation_vocab_50.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ablation_vocab_50_seed_19", "ablation_vocab_50", 19, "configs/ablation_vocab_50.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "beta_sweep_seed_00", "beta_sweep", 0, "configs/beta_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "beta_sweep_seed_01", "beta_sweep", 1, "configs/beta_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "beta_sweep_seed_02", "beta_sweep", 2, "configs/beta_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "beta_sweep_seed_03", "beta_sweep", 3, "configs/beta_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "beta_sweep_seed_04", "beta_sweep", 4, "configs/beta_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "beta_sweep_seed_05", "beta_sweep", 5, "configs/beta_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "beta_sweep_seed_06", "beta_sweep", 6, "configs/beta_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "beta_sweep_seed_07", "beta_sweep", 7, "configs/beta_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "beta_sweep_seed_08", "beta_sweep", 8, "configs/beta_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "beta_sweep_seed_09", "beta_sweep", 9, "configs/beta_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "beta_sweep_seed_10", "beta_sweep", 10, "configs/beta_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "beta_sweep_seed_11", "beta_sweep", 11, "configs/beta_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "beta_sweep_seed_12", "beta_sweep", 12, "configs/beta_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "beta_sweep_seed_13", "beta_sweep", 13, "configs/beta_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "beta_sweep_seed_14", "beta_sweep", 14, "configs/beta_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "beta_sweep_seed_15", "beta_sweep", 15, "configs/beta_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "beta_sweep_seed_16", "beta_sweep", 16, "configs/beta_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "beta_sweep_seed_17", "beta_sweep", 17, "configs/beta_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "beta_sweep_seed_18", "beta_sweep", 18, "configs/beta_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "beta_sweep_seed_19", "beta_sweep", 19, "configs/beta_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_1000_seed_00", "cohort_1000", 0, "configs/cohort_1000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_1000_seed_01", "cohort_1000", 1, "configs/cohort_1000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_1000_seed_02", "cohort_1000", 2, "configs/cohort_1000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_1000_seed_03", "cohort_1000", 3, "configs/cohort_1000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_1000_seed_04", "cohort_1000", 4, "configs/cohort_1000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_1000_seed_05", "cohort_1000", 5, "configs/cohort_1000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_1000_seed_06", "cohort_1000", 6, "configs/cohort_1000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_1000_seed_07", "cohort_1000", 7, "configs/cohort_1000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_1000_seed_08", "cohort_1000", 8, "configs/cohort_1000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_1000_seed_09", "cohort_1000", 9, "configs/cohort_1000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_1000_seed_10", "cohort_1000", 10, "configs/cohort_1000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_1000_seed_11", "cohort_1000", 11, "configs/cohort_1000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_1000_seed_12", "cohort_1000", 12, "configs/cohort_1000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_1000_seed_13", "cohort_1000", 13, "configs/cohort_1000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_1000_seed_14", "cohort_1000", 14, "configs/cohort_1000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_1000_seed_15", "cohort_1000", 15, "configs/cohort_1000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_1000_seed_16", "cohort_1000", 16, "configs/cohort_1000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_1000_seed_17", "cohort_1000", 17, "configs/cohort_1000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_1000_seed_18", "cohort_1000", 18, "configs/cohort_1000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_1000_seed_19", "cohort_1000", 19, "configs/cohort_1000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_2000_seed_00", "cohort_2000", 0, "configs/cohort_2000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_2000_seed_01", "cohort_2000", 1, "configs/cohort_2000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_2000_seed_02", "cohort_2000", 2, "configs/cohort_2000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_2000_seed_03", "cohort_2000", 3, "configs/cohort_2000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_2000_seed_04", "cohort_2000", 4, "configs/cohort_2000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_2000_seed_05", "cohort_2000", 5, "configs/cohort_2000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_2000_seed_06", "cohort_2000", 6, "configs/cohort_2000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_2000_seed_07", "cohort_2000", 7, "configs/cohort_2000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_2000_seed_08", "cohort_2000", 8, "configs/cohort_2000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_2000_seed_09", "cohort_2000", 9, "configs/cohort_2000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_2000_seed_10", "cohort_2000", 10, "configs/cohort_2000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_2000_seed_11", "cohort_2000", 11, "configs/cohort_2000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_2000_seed_12", "cohort_2000", 12, "configs/cohort_2000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_2000_seed_13", "cohort_2000", 13, "configs/cohort_2000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_2000_seed_14", "cohort_2000", 14, "configs/cohort_2000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_2000_seed_15", "cohort_2000", 15, "configs/cohort_2000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_2000_seed_16", "cohort_2000", 16, "configs/cohort_2000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_2000_seed_17", "cohort_2000", 17, "configs/cohort_2000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_2000_seed_18", "cohort_2000", 18, "configs/cohort_2000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_2000_seed_19", "cohort_2000", 19, "configs/cohort_2000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_4000_seed_00", "cohort_4000", 0, "configs/cohort_4000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_4000_seed_01", "cohort_4000", 1, "configs/cohort_4000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_4000_seed_02", "cohort_4000", 2, "configs/cohort_4000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_4000_seed_03", "cohort_4000", 3, "configs/cohort_4000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_4000_seed_04", "cohort_4000", 4, "configs/cohort_4000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_4000_seed_05", "cohort_4000", 5, "configs/cohort_4000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_4000_seed_06", "cohort_4000", 6, "configs/cohort_4000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_4000_seed_07", "cohort_4000", 7, "configs/cohort_4000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_4000_seed_08", "cohort_4000", 8, "configs/cohort_4000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_4000_seed_09", "cohort_4000", 9, "configs/cohort_4000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_4000_seed_10", "cohort_4000", 10, "configs/cohort_4000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_4000_seed_11", "cohort_4000", 11, "configs/cohort_4000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_4000_seed_12", "cohort_4000", 12, "configs/cohort_4000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_4000_seed_13", "cohort_4000", 13, "configs/cohort_4000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_4000_seed_14", "cohort_4000", 14, "configs/cohort_4000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_4000_seed_15", "cohort_4000", 15, "configs/cohort_4000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_4000_seed_16", "cohort_4000", 16, "configs/cohort_4000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_4000_seed_17", "cohort_4000", 17, "configs/cohort_4000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_4000_seed_18", "cohort_4000", 18, "configs/cohort_4000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_4000_seed_19", "cohort_4000", 19, "configs/cohort_4000.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_500_seed_00", "cohort_500", 0, "configs/cohort_500.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_500_seed_01", "cohort_500", 1, "configs/cohort_500.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_500_seed_02", "cohort_500", 2, "configs/cohort_500.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_500_seed_03", "cohort_500", 3, "configs/cohort_500.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_500_seed_04", "cohort_500", 4, "configs/cohort_500.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_500_seed_05", "cohort_500", 5, "configs/cohort_500.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_500_seed_06", "cohort_500", 6, "configs/cohort_500.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_500_seed_07", "cohort_500", 7, "configs/cohort_500.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_500_seed_08", "cohort_500", 8, "configs/cohort_500.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_500_seed_09", "cohort_500", 9, "configs/cohort_500.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_500_seed_10", "cohort_500", 10, "configs/cohort_500.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_500_seed_11", "cohort_500", 11, "configs/cohort_500.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_500_seed_12", "cohort_500", 12, "configs/cohort_500.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_500_seed_13", "cohort_500", 13, "configs/cohort_500.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_500_seed_14", "cohort_500", 14, "configs/cohort_500.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_500_seed_15", "cohort_500", 15, "configs/cohort_500.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_500_seed_16", "cohort_500", 16, "configs/cohort_500.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_500_seed_17", "cohort_500", 17, "configs/cohort_500.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_500_seed_18", "cohort_500", 18, "configs/cohort_500.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "cohort_500_seed_19", "cohort_500", 19, "configs/cohort_500.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "contrast_polarity_seed_00", "contrast_polarity", 0, "configs/contrast_polarity.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "contrast_polarity_seed_01", "contrast_polarity", 1, "configs/contrast_polarity.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "contrast_polarity_seed_02", "contrast_polarity", 2, "configs/contrast_polarity.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "contrast_polarity_seed_03", "contrast_polarity", 3, "configs/contrast_polarity.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "contrast_polarity_seed_04", "contrast_polarity", 4, "configs/contrast_polarity.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "contrast_polarity_seed_05", "contrast_polarity", 5, "configs/contrast_polarity.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "contrast_polarity_seed_06", "contrast_polarity", 6, "configs/contrast_polarity.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "contrast_polarity_seed_07", "contrast_polarity", 7, "configs/contrast_polarity.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "contrast_polarity_seed_08", "contrast_polarity", 8, "configs/contrast_polarity.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "contrast_polarity_seed_09", "contrast_polarity", 9, "configs/contrast_polarity.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "contrast_polarity_seed_10", "contrast_polarity", 10, "configs/contrast_polarity.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "contrast_polarity_seed_11", "contrast_polarity", 11, "configs/contrast_polarity.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "contrast_polarity_seed_12", "contrast_polarity", 12, "configs/contrast_polarity.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "contrast_polarity_seed_13", "contrast_polarity", 13, "configs/contrast_polarity.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "contrast_polarity_seed_14", "contrast_polarity", 14, "configs/contrast_polarity.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "contrast_polarity_seed_15", "contrast_polarity", 15, "configs/contrast_polarity.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "contrast_polarity_seed_16", "contrast_polarity", 16, "configs/contrast_polarity.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "contrast_polarity_seed_17", "contrast_polarity", 17, "configs/contrast_polarity.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "contrast_polarity_seed_18", "contrast_polarity", 18, "configs/contrast_polarity.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "contrast_polarity_seed_19", "contrast_polarity", 19, "configs/contrast_polarity.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "contrast_relaxivity_seed_00", "contrast_relaxivity", 0, "configs/contrast_relaxivity.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "contrast_relaxivity_seed_01", "contrast_relaxivity", 1, "configs/contrast_relaxivity.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "contrast_relaxivity_seed_02", "contrast_relaxivity", 2, "configs/contrast_relaxivity.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "contrast_relaxivity_seed_03", "contrast_relaxivity", 3, "configs/contrast_relaxivity.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "contrast_relaxivity_seed_04", "contrast_relaxivity", 4, "configs/contrast_relaxivity.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "contrast_relaxivity_seed_05", "contrast_relaxivity", 5, "configs/contrast_relaxivity.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "contrast_relaxivity_seed_06", "contrast_relaxivity", 6, "configs/contrast_relaxivity.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "contrast_relaxivity_seed_07", "contrast_relaxivity", 7, "configs/contrast_relaxivity.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "contrast_relaxivity_seed_08", "contrast_relaxivity", 8, "configs/contrast_relaxivity.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "contrast_relaxivity_seed_09", "contrast_relaxivity", 9, "configs/contrast_relaxivity.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "contrast_relaxivity_seed_10", "contrast_relaxivity", 10, "configs/contrast_relaxivity.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "contrast_relaxivity_seed_11", "contrast_relaxivity", 11, "configs/contrast_relaxivity.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "contrast_relaxivity_seed_12", "contrast_relaxivity", 12, "configs/contrast_relaxivity.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "contrast_relaxivity_seed_13", "contrast_relaxivity", 13, "configs/contrast_relaxivity.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "contrast_relaxivity_seed_14", "contrast_relaxivity", 14, "configs/contrast_relaxivity.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "contrast_relaxivity_seed_15", "contrast_relaxivity", 15, "configs/contrast_relaxivity.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "contrast_relaxivity_seed_16", "contrast_relaxivity", 16, "configs/contrast_relaxivity.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "contrast_relaxivity_seed_17", "contrast_relaxivity", 17, "configs/contrast_relaxivity.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "contrast_relaxivity_seed_18", "contrast_relaxivity", 18, "configs/contrast_relaxivity.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "contrast_relaxivity_seed_19", "contrast_relaxivity", 19, "configs/contrast_relaxivity.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_10_seed_00", "ensemble_10", 0, "configs/ensemble_10.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_10_seed_01", "ensemble_10", 1, "configs/ensemble_10.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_10_seed_02", "ensemble_10", 2, "configs/ensemble_10.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_10_seed_03", "ensemble_10", 3, "configs/ensemble_10.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_10_seed_04", "ensemble_10", 4, "configs/ensemble_10.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_10_seed_05", "ensemble_10", 5, "configs/ensemble_10.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_10_seed_06", "ensemble_10", 6, "configs/ensemble_10.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_10_seed_07", "ensemble_10", 7, "configs/ensemble_10.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_10_seed_08", "ensemble_10", 8, "configs/ensemble_10.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_10_seed_09", "ensemble_10", 9, "configs/ensemble_10.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_10_seed_10", "ensemble_10", 10, "configs/ensemble_10.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_10_seed_11", "ensemble_10", 11, "configs/ensemble_10.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_10_seed_12", "ensemble_10", 12, "configs/ensemble_10.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_10_seed_13", "ensemble_10", 13, "configs/ensemble_10.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_10_seed_14", "ensemble_10", 14, "configs/ensemble_10.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_10_seed_15", "ensemble_10", 15, "configs/ensemble_10.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_10_seed_16", "ensemble_10", 16, "configs/ensemble_10.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_10_seed_17", "ensemble_10", 17, "configs/ensemble_10.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_10_seed_18", "ensemble_10", 18, "configs/ensemble_10.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_10_seed_19", "ensemble_10", 19, "configs/ensemble_10.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_20_seed_00", "ensemble_20", 0, "configs/ensemble_20.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_20_seed_01", "ensemble_20", 1, "configs/ensemble_20.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_20_seed_02", "ensemble_20", 2, "configs/ensemble_20.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_20_seed_03", "ensemble_20", 3, "configs/ensemble_20.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_20_seed_04", "ensemble_20", 4, "configs/ensemble_20.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_20_seed_05", "ensemble_20", 5, "configs/ensemble_20.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_20_seed_06", "ensemble_20", 6, "configs/ensemble_20.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_20_seed_07", "ensemble_20", 7, "configs/ensemble_20.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_20_seed_08", "ensemble_20", 8, "configs/ensemble_20.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_20_seed_09", "ensemble_20", 9, "configs/ensemble_20.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_20_seed_10", "ensemble_20", 10, "configs/ensemble_20.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_20_seed_11", "ensemble_20", 11, "configs/ensemble_20.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_20_seed_12", "ensemble_20", 12, "configs/ensemble_20.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_20_seed_13", "ensemble_20", 13, "configs/ensemble_20.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_20_seed_14", "ensemble_20", 14, "configs/ensemble_20.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_20_seed_15", "ensemble_20", 15, "configs/ensemble_20.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_20_seed_16", "ensemble_20", 16, "configs/ensemble_20.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_20_seed_17", "ensemble_20", 17, "configs/ensemble_20.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_20_seed_18", "ensemble_20", 18, "configs/ensemble_20.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_20_seed_19", "ensemble_20", 19, "configs/ensemble_20.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_5_seed_00", "ensemble_5", 0, "configs/ensemble_5.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_5_seed_01", "ensemble_5", 1, "configs/ensemble_5.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_5_seed_02", "ensemble_5", 2, "configs/ensemble_5.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_5_seed_03", "ensemble_5", 3, "configs/ensemble_5.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_5_seed_04", "ensemble_5", 4, "configs/ensemble_5.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_5_seed_05", "ensemble_5", 5, "configs/ensemble_5.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_5_seed_06", "ensemble_5", 6, "configs/ensemble_5.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_5_seed_07", "ensemble_5", 7, "configs/ensemble_5.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_5_seed_08", "ensemble_5", 8, "configs/ensemble_5.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_5_seed_09", "ensemble_5", 9, "configs/ensemble_5.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_5_seed_10", "ensemble_5", 10, "configs/ensemble_5.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_5_seed_11", "ensemble_5", 11, "configs/ensemble_5.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_5_seed_12", "ensemble_5", 12, "configs/ensemble_5.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_5_seed_13", "ensemble_5", 13, "configs/ensemble_5.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_5_seed_14", "ensemble_5", 14, "configs/ensemble_5.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_5_seed_15", "ensemble_5", 15, "configs/ensemble_5.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_5_seed_16", "ensemble_5", 16, "configs/ensemble_5.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_5_seed_17", "ensemble_5", 17, "configs/ensemble_5.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_5_seed_18", "ensemble_5", 18, "configs/ensemble_5.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "ensemble_5_seed_19", "ensemble_5", 19, "configs/ensemble_5.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "gamma_sweep_seed_00", "gamma_sweep", 0, "configs/gamma_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "gamma_sweep_seed_01", "gamma_sweep", 1, "configs/gamma_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "gamma_sweep_seed_02", "gamma_sweep", 2, "configs/gamma_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "gamma_sweep_seed_03", "gamma_sweep", 3, "configs/gamma_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "gamma_sweep_seed_04", "gamma_sweep", 4, "configs/gamma_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "gamma_sweep_seed_05", "gamma_sweep", 5, "configs/gamma_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "gamma_sweep_seed_06", "gamma_sweep", 6, "configs/gamma_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "gamma_sweep_seed_07", "gamma_sweep", 7, "configs/gamma_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "gamma_sweep_seed_08", "gamma_sweep", 8, "configs/gamma_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "gamma_sweep_seed_09", "gamma_sweep", 9, "configs/gamma_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "gamma_sweep_seed_10", "gamma_sweep", 10, "configs/gamma_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "gamma_sweep_seed_11", "gamma_sweep", 11, "configs/gamma_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "gamma_sweep_seed_12", "gamma_sweep", 12, "configs/gamma_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "gamma_sweep_seed_13", "gamma_sweep", 13, "configs/gamma_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "gamma_sweep_seed_14", "gamma_sweep", 14, "configs/gamma_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "gamma_sweep_seed_15", "gamma_sweep", 15, "configs/gamma_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "gamma_sweep_seed_16", "gamma_sweep", 16, "configs/gamma_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "gamma_sweep_seed_17", "gamma_sweep", 17, "configs/gamma_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "gamma_sweep_seed_18", "gamma_sweep", 18, "configs/gamma_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "gamma_sweep_seed_19", "gamma_sweep", 19, "configs/gamma_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "learning_rate_sweep_seed_00", "learning_rate_sweep", 0, "configs/learning_rate_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "learning_rate_sweep_seed_01", "learning_rate_sweep", 1, "configs/learning_rate_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "learning_rate_sweep_seed_02", "learning_rate_sweep", 2, "configs/learning_rate_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "learning_rate_sweep_seed_03", "learning_rate_sweep", 3, "configs/learning_rate_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "learning_rate_sweep_seed_04", "learning_rate_sweep", 4, "configs/learning_rate_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "learning_rate_sweep_seed_05", "learning_rate_sweep", 5, "configs/learning_rate_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "learning_rate_sweep_seed_06", "learning_rate_sweep", 6, "configs/learning_rate_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "learning_rate_sweep_seed_07", "learning_rate_sweep", 7, "configs/learning_rate_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "learning_rate_sweep_seed_08", "learning_rate_sweep", 8, "configs/learning_rate_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "learning_rate_sweep_seed_09", "learning_rate_sweep", 9, "configs/learning_rate_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "learning_rate_sweep_seed_10", "learning_rate_sweep", 10, "configs/learning_rate_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "learning_rate_sweep_seed_11", "learning_rate_sweep", 11, "configs/learning_rate_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "learning_rate_sweep_seed_12", "learning_rate_sweep", 12, "configs/learning_rate_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "learning_rate_sweep_seed_13", "learning_rate_sweep", 13, "configs/learning_rate_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "learning_rate_sweep_seed_14", "learning_rate_sweep", 14, "configs/learning_rate_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "learning_rate_sweep_seed_15", "learning_rate_sweep", 15, "configs/learning_rate_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "learning_rate_sweep_seed_16", "learning_rate_sweep", 16, "configs/learning_rate_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "learning_rate_sweep_seed_17", "learning_rate_sweep", 17, "configs/learning_rate_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "learning_rate_sweep_seed_18", "learning_rate_sweep", 18, "configs/learning_rate_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "learning_rate_sweep_seed_19", "learning_rate_sweep", 19, "configs/learning_rate_sweep.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_imagenet_seed_00", "pretrain_imagenet", 0, "configs/pretrain_imagenet.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_imagenet_seed_01", "pretrain_imagenet", 1, "configs/pretrain_imagenet.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_imagenet_seed_02", "pretrain_imagenet", 2, "configs/pretrain_imagenet.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_imagenet_seed_03", "pretrain_imagenet", 3, "configs/pretrain_imagenet.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_imagenet_seed_04", "pretrain_imagenet", 4, "configs/pretrain_imagenet.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_imagenet_seed_05", "pretrain_imagenet", 5, "configs/pretrain_imagenet.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_imagenet_seed_06", "pretrain_imagenet", 6, "configs/pretrain_imagenet.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_imagenet_seed_07", "pretrain_imagenet", 7, "configs/pretrain_imagenet.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_imagenet_seed_08", "pretrain_imagenet", 8, "configs/pretrain_imagenet.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_imagenet_seed_09", "pretrain_imagenet", 9, "configs/pretrain_imagenet.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_imagenet_seed_10", "pretrain_imagenet", 10, "configs/pretrain_imagenet.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_imagenet_seed_11", "pretrain_imagenet", 11, "configs/pretrain_imagenet.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_imagenet_seed_12", "pretrain_imagenet", 12, "configs/pretrain_imagenet.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_imagenet_seed_13", "pretrain_imagenet", 13, "configs/pretrain_imagenet.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_imagenet_seed_14", "pretrain_imagenet", 14, "configs/pretrain_imagenet.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_imagenet_seed_15", "pretrain_imagenet", 15, "configs/pretrain_imagenet.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_imagenet_seed_16", "pretrain_imagenet", 16, "configs/pretrain_imagenet.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_imagenet_seed_17", "pretrain_imagenet", 17, "configs/pretrain_imagenet.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_imagenet_seed_18", "pretrain_imagenet", 18, "configs/pretrain_imagenet.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_imagenet_seed_19", "pretrain_imagenet", 19, "configs/pretrain_imagenet.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_radfm_seed_00", "pretrain_radfm", 0, "configs/pretrain_radfm.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_radfm_seed_01", "pretrain_radfm", 1, "configs/pretrain_radfm.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_radfm_seed_02", "pretrain_radfm", 2, "configs/pretrain_radfm.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_radfm_seed_03", "pretrain_radfm", 3, "configs/pretrain_radfm.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_radfm_seed_04", "pretrain_radfm", 4, "configs/pretrain_radfm.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_radfm_seed_05", "pretrain_radfm", 5, "configs/pretrain_radfm.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_radfm_seed_06", "pretrain_radfm", 6, "configs/pretrain_radfm.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_radfm_seed_07", "pretrain_radfm", 7, "configs/pretrain_radfm.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_radfm_seed_08", "pretrain_radfm", 8, "configs/pretrain_radfm.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_radfm_seed_09", "pretrain_radfm", 9, "configs/pretrain_radfm.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_radfm_seed_10", "pretrain_radfm", 10, "configs/pretrain_radfm.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_radfm_seed_11", "pretrain_radfm", 11, "configs/pretrain_radfm.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_radfm_seed_12", "pretrain_radfm", 12, "configs/pretrain_radfm.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_radfm_seed_13", "pretrain_radfm", 13, "configs/pretrain_radfm.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_radfm_seed_14", "pretrain_radfm", 14, "configs/pretrain_radfm.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_radfm_seed_15", "pretrain_radfm", 15, "configs/pretrain_radfm.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_radfm_seed_16", "pretrain_radfm", 16, "configs/pretrain_radfm.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_radfm_seed_17", "pretrain_radfm", 17, "configs/pretrain_radfm.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_radfm_seed_18", "pretrain_radfm", 18, "configs/pretrain_radfm.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_radfm_seed_19", "pretrain_radfm", 19, "configs/pretrain_radfm.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_scratch_seed_00", "pretrain_scratch", 0, "configs/pretrain_scratch.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_scratch_seed_01", "pretrain_scratch", 1, "configs/pretrain_scratch.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_scratch_seed_02", "pretrain_scratch", 2, "configs/pretrain_scratch.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_scratch_seed_03", "pretrain_scratch", 3, "configs/pretrain_scratch.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_scratch_seed_04", "pretrain_scratch", 4, "configs/pretrain_scratch.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_scratch_seed_05", "pretrain_scratch", 5, "configs/pretrain_scratch.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_scratch_seed_06", "pretrain_scratch", 6, "configs/pretrain_scratch.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_scratch_seed_07", "pretrain_scratch", 7, "configs/pretrain_scratch.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_scratch_seed_08", "pretrain_scratch", 8, "configs/pretrain_scratch.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_scratch_seed_09", "pretrain_scratch", 9, "configs/pretrain_scratch.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_scratch_seed_10", "pretrain_scratch", 10, "configs/pretrain_scratch.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_scratch_seed_11", "pretrain_scratch", 11, "configs/pretrain_scratch.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_scratch_seed_12", "pretrain_scratch", 12, "configs/pretrain_scratch.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_scratch_seed_13", "pretrain_scratch", 13, "configs/pretrain_scratch.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_scratch_seed_14", "pretrain_scratch", 14, "configs/pretrain_scratch.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_scratch_seed_15", "pretrain_scratch", 15, "configs/pretrain_scratch.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_scratch_seed_16", "pretrain_scratch", 16, "configs/pretrain_scratch.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_scratch_seed_17", "pretrain_scratch", 17, "configs/pretrain_scratch.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_scratch_seed_18", "pretrain_scratch", 18, "configs/pretrain_scratch.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "pretrain_scratch_seed_19", "pretrain_scratch", 19, "configs/pretrain_scratch.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_16_seed_00", "site_embedding_16", 0, "configs/site_embedding_16.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_16_seed_01", "site_embedding_16", 1, "configs/site_embedding_16.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_16_seed_02", "site_embedding_16", 2, "configs/site_embedding_16.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_16_seed_03", "site_embedding_16", 3, "configs/site_embedding_16.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_16_seed_04", "site_embedding_16", 4, "configs/site_embedding_16.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_16_seed_05", "site_embedding_16", 5, "configs/site_embedding_16.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_16_seed_06", "site_embedding_16", 6, "configs/site_embedding_16.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_16_seed_07", "site_embedding_16", 7, "configs/site_embedding_16.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_16_seed_08", "site_embedding_16", 8, "configs/site_embedding_16.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_16_seed_09", "site_embedding_16", 9, "configs/site_embedding_16.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_16_seed_10", "site_embedding_16", 10, "configs/site_embedding_16.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_16_seed_11", "site_embedding_16", 11, "configs/site_embedding_16.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_16_seed_12", "site_embedding_16", 12, "configs/site_embedding_16.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_16_seed_13", "site_embedding_16", 13, "configs/site_embedding_16.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_16_seed_14", "site_embedding_16", 14, "configs/site_embedding_16.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_16_seed_15", "site_embedding_16", 15, "configs/site_embedding_16.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_16_seed_16", "site_embedding_16", 16, "configs/site_embedding_16.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_16_seed_17", "site_embedding_16", 17, "configs/site_embedding_16.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_16_seed_18", "site_embedding_16", 18, "configs/site_embedding_16.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_16_seed_19", "site_embedding_16", 19, "configs/site_embedding_16.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_6_seed_00", "site_embedding_6", 0, "configs/site_embedding_6.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_6_seed_01", "site_embedding_6", 1, "configs/site_embedding_6.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_6_seed_02", "site_embedding_6", 2, "configs/site_embedding_6.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_6_seed_03", "site_embedding_6", 3, "configs/site_embedding_6.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_6_seed_04", "site_embedding_6", 4, "configs/site_embedding_6.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_6_seed_05", "site_embedding_6", 5, "configs/site_embedding_6.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_6_seed_06", "site_embedding_6", 6, "configs/site_embedding_6.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_6_seed_07", "site_embedding_6", 7, "configs/site_embedding_6.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_6_seed_08", "site_embedding_6", 8, "configs/site_embedding_6.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_6_seed_09", "site_embedding_6", 9, "configs/site_embedding_6.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_6_seed_10", "site_embedding_6", 10, "configs/site_embedding_6.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_6_seed_11", "site_embedding_6", 11, "configs/site_embedding_6.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_6_seed_12", "site_embedding_6", 12, "configs/site_embedding_6.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_6_seed_13", "site_embedding_6", 13, "configs/site_embedding_6.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_6_seed_14", "site_embedding_6", 14, "configs/site_embedding_6.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_6_seed_15", "site_embedding_6", 15, "configs/site_embedding_6.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_6_seed_16", "site_embedding_6", 16, "configs/site_embedding_6.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_6_seed_17", "site_embedding_6", 17, "configs/site_embedding_6.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_6_seed_18", "site_embedding_6", 18, "configs/site_embedding_6.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_6_seed_19", "site_embedding_6", 19, "configs/site_embedding_6.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_64_seed_00", "site_embedding_64", 0, "configs/site_embedding_64.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_64_seed_01", "site_embedding_64", 1, "configs/site_embedding_64.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_64_seed_02", "site_embedding_64", 2, "configs/site_embedding_64.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_64_seed_03", "site_embedding_64", 3, "configs/site_embedding_64.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_64_seed_04", "site_embedding_64", 4, "configs/site_embedding_64.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_64_seed_05", "site_embedding_64", 5, "configs/site_embedding_64.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_64_seed_06", "site_embedding_64", 6, "configs/site_embedding_64.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_64_seed_07", "site_embedding_64", 7, "configs/site_embedding_64.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_64_seed_08", "site_embedding_64", 8, "configs/site_embedding_64.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_64_seed_09", "site_embedding_64", 9, "configs/site_embedding_64.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_64_seed_10", "site_embedding_64", 10, "configs/site_embedding_64.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_64_seed_11", "site_embedding_64", 11, "configs/site_embedding_64.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_64_seed_12", "site_embedding_64", 12, "configs/site_embedding_64.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_64_seed_13", "site_embedding_64", 13, "configs/site_embedding_64.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_64_seed_14", "site_embedding_64", 14, "configs/site_embedding_64.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_64_seed_15", "site_embedding_64", 15, "configs/site_embedding_64.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_64_seed_16", "site_embedding_64", 16, "configs/site_embedding_64.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_64_seed_17", "site_embedding_64", 17, "configs/site_embedding_64.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_64_seed_18", "site_embedding_64", 18, "configs/site_embedding_64.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "site_embedding_64_seed_19", "site_embedding_64", 19, "configs/site_embedding_64.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "transfer_most_oai_seed_00", "transfer_most_oai", 0, "configs/transfer_most_oai.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "transfer_most_oai_seed_01", "transfer_most_oai", 1, "configs/transfer_most_oai.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "transfer_most_oai_seed_02", "transfer_most_oai", 2, "configs/transfer_most_oai.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "transfer_most_oai_seed_03", "transfer_most_oai", 3, "configs/transfer_most_oai.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "transfer_most_oai_seed_04", "transfer_most_oai", 4, "configs/transfer_most_oai.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "transfer_most_oai_seed_05", "transfer_most_oai", 5, "configs/transfer_most_oai.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "transfer_most_oai_seed_06", "transfer_most_oai", 6, "configs/transfer_most_oai.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "transfer_most_oai_seed_07", "transfer_most_oai", 7, "configs/transfer_most_oai.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "transfer_most_oai_seed_08", "transfer_most_oai", 8, "configs/transfer_most_oai.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "transfer_most_oai_seed_09", "transfer_most_oai", 9, "configs/transfer_most_oai.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "transfer_most_oai_seed_10", "transfer_most_oai", 10, "configs/transfer_most_oai.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "transfer_most_oai_seed_11", "transfer_most_oai", 11, "configs/transfer_most_oai.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "transfer_most_oai_seed_12", "transfer_most_oai", 12, "configs/transfer_most_oai.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "transfer_most_oai_seed_13", "transfer_most_oai", 13, "configs/transfer_most_oai.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "transfer_most_oai_seed_14", "transfer_most_oai", 14, "configs/transfer_most_oai.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "transfer_most_oai_seed_15", "transfer_most_oai", 15, "configs/transfer_most_oai.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "transfer_most_oai_seed_16", "transfer_most_oai", 16, "configs/transfer_most_oai.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "transfer_most_oai_seed_17", "transfer_most_oai", 17, "configs/transfer_most_oai.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "transfer_most_oai_seed_18", "transfer_most_oai", 18, "configs/transfer_most_oai.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "transfer_most_oai_seed_19", "transfer_most_oai", 19, "configs/transfer_most_oai.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "transfer_oai_most_seed_00", "transfer_oai_most", 0, "configs/transfer_oai_most.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "transfer_oai_most_seed_01", "transfer_oai_most", 1, "configs/transfer_oai_most.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "transfer_oai_most_seed_02", "transfer_oai_most", 2, "configs/transfer_oai_most.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "transfer_oai_most_seed_03", "transfer_oai_most", 3, "configs/transfer_oai_most.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "transfer_oai_most_seed_04", "transfer_oai_most", 4, "configs/transfer_oai_most.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "transfer_oai_most_seed_05", "transfer_oai_most", 5, "configs/transfer_oai_most.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "transfer_oai_most_seed_06", "transfer_oai_most", 6, "configs/transfer_oai_most.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "transfer_oai_most_seed_07", "transfer_oai_most", 7, "configs/transfer_oai_most.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "transfer_oai_most_seed_08", "transfer_oai_most", 8, "configs/transfer_oai_most.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "transfer_oai_most_seed_09", "transfer_oai_most", 9, "configs/transfer_oai_most.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "transfer_oai_most_seed_10", "transfer_oai_most", 10, "configs/transfer_oai_most.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "transfer_oai_most_seed_11", "transfer_oai_most", 11, "configs/transfer_oai_most.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "transfer_oai_most_seed_12", "transfer_oai_most", 12, "configs/transfer_oai_most.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "transfer_oai_most_seed_13", "transfer_oai_most", 13, "configs/transfer_oai_most.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "transfer_oai_most_seed_14", "transfer_oai_most", 14, "configs/transfer_oai_most.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "transfer_oai_most_seed_15", "transfer_oai_most", 15, "configs/transfer_oai_most.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "transfer_oai_most_seed_16", "transfer_oai_most", 16, "configs/transfer_oai_most.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "transfer_oai_most_seed_17", "transfer_oai_most", 17, "configs/transfer_oai_most.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "transfer_oai_most_seed_18", "transfer_oai_most", 18, "configs/transfer_oai_most.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "transfer_oai_most_seed_19", "transfer_oai_most", 19, "configs/transfer_oai_most.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "uncertainty_dropout_seed_00", "uncertainty_dropout", 0, "configs/uncertainty_dropout.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "uncertainty_dropout_seed_01", "uncertainty_dropout", 1, "configs/uncertainty_dropout.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "uncertainty_dropout_seed_02", "uncertainty_dropout", 2, "configs/uncertainty_dropout.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "uncertainty_dropout_seed_03", "uncertainty_dropout", 3, "configs/uncertainty_dropout.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "uncertainty_dropout_seed_04", "uncertainty_dropout", 4, "configs/uncertainty_dropout.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "uncertainty_dropout_seed_05", "uncertainty_dropout", 5, "configs/uncertainty_dropout.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "uncertainty_dropout_seed_06", "uncertainty_dropout", 6, "configs/uncertainty_dropout.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "uncertainty_dropout_seed_07", "uncertainty_dropout", 7, "configs/uncertainty_dropout.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "uncertainty_dropout_seed_08", "uncertainty_dropout", 8, "configs/uncertainty_dropout.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "uncertainty_dropout_seed_09", "uncertainty_dropout", 9, "configs/uncertainty_dropout.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "uncertainty_dropout_seed_10", "uncertainty_dropout", 10, "configs/uncertainty_dropout.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "uncertainty_dropout_seed_11", "uncertainty_dropout", 11, "configs/uncertainty_dropout.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "uncertainty_dropout_seed_12", "uncertainty_dropout", 12, "configs/uncertainty_dropout.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "uncertainty_dropout_seed_13", "uncertainty_dropout", 13, "configs/uncertainty_dropout.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "uncertainty_dropout_seed_14", "uncertainty_dropout", 14, "configs/uncertainty_dropout.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "uncertainty_dropout_seed_15", "uncertainty_dropout", 15, "configs/uncertainty_dropout.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "uncertainty_dropout_seed_16", "uncertainty_dropout", 16, "configs/uncertainty_dropout.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "uncertainty_dropout_seed_17", "uncertainty_dropout", 17, "configs/uncertainty_dropout.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "uncertainty_dropout_seed_18", "uncertainty_dropout", 18, "configs/uncertainty_dropout.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "uncertainty_dropout_seed_19", "uncertainty_dropout", 19, "configs/uncertainty_dropout.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "uncertainty_temperature_seed_00", "uncertainty_temperature", 0, "configs/uncertainty_temperature.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "uncertainty_temperature_seed_01", "uncertainty_temperature", 1, "configs/uncertainty_temperature.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "uncertainty_temperature_seed_02", "uncertainty_temperature", 2, "configs/uncertainty_temperature.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "uncertainty_temperature_seed_03", "uncertainty_temperature", 3, "configs/uncertainty_temperature.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "uncertainty_temperature_seed_04", "uncertainty_temperature", 4, "configs/uncertainty_temperature.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "uncertainty_temperature_seed_05", "uncertainty_temperature", 5, "configs/uncertainty_temperature.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "uncertainty_temperature_seed_06", "uncertainty_temperature", 6, "configs/uncertainty_temperature.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "uncertainty_temperature_seed_07", "uncertainty_temperature", 7, "configs/uncertainty_temperature.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "uncertainty_temperature_seed_08", "uncertainty_temperature", 8, "configs/uncertainty_temperature.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "uncertainty_temperature_seed_09", "uncertainty_temperature", 9, "configs/uncertainty_temperature.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "uncertainty_temperature_seed_10", "uncertainty_temperature", 10, "configs/uncertainty_temperature.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "uncertainty_temperature_seed_11", "uncertainty_temperature", 11, "configs/uncertainty_temperature.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "uncertainty_temperature_seed_12", "uncertainty_temperature", 12, "configs/uncertainty_temperature.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "uncertainty_temperature_seed_13", "uncertainty_temperature", 13, "configs/uncertainty_temperature.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "uncertainty_temperature_seed_14", "uncertainty_temperature", 14, "configs/uncertainty_temperature.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "uncertainty_temperature_seed_15", "uncertainty_temperature", 15, "configs/uncertainty_temperature.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "uncertainty_temperature_seed_16", "uncertainty_temperature", 16, "configs/uncertainty_temperature.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "uncertainty_temperature_seed_17", "uncertainty_temperature", 17, "configs/uncertainty_temperature.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "uncertainty_temperature_seed_18", "uncertainty_temperature", 18, "configs/uncertainty_temperature.yaml", 4, "bfloat16"
    ),
    RunSpec(
        "uncertainty_temperature_seed_19", "uncertainty_temperature", 19, "configs/uncertainty_temperature.yaml", 4, "bfloat16"
    ),
)


def runs_for(experiment: str) -> tuple[RunSpec, ...]:
    return tuple(run for run in RUNS if run.experiment == experiment)


def run_by_identifier(identifier: str) -> RunSpec:
    matches = tuple(run for run in RUNS if run.identifier == identifier)
    if len(matches) != 1:
        raise KeyError(identifier)
    return matches[0]


def validate_registry(root: str | Path | None = None) -> None:
    identifiers = {run.identifier for run in RUNS}
    if len(identifiers) != len(RUNS):
        raise ValueError("run identifiers must be unique")
    grouped = {experiment: runs_for(experiment) for experiment in {run.experiment for run in RUNS}}
    if any(len(values) != 20 for values in grouped.values()):
        raise ValueError("each experiment must contain twenty seeds")
    if any(tuple(run.seed for run in values) != tuple(range(20)) for values in grouped.values()):
        raise ValueError("experiment seeds must span zero through nineteen")
    if root is not None:
        base = Path(root)
        missing = [run.configuration for run in RUNS if not (base / run.configuration).is_file()]
        if missing:
            raise ValueError(f"missing configurations: {sorted(set(missing))}")


def select_runs(experiments: Iterable[str]) -> tuple[RunSpec, ...]:
    selected = set(experiments)
    unknown = selected - {run.experiment for run in RUNS}
    if unknown:
        raise ValueError(f"unknown experiments: {sorted(unknown)}")
    return tuple(run for run in RUNS if run.experiment in selected)

