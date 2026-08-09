from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


class Feature(str, Enum):
    HOFFA = "hoffa"
    EFFUSION = "effusion"
    BML = "bml"


class Source(str, Enum):
    OAI_MARYLAND = "oai_md"
    OAI_OHIO = "oai_oh"
    OAI_PITTSBURGH = "oai_pa"
    OAI_RHODE_ISLAND = "oai_ri"
    MOST_ALABAMA = "most_al"
    MOST_IOWA = "most_ia"


@dataclass(frozen=True)
class SourceProfile:
    source: Source
    cohort: str
    count: int
    vendor: str
    field_strength: float
    institution: str


@dataclass(frozen=True)
class SplitFractions:
    train: float = 0.75
    validation: float = 0.10
    test: float = 0.15

    def validate(self) -> None:
        values = (self.train, self.validation, self.test)
        if any(value <= 0 for value in values):
            raise ValueError("split fractions must be positive")
        if not np.isclose(sum(values), 1.0):
            raise ValueError("split fractions must sum to one")


@dataclass(frozen=True)
class CaseRecord:
    participant: str
    source: Source
    age: float
    sex: str
    bmi: float
    kl_grade: int
    hoffa_grade: int
    effusion_grade: int
    bml_grade: int
    volume: str


@dataclass(frozen=True)
class Partition:
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]


@dataclass(frozen=True)
class Hypothesis:
    name: str
    metric: str
    operator: str
    threshold: float
    family: str


@dataclass(frozen=True)
class ContrastCondition:
    name: str
    relaxivity: float
    polarity: int
    field_strength: float


SOURCE_PROFILES: tuple[SourceProfile, ...] = (
    SourceProfile(Source.OAI_MARYLAND, "OAI", 1243, "Siemens", 3.0, "Maryland"),
    SourceProfile(Source.OAI_OHIO, "OAI", 1197, "Siemens", 3.0, "Ohio"),
    SourceProfile(Source.OAI_PITTSBURGH, "OAI", 1189, "Siemens", 3.0, "Pittsburgh"),
    SourceProfile(Source.OAI_RHODE_ISLAND, "OAI", 1167, "Siemens", 3.0, "Rhode Island"),
    SourceProfile(Source.MOST_ALABAMA, "MOST", 1532, "GE", 1.0, "Alabama"),
    SourceProfile(Source.MOST_IOWA, "MOST", 1494, "GE", 1.0, "Iowa"),
)


PRIMARY_HYPOTHESES: tuple[Hypothesis, ...] = (
    Hypothesis("H1a", "cross_site_auc_standard_deviation", "less_equal", 0.03, "H1"),
    Hypothesis("H1b", "mean_auc", "greater_equal", 0.86, "H1"),
    Hypothesis("H2", "quadratic_weighted_kappa", "greater_equal", 0.81, "H2"),
    Hypothesis("H3", "contrast_auc_retention", "greater_equal", 0.95, "H3"),
    Hypothesis("H4", "vocabulary_to_image_slope_ratio", "greater_equal", 2.0, "H4"),
    Hypothesis("H5", "maximum_subgroup_auc_gap", "less_equal", 0.12, "H5"),
)


CONTRAST_CONDITIONS: tuple[ContrastCondition, ...] = (
    ContrastCondition("gadolinium_dtpa", 3.8, 1, 1.5),
    ContrastCondition("gadolinium_bopta", 5.6, 1, 1.5),
    ContrastCondition("mid_range", 9.2, 1, 1.5),
    ContrastCondition("spion", 14.1, -1, 1.5),
    ContrastCondition("ferumoxytol", 19.0, -1, 1.5),
)


def source_counts() -> dict[Source, int]:
    return {profile.source: profile.count for profile in SOURCE_PROFILES}


def total_cases() -> int:
    return sum(profile.count for profile in SOURCE_PROFILES)


def validate_profiles() -> None:
    if total_cases() != 7822:
        raise ValueError("source counts do not sum to the study cohort")
    if len({profile.source for profile in SOURCE_PROFILES}) != 6:
        raise ValueError("six unique acquisition sources are required")
    if len({profile.vendor for profile in SOURCE_PROFILES}) != 2:
        raise ValueError("two scanner vendors are required")
    if len({profile.field_strength for profile in SOURCE_PROFILES}) != 2:
        raise ValueError("two field strengths are required")


def age_decile(age: float) -> int:
    if age < 0:
        raise ValueError("age must be nonnegative")
    return int(age // 10)


def bmi_tertile(bmi: float) -> str:
    if bmi < 0:
        raise ValueError("BMI must be nonnegative")
    if bmi < 25:
        return "lower"
    if bmi < 30:
        return "middle"
    return "upper"


def stratification_key(record: CaseRecord) -> tuple[str, str, int, int, int, int, int]:
    return (
        record.source.value,
        record.sex,
        age_decile(record.age),
        record.kl_grade,
        record.hoffa_grade,
        record.effusion_grade,
        record.bml_grade,
    )


def grouped_participants(records: Sequence[CaseRecord]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(record.participant, []).append(index)
    return groups


def participant_labels(records: Sequence[CaseRecord]) -> dict[str, tuple[str, str, int, int, int, int, int]]:
    labels: dict[str, tuple[str, str, int, int, int, int, int]] = {}
    for record in records:
        key = stratification_key(record)
        existing = labels.get(record.participant)
        if existing is not None and existing != key:
            raise ValueError("participant records disagree on stratification fields")
        labels[record.participant] = key
    return labels


def split_participants(
    records: Sequence[CaseRecord],
    fractions: SplitFractions = SplitFractions(),
    seed: int = 0,
) -> Partition:
    fractions.validate()
    groups = grouped_participants(records)
    labels = participant_labels(records)
    strata: dict[tuple[str, str, int, int, int, int, int], list[str]] = {}
    for participant, label in labels.items():
        strata.setdefault(label, []).append(participant)
    generator = np.random.default_rng(seed)
    train_participants: set[str] = set()
    validation_participants: set[str] = set()
    test_participants: set[str] = set()
    for participants in strata.values():
        ordered = np.asarray(sorted(participants), dtype=object)
        generator.shuffle(ordered)
        count = ordered.size
        train_end = int(round(count * fractions.train))
        validation_end = train_end + int(round(count * fractions.validation))
        train_participants.update(str(value) for value in ordered[:train_end])
        validation_participants.update(str(value) for value in ordered[train_end:validation_end])
        test_participants.update(str(value) for value in ordered[validation_end:])
    train = tuple(index for participant in train_participants for index in groups[participant])
    validation = tuple(index for participant in validation_participants for index in groups[participant])
    test = tuple(index for participant in test_participants for index in groups[participant])
    partition = Partition(tuple(sorted(train)), tuple(sorted(validation)), tuple(sorted(test)))
    validate_partition(partition, len(records))
    return partition


def validate_partition(partition: Partition, record_count: int) -> None:
    train = set(partition.train)
    validation = set(partition.validation)
    test = set(partition.test)
    if train & validation or train & test or validation & test:
        raise ValueError("partitions overlap")
    if train | validation | test != set(range(record_count)):
        raise ValueError("partitions do not cover every record")


class UniformSourceSampler:
    def __init__(
        self,
        sources: Sequence[Source],
        batch_size: int,
        seed: int = 0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        self.sources = tuple(sources)
        self.batch_size = batch_size
        self.generator = np.random.default_rng(seed)
        self.indices: dict[Source, NDArray[np.int64]] = {}
        for source in Source:
            values = np.flatnonzero(np.asarray(self.sources, dtype=object) == source)
            if values.size == 0:
                raise ValueError(f"source {source.value} has no cases")
            self.indices[source] = values
        self.ordered_sources = tuple(Source)

    def allocation(self) -> dict[Source, int]:
        base = self.batch_size // len(self.ordered_sources)
        remainder = self.batch_size % len(self.ordered_sources)
        return {
            source: base + int(index < remainder)
            for index, source in enumerate(self.ordered_sources)
        }

    def sample(self) -> tuple[int, ...]:
        selected: list[int] = []
        for source, count in self.allocation().items():
            values = self.generator.choice(self.indices[source], size=count, replace=True)
            selected.extend(int(value) for value in values)
        self.generator.shuffle(selected)
        return tuple(selected)


def evaluate_hypothesis(hypothesis: Hypothesis, value: float) -> bool:
    if hypothesis.operator == "less_equal":
        return value <= hypothesis.threshold
    if hypothesis.operator == "greater_equal":
        return value >= hypothesis.threshold
    raise ValueError("unknown hypothesis operator")


def hypothesis_results(values: Mapping[str, float]) -> dict[str, bool]:
    results: dict[str, bool] = {}
    for hypothesis in PRIMARY_HYPOTHESES:
        if hypothesis.metric not in values:
            raise ValueError(f"missing metric {hypothesis.metric}")
        results[hypothesis.name] = evaluate_hypothesis(hypothesis, values[hypothesis.metric])
    return results


def feature_grade(record: CaseRecord, feature: Feature) -> int:
    if feature is Feature.HOFFA:
        return record.hoffa_grade
    if feature is Feature.EFFUSION:
        return record.effusion_grade
    return record.bml_grade


def validate_grade(value: int) -> None:
    if value not in {0, 1, 2, 3}:
        raise ValueError("MOAKS grade must be between zero and three")


def validate_record(record: CaseRecord) -> None:
    if not record.participant:
        raise ValueError("participant identifier must not be empty")
    if not record.volume:
        raise ValueError("volume location must not be empty")
    if record.age < 0 or record.bmi < 0:
        raise ValueError("age and BMI must be nonnegative")
    if record.sex not in {"female", "male"}:
        raise ValueError("sex must be female or male")
    if record.kl_grade not in {0, 1, 2, 3, 4}:
        raise ValueError("KL grade must be between zero and four")
    validate_grade(record.hoffa_grade)
    validate_grade(record.effusion_grade)
    validate_grade(record.bml_grade)


def validate_records(records: Iterable[CaseRecord]) -> tuple[CaseRecord, ...]:
    values = tuple(records)
    if not values:
        raise ValueError("records must not be empty")
    for record in values:
        validate_record(record)
    return values
