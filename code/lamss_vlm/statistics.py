from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, lgamma, log, sqrt
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class Interval:
    estimate: float
    lower: float
    upper: float


@dataclass(frozen=True)
class SiteResult:
    site: str
    count: int
    auc: Interval
    kappa: float
    spearman: float
    mcid_crossing_rate: float


@dataclass(frozen=True)
class CalibrationResult:
    ece: float
    mce: float
    brier: float
    nll: float
    reliability_gap: float


@dataclass(frozen=True)
class HeterogeneityResult:
    mean: float
    standard_deviation: float
    cochran_q: float
    i_squared: float


@dataclass(frozen=True)
class ComparisonResult:
    difference: float
    standard_error: float
    z_score: float
    p_value: float


def as_float(values: Sequence[float] | FloatArray) -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if result.size == 0:
        raise ValueError("values must not be empty")
    return result


def as_int(values: Sequence[int] | IntArray) -> IntArray:
    result = np.asarray(values, dtype=np.int64)
    if result.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if result.size == 0:
        raise ValueError("values must not be empty")
    return result


def validate_binary(targets: IntArray) -> None:
    if not np.all(np.isin(targets, [0, 1])):
        raise ValueError("targets must be binary")
    if np.unique(targets).size != 2:
        raise ValueError("both target classes are required")


def average_ranks(values: FloatArray) -> FloatArray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def binary_auc(targets: Sequence[int], scores: Sequence[float]) -> float:
    target = as_int(targets)
    score = as_float(scores)
    if target.size != score.size:
        raise ValueError("targets and scores must have equal length")
    validate_binary(target)
    positive = target == 1
    positive_count = int(positive.sum())
    negative_count = target.size - positive_count
    ranks = average_ranks(score)
    rank_sum = float(ranks[positive].sum())
    return (rank_sum - positive_count * (positive_count + 1) / 2) / (
        positive_count * negative_count
    )


def ordinal_auc(targets: Sequence[int], scores: Sequence[float], threshold: int = 1) -> float:
    values = as_int(targets)
    return binary_auc((values >= threshold).astype(np.int64), scores)


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    first = average_ranks(as_float(left))
    second = average_ranks(as_float(right))
    if first.size != second.size:
        raise ValueError("inputs must have equal length")
    first -= first.mean()
    second -= second.mean()
    denominator = sqrt(float(np.dot(first, first) * np.dot(second, second)))
    return float(np.dot(first, second) / denominator) if denominator else 0.0


def weighted_kappa(targets: Sequence[int], predictions: Sequence[int], grades: int = 4) -> float:
    target = as_int(targets)
    prediction = as_int(predictions)
    if target.size != prediction.size:
        raise ValueError("targets and predictions must have equal length")
    matrix = np.zeros((grades, grades), dtype=np.float64)
    np.add.at(matrix, (target, prediction), 1.0)
    observed = matrix / matrix.sum()
    target_hist = matrix.sum(axis=1)
    prediction_hist = matrix.sum(axis=0)
    expected = np.outer(target_hist, prediction_hist) / matrix.sum() ** 2
    indices = np.arange(grades, dtype=np.float64)
    weights = (indices[:, None] - indices[None, :]) ** 2 / max(1, (grades - 1) ** 2)
    denominator = float((weights * expected).sum())
    return 1.0 - float((weights * observed).sum()) / denominator if denominator else 1.0


def intraclass_correlation(targets: Sequence[float], predictions: Sequence[float]) -> float:
    target = as_float(targets)
    prediction = as_float(predictions)
    if target.size != prediction.size:
        raise ValueError("targets and predictions must have equal length")
    matrix = np.column_stack((target, prediction))
    sample_count, rater_count = matrix.shape
    grand_mean = float(matrix.mean())
    row_means = matrix.mean(axis=1)
    column_means = matrix.mean(axis=0)
    row_ss = rater_count * float(((row_means - grand_mean) ** 2).sum())
    error_ss = float(
        ((matrix - row_means[:, None] - column_means[None, :] + grand_mean) ** 2).sum()
    )
    row_ms = row_ss / max(1, sample_count - 1)
    error_ms = error_ss / max(1, (sample_count - 1) * (rater_count - 1))
    denominator = row_ms + (rater_count - 1) * error_ms
    return (row_ms - error_ms) / denominator if denominator else 0.0


def mcid_crossing_rate(
    targets: Sequence[int], predictions: Sequence[int], grade_threshold: int = 1
) -> float:
    target = as_int(targets)
    prediction = as_int(predictions)
    if target.size != prediction.size:
        raise ValueError("targets and predictions must have equal length")
    return float((np.abs(target - prediction) > grade_threshold).mean())


def bootstrap_interval(
    values: Sequence[float],
    statistic: Callable[[FloatArray], float] = lambda x: float(x.mean()),
    iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    sample = as_float(values)
    generator = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        selection = generator.integers(0, sample.size, sample.size)
        estimates[index] = statistic(sample[selection])
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(estimates, [tail, 1.0 - tail])
    return Interval(statistic(sample), float(lower), float(upper))


def stratified_auc_interval(
    targets: Sequence[int],
    scores: Sequence[float],
    strata: Sequence[int],
    iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    target = as_int(targets)
    score = as_float(scores)
    stratum = as_int(strata)
    if not (target.size == score.size == stratum.size):
        raise ValueError("all inputs must have equal length")
    groups = [np.flatnonzero(stratum == value) for value in np.unique(stratum)]
    generator = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        selected = np.concatenate(
            [group[generator.integers(0, group.size, group.size)] for group in groups]
        )
        estimates[iteration] = binary_auc(target[selected], score[selected])
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(estimates, [tail, 1.0 - tail])
    return Interval(binary_auc(target, score), float(lower), float(upper))


def cross_site_summary(aucs: Mapping[str, float]) -> HeterogeneityResult:
    values = as_float(list(aucs.values()))
    mean = float(values.mean())
    standard_deviation = float(values.std(ddof=0))
    variances = np.full(values.size, max(standard_deviation**2, 1e-12))
    weights = 1.0 / variances
    pooled = float(np.dot(weights, values) / weights.sum())
    q_value = float(np.dot(weights, (values - pooled) ** 2))
    degrees = values.size - 1
    i_squared = max(0.0, (q_value - degrees) / q_value * 100.0) if q_value else 0.0
    return HeterogeneityResult(mean, standard_deviation, q_value, i_squared)


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def delong_components(targets: IntArray, scores: FloatArray) -> tuple[FloatArray, FloatArray]:
    positive = scores[targets == 1]
    negative = scores[targets == 0]
    comparisons = np.zeros((positive.size, negative.size), dtype=np.float64)
    comparisons[positive[:, None] > negative[None, :]] = 1.0
    comparisons[positive[:, None] == negative[None, :]] = 0.5
    return comparisons.mean(axis=1), comparisons.mean(axis=0)


def paired_delong(
    targets: Sequence[int], first_scores: Sequence[float], second_scores: Sequence[float]
) -> ComparisonResult:
    target = as_int(targets)
    first = as_float(first_scores)
    second = as_float(second_scores)
    if not (target.size == first.size == second.size):
        raise ValueError("all inputs must have equal length")
    validate_binary(target)
    first_positive, first_negative = delong_components(target, first)
    second_positive, second_negative = delong_components(target, second)
    first_auc = binary_auc(target, first)
    second_auc = binary_auc(target, second)
    positive_difference = first_positive - second_positive
    negative_difference = first_negative - second_negative
    variance = positive_difference.var(ddof=1) / positive_difference.size
    variance += negative_difference.var(ddof=1) / negative_difference.size
    standard_error = sqrt(max(float(variance), 1e-15))
    difference = first_auc - second_auc
    z_score = difference / standard_error
    p_value = 2.0 * (1.0 - normal_cdf(abs(z_score)))
    return ComparisonResult(difference, standard_error, z_score, p_value)


def holm_bonferroni(p_values: Mapping[str, float], alpha: float = 0.05) -> dict[str, bool]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    decisions: dict[str, bool] = {name: False for name in p_values}
    active = True
    count = len(ordered)
    for index, (name, value) in enumerate(ordered):
        threshold = alpha / (count - index)
        active = active and value <= threshold
        decisions[name] = active
    return decisions


def softmax(logits: FloatArray) -> FloatArray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def calibration_metrics(
    targets: Sequence[int], probabilities: NDArray[np.float64], bins: int = 10
) -> CalibrationResult:
    target = as_int(targets)
    probability = np.asarray(probabilities, dtype=np.float64)
    if probability.ndim != 2 or probability.shape[0] != target.size:
        raise ValueError("probabilities must be a case by class matrix")
    confidence = probability.max(axis=1)
    prediction = probability.argmax(axis=1)
    correctness = (prediction == target).astype(np.float64)
    order = np.argsort(confidence)
    groups = np.array_split(order, bins)
    gaps = np.asarray(
        [abs(float(correctness[group].mean() - confidence[group].mean())) for group in groups]
    )
    weights = np.asarray([group.size / target.size for group in groups])
    one_hot = np.eye(probability.shape[1], dtype=np.float64)[target]
    brier = float(((probability - one_hot) ** 2).sum(axis=1).mean())
    selected = probability[np.arange(target.size), target].clip(1e-12, 1.0)
    nll = float(-np.log(selected).mean())
    return CalibrationResult(
        float(np.dot(weights, gaps)),
        float(gaps.max()),
        brier,
        nll,
        float(gaps.max()),
    )


def temperature_scale(logits: NDArray[np.float64], temperature: float) -> FloatArray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return softmax(np.asarray(logits, dtype=np.float64) / temperature)


def optimal_temperature(
    targets: Sequence[int], logits: NDArray[np.float64], candidates: Iterable[float]
) -> float:
    target = as_int(targets)
    values = np.asarray(logits, dtype=np.float64)
    best_temperature = 1.0
    best_loss = float("inf")
    for temperature in candidates:
        probability = temperature_scale(values, temperature)
        selected = probability[np.arange(target.size), target].clip(1e-12, 1.0)
        loss = float(-np.log(selected).mean())
        if loss < best_loss:
            best_temperature = temperature
            best_loss = loss
    return best_temperature


def fixed_effect_mean(estimates: Sequence[float], standard_errors: Sequence[float]) -> Interval:
    values = as_float(estimates)
    errors = as_float(standard_errors)
    if values.size != errors.size:
        raise ValueError("estimates and errors must have equal length")
    weights = 1.0 / np.maximum(errors**2, 1e-15)
    estimate = float(np.dot(weights, values) / weights.sum())
    standard_error = sqrt(1.0 / float(weights.sum()))
    return Interval(estimate, estimate - 1.96 * standard_error, estimate + 1.96 * standard_error)


def subgroup_gaps(
    targets: Sequence[int], scores: Sequence[float], groups: Sequence[str]
) -> dict[str, float]:
    target = as_int(targets)
    score = as_float(scores)
    group = np.asarray(groups, dtype=str)
    if not (target.size == score.size == group.size):
        raise ValueError("all inputs must have equal length")
    results: dict[str, float] = {}
    for name in np.unique(group):
        selection = group == name
        if np.unique(target[selection]).size == 2:
            results[str(name)] = binary_auc(target[selection], score[selection])
    return results


def maximum_subgroup_gap(values: Mapping[str, float]) -> float:
    scores = as_float(list(values.values()))
    return float(scores.max() - scores.min())


def bland_altman(targets: Sequence[float], predictions: Sequence[float]) -> tuple[float, float, float]:
    target = as_float(targets)
    prediction = as_float(predictions)
    if target.size != prediction.size:
        raise ValueError("targets and predictions must have equal length")
    differences = prediction - target
    bias = float(differences.mean())
    deviation = float(differences.std(ddof=1))
    return bias, bias - 1.96 * deviation, bias + 1.96 * deviation


def binomial_log_likelihood(successes: int, trials: int, probability: float) -> float:
    if not 0 <= successes <= trials:
        raise ValueError("successes must lie within trials")
    if not 0 < probability < 1:
        raise ValueError("probability must lie strictly between zero and one")
    combination = lgamma(trials + 1) - lgamma(successes + 1) - lgamma(trials - successes + 1)
    return combination + successes * log(probability) + (trials - successes) * log(1 - probability)


def auc_retention(reference: float, shifted: float) -> float:
    if reference <= 0:
        raise ValueError("reference AUC must be positive")
    return shifted / reference


def vocabulary_slope(sizes: Sequence[int], variances: Sequence[float]) -> float:
    size = np.log(as_float(sizes))
    variance = as_float(variances)
    if size.size != variance.size:
        raise ValueError("sizes and variances must have equal length")
    design = np.column_stack((size, np.ones(size.size)))
    coefficient, _ = np.linalg.lstsq(design, variance, rcond=None)[0]
    return float(coefficient)


def power_law_exponent(sizes: Sequence[int], aucs: Sequence[float]) -> float:
    size = np.log(as_float(sizes))
    error = np.log(1.0 - as_float(aucs))
    design = np.column_stack((size, np.ones(size.size)))
    coefficient, _ = np.linalg.lstsq(design, error, rcond=None)[0]
    return float(-coefficient)


def annealed_beta(epoch: int) -> float:
    return 0.5 * (1.0 - exp(-epoch / 10.0))


def annealed_gamma(epoch: int) -> float:
    return min(1.0, epoch / 20.0)
