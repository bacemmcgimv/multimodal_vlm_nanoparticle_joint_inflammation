from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from lamss_vlm.model import ModelOutput


@dataclass(frozen=True)
class LossBreakdown:
    total: Tensor
    ordinal: Tensor
    kl: Tensor
    language: Tensor
    site: Tensor


def coral_levels(targets: Tensor, grade_count: int) -> Tensor:
    thresholds = torch.arange(grade_count - 1, device=targets.device)
    return (targets.unsqueeze(-1) > thresholds).to(dtype=torch.float32)


def coral_loss(logits: Tensor, targets: Tensor, grade_count: int) -> Tensor:
    levels = coral_levels(targets, grade_count)
    return functional.binary_cross_entropy_with_logits(logits.float(), levels)


def multi_feature_coral_loss(
    logits: Mapping[str, Tensor],
    targets: Tensor,
    feature_order: Sequence[str],
    grade_counts: Mapping[str, int],
) -> Tensor:
    if targets.ndim != 2 or targets.shape[1] != len(feature_order):
        raise ValueError("ordinal targets do not match feature order")
    losses = [
        coral_loss(logits[name], targets[:, index], grade_counts[name])
        for index, name in enumerate(feature_order)
    ]
    return torch.stack(losses).mean()


def gaussian_kl(mean: Tensor, log_variance: Tensor) -> Tensor:
    values = -0.5 * (1.0 + log_variance - mean.square() - log_variance.exp())
    return values.sum(dim=-1).mean()


def language_alignment_loss(visual: Tensor, phrase: Tensor, temperature: float = 0.07) -> Tensor:
    if visual.shape != phrase.shape:
        raise ValueError("visual and phrase embeddings must have equal shapes")
    scores = visual @ phrase.transpose(0, 1) / temperature
    targets = torch.arange(scores.shape[0], device=scores.device)
    visual_loss = functional.cross_entropy(scores, targets)
    phrase_loss = functional.cross_entropy(scores.transpose(0, 1), targets)
    return 0.5 * (visual_loss + phrase_loss)


class ConditionalGaussian(nn.Module):
    def __init__(self, latent_dim: int, site_count: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden = hidden_dim or latent_dim * 2
        self.site_embedding = nn.Embedding(site_count, hidden)
        self.network = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.mean = nn.Linear(hidden, latent_dim)
        self.log_variance = nn.Linear(hidden, latent_dim)

    def parameters_for(self, sites: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.network(self.site_embedding(sites))
        return self.mean(hidden), self.log_variance(hidden).clamp(-12.0, 8.0)

    def log_likelihood(self, latent: Tensor, sites: Tensor) -> Tensor:
        mean, log_variance = self.parameters_for(sites)
        normalizer = torch.log(torch.tensor(2.0 * torch.pi, device=latent.device))
        values = -0.5 * (
            normalizer + log_variance + (latent - mean).square() / log_variance.exp()
        )
        return values.sum(dim=-1)


class ClubSiteEstimator(nn.Module):
    def __init__(self, latent_dim: int, site_count: int) -> None:
        super().__init__()
        self.conditional = ConditionalGaussian(latent_dim, site_count)

    def learning_loss(self, latent: Tensor, sites: Tensor) -> Tensor:
        return -self.conditional.log_likelihood(latent.detach(), sites).mean()

    def forward(self, latent: Tensor, sites: Tensor) -> Tensor:
        batch = latent.shape[0]
        if batch < 2:
            return latent.new_zeros(())
        positive = self.conditional.log_likelihood(latent, sites)
        permutations = []
        for shift in range(1, batch):
            shifted_sites = sites.roll(shifts=shift, dims=0)
            permutations.append(self.conditional.log_likelihood(latent, shifted_sites))
        negative = torch.stack(permutations, dim=0).mean(dim=0)
        return (positive - negative).mean().clamp_min(0.0)


class LamssObjective(nn.Module):
    def __init__(
        self,
        feature_order: Sequence[str],
        grade_counts: Mapping[str, int],
        beta_kl: float,
        beta_text: float,
        gamma_site: float,
    ) -> None:
        super().__init__()
        self.feature_order = tuple(feature_order)
        self.grade_counts = dict(grade_counts)
        self.beta_kl = beta_kl
        self.beta_text = beta_text
        self.gamma_site = gamma_site

    def forward(
        self,
        output: ModelOutput,
        targets: Tensor,
        site_information: Tensor,
    ) -> LossBreakdown:
        ordinal = multi_feature_coral_loss(
            output.ordinal_logits,
            targets,
            self.feature_order,
            self.grade_counts,
        )
        kl = gaussian_kl(output.latent_mean, output.latent_log_variance)
        language = language_alignment_loss(output.visual_embedding, output.phrase_embedding)
        total = ordinal + self.beta_kl * kl + self.beta_text * language
        total = total + self.gamma_site * site_information
        return LossBreakdown(total, ordinal, kl, language, site_information)


def ordinal_predictions(logits: Tensor) -> Tensor:
    return (logits.sigmoid() >= 0.5).sum(dim=-1)


def quadratic_weighted_kappa(
    predictions: Tensor,
    targets: Tensor,
    grades: int,
) -> Tensor:
    predictions = predictions.long().flatten()
    targets = targets.long().flatten()
    observed = torch.zeros((grades, grades), dtype=torch.float64, device=targets.device)
    indices = targets * grades + predictions
    observed.flatten().scatter_add_(0, indices, torch.ones_like(indices, dtype=torch.float64))
    target_histogram = torch.bincount(targets, minlength=grades).double()
    prediction_histogram = torch.bincount(predictions, minlength=grades).double()
    expected = target_histogram[:, None] * prediction_histogram[None, :]
    expected = expected / expected.sum().clamp_min(1.0)
    observed = observed / observed.sum().clamp_min(1.0)
    coordinates = torch.arange(grades, dtype=torch.float64, device=targets.device)
    weights = (coordinates[:, None] - coordinates[None, :]).square()
    weights = weights / max(1, (grades - 1) ** 2)
    numerator = (weights * observed).sum()
    denominator = (weights * expected).sum().clamp_min(1e-12)
    return (1.0 - numerator / denominator).float()
