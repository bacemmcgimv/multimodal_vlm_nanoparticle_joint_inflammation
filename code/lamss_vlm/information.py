from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
from torch import Tensor, nn
from torch.autograd import Function
from torch.nn import functional as functional


@dataclass(frozen=True)
class MutualInformationEstimate:
    value: Tensor
    stable: bool
    method: str


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx: object, inputs: Tensor, scale: float) -> Tensor:
        setattr(ctx, "scale", scale)
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx: object, gradient: Tensor) -> tuple[Tensor, None]:
        return -float(getattr(ctx, "scale")) * gradient, None


class GradientReversal(nn.Module):
    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, inputs: Tensor) -> Tensor:
        return GradientReversalFunction.apply(inputs, self.scale)


class SeparableCritic(nn.Module):
    def __init__(self, latent_dim: int, site_count: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.latent_network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.site_embedding = nn.Embedding(site_count, hidden_dim)
        self.site_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.scale = hidden_dim**-0.5

    def forward(self, latent: Tensor, sites: Tensor) -> Tensor:
        latent_features = self.latent_network(latent)
        site_features = self.site_network(self.site_embedding(sites))
        return (latent_features * site_features).sum(dim=-1) * self.scale


class MineEstimator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        site_count: int,
        hidden_dim: int = 256,
        moving_average_rate: float = 0.01,
    ) -> None:
        super().__init__()
        self.critic = SeparableCritic(latent_dim, site_count, hidden_dim)
        self.moving_average_rate = moving_average_rate
        self.register_buffer("exponential_mean", torch.ones(()))
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))

    def joint_scores(self, latent: Tensor, sites: Tensor) -> Tensor:
        return self.critic(latent, sites)

    def marginal_scores(self, latent: Tensor, sites: Tensor) -> Tensor:
        if latent.shape[0] < 2:
            return self.critic(latent, sites)
        permutation = torch.randperm(sites.shape[0], device=sites.device)
        return self.critic(latent, sites[permutation])

    def estimate(self, latent: Tensor, sites: Tensor, update_average: bool = True) -> Tensor:
        joint = self.joint_scores(latent, sites)
        marginal = self.marginal_scores(latent, sites)
        marginal_exponential = marginal.exp().mean().clamp_min(1e-12)
        if update_average and self.training:
            with torch.no_grad():
                rate = self.moving_average_rate
                self.exponential_mean.mul_(1.0 - rate).add_(marginal_exponential * rate)
                self.updates.add_(1)
        return joint.mean() - marginal_exponential.log()

    def learning_loss(self, latent: Tensor, sites: Tensor) -> Tensor:
        joint = self.joint_scores(latent.detach(), sites)
        marginal = self.marginal_scores(latent.detach(), sites)
        exponential = marginal.exp().mean().clamp_min(1e-12)
        correction = exponential / self.exponential_mean.detach().clamp_min(1e-12)
        return -(joint.mean() - correction)

    def forward(self, latent: Tensor, sites: Tensor) -> Tensor:
        return self.estimate(latent, sites)


class ClubEstimator(nn.Module):
    def __init__(self, latent_dim: int, site_count: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.site_embedding = nn.Embedding(site_count, hidden_dim)
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_variance = nn.Linear(hidden_dim, latent_dim)

    def parameters_for(self, sites: Tensor) -> tuple[Tensor, Tensor]:
        features = self.trunk(self.site_embedding(sites))
        return self.mean(features), self.log_variance(features).clamp(-12.0, 8.0)

    def log_likelihood(self, latent: Tensor, sites: Tensor) -> Tensor:
        mean, log_variance = self.parameters_for(sites)
        values = -0.5 * (log_variance + (latent - mean).square() / log_variance.exp())
        return values.sum(dim=-1)

    def learning_loss(self, latent: Tensor, sites: Tensor) -> Tensor:
        return -self.log_likelihood(latent.detach(), sites).mean()

    def forward(self, latent: Tensor, sites: Tensor) -> Tensor:
        positive = self.log_likelihood(latent, sites)
        if latent.shape[0] < 2:
            return positive.new_zeros(())
        negative_values = []
        for shift in range(1, latent.shape[0]):
            negative_values.append(self.log_likelihood(latent, sites.roll(shift)))
        negative = torch.stack(negative_values).mean(dim=0)
        return (positive - negative).mean().clamp_min(0.0)


class StableInformationEstimator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        site_count: int,
        maximum_absolute_value: float = 20.0,
    ) -> None:
        super().__init__()
        self.mine = MineEstimator(latent_dim, site_count)
        self.club = ClubEstimator(latent_dim, site_count)
        self.maximum_absolute_value = maximum_absolute_value
        self.register_buffer("mine_failures", torch.zeros((), dtype=torch.long))
        self.register_buffer("club_uses", torch.zeros((), dtype=torch.long))

    def mine_is_stable(self, value: Tensor) -> bool:
        return bool(torch.isfinite(value).all()) and abs(float(value.detach())) <= self.maximum_absolute_value

    def estimate(self, latent: Tensor, sites: Tensor) -> MutualInformationEstimate:
        mine_value = self.mine(latent, sites)
        stable = self.mine_is_stable(mine_value)
        if stable:
            return MutualInformationEstimate(mine_value, True, "mine")
        self.mine_failures.add_(1)
        self.club_uses.add_(1)
        club_value = self.club(latent, sites)
        return MutualInformationEstimate(club_value, False, "club")

    def estimator_losses(self, latent: Tensor, sites: Tensor) -> tuple[Tensor, Tensor]:
        return self.mine.learning_loss(latent, sites), self.club.learning_loss(latent, sites)

    def learning_loss(self, latent: Tensor, sites: Tensor) -> Tensor:
        mine_loss, club_loss = self.estimator_losses(latent, sites)
        return mine_loss + club_loss

    def forward(self, latent: Tensor, sites: Tensor) -> Tensor:
        return self.estimate(latent, sites).value


def log_mean_exp(values: Tensor, dimension: int = 0) -> Tensor:
    maximum = values.max(dim=dimension, keepdim=True).values
    return maximum.squeeze(dimension) + (values - maximum).exp().mean(dim=dimension).log()


def categorical_information(scores: Tensor, labels: Tensor, temperature: float = 0.07) -> Tensor:
    if scores.ndim != 2:
        raise ValueError("scores must be a matrix")
    if labels.ndim != 1 or labels.shape[0] != scores.shape[0]:
        raise ValueError("labels must align with scores")
    logits = scores / temperature
    conditional = functional.cross_entropy(logits, labels)
    marginal = -(logits.log_softmax(dim=-1).exp().mean(dim=0).clamp_min(1e-12).log()).mean()
    return marginal - conditional


def cosine_grounding_scores(latent: Tensor, text: Tensor) -> Tensor:
    if latent.ndim != 2 or text.ndim != 2:
        raise ValueError("latent and text inputs must be matrices")
    if latent.shape[1] != text.shape[1]:
        raise ValueError("latent and text dimensions must match")
    normalized_latent = functional.normalize(latent, dim=-1)
    normalized_text = functional.normalize(text, dim=-1)
    return normalized_latent @ normalized_text.transpose(0, 1)


def language_information_loss(
    latent: Tensor,
    text: Tensor,
    phrase_labels: Tensor,
    temperature: float = 0.07,
) -> Tensor:
    scores = cosine_grounding_scores(latent, text)
    return functional.cross_entropy(scores / temperature, phrase_labels)


def gaussian_information_budget(mean: Tensor, log_variance: Tensor) -> Tensor:
    if mean.shape != log_variance.shape:
        raise ValueError("mean and log variance must have equal shapes")
    return 0.5 * (mean.square() + log_variance.exp() - log_variance - 1.0).sum(dim=-1).mean()


def beta_schedule(epochs: int) -> Iterator[float]:
    for epoch in range(epochs):
        yield 0.5 * (1.0 - torch.exp(torch.tensor(-epoch / 10.0)).item())


def gamma_schedule(epochs: int) -> Iterator[float]:
    for epoch in range(epochs):
        yield min(1.0, epoch / 20.0)


def invariance_bound(gamma: float, residual_information: float, text_leakage: float) -> float:
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if residual_information < 0 or text_leakage < 0:
        raise ValueError("information values must be nonnegative")
    return 2.0 * (residual_information / gamma) ** 0.5 + text_leakage
