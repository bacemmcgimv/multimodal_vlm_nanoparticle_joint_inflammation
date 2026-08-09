from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as functional


@dataclass(frozen=True)
class ModelOutput:
    ordinal_logits: Mapping[str, Tensor]
    latent: Tensor
    latent_mean: Tensor
    latent_log_variance: Tensor
    visual_embedding: Tensor
    phrase_embedding: Tensor
    attention: Tensor


class DropPath(nn.Module):
    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        self.probability = probability

    def forward(self, inputs: Tensor) -> Tensor:
        if self.probability == 0.0 or not self.training:
            return inputs
        keep = 1.0 - self.probability
        shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)
        mask = inputs.new_empty(shape).bernoulli_(keep)
        return inputs * mask / keep


class LayerNormChannels(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = inputs.permute(0, 2, 3, 4, 1)
        outputs = self.norm(outputs)
        return outputs.permute(0, 4, 1, 2, 3).contiguous()


class PatchEmbedding3D(nn.Module):
    def __init__(self, input_channels: int, embed_dim: int, patch_size: int = 2) -> None:
        super().__init__()
        self.projection = nn.Conv3d(
            input_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.normalization = LayerNormChannels(embed_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.normalization(self.projection(inputs))


def _window_partition(inputs: Tensor, window: tuple[int, int, int]) -> Tensor:
    batch, depth, height, width, channels = inputs.shape
    wd, wh, ww = window
    outputs = inputs.view(
        batch,
        depth // wd,
        wd,
        height // wh,
        wh,
        width // ww,
        ww,
        channels,
    )
    outputs = outputs.permute(0, 1, 3, 5, 2, 4, 6, 7)
    return outputs.contiguous().view(-1, wd * wh * ww, channels)


def _window_reverse(
    windows: Tensor,
    window: tuple[int, int, int],
    batch: int,
    depth: int,
    height: int,
    width: int,
) -> Tensor:
    wd, wh, ww = window
    channels = windows.shape[-1]
    outputs = windows.view(
        batch,
        depth // wd,
        height // wh,
        width // ww,
        wd,
        wh,
        ww,
        channels,
    )
    outputs = outputs.permute(0, 1, 4, 2, 5, 3, 6, 7)
    return outputs.contiguous().view(batch, depth, height, width, channels)


def _pad_to_window(
    inputs: Tensor, window: tuple[int, int, int]
) -> tuple[Tensor, tuple[int, int, int]]:
    _, depth, height, width, _ = inputs.shape
    wd, wh, ww = window
    pad_d = (wd - depth % wd) % wd
    pad_h = (wh - height % wh) % wh
    pad_w = (ww - width % ww) % ww
    channels_first = inputs.permute(0, 4, 1, 2, 3)
    channels_first = functional.pad(channels_first, (0, pad_w, 0, pad_h, 0, pad_d))
    return channels_first.permute(0, 2, 3, 4, 1), (depth, height, width)


class WindowAttention3D(nn.Module):
    def __init__(self, dimension: int, heads: int, window: tuple[int, int, int]) -> None:
        super().__init__()
        if dimension % heads != 0:
            raise ValueError("dimension must be divisible by heads")
        self.dimension = dimension
        self.heads = heads
        self.window = window
        self.head_dim = dimension // heads
        self.scale = self.head_dim**-0.5
        self.query_key_value = nn.Linear(dimension, dimension * 3)
        self.output = nn.Linear(dimension, dimension)
        wd, wh, ww = window
        count = (2 * wd - 1) * (2 * wh - 1) * (2 * ww - 1)
        self.relative_bias = nn.Parameter(torch.zeros(count, heads))
        coordinates = torch.stack(
            torch.meshgrid(
                torch.arange(wd),
                torch.arange(wh),
                torch.arange(ww),
                indexing="ij",
            )
        )
        flattened = coordinates.flatten(1)
        relative = flattened[:, :, None] - flattened[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[:, :, 0] += wd - 1
        relative[:, :, 1] += wh - 1
        relative[:, :, 2] += ww - 1
        relative[:, :, 0] *= (2 * wh - 1) * (2 * ww - 1)
        relative[:, :, 1] *= 2 * ww - 1
        self.register_buffer("relative_index", relative.sum(-1), persistent=False)
        nn.init.trunc_normal_(self.relative_bias, std=0.02)

    def forward(self, inputs: Tensor) -> Tensor:
        batch, tokens, channels = inputs.shape
        qkv = self.query_key_value(inputs)
        qkv = qkv.view(batch, tokens, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        query = query * self.scale
        scores = query @ key.transpose(-2, -1)
        index = self.relative_index.reshape(-1)
        bias = self.relative_bias[index]
        bias = bias.view(tokens, tokens, self.heads).permute(2, 0, 1)
        scores = scores + bias.unsqueeze(0)
        weights = scores.softmax(dim=-1)
        outputs = weights @ value
        outputs = outputs.transpose(1, 2).reshape(batch, tokens, channels)
        return self.output(outputs)


class FeedForward(nn.Module):
    def __init__(self, dimension: int, expansion: int = 4) -> None:
        super().__init__()
        hidden = dimension * expansion
        self.layers = nn.Sequential(
            nn.Linear(dimension, hidden),
            nn.GELU(),
            nn.Linear(hidden, dimension),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class SwinBlock3D(nn.Module):
    def __init__(
        self,
        dimension: int,
        heads: int,
        window: tuple[int, int, int],
        shifted: bool,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.window = window
        self.shifted = shifted
        self.normalization_one = nn.LayerNorm(dimension)
        self.attention = WindowAttention3D(dimension, heads, window)
        self.path_one = DropPath(drop_path)
        self.normalization_two = nn.LayerNorm(dimension)
        self.feed_forward = FeedForward(dimension)
        self.path_two = DropPath(drop_path)

    def forward(self, inputs: Tensor) -> Tensor:
        batch, depth, height, width, _ = inputs.shape
        shortcut = inputs
        normalized = self.normalization_one(inputs)
        padded, original = _pad_to_window(normalized, self.window)
        _, pd, ph, pw, _ = padded.shape
        shifts = tuple(size // 2 for size in self.window)
        if self.shifted:
            padded = torch.roll(padded, tuple(-value for value in shifts), dims=(1, 2, 3))
        windows = _window_partition(padded, self.window)
        windows = self.attention(windows)
        outputs = _window_reverse(windows, self.window, batch, pd, ph, pw)
        if self.shifted:
            outputs = torch.roll(outputs, shifts, dims=(1, 2, 3))
        od, oh, ow = original
        outputs = outputs[:, :od, :oh, :ow]
        outputs = shortcut + self.path_one(outputs)
        outputs = outputs + self.path_two(self.feed_forward(self.normalization_two(outputs)))
        if outputs.shape[1:4] != (depth, height, width):
            raise RuntimeError("spatial shape changed inside attention block")
        return outputs


class PatchMerging3D(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(dimension * 8)
        self.reduction = nn.Linear(dimension * 8, dimension * 2, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        batch, depth, height, width, channels = inputs.shape
        pad_d = depth % 2
        pad_h = height % 2
        pad_w = width % 2
        if pad_d or pad_h or pad_w:
            channels_first = inputs.permute(0, 4, 1, 2, 3)
            channels_first = functional.pad(channels_first, (0, pad_w, 0, pad_h, 0, pad_d))
            inputs = channels_first.permute(0, 2, 3, 4, 1)
        parts = [
            inputs[:, d::2, h::2, w::2, :]
            for d in range(2)
            for h in range(2)
            for w in range(2)
        ]
        merged = torch.cat(parts, dim=-1)
        return self.reduction(self.normalization(merged))


class SwinStage3D(nn.Module):
    def __init__(
        self,
        dimension: int,
        depth: int,
        heads: int,
        window: tuple[int, int, int],
        drop_paths: Sequence[float],
        downsample: bool,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            SwinBlock3D(dimension, heads, window, index % 2 == 1, drop_paths[index])
            for index in range(depth)
        )
        self.downsample = PatchMerging3D(dimension) if downsample else nn.Identity()

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        for block in self.blocks:
            inputs = block(inputs)
        return inputs, self.downsample(inputs)


class VisionEncoder3D(nn.Module):
    def __init__(
        self,
        input_channels: int,
        embed_dim: int,
        depths: Sequence[int],
        heads: Sequence[int],
        window: tuple[int, int, int],
        drop_path: float = 0.2,
    ) -> None:
        super().__init__()
        self.patch_embedding = PatchEmbedding3D(input_channels, embed_dim)
        total = sum(depths)
        probabilities = torch.linspace(0, drop_path, total).tolist()
        stages: list[nn.Module] = []
        cursor = 0
        for index, (depth, head_count) in enumerate(zip(depths, heads)):
            dimension = embed_dim * 2**index
            stages.append(
                SwinStage3D(
                    dimension,
                    depth,
                    head_count,
                    window,
                    probabilities[cursor : cursor + depth],
                    index < len(depths) - 1,
                )
            )
            cursor += depth
        self.stages = nn.ModuleList(stages)
        self.output_dim = embed_dim * 2 ** (len(depths) - 1)
        self.output_norm = nn.LayerNorm(self.output_dim)

    def forward(self, inputs: Tensor) -> tuple[Tensor, tuple[Tensor, ...]]:
        features = self.patch_embedding(inputs).permute(0, 2, 3, 4, 1)
        pyramid: list[Tensor] = []
        for stage in self.stages:
            current, features = stage(features)
            pyramid.append(current.permute(0, 4, 1, 2, 3).contiguous())
        tokens = current.flatten(1, 3)
        return self.output_norm(tokens), tuple(pyramid)


class VariationalBottleneck(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.mean = nn.Linear(input_dim, latent_dim)
        self.log_variance = nn.Linear(input_dim, latent_dim)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        pooled = inputs.mean(dim=1)
        mean = self.mean(pooled)
        log_variance = self.log_variance(pooled).clamp(-12.0, 8.0)
        if self.training:
            noise = torch.randn_like(mean)
            latent = mean + noise * torch.exp(0.5 * log_variance)
        else:
            latent = mean
        return latent, mean, log_variance


class FrozenPhraseBank(nn.Module):
    def __init__(self, embeddings: Tensor, phrases: Sequence[str]) -> None:
        super().__init__()
        if embeddings.ndim != 2:
            raise ValueError("phrase embeddings must be two-dimensional")
        if len(phrases) != embeddings.shape[0]:
            raise ValueError("phrase and embedding counts differ")
        normalized = functional.normalize(embeddings.float(), dim=-1)
        self.register_buffer("embeddings", normalized)
        self.phrases = tuple(phrases)

    def forward(self, indices: Tensor | None = None) -> Tensor:
        if indices is None:
            return self.embeddings
        return self.embeddings[indices]


class CrossAttentionBridge(nn.Module):
    def __init__(self, visual_dim: int, phrase_dim: int, latent_dim: int, heads: int) -> None:
        super().__init__()
        if latent_dim % heads != 0:
            raise ValueError("latent dimension must be divisible by heads")
        self.visual_projection = nn.Linear(visual_dim, latent_dim)
        self.phrase_projection = nn.Linear(phrase_dim, latent_dim)
        self.attention = nn.MultiheadAttention(latent_dim, heads, batch_first=True)
        self.output_norm = nn.LayerNorm(latent_dim)

    def forward(self, visual_tokens: Tensor, phrases: Tensor) -> tuple[Tensor, Tensor]:
        query = self.visual_projection(visual_tokens)
        keys = self.phrase_projection(phrases)
        if keys.ndim == 2:
            keys = keys.unsqueeze(0).expand(query.shape[0], -1, -1)
        outputs, weights = self.attention(query, keys, keys, need_weights=True)
        pooled = self.output_norm(outputs + query).mean(dim=1)
        return pooled, weights


class CoralHead(nn.Module):
    def __init__(self, input_dim: int, grades: int) -> None:
        super().__init__()
        if grades < 2:
            raise ValueError("CORAL requires at least two grades")
        self.score = nn.Linear(input_dim, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(grades - 1))
        self.grades = grades

    def forward(self, inputs: Tensor) -> Tensor:
        return self.score(inputs) + self.bias


class MultiFeatureOrdinalHead(nn.Module):
    def __init__(self, input_dim: int, grade_counts: Mapping[str, int]) -> None:
        super().__init__()
        self.heads = nn.ModuleDict(
            {name: CoralHead(input_dim, count) for name, count in grade_counts.items()}
        )

    def forward(self, inputs: Tensor) -> dict[str, Tensor]:
        return {name: head(inputs) for name, head in self.heads.items()}


class LamssVLM(nn.Module):
    def __init__(
        self,
        input_channels: int,
        embed_dim: int,
        latent_dim: int,
        depths: Sequence[int],
        heads: Sequence[int],
        window: tuple[int, int, int],
        phrase_embeddings: Tensor,
        phrases: Sequence[str],
        grade_counts: Mapping[str, int],
    ) -> None:
        super().__init__()
        self.vision = VisionEncoder3D(input_channels, embed_dim, depths, heads, window)
        self.bottleneck = VariationalBottleneck(self.vision.output_dim, latent_dim)
        self.phrases = FrozenPhraseBank(phrase_embeddings, phrases)
        bridge_heads = next(value for value in reversed(heads) if latent_dim % value == 0)
        self.bridge = CrossAttentionBridge(
            self.vision.output_dim,
            phrase_embeddings.shape[1],
            latent_dim,
            bridge_heads,
        )
        self.latent_projection = nn.Linear(latent_dim, latent_dim)
        self.phrase_projection = nn.Linear(phrase_embeddings.shape[1], latent_dim)
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
        )
        self.ordinal = MultiFeatureOrdinalHead(latent_dim, grade_counts)

    def forward(self, volumes: Tensor, phrase_indices: Tensor) -> ModelOutput:
        tokens, _ = self.vision(volumes)
        latent, mean, log_variance = self.bottleneck(tokens)
        phrase_bank = self.phrases()
        attended, attention = self.bridge(tokens, phrase_bank)
        fused = self.fusion(torch.cat((self.latent_projection(latent), attended), dim=-1))
        assigned_phrases = self.phrases(phrase_indices)
        phrase_embedding = functional.normalize(self.phrase_projection(assigned_phrases), dim=-1)
        visual_embedding = functional.normalize(fused, dim=-1)
        return ModelOutput(
            ordinal_logits=self.ordinal(fused),
            latent=latent,
            latent_mean=mean,
            latent_log_variance=log_variance,
            visual_embedding=visual_embedding,
            phrase_embedding=phrase_embedding,
            attention=attention,
        )
