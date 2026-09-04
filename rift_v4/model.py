from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class CorrectedAttention(nn.Module):
    """Masked fused attention with conventional Transformer parameterization."""

    def __init__(self, dim: int, head_dim: int) -> None:
        super().__init__()
        if dim % head_dim:
            raise ValueError("dim must be divisible by head_dim")
        self.head_dim = head_dim
        self.heads = dim // head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_norm = nn.LayerNorm(head_dim, elementwise_affine=False)
        self.k_norm = nn.LayerNorm(head_dim, elementwise_affine=False)
        self.output = nn.Linear(dim, dim, bias=False)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        batch, frames, dim = x.shape
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = self.q_norm(q.view(batch, frames, self.heads, self.head_dim)).transpose(
            1, 2
        )
        k = self.k_norm(k.view(batch, frames, self.heads, self.head_dim)).transpose(
            1, 2
        )
        v = v.view(batch, frames, self.heads, self.head_dim).transpose(1, 2)
        q, k = _rotary(q, k)
        attention_mask = mask[:, None, None, :] if mask is not None else None
        attended = (
            F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=0.0,
                scale=self.scale,
            )
            .transpose(1, 2)
            .reshape(batch, frames, dim)
        )
        if mask is not None:
            attended = attended * mask.unsqueeze(-1)
        return self.output(attended)


class ConvFeedForward(nn.Module):
    def __init__(self, dim: int, hidden: int, kernel_size: int) -> None:
        super().__init__()
        self.input = nn.Linear(dim, hidden * 2)
        self.conv = nn.Conv1d(
            hidden, hidden, kernel_size, padding=kernel_size // 2, groups=hidden
        )
        self.output = nn.Linear(hidden, dim)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)
        value, gate = self.input(x).chunk(2, dim=-1)
        if mask is not None:
            numeric_mask = mask.unsqueeze(-1).to(value.dtype)
            value = value * numeric_mask
            gate = gate * numeric_mask
        value = self.conv(value.transpose(1, 2)).transpose(1, 2)
        output = self.output((value * F.silu(gate)).contiguous())
        if mask is not None:
            output = output * mask.unsqueeze(-1).to(output.dtype)
        return output


class AdaLNBlock(nn.Module):
    def __init__(
        self, dim: int, head_dim: int, ff_hidden_dim: int, kernel_size: int
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attention = CorrectedAttention(dim, head_dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.feed_forward = ConvFeedForward(dim, ff_hidden_dim, kernel_size)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.attention_residual_scale = 1.0
        self.ffn_residual_scale = 1.0
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x: Tensor, conditioning: Tensor, mask: Tensor | None) -> Tensor:
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.modulation(
            conditioning
        ).chunk(6, dim=-1)
        attended = self.attention(_modulate(self.norm1(x), shift_a, scale_a), mask)
        x = x + self.attention_residual_scale * gate_a * attended
        fed = self.feed_forward(_modulate(self.norm2(x), shift_f, scale_f), mask)
        x = x + self.ffn_residual_scale * gate_f * fed
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        return x


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int, frequency_dim: int = 256) -> None:
        super().__init__()
        self.frequency_dim = frequency_dim
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

    def forward(self, timestep: Tensor) -> Tensor:
        half = self.frequency_dim // 2
        frequencies = torch.exp(
            -math.log(10_000) * torch.arange(half, device=timestep.device) / half
        )
        phases = (timestep.float() * 1000.0)[:, None] * frequencies[None]
        return self.mlp(torch.cat((phases.cos(), phases.sin()), dim=-1))


class RIFTV4(nn.Module):
    def __init__(
        self,
        mel_channels: int,
        content_dim: int,
        num_speakers: int,
        dim: int,
        depth: int,
        head_dim: int,
        ff_hidden_dim: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        self.mel_input = nn.Linear(mel_channels, dim)
        self.content_input = nn.Linear(content_dim, dim)
        self.pitch_input = nn.Sequential(
            nn.Linear(2, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.rms_input = nn.Linear(1, dim)
        self.speaker = nn.Embedding(num_speakers + 1, dim)
        self.null_speaker_id = num_speakers
        self.time = TimestepEmbedding(dim)
        self.blocks = nn.ModuleList(
            [
                AdaLNBlock(dim, head_dim, ff_hidden_dim, kernel_size)
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 2))
        self.output = nn.Linear(dim, mel_channels)
        self.apply(_initialize_weights)
        for block in self.blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        noisy_mel: Tensor,
        content: Tensor,
        f0: Tensor,
        rms: Tensor,
        speaker: Tensor,
        timestep: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        voiced = (f0 > 0).to(f0.dtype)
        log_f0 = torch.where(voiced.bool(), f0.clamp_min(1).log2() / 10.0, 0.0)
        x = (
            self.mel_input(noisy_mel)
            + self.content_input(content)
            + self.pitch_input(torch.cat((log_f0, voiced), dim=-1))
            + self.rms_input(rms)
        )
        conditioning = self.time(timestep) + self.speaker(speaker)
        conditioning = conditioning[:, None, :]
        for block in self.blocks:
            x = block(x, conditioning, mask)
        shift, scale = self.final_modulation(conditioning).chunk(2, dim=-1)
        output = self.output(_modulate(self.final_norm(x), shift, scale))
        if mask is not None:
            output = output * mask.unsqueeze(-1).to(output.dtype)
        return output


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


def _initialize_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=0.02)


def _rotary(q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
    dimension = q.shape[-1]
    if dimension % 2:
        raise ValueError("head_dim must be even for rotary embeddings")
    positions = torch.arange(q.shape[-2], device=q.device, dtype=torch.float32)
    frequencies = torch.exp(
        -math.log(10_000)
        * torch.arange(0, dimension, 2, device=q.device, dtype=torch.float32)
        / dimension
    )
    angles = positions[:, None] * frequencies[None]
    cosine = angles.cos().to(q.dtype)[None, None]
    sine = angles.sin().to(q.dtype)[None, None]
    return _apply_rotary(q, cosine, sine), _apply_rotary(k, cosine, sine)


def _apply_rotary(x: Tensor, cosine: Tensor, sine: Tensor) -> Tensor:
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
    ).flatten(-2)
