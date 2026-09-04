from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .model import RIFTV4


@dataclass
class FlowLoss:
    total: Tensor
    flow: Tensor
    flow_by_sample: Tensor


class FlowMatchingSystem(nn.Module):
    def __init__(
        self,
        model: RIFTV4,
        speaker_drop_probability: float,
    ) -> None:
        super().__init__()
        self.model = model
        self.speaker_drop_probability = speaker_drop_probability

    def forward(self, batch: dict[str, Tensor]) -> FlowLoss:
        mel = batch["mel"]
        mask = batch["mask"]
        speaker = batch["speaker"]

        timestep = _sample_timestep(mel.shape[0], mel.device, mel.dtype)
        noise = torch.randn_like(mel)
        expanded_t = timestep[:, None, None]
        noisy = (1 - expanded_t) * noise + expanded_t * mel
        target_velocity = mel - noise
        effective_speaker = speaker.clone()
        if self.training and self.speaker_drop_probability:
            dropped = (
                torch.rand(speaker.shape, device=speaker.device)
                < self.speaker_drop_probability
            )
            effective_speaker[dropped] = self.model.null_speaker_id
        prediction = self.model(
            noisy,
            batch["content"],
            batch["f0"],
            batch["rms"],
            effective_speaker,
            timestep,
            mask,
        )
        weights = mask.unsqueeze(-1).to(mel.dtype)
        squared = (prediction - target_velocity).square() * weights
        flow_by_sample = squared.sum(dim=(1, 2)) / (
            weights.sum(dim=(1, 2)).clamp_min(1) * mel.shape[-1]
        )
        flow = squared.sum() / (weights.sum().clamp_min(1) * mel.shape[-1])
        return FlowLoss(total=flow, flow=flow, flow_by_sample=flow_by_sample)

    @torch.inference_mode()
    def sample(
        self,
        content: Tensor,
        f0: Tensor,
        rms: Tensor,
        speaker: Tensor,
        mask: Tensor,
        mel_channels: int,
        steps: int = 32,
        guidance_strength: float = 1.0,
        method: str = "heun",
        generator: torch.Generator | None = None,
        time_schedule: str = "cosine",
        initial_noise: Tensor | None = None,
    ) -> Tensor:
        if steps <= 0:
            raise ValueError("inference steps must be positive")
        if method not in {"euler", "heun"}:
            raise ValueError("method must be euler or heun")
        if time_schedule not in {"linear", "cosine"}:
            raise ValueError("time schedule must be linear or cosine")
        expected_shape = (content.shape[0], content.shape[1], mel_channels)
        if initial_noise is None:
            state = torch.randn(
                expected_shape,
                device=content.device,
                dtype=content.dtype,
                generator=generator,
            )
        else:
            if initial_noise.shape != expected_shape:
                raise ValueError("initial noise shape does not match conditioning")
            state = initial_noise.to(device=content.device, dtype=content.dtype)
        state = state * mask.unsqueeze(-1).to(state.dtype)
        times = _time_grid(steps, state.device, time_schedule)
        for index in range(steps):
            time = times[index].expand(state.shape[0]).to(state.dtype)
            delta = (times[index + 1] - times[index]).to(state.dtype)
            velocity = self._guided_velocity(
                state, content, f0, rms, speaker, time, mask, guidance_strength
            )
            proposal = state + delta * velocity
            if method == "heun" and index + 1 < steps:
                next_time = times[index + 1].expand(state.shape[0]).to(state.dtype)
                next_velocity = self._guided_velocity(
                    proposal,
                    content,
                    f0,
                    rms,
                    speaker,
                    next_time,
                    mask,
                    guidance_strength,
                )
                state = state + delta * 0.5 * (velocity + next_velocity)
            else:
                state = proposal
            state = state * mask.unsqueeze(-1).to(state.dtype)
        return state

    def _guided_velocity(
        self,
        state: Tensor,
        content: Tensor,
        f0: Tensor,
        rms: Tensor,
        speaker: Tensor,
        timestep: Tensor,
        mask: Tensor,
        strength: float,
    ) -> Tensor:
        conditional = self.model(state, content, f0, rms, speaker, timestep, mask)
        if strength == 1.0:
            return conditional
        null_speaker = torch.full_like(speaker, self.model.null_speaker_id)
        unconditional = self.model(
            state, content, f0, rms, null_speaker, timestep, mask
        )
        return unconditional + strength * (conditional - unconditional)


def _sample_timestep(batch: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    # Stratify the base uniform variable, then transform it to logit-normal.
    work_dtype = torch.float32
    quantiles = torch.arange(batch, device=device, dtype=work_dtype) / batch
    uniform = quantiles + torch.rand(batch, device=device, dtype=work_dtype) / batch
    epsilon = torch.finfo(work_dtype).eps
    uniform = uniform.clamp(epsilon, 1.0 - epsilon)
    normal = torch.erfinv(uniform.mul(2).sub(1)).mul(math.sqrt(2.0))
    return torch.sigmoid(normal)[torch.randperm(batch, device=device)].to(dtype)


def _time_grid(steps: int, device: torch.device, schedule: str) -> Tensor:
    base = torch.linspace(0, 1, steps + 1, device=device)
    if schedule == "linear":
        return base
    if schedule == "cosine":
        return 1 - torch.cos(base * (math.pi / 2))
    raise ValueError("time schedule must be linear or cosine")
