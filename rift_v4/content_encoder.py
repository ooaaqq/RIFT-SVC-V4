from __future__ import annotations

from pathlib import Path
from typing import Any

import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoModel


class FrozenContentEncoder(nn.Module):
    """Pinned ContentVec used only to produce immutable raw hidden states."""

    def __init__(self, backbone: nn.Module, content_dim: int) -> None:
        super().__init__()
        self.backbone = backbone
        hidden_size = (
            int(backbone.config.hidden_size)
            if hasattr(backbone, "config")
            else content_dim
        )
        if hidden_size != content_dim:
            raise ValueError(
                f"ContentVec hidden size {hidden_size} does not match {content_dim}"
            )
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.eval()

    @classmethod
    def from_local_pretrained(
        cls,
        model_path: str | Path,
        content_dim: int,
    ) -> FrozenContentEncoder:
        backbone = AutoModel.from_pretrained(
            str(Path(model_path)), local_files_only=True, trust_remote_code=False
        )
        return cls(backbone, content_dim)

    def train(self, mode: bool = True) -> FrozenContentEncoder:
        # Frozen SSL layers must not silently re-enable dropout or layerdrop.
        super().train(False)
        return self

    def forward(self, waveform: Tensor, waveform_mask: Tensor) -> tuple[Tensor, Tensor]:
        output: Any = self.backbone(
            input_values=waveform,
            attention_mask=waveform_mask.long(),
            return_dict=True,
        )
        hidden = output.last_hidden_state
        mask_builder = getattr(
            self.backbone, "_get_feature_vector_attention_mask", None
        )
        if mask_builder is not None:
            hidden_mask = mask_builder(hidden.shape[1], waveform_mask.long()).bool()
        else:
            hidden_mask = (
                F.interpolate(
                    waveform_mask.float().unsqueeze(1),
                    size=hidden.shape[1],
                    mode="nearest",
                )
                .squeeze(1)
                .bool()
            )
        return hidden, hidden_mask
