import pytest
import torch

from rift_v4.infer import sample_chunked


class NoiseEchoSystem:
    def __init__(self) -> None:
        self.lengths: list[int] = []

    def sample(self, content, *args, initial_noise, **kwargs):
        del args, kwargs
        self.lengths.append(content.shape[1])
        return initial_noise


def test_chunked_inference_reuses_overlap_noise_and_preserves_length() -> None:
    system = NoiseEchoSystem()
    frames = 1_200
    content = torch.zeros(1, frames, 4)
    scalar = torch.zeros(1, frames, 1)
    mask = torch.ones(1, frames, dtype=torch.bool)
    speaker = torch.zeros(1, dtype=torch.long)
    expected_generator = torch.Generator().manual_seed(17)
    expected = torch.randn(1, frames, 3, generator=expected_generator)[0]
    actual = sample_chunked(
        system,
        content,
        scalar,
        scalar,
        speaker,
        mask,
        mel_channels=3,
        steps=2,
        guidance=1.0,
        method="heun",
        generator=torch.Generator().manual_seed(17),
        time_schedule="cosine",
        chunk_frames=512,
        overlap_frames=64,
    )
    torch.testing.assert_close(actual, expected)
    assert system.lengths == [512, 512, 512]


def test_chunked_inference_rejects_invalid_overlap() -> None:
    system = NoiseEchoSystem()
    value = torch.zeros(1, 10, 1)
    with pytest.raises(ValueError, match="overlap frames"):
        sample_chunked(
            system,
            value,
            value,
            value,
            torch.zeros(1, dtype=torch.long),
            torch.ones(1, 10, dtype=torch.bool),
            1,
            1,
            1.0,
            "euler",
            torch.Generator(),
            "cosine",
            8,
            8,
        )
