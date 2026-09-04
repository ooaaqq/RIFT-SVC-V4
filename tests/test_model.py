import pytest
import torch

from rift_v4.flow import FlowMatchingSystem, _sample_timestep
from rift_v4.model import RIFTV4, CorrectedAttention
from rift_v4.train import _require_finite


def test_attention_uses_inverse_sqrt_head_dimension() -> None:
    attention = CorrectedAttention(dim=32, head_dim=8)
    assert attention.scale == 1 / 8**0.5


def test_small_system_forward_and_backward() -> None:
    model = RIFTV4(
        mel_channels=8,
        content_dim=16,
        num_speakers=3,
        dim=32,
        depth=2,
        head_dim=8,
        ff_hidden_dim=64,
        kernel_size=5,
    )
    system = FlowMatchingSystem(model, speaker_drop_probability=0.2)
    batch = {
        "mel": torch.randn(2, 12, 8),
        "content": torch.randn(2, 12, 16),
        "f0": torch.rand(2, 12, 1) * 400,
        "rms": torch.rand(2, 12, 1),
        "speaker": torch.tensor([0, 2]),
        "mask": torch.tensor([[True] * 12, [True] * 9 + [False] * 3]),
    }
    loss = system(batch)
    loss.total.backward()
    assert loss.total.isfinite()
    assert model.mel_input.weight.grad is not None


def test_block_residual_scales_can_disable_both_branches() -> None:
    model = RIFTV4(
        8, 16, 2, dim=32, depth=1, head_dim=8, ff_hidden_dim=64, kernel_size=5
    )
    block = model.blocks[0]
    torch.nn.init.normal_(block.modulation[-1].weight, std=0.05)
    torch.nn.init.normal_(block.modulation[-1].bias, std=0.05)
    block.attention_residual_scale = 0.0
    block.ffn_residual_scale = 0.0
    x = torch.randn(2, 9, 32)
    mask = torch.tensor([[True] * 9, [True] * 7 + [False] * 2])

    output = block(x, torch.randn(2, 1, 32), mask)

    torch.testing.assert_close(output[0], x[0])
    torch.testing.assert_close(output[1, :7], x[1, :7])
    assert output[1, 7:].count_nonzero() == 0


def test_padding_cannot_change_valid_frames() -> None:
    torch.manual_seed(7)
    model = RIFTV4(
        8, 16, 2, dim=32, depth=2, head_dim=8, ff_hidden_dim=64, kernel_size=5
    )
    for block in model.blocks:
        torch.nn.init.normal_(block.modulation[-1].weight, std=0.05)
        torch.nn.init.normal_(block.modulation[-1].bias, std=0.05)
    torch.nn.init.normal_(model.final_modulation[-1].weight, std=0.05)
    torch.nn.init.normal_(model.output.weight, std=0.05)
    common = {
        "f0": torch.rand(1, 8, 1) * 400,
        "rms": torch.rand(1, 8, 1),
        "speaker": torch.tensor([0]),
        "timestep": torch.tensor([0.4]),
    }
    mel = torch.randn(1, 8, 8)
    content = torch.randn(1, 8, 16)
    short = model(mel, content, mask=torch.ones(1, 8, dtype=torch.bool), **common)
    padded_common = {
        "f0": torch.nn.functional.pad(common["f0"], (0, 0, 0, 4)),
        "rms": torch.nn.functional.pad(common["rms"], (0, 0, 0, 4)),
        "speaker": common["speaker"],
        "timestep": common["timestep"],
    }
    long = model(
        torch.nn.functional.pad(mel, (0, 0, 0, 4)),
        torch.nn.functional.pad(content, (0, 0, 0, 4)),
        mask=torch.tensor([[True] * 8 + [False] * 4]),
        **padded_common,
    )
    torch.testing.assert_close(short, long[:, :8], atol=2e-6, rtol=2e-6)


def test_heun_sampler_returns_masked_finite_mel() -> None:
    model = RIFTV4(
        8, 16, 2, dim=32, depth=1, head_dim=8, ff_hidden_dim=64, kernel_size=5
    )
    system = FlowMatchingSystem(model, speaker_drop_probability=0.2).eval()
    mask = torch.tensor([[True] * 7 + [False] * 2])
    output = system.sample(
        torch.randn(1, 9, 16),
        torch.rand(1, 9, 1) * 400,
        torch.rand(1, 9, 1),
        torch.tensor([0]),
        mask,
        8,
        steps=3,
    )
    assert output.shape == (1, 9, 8)
    assert output.isfinite().all()
    assert output[:, 7:].count_nonzero() == 0


def test_cosine_time_schedule_has_correct_endpoints() -> None:
    from rift_v4.flow import _time_grid

    linear = _time_grid(4, torch.device("cpu"), "linear")
    cosine = _time_grid(4, torch.device("cpu"), "cosine")
    assert torch.allclose(linear, torch.linspace(0, 1, 5))
    assert torch.allclose(cosine[[0, -1]], torch.tensor([0.0, 1.0]))
    assert cosine[1] < linear[1]


def test_timestep_sampling_is_stratified_and_shuffled() -> None:
    torch.manual_seed(4)
    values = _sample_timestep(32, torch.device("cpu"), torch.float32)
    assert torch.isfinite(values).all()
    assert torch.all((values > 0) & (values < 1))
    ordered = torch.sort(values).values
    assert ordered[0] < 0.15
    assert ordered[-1] > 0.85
    assert not torch.equal(values, ordered)


def test_non_finite_training_values_fail_before_an_update() -> None:
    _require_finite(torch.tensor(1.0), "total loss", 7)
    with pytest.raises(FloatingPointError, match="non-finite total loss at step 7"):
        _require_finite(torch.tensor(float("nan")), "total loss", 7)
    with pytest.raises(FloatingPointError, match="non-finite gradient norm"):
        _require_finite(torch.tensor(float("inf")), "gradient norm", 8)
