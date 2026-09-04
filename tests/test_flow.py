import torch

from rift_v4.flow import _sample_timestep


def test_timestep_sampling_is_finite_for_bfloat16() -> None:
    value = _sample_timestep(128, torch.device("cpu"), torch.bfloat16)
    assert value.dtype == torch.bfloat16
    assert torch.isfinite(value).all()
    assert (value > 0).all() and (value < 1).all()
