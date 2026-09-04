from __future__ import annotations

from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version

import torch
from torch import nn

from .config import PerformanceConfig

PINNED_TORCHAO_VERSION = "0.18.0"


def configure_performance(
    config: PerformanceConfig, device: torch.device
) -> dict[str, object]:
    runtime = asdict(config)
    runtime["device_type"] = device.type
    runtime["flash_sdpa_enforced"] = False
    if device.type != "cuda":
        return runtime

    if config.compile_mode.startswith("max-autotune"):
        torch.set_float32_matmul_precision(
            "high" if config.matmul_fp32_precision == "tf32" else "highest"
        )
        torch.backends.cudnn.allow_tf32 = config.conv_fp32_precision == "tf32"
        runtime["precision_api"] = "legacy_for_inductor_autotune"
    else:
        torch.backends.cuda.matmul.fp32_precision = config.matmul_fp32_precision
        torch.backends.cudnn.conv.fp32_precision = config.conv_fp32_precision
        runtime["precision_api"] = "backend_specific"
    torch.backends.cuda.enable_flash_sdp(config.sdpa_backend == "flash")
    torch.backends.cuda.enable_cudnn_sdp(config.sdpa_backend == "cudnn")
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)
    runtime["flash_sdpa_enforced"] = config.sdpa_backend == "flash"
    runtime["cudnn_sdpa_enforced"] = config.sdpa_backend == "cudnn"
    runtime["flash_sdp_enabled"] = torch.backends.cuda.flash_sdp_enabled()
    runtime["memory_efficient_sdp_enabled"] = (
        torch.backends.cuda.mem_efficient_sdp_enabled()
    )
    runtime["math_sdp_enabled"] = torch.backends.cuda.math_sdp_enabled()
    runtime["cudnn_sdp_enabled"] = torch.backends.cuda.cudnn_sdp_enabled()
    return runtime


def compile_model_in_place(model: nn.Module, config: PerformanceConfig) -> bool:
    if not config.compile_model:
        return False
    state_keys = tuple(model.state_dict())
    model.compile(mode=config.compile_mode, dynamic=False)
    if tuple(model.state_dict()) != state_keys:
        raise RuntimeError("torch.compile changed model state-dict keys")
    return True


def apply_selective_float8_training(
    model: nn.Module, config: PerformanceConfig, device: torch.device
) -> dict[str, object]:
    if not config.float8_training:
        return {
            "enabled": False,
            "recipe": config.float8_recipe,
            "modules": [],
        }
    if device.type != "cuda":
        raise RuntimeError("FP8 training requires a CUDA device")
    version_parts = torch.__version__.split("+")[0].split(".")[:2]
    torch_version = tuple(int(part) for part in version_parts)
    if torch_version < (2, 11):
        raise RuntimeError("torchao 0.18 FP8 training requires PyTorch 2.11 or newer")
    if torch.cuda.get_device_capability(device) < (8, 9):
        raise RuntimeError("FP8 training requires compute capability 8.9 or newer")
    try:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training
    except ImportError as error:
        raise RuntimeError(
            "FP8 training requires the pinned torchao dependency"
        ) from error
    torchao_version = version("torchao")
    if torchao_version != PINNED_TORCHAO_VERSION:
        raise RuntimeError(
            f"FP8 training requires torchao {PINNED_TORCHAO_VERSION}, "
            f"found {torchao_version}"
        )

    selected = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and _is_float8_linear(name)
    ]
    expected = len(getattr(model, "blocks", ())) * 4
    if len(selected) != expected or expected == 0:
        raise RuntimeError(
            f"selective FP8 expected {expected} Linear modules, found {len(selected)}"
        )
    modules = dict(model.named_modules())
    misaligned = [
        name
        for name in selected
        if modules[name].in_features % 16 or modules[name].out_features % 16
    ]
    if misaligned:
        raise RuntimeError(f"FP8 Linear dimensions are not aligned: {misaligned}")

    state_keys = tuple(model.state_dict())
    recipe = Float8LinearConfig.from_recipe_name(config.float8_recipe)
    converted = convert_to_float8_training(
        model,
        config=recipe,
        module_filter_fn=lambda module, fqn: fqn in selected,
    )
    if converted is not model:
        raise RuntimeError("selective FP8 unexpectedly replaced the root model")
    if tuple(model.state_dict()) != state_keys:
        raise RuntimeError("selective FP8 changed canonical state-dict keys")
    return {
        "enabled": True,
        "recipe": config.float8_recipe,
        "torchao_version": torchao_version,
        "modules": selected,
    }


def software_versions(_config: PerformanceConfig) -> dict[str, str | None]:
    torchao_version: str | None = None
    try:
        torchao_version = version("torchao")
    except PackageNotFoundError:
        torchao_version = "missing"
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "torchao": torchao_version,
    }


def _is_float8_linear(name: str) -> bool:
    return name.endswith(
        (
            ".attention.qkv",
            ".attention.output",
            ".feed_forward.input",
            ".feed_forward.output",
        )
    )
