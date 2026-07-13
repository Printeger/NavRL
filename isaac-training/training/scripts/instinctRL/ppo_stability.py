"""PPO numerical-stability helpers for learned-governor training."""

import os
import time
from typing import Dict, Iterable, Mapping, Optional

import torch
import torch.nn as nn


_EXACT_COUNT_LIMIT = 2_000_000
_STAT_SAMPLE_LIMIT = 65_536


def _sample_flat_tensor(flat: torch.Tensor, max_elements: int = _STAT_SAMPLE_LIMIT) -> torch.Tensor:
    if flat.numel() <= max_elements:
        return flat
    step = max(1, flat.numel() // max_elements)
    return flat[::step][:max_elements]


def tensor_stats(tensor: Optional[torch.Tensor]) -> Dict[str, object]:
    """Return compact finite/non-finite statistics for a tensor."""
    if tensor is None:
        return {"present": False}
    detached = tensor.detach()
    flat = detached.reshape(-1)
    numel = int(flat.numel())
    sampled = numel > _EXACT_COUNT_LIMIT
    count_source = _sample_flat_tensor(flat) if sampled else flat
    finite = torch.isfinite(count_source)
    finite_count = int(finite.sum().item())
    nonfinite_count = int((~finite).sum().item())
    stats: Dict[str, object] = {
        "present": True,
        "shape": tuple(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "finite_count": finite_count,
        "numel": numel,
        "nonfinite_count": nonfinite_count,
        "count_exact": not sampled,
    }
    if finite.any():
        finite_values = count_source[finite].float().cpu()
        stats.update({
            "min": float(finite_values.min().item()),
            "max": float(finite_values.max().item()),
            "mean": float(finite_values.mean().item()),
            "std": float(finite_values.std(unbiased=False).item()),
            "norm": float(torch.linalg.norm(finite_values).item()),
        })
    return stats


def _iter_named_parameters(module: nn.Module) -> Iterable[tuple[str, torch.nn.Parameter]]:
    for name, parameter in module.named_parameters():
        if parameter is not None:
            yield name, parameter


def module_parameter_stats(module: nn.Module) -> Dict[str, object]:
    """Return compact parameter stats for a module."""
    per_parameter = {}
    total_norm_sq = 0.0
    finite = True
    for name, parameter in _iter_named_parameters(module):
        stats = tensor_stats(parameter.data)
        per_parameter[name] = stats
        finite = finite and stats.get("nonfinite_count", 0) == 0
        norm = stats.get("norm")
        if norm is not None:
            total_norm_sq += float(norm) ** 2
    return {
        "finite": finite,
        "total_norm": total_norm_sq ** 0.5,
        "parameters": per_parameter,
    }


def module_gradient_stats(module: nn.Module) -> Dict[str, object]:
    """Return compact gradient stats for a module."""
    per_parameter = {}
    total_norm_sq = 0.0
    finite = True
    for name, parameter in _iter_named_parameters(module):
        stats = tensor_stats(parameter.grad)
        per_parameter[name] = stats
        if parameter.grad is not None:
            finite = finite and stats.get("nonfinite_count", 0) == 0
            norm = stats.get("norm")
            if norm is not None:
                total_norm_sq += float(norm) ** 2
    return {
        "finite": finite,
        "total_norm": total_norm_sq ** 0.5,
        "gradients": per_parameter,
    }


def assert_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    """Fail fast if a tensor contains non-finite values."""
    if not torch.isfinite(tensor).all():
        stats = tensor_stats(tensor)
        raise ValueError(
            f"non-finite tensor detected: {name} "
            f"shape={stats.get('shape')} nonfinite={stats.get('nonfinite_count')}"
        )


def assert_module_gradients_finite(name: str, module: nn.Module) -> None:
    stats = module_gradient_stats(module)
    if not stats["finite"]:
        raise ValueError(f"non-finite gradients detected in {name}")


def assert_module_parameters_finite(name: str, module: nn.Module) -> None:
    stats = module_parameter_stats(module)
    if not stats["finite"]:
        raise ValueError(f"non-finite parameters detected in {name}")


def safe_normalize_advantage(advantage: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize advantages with a denominator that cannot underflow to zero."""
    normalized = (advantage - advantage.mean()) / advantage.std().clamp_min(eps)
    assert_finite_tensor("advantage_normalized", normalized)
    return normalized


def save_diagnostic_snapshot(
    diagnostic_dir: str,
    reason: str,
    context: Mapping[str, object],
    tensors: Optional[Mapping[str, Optional[torch.Tensor]]] = None,
    modules: Optional[Mapping[str, nn.Module]] = None,
) -> str:
    """Save a compact .pt snapshot for non-finite PPO failures."""
    os.makedirs(diagnostic_dir, exist_ok=True)
    timestamp = int(time.time() * 1000)
    safe_reason = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in reason)
    path = os.path.join(diagnostic_dir, f"ppo_stability_{timestamp}_{safe_reason}.pt")

    payload: Dict[str, object] = {
        "reason": reason,
        "context": dict(context),
        "tensors": {},
        "modules": {},
    }
    if tensors:
        payload["tensors"] = {}
        for name, value in tensors.items():
            try:
                payload["tensors"][name] = tensor_stats(value)
            except Exception as exc:
                payload["tensors"][name] = {
                    "present": value is not None,
                    "stats_error": f"{type(exc).__name__}: {exc}",
                }
    if modules:
        payload["modules"] = {}
        for name, module in modules.items():
            try:
                payload["modules"][name] = {
                    "parameters": module_parameter_stats(module),
                    "gradients": module_gradient_stats(module),
                }
            except Exception as exc:
                payload["modules"][name] = {
                    "stats_error": f"{type(exc).__name__}: {exc}",
                }
    torch.save(payload, path)
    return path
