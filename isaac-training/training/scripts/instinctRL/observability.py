"""
instinctRL Range-Jacobian Observability Logger
==============================================

Evaluation-only observability diagnostics for MID360 measurement space.
This module must not become a deployed control dependency.
"""

from dataclasses import dataclass
import math
from typing import Dict, Optional

import torch


OBS_MODE_PROXY = 0
OBS_MODE_NORMAL = 1
OBS_MODE_FINITE_DIFFERENCE = 2

OBSERVABILITY_METRIC_KEYS = (
    "observability_valid_fraction",
    "observability_weighted_valid_fraction",
    "observability_rank",
    "observability_sigma_min",
    "observability_sigma_max",
    "observability_condition_number",
    "observability_score",
    "observability_drift_projection",
    "observability_drift_norm",
    "observability_is_proxy",
    "observability_mode_code",
)


@dataclass
class ObservabilityConfig:
    """Configuration for evaluation-only observability diagnostics."""

    enabled: bool = False
    mode: str = "offline"
    rank_tol: float = 1e-4
    condition_number_cap: float = 1e6
    min_valid_fraction: float = 0.01
    log_interval: int = 50
    use_surface_normals: bool = False
    use_finite_difference: bool = True
    eps: float = 1e-8

    def __post_init__(self):
        if self.mode not in ("offline", "proxy"):
            raise ValueError("observability mode must be 'offline' or 'proxy'")
        for name in ("rank_tol", "condition_number_cap", "min_valid_fraction", "eps"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            setattr(self, name, value)
        if self.rank_tol <= 0.0:
            raise ValueError("rank_tol must be > 0")
        if self.condition_number_cap <= 0.0:
            raise ValueError("condition_number_cap must be > 0")
        if not (0.0 <= self.min_valid_fraction <= 1.0):
            raise ValueError("min_valid_fraction must satisfy 0.0 <= value <= 1.0")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")
        if self.eps <= 0.0:
            raise ValueError("eps must be > 0")

    @classmethod
    def from_namespace(cls, cfg) -> "ObservabilityConfig":
        return cls(
            enabled=bool(getattr(cfg, "enabled", False)),
            mode=str(getattr(cfg, "mode", "offline")),
            rank_tol=float(getattr(cfg, "rank_tol", 1e-4)),
            condition_number_cap=float(getattr(cfg, "condition_number_cap", 1e6)),
            min_valid_fraction=float(getattr(cfg, "min_valid_fraction", 0.01)),
            log_interval=int(getattr(cfg, "log_interval", 50)),
            use_surface_normals=bool(getattr(cfg, "use_surface_normals", False)),
            use_finite_difference=bool(getattr(cfg, "use_finite_difference", True)),
        )


@dataclass
class ObservabilityOutput:
    """Separated public scalar metrics and dense internal runtime cache."""

    metrics: Dict[str, torch.Tensor]
    cache: Dict[str, torch.Tensor]


class RangeJacobianObservabilityLogger:
    """Evaluation-only range-Jacobian observability logger."""

    def __init__(self, config: ObservabilityConfig, *, device: str = "cuda:0"):
        self.cfg = config
        self.device = torch.device(device)

    def compute(
        self,
        *,
        ray_directions_b: torch.Tensor,
        valid_mask: torch.Tensor,
        reliability_weight: torch.Tensor,
        surface_normals_b: Optional[torch.Tensor] = None,
        fd_perturbations_b: Optional[torch.Tensor] = None,
        fd_range_delta: Optional[torch.Tensor] = None,
        drift_b: Optional[torch.Tensor] = None,
        scenario_id: Optional[torch.Tensor] = None,
    ) -> ObservabilityOutput:
        """Compute observability metrics from flattened MID360 measurement geometry."""
        valid_mask, reliability_weight = self._validate_mask_weight(valid_mask, reliability_weight)
        N, R = valid_mask.shape
        ray_dirs = self._validate_ray_directions(ray_directions_b, N, R)
        drift_b = self._validate_optional_drift(drift_b, N, valid_mask.device)
        scenario_id = self._validate_optional_scenario_id(scenario_id, N, valid_mask.device)

        mode_code = self._select_mode(
            surface_normals_b=surface_normals_b,
            fd_perturbations_b=fd_perturbations_b,
            fd_range_delta=fd_range_delta,
        )
        if mode_code == OBS_MODE_FINITE_DIFFERENCE:
            jacobian_rows, source_valid = self._finite_difference_rows(
                fd_perturbations_b, fd_range_delta, N, R
            )
        elif mode_code == OBS_MODE_NORMAL:
            jacobian_rows, source_valid = self._normal_rows(surface_normals_b, N, R)
        else:
            jacobian_rows, source_valid = self._proxy_rows(ray_dirs)

        valid_bool = self._mask_to_bool(valid_mask)
        weights = reliability_weight.clamp(0.0, 1.0)
        effective_mask = valid_bool & source_valid & (weights > 0)
        sqrt_w = torch.sqrt(weights).unsqueeze(-1)
        weighted_rows = jacobian_rows * sqrt_w
        weighted_rows = torch.where(effective_mask.unsqueeze(-1), weighted_rows, torch.zeros_like(weighted_rows))

        metrics, cache = self._svd_metrics(weighted_rows, effective_mask)
        self._add_common_metrics(metrics, effective_mask, weights, mode_code, drift_b, cache)
        if scenario_id is not None:
            metrics["observability_scenario_id"] = scenario_id.reshape(N, 1)
        cache.update({
            "jacobian_rows": jacobian_rows,
            "weighted_jacobian_rows": weighted_rows,
            "effective_row_mask": effective_mask,
        })
        return ObservabilityOutput(metrics=metrics, cache=cache)

    def _select_mode(
        self,
        *,
        surface_normals_b: Optional[torch.Tensor],
        fd_perturbations_b: Optional[torch.Tensor],
        fd_range_delta: Optional[torch.Tensor],
    ) -> int:
        fd_any = fd_perturbations_b is not None or fd_range_delta is not None
        normal_any = surface_normals_b is not None
        if self.cfg.mode == "proxy":
            if fd_any:
                self._validate_fd_inputs(fd_perturbations_b, fd_range_delta)
            if normal_any:
                self._validate_supplied_normals(surface_normals_b)
            return OBS_MODE_PROXY
        if fd_any:
            self._validate_fd_inputs(fd_perturbations_b, fd_range_delta)
            if self.cfg.use_finite_difference:
                return OBS_MODE_FINITE_DIFFERENCE
        if normal_any:
            if self.cfg.use_surface_normals:
                return OBS_MODE_NORMAL
            self._validate_supplied_normals(surface_normals_b)
        return OBS_MODE_PROXY

    def _validate_supplied_normals(self, surface_normals_b: torch.Tensor):
        if surface_normals_b.device != self.device:
            raise ValueError("surface_normals_b must be on logger device")
        if surface_normals_b.dim() != 3 or surface_normals_b.shape[-1] != 3:
            raise ValueError("surface_normals_b must have shape [N, R, 3]")
        if not torch.isfinite(surface_normals_b).all():
            raise ValueError("surface_normals_b must be finite")

    def _validate_mask_weight(self, valid_mask: torch.Tensor, reliability_weight: torch.Tensor):
        if valid_mask.dim() != 2:
            raise ValueError("valid_mask must have shape [N, R]")
        if reliability_weight.shape != valid_mask.shape:
            raise ValueError("reliability_weight must have shape [N, R]")
        if valid_mask.device != self.device or reliability_weight.device != self.device:
            raise ValueError("observability inputs must be on logger device")
        if not torch.isfinite(reliability_weight).all():
            raise ValueError("reliability_weight must be finite")
        if valid_mask.dtype != torch.bool and not torch.isfinite(valid_mask).all():
            raise ValueError("numeric valid_mask must be finite")
        return valid_mask, reliability_weight

    def _validate_ray_directions(self, ray_directions_b: torch.Tensor, N: int, R: int):
        if ray_directions_b.device != self.device:
            raise ValueError("ray_directions_b must be on logger device")
        if ray_directions_b.dim() == 2 and ray_directions_b.shape == (R, 3):
            ray_directions_b = ray_directions_b.unsqueeze(0).expand(N, R, 3)
        elif not (ray_directions_b.dim() == 3 and ray_directions_b.shape == (N, R, 3)):
            raise ValueError("ray_directions_b must have shape [R,3] or [N,R,3]")
        if not torch.isfinite(ray_directions_b).all():
            raise ValueError("ray_directions_b must be finite")
        return ray_directions_b

    def _validate_optional_drift(self, drift_b: Optional[torch.Tensor], N: int, device: torch.device):
        if drift_b is None:
            return None
        if drift_b.shape != (N, 3):
            raise ValueError("drift_b must have shape [N,3]")
        if drift_b.device != device:
            raise ValueError("drift_b must share input device")
        if not torch.isfinite(drift_b).all():
            raise ValueError("drift_b must be finite")
        return drift_b

    def _validate_optional_scenario_id(self, scenario_id: Optional[torch.Tensor], N: int, device: torch.device):
        if scenario_id is None:
            return None
        if scenario_id.device != device:
            raise ValueError("scenario_id must share input device")
        if scenario_id.shape == (N,):
            scenario_id = scenario_id.reshape(N, 1)
        if scenario_id.shape != (N, 1):
            raise ValueError("scenario_id must have shape [N] or [N,1]")
        return scenario_id.to(dtype=torch.long)

    def _validate_fd_inputs(
        self,
        fd_perturbations_b: Optional[torch.Tensor],
        fd_range_delta: Optional[torch.Tensor],
    ):
        if fd_perturbations_b is None or fd_range_delta is None:
            raise ValueError("finite-difference inputs must include fd_perturbations_b and fd_range_delta")
        if fd_perturbations_b.device != self.device or fd_range_delta.device != self.device:
            raise ValueError("finite-difference inputs must be on logger device")
        if fd_perturbations_b.dim() != 2 or fd_perturbations_b.shape[1] != 3:
            raise ValueError("fd_perturbations_b must have shape [K,3]")
        if fd_perturbations_b.shape[0] < 3:
            raise ValueError("finite-difference requires K >= 3")
        if fd_range_delta.dim() != 3 or fd_range_delta.shape[1] != fd_perturbations_b.shape[0]:
            raise ValueError("fd_range_delta must have shape [N,K,R]")
        if not torch.isfinite(fd_perturbations_b).all() or not torch.isfinite(fd_range_delta).all():
            raise ValueError("finite-difference inputs must be finite")
        s = torch.linalg.svdvals(fd_perturbations_b)
        if s.numel() < 3 or s[-1] <= self.cfg.rank_tol:
            raise ValueError("fd_perturbations_b must have rank 3")

    def _mask_to_bool(self, mask: torch.Tensor) -> torch.Tensor:
        if mask.dtype == torch.bool:
            return mask
        return mask > 0

    def _normalize_vectors(self, vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        norms = torch.linalg.norm(vectors, dim=-1, keepdim=True)
        valid = torch.isfinite(vectors).all(dim=-1) & (norms.squeeze(-1) > self.cfg.eps)
        unit = torch.zeros_like(vectors)
        unit[valid] = vectors[valid] / norms[valid]
        return unit, valid

    def _proxy_rows(self, ray_dirs: torch.Tensor):
        ray_unit, source_valid = self._normalize_vectors(ray_dirs)
        return -ray_unit, source_valid

    def _normal_rows(self, surface_normals_b: torch.Tensor, N: int, R: int):
        if surface_normals_b.device != self.device:
            raise ValueError("surface_normals_b must be on logger device")
        if surface_normals_b.shape != (N, R, 3):
            raise ValueError("surface_normals_b must have shape [N,R,3]")
        if not torch.isfinite(surface_normals_b).all():
            raise ValueError("surface_normals_b must be finite")
        normal_unit, source_valid = self._normalize_vectors(surface_normals_b)
        return -normal_unit, source_valid

    def _finite_difference_rows(
        self,
        fd_perturbations_b: torch.Tensor,
        fd_range_delta: torch.Tensor,
        N: int,
        R: int,
    ):
        self._validate_fd_inputs(fd_perturbations_b, fd_range_delta)
        if fd_range_delta.shape[0] != N or fd_range_delta.shape[2] != R:
            raise ValueError("fd_range_delta must have shape [N,K,R]")
        p_pinv = torch.linalg.pinv(fd_perturbations_b)
        j_tmp = torch.einsum("dk,nkr->ndr", p_pinv, fd_range_delta)
        jacobian_rows = j_tmp.transpose(1, 2)
        source_valid = torch.isfinite(jacobian_rows).all(dim=-1)
        if not source_valid.all():
            raise ValueError("finite-difference estimated J must be finite")
        return jacobian_rows, source_valid

    def _svd_metrics(self, weighted_rows: torch.Tensor, effective_mask: torch.Tensor):
        N = weighted_rows.shape[0]
        dtype = weighted_rows.dtype
        device = weighted_rows.device
        rank = torch.zeros(N, 1, dtype=dtype, device=device)
        sigma_min = torch.zeros(N, 1, dtype=dtype, device=device)
        sigma_max = torch.zeros(N, 1, dtype=dtype, device=device)
        condition = torch.full((N, 1), self.cfg.condition_number_cap, dtype=dtype, device=device)
        score = torch.zeros(N, 1, dtype=dtype, device=device)
        weak = torch.zeros(N, 3, dtype=dtype, device=device)
        singular_values = torch.zeros(N, 3, dtype=dtype, device=device)

        for env_id in range(N):
            rows = weighted_rows[env_id][effective_mask[env_id]]
            if rows.shape[0] < 3 or torch.linalg.norm(rows) <= self.cfg.eps:
                continue
            _, s, vh = torch.linalg.svd(rows, full_matrices=False)
            if s.numel() < 3:
                s = torch.cat([s, torch.zeros(3 - s.numel(), dtype=dtype, device=device)])
            else:
                s = s[:3]
            singular_values[env_id] = s
            rank_value = (s > self.cfg.rank_tol).sum().to(dtype=dtype)
            rank[env_id, 0] = rank_value
            sigma_max[env_id, 0] = s[0]
            sigma_min[env_id, 0] = s[-1]
            condition_value = s[0] / torch.clamp(s[-1], min=self.cfg.eps)
            condition[env_id, 0] = torch.clamp(condition_value, max=self.cfg.condition_number_cap)
            score[env_id, 0] = s[-1] / (s[0] + self.cfg.eps)
            if int(rank_value.item()) > 0:
                weak[env_id] = vh[-1]

        metrics = {
            "observability_rank": rank,
            "observability_sigma_min": sigma_min,
            "observability_sigma_max": sigma_max,
            "observability_condition_number": condition,
            "observability_score": score,
        }
        cache = {
            "singular_values": singular_values,
            "observability_weak_direction": weak,
        }
        return metrics, cache

    def _add_common_metrics(
        self,
        metrics: Dict[str, torch.Tensor],
        effective_mask: torch.Tensor,
        weights: torch.Tensor,
        mode_code: int,
        drift_b: Optional[torch.Tensor],
        cache: Dict[str, torch.Tensor],
    ):
        N, R = effective_mask.shape
        dtype = weights.dtype
        valid_count = effective_mask.sum(dim=1, keepdim=True).to(dtype=dtype)
        valid_fraction = valid_count / float(R)
        weighted_valid_fraction = (weights * effective_mask.to(dtype=dtype)).sum(dim=1, keepdim=True) / float(R)
        metrics["observability_valid_fraction"] = valid_fraction
        metrics["observability_weighted_valid_fraction"] = weighted_valid_fraction

        weak = cache["observability_weak_direction"]
        drift_projection = torch.zeros(N, 1, dtype=dtype, device=self.device)
        drift_norm = torch.zeros(N, 1, dtype=dtype, device=self.device)
        if drift_b is not None:
            drift_norm = torch.linalg.norm(drift_b, dim=-1, keepdim=True)
            weak_norm = torch.linalg.norm(weak, dim=-1, keepdim=True)
            weak_unit = torch.where(weak_norm > self.cfg.eps, weak / weak_norm.clamp_min(self.cfg.eps), torch.zeros_like(weak))
            drift_projection = (drift_b * weak_unit).sum(dim=-1, keepdim=True).abs()
        metrics["observability_drift_projection"] = drift_projection
        metrics["observability_drift_norm"] = drift_norm

        metrics["observability_is_proxy"] = torch.full(
            (N, 1), 1.0 if mode_code == OBS_MODE_PROXY else 0.0, dtype=dtype, device=self.device
        )
        metrics["observability_mode_code"] = torch.full(
            (N, 1), int(mode_code), dtype=torch.long, device=self.device
        )

        for key, value in list(metrics.items()):
            if value.dtype.is_floating_point:
                metrics[key] = torch.nan_to_num(
                    value,
                    nan=0.0,
                    posinf=self.cfg.condition_number_cap,
                    neginf=0.0,
                )
