"""
instinctRL Measurement-Space Anchor Manager
===========================================

Actor-clean station-keeping anchor over MID360 measurement space.

Inputs are limited to MID360 range/mask/weight, body-frame command, and
reset flags. Dense anchor tensors are internal runtime state/cache only.
"""

from dataclasses import dataclass
import math
from typing import Dict, Optional

import torch


ANCHOR_RESET_NONE = 0
ANCHOR_RESET_EPISODE = 1
ANCHOR_RESET_EXPLICIT = 2
ANCHOR_RESET_COMMAND = 3
ANCHOR_RESET_INVALID = 4

ANCHOR_RESET_CODES = {
    ANCHOR_RESET_NONE,
    ANCHOR_RESET_EPISODE,
    ANCHOR_RESET_EXPLICIT,
    ANCHOR_RESET_COMMAND,
    ANCHOR_RESET_INVALID,
}

ANCHOR_METRIC_KEYS = (
    "anchor_active",
    "anchor_loss",
    "anchor_valid_fraction",
    "anchor_error_mean",
    "anchor_error_max",
    "anchor_hold_steps",
    "anchor_activation_count",
    "anchor_reset_reason",
)


@dataclass
class AnchorConfig:
    """Configuration for measurement-space anchor lifecycle and loss."""

    lidar_hbeams: int = 360
    lidar_vbeams: int = 59
    enabled: bool = True
    eps_enter: float = 0.05
    eps_exit: float = 0.15
    min_valid_anchor_fraction: float = 0.10
    huber_delta: float = 0.25
    reset_on_large_command: bool = True

    def __post_init__(self):
        if int(self.lidar_hbeams) <= 0 or int(self.lidar_vbeams) <= 0:
            raise ValueError("lidar_hbeams and lidar_vbeams must be positive")
        self.lidar_hbeams = int(self.lidar_hbeams)
        self.lidar_vbeams = int(self.lidar_vbeams)

        for name in ("eps_enter", "eps_exit", "min_valid_anchor_fraction", "huber_delta"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            setattr(self, name, value)

        if not self.eps_enter < self.eps_exit:
            raise ValueError("eps_enter must be < eps_exit")
        if not (0.0 < self.min_valid_anchor_fraction <= 1.0):
            raise ValueError("min_valid_anchor_fraction must satisfy 0.0 < value <= 1.0")
        if self.huber_delta <= 0.0:
            raise ValueError("huber_delta must be > 0")

    @classmethod
    def from_namespace(cls, cfg, *, lidar_hbeams: int, lidar_vbeams: int) -> "AnchorConfig":
        """Build from a Hydra/OmegaConf-style namespace and reject legacy keys."""
        if hasattr(cfg, "min_valid_fraction"):
            raise ValueError(
                "Unsupported key instinctRL.anchor.min_valid_fraction; "
                "use instinctRL.anchor.min_valid_anchor_fraction"
            )
        return cls(
            lidar_hbeams=lidar_hbeams,
            lidar_vbeams=lidar_vbeams,
            enabled=bool(getattr(cfg, "enabled", True)),
            eps_enter=float(getattr(cfg, "eps_enter", 0.05)),
            eps_exit=float(getattr(cfg, "eps_exit", 0.15)),
            min_valid_anchor_fraction=float(
                getattr(cfg, "min_valid_anchor_fraction", 0.10)
            ),
            huber_delta=float(getattr(cfg, "huber_delta", 0.25)),
            reset_on_large_command=bool(getattr(cfg, "reset_on_large_command", True)),
        )


@dataclass
class AnchorState:
    """Vectorized anchor state over parallel environments."""

    active: torch.Tensor
    r_star: torch.Tensor
    m_star: torch.Tensor
    w_star: torch.Tensor
    hold_steps: torch.Tensor
    activation_count: torch.Tensor
    reset_reason: torch.Tensor


@dataclass
class AnchorStepOutput:
    """Separated public scalar metrics and dense internal runtime cache."""

    metrics: Dict[str, torch.Tensor]
    cache: Dict[str, torch.Tensor]


def huber_loss(x: torch.Tensor, delta: float) -> torch.Tensor:
    """Return per-element standard Huber loss with the same shape as ``x``."""
    delta = float(delta)
    if not math.isfinite(delta) or delta <= 0.0:
        raise ValueError("delta must be finite and > 0")
    abs_x = x.abs()
    quadratic = 0.5 * x * x
    linear = delta * (abs_x - 0.5 * delta)
    return torch.where(abs_x <= delta, quadratic, linear)


class MeasurementSpaceAnchorManager:
    """Measurement-space station-keeping anchor manager."""

    def __init__(
        self,
        config: AnchorConfig,
        *,
        num_envs: Optional[int] = None,
        device: str = "cuda:0",
        structural_mask: Optional[torch.Tensor] = None,
    ):
        self.cfg = config
        self.device = torch.device(device)
        self._dtype = torch.float32
        self.structural_mask = self._build_structural_mask(structural_mask)
        self._structural_denominator = self.structural_mask.sum()
        if self._structural_denominator.item() <= 0:
            raise ValueError("structural_mask must contain at least one active slot")
        self.state: Optional[AnchorState] = None
        self._pending_reset_reason: Optional[torch.Tensor] = None
        if num_envs is not None:
            self._allocate(int(num_envs), self._dtype)

    def _build_structural_mask(self, structural_mask: Optional[torch.Tensor]) -> torch.Tensor:
        H, V = self.cfg.lidar_hbeams, self.cfg.lidar_vbeams
        if structural_mask is None:
            return torch.ones(H, V, dtype=torch.bool, device=self.device)
        if structural_mask.shape != (H, V):
            raise ValueError("structural_mask must have shape [H, V]; per-env masks are not supported")
        mask = structural_mask.to(self.device)
        if mask.dtype == torch.bool:
            return mask.clone()
        if not torch.isfinite(mask).all():
            raise ValueError("structural_mask must be finite")
        return (mask > 0).bool()

    def _allocate(self, num_envs: int, dtype: torch.dtype):
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        H, V = self.cfg.lidar_hbeams, self.cfg.lidar_vbeams
        self._dtype = dtype
        self.state = AnchorState(
            active=torch.zeros(num_envs, dtype=torch.bool, device=self.device),
            r_star=torch.zeros(num_envs, H, V, dtype=dtype, device=self.device),
            m_star=torch.zeros(num_envs, H, V, dtype=torch.bool, device=self.device),
            w_star=torch.zeros(num_envs, H, V, dtype=dtype, device=self.device),
            hold_steps=torch.zeros(num_envs, dtype=torch.long, device=self.device),
            activation_count=torch.zeros(num_envs, dtype=torch.long, device=self.device),
            reset_reason=torch.zeros(num_envs, dtype=torch.long, device=self.device),
        )
        self._pending_reset_reason = torch.zeros(num_envs, dtype=torch.long, device=self.device)

    def _ensure_state(self, num_envs: int, dtype: torch.dtype, device: torch.device):
        if device != self.device:
            raise ValueError("input tensors must be on the manager device")
        if self.state is None:
            self._allocate(num_envs, dtype)
        elif self.state.active.shape[0] != num_envs:
            raise ValueError("batch size changed after anchor manager initialization")

    def reset(self, env_ids=None, reason: int = ANCHOR_RESET_EPISODE):
        """Reset selected environments, or all when ``env_ids`` is None."""
        if int(reason) not in ANCHOR_RESET_CODES:
            raise ValueError("unknown anchor reset reason code")
        if self.state is None:
            if env_ids is None:
                return
            raise ValueError("anchor state is not initialized")
        ids = self._normalize_env_ids(env_ids)
        self._clear_anchor(ids)
        if int(reason) == ANCHOR_RESET_EPISODE:
            self.state.activation_count[ids] = 0
        self._pending_reset_reason[ids] = int(reason)

    def step(
        self,
        r_t: torch.Tensor,
        m_t: torch.Tensor,
        w_t: torch.Tensor,
        v_cmd: torch.Tensor,
        *,
        explicit_reset_mask: Optional[torch.Tensor] = None,
        episode_reset_mask: Optional[torch.Tensor] = None,
    ) -> AnchorStepOutput:
        """Advance anchor lifecycle by one vectorized environment step."""
        r_t, m_bool, w_t, v_cmd = self._validate_step_inputs(r_t, m_t, w_t, v_cmd)
        N = r_t.shape[0]
        self._ensure_state(N, r_t.dtype, r_t.device)
        explicit_reset_mask = self._validate_reset_mask(explicit_reset_mask, N, "explicit_reset_mask")
        episode_reset_mask = self._validate_reset_mask(episode_reset_mask, N, "episode_reset_mask")

        state = self.state
        reset_reason = torch.zeros(N, dtype=torch.long, device=self.device)
        if self._pending_reset_reason is not None:
            reset_reason = self._pending_reset_reason.clone()
            self._pending_reset_reason.zero_()

        cmd_norm = torch.linalg.norm(v_cmd, dim=-1)
        command_mask = (
            state.active
            & bool(self.cfg.reset_on_large_command)
            & (cmd_norm >= self.cfg.eps_exit)
        )

        reset_reason = torch.where(
            command_mask & (reset_reason == ANCHOR_RESET_NONE),
            torch.full_like(reset_reason, ANCHOR_RESET_COMMAND),
            reset_reason,
        )
        reset_reason = torch.where(
            explicit_reset_mask & (reset_reason != ANCHOR_RESET_EPISODE),
            torch.full_like(reset_reason, ANCHOR_RESET_EXPLICIT),
            reset_reason,
        )
        reset_reason = torch.where(
            episode_reset_mask,
            torch.full_like(reset_reason, ANCHOR_RESET_EPISODE),
            reset_reason,
        )

        high_reset_mask = reset_reason != ANCHOR_RESET_NONE
        if high_reset_mask.any():
            self._clear_anchor(high_reset_mask)
            episode_mask = reset_reason == ANCHOR_RESET_EPISODE
            state.activation_count[episode_mask] = 0

        continued = state.active & ~high_reset_mask
        state.hold_steps[continued] += 1

        capture_mask = (~state.active) & (~high_reset_mask) & (cmd_norm <= self.cfg.eps_enter)
        if capture_mask.any():
            state.r_star[capture_mask] = r_t[capture_mask]
            state.m_star[capture_mask] = m_bool[capture_mask]
            state.w_star[capture_mask] = w_t[capture_mask]
            state.active[capture_mask] = True
            state.hold_steps[capture_mask] = 1
            state.activation_count[capture_mask] += 1

        pre_invalid = self._compute_outputs(r_t, m_bool, w_t, reset_reason)
        invalid_mask = (
            state.active
            & (pre_invalid.metrics["anchor_valid_fraction"].squeeze(-1) < self.cfg.min_valid_anchor_fraction)
            & (reset_reason == ANCHOR_RESET_NONE)
        )
        if invalid_mask.any():
            self._clear_anchor(invalid_mask)
            reset_reason[invalid_mask] = ANCHOR_RESET_INVALID

        state.reset_reason[:] = reset_reason
        return self._compute_outputs(r_t, m_bool, w_t, reset_reason)

    def _compute_outputs(
        self,
        r_t: torch.Tensor,
        m_bool: torch.Tensor,
        w_t: torch.Tensor,
        reset_reason: torch.Tensor,
    ) -> AnchorStepOutput:
        state = self.state
        N, H, V = r_t.shape
        structural = self.structural_mask.reshape(1, H, V).expand(N, H, V)
        active = state.active.reshape(N, 1, 1)
        usable = (
            structural
            & active
            & m_bool
            & state.m_star
            & (w_t > 0)
            & (state.w_star > 0)
        )
        m_t_f = m_bool.to(dtype=r_t.dtype)
        m_star_f = state.m_star.to(dtype=r_t.dtype)
        active_f = active.to(dtype=r_t.dtype)
        anchor_error = active_f * m_t_f * m_star_f * w_t * (r_t - state.r_star)

        denominator = self._structural_denominator.to(dtype=r_t.dtype)
        usable_count = usable.sum(dim=(1, 2))
        valid_fraction = usable_count.to(dtype=r_t.dtype) / denominator
        valid_fraction = torch.where(state.active, valid_fraction, torch.zeros_like(valid_fraction))

        per_beam_loss = huber_loss(anchor_error, self.cfg.huber_delta)
        loss = (per_beam_loss * usable.to(dtype=r_t.dtype)).sum(dim=(1, 2)) / denominator
        loss = torch.where(state.active, loss, torch.zeros_like(loss))

        abs_error = anchor_error.abs()
        safe_count = usable_count.clamp_min(1).to(dtype=r_t.dtype)
        error_mean = (abs_error * usable.to(dtype=r_t.dtype)).sum(dim=(1, 2)) / safe_count
        error_mean = torch.where((state.active & (usable_count > 0)), error_mean, torch.zeros_like(error_mean))

        error_max = torch.zeros(N, dtype=r_t.dtype, device=self.device)
        if usable.any():
            masked_abs = abs_error.masked_fill(~usable, 0.0)
            error_max = masked_abs.amax(dim=(1, 2))
            error_max = torch.where(state.active & (usable_count > 0), error_max, torch.zeros_like(error_max))

        metrics = {
            "anchor_active": state.active.to(dtype=r_t.dtype).reshape(N, 1),
            "anchor_loss": loss.reshape(N, 1),
            "anchor_valid_fraction": valid_fraction.reshape(N, 1),
            "anchor_error_mean": error_mean.reshape(N, 1),
            "anchor_error_max": error_max.reshape(N, 1),
            "anchor_hold_steps": state.hold_steps.to(dtype=r_t.dtype).reshape(N, 1),
            "anchor_activation_count": state.activation_count.to(dtype=r_t.dtype).reshape(N, 1),
            "anchor_reset_reason": reset_reason.reshape(N, 1),
        }
        cache = {
            "anchor_error": anchor_error,
            "usable_anchor_mask": usable,
        }
        return AnchorStepOutput(metrics=metrics, cache=cache)

    def _clear_anchor(self, env_selector):
        state = self.state
        state.active[env_selector] = False
        state.r_star[env_selector] = 0.0
        state.m_star[env_selector] = False
        state.w_star[env_selector] = 0.0
        state.hold_steps[env_selector] = 0

    def _normalize_env_ids(self, env_ids):
        if env_ids is None:
            return torch.arange(self.state.active.shape[0], device=self.device)
        if isinstance(env_ids, int):
            env_ids = torch.tensor([env_ids], dtype=torch.long, device=self.device)
        elif isinstance(env_ids, (list, tuple)):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        elif isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        else:
            raise TypeError("env_ids must be None, int, list, tuple, or tensor")
        return env_ids

    def _validate_step_inputs(
        self,
        r_t: torch.Tensor,
        m_t: torch.Tensor,
        w_t: torch.Tensor,
        v_cmd: torch.Tensor,
    ):
        H, V = self.cfg.lidar_hbeams, self.cfg.lidar_vbeams
        if r_t.dim() != 3 or r_t.shape[1:] != (H, V):
            raise ValueError("r_t must have shape [N, H, V]")
        if m_t.shape != r_t.shape:
            raise ValueError("m_t must have shape [N, H, V]")
        if w_t.shape != r_t.shape:
            raise ValueError("w_t must have shape [N, H, V]")
        if m_t.device != r_t.device or w_t.device != r_t.device:
            raise ValueError("r_t, m_t, and w_t must share device")
        if not torch.isfinite(r_t).all():
            raise ValueError("r_t must be finite")
        if not torch.isfinite(w_t).all():
            raise ValueError("w_t must be finite")
        if m_t.dtype == torch.bool:
            m_bool = m_t
        else:
            if not torch.isfinite(m_t).all():
                raise ValueError("numeric m_t must be finite")
            m_bool = m_t > 0
        if v_cmd.device != r_t.device:
            raise ValueError("v_cmd must share device with r_t")
        if v_cmd.dim() == 3 and v_cmd.shape[1:] == (1, 3):
            v_cmd = v_cmd.squeeze(1)
        elif not (v_cmd.dim() == 2 and v_cmd.shape[1] == 3):
            raise ValueError("v_cmd must have shape [N, 3] or [N, 1, 3]")
        if v_cmd.shape[0] != r_t.shape[0]:
            raise ValueError("v_cmd batch size must match r_t")
        if not torch.isfinite(v_cmd).all():
            raise ValueError("v_cmd must be finite")
        return r_t, m_bool.bool(), w_t.clamp(0.0, 1.0), v_cmd

    def _validate_reset_mask(self, mask: Optional[torch.Tensor], N: int, name: str) -> torch.Tensor:
        if mask is None:
            return torch.zeros(N, dtype=torch.bool, device=self.device)
        if mask.dtype != torch.bool:
            raise TypeError(f"{name} must be bool")
        if mask.shape != (N,):
            raise ValueError(f"{name} must have shape [N]")
        if mask.device != self.device:
            raise ValueError(f"{name} must be on the manager device")
        return mask
