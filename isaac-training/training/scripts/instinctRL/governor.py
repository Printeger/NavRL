"""
instinctRL Velocity Governor
=============================
Actor-clean body-frame velocity governor helpers.

B0 (instinctRL-A): Minimal governor — alpha=1, v_corr=0, v_gov=v_cmd.
A2: Trainable governor action decoder producing alpha_t and v_corr from the
actor's normalized action without reading privileged simulator state.

Action formula:
    v_gov = alpha_t * v_cmd + v_corr         (body/governor frame)
    v_fin = beta_t * v_gov + (1-beta_t) * v_brake  (after ICS)
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


@dataclass
class GovernorOutput:
    """Output of the velocity governor."""
    alpha: torch.Tensor    # [..., 1]  command scaling factor ∈ [0, 1]
    v_corr: torch.Tensor   # [..., 3]  body-frame correction velocity
    v_gov: torch.Tensor    # [..., 3]  governed velocity = alpha * v_cmd + v_corr


def clip_vector_norm(vector: torch.Tensor, limit: float, eps: float = 1e-6) -> torch.Tensor:
    """Clip vectors by Euclidean norm while preserving direction."""
    if limit <= 0:
        raise ValueError(f"velocity_limit must be > 0, got {limit}")
    norm = torch.linalg.norm(vector, dim=-1, keepdim=True)
    scale = torch.clamp(limit / norm.clamp_min(eps), max=1.0)
    return vector * scale


def latest_state_frame(state_vec: torch.Tensor, frame_dim: int = 13) -> torch.Tensor:
    """Return the latest observation history frame from state_vec."""
    if state_vec.shape[-1] < frame_dim:
        raise ValueError(
            f"state_vec last dimension must be at least {frame_dim}, got {state_vec.shape[-1]}"
        )
    return state_vec[..., -frame_dim:]


def extract_latest_v_cmd_b(state_vec: torch.Tensor) -> torch.Tensor:
    """Actor-clean v_cmd_b from latest state frame: imu6 + v_cmd3 + prev_action3 + age1."""
    frame = latest_state_frame(state_vec)
    return frame[..., 6:9]


def extract_latest_prev_action_b(state_vec: torch.Tensor) -> torch.Tensor:
    """Actor-clean previous issued body command from latest state frame."""
    frame = latest_state_frame(state_vec)
    return frame[..., 9:12]


class TrainableGovernorDecoder(nn.Module):
    """
    Decode the actor's normalized 4D governor action into body-frame v_gov.

    Inputs are actor-clean:
        action_normalized[..., 0]   -> alpha in [0, 1]
        action_normalized[..., 1:4] -> bounded v_corr
        state_vec latest frame      -> v_cmd_b and previous issued body command
    """

    action_dim = 4

    def __init__(
        self,
        v_corr_limit: float = 0.5,
        v_corr_z_limit: Optional[float] = None,
        velocity_limit: float = 2.0,
        smoothing_tau: float = 0.0,
        null_vcorr_gate_enabled: bool = False,
        null_vcorr_gate_eps: float = 0.25,
        null_vcorr_gate_min: float = 0.25,
        tracking_vcorr_z_gate_enabled: bool = False,
        tracking_vcorr_z_gate_eps: float = 1e-3,
        tracking_vcorr_z_gain: float = 1.0,
    ):
        super().__init__()
        if not torch.isfinite(torch.tensor(float(v_corr_limit))):
            raise ValueError("v_corr_limit must be finite")
        if v_corr_limit < 0:
            raise ValueError(f"v_corr_limit must be >= 0, got {v_corr_limit}")
        if v_corr_z_limit is None:
            v_corr_z_limit = v_corr_limit
        if not torch.isfinite(torch.tensor(float(v_corr_z_limit))):
            raise ValueError("v_corr_z_limit must be finite")
        if v_corr_z_limit < 0:
            raise ValueError(f"v_corr_z_limit must be >= 0, got {v_corr_z_limit}")
        if velocity_limit <= 0:
            raise ValueError(f"velocity_limit must be > 0, got {velocity_limit}")
        if not 0.0 <= smoothing_tau < 1.0:
            raise ValueError(f"smoothing_tau must be in [0, 1), got {smoothing_tau}")
        if null_vcorr_gate_eps <= 0.0:
            raise ValueError(
                f"null_vcorr_gate_eps must be > 0, got {null_vcorr_gate_eps}"
            )
        if not 0.0 <= null_vcorr_gate_min <= 1.0:
            raise ValueError(
                "null_vcorr_gate_min must satisfy 0.0 <= value <= 1.0, "
                f"got {null_vcorr_gate_min}"
            )
        if not torch.isfinite(torch.tensor(float(tracking_vcorr_z_gate_eps))):
            raise ValueError("tracking_vcorr_z_gate_eps must be finite")
        if tracking_vcorr_z_gate_eps <= 0.0:
            raise ValueError(
                "tracking_vcorr_z_gate_eps must be > 0, "
                f"got {tracking_vcorr_z_gate_eps}"
            )
        if not torch.isfinite(torch.tensor(float(tracking_vcorr_z_gain))):
            raise ValueError("tracking_vcorr_z_gain must be finite")
        if not 0.0 <= tracking_vcorr_z_gain <= 1.0:
            raise ValueError(
                "tracking_vcorr_z_gain must satisfy 0.0 <= value <= 1.0, "
                f"got {tracking_vcorr_z_gain}"
            )
        self.v_corr_limit = float(v_corr_limit)
        self.v_corr_z_limit = float(v_corr_z_limit)
        self.velocity_limit = float(velocity_limit)
        self.smoothing_tau = float(smoothing_tau)
        self.null_vcorr_gate_enabled = bool(null_vcorr_gate_enabled)
        self.null_vcorr_gate_eps = float(null_vcorr_gate_eps)
        self.null_vcorr_gate_min = float(null_vcorr_gate_min)
        self.tracking_vcorr_z_gate_enabled = bool(tracking_vcorr_z_gate_enabled)
        self.tracking_vcorr_z_gate_eps = float(tracking_vcorr_z_gate_eps)
        self.tracking_vcorr_z_gain = float(tracking_vcorr_z_gain)

    def forward(
        self,
        action_normalized: torch.Tensor,
        state_vec: torch.Tensor,
        prev_action_b: Optional[torch.Tensor] = None,
    ) -> GovernorOutput:
        if action_normalized.shape[-1] != self.action_dim:
            raise ValueError(
                f"learned governor action must have dim {self.action_dim}, "
                f"got {action_normalized.shape[-1]}"
            )
        if not torch.isfinite(action_normalized).all():
            raise ValueError("learned governor action contains non-finite values")
        if not torch.isfinite(state_vec).all():
            raise ValueError("state_vec contains non-finite values")

        action = action_normalized.clamp(0.0, 1.0)
        alpha = action[..., 0:1]
        v_corr_xy = (2.0 * action[..., 1:3] - 1.0) * self.v_corr_limit
        v_corr_z = (2.0 * action[..., 3:4] - 1.0) * self.v_corr_z_limit
        v_corr = torch.cat((v_corr_xy, v_corr_z), dim=-1)
        v_cmd_b = extract_latest_v_cmd_b(state_vec)
        if self.null_vcorr_gate_enabled:
            command_norm = v_cmd_b.norm(dim=-1, keepdim=True)
            gate = (command_norm / self.null_vcorr_gate_eps).clamp(
                self.null_vcorr_gate_min,
                1.0,
            )
            v_corr = v_corr * gate
        if self.tracking_vcorr_z_gate_enabled:
            tracking_active = v_cmd_b[..., 2:3].abs() > self.tracking_vcorr_z_gate_eps
            z_gain = torch.where(
                tracking_active,
                torch.full_like(v_corr[..., 2:3], self.tracking_vcorr_z_gain),
                torch.ones_like(v_corr[..., 2:3]),
            )
            v_corr = torch.cat((v_corr[..., :2], v_corr[..., 2:3] * z_gain), dim=-1)
        if prev_action_b is None:
            prev_action_b = extract_latest_prev_action_b(state_vec)

        v_gov = alpha * v_cmd_b + v_corr
        if self.smoothing_tau > 0.0:
            v_gov = (1.0 - self.smoothing_tau) * v_gov + self.smoothing_tau * prev_action_b
        v_gov = clip_vector_norm(v_gov, self.velocity_limit)

        return GovernorOutput(alpha=alpha, v_corr=v_corr, v_gov=v_gov)


class MinimalGovernor(nn.Module):
    """
    B0 baseline governor: alpha=1, v_corr=0, v_gov=v_cmd.

    This is the simplest possible governor — the operator command passes
    through unchanged. It establishes the governor interface for future
    trainable versions.

    Interface preserved for future trainable governor:
        forward(v_cmd, obs=None) -> GovernorOutput
    """

    def __init__(self, v_corr_limit: float = 0.0, velocity_limit: float = 2.0):
        super().__init__()
        self.v_corr_limit = v_corr_limit
        self.velocity_limit = velocity_limit

    def forward(
        self,
        v_cmd: torch.Tensor,
        obs: dict = None,  # reserved for future trainable governor
    ) -> GovernorOutput:
        """
        B0 pass-through: alpha=1, v_corr=0, v_gov=v_cmd.

        Args:
            v_cmd: Operator velocity command in body frame [..., 3]
            obs:   Observation dict (unused in B0, reserved for future)

        Returns:
            GovernorOutput with alpha, v_corr, v_gov
        """
        batch_shape = v_cmd.shape[:-1]
        device = v_cmd.device

        alpha = torch.ones(*batch_shape, 1, device=device)
        v_corr = torch.zeros(*batch_shape, 3, device=device)
        v_gov = clip_vector_norm(alpha * v_cmd + v_corr, self.velocity_limit)

        return GovernorOutput(alpha=alpha, v_corr=v_corr, v_gov=v_gov)
