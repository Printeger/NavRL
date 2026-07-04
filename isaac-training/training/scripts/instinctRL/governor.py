"""
instinctRL Velocity Governor
=============================
Learned velocity governor producing corrected body-frame velocity commands.

B0 (instinctRL-A): Minimal governor — alpha=1, v_corr=0, v_gov=v_cmd.
Future (instinctRL-A2): Trainable governor head producing (alpha_t, v_corr).

Action formula:
    v_gov = alpha_t * v_cmd + v_corr         (body/governor frame)
    v_fin = beta_t * v_gov + (1-beta_t) * v_brake  (after ICS — deferred)

instinctRL-A
"""

import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class GovernorOutput:
    """Output of the velocity governor."""
    alpha: torch.Tensor    # [..., 1]  command scaling factor ∈ [0, 1]
    v_corr: torch.Tensor   # [..., 3]  body-frame correction velocity
    v_gov: torch.Tensor    # [..., 3]  governed velocity = alpha * v_cmd + v_corr


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
        v_gov = alpha * v_cmd + v_corr

        # Clip to velocity bounds
        v_gov = v_gov.clamp(-self.velocity_limit, self.velocity_limit)

        return GovernorOutput(alpha=alpha, v_corr=v_corr, v_gov=v_gov)
