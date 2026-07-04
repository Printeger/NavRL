"""
instinctRL Command Adapter
===========================
Body-frame velocity → world-frame transform for VelController.

The instinctRL governor outputs body-frame velocity commands.
VelController(LeePositionController) expects world-frame velocity.
This adapter rotates body-frame commands to world-frame using
the drone attitude quaternion from privileged controller state.

The actor NEVER accesses the quaternion or any pose information.
The rotation is performed at the controller boundary.

instinctRL-0 / instinctRL-A
"""

import torch
import torch.nn as nn


def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(q.shape[0], 1, 3), v.view(q.shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


class BodyToWorldVelocityAdapter(nn.Module):
    """
    Transforms body-frame velocity commands to world-frame for VelController.

    Frame conventions:
        Body/governor frame: X-forward, Y-left, Z-up
        World frame:         Z-up (gravity-aligned), X,Y inertial horizontal

    Args:
        body_vel:   Velocity in body/governor frame [..., 3]
        drone_quat: Drone attitude quaternion [..., 4] (w, x, y, z)
                    from info["drone_state"][..., 3:7] — privileged controller state

    Returns:
        world_vel:  Velocity in world frame [..., 3]
    """

    def forward(
        self, body_vel: torch.Tensor, drone_quat: torch.Tensor
    ) -> torch.Tensor:
        """
        Rotate body-frame velocity to world-frame.

        Uses OmniDrones' quat_rotate for body→world.  In OmniDrones,
        quat_rotate_inverse is used for world→body velocity.
        """
        orig_shape = body_vel.shape
        body_vel_flat = body_vel.reshape(-1, 3)
        drone_quat_flat = drone_quat.reshape(-1, 4)

        world_vel_flat = _quat_rotate(drone_quat_flat, body_vel_flat)

        return world_vel_flat.reshape(orig_shape)
