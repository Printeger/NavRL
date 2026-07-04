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
from omni_drones.utils.torch import quat_rotate_inverse


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

        Uses quat_rotate_inverse: if q rotates world→body,
        then q_inverse rotates body→world.
        """
        orig_shape = body_vel.shape
        body_vel_flat = body_vel.reshape(-1, 3)
        drone_quat_flat = drone_quat.reshape(-1, 4)

        world_vel_flat = quat_rotate_inverse(drone_quat_flat, body_vel_flat)

        return world_vel_flat.reshape(orig_shape)
