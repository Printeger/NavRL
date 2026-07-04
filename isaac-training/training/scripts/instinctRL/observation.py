"""
instinctRL Observation Builder
===============================
Builds actor-clean observation from MID360 raw range, masks,
reliability weights, IMU cues, command, and history buffer.

Components:
  r_t         — Raw MID360 range vector (true distance, NOT danger-coded)
  m_t         — Valid-return mask (finite + in-range, non-dropout)
  w_t         — Staleness-weighted reliability: w_t = m_t * exp(-age/tau)
  timestamps  — Frame age, simulation time
  IMU cues    — Body angular velocity + gravity direction (no position/velocity)
  v_cmd       — Operator velocity command in body frame
  prev_action — Previous issued governor command
  h_t         — Fixed-size history buffer over all allowed fields

instinctRL-B (D-002)
"""

import torch
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ObservationConfig:
    """Configuration for the instinctRL observation pipeline."""

    history_len: int = 4
    """Number of past frames in the history buffer."""

    enable_noise: bool = False
    """Enable MID360 range noise (Gaussian). Deferred to robustness stage."""

    enable_dropout: bool = False
    """Enable MID360 random dropout. Deferred to robustness stage."""

    tau_staleness: float = 0.5
    """Staleness time constant (seconds) for reliability weight decay."""

    lidar_hbeams: int = 360
    lidar_vbeams: int = 59
    lidar_range: float = 40.0


class MID360ObservationBuilder:
    """
    Builds actor-clean observations from raw MID360 data.

    Usage:
        builder = MID360ObservationBuilder(cfg, device="cuda:0")
        obs = builder.build(ray_hits_w, lidar_pos, drone_state, v_cmd, dt, N)
        hist = builder.build_history(obs)
        actor_input = hist["lidar_grid"], hist["state_vec"]
    """

    def __init__(self, config: ObservationConfig, device: str = "cuda:0"):
        self.cfg = config
        self.device = device
        self._step_counter = 0
        self._last_frame_time = 0.0
        self._prev_action: Optional[torch.Tensor] = None
        self._history: Optional[Dict[str, torch.Tensor]] = None

    # ------------------------------------------------------------------
    # Single-frame observation
    # ------------------------------------------------------------------
    def build(
        self,
        ray_hits_w: torch.Tensor,
        lidar_pos_w: torch.Tensor,
        drone_state: torch.Tensor,
        v_cmd: torch.Tensor,
        dt: float,
        num_envs: int,
        prev_action: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Build a single-frame observation from raw sensor data.

        Args:
            ray_hits_w:  Ray hit positions in world frame [N, num_rays, 3]
            lidar_pos_w: LiDAR sensor position in world frame [N, 3]
            drone_state: Full drone state [N, 13] (pos, quat, vel, ang_vel)
            v_cmd:       Body-frame velocity command [N, 1, 3] or [N, 3]
            dt:          Physics timestep (seconds)
            num_envs:    Number of parallel environments
            prev_action: Previous issued governor command [N, 3] (optional)

        Returns:
            Dict with: range, mask, weight, imu, v_cmd, prev_action,
            frame_age, sim_time
        """
        N = num_envs
        H, V = self.cfg.lidar_hbeams, self.cfg.lidar_vbeams
        max_range = self.cfg.lidar_range

        # --- 1. Raw MID360 range r_t (true distance, NOT danger-coded) ---
        num_rays = H * V
        ray_hits = ray_hits_w.reshape(N, num_rays, 3)
        pos = lidar_pos_w.reshape(N, 1, 3)
        raw_range = (ray_hits - pos).norm(dim=-1).clamp_max(max_range)
        raw_range = raw_range.reshape(N, H, V)

        # --- 2. Valid-return mask m_t ---
        # Finite, in-range, above blind-spot (0.01m), non-dropout
        mask = (
            torch.isfinite(raw_range)
            & (raw_range > 0.01)
            & (raw_range < max_range)
        ).float()

        # --- 3. Staleness-weighted reliability w_t = m_t * exp(-age/tau) ---
        self._step_counter += 1
        self._last_frame_time += dt
        if self._step_counter == 1:
            frame_age = torch.zeros(N, 1, device=self.device)
        else:
            frame_age = torch.full((N, 1), dt, device=self.device)

        tau = max(self.cfg.tau_staleness, 0.01)
        staleness_factor = torch.exp(-frame_age / tau).reshape(N, 1, 1)
        weight = mask * staleness_factor

        # --- 4. IMU cues: body angular velocity + gravity direction ---
        # These are allowed actor inputs — no position, linear velocity, or
        # privileged simulator state.
        ang_vel_body = drone_state[..., 10:13]  # [N, 3]  body angular velocity
        quat = drone_state[..., 3:7]            # [N, 4]  attitude quaternion

        # Gravity direction in body frame (world→body rotation)
        gravity_world = torch.tensor([0.0, 0.0, -1.0], device=self.device)
        gravity_dir_body = self._rotate_world_to_body(gravity_world, quat)
        imu_cues = torch.cat([ang_vel_body, gravity_dir_body], dim=-1)  # [N, 6]

        # --- 5. Body-frame v_cmd ---
        if v_cmd.dim() == 3 and v_cmd.shape[1] == 1:
            v_cmd = v_cmd.squeeze(1)  # [N, 1, 3] → [N, 3]
        v_cmd = v_cmd.reshape(N, 3)

        # --- 6. Previous action ---
        if prev_action is None:
            prev_action = (
                self._prev_action
                if self._prev_action is not None
                else torch.zeros(N, 3, device=self.device)
            )
        self._prev_action = prev_action.reshape(N, 3).clone()

        # --- 7. Sim time ---
        sim_time = torch.full((N, 1), self._last_frame_time, device=self.device)

        return {
            "range": raw_range,
            "mask": mask,
            "weight": weight,
            "imu": imu_cues,
            "v_cmd": v_cmd,
            "prev_action": prev_action.reshape(N, 3),
            "frame_age": frame_age,
            "sim_time": sim_time,
        }

    # ------------------------------------------------------------------
    # History buffer
    # ------------------------------------------------------------------
    def build_history(
        self, current: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Stack current frame into history buffer.

        Returns:
            "lidar_grid": [N, L*3, H, V]  interleaved range/mask/weight channels
            "state_vec":  [N, L*13]       stacked IMU+v_cmd+prev_action+age
        """
        N = current["range"].shape[0]
        H, V = self.cfg.lidar_hbeams, self.cfg.lidar_vbeams
        L = self.cfg.history_len

        # Lazy-init history buffer
        if self._history is None:
            self._history = {}
            for key in ["range", "mask", "weight"]:
                self._history[key] = torch.zeros(N, L, H, V, device=self.device)
            # state: imu(6) + v_cmd(3) + prev_action(3) + frame_age(1) = 13
            self._history["state"] = torch.zeros(N, L, 13, device=self.device)

        # Shift buffer (oldest out, newest in at position -1)
        for key in ["range", "mask", "weight"]:
            self._history[key] = torch.roll(self._history[key], shifts=-1, dims=1)
            self._history[key][:, -1] = current[key]

        # State vector for this frame
        state_now = torch.cat(
            [
                current["imu"],
                current["v_cmd"],
                current["prev_action"],
                current["frame_age"],
            ],
            dim=-1,
        )  # [N, 13]
        self._history["state"] = torch.roll(self._history["state"], shifts=-1, dims=1)
        self._history["state"][:, -1] = state_now

        # Build lidar grid: interleave channels (r0,m0,w0, r1,m1,w1, ...)
        channels = []
        for t in range(L):
            channels.append(self._history["range"][:, t])
            channels.append(self._history["mask"][:, t])
            channels.append(self._history["weight"][:, t])
        lidar_grid = torch.stack(channels, dim=1)  # [N, L*3, H, V]

        # State vector: flatten across history frames
        state_vec_out = self._history["state"].reshape(N, -1)  # [N, L*13]

        return {"lidar_grid": lidar_grid, "state_vec": state_vec_out}

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset_history(self, env_ids: Optional[torch.Tensor] = None):
        """Reset history buffer for specified environments (or all)."""
        if self._history is not None:
            if env_ids is None:
                self._history = None
            else:
                for k in self._history:
                    self._history[k][env_ids] = 0.0
        self._step_counter = 0
        self._last_frame_time = 0.0
        self._prev_action = None

    # ------------------------------------------------------------------
    # Utility: world → body vector rotation
    # ------------------------------------------------------------------
    @staticmethod
    def _rotate_world_to_body(v: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
        """
        Rotate a world-frame vector to body-frame using attitude quaternion.

        Uses inverse quaternion rotation: if q rotates world→body,
        then v_body = q_conj * v_world_quat * q.
        """
        v = v.reshape(1, 3).expand(quat.shape[0], 3)
        q_w = quat[..., 0]
        q_vec = quat[..., 1:]
        a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
        b = torch.linalg.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
        c = q_vec * (q_vec * v).sum(dim=-1, keepdim=True) * 2.0
        return a - b + c  # inverse rotation: world → body
