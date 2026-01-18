import torch
import math
from typing import Optional, Sequence


class AdversarialCommandGenerator:
    """
    Vectorized Adversarial Command Generator implementing a state-machine with temporal persistence.

    Modes:
      0: Normal Nav
      1: Aggressive Step
      2: Adversarial Suicide (towards nearest obstacle)
      3: Oscillation
      4: Recovery Hover

    All operations are vectorized PyTorch; no Python for-loops.
    """

    def __init__(self, num_envs: int, device: torch.device, max_vel: float, dt: float):
        self.num_envs = num_envs
        self.device = device
        self.max_vel = float(max_vel)
        self.dt = float(dt)

        # timers: remaining duration for current mode (seconds)
        self.timers = torch.zeros(num_envs, device=device, dtype=torch.float32)

        # current mode per env (0-4)
        self.current_modes = torch.zeros(
            num_envs, device=device, dtype=torch.long)

        # base persistent targets (num_envs, 3)
        self.current_targets = torch.zeros(
            num_envs, 3, device=device, dtype=torch.float32)

        # elapsed time since mode start (for oscillation phase)
        self.elapsed = torch.zeros(
            num_envs, device=device, dtype=torch.float32)

        # Oscillation params per-env (freq Hz, amplitude m/s)
        self.osc_freq = torch.zeros(
            num_envs, device=device, dtype=torch.float32)
        self.osc_amp = torch.zeros(
            num_envs, device=device, dtype=torch.float32)

        # small epsilon for normalization
        self.eps = 1e-8

    def _sample_modes(self, prev_modes: torch.LongTensor, mask: torch.BoolTensor, probabilities: Optional[Sequence[float]] = None):
        """
        Vectorized mode sampling with transition bias: if prev_mode==2 (Adversarial) then bias toward Recovery(4).
        - probabilities: optional base weight list of length 5.
        Returns a tensor of new modes sized (num_envs,) where entries with mask==False are unchanged.
        """
        device = self.device
        n = self.num_envs

        # default uniform base probs
        if probabilities is None:
            base = torch.ones(5, device=device, dtype=torch.float32)
        else:
            base = torch.tensor(
                probabilities, device=device, dtype=torch.float32)
            if base.numel() != 5:
                raise ValueError("probabilities must have length 5")

        # Build per-env probability matrix: (n,5)
        probs = base.unsqueeze(0).repeat(n, 1)

        # If previous mode == 2, bias toward recovery (mode 4)
        prev_is_adv = (prev_modes == 2)
        if prev_is_adv.any():
            # increase weight for recovery
            # We'll set recovery weight to 0.7 and redistribute remaining 0.3 across others proportionally
            recovery_weight = 0.7
            other_total = 1.0 - recovery_weight
            # Normalize other probabilities excluding index 4
            other = base.clone()
            other4 = other[4].item()
            other[4] = 0.0
            other_sum = other.sum().item()
            if other_sum <= 0:
                # fallback uniform for others
                other_norm = torch.ones(5, device=device, dtype=torch.float32)
                other_norm[4] = 0.0
                other_norm = other_norm / other_norm.sum()
            else:
                other_norm = other / other_sum

            # create per-env customized vector
            custom = other_norm * other_total
            custom[4] = recovery_weight

            # apply where prev_is_adv
            probs[prev_is_adv, :] = custom.unsqueeze(
                0).repeat(prev_is_adv.sum().item(), 1)

        # For mask==False keep a dummy row; we'll only sample where mask is True
        to_sample_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        m = to_sample_idx.numel()
        if m == 0:
            return self.current_modes.clone()

        # cumulative sums for categorical sampling
        sel_probs = probs[to_sample_idx, :]
        cdf = torch.cumsum(sel_probs, dim=1)
        totals = cdf[:, -1:].clone()
        # avoid division by zero
        totals = torch.where(totals <= 0, torch.ones_like(totals), totals)
        cdf = cdf / totals

        # sample uniform and map into categories
        rnd = torch.rand(m, device=device).unsqueeze(1)
        # category = first index where rnd <= cdf
        cmp = rnd <= cdf
        # argmax over bool returns first True since cdf is monotonic
        new_cats = torch.argmax(cmp.to(torch.int64), dim=1).to(torch.long)

        new_modes = self.current_modes.clone()
        new_modes[to_sample_idx] = new_cats
        return new_modes

    def _sample_new_targets(self, modes: torch.LongTensor, obstacle_relative_vectors: torch.Tensor):
        """
        Generate base targets for each env according to mode. Vectorized over all envs.
        obstacle_relative_vectors: (num_envs, 3)
        Returns: base_targets (num_envs,3), osc_freq, osc_amp
        """
        device = self.device
        n = self.num_envs
        maxv = self.max_vel

        base_targets = torch.zeros(n, 3, device=device, dtype=torch.float32)
        osc_freq = torch.zeros(n, device=device, dtype=torch.float32)
        osc_amp = torch.zeros(n, device=device, dtype=torch.float32)

        # Mode 0: Normal Nav -> moderate random velocity (0.0 - 0.6*maxv)
        mode0 = (modes == 0)
        if mode0.any():
            s = torch.rand(mode0.sum().item(), 3,
                           device=device) * 2.0 - 1.0  # [-1,1]
            # scale to magnitude in [0, 0.6*maxv]
            mags = torch.rand(mode0.sum().item(), device=device) * 0.6 * maxv
            dir = s / (s.norm(dim=1, keepdim=True) + self.eps)
            base_targets[mode0] = dir * mags.unsqueeze(1)

        # Mode 1: Aggressive Step -> high velocity (0.8-1.0)*maxv in random direction
        mode1 = (modes == 1)
        if mode1.any():
            s = torch.rand(mode1.sum().item(), 3, device=device) * 2.0 - 1.0
            dir = s / (s.norm(dim=1, keepdim=True) + self.eps)
            mags = torch.rand(mode1.sum().item(),
                              device=device) * 0.2 * maxv + 0.8 * maxv
            base_targets[mode1] = dir * mags.unsqueeze(1)

        # Mode 2: Adversarial Suicide -> toward nearest obstacle, normalized * max_vel
        mode2 = (modes == 2)
        if mode2.any():
            vecs = obstacle_relative_vectors[mode2]
            norm = vecs.norm(dim=1, keepdim=True)
            # if zero-vector, fallback to random direction
            zero_mask = (norm.squeeze(-1) <= self.eps)
            if zero_mask.any():
                # random direction for zeros
                s = torch.rand(zero_mask.sum().item(), 3,
                               device=device) * 2.0 - 1.0
                dir_rand = s / (s.norm(dim=1, keepdim=True) + self.eps)
                vecs[zero_mask] = dir_rand * (0.1 * maxv)
                norm = vecs.norm(dim=1, keepdim=True)

            dir_to_obs = vecs / (norm + self.eps)
            base_targets[mode2] = dir_to_obs * maxv

        # Mode 3: Oscillation -> base moderate velocity + per-env freq/amp
        mode3 = (modes == 3)
        if mode3.any():
            s = torch.rand(mode3.sum().item(), 3, device=device) * 2.0 - 1.0
            dir = s / (s.norm(dim=1, keepdim=True) + self.eps)
            # base magnitude in [0.2, 0.6]*maxv
            mags = torch.rand(mode3.sum().item(),
                              device=device) * 0.4 * maxv + 0.2 * maxv
            base_targets[mode3] = dir * mags.unsqueeze(1)
            # freq between 0.5 and 2.0 Hz
            osc_freq[mode3] = torch.rand(
                mode3.sum().item(), device=device) * 1.5 + 0.5
            # amplitude between 0.1*maxv and 0.4*maxv
            osc_amp[mode3] = torch.rand(
                mode3.sum().item(), device=device) * 0.3 * maxv + 0.1 * maxv

        # Mode 4: Recovery Hover -> zero
        # already zero

        return base_targets, osc_freq, osc_amp

    def update_commands(self, drone_pos: torch.Tensor, drone_vel: torch.Tensor, obstacle_relative_vectors: torch.Tensor, probabilities: Optional[Sequence[float]] = None) -> torch.Tensor:
        """
        Update internal state and return commands (num_envs, 3) for this frame.

        Inputs are expected to be tensors on the same device with shapes:
          drone_pos: (num_envs, 3) -- unused directly here but accepted for extensibility
          drone_vel: (num_envs, 3) -- unused directly but accepted for extensibility
          obstacle_relative_vectors: (num_envs, 3) -- vector pointing from drone to nearest obstacle

        Returns:
          commands: Tensor (num_envs, 3)
        """
        device = self.device
        n = self.num_envs

        # Step A: countdown timers
        self.timers = self.timers - self.dt
        # ensure non-negative (not strictly necessary)
        # mask of which envs need resetting
        need_reset = self.timers <= 0.0

        # Step B: For envs where timer <= 0 sample new mode, duration, base target
        if need_reset.any():
            # sample new modes with transition logic
            new_modes = self._sample_modes(
                self.current_modes, need_reset, probabilities=probabilities)

            # sample durations uniformly [1,4]
            m = need_reset.sum().item()
            durations = torch.rand(m, device=device) * 3.0 + 1.0
            # assign
            idxs = torch.nonzero(need_reset, as_tuple=False).squeeze(-1)
            self.timers[idxs] = durations

            # reset elapsed for those envs
            self.elapsed[idxs] = 0.0

            # set new modes
            self.current_modes = new_modes

            # sample base commands and oscillation params for *all* newly chosen modes
            base_targets, osc_freq, osc_amp = self._sample_new_targets(
                self.current_modes, obstacle_relative_vectors)

            # Only update stored params at indices that were reset (to preserve persistence for others)
            self.current_targets[idxs] = base_targets[idxs]
            self.osc_freq[idxs] = osc_freq[idxs]
            self.osc_amp[idxs] = osc_amp[idxs]

        # Step C: Compute output for this frame based on persistent current_targets etc.
        # Increment elapsed for everyone by dt
        self.elapsed = self.elapsed + self.dt

        modes = self.current_modes
        out = self.current_targets.clone()

        # Mode 3: Oscillation -- add sine term
        mask3 = (modes == 3)
        if mask3.any():
            idx3 = torch.nonzero(mask3, as_tuple=False).squeeze(-1)
            freqs = self.osc_freq[idx3]
            amps = self.osc_amp[idx3]
            t = self.elapsed[idx3]
            # sin(2*pi*f*t)
            osc = torch.sin(2.0 * math.pi * freqs * t)
            # oscillation along the base direction unit vector
            base_vecs = self.current_targets[idx3]
            base_dir = base_vecs / \
                (base_vecs.norm(dim=1, keepdim=True) + self.eps)
            # broadcast
            osc_terms = base_dir * (osc.unsqueeze(1) * amps.unsqueeze(1))
            out[idx3] = out[idx3] + osc_terms

        # Mode 4: Recovery hover -> zero
        mask4 = (modes == 4)
        if mask4.any():
            out[mask4] = 0.0

        # Ensure not exceeding max velocity by clamping magnitude
        mags = out.norm(dim=1, keepdim=True)
        too_fast = mags > self.max_vel
        if too_fast.any():
            out = torch.where(too_fast, out / mags * self.max_vel, out)

        return out


__all__ = ["AdversarialCommandGenerator"]
