"""
工具函数集合 (Utility Functions)
==================================
包含训练中使用的各种辅助类和函数：
1. ValueNorm: 价值归一化（稳定训练）
2. GAE: 广义优势估计
3. Actor/Critic 网络结构
4. 坐标变换函数
5. 评估函数

这些工具是强化学习训练的基础组件。
"""

import math
import torch
import torch.nn as nn
import wandb
import numpy as np
from typing import Iterable, Union
from tensordict.tensordict import TensorDict
from omni_drones.utils.torchrl import RenderCallback
from torchrl.envs.utils import ExplorationType, set_exploration_type

from instinctRL.task_metrics import (
    R5E1_LAG_STEPS,
    R5E2_COLLISION_REASON_LABELS,
    R5E2_COLLISION_WINDOW_STEPS,
    R5E2_COLLISION_WINDOW_VALUE_FIELDS,
    R5E2_DIAGNOSTIC_FIELDS,
    R5E2_REASON_NOT_COLLISION_TERMINATION,
    R5E3_COLLISION_WINDOW_STEPS,
    R5E3_DIAGNOSTIC_FIELDS,
    R5E3_LOW_BETA_WINDOW_STEPS,
    R5E3_WINDOW_VALUE_FIELDS,
    R5H_COLLISION_WINDOW_STEPS,
    R5H_COLLISION_WINDOW_VALUE_FIELDS,
    R5H_CONCENTRATION_SAMPLE_VALUE_NAMES,
    R5H_CONCENTRATION_VALUE_NAMES,
    R5H_CONDITION_NAMES,
    R5H_ANCHOR_CONDITION_NAMES,
    R5H_ANCHOR_VALUE_NAMES,
    R5H_DIAGNOSTIC_FIELDS,
    R5H_STATION_VALUE_NAMES,
    TERMINATION_ABOVE_BOUND,
    TERMINATION_BELOW_BOUND,
    TERMINATION_COLLISION,
    TERMINATION_TIMEOUT,
    compute_r5e1_controller_latency_step_metrics,
    compute_r5e1_lagged_command_metrics,
    compute_r5h_mechanism_step_metrics,
    compute_r5g_downward_step_metrics,
    compute_r5g_station_anchor_step_metrics,
    compute_r5e_mechanism_step_metrics,
    compute_vertical_channel_step_metrics,
)

# ============================================
# 价值归一化（Value Normalization）
# ============================================
class ValueNorm(nn.Module):
    """
    价值归一化模块
    
    作用：将价值函数 V(s) 归一化到合适的范围，稳定训练。
    
    原理：维护回报的滑动平均和方差，使用它们归一化价值。
    V_normalized = (V - mean) / sqrt(var)
    
    参数:
        input_shape: 输入形状（通常是1，表示标量价值）
        beta: 滑动平均系数（0.995 表示慢速更新）
        epsilon: 数值稳定性常数
    """
    def __init__(
        self,
        input_shape: Union[int, Iterable],
        beta=0.995,  # 滑动平均系数
        epsilon=1e-5,  # 防止除零
    ) -> None:
        super().__init__()

        self.input_shape = (
            torch.Size(input_shape)
            if isinstance(input_shape, Iterable)
            else torch.Size((input_shape,))
        )
        self.epsilon = epsilon
        self.beta = beta

        # 注册缓冲区（会被保存到模型中，但不会被优化）
        self.running_mean: torch.Tensor  # 滑动平均
        self.running_mean_sq: torch.Tensor  # 平方的滑动平均（用于计算方差）
        self.debiasing_term: torch.Tensor  # 去偏项
        self.register_buffer("running_mean", torch.zeros(input_shape, dtype=torch.float64))
        self.register_buffer("running_mean_sq", torch.zeros(input_shape, dtype=torch.float64))
        self.register_buffer("debiasing_term", torch.tensor(0.0, dtype=torch.float64))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_mean_sq.zero_()
        self.debiasing_term.zero_()

    def running_mean_var(self):
        """计算去偏后的均值和方差"""
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(
            min=self.epsilon
        )
        # Var(X) = E[X²] - E[X]²
        debiased_var = (debiased_mean_sq - debiased_mean**2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    @torch.no_grad()
    def update(self, input_vector: torch.Tensor):
        """
        更新滑动平均统计量
        
        参数:
            input_vector: 一批回报值 G_t
        """
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        if not torch.isfinite(input_vector).all():
            raise ValueError("ValueNorm.update received non-finite returns")
        dim = tuple(range(input_vector.dim() - len(self.input_shape)))
        input_fp64 = input_vector.to(dtype=torch.float64)
        batch_mean = input_fp64.mean(dim=dim)
        batch_sq_mean = input_fp64.square().mean(dim=dim)
        if not torch.isfinite(batch_mean).all() or not torch.isfinite(batch_sq_mean).all():
            raise ValueError("ValueNorm.update produced non-finite batch statistics")

        weight = self.beta  # 滑动平均权重

        # 指数移动平均：new = weight * old + (1 - weight) * new_sample
        next_mean = self.running_mean * weight + batch_mean * (1.0 - weight)
        next_mean_sq = self.running_mean_sq * weight + batch_sq_mean * (1.0 - weight)
        next_debiasing = self.debiasing_term * weight + 1.0 * (1.0 - weight)
        if (
            not torch.isfinite(next_mean).all()
            or not torch.isfinite(next_mean_sq).all()
            or not torch.isfinite(next_debiasing).all()
        ):
            raise ValueError("ValueNorm.update would make running statistics non-finite")
        self.running_mean.copy_(next_mean)
        self.running_mean_sq.copy_(next_mean_sq)
        self.debiasing_term.copy_(next_debiasing)

    def normalize(self, input_vector: torch.Tensor):
        """归一化：(x - mean) / std"""
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        mean, var = self.running_mean_var()
        input_fp64 = input_vector.to(dtype=torch.float64)
        out = (input_fp64 - mean.to(device=input_vector.device)) / torch.sqrt(
            var.to(device=input_vector.device)
        )
        return out.to(dtype=input_vector.dtype)

    def denormalize(self, input_vector: torch.Tensor, max_abs: float = None):
        """反归一化：x * std + mean"""
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        mean, var = self.running_mean_var()
        input_fp64 = input_vector.to(dtype=torch.float64)
        out = input_fp64 * torch.sqrt(var.to(device=input_vector.device)) + mean.to(
            device=input_vector.device
        )
        if max_abs is not None:
            max_abs = float(max_abs)
            if max_abs <= 0:
                raise ValueError(f"ValueNorm denormalize max_abs must be > 0, got {max_abs}")
            out = out.clamp(-max_abs, max_abs)
        return out.to(dtype=input_vector.dtype)

# ============================================
# MLP 构建器
# ============================================
def make_mlp(num_units):
    """
    创建多层感知机（MLP）
    
    参数:
        num_units: 列表，每个元素是一层的神经元数
                  例如 [128, 64] 表示两层，128 -> 64
    
    返回:
        nn.Sequential: MLP 模块
    
    每层结构：Linear -> LeakyReLU -> LayerNorm
    """
    layers = []
    for n in num_units:
        layers.append(nn.LazyLinear(n))  # 全连接层
        layers.append(nn.LeakyReLU())     # 激活函数
        layers.append(nn.LayerNorm(n))    # 层归一化（稳定训练）
    return nn.Sequential(*layers)

# ============================================
# 概率分布类
# ============================================

class IndependentNormal(torch.distributions.Independent):
    """
    独立正态分布
    
    用于连续动作空间，每个动作维度独立采样。
    例如：3维速度 [vx, vy, vz]，每个维度服从独立的正态分布。
    
    参数:
        loc: 均值 μ
        scale: 标准差 σ（必须 > 0）
    """
    arg_constraints = {
        "loc": torch.distributions.constraints.real, 
        "scale": torch.distributions.constraints.positive
    } 
    
    def __init__(self, loc, scale, validate_args=None):
        scale = torch.clamp_min(scale, 1e-6)  # 确保标准差 > 0
        base_dist = torch.distributions.Normal(loc, scale)
        super().__init__(base_dist, 1, validate_args=validate_args)

class IndependentBeta(torch.distributions.Independent):
    """
    独立 Beta 分布
    
    用于有界动作空间 [0, 1]。Beta 分布比正态分布更适合有界空间，
    因为它自然地在 [0, 1] 内采样，不需要额外裁剪。
    
    参数:
        alpha: Beta 分布参数 α（必须 > 0）
        beta: Beta 分布参数 β（必须 > 0）
    
    性质:
        - α = β: 对称分布
        - α > β: 偏向 1
        - α < β: 偏向 0
    """
    arg_constraints = {
        "alpha": torch.distributions.constraints.positive, 
        "beta": torch.distributions.constraints.positive
    }

    def __init__(self, alpha, beta, validate_args=None):
        beta_dist = torch.distributions.Beta(alpha, beta)
        super().__init__(beta_dist, 1, validate_args=validate_args)

# ============================================
# Actor 网络类
# ============================================

class Actor(nn.Module):
    """
    Gaussian Actor（高斯策略）
    
    输出正态分布的参数：均值 μ 和标准差 σ
    动作采样：a ~ N(μ(s), σ)
    
    参数:
        action_dim: 动作维度（例如 3 表示 [vx, vy, vz]）
    """
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.actor_mean = nn.LazyLinear(action_dim)  # 输出均值 μ
        self.actor_std = nn.Parameter(torch.zeros(action_dim))  # 可学习的标准差 log(σ)
    
    def forward(self, features: torch.Tensor):
        """
        前向传播
        
        参数:
            features: 特征向量（来自 feature_extractor）
        
        返回:
            loc: 均值 μ
            scale: 标准差 σ = exp(actor_std)
        """
        loc = self.actor_mean(features)
        scale = torch.exp(self.actor_std).expand_as(loc)
        return loc, scale

class BetaActor(nn.Module):
    """
    Beta Actor（Beta 策略）
    
    输出 Beta 分布的参数：α 和 β
    动作采样：a ~ Beta(α(s), β(s))，a ∈ [0, 1]
    
    优势：相比正态分布，Beta 分布自然支持有界动作空间
    
    参数:
        action_dim: 动作维度
    """
    def __init__(
        self,
        action_dim: int,
        alpha_min: float = 1.0,
        alpha_max: float = 30.0,
        beta_min: float = 1.0,
        beta_max: float = 30.0,
    ) -> None:
        super().__init__()
        for name, value in {
            "alpha_min": alpha_min,
            "alpha_max": alpha_max,
            "beta_min": beta_min,
            "beta_max": beta_max,
        }.items():
            if not torch.isfinite(torch.tensor(float(value))):
                raise ValueError(f"{name} must be finite")
        if alpha_min <= 0 or beta_min <= 0:
            raise ValueError("BetaActor concentration minimums must be > 0")
        if alpha_max <= alpha_min:
            raise ValueError("alpha_max must be > alpha_min")
        if beta_max <= beta_min:
            raise ValueError("beta_max must be > beta_min")
        self.alpha_layer = nn.LazyLinear(action_dim)  # 输出 α
        self.beta_layer = nn.LazyLinear(action_dim)   # 输出 β
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
    
    def forward(self, features: torch.Tensor):
        """
        前向传播
        
        参数:
            features: 特征向量
        
        返回:
            alpha: Beta 分布参数 α (> 1)
            beta: Beta 分布参数 β (> 1)
        """
        raw_alpha = self.alpha_layer(features)
        raw_beta = self.beta_layer(features)
        if not torch.isfinite(raw_alpha).all():
            raise ValueError("BetaActor raw_alpha contains non-finite values")
        if not torch.isfinite(raw_beta).all():
            raise ValueError("BetaActor raw_beta contains non-finite values")
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(raw_alpha)
        beta = self.beta_min + (self.beta_max - self.beta_min) * torch.sigmoid(raw_beta)
        if not torch.isfinite(alpha).all() or not torch.isfinite(beta).all():
            raise ValueError("BetaActor concentration parameters contain non-finite values")
        if (alpha < self.alpha_min).any() or (alpha > self.alpha_max).any():
            raise ValueError("BetaActor alpha escaped configured bounds")
        if (beta < self.beta_min).any() or (beta > self.beta_max).any():
            raise ValueError("BetaActor beta escaped configured bounds")
        return alpha, beta

# ============================================
# GAE (Generalized Advantage Estimation)
# ============================================
class GAE(nn.Module):
    """
    广义优势估计（GAE）
    
    GAE 是一种权衡偏差和方差的优势函数估计方法：
    A^GAE_t = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
    其中 δ_t = r_t + γV(s_{t+1}) - V(s_t) 是 TD 误差
    
    参数:
        gamma: 折扣因子 γ ∈ [0, 1]
               - 接近 1: 更关注长期回报
               - 接近 0: 更关注即时回报
        lmbda: GAE 参数 λ ∈ [0, 1]
               - λ = 0: TD(0)，低方差高偏差
               - λ = 1: 蒙特卡洛，高方差低偏差
               - λ = 0.95: 常用折中值
    
    参考：https://arxiv.org/abs/1506.02438
    """
    def __init__(self, gamma, lmbda):
        super().__init__()
        self.register_buffer("gamma", torch.tensor(gamma))
        self.register_buffer("lmbda", torch.tensor(lmbda))
        self.gamma: torch.Tensor
        self.lmbda: torch.Tensor
    
    def forward(
        self, 
        reward: torch.Tensor,      # r_t
        terminated: torch.Tensor,  # 是否终止
        value: torch.Tensor,       # V(s_t)
        next_value: torch.Tensor   # V(s_{t+1})
    ):
        """
        计算 GAE 优势函数和回报
        
        参数:
            reward: [num_envs, num_steps]
            terminated: [num_envs, num_steps]
            value: [num_envs, num_steps]
            next_value: [num_envs, num_steps]
        
        返回:
            advantages: 优势函数 A(s,a)
            returns: 回报 G_t = A_t + V_t
        """
        num_steps = terminated.shape[1]
        advantages = torch.zeros_like(reward)
        not_done = 1 - terminated.float()  # 如果终止，后续价值为 0
        gae = 0
        
        # 从后向前计算 GAE
        for step in reversed(range(num_steps)):
            # TD 误差: δ_t = r_t + γV(s_{t+1}) - V(s_t)
            delta = (
                reward[:, step] 
                + self.gamma * next_value[:, step] * not_done[:, step] 
                - value[:, step]
            )
            # GAE: A_t = δ_t + γλ δ_{t+1} + (γλ)² δ_{t+2} + ...
            advantages[:, step] = gae = delta + (self.gamma * self.lmbda * not_done[:, step] * gae) 
        
        # 回报 G_t = A_t + V_t
        returns = advantages + value
        return advantages, returns

# ============================================
# 辅助函数
# ============================================

def make_batch(tensordict: TensorDict, num_minibatches: int):
    """
    将数据分成多个 minibatch
    
    PPO 使用小批量更新，而不是全批量更新，可以：
    1. 提高样本效率
    2. 稳定训练
    3. 节省内存
    
    参数:
        tensordict: 完整的训练数据
        num_minibatches: minibatch 数量
    
    返回:
        生成器，每次 yield 一个 minibatch
    """
    # 展平为一维
    tensordict = tensordict.reshape(-1) 
    usable = (tensordict.shape[0] // num_minibatches) * num_minibatches
    if usable <= 0:
        raise ValueError(
            "num_minibatches is larger than the collected PPO batch. "
            f"batch={tensordict.shape[0]}, num_minibatches={num_minibatches}"
        )
    
    # 随机打乱索引并分成 num_minibatches 组
    perm = torch.randperm(
        usable,
        device=tensordict.device,
    ).reshape(num_minibatches, -1)
    
    # 逐个返回 minibatch
    for indices in perm:
        yield tensordict[indices]

def _json_safe_scalar(value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("expected a scalar tensor")
        value = value.detach().cpu().item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _tensor_summary(value: torch.Tensor):
    tensor = value.detach().float().cpu().reshape(-1)
    count = int(tensor.numel())
    finite = torch.isfinite(tensor)
    finite_count = int(finite.sum().item())
    summary = {
        "count": count,
        "finite_count": finite_count,
    }
    if finite_count == 0:
        return summary

    finite_tensor = tensor[finite]
    summary.update({
        "mean": float(finite_tensor.mean().item()),
        "min": float(finite_tensor.min().item()),
        "max": float(finite_tensor.max().item()),
    })
    summary["std"] = (
        float(finite_tensor.std(unbiased=False).item())
        if finite_count > 1
        else 0.0
    )
    return summary


def _get_optional_tensor(tensordict, candidates):
    for key in candidates:
        try:
            value = tensordict.get(key)
        except KeyError:
            continue
        if isinstance(value, torch.Tensor):
            return value
    return None


def _r5e2_optional_field_candidates():
    return {
        field_name: [
            ("info", field_name),
            ("next", "info", field_name),
        ]
        for field_name in R5E2_DIAGNOSTIC_FIELDS
    }


def _r5e3_optional_field_candidates():
    return {
        field_name: [
            ("info", field_name),
            ("next", "info", field_name),
        ]
        for field_name in R5E3_DIAGNOSTIC_FIELDS
    }


def _json_safe_eval_summary(info, trajs, cfg=None):
    summary = {
        key: _json_safe_scalar(value)
        for key, value in info.items()
        if key != "recording"
    }

    optional_fields = {
        "governor_alpha": ["governor_alpha", ("next", "governor_alpha")],
        "governor_v_corr": ["governor_v_corr", ("next", "governor_v_corr")],
        "governor_v_corr_z": ["governor_v_corr_z", ("next", "governor_v_corr_z")],
        "governor_v_cmd_b": ["governor_v_cmd_b", ("next", "governor_v_cmd_b")],
        "governor_v_cmd_b_z": ["governor_v_cmd_b_z", ("next", "governor_v_cmd_b_z")],
        "governor_v_gov_b": ["governor_v_gov_b", ("next", "governor_v_gov_b")],
        "governor_v_gov_b_z": ["governor_v_gov_b_z", ("next", "governor_v_gov_b_z")],
        "governor_v_final_b": ["governor_v_final_b", ("next", "governor_v_final_b")],
        "governor_v_final_b_z": ["governor_v_final_b_z", ("next", "governor_v_final_b_z")],
        "actual_velocity_b": [
            ("info", "actual_velocity_b"),
            ("next", "info", "actual_velocity_b"),
        ],
        "r5e1_controller_command_w": [
            ("info", "r5e1_controller_command_w"),
            ("next", "info", "r5e1_controller_command_w"),
        ],
        "r5e1_actual_velocity_w": [
            ("info", "r5e1_actual_velocity_w"),
            ("next", "info", "r5e1_actual_velocity_w"),
        ],
        "drone_state": [
            ("info", "drone_state"),
            ("next", "info", "drone_state"),
        ],
        "min_clearance": [
            ("info", "min_clearance"),
            ("next", "info", "min_clearance"),
            ("info", "safety_min_clearance"),
            ("next", "info", "safety_min_clearance"),
        ],
        "ics_beta": [("info", "ics_beta"), ("next", "info", "ics_beta")],
        "ics_active_beam_count": [
            ("info", "ics_active_beam_count"),
            ("next", "info", "ics_active_beam_count"),
        ],
        "ics_min_clearance": [
            ("info", "ics_min_clearance"),
            ("next", "info", "ics_min_clearance"),
        ],
        "ics_worst_beam_index": [
            ("info", "ics_worst_beam_index"),
            ("next", "info", "ics_worst_beam_index"),
        ],
        "ics_emergency": [("info", "ics_emergency"), ("next", "info", "ics_emergency")],
        "ics_command_speed": [
            ("info", "ics_command_speed"),
            ("next", "info", "ics_command_speed"),
        ],
        "ics_final_speed": [
            ("info", "ics_final_speed"),
            ("next", "info", "ics_final_speed"),
        ],
        "ics_downward_active": [
            ("info", "ics_downward_active"),
            ("next", "info", "ics_downward_active"),
        ],
        "ics_downward_has_ray": [
            ("info", "ics_downward_has_ray"),
            ("next", "info", "ics_downward_has_ray"),
        ],
        "ics_downward_beta": [
            ("info", "ics_downward_beta"),
            ("next", "info", "ics_downward_beta"),
        ],
        "ics_downward_min_clearance": [
            ("info", "ics_downward_min_clearance"),
            ("next", "info", "ics_downward_min_clearance"),
        ],
        "ics_downward_pre_z": [
            ("info", "ics_downward_pre_z"),
            ("next", "info", "ics_downward_pre_z"),
        ],
        "ics_downward_post_z": [
            ("info", "ics_downward_post_z"),
            ("next", "info", "ics_downward_post_z"),
        ],
        "ics_downward_z_delta_abs": [
            ("info", "ics_downward_z_delta_abs"),
            ("next", "info", "ics_downward_z_delta_abs"),
        ],
        "ics_downward_attenuation_ratio": [
            ("info", "ics_downward_attenuation_ratio"),
            ("next", "info", "ics_downward_attenuation_ratio"),
        ],
        "ics_residual_preemption_trigger": [
            ("info", "ics_residual_preemption_trigger"),
            ("next", "info", "ics_residual_preemption_trigger"),
        ],
        "ics_residual_preemption_range_rate_available": [
            ("info", "ics_residual_preemption_range_rate_available"),
            ("next", "info", "ics_residual_preemption_range_rate_available"),
        ],
        "null_command_speed": [
            ("info", "null_command_speed"),
            ("next", "info", "null_command_speed"),
        ],
        "null_command_output_speed": [
            ("info", "null_command_output_speed"),
            ("next", "info", "null_command_output_speed"),
        ],
        "command_amplification": [
            ("info", "command_amplification"),
            ("next", "info", "command_amplification"),
        ],
        "command_amplification_active": [
            ("info", "command_amplification_active"),
            ("next", "info", "command_amplification_active"),
        ],
        "command_amplification_horizontal": [
            ("info", "command_amplification_horizontal"),
            ("next", "info", "command_amplification_horizontal"),
        ],
        "command_amplification_horizontal_active": [
            ("info", "command_amplification_horizontal_active"),
            ("next", "info", "command_amplification_horizontal_active"),
        ],
        "command_amplification_vertical": [
            ("info", "command_amplification_vertical"),
            ("next", "info", "command_amplification_vertical"),
        ],
        "command_amplification_vertical_active": [
            ("info", "command_amplification_vertical_active"),
            ("next", "info", "command_amplification_vertical_active"),
        ],
        "height_world_z": [("info", "height_world_z"), ("next", "info", "height_world_z")],
        "height_floor_violation": [
            ("info", "height_floor_violation"),
            ("next", "info", "height_floor_violation"),
        ],
        "height_ceiling_violation": [
            ("info", "height_ceiling_violation"),
            ("next", "info", "height_ceiling_violation"),
        ],
        "height_ceiling_margin": [
            ("info", "height_ceiling_margin"),
            ("next", "info", "height_ceiling_margin"),
        ],
        "v_cmd_z": [("info", "v_cmd_z"), ("next", "info", "v_cmd_z")],
        "v_final_b_z": [("info", "v_final_b_z"), ("next", "info", "v_final_b_z")],
    }
    optional_fields.update(_r5e2_optional_field_candidates())
    optional_fields.update(_r5e3_optional_field_candidates())

    absent_fields = []
    r5e2_step_accumulators = {
        field_name: _TensorSummaryAccumulator()
        for field_name in R5E2_DIAGNOSTIC_FIELDS
    }
    r5e3_step_accumulators = {
        field_name: _TensorSummaryAccumulator()
        for field_name in R5E3_DIAGNOSTIC_FIELDS
    }
    for field_name, candidates in optional_fields.items():
        value = _get_optional_tensor(trajs, candidates)
        if value is None:
            absent_fields.append(field_name)
            continue
        summary[f"eval/diagnostics.{field_name}"] = _tensor_summary(value)
        if field_name in r5e2_step_accumulators:
            r5e2_step_accumulators[field_name].add(value)
        if field_name in r5e3_step_accumulators:
            r5e3_step_accumulators[field_name].add(value)

    reward = _get_optional_tensor(trajs, [("next", "agents", "reward")])
    if reward is not None:
        reward_summary = _tensor_summary(reward)
        finite_reward = reward.detach().float().cpu().reshape(-1)
        finite_reward = finite_reward[torch.isfinite(finite_reward)]
        if finite_reward.numel() > 0:
            reward_summary["sum"] = float(finite_reward.sum().item())
        summary["eval/reward"] = reward_summary

    if absent_fields:
        summary["absent_optional_fields"] = absent_fields

    r5e1_accumulators = _make_r5e1_diagnostic_accumulators()
    if _accumulate_r5e1_metrics_from_tensordict(
        r5e1_accumulators,
        trajs,
        optional_fields,
        cfg,
    ):
        _add_r5e1_diagnostic_summaries(summary, r5e1_accumulators)
        _add_r5e1_handbook_summary(summary, r5e1_accumulators)

    r5e_accumulators = _make_r5e_diagnostic_accumulators()
    if _accumulate_r5e_metrics_from_tensordict(
        r5e_accumulators,
        trajs,
        optional_fields,
        cfg,
    ):
        _add_r5e_diagnostic_summaries(summary, r5e_accumulators)
        _add_r5e_handbook_summary(summary, r5e_accumulators)

    r5h_accumulators = _make_r5h_diagnostic_accumulators()
    r5h_candidates = _make_optional_eval_field_candidates()
    if _accumulate_r5h_metrics_from_tensordict(
        r5h_accumulators,
        trajs,
        r5h_candidates,
        cfg,
    ):
        _add_r5h_diagnostic_summaries(summary, r5h_accumulators)
        _add_r5h_handbook_summary(summary, r5h_accumulators)
    _add_r5e2_step_handbook_summary(summary, r5e2_step_accumulators)
    _add_r5e3_step_handbook_summary(summary, r5e3_step_accumulators)

    return summary


class _TensorSummaryAccumulator:
    def __init__(self):
        self.count = 0
        self.finite_count = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.min = None
        self.max = None
        self._values = []

    def add(self, value: torch.Tensor):
        tensor = value.detach().float().cpu().reshape(-1)
        self.count += int(tensor.numel())
        finite = tensor[torch.isfinite(tensor)]
        finite_count = int(finite.numel())
        self.finite_count += finite_count
        if finite_count == 0:
            return
        finite_sum = float(finite.sum().item())
        finite_sum_sq = float(finite.square().sum().item())
        finite_min = float(finite.min().item())
        finite_max = float(finite.max().item())
        self.sum += finite_sum
        self.sum_sq += finite_sum_sq
        self.min = finite_min if self.min is None else min(self.min, finite_min)
        self.max = finite_max if self.max is None else max(self.max, finite_max)
        self._values.append(finite)

    def mean(self, default=None):
        if self.finite_count == 0:
            return default
        return self.sum / self.finite_count

    def quantile(self, q: float, default=None):
        if self.finite_count == 0 or not self._values:
            return default
        values = torch.cat(self._values)
        return float(torch.quantile(values, float(q)).item())

    def summary(self):
        result = {
            "count": int(self.count),
            "finite_count": int(self.finite_count),
        }
        if self.finite_count == 0:
            return result
        mean = self.sum / self.finite_count
        variance = max((self.sum_sq / self.finite_count) - mean**2, 0.0)
        result.update({
            "mean": float(mean),
            "min": float(self.min),
            "max": float(self.max),
            "std": float(math.sqrt(variance)),
        })
        return result


def _make_optional_eval_field_candidates():
    candidates = {
        "governor_alpha": ["governor_alpha", ("next", "governor_alpha")],
        "governor_v_corr": ["governor_v_corr", ("next", "governor_v_corr")],
        "governor_v_corr_z": ["governor_v_corr_z", ("next", "governor_v_corr_z")],
        "governor_v_cmd_b": ["governor_v_cmd_b", ("next", "governor_v_cmd_b")],
        "governor_v_cmd_b_z": ["governor_v_cmd_b_z", ("next", "governor_v_cmd_b_z")],
        "governor_v_gov_b": ["governor_v_gov_b", ("next", "governor_v_gov_b")],
        "governor_v_gov_b_z": ["governor_v_gov_b_z", ("next", "governor_v_gov_b_z")],
        "governor_v_final_b": ["governor_v_final_b", ("next", "governor_v_final_b")],
        "governor_v_final_b_z": ["governor_v_final_b_z", ("next", "governor_v_final_b_z")],
        "actual_velocity_b": [
            ("info", "actual_velocity_b"),
            ("next", "info", "actual_velocity_b"),
        ],
        "r5e1_controller_command_w": [
            ("info", "r5e1_controller_command_w"),
            ("next", "info", "r5e1_controller_command_w"),
        ],
        "r5e1_actual_velocity_w": [
            ("info", "r5e1_actual_velocity_w"),
            ("next", "info", "r5e1_actual_velocity_w"),
        ],
        "drone_state": [
            ("info", "drone_state"),
            ("next", "info", "drone_state"),
        ],
        "min_clearance": [
            ("info", "min_clearance"),
            ("next", "info", "min_clearance"),
            ("info", "safety_min_clearance"),
            ("next", "info", "safety_min_clearance"),
        ],
        "tracking_actual_error_sq": [
            ("info", "tracking_actual_error_sq"),
            ("next", "info", "tracking_actual_error_sq"),
        ],
        "tracking_proxy_error_sq": [
            ("info", "tracking_proxy_error_sq"),
            ("next", "info", "tracking_proxy_error_sq"),
        ],
        "command_preservation_ratio": [
            ("info", "command_preservation_ratio"),
            ("next", "info", "command_preservation_ratio"),
        ],
        "null_command_speed": [
            ("info", "null_command_speed"),
            ("next", "info", "null_command_speed"),
        ],
        "null_command_output_speed": [
            ("info", "null_command_output_speed"),
            ("next", "info", "null_command_output_speed"),
        ],
        "command_amplification": [
            ("info", "command_amplification"),
            ("next", "info", "command_amplification"),
        ],
        "command_amplification_active": [
            ("info", "command_amplification_active"),
            ("next", "info", "command_amplification_active"),
        ],
        "command_amplification_horizontal": [
            ("info", "command_amplification_horizontal"),
            ("next", "info", "command_amplification_horizontal"),
        ],
        "command_amplification_horizontal_active": [
            ("info", "command_amplification_horizontal_active"),
            ("next", "info", "command_amplification_horizontal_active"),
        ],
        "command_amplification_vertical": [
            ("info", "command_amplification_vertical"),
            ("next", "info", "command_amplification_vertical"),
        ],
        "command_amplification_vertical_active": [
            ("info", "command_amplification_vertical_active"),
            ("next", "info", "command_amplification_vertical_active"),
        ],
        "height_world_z": [
            ("info", "height_world_z"),
            ("next", "info", "height_world_z"),
        ],
        "height_floor_violation": [
            ("info", "height_floor_violation"),
            ("next", "info", "height_floor_violation"),
        ],
        "height_ceiling_violation": [
            ("info", "height_ceiling_violation"),
            ("next", "info", "height_ceiling_violation"),
        ],
        "height_ceiling_margin": [
            ("info", "height_ceiling_margin"),
            ("next", "info", "height_ceiling_margin"),
        ],
        "v_cmd_z": [
            ("info", "v_cmd_z"),
            ("next", "info", "v_cmd_z"),
        ],
        "v_final_b_z": [
            ("info", "v_final_b_z"),
            ("next", "info", "v_final_b_z"),
        ],
        "command_mode_code": [
            ("info", "command_mode_code"),
            ("next", "info", "command_mode_code"),
        ],
        "station_keeping_drift": [
            ("info", "station_keeping_drift"),
            ("next", "info", "station_keeping_drift"),
        ],
        "anchor_active": [("info", "anchor_active"), ("next", "info", "anchor_active")],
        "anchor_valid_fraction": [
            ("info", "anchor_valid_fraction"),
            ("next", "info", "anchor_valid_fraction"),
        ],
        "anchor_error_mean": [
            ("info", "anchor_error_mean"),
            ("next", "info", "anchor_error_mean"),
        ],
        "anchor_error_max": [
            ("info", "anchor_error_max"),
            ("next", "info", "anchor_error_max"),
        ],
        "anchor_loss": [("info", "anchor_loss"), ("next", "info", "anchor_loss")],
        "safety_min_clearance": [
            ("info", "safety_min_clearance"),
            ("next", "info", "safety_min_clearance"),
        ],
        "safety_collision": [
            ("info", "safety_collision"),
            ("next", "info", "safety_collision"),
        ],
        "ics_beta": [("info", "ics_beta"), ("next", "info", "ics_beta")],
        "ics_active_beam_count": [
            ("info", "ics_active_beam_count"),
            ("next", "info", "ics_active_beam_count"),
        ],
        "ics_min_clearance": [
            ("info", "ics_min_clearance"),
            ("next", "info", "ics_min_clearance"),
        ],
        "ics_worst_beam_index": [
            ("info", "ics_worst_beam_index"),
            ("next", "info", "ics_worst_beam_index"),
        ],
        "ics_intervention": [
            ("info", "ics_intervention"),
            ("next", "info", "ics_intervention"),
        ],
        "ics_emergency": [("info", "ics_emergency"), ("next", "info", "ics_emergency")],
        "ics_violation": [("info", "ics_violation"), ("next", "info", "ics_violation")],
        "ics_command_speed": [
            ("info", "ics_command_speed"),
            ("next", "info", "ics_command_speed"),
        ],
        "ics_final_speed": [
            ("info", "ics_final_speed"),
            ("next", "info", "ics_final_speed"),
        ],
        "ics_downward_active": [
            ("info", "ics_downward_active"),
            ("next", "info", "ics_downward_active"),
        ],
        "ics_downward_has_ray": [
            ("info", "ics_downward_has_ray"),
            ("next", "info", "ics_downward_has_ray"),
        ],
        "ics_downward_beta": [
            ("info", "ics_downward_beta"),
            ("next", "info", "ics_downward_beta"),
        ],
        "ics_downward_min_clearance": [
            ("info", "ics_downward_min_clearance"),
            ("next", "info", "ics_downward_min_clearance"),
        ],
        "ics_downward_pre_z": [
            ("info", "ics_downward_pre_z"),
            ("next", "info", "ics_downward_pre_z"),
        ],
        "ics_downward_post_z": [
            ("info", "ics_downward_post_z"),
            ("next", "info", "ics_downward_post_z"),
        ],
        "ics_downward_z_delta_abs": [
            ("info", "ics_downward_z_delta_abs"),
            ("next", "info", "ics_downward_z_delta_abs"),
        ],
        "ics_downward_attenuation_ratio": [
            ("info", "ics_downward_attenuation_ratio"),
            ("next", "info", "ics_downward_attenuation_ratio"),
        ],
        "ics_residual_preemption_trigger": [
            ("info", "ics_residual_preemption_trigger"),
            ("next", "info", "ics_residual_preemption_trigger"),
        ],
        "ics_residual_preemption_range_rate_available": [
            ("info", "ics_residual_preemption_range_rate_available"),
            ("next", "info", "ics_residual_preemption_range_rate_available"),
        ],
        "observability_valid_fraction": [
            ("info", "observability_valid_fraction"),
            ("next", "info", "observability_valid_fraction"),
        ],
        "observability_weighted_valid_fraction": [
            ("info", "observability_weighted_valid_fraction"),
            ("next", "info", "observability_weighted_valid_fraction"),
        ],
        "observability_rank": [
            ("info", "observability_rank"),
            ("next", "info", "observability_rank"),
        ],
        "observability_sigma_min": [
            ("info", "observability_sigma_min"),
            ("next", "info", "observability_sigma_min"),
        ],
        "observability_sigma_max": [
            ("info", "observability_sigma_max"),
            ("next", "info", "observability_sigma_max"),
        ],
        "observability_condition_number": [
            ("info", "observability_condition_number"),
            ("next", "info", "observability_condition_number"),
        ],
        "observability_score": [
            ("info", "observability_score"),
            ("next", "info", "observability_score"),
        ],
        "observability_drift_projection": [
            ("info", "observability_drift_projection"),
            ("next", "info", "observability_drift_projection"),
        ],
        "observability_drift_norm": [
            ("info", "observability_drift_norm"),
            ("next", "info", "observability_drift_norm"),
        ],
        "observability_is_proxy": [
            ("info", "observability_is_proxy"),
            ("next", "info", "observability_is_proxy"),
        ],
        "observability_mode_code": [
            ("info", "observability_mode_code"),
            ("next", "info", "observability_mode_code"),
        ],
        "observability_scenario_id": [
            ("info", "observability_scenario_id"),
            ("next", "info", "observability_scenario_id"),
        ],
    }
    candidates.update(_r5e2_optional_field_candidates())
    candidates.update(_r5e3_optional_field_candidates())
    return candidates


def _categorical_fractions(accumulator: _TensorSummaryAccumulator, labels):
    if accumulator.finite_count == 0 or not accumulator._values:
        return {}
    values = torch.cat(accumulator._values).round().long()
    total = max(int(values.numel()), 1)
    return {
        label: float((values == int(code)).sum().item() / total)
        for code, label in labels.items()
    }


def _governor_v_corr_limit(cfg) -> float:
    gov_cfg = getattr(getattr(getattr(cfg, "algo", None), "instinctRL", None), "governor", None)
    return float(getattr(gov_cfg, "v_corr_limit", 0.0))


def _masked_sum_mean(
    numerator: _TensorSummaryAccumulator,
    denominator: _TensorSummaryAccumulator,
):
    if denominator.sum <= 0.0:
        return None
    return numerator.sum / denominator.sum


def _add_r5e2_step_handbook_summary(summary, accumulators):
    if accumulators["r5e2_collision"].count == 0:
        return
    for field_name, handbook_name in (
        ("r5e2_collision", "r5e2_collision_rate"),
        ("r5e2_terminated_collision", "r5e2_terminated_collision_step_rate"),
        ("r5e2_below_bound_adjacent", "r5e2_below_bound_adjacent_rate"),
        ("r5e2_ceiling_adjacent", "r5e2_ceiling_adjacent_rate"),
        ("r5e2_height_adjacent", "r5e2_height_adjacent_rate"),
        ("r5e2_min_clearance_source_available", "r5e2_min_clearance_source_available_rate"),
        ("r5e2_missing_clearance_source", "r5e2_missing_clearance_source_rate"),
        ("r5e2_lidar_collision_evidence", "r5e2_lidar_collision_evidence_rate"),
        ("r5e2_contact_telemetry_available", "r5e2_contact_telemetry_available_rate"),
        ("r5e2_missing_contact_telemetry", "r5e2_missing_contact_telemetry_rate"),
        ("r5e2_collision_termination_same_step", "r5e2_collision_termination_same_step_rate"),
        ("r5e2_collision_without_termination", "r5e2_collision_without_termination_rate"),
        (
            "r5e2_termination_collision_without_collision",
            "r5e2_termination_collision_without_collision_rate",
        ),
    ):
        value = accumulators[field_name].mean()
        if value is not None:
            summary[f"eval/handbook.{handbook_name}"] = float(value)

    clearance_p05 = accumulators["r5e2_min_clearance"].quantile(0.05)
    if clearance_p05 is not None:
        summary["eval/handbook.r5e2_min_clearance_p05"] = float(clearance_p05)
    root_z_min = accumulators["r5e2_root_z"].summary().get("min")
    if root_z_min is not None:
        summary["eval/handbook.r5e2_root_z_min"] = float(root_z_min)


_R5E3_STEP_P05_FIELDS = (
    "r5e3_raw_min_clearance",
    "r5e3_ics_min_clearance",
    "r5e3_required_stop_distance_conservative",
    "r5e3_residual_to_collision_threshold",
    "r5e3_residual_to_emergency",
    "r5e3_residual_to_d_safe",
    "r5e3_worst_beam_required_stop_distance",
    "r5e3_worst_beam_residual_to_collision_threshold",
    "r5e3_worst_beam_residual_to_emergency",
    "r5e3_worst_beam_residual_to_d_safe",
)

_R5E3_STEP_MEAN_FIELDS = (
    "r5e3_collision_clearance_threshold",
    "r5e3_emergency_clearance",
    "r5e3_d_safe",
    "r5e3_a_max",
    "r5e3_latency_sec",
    "r5e3_command_eps",
    "r5e3_low_beta_threshold",
    "r5e3_actual_body_speed_xy",
    "r5e3_actual_body_speed_z_abs",
    "r5e3_actual_body_speed_norm",
    "r5e3_v_final_body_speed_xy",
    "r5e3_v_final_body_speed_z_abs",
    "r5e3_v_final_body_speed_norm",
) + _R5E3_STEP_P05_FIELDS

_R5E3_STEP_RATE_FIELDS = (
    "r5e3_collision_margin_exhausted",
    "r5e3_emergency_margin_exhausted",
    "r5e3_d_safe_margin_exhausted",
    "r5e3_low_beta",
    "r5e3_full_stop_commanded",
    "r5e3_full_stop_after_collision_margin_exhausted",
    "r5e3_full_stop_after_emergency_margin_exhausted",
    "r5e3_worst_beam_collision_margin_exhausted",
    "r5e3_worst_beam_emergency_margin_exhausted",
    "r5e3_worst_beam_d_safe_margin_exhausted",
    "r5e3_worst_beam_full_stop_after_collision_margin_exhausted",
    "r5e3_worst_beam_full_stop_after_emergency_margin_exhausted",
    "r5e3_raw_min_clearance_source_available",
    "r5e3_ics_min_clearance_source_available",
    "r5e3_contact_telemetry_available",
    "r5e3_body_telemetry_available",
    "r5e3_surface_normal_available",
    "r5e3_measured_deceleration_available",
    "r5e3_worst_ics_beam_source_available",
    "r5e3_missing_clearance_source",
    "r5e3_missing_contact_telemetry",
    "r5e3_missing_body_telemetry",
    "r5e3_missing_contact_body_telemetry",
    "r5e3_missing_surface_normal",
    "r5e3_missing_measured_deceleration",
    "r5e3_missing_worst_ics_beam",
    "r5e3_conservative_approximation_used",
)

_R5E3_HANDBOOK_CONTRACT_KEYS = (
    "eval/handbook.r5e3_residual_to_collision_threshold_mean",
    "eval/handbook.r5e3_residual_to_collision_threshold_p05",
    "eval/handbook.r5e3_low_beta_rate",
    "eval/handbook.r5e3_missing_contact_telemetry_rate",
    "eval/handbook.r5e3_full_stop_after_collision_margin_exhausted_rate",
    "eval/handbook.r5e3_collision_window{window}_steps",
    "eval/handbook.r5e3_collision_window{window}_{field_name}_{suffix}",
    "eval/handbook.r5e3_low_beta_window{window}_steps",
)


def _add_r5e3_step_handbook_summary(summary, accumulators):
    if accumulators["r5e3_required_stop_distance_conservative"].count == 0:
        return
    for field_name in _R5E3_STEP_MEAN_FIELDS:
        value = accumulators[field_name].mean()
        if value is not None:
            summary[f"eval/handbook.{field_name}_mean"] = float(value)
    for field_name in _R5E3_STEP_P05_FIELDS:
        value = accumulators[field_name].quantile(0.05)
        if value is not None:
            summary[f"eval/handbook.{field_name}_p05"] = float(value)
    for field_name in _R5E3_STEP_RATE_FIELDS:
        value = accumulators[field_name].mean()
        if value is not None:
            summary[f"eval/handbook.{field_name}_rate"] = float(value)


_R5E_DIAGNOSTIC_FIELDS = (
    "r5e_null_command",
    "r5e_null_actual_speed_xy",
    "r5e_null_actual_speed_z_abs",
    "r5e_null_output_speed_xy",
    "r5e_null_output_speed_z_abs",
    "r5e_command_active",
    "r5e_command_preservation_pre_ics",
    "r5e_command_preservation_post_ics",
    "r5e_command_preservation_ics_loss",
    "r5e_command_horizontal_active",
    "r5e_command_preservation_horizontal",
    "r5e_command_vertical_active",
    "r5e_command_preservation_vertical_abs",
    "r5e_near_floor",
    "r5e_near_floor_v_cmd_z",
    "r5e_near_floor_v_gov_z",
    "r5e_near_floor_v_final_z",
    "r5e_near_floor_ics_beta",
    "r5e_near_floor_clearance",
    "r5e_ics_violation_near_floor",
)


def _make_r5e_diagnostic_accumulators():
    return {
        field_name: _TensorSummaryAccumulator()
        for field_name in _R5E_DIAGNOSTIC_FIELDS
    }


def _r5e_eval_config(cfg):
    instinct_cfg = getattr(cfg, "instinctRL", None) if cfg is not None else None
    reward_cfg = getattr(instinct_cfg, "reward", None)
    ics_cfg = getattr(instinct_cfg, "ics", None)
    command_eps = float(getattr(reward_cfg, "command_eps", 1e-3))
    height_floor = float(getattr(reward_cfg, "height_floor", 0.5))
    d_safe = float(getattr(ics_cfg, "d_safe", 0.8))
    return command_eps, height_floor, d_safe


_R5E1_CURRENT_FIELDS = (
    "r5e1_v_final_body_speed_xy",
    "r5e1_v_final_body_speed_z_abs",
    "r5e1_controller_command_world_speed_xy",
    "r5e1_controller_command_world_speed_z_abs",
    "r5e1_actual_body_speed_xy",
    "r5e1_actual_body_speed_z_abs",
    "r5e1_actual_world_speed_xy",
    "r5e1_actual_world_speed_z_abs",
    "r5e1_command_actual_body_mismatch_xy",
    "r5e1_command_actual_body_mismatch_z_abs",
    "r5e1_command_actual_world_mismatch_xy",
    "r5e1_command_actual_world_mismatch_z_abs",
    "r5e1_command_actual_body_alignment_xy",
    "r5e1_command_actual_body_alignment_xy_active",
    "r5e1_command_actual_world_alignment_xy",
    "r5e1_command_actual_world_alignment_xy_active",
    "r5e1_prev_action_available",
    "r5e1_prev_action_v_final_mismatch_xy",
    "r5e1_prev_action_v_final_mismatch_z_abs",
)

_R5E1_LAG_FIELDS = (
    "r5e1_lag0_command_actual_world_mismatch_xy",
    "r5e1_lag0_command_actual_world_mismatch_z_abs",
) + tuple(
    field
    for lag in R5E1_LAG_STEPS
    for field in (
        f"r5e1_lag{lag}_available",
        f"r5e1_lag{lag}_command_actual_world_mismatch_xy",
        f"r5e1_lag{lag}_command_actual_world_mismatch_z_abs",
    )
) + (
    "r5e1_lag_best_command_actual_world_mismatch_xy",
    "r5e1_lag_best_command_actual_world_mismatch_z_abs",
    "r5e1_lag_best_step_xy",
    "r5e1_lag_best_step_z_abs",
    "r5e1_lag_best_improvement_xy",
    "r5e1_lag_best_improvement_z_abs",
)

_R5E1_ALIGNMENT_DENOMINATORS = {
    "r5e1_command_actual_body_alignment_xy": "r5e1_command_actual_body_alignment_xy_active",
    "r5e1_command_actual_world_alignment_xy": "r5e1_command_actual_world_alignment_xy_active",
}


def _make_r5e1_diagnostic_accumulators():
    return {
        field_name: _TensorSummaryAccumulator()
        for field_name in _R5E1_CURRENT_FIELDS
    }


def _actual_velocity_w_from_tensordict(tensordict, optional_candidates):
    actual_w = _get_optional_tensor(
        tensordict,
        optional_candidates.get("r5e1_actual_velocity_w", []),
    )
    if actual_w is not None:
        return actual_w
    drone_state = _get_optional_tensor(
        tensordict,
        optional_candidates.get("drone_state", [])
        + [("info", "drone_state"), ("next", "info", "drone_state")],
    )
    if drone_state is None:
        return None
    return drone_state[..., 7:10]


def _accumulate_r5e1_metrics_from_tensordict(
    accumulators,
    tensordict,
    optional_candidates,
    cfg,
):
    v_final_b = _get_optional_tensor(
        tensordict,
        optional_candidates.get("governor_v_final_b", []),
    )
    if v_final_b is None:
        v_final_b = _get_optional_tensor(
            tensordict,
            optional_candidates.get("governor_v_gov_b", []),
        )
    controller_command_w = _get_optional_tensor(
        tensordict,
        optional_candidates.get("r5e1_controller_command_w", []),
    )
    actual_velocity_b = _get_optional_tensor(
        tensordict,
        optional_candidates.get("actual_velocity_b", []),
    )
    actual_velocity_w = _actual_velocity_w_from_tensordict(
        tensordict,
        optional_candidates,
    )
    if any(
        value is None
        for value in (
            v_final_b,
            controller_command_w,
            actual_velocity_b,
            actual_velocity_w,
        )
    ):
        return False

    state_vec = _get_optional_tensor(
        tensordict,
        [
            ("agents", "observation", "state_vec"),
            ("next", "agents", "observation", "state_vec"),
        ],
    )
    command_eps, _, _ = _r5e_eval_config(cfg)
    metrics = compute_r5e1_controller_latency_step_metrics(
        v_final_b=v_final_b,
        controller_command_w=controller_command_w,
        actual_velocity_b=actual_velocity_b,
        actual_velocity_w=actual_velocity_w,
        prev_action_b=_latest_prev_action_from_state_vec(state_vec),
        command_eps=command_eps,
    )
    for field_name, value in metrics.items():
        accumulators[field_name].add(value)
    return True


def _add_r5e1_diagnostic_summaries(summary, accumulators):
    for field_name, accumulator in accumulators.items():
        if accumulator.count > 0:
            summary[f"eval/diagnostics.{field_name}"] = accumulator.summary()


def _r5e1_handbook_suffix(field_name):
    if field_name.endswith("_available") or field_name.endswith("_active"):
        return "rate"
    return "mean"


def _add_r5e1_field_summary(summary, accumulators, field_name, key_prefix):
    if field_name in _R5E1_ALIGNMENT_DENOMINATORS:
        value = _masked_sum_mean(
            accumulators[field_name],
            accumulators[_R5E1_ALIGNMENT_DENOMINATORS[field_name]],
        )
    else:
        value = accumulators[field_name].mean()
    suffix = field_name[len("r5e1_"):]
    summary[f"{key_prefix}{suffix}_{_r5e1_handbook_suffix(field_name)}"] = (
        float(value) if value is not None else None
    )


def _add_r5e1_handbook_summary(summary, accumulators):
    if accumulators["r5e1_v_final_body_speed_xy"].count == 0:
        return
    for field_name in _R5E1_CURRENT_FIELDS:
        _add_r5e1_field_summary(
            summary,
            accumulators,
            field_name,
            "eval/handbook.r5e1_",
        )


def _accumulate_r5e_metrics_from_tensordict(
    accumulators,
    tensordict,
    optional_candidates,
    cfg,
):
    v_cmd_b = _get_optional_tensor(
        tensordict,
        optional_candidates.get("governor_v_cmd_b", [])
        + [("info", "v_cmd"), ("next", "info", "v_cmd")],
    )
    actual_velocity_b = _get_optional_tensor(
        tensordict,
        optional_candidates.get("actual_velocity_b", []),
    )
    v_gov_b = _get_optional_tensor(
        tensordict,
        optional_candidates.get("governor_v_gov_b", []),
    )
    v_final_b = _get_optional_tensor(
        tensordict,
        optional_candidates.get("governor_v_final_b", []),
    )
    if v_final_b is None:
        v_final_b = v_gov_b
    height_world_z = _get_optional_tensor(
        tensordict,
        optional_candidates.get("height_world_z", []),
    )
    min_clearance = _get_optional_tensor(
        tensordict,
        optional_candidates.get("min_clearance", []),
    )
    ics_beta = _get_optional_tensor(
        tensordict,
        optional_candidates.get("ics_beta", []),
    )
    ics_emergency = _get_optional_tensor(
        tensordict,
        optional_candidates.get("ics_emergency", []),
    )
    if any(
        value is None
        for value in (
            v_cmd_b,
            actual_velocity_b,
            v_gov_b,
            v_final_b,
            height_world_z,
            min_clearance,
        )
    ):
        return False

    command_eps, height_floor, d_safe = _r5e_eval_config(cfg)
    metrics = compute_r5e_mechanism_step_metrics(
        v_cmd_b=v_cmd_b,
        actual_velocity_b=actual_velocity_b,
        v_gov_b=v_gov_b,
        v_final_b=v_final_b,
        height_world_z=height_world_z,
        min_clearance=min_clearance,
        ics_beta=ics_beta,
        ics_emergency=ics_emergency,
        d_safe=d_safe,
        height_floor=height_floor,
        command_eps=command_eps,
    )
    for field_name, value in metrics.items():
        accumulators[field_name].add(value)
    return True


def _add_r5e_diagnostic_summaries(summary, accumulators):
    for field_name, accumulator in accumulators.items():
        if accumulator.count > 0:
            summary[f"eval/diagnostics.{field_name}"] = accumulator.summary()


def _add_r5e_handbook_summary(summary, accumulators):
    if accumulators["r5e_null_command"].count == 0:
        return
    for numerator_name, denominator_name, handbook_key in (
        (
            "r5e_null_actual_speed_xy",
            "r5e_null_command",
            "eval/handbook.r5e_null_actual_speed_xy_mean",
        ),
        (
            "r5e_null_actual_speed_z_abs",
            "r5e_null_command",
            "eval/handbook.r5e_null_actual_speed_z_abs_mean",
        ),
        (
            "r5e_null_output_speed_xy",
            "r5e_null_command",
            "eval/handbook.r5e_null_output_speed_xy_mean",
        ),
        (
            "r5e_null_output_speed_z_abs",
            "r5e_null_command",
            "eval/handbook.r5e_null_output_speed_z_abs_mean",
        ),
        (
            "r5e_command_preservation_pre_ics",
            "r5e_command_active",
            "eval/handbook.r5e_command_preservation_pre_ics_ratio",
        ),
        (
            "r5e_command_preservation_post_ics",
            "r5e_command_active",
            "eval/handbook.r5e_command_preservation_post_ics_ratio",
        ),
        (
            "r5e_command_preservation_ics_loss",
            "r5e_command_active",
            "eval/handbook.r5e_command_preservation_ics_loss_ratio",
        ),
        (
            "r5e_command_preservation_horizontal",
            "r5e_command_horizontal_active",
            "eval/handbook.r5e_command_preservation_horizontal_ratio",
        ),
        (
            "r5e_command_preservation_vertical_abs",
            "r5e_command_vertical_active",
            "eval/handbook.r5e_command_preservation_vertical_abs_ratio",
        ),
        (
            "r5e_near_floor_v_cmd_z",
            "r5e_near_floor",
            "eval/handbook.r5e_near_floor_v_cmd_z_mean",
        ),
        (
            "r5e_near_floor_v_gov_z",
            "r5e_near_floor",
            "eval/handbook.r5e_near_floor_v_gov_z_mean",
        ),
        (
            "r5e_near_floor_v_final_z",
            "r5e_near_floor",
            "eval/handbook.r5e_near_floor_v_final_z_mean",
        ),
        (
            "r5e_near_floor_ics_beta",
            "r5e_near_floor",
            "eval/handbook.r5e_near_floor_ics_beta_mean",
        ),
        (
            "r5e_ics_violation_near_floor",
            "r5e_near_floor",
            "eval/handbook.r5e_ics_violation_near_floor_rate",
        ),
    ):
        value = _masked_sum_mean(
            accumulators[numerator_name],
            accumulators[denominator_name],
        )
        summary[handbook_key] = float(value) if value is not None else None

    near_floor_rate = accumulators["r5e_near_floor"].mean()
    summary["eval/handbook.r5e_near_floor_rate"] = (
        float(near_floor_rate) if near_floor_rate is not None else None
    )
    near_floor_clearance_p05 = accumulators["r5e_near_floor_clearance"].quantile(0.05)
    summary["eval/handbook.r5e_near_floor_clearance_p05"] = (
        float(near_floor_clearance_p05)
        if near_floor_clearance_p05 is not None
        else None
    )


_R5G_STATION_FIELDS = (
    "r5g_station_null_command",
    "r5g_station_null_actual_speed_xy",
    "r5g_station_null_output_speed_xy",
    "r5g_station_null_mismatch_xy",
    "r5g_station_null_mismatch_z_abs",
    "r5g_station_null_alignment_xy",
    "r5g_station_null_alignment_xy_active",
    "r5g_station_null_actual_output_xy_ratio",
    "r5g_station_null_output_xy_active",
    "r5g_anchor_active",
    "r5g_anchor_station_drift_when_active",
    "r5g_anchor_error_when_active",
    "r5g_anchor_loss_when_active",
    "r5g_anchor_valid",
    "r5g_anchor_station_drift_when_valid",
    "r5g_anchor_error_when_valid",
    "r5g_anchor_loss_when_valid",
    "r5g_anchor_invalid",
    "r5g_anchor_station_drift_when_invalid",
    "r5g_anchor_error_when_invalid",
    "r5g_anchor_loss_when_invalid",
    "r5g_anchor_high_loss",
    "r5g_anchor_station_drift_when_high_loss",
    "r5g_anchor_error_when_high_loss",
    "r5g_anchor_loss_when_high_loss",
    "r5g_anchor_obs_valid",
    "r5g_anchor_station_drift_when_obs_valid",
    "r5g_anchor_error_when_obs_valid",
    "r5g_anchor_loss_when_obs_valid",
    "r5g_anchor_obs_poor",
    "r5g_anchor_station_drift_when_obs_poor",
    "r5g_anchor_error_when_obs_poor",
    "r5g_anchor_loss_when_obs_poor",
)

_R5G_DOWNWARD_FIELDS = (
    "r5g_downward_active",
    "r5g_downward_has_ray",
    "r5g_downward_beta_when_active",
    "r5g_downward_min_clearance_when_active",
    "r5g_downward_pre_z_when_active",
    "r5g_downward_post_z_when_active",
    "r5g_downward_z_delta_abs_when_active",
    "r5g_downward_attenuation_ratio_when_active",
)


def _make_r5g_station_accumulators():
    return {field_name: _TensorSummaryAccumulator() for field_name in _R5G_STATION_FIELDS}


def _make_r5g_downward_accumulators():
    return {field_name: _TensorSummaryAccumulator() for field_name in _R5G_DOWNWARD_FIELDS}


def _r5g_eval_config(cfg):
    instinct_cfg = getattr(cfg, "instinctRL", None) if cfg is not None else None
    reward_cfg = getattr(instinct_cfg, "reward", None)
    observability_cfg = getattr(instinct_cfg, "observability", None)
    command_eps = float(getattr(reward_cfg, "command_eps", 1e-3))
    min_anchor_valid_fraction = float(
        getattr(reward_cfg, "min_anchor_valid_fraction", 0.1)
    )
    anchor_loss_high_threshold = float(
        getattr(reward_cfg, "null_output_anchor_loss_threshold", 0.05)
    )
    observability_min_valid_fraction = float(
        getattr(observability_cfg, "min_valid_fraction", 0.01)
    )
    return (
        command_eps,
        min_anchor_valid_fraction,
        anchor_loss_high_threshold,
        observability_min_valid_fraction,
    )


def _accumulate_r5g_station_metrics_from_tensordict(
    accumulators,
    tensordict,
    optional_candidates,
    cfg,
):
    v_cmd_b = _get_optional_tensor(
        tensordict,
        optional_candidates.get("governor_v_cmd_b", [])
        + [("info", "v_cmd"), ("next", "info", "v_cmd")],
    )
    actual_velocity_b = _get_optional_tensor(
        tensordict,
        optional_candidates.get("actual_velocity_b", []),
    )
    v_final_b = _get_optional_tensor(
        tensordict,
        optional_candidates.get("governor_v_final_b", []),
    )
    station_drift = _get_optional_tensor(
        tensordict,
        optional_candidates.get("station_keeping_drift", []),
    )
    if any(value is None for value in (v_cmd_b, actual_velocity_b, v_final_b, station_drift)):
        return False

    (
        command_eps,
        min_anchor_valid_fraction,
        anchor_loss_high_threshold,
        observability_min_valid_fraction,
    ) = _r5g_eval_config(cfg)
    metrics = compute_r5g_station_anchor_step_metrics(
        v_cmd_b=v_cmd_b,
        actual_velocity_b=actual_velocity_b,
        v_final_b=v_final_b,
        station_drift=station_drift,
        anchor_active=_get_optional_tensor(
            tensordict,
            optional_candidates.get("anchor_active", []),
        ),
        anchor_valid_fraction=_get_optional_tensor(
            tensordict,
            optional_candidates.get("anchor_valid_fraction", []),
        ),
        anchor_error_mean=_get_optional_tensor(
            tensordict,
            optional_candidates.get("anchor_error_mean", []),
        ),
        anchor_loss=_get_optional_tensor(
            tensordict,
            optional_candidates.get("anchor_loss", []),
        ),
        observability_valid_fraction=_get_optional_tensor(
            tensordict,
            optional_candidates.get("observability_valid_fraction", []),
        ),
        command_eps=command_eps,
        min_anchor_valid_fraction=min_anchor_valid_fraction,
        anchor_loss_high_threshold=anchor_loss_high_threshold,
        observability_min_valid_fraction=observability_min_valid_fraction,
    )
    for field_name, value in metrics.items():
        accumulators[field_name].add(value)
    return True


def _accumulate_r5g_downward_metrics_from_tensordict(
    accumulators,
    tensordict,
    optional_candidates,
):
    active = _get_optional_tensor(
        tensordict,
        optional_candidates.get("ics_downward_active", []),
    )
    if active is None:
        return False
    metrics = compute_r5g_downward_step_metrics(
        downward_active=active,
        downward_has_ray=_get_optional_tensor(
            tensordict,
            optional_candidates.get("ics_downward_has_ray", []),
        ),
        downward_beta=_get_optional_tensor(
            tensordict,
            optional_candidates.get("ics_downward_beta", []),
        ),
        downward_min_clearance=_get_optional_tensor(
            tensordict,
            optional_candidates.get("ics_downward_min_clearance", []),
        ),
        downward_pre_z=_get_optional_tensor(
            tensordict,
            optional_candidates.get("ics_downward_pre_z", []),
        ),
        downward_post_z=_get_optional_tensor(
            tensordict,
            optional_candidates.get("ics_downward_post_z", []),
        ),
        downward_z_delta_abs=_get_optional_tensor(
            tensordict,
            optional_candidates.get("ics_downward_z_delta_abs", []),
        ),
        downward_attenuation_ratio=_get_optional_tensor(
            tensordict,
            optional_candidates.get("ics_downward_attenuation_ratio", []),
        ),
    )
    for field_name, value in metrics.items():
        accumulators[field_name].add(value)
    return True


def _add_r5g_diagnostic_summaries(summary, station_accumulators, downward_accumulators):
    for field_name, accumulator in station_accumulators.items():
        if accumulator.count > 0:
            summary[f"eval/diagnostics.{field_name}"] = accumulator.summary()
    for field_name, accumulator in downward_accumulators.items():
        if accumulator.count > 0:
            summary[f"eval/diagnostics.{field_name}"] = accumulator.summary()


def _add_r5g_handbook_summary(summary, station_accumulators, downward_accumulators):
    null_count = station_accumulators["r5g_station_null_command"]
    for numerator_name, denominator_name, handbook_key in (
        (
            "r5g_station_null_actual_speed_xy",
            "r5g_station_null_command",
            "eval/handbook.r5g_station_null_actual_speed_xy_mean",
        ),
        (
            "r5g_station_null_output_speed_xy",
            "r5g_station_null_command",
            "eval/handbook.r5g_station_null_output_speed_xy_mean",
        ),
        (
            "r5g_station_null_mismatch_xy",
            "r5g_station_null_command",
            "eval/handbook.r5g_station_null_mismatch_xy_mean",
        ),
        (
            "r5g_station_null_mismatch_z_abs",
            "r5g_station_null_command",
            "eval/handbook.r5g_station_null_mismatch_z_abs_mean",
        ),
        (
            "r5g_station_null_alignment_xy",
            "r5g_station_null_alignment_xy_active",
            "eval/handbook.r5g_station_null_alignment_xy_mean",
        ),
        (
            "r5g_station_null_actual_output_xy_ratio",
            "r5g_station_null_output_xy_active",
            "eval/handbook.r5g_station_null_actual_output_xy_ratio_mean",
        ),
    ):
        value = _masked_sum_mean(
            station_accumulators[numerator_name],
            station_accumulators[denominator_name],
        )
        if null_count.count > 0:
            summary[handbook_key] = float(value) if value is not None else None

    for condition in ("active", "valid", "invalid", "high_loss", "obs_valid", "obs_poor"):
        denominator = station_accumulators[f"r5g_anchor_{condition}"]
        for field_name, suffix in (
            ("station_drift", "station_drift_mean"),
            ("error", "error_mean"),
            ("loss", "loss_mean"),
        ):
            value = _masked_sum_mean(
                station_accumulators[f"r5g_anchor_{field_name}_when_{condition}"],
                denominator,
            )
            if denominator.count > 0:
                summary[f"eval/handbook.r5g_anchor_{suffix}_when_{condition}"] = (
                    float(value) if value is not None else None
                )
        rate = denominator.mean()
        if rate is not None:
            summary[f"eval/handbook.r5g_anchor_{condition}_rate"] = float(rate)

    active = downward_accumulators["r5g_downward_active"]
    has_ray = downward_accumulators["r5g_downward_has_ray"]
    active_rate = active.mean()
    has_ray_rate = has_ray.mean()
    if active_rate is not None:
        summary["eval/handbook.r5g_downward_active_rate"] = float(active_rate)
    if has_ray_rate is not None:
        summary["eval/handbook.r5g_downward_has_ray_rate"] = float(has_ray_rate)
    for numerator_name, handbook_key in (
        ("r5g_downward_beta_when_active", "eval/handbook.r5g_downward_beta_mean_when_active"),
        ("r5g_downward_pre_z_when_active", "eval/handbook.r5g_downward_pre_z_mean_when_active"),
        ("r5g_downward_post_z_when_active", "eval/handbook.r5g_downward_post_z_mean_when_active"),
        (
            "r5g_downward_z_delta_abs_when_active",
            "eval/handbook.r5g_downward_z_delta_abs_mean_when_active",
        ),
        (
            "r5g_downward_attenuation_ratio_when_active",
            "eval/handbook.r5g_downward_attenuation_ratio_mean_when_active",
        ),
    ):
        value = _masked_sum_mean(downward_accumulators[numerator_name], active)
        if active.count > 0:
            summary[handbook_key] = float(value) if value is not None else None
    clearance_p05 = downward_accumulators[
        "r5g_downward_min_clearance_when_active"
    ].quantile(0.05)
    if active.count > 0:
        summary["eval/handbook.r5g_downward_min_clearance_p05_when_active"] = (
            float(clearance_p05) if clearance_p05 is not None else None
        )


def _make_r5h_diagnostic_accumulators():
    return {field_name: _TensorSummaryAccumulator() for field_name in R5H_DIAGNOSTIC_FIELDS}


def _latest_prev_action_from_state_vec(state_vec: torch.Tensor):
    if state_vec is None or state_vec.shape[-1] < 13:
        return None
    latest_frame = state_vec[..., -13:]
    return latest_frame[..., 9:12]


def _r5h_eval_config(cfg):
    command_eps, height_floor, d_safe = _r5e_eval_config(cfg)
    (
        _,
        min_anchor_valid_fraction,
        anchor_loss_high_threshold,
        _observability_min_valid_fraction,
    ) = _r5g_eval_config(cfg)
    instinct_cfg = getattr(cfg, "instinctRL", None) if cfg is not None else None
    ics_cfg = getattr(instinct_cfg, "ics", None)
    low_beta_threshold = float(getattr(ics_cfg, "low_beta_threshold", 0.999))
    return (
        command_eps,
        height_floor,
        d_safe,
        low_beta_threshold,
        min_anchor_valid_fraction,
        anchor_loss_high_threshold,
    )


def _accumulate_r5h_metrics_from_tensordict(
    accumulators,
    tensordict,
    optional_candidates,
    cfg,
):
    v_cmd_b = _get_optional_tensor(
        tensordict,
        optional_candidates.get("governor_v_cmd_b", [])
        + [("info", "v_cmd"), ("next", "info", "v_cmd")],
    )
    actual_velocity_b = _get_optional_tensor(
        tensordict,
        optional_candidates.get("actual_velocity_b", []),
    )
    v_gov_b = _get_optional_tensor(
        tensordict,
        optional_candidates.get("governor_v_gov_b", []),
    )
    v_final_b = _get_optional_tensor(
        tensordict,
        optional_candidates.get("governor_v_final_b", []),
    )
    if v_final_b is None:
        v_final_b = v_gov_b
    min_clearance = _get_optional_tensor(
        tensordict,
        optional_candidates.get("min_clearance", []),
    )
    if any(
        value is None
        for value in (
            v_cmd_b,
            actual_velocity_b,
            v_gov_b,
            v_final_b,
            min_clearance,
        )
    ):
        return False

    state_vec = _get_optional_tensor(
        tensordict,
        [
            ("agents", "observation", "state_vec"),
            ("next", "agents", "observation", "state_vec"),
        ],
    )
    (
        command_eps,
        height_floor,
        d_safe,
        low_beta_threshold,
        min_anchor_valid_fraction,
        anchor_loss_high_threshold,
    ) = _r5h_eval_config(cfg)
    metrics = compute_r5h_mechanism_step_metrics(
        v_cmd_b=v_cmd_b,
        actual_velocity_b=actual_velocity_b,
        v_gov_b=v_gov_b,
        v_final_b=v_final_b,
        min_clearance=min_clearance,
        height_world_z=_get_optional_tensor(
            tensordict,
            optional_candidates.get("height_world_z", []),
        ),
        ics_beta=_get_optional_tensor(
            tensordict,
            optional_candidates.get("ics_beta", []),
        ),
        ics_emergency=_get_optional_tensor(
            tensordict,
            optional_candidates.get("ics_emergency", []),
        ),
        ics_violation=_get_optional_tensor(
            tensordict,
            optional_candidates.get("ics_violation", []),
        ),
        ics_active_beam_count=_get_optional_tensor(
            tensordict,
            optional_candidates.get("ics_active_beam_count", []),
        ),
        ics_downward_active=_get_optional_tensor(
            tensordict,
            optional_candidates.get("ics_downward_active", []),
        ),
        ics_downward_beta=_get_optional_tensor(
            tensordict,
            optional_candidates.get("ics_downward_beta", []),
        ),
        ics_downward_min_clearance=_get_optional_tensor(
            tensordict,
            optional_candidates.get("ics_downward_min_clearance", []),
        ),
        collision=_get_optional_tensor(
            tensordict,
            optional_candidates.get("safety_collision", []),
        ),
        governor_alpha=_get_optional_tensor(
            tensordict,
            optional_candidates.get("governor_alpha", []),
        ),
        governor_v_corr=_get_optional_tensor(
            tensordict,
            optional_candidates.get("governor_v_corr", []),
        ),
        prev_action_b=_latest_prev_action_from_state_vec(state_vec),
        station_drift=_get_optional_tensor(
            tensordict,
            optional_candidates.get("station_keeping_drift", []),
        ),
        anchor_active=_get_optional_tensor(
            tensordict,
            optional_candidates.get("anchor_active", []),
        ),
        anchor_valid_fraction=_get_optional_tensor(
            tensordict,
            optional_candidates.get("anchor_valid_fraction", []),
        ),
        anchor_error_mean=_get_optional_tensor(
            tensordict,
            optional_candidates.get("anchor_error_mean", []),
        ),
        anchor_loss=_get_optional_tensor(
            tensordict,
            optional_candidates.get("anchor_loss", []),
        ),
        command_eps=command_eps,
        height_floor=height_floor,
        d_safe=d_safe,
        low_beta_threshold=low_beta_threshold,
        min_anchor_valid_fraction=min_anchor_valid_fraction,
        anchor_loss_high_threshold=anchor_loss_high_threshold,
    )
    for field_name, value in metrics.items():
        accumulators[field_name].add(value)
    return True


def _add_r5h_diagnostic_summaries(summary, accumulators):
    for field_name, accumulator in accumulators.items():
        if accumulator.count > 0:
            summary[f"eval/diagnostics.{field_name}"] = accumulator.summary()


def _add_r5h_handbook_summary(summary, accumulators):
    if accumulators["r5h_collision"].count == 0:
        return

    for condition in R5H_CONDITION_NAMES:
        denominator = accumulators[f"r5h_{condition}"]
        rate = denominator.mean()
        if rate is not None:
            summary[f"eval/handbook.r5h_{condition}_rate"] = float(rate)
        for value_name in R5H_CONCENTRATION_VALUE_NAMES:
            value = _masked_sum_mean(
                accumulators[f"r5h_{value_name}_when_{condition}"],
                denominator,
            )
            summary[f"eval/handbook.r5h_{value_name}_mean_when_{condition}"] = (
                float(value) if value is not None else None
            )
        for value_name in R5H_CONCENTRATION_SAMPLE_VALUE_NAMES:
            p05 = accumulators[f"r5h_{value_name}_sample_when_{condition}"].quantile(0.05)
            summary[f"eval/handbook.r5h_{value_name}_p05_when_{condition}"] = (
                float(p05) if p05 is not None else None
            )

    station_denominator = accumulators["r5h_station_null_command"]
    station_rate = station_denominator.mean()
    if station_rate is not None:
        summary["eval/handbook.r5h_station_null_command_rate"] = float(station_rate)
    for value_name in R5H_STATION_VALUE_NAMES:
        value = _masked_sum_mean(
            accumulators[f"r5h_station_null_{value_name}"],
            station_denominator,
        )
        summary[f"eval/handbook.r5h_station_null_{value_name}_mean"] = (
            float(value) if value is not None else None
        )

    for condition in R5H_ANCHOR_CONDITION_NAMES:
        denominator = accumulators[f"r5h_anchor_{condition}"]
        rate = denominator.mean()
        if rate is not None:
            summary[f"eval/handbook.r5h_anchor_{condition}_rate"] = float(rate)
        for value_name in R5H_ANCHOR_VALUE_NAMES:
            value = _masked_sum_mean(
                accumulators[f"r5h_anchor_{value_name}_when_{condition}"],
                denominator,
            )
            summary[f"eval/handbook.r5h_anchor_{value_name}_mean_when_{condition}"] = (
                float(value) if value is not None else None
            )

    tracking_groups = (
        (
            "r5h_tracking_active",
            (
                "pre_ics_preservation",
                "post_ics_preservation",
                "governor_preservation_loss",
                "post_ics_preservation_loss",
            ),
        ),
        (
            "r5h_tracking_horizontal_active",
            (
                "horizontal_pre_ics_preservation",
                "horizontal_post_ics_preservation",
                "horizontal_governor_preservation_loss",
                "horizontal_post_ics_preservation_loss",
            ),
        ),
        (
            "r5h_tracking_vertical_active",
            (
                "vertical_pre_ics_preservation",
                "vertical_post_ics_preservation",
                "vertical_governor_preservation_loss",
                "vertical_post_ics_preservation_loss",
            ),
        ),
    )
    for denominator_name, value_names in tracking_groups:
        denominator = accumulators[denominator_name]
        rate = denominator.mean()
        if rate is not None:
            summary[f"eval/handbook.{denominator_name}_rate"] = float(rate)
        for value_name in value_names:
            accumulator_name = f"r5h_tracking_{value_name}"
            value = _masked_sum_mean(accumulators[accumulator_name], denominator)
            summary[f"eval/handbook.{accumulator_name}_ratio"] = (
                float(value) if value is not None else None
            )


_R5G_TERMINATION_LABELS = {
    TERMINATION_BELOW_BOUND: "below_bound",
    TERMINATION_ABOVE_BOUND: "above_bound",
    TERMINATION_COLLISION: "collision",
    TERMINATION_TIMEOUT: "timeout",
}


class _R5GTerminationWindowTracker:
    def __init__(self, num_envs: int, *, height_floor: float, window_steps: int = 25):
        self.window_steps = int(window_steps)
        self.height_floor = float(height_floor)
        self._buffers = [[] for _ in range(int(num_envs))]
        self._accumulators = {
            label: {
                field_name: _TensorSummaryAccumulator()
                for field_name in (
                    "window_step",
                    "near_floor",
                    "v_cmd_z",
                    "v_gov_z",
                    "v_final_z",
                    "ics_beta",
                    "clearance",
                )
            }
            for label in ("below_bound", "above_bound", "collision", "timeout", "other")
        }

    def add_step(self, tensordict, optional_candidates, recorded):
        values = {
            "height": _get_optional_tensor(
                tensordict,
                optional_candidates.get("height_world_z", []),
            ),
            "v_cmd_z": _get_optional_tensor(
                tensordict,
                optional_candidates.get("governor_v_cmd_b_z", []),
            ),
            "v_gov_z": _get_optional_tensor(
                tensordict,
                optional_candidates.get("governor_v_gov_b_z", []),
            ),
            "v_final_z": _get_optional_tensor(
                tensordict,
                optional_candidates.get("governor_v_final_b_z", []),
            ),
            "ics_beta": _get_optional_tensor(
                tensordict,
                optional_candidates.get("ics_beta", []),
            ),
            "clearance": _get_optional_tensor(
                tensordict,
                optional_candidates.get("min_clearance", []),
            ),
        }
        if any(value is None for value in values.values()):
            return False
        recorded_cpu = recorded.detach().cpu().reshape(-1).bool()
        flat = {
            key: value.detach().float().cpu().reshape(-1)
            for key, value in values.items()
        }
        count = min(len(self._buffers), int(flat["height"].numel()))
        near_floor = (flat["height"] <= self.height_floor + 0.10).float()
        for env_id in range(count):
            if recorded_cpu[env_id]:
                continue
            self._buffers[env_id].append({
                "near_floor": float(near_floor[env_id].item()),
                "v_cmd_z": float(flat["v_cmd_z"][env_id].item()),
                "v_gov_z": float(flat["v_gov_z"][env_id].item()),
                "v_final_z": float(flat["v_final_z"][env_id].item()),
                "ics_beta": float(flat["ics_beta"][env_id].item()),
                "clearance": float(flat["clearance"][env_id].item()),
            })
            if len(self._buffers[env_id]) > self.window_steps:
                self._buffers[env_id].pop(0)
        return True

    def flush(self, newly_done, stats):
        done_cpu = newly_done.detach().cpu().reshape(-1).bool()
        reason_codes = self._reason_codes(stats)
        for env_id in done_cpu.nonzero(as_tuple=False).reshape(-1).tolist():
            label = _R5G_TERMINATION_LABELS.get(int(reason_codes[env_id].item()), "other")
            target = self._accumulators[label]
            for record in self._buffers[env_id]:
                near_floor = record["near_floor"]
                target["window_step"].add(torch.ones(1))
                target["near_floor"].add(torch.tensor([near_floor]))
                for field_name in ("v_cmd_z", "v_gov_z", "v_final_z", "ics_beta", "clearance"):
                    value = record[field_name] if near_floor >= 0.5 else float("nan")
                    target[field_name].add(torch.tensor([value]))
            self._buffers[env_id] = []

    def add_summaries(self, summary):
        summary["eval/handbook.r5g_near_floor_pre_termination_window_steps"] = self.window_steps
        for label, accumulators in self._accumulators.items():
            window_steps = accumulators["window_step"].finite_count
            if window_steps <= 0:
                continue
            near_count = accumulators["near_floor"].sum
            summary[f"eval/handbook.r5g_near_floor_window_steps_before_{label}"] = int(window_steps)
            summary[f"eval/handbook.r5g_near_floor_count_before_{label}"] = float(near_count)
            summary[f"eval/handbook.r5g_near_floor_rate_before_{label}"] = float(
                near_count / max(window_steps, 1)
            )
            for field_name, suffix in (
                ("v_cmd_z", "v_cmd_z_mean"),
                ("v_gov_z", "v_gov_z_mean"),
                ("v_final_z", "v_final_z_mean"),
                ("ics_beta", "ics_beta_mean"),
            ):
                value = accumulators[field_name].mean()
                summary[f"eval/handbook.r5g_near_floor_{suffix}_before_{label}"] = (
                    float(value) if value is not None else None
                )
            clearance_p05 = accumulators["clearance"].quantile(0.05)
            summary[f"eval/handbook.r5g_near_floor_clearance_p05_before_{label}"] = (
                float(clearance_p05) if clearance_p05 is not None else None
            )

    @staticmethod
    def _reason_codes(stats):
        keys = stats.keys() if hasattr(stats, "keys") else stats
        if "termination_reason_code" in keys:
            return stats["termination_reason_code"].detach().cpu().reshape(-1).long()
        below = _R5GTerminationWindowTracker._stat_tensor(
            stats,
            "terminated_below_bound",
            torch.zeros(1),
        ).bool()
        above = _R5GTerminationWindowTracker._stat_tensor(
            stats,
            "terminated_above_bound",
            torch.zeros_like(below),
        ).bool()
        collision = _R5GTerminationWindowTracker._stat_tensor(
            stats,
            "terminated_collision",
            torch.zeros_like(below),
        ).bool()
        timeout = _R5GTerminationWindowTracker._stat_tensor(
            stats,
            "truncated_timeout",
            torch.zeros_like(below),
        ).bool()
        codes = torch.zeros_like(below, dtype=torch.long)
        codes = torch.where(timeout, torch.full_like(codes, TERMINATION_TIMEOUT), codes)
        codes = torch.where(above, torch.full_like(codes, TERMINATION_ABOVE_BOUND), codes)
        codes = torch.where(below, torch.full_like(codes, TERMINATION_BELOW_BOUND), codes)
        codes = torch.where(collision, torch.full_like(codes, TERMINATION_COLLISION), codes)
        return codes

    @staticmethod
    def _stat_tensor(stats, name, default):
        keys = stats.keys() if hasattr(stats, "keys") else stats
        if name in keys:
            return stats[name].detach().cpu().reshape(-1)
        return default.detach().cpu().reshape(-1)


class R5E1ControllerLatencyTracker:
    def __init__(
        self,
        num_envs: int,
        *,
        command_eps: float,
        window_steps=R5H_COLLISION_WINDOW_STEPS,
        lag_steps=R5E1_LAG_STEPS,
    ):
        self.command_eps = float(command_eps)
        self.window_steps = tuple(int(value) for value in window_steps)
        self.lag_steps = tuple(int(value) for value in lag_steps)
        self._buffers = [[] for _ in range(int(num_envs))]
        self._max_window = max(
            [0] + list(self.window_steps) + list(self.lag_steps),
        )
        self._fields = _R5E1_CURRENT_FIELDS + _R5E1_LAG_FIELDS
        self._station_null_accumulators = self._make_accumulators(include_step=True)
        self._lag_accumulators = {
            field_name: _TensorSummaryAccumulator()
            for field_name in _R5E1_LAG_FIELDS
        }
        self._collision_episodes = {window: 0 for window in self.window_steps}
        self._collision_accumulators = {
            window: self._make_accumulators(include_step=True)
            for window in self.window_steps
        }

    def _make_accumulators(self, *, include_step: bool):
        field_names = (("step",) if include_step else ()) + self._fields
        return {
            field_name: _TensorSummaryAccumulator()
            for field_name in field_names
        }

    @staticmethod
    def _as_scalar_tensor(value):
        return torch.tensor([float(value)], dtype=torch.float32)

    def _add_record_metrics(self, accumulators, record):
        for field_name in self._fields:
            accumulators[field_name].add(self._as_scalar_tensor(record[field_name]))

    def _add_lag_metrics(self, record):
        for field_name in _R5E1_LAG_FIELDS:
            self._lag_accumulators[field_name].add(
                self._as_scalar_tensor(record[field_name])
            )

    def _make_record(self, metrics, lag_metrics, env_id, command_w, null_command):
        record = {
            "controller_command_w_vec": [
                float(value)
                for value in command_w[env_id].detach().cpu().reshape(3).tolist()
            ],
            "null_command": bool(null_command),
        }
        for field_name in _R5E1_CURRENT_FIELDS:
            record[field_name] = float(
                metrics[field_name][env_id].detach().float().cpu().reshape(-1)[0].item()
            )
        for field_name in _R5E1_LAG_FIELDS:
            record[field_name] = float(
                lag_metrics[field_name].detach().float().cpu().reshape(-1)[0].item()
            )
        return record

    def add_step(self, tensordict, optional_candidates, recorded):
        v_cmd_b = _get_optional_tensor(
            tensordict,
            optional_candidates.get("governor_v_cmd_b", [])
            + [("info", "v_cmd"), ("next", "info", "v_cmd")],
        )
        v_final_b = _get_optional_tensor(
            tensordict,
            optional_candidates.get("governor_v_final_b", []),
        )
        if v_final_b is None:
            v_final_b = _get_optional_tensor(
                tensordict,
                optional_candidates.get("governor_v_gov_b", []),
            )
        controller_command_w = _get_optional_tensor(
            tensordict,
            optional_candidates.get("r5e1_controller_command_w", []),
        )
        actual_velocity_b = _get_optional_tensor(
            tensordict,
            optional_candidates.get("actual_velocity_b", []),
        )
        actual_velocity_w = _actual_velocity_w_from_tensordict(
            tensordict,
            optional_candidates,
        )
        if any(
            value is None
            for value in (
                v_cmd_b,
                v_final_b,
                controller_command_w,
                actual_velocity_b,
                actual_velocity_w,
            )
        ):
            return False

        state_vec = _get_optional_tensor(
            tensordict,
            [
                ("agents", "observation", "state_vec"),
                ("next", "agents", "observation", "state_vec"),
            ],
        )
        prev_action_b = _latest_prev_action_from_state_vec(state_vec)
        metrics = compute_r5e1_controller_latency_step_metrics(
            v_final_b=v_final_b,
            controller_command_w=controller_command_w,
            actual_velocity_b=actual_velocity_b,
            actual_velocity_w=actual_velocity_w,
            prev_action_b=prev_action_b,
            command_eps=self.command_eps,
        )

        recorded_cpu = recorded.detach().cpu().reshape(-1).bool()
        v_cmd = v_cmd_b.detach().float().cpu().reshape(-1, 3)
        command_w = controller_command_w.detach().float().cpu().reshape(-1, 3)
        actual_w = actual_velocity_w.detach().float().cpu().reshape(-1, 3)
        count = min(
            len(self._buffers),
            int(v_cmd.shape[0]),
            int(command_w.shape[0]),
            int(actual_w.shape[0]),
        )
        null_commands = v_cmd.norm(dim=-1) <= self.command_eps
        for env_id in range(count):
            if recorded_cpu[env_id]:
                continue
            history = self._buffers[env_id]
            lagged_commands = {}
            for lag in self.lag_steps:
                if len(history) >= lag:
                    lagged_commands[lag] = torch.tensor(
                        [history[-lag]["controller_command_w_vec"]],
                        dtype=torch.float32,
                    )
                else:
                    lagged_commands[lag] = None
            lag_metrics = compute_r5e1_lagged_command_metrics(
                current_controller_command_w=command_w[env_id : env_id + 1],
                actual_velocity_w=actual_w[env_id : env_id + 1],
                lagged_controller_commands_w=lagged_commands,
                lag_steps=self.lag_steps,
            )
            record = self._make_record(
                metrics,
                lag_metrics,
                env_id,
                command_w,
                bool(null_commands[env_id].item()),
            )
            if record["null_command"]:
                self._station_null_accumulators["step"].add(torch.ones(1))
                self._add_record_metrics(self._station_null_accumulators, record)
            self._add_lag_metrics(record)
            history.append(record)
            if len(history) > self._max_window:
                history.pop(0)
        return True

    def flush(self, newly_done, stats):
        done_cpu = newly_done.detach().cpu().reshape(-1).bool()
        reason_codes = _R5GTerminationWindowTracker._reason_codes(stats)
        for env_id in done_cpu.nonzero(as_tuple=False).reshape(-1).tolist():
            is_collision = int(reason_codes[env_id].item()) == TERMINATION_COLLISION
            if is_collision:
                records = self._buffers[env_id]
                for window in self.window_steps:
                    target = self._collision_accumulators[window]
                    self._collision_episodes[window] += 1
                    for record in records[-window:]:
                        target["step"].add(torch.ones(1))
                        self._add_record_metrics(target, record)
            self._buffers[env_id] = []

    def _add_prefixed_summaries(self, summary, accumulators, *, key_prefix):
        for field_name in self._fields:
            _add_r5e1_field_summary(summary, accumulators, field_name, key_prefix)

    def add_diagnostic_summaries(self, summary):
        for field_name, accumulator in self._lag_accumulators.items():
            if accumulator.count > 0:
                summary[f"eval/diagnostics.{field_name}"] = accumulator.summary()

    def add_summaries(self, summary):
        station_steps = self._station_null_accumulators["step"].finite_count
        summary["eval/handbook.r5e1_station_null_steps"] = int(station_steps)
        if station_steps > 0:
            self._add_prefixed_summaries(
                summary,
                self._station_null_accumulators,
                key_prefix="eval/handbook.r5e1_station_null_",
            )

        for field_name in _R5E1_LAG_FIELDS:
            _add_r5e1_field_summary(
                summary,
                self._lag_accumulators,
                field_name,
                "eval/handbook.r5e1_",
            )

        for window in self.window_steps:
            accumulators = self._collision_accumulators[window]
            window_steps = accumulators["step"].finite_count
            summary[f"eval/handbook.r5e1_collision_window{window}_steps"] = int(
                window_steps
            )
            summary[f"eval/handbook.r5e1_collision_window{window}_episodes"] = int(
                self._collision_episodes[window]
            )
            if window_steps <= 0:
                continue
            self._add_prefixed_summaries(
                summary,
                accumulators,
                key_prefix=f"eval/handbook.r5e1_collision_window{window}_",
            )


_R5E2_WINDOW_RATE_FIELDS = {
    "below_bound_adjacent",
    "ceiling_adjacent",
    "height_adjacent",
    "lidar_collision_evidence",
    "min_clearance_source_available",
    "missing_clearance_source",
    "contact_telemetry_available",
    "missing_contact_telemetry",
    "collision",
    "terminated_collision",
    "collision_termination_same_step",
    "collision_without_termination",
    "termination_collision_without_collision",
}


_R5E3_WINDOW_RATE_FIELDS = {
    "low_beta",
    "full_stop_commanded",
    "full_stop_after_collision_margin_exhausted",
    "full_stop_after_emergency_margin_exhausted",
    "missing_contact_telemetry",
    "missing_body_telemetry",
    "missing_surface_normal",
    "missing_measured_deceleration",
    "missing_worst_ics_beam",
    "conservative_approximation_used",
}

_R5E3_WINDOW_P05_FIELDS = {
    "ics_min_clearance",
    "raw_min_clearance",
    "required_stop_distance_conservative",
    "residual_to_collision_threshold",
    "residual_to_emergency",
    "residual_to_d_safe",
    "worst_beam_required_stop_distance",
    "worst_beam_residual_to_collision_threshold",
    "worst_beam_residual_to_emergency",
    "worst_beam_residual_to_d_safe",
}


class _R5E3BrakingResidualTracker:
    def __init__(
        self,
        num_envs: int,
        *,
        collision_window_steps=R5E3_COLLISION_WINDOW_STEPS,
        low_beta_window_steps=R5E3_LOW_BETA_WINDOW_STEPS,
    ):
        self.collision_window_steps = tuple(int(value) for value in collision_window_steps)
        self.low_beta_window_steps = tuple(int(value) for value in low_beta_window_steps)
        self._buffers = [[] for _ in range(int(num_envs))]
        all_windows = list(self.collision_window_steps) + list(self.low_beta_window_steps)
        self._max_window = max(all_windows) if all_windows else 0
        self._low_beta_recorded = [False for _ in range(int(num_envs))]
        self._collision_episodes = {window: 0 for window in self.collision_window_steps}
        self._low_beta_episodes = {window: 0 for window in self.low_beta_window_steps}
        self._collision_accumulators = {
            window: self._make_window_accumulators("steps_before_termination")
            for window in self.collision_window_steps
        }
        self._low_beta_accumulators = {
            window: self._make_window_accumulators("steps_before_low_beta")
            for window in self.low_beta_window_steps
        }

    @staticmethod
    def _as_scalar_tensor(value):
        return torch.tensor([float(value)], dtype=torch.float32)

    @staticmethod
    def _make_window_accumulators(step_field):
        return {
            field_name: _TensorSummaryAccumulator()
            for field_name in (
                "window_step",
                step_field,
            ) + R5E3_WINDOW_VALUE_FIELDS
        }

    @staticmethod
    def _r5e3_tensor(tensordict, optional_candidates, suffix):
        return _get_optional_tensor(
            tensordict,
            optional_candidates.get(f"r5e3_{suffix}", []),
        )

    @staticmethod
    def _vector_record(prefix, value):
        vector = value.detach().float().cpu().reshape(3)
        return {
            f"{prefix}_x": float(vector[0].item()),
            f"{prefix}_y": float(vector[1].item()),
            f"{prefix}_z": float(vector[2].item()),
            f"{prefix}_speed_xy": float(vector[:2].norm().item()),
            f"{prefix}_speed_z_abs": float(vector[2].abs().item()),
            f"{prefix}_speed_norm": float(vector.norm().item()),
        }

    def _make_record(
        self,
        *,
        env_id,
        v_gov_b,
        v_final_b,
        controller_command_w,
        actual_velocity_b,
        actual_velocity_w,
        beta,
        emergency,
        active_beams,
        flat,
    ):
        record = {}
        record.update(self._vector_record("v_gov_body", v_gov_b[env_id]))
        record.update(self._vector_record("v_final_body", v_final_b[env_id]))
        record.update(
            self._vector_record(
                "controller_command_world",
                controller_command_w[env_id],
            )
        )
        record.update(self._vector_record("actual_body", actual_velocity_b[env_id]))
        record.update(self._vector_record("actual_world", actual_velocity_w[env_id]))
        record["ics_beta"] = float(beta[env_id].item())
        record["ics_emergency"] = float(emergency[env_id].item())
        record["ics_active_beam_count"] = float(active_beams[env_id].item())
        for suffix in (
            "ics_min_clearance",
            "raw_min_clearance",
            "collision_clearance_threshold",
            "emergency_clearance",
            "d_safe",
            "required_stop_distance_conservative",
            "residual_to_collision_threshold",
            "residual_to_emergency",
            "residual_to_d_safe",
            "worst_beam_closing_speed",
            "worst_beam_required_stop_distance",
            "worst_beam_residual_to_collision_threshold",
            "worst_beam_residual_to_emergency",
            "worst_beam_residual_to_d_safe",
            "low_beta",
            "full_stop_commanded",
            "full_stop_after_collision_margin_exhausted",
            "full_stop_after_emergency_margin_exhausted",
            "missing_contact_telemetry",
            "missing_body_telemetry",
            "missing_surface_normal",
            "missing_measured_deceleration",
            "missing_worst_ics_beam",
            "conservative_approximation_used",
        ):
            record[suffix] = float(flat[suffix][env_id].item())
        return record

    def _add_window_records(self, target, records, step_field):
        selected_len = len(records)
        for index, record in enumerate(records):
            target["window_step"].add(torch.ones(1))
            target[step_field].add(self._as_scalar_tensor(selected_len - index - 1))
            for field_name in R5E3_WINDOW_VALUE_FIELDS:
                target[field_name].add(self._as_scalar_tensor(record[field_name]))

    def add_step(self, tensordict, optional_candidates, recorded):
        v_gov_b = _get_optional_tensor(
            tensordict,
            optional_candidates.get("governor_v_gov_b", []),
        )
        v_final_b = _get_optional_tensor(
            tensordict,
            optional_candidates.get("governor_v_final_b", []),
        )
        if v_final_b is None:
            v_final_b = v_gov_b
        controller_command_w = _get_optional_tensor(
            tensordict,
            optional_candidates.get("r5e1_controller_command_w", []),
        )
        actual_velocity_b = _get_optional_tensor(
            tensordict,
            optional_candidates.get("actual_velocity_b", []),
        )
        actual_velocity_w = _actual_velocity_w_from_tensordict(
            tensordict,
            optional_candidates,
        )
        required_suffixes = (
            "ics_min_clearance",
            "raw_min_clearance",
            "collision_clearance_threshold",
            "emergency_clearance",
            "d_safe",
            "required_stop_distance_conservative",
            "residual_to_collision_threshold",
            "residual_to_emergency",
            "residual_to_d_safe",
            "worst_beam_closing_speed",
            "worst_beam_required_stop_distance",
            "worst_beam_residual_to_collision_threshold",
            "worst_beam_residual_to_emergency",
            "worst_beam_residual_to_d_safe",
            "low_beta",
            "full_stop_commanded",
            "full_stop_after_collision_margin_exhausted",
            "full_stop_after_emergency_margin_exhausted",
            "missing_contact_telemetry",
            "missing_body_telemetry",
            "missing_surface_normal",
            "missing_measured_deceleration",
            "missing_worst_ics_beam",
            "conservative_approximation_used",
        )
        required = {
            suffix: self._r5e3_tensor(tensordict, optional_candidates, suffix)
            for suffix in required_suffixes
        }
        if any(
            value is None
            for value in (
                v_gov_b,
                v_final_b,
                controller_command_w,
                actual_velocity_b,
                actual_velocity_w,
                *required.values(),
            )
        ):
            return False

        beta = _get_optional_tensor(
            tensordict,
            optional_candidates.get("ics_beta", []),
        )
        emergency = _get_optional_tensor(
            tensordict,
            optional_candidates.get("ics_emergency", []),
        )
        active_beams = _get_optional_tensor(
            tensordict,
            optional_candidates.get("ics_active_beam_count", []),
        )

        recorded_cpu = recorded.detach().cpu().reshape(-1).bool()
        gov_b = v_gov_b.detach().float().cpu().reshape(-1, 3)
        final_b = v_final_b.detach().float().cpu().reshape(-1, 3)
        command_w = controller_command_w.detach().float().cpu().reshape(-1, 3)
        actual_b = actual_velocity_b.detach().float().cpu().reshape(-1, 3)
        actual_w = actual_velocity_w.detach().float().cpu().reshape(-1, 3)
        flat = {
            suffix: value.detach().float().cpu().reshape(-1)
            for suffix, value in required.items()
        }
        beta = (
            beta.detach().float().cpu().reshape(-1)
            if beta is not None
            else torch.ones_like(flat["low_beta"])
        )
        emergency = (
            emergency.detach().float().cpu().reshape(-1)
            if emergency is not None
            else torch.zeros_like(flat["low_beta"])
        )
        active_beams = (
            active_beams.detach().float().cpu().reshape(-1)
            if active_beams is not None
            else torch.zeros_like(flat["low_beta"])
        )

        count = min(
            len(self._buffers),
            int(gov_b.shape[0]),
            int(final_b.shape[0]),
            int(command_w.shape[0]),
            int(actual_b.shape[0]),
            int(actual_w.shape[0]),
            int(flat["low_beta"].numel()),
        )
        for env_id in range(count):
            if recorded_cpu[env_id]:
                continue
            record = self._make_record(
                env_id=env_id,
                v_gov_b=gov_b,
                v_final_b=final_b,
                controller_command_w=command_w,
                actual_velocity_b=actual_b,
                actual_velocity_w=actual_w,
                beta=beta,
                emergency=emergency,
                active_beams=active_beams,
                flat=flat,
            )
            self._buffers[env_id].append(record)
            if len(self._buffers[env_id]) > self._max_window:
                self._buffers[env_id].pop(0)
            if record["low_beta"] >= 0.5 and not self._low_beta_recorded[env_id]:
                self._low_beta_recorded[env_id] = True
                for window in self.low_beta_window_steps:
                    self._low_beta_episodes[window] += 1
                    self._add_window_records(
                        self._low_beta_accumulators[window],
                        self._buffers[env_id][-window:],
                        "steps_before_low_beta",
                    )
        return True

    def flush(self, newly_done, stats):
        done_cpu = newly_done.detach().cpu().reshape(-1).bool()
        reason_codes = _R5GTerminationWindowTracker._reason_codes(stats)
        for env_id in done_cpu.nonzero(as_tuple=False).reshape(-1).tolist():
            records = self._buffers[env_id]
            if int(reason_codes[env_id].item()) == TERMINATION_COLLISION and records:
                for window in self.collision_window_steps:
                    self._collision_episodes[window] += 1
                    self._add_window_records(
                        self._collision_accumulators[window],
                        records[-window:],
                        "steps_before_termination",
                    )
            self._buffers[env_id] = []
            self._low_beta_recorded[env_id] = False

    def _add_event_summaries(self, summary, *, prefix, accumulators, episodes, step_field):
        for window, fields in accumulators.items():
            window_steps = fields["window_step"].finite_count
            summary[f"eval/handbook.{prefix}_window{window}_steps"] = int(window_steps)
            summary[f"eval/handbook.{prefix}_window{window}_episodes"] = int(
                episodes[window]
            )
            if window_steps <= 0:
                continue
            step_mean = fields[step_field].mean()
            step_max = fields[step_field].summary().get("max")
            summary[f"eval/handbook.{prefix}_window{window}_{step_field}_mean"] = (
                float(step_mean) if step_mean is not None else None
            )
            if step_max is not None:
                summary[f"eval/handbook.{prefix}_window{window}_{step_field}_max"] = (
                    float(step_max)
                )
            for field_name in R5E3_WINDOW_VALUE_FIELDS:
                accumulator = fields[field_name]
                value = accumulator.mean()
                suffix = "rate" if field_name in _R5E3_WINDOW_RATE_FIELDS else "mean"
                summary[f"eval/handbook.{prefix}_window{window}_{field_name}_{suffix}"] = (
                    float(value) if value is not None else None
                )
                if field_name in _R5E3_WINDOW_P05_FIELDS:
                    p05 = accumulator.quantile(0.05)
                    summary[f"eval/handbook.{prefix}_window{window}_{field_name}_p05"] = (
                        float(p05) if p05 is not None else None
                    )

    def add_summaries(self, summary):
        self._add_event_summaries(
            summary,
            prefix="r5e3_collision",
            accumulators=self._collision_accumulators,
            episodes=self._collision_episodes,
            step_field="steps_before_termination",
        )
        self._add_event_summaries(
            summary,
            prefix="r5e3_low_beta",
            accumulators=self._low_beta_accumulators,
            episodes=self._low_beta_episodes,
            step_field="steps_before_low_beta",
        )


class _R5E2CollisionGeometryTracker:
    def __init__(
        self,
        num_envs: int,
        *,
        window_steps=R5E2_COLLISION_WINDOW_STEPS,
    ):
        self.window_steps = tuple(int(value) for value in window_steps)
        self._buffers = [[] for _ in range(int(num_envs))]
        self._max_window = max(self.window_steps) if self.window_steps else 0
        self._collision_termination_episodes = 0
        self._reason_counts = {
            label: 0
            for label in R5E2_COLLISION_REASON_LABELS.values()
        }
        self._terminal_accumulators = {
            field_name: _TensorSummaryAccumulator()
            for field_name in (
                "min_clearance",
                "root_z",
                "below_bound_adjacent",
                "ceiling_adjacent",
                "height_adjacent",
                "lidar_collision_evidence",
                "min_clearance_source_available",
                "missing_clearance_source",
                "contact_telemetry_available",
                "missing_contact_telemetry",
                "collision_termination_same_step",
                "termination_collision_without_collision",
            )
        }
        self._window_accumulators = {
            window: {
                field_name: _TensorSummaryAccumulator()
                for field_name in ("window_step",) + R5E2_COLLISION_WINDOW_VALUE_FIELDS
            }
            for window in self.window_steps
        }
        self._collision_episodes = {window: 0 for window in self.window_steps}

    @staticmethod
    def _as_scalar_tensor(value):
        return torch.tensor([float(value)], dtype=torch.float32)

    @staticmethod
    def _r5e2_tensor(tensordict, optional_candidates, suffix):
        field_name = f"r5e2_{suffix}"
        return _get_optional_tensor(
            tensordict,
            optional_candidates.get(field_name, []),
        )

    def add_step(self, tensordict, optional_candidates, recorded):
        v_final_b = _get_optional_tensor(
            tensordict,
            optional_candidates.get("governor_v_final_b", []),
        )
        if v_final_b is None:
            v_final_b = _get_optional_tensor(
                tensordict,
                optional_candidates.get("governor_v_gov_b", []),
            )
        controller_command_w = _get_optional_tensor(
            tensordict,
            optional_candidates.get("r5e1_controller_command_w", []),
        )
        actual_velocity_b = _get_optional_tensor(
            tensordict,
            optional_candidates.get("actual_velocity_b", []),
        )
        actual_velocity_w = _actual_velocity_w_from_tensordict(
            tensordict,
            optional_candidates,
        )
        r5e2_required = {
            suffix: self._r5e2_tensor(tensordict, optional_candidates, suffix)
            for suffix in (
                "collision",
                "terminated_collision",
                "root_z",
                "below_bound_adjacent",
                "ceiling_adjacent",
                "height_adjacent",
                "min_clearance",
                "min_clearance_source_available",
                "missing_clearance_source",
                "lidar_collision_evidence",
                "contact_telemetry_available",
                "missing_contact_telemetry",
                "collision_termination_same_step",
                "collision_without_termination",
                "termination_collision_without_collision",
                "reason_code",
            )
        }
        if any(
            value is None
            for value in (
                v_final_b,
                controller_command_w,
                actual_velocity_b,
                actual_velocity_w,
                *r5e2_required.values(),
            )
        ):
            return False

        beta = _get_optional_tensor(
            tensordict,
            optional_candidates.get("ics_beta", []),
        )
        emergency = _get_optional_tensor(
            tensordict,
            optional_candidates.get("ics_emergency", []),
        )
        active_beams = _get_optional_tensor(
            tensordict,
            optional_candidates.get("ics_active_beam_count", []),
        )

        recorded_cpu = recorded.detach().cpu().reshape(-1).bool()
        final_b = v_final_b.detach().float().cpu().reshape(-1, 3)
        command_w = controller_command_w.detach().float().cpu().reshape(-1, 3)
        actual_b = actual_velocity_b.detach().float().cpu().reshape(-1, 3)
        actual_w = actual_velocity_w.detach().float().cpu().reshape(-1, 3)
        flat = {
            suffix: value.detach().float().cpu().reshape(-1)
            for suffix, value in r5e2_required.items()
            if suffix != "reason_code"
        }
        reason_code = r5e2_required["reason_code"].detach().cpu().reshape(-1).long()
        beta = (
            beta.detach().float().cpu().reshape(-1)
            if beta is not None
            else torch.ones_like(flat["collision"])
        )
        emergency = (
            emergency.detach().float().cpu().reshape(-1)
            if emergency is not None
            else torch.zeros_like(flat["collision"])
        )
        active_beams = (
            active_beams.detach().float().cpu().reshape(-1)
            if active_beams is not None
            else torch.zeros_like(flat["collision"])
        )

        count = min(
            len(self._buffers),
            int(final_b.shape[0]),
            int(command_w.shape[0]),
            int(actual_b.shape[0]),
            int(actual_w.shape[0]),
            int(flat["collision"].numel()),
            int(reason_code.numel()),
        )
        for env_id in range(count):
            if recorded_cpu[env_id]:
                continue
            record = {
                "v_final_body_x": float(final_b[env_id, 0].item()),
                "v_final_body_y": float(final_b[env_id, 1].item()),
                "v_final_body_z": float(final_b[env_id, 2].item()),
                "v_final_body_speed_xy": float(final_b[env_id, :2].norm().item()),
                "v_final_body_speed_z_abs": float(final_b[env_id, 2].abs().item()),
                "controller_command_world_x": float(command_w[env_id, 0].item()),
                "controller_command_world_y": float(command_w[env_id, 1].item()),
                "controller_command_world_z": float(command_w[env_id, 2].item()),
                "controller_command_world_speed_xy": float(command_w[env_id, :2].norm().item()),
                "controller_command_world_speed_z_abs": float(command_w[env_id, 2].abs().item()),
                "actual_body_x": float(actual_b[env_id, 0].item()),
                "actual_body_y": float(actual_b[env_id, 1].item()),
                "actual_body_z": float(actual_b[env_id, 2].item()),
                "actual_body_speed_xy": float(actual_b[env_id, :2].norm().item()),
                "actual_body_speed_z_abs": float(actual_b[env_id, 2].abs().item()),
                "actual_world_x": float(actual_w[env_id, 0].item()),
                "actual_world_y": float(actual_w[env_id, 1].item()),
                "actual_world_z": float(actual_w[env_id, 2].item()),
                "actual_world_speed_xy": float(actual_w[env_id, :2].norm().item()),
                "actual_world_speed_z_abs": float(actual_w[env_id, 2].abs().item()),
                "ics_beta": float(beta[env_id].item()),
                "ics_emergency": float(emergency[env_id].item()),
                "ics_active_beam_count": float(active_beams[env_id].item()),
                "reason_code": int(reason_code[env_id].item()),
            }
            for suffix in (
                "min_clearance",
                "root_z",
                "below_bound_adjacent",
                "ceiling_adjacent",
                "height_adjacent",
                "lidar_collision_evidence",
                "min_clearance_source_available",
                "missing_clearance_source",
                "contact_telemetry_available",
                "missing_contact_telemetry",
                "collision",
                "terminated_collision",
                "collision_termination_same_step",
                "collision_without_termination",
                "termination_collision_without_collision",
            ):
                record[suffix] = float(flat[suffix][env_id].item())
            self._buffers[env_id].append(record)
            if len(self._buffers[env_id]) > self._max_window:
                self._buffers[env_id].pop(0)
        return True

    def flush(self, newly_done, stats):
        done_cpu = newly_done.detach().cpu().reshape(-1).bool()
        reason_codes = _R5GTerminationWindowTracker._reason_codes(stats)
        for env_id in done_cpu.nonzero(as_tuple=False).reshape(-1).tolist():
            records = self._buffers[env_id]
            if not records:
                continue
            final_record = records[-1]
            is_collision_termination = (
                int(reason_codes[env_id].item()) == TERMINATION_COLLISION
                or final_record["terminated_collision"] >= 0.5
            )
            if is_collision_termination:
                self._collision_termination_episodes += 1
                code = int(final_record["reason_code"])
                if code == R5E2_REASON_NOT_COLLISION_TERMINATION:
                    label = "unknown"
                else:
                    label = R5E2_COLLISION_REASON_LABELS.get(code, "unknown")
                self._reason_counts[label] += 1
                for field_name, accumulator in self._terminal_accumulators.items():
                    accumulator.add(self._as_scalar_tensor(final_record[field_name]))
                for window in self.window_steps:
                    target = self._window_accumulators[window]
                    self._collision_episodes[window] += 1
                    selected = records[-window:]
                    for index, record in enumerate(selected):
                        target["window_step"].add(torch.ones(1))
                        for field_name in R5E2_COLLISION_WINDOW_VALUE_FIELDS:
                            if field_name == "steps_before_termination":
                                value = float(len(selected) - index - 1)
                            else:
                                value = record[field_name]
                            target[field_name].add(self._as_scalar_tensor(value))
            self._buffers[env_id] = []

    def add_summaries(self, summary):
        episodes = int(self._collision_termination_episodes)
        summary["eval/handbook.r5e2_collision_termination_episodes"] = episodes
        for label, count in self._reason_counts.items():
            summary[f"eval/handbook.r5e2_reason_{label}_count"] = int(count)
            summary[f"eval/handbook.r5e2_reason_{label}_fraction"] = (
                float(count / episodes) if episodes > 0 else None
            )

        for field_name, handbook_key in (
            (
                "missing_contact_telemetry",
                "eval/handbook.r5e2_missing_contact_telemetry_collision_termination_rate",
            ),
            (
                "contact_telemetry_available",
                "eval/handbook.r5e2_contact_telemetry_collision_termination_rate",
            ),
            (
                "missing_clearance_source",
                "eval/handbook.r5e2_missing_clearance_source_collision_termination_rate",
            ),
            (
                "min_clearance_source_available",
                "eval/handbook.r5e2_min_clearance_source_collision_termination_rate",
            ),
            (
                "lidar_collision_evidence",
                "eval/handbook.r5e2_lidar_collision_evidence_collision_termination_rate",
            ),
            (
                "below_bound_adjacent",
                "eval/handbook.r5e2_below_bound_adjacent_collision_termination_rate",
            ),
            (
                "ceiling_adjacent",
                "eval/handbook.r5e2_ceiling_adjacent_collision_termination_rate",
            ),
            (
                "height_adjacent",
                "eval/handbook.r5e2_height_adjacent_collision_termination_rate",
            ),
            (
                "collision_termination_same_step",
                "eval/handbook.r5e2_collision_termination_same_step_collision_termination_rate",
            ),
            (
                "termination_collision_without_collision",
                "eval/handbook.r5e2_termination_collision_without_collision_collision_termination_rate",
            ),
        ):
            value = self._terminal_accumulators[field_name].mean()
            if episodes > 0:
                summary[handbook_key] = float(value) if value is not None else None
        clearance_p05 = self._terminal_accumulators["min_clearance"].quantile(0.05)
        if episodes > 0:
            summary["eval/handbook.r5e2_min_clearance_p05_at_collision_termination"] = (
                float(clearance_p05) if clearance_p05 is not None else None
            )

        for window in self.window_steps:
            accumulators = self._window_accumulators[window]
            window_steps = accumulators["window_step"].finite_count
            summary[f"eval/handbook.r5e2_collision_window{window}_steps"] = int(window_steps)
            summary[f"eval/handbook.r5e2_collision_window{window}_episodes"] = int(
                self._collision_episodes[window]
            )
            if window_steps <= 0:
                continue
            for field_name in R5E2_COLLISION_WINDOW_VALUE_FIELDS:
                accumulator = accumulators[field_name]
                if field_name == "min_clearance":
                    mean = accumulator.mean()
                    p05 = accumulator.quantile(0.05)
                    summary[f"eval/handbook.r5e2_collision_window{window}_min_clearance_mean"] = (
                        float(mean) if mean is not None else None
                    )
                    summary[f"eval/handbook.r5e2_collision_window{window}_min_clearance_p05"] = (
                        float(p05) if p05 is not None else None
                    )
                elif field_name == "root_z":
                    root_summary = accumulator.summary()
                    mean = accumulator.mean()
                    summary[f"eval/handbook.r5e2_collision_window{window}_root_z_mean"] = (
                        float(mean) if mean is not None else None
                    )
                    for suffix in ("min", "max"):
                        if suffix in root_summary:
                            summary[
                                f"eval/handbook.r5e2_collision_window{window}_root_z_{suffix}"
                            ] = float(root_summary[suffix])
                elif field_name == "steps_before_termination":
                    mean = accumulator.mean()
                    maximum = accumulator.summary().get("max")
                    summary[
                        f"eval/handbook.r5e2_collision_window{window}_steps_before_termination_mean"
                    ] = float(mean) if mean is not None else None
                    if maximum is not None:
                        summary[
                            f"eval/handbook.r5e2_collision_window{window}_steps_before_termination_max"
                        ] = float(maximum)
                else:
                    value = accumulator.mean()
                    suffix = "rate" if field_name in _R5E2_WINDOW_RATE_FIELDS else "mean"
                    summary[f"eval/handbook.r5e2_collision_window{window}_{field_name}_{suffix}"] = (
                        float(value) if value is not None else None
                    )


class _R5HCollisionWindowTracker:
    def __init__(
        self,
        num_envs: int,
        *,
        height_floor: float,
        window_steps=R5H_COLLISION_WINDOW_STEPS,
    ):
        self.window_steps = tuple(int(value) for value in window_steps)
        self.height_floor = float(height_floor)
        self._buffers = [[] for _ in range(int(num_envs))]
        self._collision_episodes = {window: 0 for window in self.window_steps}
        self._accumulators = {
            window: {
                field_name: _TensorSummaryAccumulator()
                for field_name in ("window_step",) + R5H_COLLISION_WINDOW_VALUE_FIELDS
            }
            for window in self.window_steps
        }
        self._max_window = max(self.window_steps) if self.window_steps else 0

    def add_step(self, tensordict, optional_candidates, recorded):
        required = {
            "height": _get_optional_tensor(
                tensordict,
                optional_candidates.get("height_world_z", []),
            ),
            "v_cmd": _get_optional_tensor(
                tensordict,
                optional_candidates.get("governor_v_cmd_b", [])
                + [("info", "v_cmd"), ("next", "info", "v_cmd")],
            ),
            "v_gov": _get_optional_tensor(
                tensordict,
                optional_candidates.get("governor_v_gov_b", []),
            ),
            "v_final": _get_optional_tensor(
                tensordict,
                optional_candidates.get("governor_v_final_b", []),
            ),
            "actual": _get_optional_tensor(
                tensordict,
                optional_candidates.get("actual_velocity_b", []),
            ),
            "min_clearance": _get_optional_tensor(
                tensordict,
                optional_candidates.get("min_clearance", []),
            ),
            "ics_beta": _get_optional_tensor(
                tensordict,
                optional_candidates.get("ics_beta", []),
            ),
        }
        if required["v_final"] is None:
            required["v_final"] = required["v_gov"]
        if any(value is None for value in required.values()):
            return False

        active_beams = _get_optional_tensor(
            tensordict,
            optional_candidates.get("ics_active_beam_count", []),
        )
        downward_beta = _get_optional_tensor(
            tensordict,
            optional_candidates.get("ics_downward_beta", []),
        )
        downward_active = _get_optional_tensor(
            tensordict,
            optional_candidates.get("ics_downward_active", []),
        )

        recorded_cpu = recorded.detach().cpu().reshape(-1).bool()
        height = required["height"].detach().float().cpu().reshape(-1)
        clearance = required["min_clearance"].detach().float().cpu().reshape(-1)
        beta = required["ics_beta"].detach().float().cpu().reshape(-1)
        v_cmd = required["v_cmd"].detach().float().cpu().reshape(-1, 3)
        v_gov = required["v_gov"].detach().float().cpu().reshape(-1, 3)
        v_final = required["v_final"].detach().float().cpu().reshape(-1, 3)
        actual = required["actual"].detach().float().cpu().reshape(-1, 3)
        active_beams = (
            active_beams.detach().float().cpu().reshape(-1)
            if active_beams is not None
            else torch.zeros_like(clearance)
        )
        downward_beta = (
            downward_beta.detach().float().cpu().reshape(-1)
            if downward_beta is not None
            else torch.ones_like(clearance)
        )
        downward_active = (
            downward_active.detach().float().cpu().reshape(-1)
            if downward_active is not None
            else torch.zeros_like(clearance)
        )
        count = min(
            len(self._buffers),
            int(height.numel()),
            int(v_cmd.shape[0]),
            int(v_gov.shape[0]),
            int(v_final.shape[0]),
            int(actual.shape[0]),
        )
        near_floor = (height <= self.height_floor + 0.10).float()
        for env_id in range(count):
            if recorded_cpu[env_id]:
                continue
            self._buffers[env_id].append({
                "min_clearance": float(clearance[env_id].item()),
                "ics_beta": float(beta[env_id].item()),
                "ics_downward_beta": float(downward_beta[env_id].item()),
                "ics_active_beam_count": float(active_beams[env_id].item()),
                "v_cmd_xy_norm": float(v_cmd[env_id, :2].norm().item()),
                "v_cmd_z_abs": float(v_cmd[env_id, 2].abs().item()),
                "v_gov_xy_norm": float(v_gov[env_id, :2].norm().item()),
                "v_gov_z_abs": float(v_gov[env_id, 2].abs().item()),
                "v_final_xy_norm": float(v_final[env_id, :2].norm().item()),
                "v_final_z_abs": float(v_final[env_id, 2].abs().item()),
                "actual_xy_speed": float(actual[env_id, :2].norm().item()),
                "actual_z_abs": float(actual[env_id, 2].abs().item()),
                "near_floor": float(near_floor[env_id].item()),
                "downward_active": float(downward_active[env_id].item()),
            })
            if len(self._buffers[env_id]) > self._max_window:
                self._buffers[env_id].pop(0)
        return True

    def flush(self, newly_done, stats):
        done_cpu = newly_done.detach().cpu().reshape(-1).bool()
        reason_codes = _R5GTerminationWindowTracker._reason_codes(stats)
        for env_id in done_cpu.nonzero(as_tuple=False).reshape(-1).tolist():
            is_collision = int(reason_codes[env_id].item()) == TERMINATION_COLLISION
            if is_collision:
                records = self._buffers[env_id]
                for window in self.window_steps:
                    target = self._accumulators[window]
                    self._collision_episodes[window] += 1
                    for record in records[-window:]:
                        target["window_step"].add(torch.ones(1))
                        for field_name in R5H_COLLISION_WINDOW_VALUE_FIELDS:
                            target[field_name].add(torch.tensor([record[field_name]]))
            self._buffers[env_id] = []

    def add_summaries(self, summary):
        for window in self.window_steps:
            accumulators = self._accumulators[window]
            window_steps = accumulators["window_step"].finite_count
            summary[f"eval/handbook.r5h_collision_window{window}_steps"] = int(window_steps)
            summary[f"eval/handbook.r5h_collision_window{window}_episodes"] = int(
                self._collision_episodes[window]
            )
            if window_steps <= 0:
                continue
            clearance_mean = accumulators["min_clearance"].mean()
            summary[f"eval/handbook.r5h_collision_window{window}_min_clearance_mean"] = (
                float(clearance_mean) if clearance_mean is not None else None
            )
            clearance_p05 = accumulators["min_clearance"].quantile(0.05)
            summary[f"eval/handbook.r5h_collision_window{window}_min_clearance_p05"] = (
                float(clearance_p05) if clearance_p05 is not None else None
            )
            for field_name in R5H_COLLISION_WINDOW_VALUE_FIELDS:
                if field_name == "min_clearance":
                    continue
                value = accumulators[field_name].mean()
                suffix = "rate" if field_name in {"near_floor", "downward_active"} else "mean"
                summary[f"eval/handbook.r5h_collision_window{window}_{field_name}_{suffix}"] = (
                    float(value) if value is not None else None
                )


@torch.no_grad()
def _evaluate_streaming(
    env,
    policy,
    cfg,
    seed: int,
    exploration_type: ExplorationType,
    return_summary: bool,
    record_video: bool,
):
    env.enable_render(record_video)
    env.eval()
    env.set_seed(seed)

    render_callback = RenderCallback(interval=2) if record_video else None
    td = env.reset()
    num_envs = int(env.num_envs)
    recorded = torch.zeros(num_envs, dtype=torch.bool, device=td.device)
    first_episode_stats = {}
    last_stats = None
    optional_candidates = _make_optional_eval_field_candidates()
    diagnostic_accumulators = {
        field_name: _TensorSummaryAccumulator()
        for field_name in optional_candidates
    }
    vertical_diagnostic_accumulators = {
        field_name: _TensorSummaryAccumulator()
        for field_name in (
            "vertical_command_active",
            "vertical_command_null",
            "vertical_corr_z",
            "vertical_corr_z_abs",
            "vertical_corr_z_positive",
            "vertical_corr_z_negative",
            "vertical_corr_z_saturated",
            "vertical_gov_minus_cmd_z",
            "vertical_gov_minus_cmd_z_abs",
            "vertical_final_minus_cmd_z",
            "vertical_final_minus_cmd_z_abs",
            "vertical_ics_delta_z",
            "vertical_ics_delta_z_abs",
            "vertical_corr_reinforces_command",
            "vertical_corr_opposes_command",
            "vertical_null_corr_active",
            "vertical_null_corr_abs",
            "vertical_null_station_drift_when_corr_active",
            "vertical_tracking_corr_active",
            "vertical_tracking_amplification_when_corr_active",
            "vertical_tracking_preservation_when_corr_active",
            "vertical_ics_beta",
            "vertical_ics_emergency",
        )
    }
    r5e1_diagnostic_accumulators = _make_r5e1_diagnostic_accumulators()
    r5e_diagnostic_accumulators = _make_r5e_diagnostic_accumulators()
    r5g_station_accumulators = _make_r5g_station_accumulators()
    r5g_downward_accumulators = _make_r5g_downward_accumulators()
    r5h_diagnostic_accumulators = _make_r5h_diagnostic_accumulators()
    r5e2_collision_tracker = _R5E2CollisionGeometryTracker(num_envs)
    r5e3_braking_tracker = _R5E3BrakingResidualTracker(num_envs)
    v_corr_limit = _governor_v_corr_limit(cfg)
    _, r5g_height_floor, _ = _r5e_eval_config(cfg)
    r5e1_command_eps, _, _ = _r5e_eval_config(cfg)
    r5e1_latency_tracker = R5E1ControllerLatencyTracker(
        num_envs,
        command_eps=r5e1_command_eps,
    )
    r5g_termination_tracker = _R5GTerminationWindowTracker(
        num_envs,
        height_floor=r5g_height_floor,
    )
    r5h_collision_tracker = _R5HCollisionWindowTracker(
        num_envs,
        height_floor=r5g_height_floor,
    )
    reward_accumulator = _TensorSummaryAccumulator()
    reward_sum = 0.0

    with set_exploration_type(exploration_type):
        for _ in range(env.max_episode_length):
            policy(td)
            step_td, td = env.step_and_maybe_reset(td)
            if render_callback is not None:
                render_callback(env)

            for field_name, candidates in optional_candidates.items():
                value = _get_optional_tensor(step_td, candidates)
                if value is not None:
                    diagnostic_accumulators[field_name].add(value)

            vertical_inputs = {
                field_name: _get_optional_tensor(step_td, optional_candidates[field_name])
                for field_name in (
                    "governor_v_cmd_b_z",
                    "governor_v_corr_z",
                    "governor_v_gov_b_z",
                    "governor_v_final_b_z",
                )
            }
            if all(value is not None for value in vertical_inputs.values()):
                vertical_metrics = compute_vertical_channel_step_metrics(
                    v_cmd_z=vertical_inputs["governor_v_cmd_b_z"],
                    v_corr_z=vertical_inputs["governor_v_corr_z"],
                    v_gov_z=vertical_inputs["governor_v_gov_b_z"],
                    v_final_z=vertical_inputs["governor_v_final_b_z"],
                    station_drift=_get_optional_tensor(
                        step_td,
                        optional_candidates["station_keeping_drift"],
                    ),
                    command_preservation_ratio=_get_optional_tensor(
                        step_td,
                        optional_candidates["command_preservation_ratio"],
                    ),
                    command_amplification_vertical=_get_optional_tensor(
                        step_td,
                        optional_candidates["command_amplification_vertical"],
                    ),
                    ics_beta=_get_optional_tensor(
                        step_td,
                        optional_candidates["ics_beta"],
                    ),
                    ics_emergency=_get_optional_tensor(
                        step_td,
                        optional_candidates["ics_emergency"],
                    ),
                    v_corr_limit=v_corr_limit,
                )
                for field_name, value in vertical_metrics.items():
                    vertical_diagnostic_accumulators[field_name].add(value)

            _accumulate_r5e1_metrics_from_tensordict(
                r5e1_diagnostic_accumulators,
                step_td,
                optional_candidates,
                cfg,
            )
            _accumulate_r5e_metrics_from_tensordict(
                r5e_diagnostic_accumulators,
                step_td,
                optional_candidates,
                cfg,
            )
            _accumulate_r5g_station_metrics_from_tensordict(
                r5g_station_accumulators,
                step_td,
                optional_candidates,
                cfg,
            )
            _accumulate_r5g_downward_metrics_from_tensordict(
                r5g_downward_accumulators,
                step_td,
                optional_candidates,
            )
            _accumulate_r5h_metrics_from_tensordict(
                r5h_diagnostic_accumulators,
                step_td,
                optional_candidates,
                cfg,
            )
            r5e1_latency_tracker.add_step(step_td, optional_candidates, recorded)
            r5g_termination_tracker.add_step(step_td, optional_candidates, recorded)
            r5h_collision_tracker.add_step(step_td, optional_candidates, recorded)
            r5e2_collision_tracker.add_step(step_td, optional_candidates, recorded)
            r5e3_braking_tracker.add_step(step_td, optional_candidates, recorded)

            reward = _get_optional_tensor(step_td, [("next", "agents", "reward")])
            if reward is not None:
                reward_accumulator.add(reward)
                finite_reward = reward.detach().float().cpu().reshape(-1)
                finite_reward = finite_reward[torch.isfinite(finite_reward)]
                if finite_reward.numel() > 0:
                    reward_sum += float(finite_reward.sum().item())

            next_stats = step_td["next", "stats"]
            last_stats = {
                key: value.detach().cpu()
                for key, value in next_stats.items()
            }
            done = step_td["next", "done"].squeeze(-1).bool()
            newly_done = done & ~recorded
            if newly_done.any():
                for key, value in next_stats.items():
                    value_cpu = value.detach().cpu()
                    if key not in first_episode_stats:
                        first_episode_stats[key] = torch.zeros_like(value_cpu)
                    first_episode_stats[key][newly_done.cpu()] = value_cpu[newly_done.cpu()]
                r5e1_latency_tracker.flush(newly_done, next_stats)
                r5g_termination_tracker.flush(newly_done, next_stats)
                r5h_collision_tracker.flush(newly_done, next_stats)
                r5e2_collision_tracker.flush(newly_done, next_stats)
                r5e3_braking_tracker.flush(newly_done, next_stats)
                recorded |= newly_done
            if recorded.all():
                break

    if not recorded.all() and last_stats is not None:
        missing = ~recorded
        for key, value_cpu in last_stats.items():
            if key not in first_episode_stats:
                first_episode_stats[key] = torch.zeros_like(value_cpu)
            first_episode_stats[key][missing.cpu()] = value_cpu[missing.cpu()]
        r5e1_latency_tracker.flush(missing, last_stats)
        r5g_termination_tracker.flush(missing, last_stats)
        r5h_collision_tracker.flush(missing, last_stats)
        r5e2_collision_tracker.flush(missing, last_stats)
        r5e3_braking_tracker.flush(missing, last_stats)

    env.enable_render(not cfg.headless)
    env.reset()

    info = {
        "eval/stats." + key: torch.mean(value.float()).item()
        for key, value in first_episode_stats.items()
    }

    if render_callback is not None:
        info["recording"] = wandb.Video(
            render_callback.get_video_array(axes="t c h w"),
            fps=0.5 / (cfg.sim.dt * cfg.sim.substeps),
            format="mp4",
        )

    env.train()

    if not return_summary:
        return info

    summary = {
        key: _json_safe_scalar(value)
        for key, value in info.items()
        if key != "recording"
    }
    absent_fields = []
    for field_name, accumulator in diagnostic_accumulators.items():
        if accumulator.count == 0:
            absent_fields.append(field_name)
            continue
        summary[f"eval/diagnostics.{field_name}"] = accumulator.summary()
    for field_name, accumulator in vertical_diagnostic_accumulators.items():
        if accumulator.count > 0:
            summary[f"eval/diagnostics.{field_name}"] = accumulator.summary()
    _add_r5e1_diagnostic_summaries(summary, r5e1_diagnostic_accumulators)
    r5e1_latency_tracker.add_diagnostic_summaries(summary)
    _add_r5e_diagnostic_summaries(summary, r5e_diagnostic_accumulators)
    _add_r5g_diagnostic_summaries(
        summary,
        r5g_station_accumulators,
        r5g_downward_accumulators,
    )
    _add_r5h_diagnostic_summaries(summary, r5h_diagnostic_accumulators)
    _add_r5e2_step_handbook_summary(summary, diagnostic_accumulators)
    _add_r5e3_step_handbook_summary(summary, diagnostic_accumulators)
    actual_error = diagnostic_accumulators["tracking_actual_error_sq"].mean()
    if actual_error is not None:
        summary["eval/handbook.tracking_rmse_actual_body_vs_v_cmd"] = float(
            math.sqrt(max(actual_error, 0.0))
        )
    proxy_error = diagnostic_accumulators["tracking_proxy_error_sq"].mean()
    if proxy_error is not None:
        summary["eval/handbook.tracking_rmse_v_final_body_vs_v_cmd"] = float(
            math.sqrt(max(proxy_error, 0.0))
        )
    preservation = diagnostic_accumulators["command_preservation_ratio"].mean()
    if preservation is not None:
        summary["eval/handbook.command_preservation_ratio"] = float(preservation)
    null_command_speed = diagnostic_accumulators["null_command_speed"].mean()
    if null_command_speed is not None:
        summary["eval/handbook.null_command_speed_mean"] = float(null_command_speed)
    null_command_output = diagnostic_accumulators["null_command_output_speed"].mean()
    if null_command_output is not None:
        summary["eval/handbook.null_command_output_speed_mean"] = float(null_command_output)
    command_amplification = diagnostic_accumulators["command_amplification"].mean()
    if command_amplification is not None:
        summary["eval/handbook.command_amplification_mean"] = float(command_amplification)
    command_amplification_active = diagnostic_accumulators["command_amplification_active"].mean()
    if command_amplification_active is not None:
        summary["eval/handbook.command_amplification_rate"] = float(command_amplification_active)
    command_amplification_horizontal = diagnostic_accumulators[
        "command_amplification_horizontal"
    ].mean()
    if command_amplification_horizontal is not None:
        summary["eval/handbook.command_amplification_horizontal_mean"] = float(
            command_amplification_horizontal
        )
    command_amplification_horizontal_active = diagnostic_accumulators[
        "command_amplification_horizontal_active"
    ].mean()
    if command_amplification_horizontal_active is not None:
        summary["eval/handbook.command_amplification_horizontal_rate"] = float(
            command_amplification_horizontal_active
        )
    command_amplification_vertical = diagnostic_accumulators[
        "command_amplification_vertical"
    ].mean()
    if command_amplification_vertical is not None:
        summary["eval/handbook.command_amplification_vertical_mean"] = float(
            command_amplification_vertical
        )
    command_amplification_vertical_active = diagnostic_accumulators[
        "command_amplification_vertical_active"
    ].mean()
    if command_amplification_vertical_active is not None:
        summary["eval/handbook.command_amplification_vertical_rate"] = float(
            command_amplification_vertical_active
        )
    height = diagnostic_accumulators["height_world_z"]
    height_mean = height.mean()
    if height_mean is not None:
        height_summary = height.summary()
        summary["eval/handbook.height_world_z_mean"] = float(height_mean)
        summary["eval/handbook.height_world_z_p05"] = float(height.quantile(0.05))
        summary["eval/handbook.height_world_z_p95"] = float(height.quantile(0.95))
        summary["eval/handbook.height_world_z_min"] = float(
            height_summary.get("min", height_mean)
        )
        summary["eval/handbook.height_world_z_max"] = float(
            height_summary.get("max", height_mean)
        )
    floor_violation = diagnostic_accumulators["height_floor_violation"]
    floor_violation_mean = floor_violation.mean()
    if floor_violation_mean is not None:
        floor_summary = floor_violation.summary()
        summary["eval/handbook.height_floor_violation_mean"] = float(
            floor_violation_mean
        )
        summary["eval/handbook.height_floor_violation_p95"] = float(
            floor_violation.quantile(0.95)
        )
        summary["eval/handbook.height_floor_violation_max"] = float(
            floor_summary.get("max", floor_violation_mean)
        )
    ceiling_violation = diagnostic_accumulators["height_ceiling_violation"]
    ceiling_violation_mean = ceiling_violation.mean()
    if ceiling_violation_mean is not None:
        ceiling_summary = ceiling_violation.summary()
        summary["eval/handbook.height_ceiling_violation_mean"] = float(
            ceiling_violation_mean
        )
        summary["eval/handbook.height_ceiling_violation_p95"] = float(
            ceiling_violation.quantile(0.95)
        )
        summary["eval/handbook.height_ceiling_violation_max"] = float(
            ceiling_summary.get("max", ceiling_violation_mean)
        )
    ceiling_margin = diagnostic_accumulators["height_ceiling_margin"]
    ceiling_margin_mean = ceiling_margin.mean()
    if ceiling_margin_mean is not None:
        margin_summary = ceiling_margin.summary()
        summary["eval/handbook.height_ceiling_margin_mean"] = float(
            ceiling_margin_mean
        )
        summary["eval/handbook.height_ceiling_margin_p05"] = float(
            ceiling_margin.quantile(0.05)
        )
        summary["eval/handbook.height_ceiling_margin_min"] = float(
            margin_summary.get("min", ceiling_margin_mean)
        )
    for label, fraction in _categorical_fractions(
        diagnostic_accumulators["command_mode_code"],
        {
            0: "normal",
            1: "aggressive",
            2: "adversarial",
            3: "oscillation",
            4: "recovery",
        },
    ).items():
        summary[f"eval/handbook.command_mode_fraction.{label}"] = fraction
    station_drift = diagnostic_accumulators["station_keeping_drift"]
    station_drift_mean = station_drift.mean()
    if station_drift_mean is not None:
        station_drift_summary = station_drift.summary()
        summary["eval/handbook.station_keeping_drift_mean"] = float(station_drift_mean)
        summary["eval/handbook.station_keeping_drift_max"] = float(
            station_drift_summary.get("max", station_drift_mean)
        )
        station_drift_p95 = station_drift.quantile(0.95)
        if station_drift_p95 is not None:
            summary["eval/handbook.station_keeping_drift_p95"] = float(station_drift_p95)
    anchor_active = diagnostic_accumulators["anchor_active"].mean()
    if anchor_active is not None:
        summary["eval/handbook.anchor_active_fraction"] = float(anchor_active)
    anchor_error_mean = diagnostic_accumulators["anchor_error_mean"].mean()
    if anchor_error_mean is not None:
        summary["eval/handbook.anchor_error_mean"] = float(anchor_error_mean)
    anchor_error_max = diagnostic_accumulators["anchor_error_max"].summary().get("max")
    if anchor_error_max is not None:
        summary["eval/handbook.anchor_error_max"] = float(anchor_error_max)
    anchor_loss = diagnostic_accumulators["anchor_loss"].mean()
    if anchor_loss is not None:
        summary["eval/handbook.anchor_loss"] = float(anchor_loss)
    clearance = diagnostic_accumulators["safety_min_clearance"]
    clearance_mean = clearance.mean()
    if clearance_mean is not None:
        summary["eval/handbook.safety_min_clearance_mean"] = float(clearance_mean)
        summary["eval/handbook.safety_min_clearance_p05"] = float(clearance.quantile(0.05))
    collision_rate = diagnostic_accumulators["safety_collision"].mean()
    if collision_rate is not None:
        summary["eval/handbook.safety_collision_rate"] = float(collision_rate)
    beta_mean = diagnostic_accumulators["ics_beta"].mean()
    if beta_mean is not None:
        summary["eval/handbook.ics_beta_mean"] = float(beta_mean)
    intervention_frequency = diagnostic_accumulators["ics_intervention"].mean()
    if intervention_frequency is not None:
        summary["eval/handbook.ics_intervention_frequency"] = float(intervention_frequency)
    emergency_rate = diagnostic_accumulators["ics_emergency"].mean()
    if emergency_rate is not None:
        summary["eval/handbook.ics_emergency_rate"] = float(emergency_rate)
    violation_rate = diagnostic_accumulators["ics_violation"].mean()
    if violation_rate is not None:
        summary["eval/handbook.ics_violation_rate"] = float(violation_rate)
    _add_r5e1_handbook_summary(summary, r5e1_diagnostic_accumulators)
    r5e1_latency_tracker.add_summaries(summary)
    _add_r5e_handbook_summary(summary, r5e_diagnostic_accumulators)
    _add_r5g_handbook_summary(
        summary,
        r5g_station_accumulators,
        r5g_downward_accumulators,
    )
    _add_r5h_handbook_summary(summary, r5h_diagnostic_accumulators)
    r5g_termination_tracker.add_summaries(summary)
    r5h_collision_tracker.add_summaries(summary)
    r5e2_collision_tracker.add_summaries(summary)
    r5e3_braking_tracker.add_summaries(summary)
    vertical_corr = vertical_diagnostic_accumulators["vertical_corr_z"].mean()
    if vertical_corr is not None:
        summary["eval/handbook.vertical_v_corr_limit"] = float(v_corr_limit)
        summary["eval/handbook.vertical_corr_z_mean"] = float(vertical_corr)
        summary["eval/handbook.vertical_corr_z_abs_mean"] = float(
            vertical_diagnostic_accumulators["vertical_corr_z_abs"].mean()
        )
        summary["eval/handbook.vertical_corr_z_positive_fraction"] = float(
            vertical_diagnostic_accumulators["vertical_corr_z_positive"].mean()
        )
        summary["eval/handbook.vertical_corr_z_negative_fraction"] = float(
            vertical_diagnostic_accumulators["vertical_corr_z_negative"].mean()
        )
        summary["eval/handbook.vertical_corr_z_saturation_rate"] = float(
            vertical_diagnostic_accumulators["vertical_corr_z_saturated"].mean()
        )
        summary["eval/handbook.vertical_gov_minus_cmd_z_abs_mean"] = float(
            vertical_diagnostic_accumulators["vertical_gov_minus_cmd_z_abs"].mean()
        )
        summary["eval/handbook.vertical_final_minus_cmd_z_abs_mean"] = float(
            vertical_diagnostic_accumulators["vertical_final_minus_cmd_z_abs"].mean()
        )
        summary["eval/handbook.vertical_ics_delta_z_abs_mean"] = float(
            vertical_diagnostic_accumulators["vertical_ics_delta_z_abs"].mean()
        )
        summary["eval/handbook.vertical_corr_reinforcing_fraction"] = float(
            vertical_diagnostic_accumulators["vertical_corr_reinforces_command"].mean()
        )
        summary["eval/handbook.vertical_corr_opposing_fraction"] = float(
            vertical_diagnostic_accumulators["vertical_corr_opposes_command"].mean()
        )
    vertical_null_count = vertical_diagnostic_accumulators["vertical_command_null"]
    vertical_null_rate = _masked_sum_mean(
        vertical_diagnostic_accumulators["vertical_null_corr_active"],
        vertical_null_count,
    )
    if vertical_null_rate is not None:
        summary["eval/handbook.vertical_null_corr_active_rate"] = float(
            vertical_null_rate
        )
        summary["eval/handbook.vertical_null_corr_abs_mean"] = float(
            _masked_sum_mean(
                vertical_diagnostic_accumulators["vertical_null_corr_abs"],
                vertical_null_count,
            )
        )
    null_active = vertical_diagnostic_accumulators["vertical_null_corr_active"]
    null_station_drift = _masked_sum_mean(
        vertical_diagnostic_accumulators["vertical_null_station_drift_when_corr_active"],
        null_active,
    )
    if null_station_drift is not None:
        summary[
            "eval/handbook.vertical_null_station_drift_mean_when_corr_active"
        ] = float(null_station_drift)
    vertical_active_count = vertical_diagnostic_accumulators["vertical_command_active"]
    vertical_tracking_rate = _masked_sum_mean(
        vertical_diagnostic_accumulators["vertical_tracking_corr_active"],
        vertical_active_count,
    )
    if vertical_tracking_rate is not None:
        summary["eval/handbook.vertical_tracking_corr_active_rate"] = float(
            vertical_tracking_rate
        )
    tracking_active = vertical_diagnostic_accumulators["vertical_tracking_corr_active"]
    tracking_amplification = _masked_sum_mean(
        vertical_diagnostic_accumulators[
            "vertical_tracking_amplification_when_corr_active"
        ],
        tracking_active,
    )
    if tracking_amplification is not None:
        summary[
            "eval/handbook.vertical_tracking_amplification_mean_when_corr_active"
        ] = float(tracking_amplification)
    tracking_preservation = _masked_sum_mean(
        vertical_diagnostic_accumulators[
            "vertical_tracking_preservation_when_corr_active"
        ],
        tracking_active,
    )
    if tracking_preservation is not None:
        summary[
            "eval/handbook.vertical_tracking_preservation_mean_when_corr_active"
        ] = float(tracking_preservation)
    for field_name, handbook_name in (
        ("observability_valid_fraction", "observability_valid_fraction_mean"),
        ("observability_weighted_valid_fraction", "observability_weighted_valid_fraction_mean"),
        ("observability_rank", "observability_rank_mean"),
        ("observability_sigma_min", "observability_sigma_min_mean"),
        ("observability_sigma_max", "observability_sigma_max_mean"),
        ("observability_condition_number", "observability_condition_number_mean"),
        ("observability_score", "observability_score_mean"),
        ("observability_drift_projection", "observability_drift_projection_mean"),
        ("observability_drift_norm", "observability_drift_norm_mean"),
    ):
        value = diagnostic_accumulators[field_name].mean()
        if value is not None:
            summary[f"eval/handbook.{handbook_name}"] = float(value)
    observability_is_proxy = diagnostic_accumulators["observability_is_proxy"].mean()
    if observability_is_proxy is not None:
        summary["eval/handbook.observability_is_proxy"] = float(observability_is_proxy)
    observability_mode_code = diagnostic_accumulators["observability_mode_code"].mean()
    if observability_mode_code is not None:
        summary["eval/handbook.observability_mode_code_mean"] = float(observability_mode_code)
    for stats_key, handbook_key in (
        ("terminated_below_bound", "eval/handbook.termination_below_bound"),
        ("terminated_above_bound", "eval/handbook.termination_above_bound"),
        ("terminated_collision", "eval/handbook.termination_collision"),
        ("truncated_timeout", "eval/handbook.termination_timeout"),
    ):
        eval_key = f"eval/stats.{stats_key}"
        if eval_key in summary:
            summary[handbook_key] = summary[eval_key]
    if reward_accumulator.count > 0:
        reward_summary = reward_accumulator.summary()
        reward_summary["sum"] = float(reward_sum)
        summary["eval/reward"] = reward_summary
    if absent_fields:
        summary["absent_optional_fields"] = absent_fields
    summary["eval/episodes_recorded"] = int(recorded.sum().item())
    summary["eval/episodes_expected"] = num_envs

    return info, summary


@torch.no_grad()
def evaluate(
    env,
    policy,
    cfg,
    seed: int=0, 
    exploration_type: ExplorationType=ExplorationType.MEAN,
    return_summary: bool=False,
    streaming: bool=False,
    record_video: bool=True,
):
    """
    评估函数：测试训练好的策略
    
    功能：
    1. 运行完整的 episode
    2. 使用确定性策略（或随机策略）
    3. 录制视频
    4. 统计 tracking、safety、ICS、termination 等指标
    
    参数:
        env: 环境
        policy: 策略网络
        cfg: 配置
        seed: 随机种子
        exploration_type: 探索类型
            - MEAN: 确定性（取均值）
            - RANDOM: 随机采样
    
    返回:
        dict: 评估统计信息
            - eval/stats.return: 平均回报
            - eval/handbook.*: instinctRL command-governor metrics
            - eval/stats.legacy_reach_goal: legacy NavRL diagnostic only
            - eval/stats.collision: 碰撞率
            - recording: WandB 视频对象
    """
    if streaming:
        return _evaluate_streaming(
            env=env,
            policy=policy,
            cfg=cfg,
            seed=seed,
            exploration_type=exploration_type,
            return_summary=return_summary,
            record_video=record_video,
        )

    # 开启渲染（用于录制视频）
    env.enable_render(True)
    env.eval()  # 评估模式
    env.set_seed(seed)

    # 视频录制回调（每 2 步保存一帧）
    render_callback = RenderCallback(interval=2)
    
    # 设置探索类型并运行 rollout
    with set_exploration_type(exploration_type):
        trajs = env.rollout(
            max_steps=env.max_episode_length,  # 最大步数
            policy=policy,                      # 策略
            callback=render_callback,           # 录制视频
            auto_reset=True,                    # 自动重置环境
            break_when_any_done=False,          # 不因单个环境完成而中断
            return_contiguous=False,
        )
    
    # 恢复原始渲染设置
    env.enable_render(not cfg.headless)
    env.reset()
    
    # ============================================
    # 提取每个环境的第一个 episode 的统计信息
    # ============================================
    done = trajs.get(("next", "done")) 
    first_done = torch.argmax(done.long(), dim=1).cpu()  # 第一次 done 的索引

    def take_first_episode(tensor: torch.Tensor):
        """提取第一个 episode 的数据"""
        indices = first_done.reshape(first_done.shape+(1,)*(tensor.ndim-2))
        return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

    # 提取统计信息
    traj_stats = {
        k: take_first_episode(v)
        for k, v in trajs[("next", "stats")].cpu().items()
    }

    # 计算平均值
    info = {
        "eval/stats." + k: torch.mean(v.float()).item() 
        for k, v in traj_stats.items()
    }

    # 添加视频到 WandB
    # fps 计算：0.5 是因为 RenderCallback 的 interval=2
    info["recording"] = wandb.Video(
        render_callback.get_video_array(axes="t c h w"), 
        fps=0.5 / (cfg.sim.dt * cfg.sim.substeps), 
        format="mp4"
    )
    
    env.train()  # 恢复训练模式

    if return_summary:
        return info, _json_safe_eval_summary(info, trajs, cfg)

    return info

# ============================================
# 坐标变换函数
# ============================================

def vec_to_new_frame(vec, goal_direction):
    """
    将向量从世界坐标系转换到目标方向坐标系
    
    为什么需要坐标变换？
    - 策略网络在目标方向坐标系下更容易学习
    - 例如："向前飞"总是在 x 方向，无论世界坐标如何
    
    坐标系定义：
    - x 轴：指向目标方向（水平投影）
    - y 轴：垂直于 x 和 z
    - z 轴：竖直向上 [0, 0, 1]
    
    参数:
        vec: 世界坐标系下的向量 [batch, 3] 或 [batch, N, 3]
        goal_direction: 目标方向 [batch, 3]
    
    返回:
        vec_new: 目标坐标系下的向量
    """
    if (len(vec.size()) == 1):
        vec = vec.unsqueeze(0)

    # 构建目标坐标系
    # x 轴：目标方向（归一化）
    goal_direction_x = goal_direction / goal_direction.norm(dim=-1, keepdim=True)
    z_direction = torch.tensor([0, 0, 1.], device=vec.device)
    
    # y 轴：z × x（右手定则）
    goal_direction_y = torch.cross(z_direction.expand_as(goal_direction_x), goal_direction_x)
    goal_direction_y /= goal_direction_y.norm(dim=-1, keepdim=True)
    
    # z 轴：x × y
    goal_direction_z = torch.cross(goal_direction_x, goal_direction_y)
    goal_direction_z /= goal_direction_z.norm(dim=-1, keepdim=True)

    # 计算向量在新坐标系下的坐标
    # 原理：v_new = R^T * v，其中 R = [x, y, z] 是旋转矩阵
    n = vec.size(0)
    if len(vec.size()) == 3:
        vec_x_new = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_x.view(n, 3, 1)) 
        vec_y_new = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_y.view(n, 3, 1))
        vec_z_new = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_z.view(n, 3, 1))
    else:
        vec_x_new = torch.bmm(vec.view(n, 1, 3), goal_direction_x.view(n, 3, 1))
        vec_y_new = torch.bmm(vec.view(n, 1, 3), goal_direction_y.view(n, 3, 1))
        vec_z_new = torch.bmm(vec.view(n, 1, 3), goal_direction_z.view(n, 3, 1))

    vec_new = torch.cat((vec_x_new, vec_y_new, vec_z_new), dim=-1)
    return vec_new


def vec_to_world(vec, goal_direction):
    """
    将向量从目标方向坐标系转换到世界坐标系
    
    这是 vec_to_new_frame 的逆变换。
    
    用途：
    - 策略网络输出目标坐标系下的速度
    - 需要转换到世界坐标系才能应用到无人机
    
    参数:
        vec: 目标坐标系下的向量 [batch, 3]
        goal_direction: 目标方向 [batch, 3]
    
    返回:
        world_frame_vel: 世界坐标系下的向量
    """
    # 世界坐标系的 x 轴方向 [1, 0, 0]
    world_dir = torch.tensor([1., 0, 0], device=vec.device).expand_as(goal_direction)
    
    # 计算世界坐标系在目标坐标系下的表示
    world_frame_new = vec_to_new_frame(world_dir, goal_direction)

    # 将目标坐标系的向量转换到世界坐标系
    world_frame_vel = vec_to_new_frame(vec, world_frame_new)
    return world_frame_vel


def construct_input(start, end):
    """
    构造 USD 路径通配符
    
    例如：construct_input(0, 3) -> "(0|1|2)"
    用于匹配多个 USD 对象：/World/Origin(0|1|2)/Cuboid
    
    参数:
        start: 起始索引
        end: 结束索引（不包含）
    
    返回:
        str: 通配符字符串
    """
    input = []
    for n in range(start, end):
        input.append(f"{n}")
    return "(" + "|".join(input) + ")"
