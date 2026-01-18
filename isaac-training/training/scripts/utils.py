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

import torch
import torch.nn as nn
import wandb
import numpy as np
from typing import Iterable, Union
from tensordict.tensordict import TensorDict
from omni_drones.utils.torchrl import RenderCallback
from torchrl.envs.utils import ExplorationType, set_exploration_type

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
        self.register_buffer("running_mean", torch.zeros(input_shape))
        self.register_buffer("running_mean_sq", torch.zeros(input_shape))
        self.register_buffer("debiasing_term", torch.tensor(0.0))

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
        dim = tuple(range(input_vector.dim() - len(self.input_shape)))
        batch_mean = input_vector.mean(dim=dim)
        batch_sq_mean = (input_vector**2).mean(dim=dim)

        weight = self.beta  # 滑动平均权重

        # 指数移动平均：new = weight * old + (1 - weight) * new_sample
        self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
        self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
        self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

    def normalize(self, input_vector: torch.Tensor):
        """归一化：(x - mean) / std"""
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        mean, var = self.running_mean_var()
        out = (input_vector - mean) / torch.sqrt(var)
        return out

    def denormalize(self, input_vector: torch.Tensor):
        """反归一化：x * std + mean"""
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        mean, var = self.running_mean_var()
        out = input_vector * torch.sqrt(var) + mean
        return out

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
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.alpha_layer = nn.LazyLinear(action_dim)  # 输出 α
        self.beta_layer = nn.LazyLinear(action_dim)   # 输出 β
        # Softplus: 将 (-∞, +∞) 映射到 (0, +∞)，确保 α, β > 0
        self.alpha_softplus = nn.Softplus()
        self.beta_softplus = nn.Softplus()
    
    def forward(self, features: torch.Tensor):
        """
        前向传播
        
        参数:
            features: 特征向量
        
        返回:
            alpha: Beta 分布参数 α (> 1)
            beta: Beta 分布参数 β (> 1)
        """
        # 加 1 确保 α, β > 1，避免退化分布
        alpha = 1. + self.alpha_softplus(self.alpha_layer(features)) + 1e-6
        beta = 1. + self.beta_softplus(self.beta_layer(features)) + 1e-6
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
    
    # 随机打乱索引并分成 num_minibatches 组
    perm = torch.randperm(
        (tensordict.shape[0] // num_minibatches) * num_minibatches,
        device=tensordict.device,
    ).reshape(num_minibatches, -1)
    
    # 逐个返回 minibatch
    for indices in perm:
        yield tensordict[indices]

@torch.no_grad()
def evaluate(
    env,
    policy,
    cfg,
    seed: int=0, 
    exploration_type: ExplorationType=ExplorationType.MEAN
):
    """
    评估函数：测试训练好的策略
    
    功能：
    1. 运行完整的 episode
    2. 使用确定性策略（或随机策略）
    3. 录制视频
    4. 统计成功率、碰撞率等指标
    
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
            - eval/stats.reach_goal: 成功率
            - eval/stats.collision: 碰撞率
            - recording: WandB 视频对象
    """
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

