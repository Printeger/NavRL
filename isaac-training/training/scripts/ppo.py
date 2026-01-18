"""
PPO (Proximal Policy Optimization) 算法实现
============================================
PPO 是一种流行的策略梯度强化学习算法，特点是训练稳定且高效。

核心组件：
1. Feature Extractor（特征提取器）: 
   - CNN 处理 LiDAR 点云数据
   - MLP 处理动态障碍物信息
   - 拼接所有特征

2. Actor（策略网络）:
   - 输入：提取的特征
   - 输出：动作分布参数（alpha, beta）
   - 使用 Beta 分布（适合有界动作空间）

3. Critic（价值网络）:
   - 输入：提取的特征
   - 输出：状态价值 V(s)
   - 用于计算优势函数

训练流程：
1. 收集经验数据
2. 计算 GAE 优势函数
3. 多轮（epochs）小批量（minibatch）更新
4. 使用 PPO-Clip 目标函数

参考：https://arxiv.org/abs/1707.06347
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict.tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictSequential, TensorDictModule
from einops.layers.torch import Rearrange  # 用于方便地重排张量维度
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors
from utils import ValueNorm, make_mlp, IndependentNormal, Actor, GAE, make_batch, IndependentBeta, BetaActor, vec_to_world


class PPO(TensorDictModuleBase):
    """
    PPO 策略类
    
    参数:
        cfg: 算法配置（来自 ppo.yaml）
        observation_spec: 观测空间规范
        action_spec: 动作空间规范
        device: 运行设备（'cuda' 或 'cpu'）
    """
    def __init__(self, cfg, observation_spec, action_spec, device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        # ============================================
        # 1. LiDAR 特征提取器（CNN）
        # ============================================
        # 输入形状：[batch, 1, 36, 4] (36个水平角度 × 4个垂直角度)
        # 输出：128 维特征向量
        feature_extractor_network = nn.Sequential(
            # 第1层卷积：提取局部特征 [batch, 1, 36, 4] -> [batch, 4, 36, 4]
            nn.LazyConv2d(out_channels=4, kernel_size=[5, 3], padding=[2, 1]), 
            nn.ELU(),  # 激活函数
            
            # 第2层卷积：降采样 [batch, 4, 36, 4] -> [batch, 16, 18, 4]
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 1], padding=[2, 1]), 
            nn.ELU(),
            
            # 第3层卷积：进一步降采样 [batch, 16, 18, 4] -> [batch, 16, 9, 2]
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 2], padding=[2, 1]), 
            nn.ELU(),
            
            # 展平：[batch, 16, 9, 2] -> [batch, 288]
            Rearrange("n c w h -> n (c w h)"),
            
            # 全连接层：[batch, 288] -> [batch, 128]
            nn.LazyLinear(128), 
            nn.LayerNorm(128),  # 层归一化，稳定训练
        ).to(self.device)
        
        # ============================================
        # 2. 动态障碍物特征提取器（MLP）
        # ============================================
        # 输入：最近的 N 个动态障碍物的状态信息
        # 输出：64 维特征向量
        dynamic_obstacle_network = nn.Sequential(
            Rearrange("n c w h -> n (c w h)"),  # 展平
            make_mlp([128, 64])  # 两层 MLP: 输入 -> 128 -> 64
        ).to(self.device)

        # ============================================
        # 3. 完整特征提取器（组合所有输入）
        # ============================================
        # 将三种输入拼接：
        #   - LiDAR 特征 (128维)
        #   - 无人机状态 (8维：距离、方向、速度)
        #   - 动态障碍物特征 (64维)
        # 总共：128 + 8 + 64 = 200维 -> 经过 MLP -> 256维
        self.feature_extractor = TensorDictSequential(
            TensorDictModule(feature_extractor_network, [("agents", "observation", "lidar")], ["_cnn_feature"]),
            TensorDictModule(dynamic_obstacle_network, [("agents", "observation", "dynamic_obstacle")], ["_dynamic_obstacle_feature"]),
            CatTensors(["_cnn_feature", ("agents", "observation", "state"), "_dynamic_obstacle_feature"], "_feature", del_keys=False), 
            TensorDictModule(make_mlp([256, 256]), ["_feature"], ["_feature"]),
        ).to(self.device)

        # ============================================
        # 4. Actor 网络（策略网络）
        # ============================================
        # 输入：256维特征
        # 输出：Beta 分布的参数 (alpha, beta)
        # 动作空间：3维速度指令 [vx, vy, vz]，范围 [0, 1]（后续会缩放）
        self.n_agents, self.action_dim = action_spec.shape
        self.actor = ProbabilisticActor(
            TensorDictModule(BetaActor(self.action_dim), ["_feature"], ["alpha", "beta"]),
            in_keys=["alpha", "beta"],
            out_keys=[("agents", "action_normalized")],  # 输出在 [0, 1] 之间
            distribution_class=IndependentBeta,  # 使用 Beta 分布
            return_log_prob=True  # 返回对数概率（用于计算损失）
        ).to(self.device)

        # ============================================
        # 5. Critic 网络（价值网络）
        # ============================================
        # 输入：256维特征
        # 输出：状态价值 V(s)（1个标量）
        self.critic = TensorDictModule(
            nn.LazyLinear(1), ["_feature"], ["state_value"] 
        ).to(self.device)
        
        # 价值归一化：稳定训练，避免价值爆炸
        self.value_norm = ValueNorm(1).to(self.device)

        # ============================================
        # 6. 损失函数和优化器
        # ============================================
        # GAE (Generalized Advantage Estimation): 计算优势函数
        # gamma=0.99: 折扣因子
        # lambda=0.95: GAE 参数，权衡偏差和方差
        self.gae = GAE(0.99, 0.95)
        
        # Huber Loss: 结合 L1 和 L2 损失的优点，对异常值鲁棒
        self.critic_loss_fn = nn.HuberLoss(delta=10)

        # 为三个组件分别创建优化器
        self.feature_extractor_optim = torch.optim.Adam(
            self.feature_extractor.parameters(), 
            lr=cfg.feature_extractor.learning_rate
        )
        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(), 
            lr=cfg.actor.learning_rate
        )
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), 
            lr=cfg.actor.learning_rate
        )

        # ============================================
        # 7. 初始化网络权重
        # ============================================
        # 使用虚拟输入初始化 LazyModule（自动推断输入维度）
        dummy_input = observation_spec.zero()
        self.__call__(dummy_input)

        # 正交初始化：经典的初始化方法，有助于训练稳定
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)  # 正交初始化权重
                nn.init.constant_(module.bias, 0.)  # 偏置初始化为 0
        self.actor.apply(init_)
        self.critic.apply(init_)

    def __call__(self, tensordict):
        """
        前向传播：给定观测，输出动作和价值
        
        参数:
            tensordict: 包含观测的字典
                - ("agents", "observation", "lidar"): LiDAR 数据
                - ("agents", "observation", "state"): 无人机状态
                - ("agents", "observation", "dynamic_obstacle"): 动态障碍物
        
        返回:
            tensordict: 添加了动作和价值的字典
                - ("agents", "action"): 世界坐标系下的速度指令
                - "state_value": 状态价值
        """
        # 提取特征
        self.feature_extractor(tensordict)
        
        # Actor 前向：生成动作分布并采样
        # 输出：("agents", "action_normalized") ∈ [0, 1]^3
        self.actor(tensordict)
        
        # Critic 前向：评估状态价值
        # 输出："state_value" ∈ R
        self.critic(tensordict)

        # ============================================
        # 坐标变换：局部坐标 -> 世界坐标
        # ============================================
        # 1. 将 [0, 1] 缩放到 [-action_limit, action_limit]
        # 例如：action_limit=2 -> [-2, 2] m/s
        actions = (2 * tensordict["agents", "action_normalized"] * self.cfg.actor.action_limit) - self.cfg.actor.action_limit
        
        # 2. 从目标方向坐标系转换到世界坐标系
        # 原因：策略在目标方向坐标系下输出动作，更容易学习
        actions_world = vec_to_world(actions, tensordict["agents", "observation", "direction"])
        tensordict["agents", "action"] = actions_world
        
        return tensordict

    def train(self, tensordict):
        """
        PPO 训练函数
        
        参数:
            tensordict: 收集的经验数据
                形状：[num_envs, num_frames]
                包含：state, action, reward, next_state, done 等
        
        返回:
            dict: 训练统计信息
                - actor_loss: 策略损失
                - critic_loss: 价值损失
                - entropy: 熵（衡量探索程度）
                - actor_grad_norm: 梯度范数（监控训练稳定性）
        """
        # ============================================
        # 第 1 步：计算下一个状态的价值（用于 TD 误差）
        # ============================================
        next_tensordict = tensordict["next"]
        with torch.no_grad():
            # 使用 vmap 批量处理多个时间步
            next_tensordict = torch.vmap(self.feature_extractor)(next_tensordict)
            next_values = self.critic(next_tensordict)["state_value"]
        
        # 获取奖励和终止标志
        rewards = tensordict["next", "agents", "reward"]  # r_t
        dones = tensordict["next", "terminated"]  # 是否终止

        # ============================================
        # 第 2 步：反归一化价值（恢复真实尺度）
        # ============================================
        values = tensordict["state_value"]  # V(s_t)，在数据收集时已计算
        values = self.value_norm.denormalize(values)
        next_values = self.value_norm.denormalize(next_values)

        # ============================================
        # 第 3 步：计算 GAE 优势函数
        # ============================================
        # GAE 是一种权衡偏差和方差的优势函数估计方法
        # A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
        # 其中 δ_t = r_t + γV(s_{t+1}) - V(s_t)
        adv, ret = self.gae(rewards, dones, values, next_values)
        
        # 标准化优势函数：均值为0，标准差为1
        # 作用：稳定训练，避免梯度爆炸
        adv_mean = adv.mean()
        adv_std = adv.std()
        adv = (adv - adv_mean) / adv_std.clip(1e-7)
        
        # 更新价值归一化统计量
        self.value_norm.update(ret)
        ret = self.value_norm.normalize(ret)  # 归一化回报
        
        # 将优势和回报添加到 tensordict
        tensordict.set("adv", adv)
        tensordict.set("ret", ret)

        # ============================================
        # 第 4 步：多轮小批量更新
        # ============================================
        # PPO 使用 on-policy 数据进行多轮更新
        # 每轮将数据分成多个 minibatch 进行梯度更新
        infos = []
        for epoch in range(self.cfg.training_epoch_num):
            # 将数据随机打乱并分成 minibatch
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                # 对每个 minibatch 执行一次梯度更新
                infos.append(self._update(minibatch))
        
        # 聚合所有更新的统计信息
        infos = torch.stack(infos).to_tensordict()
        infos = infos.apply(torch.mean, batch_size=[])
        
        return {k: v.item() for k, v in infos.items()}    

    
    def _update(self, tensordict):
        """
        单次梯度更新
        
        参数:
            tensordict: 一个 minibatch 的数据
                包含：observation, action, advantage, return, old_log_prob
        
        返回:
            TensorDict: 更新统计信息
        """
        # ============================================
        # 第 1 步：重新计算当前策略下的动作概率
        # ============================================
        self.feature_extractor(tensordict)
        
        # 获取当前策略的动作分布
        action_dist = self.actor.get_dist(tensordict)  # Beta(alpha, beta)
        
        # 计算当前策略下采取该动作的对数概率
        log_probs = action_dist.log_prob(tensordict[("agents", "action_normalized")])

        # ============================================
        # 第 2 步：计算熵损失（鼓励探索）
        # ============================================
        # 熵越大，策略越随机，探索性越强
        # 熵损失：-c * H(π)，最小化负熵 = 最大化熵
        action_entropy = action_dist.entropy()
        entropy_loss = -self.cfg.entropy_loss_coefficient * torch.mean(action_entropy)

        # ============================================
        # 第 3 步：计算 PPO-Clip Actor 损失
        # ============================================
        # PPO 的核心：限制策略更新幅度，避免性能崩溃
        advantage = tensordict["adv"]  # 优势函数 A(s,a)
        
        # 重要性采样比率：π_new(a|s) / π_old(a|s)
        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        
        # PPO-Clip 目标：
        # L^CLIP = E[min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)]
        # 作用：当 ratio 超出 [1-ε, 1+ε] 时，停止更新
        surr1 = advantage * ratio  # 未裁剪的目标
        surr2 = advantage * ratio.clamp(
            1. - self.cfg.actor.clip_ratio, 
            1. + self.cfg.actor.clip_ratio
        )  # 裁剪的目标
        
        # 取两者的最小值（悲观更新）
        # 乘以 action_dim 是为了缩放损失
        actor_loss = -torch.mean(torch.min(surr1, surr2)) * self.action_dim 

        # ============================================
        # 第 4 步：计算 Critic 损失
        # ============================================
        # 目标：让 V(s) 接近真实回报 G_t
        b_value = tensordict["state_value"]  # 旧的价值估计
        ret = tensordict["ret"]  # 真实回报 G_t
        value = self.critic(tensordict)["state_value"]  # 新的价值估计
        
        # Value Clipping：限制价值函数的更新幅度
        # 原因：防止价值函数变化过大，导致训练不稳定
        value_clipped = b_value + (value - b_value).clamp(
            -self.cfg.critic.clip_ratio, 
            self.cfg.critic.clip_ratio
        )
        
        # 计算两种损失，取最大值（悲观更新）
        critic_loss_clipped = self.critic_loss_fn(ret, value_clipped)
        critic_loss_original = self.critic_loss_fn(ret, value)
        critic_loss = torch.max(critic_loss_clipped, critic_loss_original)

        # ============================================
        # 第 5 步：总损失 = 熵损失 + Actor 损失 + Critic 损失
        # ============================================
        loss = entropy_loss + actor_loss + critic_loss

        # ============================================
        # 第 6 步：反向传播和梯度更新
        # ============================================
        self.feature_extractor_optim.zero_grad()
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        loss.backward()  # 计算梯度

        # 梯度裁剪：防止梯度爆炸
        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(
            self.actor.parameters(), max_norm=5.
        )
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), max_norm=5.
        )
        
        # 应用梯度
        self.feature_extractor_optim.step()
        self.actor_optim.step()
        self.critic_optim.step()
        
        # ============================================
        # 第 7 步：计算解释方差（评估 Critic 质量）
        # ============================================
        # Explained Variance = 1 - Var(V - G) / Var(G)
        # 接近 1：Critic 准确；接近 0：Critic 无用
        explained_var = 1 - F.mse_loss(value, ret) / ret.var()
        
        return TensorDict({
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "entropy": entropy_loss,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "explained_var": explained_var
        }, [])