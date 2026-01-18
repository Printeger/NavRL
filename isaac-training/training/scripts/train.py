"""
训练脚本 (Training Script)
===========================
这是整个项目的主入口，负责训练无人机导航的强化学习模型。

主要流程：
1. 启动 Isaac Sim 仿真器
2. 初始化 WandB 日志记录
3. 创建训练环境（地形、障碍物、传感器）
4. 创建 PPO 策略网络
5. 收集交互数据并训练模型
6. 周期性评估和保存模型

作者：NavRL 项目
"""

import argparse
import os
import hydra              # 配置管理框架
import datetime
import wandb              # 实验跟踪工具
import torch
from omegaconf import DictConfig, OmegaConf
from omni.isaac.kit import SimulationApp     # Isaac Sim 应用
from ppo import PPO                          # PPO 算法实现
from omni_drones.controllers import LeePositionController  # 低层控制器
from omni_drones.utils.torchrl.transforms import VelController, ravel_composite
from omni_drones.utils.torchrl import SyncDataCollector, EpisodeStats  # 数据收集器
from torchrl.envs.transforms import TransformedEnv, Compose
from utils import evaluate  # 评估函数
from torchrl.envs.utils import ExplorationType


# ============================================
# 配置文件路径（train.yaml 等）
# ============================================
FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")

@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg):
    """
    主训练函数
    
    参数:
        cfg: Hydra 配置对象，包含所有训练参数
             - cfg.headless: 是否无头模式（不显示GUI）
             - cfg.env.num_envs: 并行环境数量
             - cfg.max_frame_num: 总训练帧数
             - cfg.algo: PPO 算法超参数
             - cfg.sensor: 传感器配置（LiDAR）
    """
    # ============================================
    # 第 1 步：启动 Isaac Sim 仿真器
    # ============================================
    # SimulationApp 是 Isaac Sim 的核心，负责：
    #   - 创建 3D 仿真场景
    #   - 运行物理引擎（PhysX）
    #   - 渲染图形（如果 headless=False）
    sim_app = SimulationApp({"headless": cfg.headless, "anti_aliasing": 1})

    # ============================================
    # 第 2 步：初始化 WandB 实验跟踪
    # ============================================
    # WandB 用于记录和可视化训练过程：
    #   - 损失曲线（actor_loss, critic_loss）
    #   - 训练指标（成功率、碰撞率、回报）
    #   - 视频录制（评估时的无人机飞行）
    
    # 将 Hydra 的 DictConfig 转换为普通字典，避免序列化错误
    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    
    if (cfg.wandb.run_id is None):
        # 新建一个训练运行（run）
        run = wandb.init(
            project=cfg.wandb.project,  # WandB 项目名称
            name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
            entity=cfg.wandb.entity,    # WandB 用户名/团队名
            config=wandb_config,        # 保存所有配置参数
            mode=cfg.wandb.mode,        # "offline" 或 "online"
            id=wandb.util.generate_id(),
        )
    else:
        # 恢复之前中断的训练运行
        run = wandb.init(
            project=cfg.wandb.project,
            name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
            entity=cfg.wandb.entity,
            config=wandb_config,
            mode=cfg.wandb.mode,
            id=cfg.wandb.run_id,
            resume="must"  # 必须恢复之前的运行
        )

    # ============================================
    # 第 3 步：创建导航训练环境
    # ============================================
    # NavigationEnv 包含所有仿真元素：
    #   - 无人机模型（Hummingbird 四旋翼）
    #   - LiDAR 传感器（36×4=144 个测量点）
    #   - 静态障碍物（地形上的障碍物）
    #   - 动态障碍物（移动的立方体和圆柱）
    #   - 奖励函数和终止条件
    from env import NavigationEnv
    env = NavigationEnv(cfg)

    # ============================================
    # 第 4 步：包装环境（添加控制器）
    # ============================================
    # 为什么需要包装？
    # - 策略网络输出：速度指令（vx, vy, vz）
    # - 无人机需要：电机推力（4 个电机的转速）
    # - VelController：将速度指令转换为电机推力
    transforms = []
    
    # Lee Position Controller: 经典的四旋翼控制算法
    # 参数：重力加速度 9.81 m/s², 无人机物理参数
    controller = LeePositionController(9.81, env.drone.params).to(cfg.device)
    vel_transform = VelController(controller, yaw_control=False)  # 不控制偏航角
    transforms.append(vel_transform)
    
    # 应用变换并设置为训练模式
    transformed_env = TransformedEnv(env, Compose(*transforms)).train()
    transformed_env.set_seed(cfg.seed)    
    
    # ============================================
    # 第 5 步：创建 PPO 策略网络
    # ============================================
    # PPO（Proximal Policy Optimization）包含：
    #   1. Feature Extractor: CNN 处理 LiDAR 数据
    #   2. Actor（策略网络）: 输出动作分布
    #   3. Critic（价值网络）: 评估状态价值
    policy = PPO(cfg.algo, transformed_env.observation_spec, transformed_env.action_spec, cfg.device)

    # ============================================
    # 第 6 步：（可选）加载预训练模型
    # ============================================
    # 如果想继续之前的训练，可以取消注释以下代码：
    # checkpoint = "/path/to/checkpoint.pt"
    # policy.load_state_dict(torch.load(checkpoint))
    
    # ============================================
    # 第 7 步：创建统计数据收集器
    # ============================================
    # EpisodeStats 用于跟踪每个 episode 的统计信息：
    #   - return: 累积奖励
    #   - reach_goal: 是否到达目标
    #   - collision: 是否发生碰撞
    #   - episode_len: episode 长度
    episode_stats_keys = [
        k for k in transformed_env.observation_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(episode_stats_keys)

    # ============================================
    # 第 8 步：创建强化学习数据收集器
    # ============================================
    # SyncDataCollector 负责：
    #   1. 让策略与环境交互，收集经验数据
    #   2. 每次收集 frames_per_batch 帧数据
    #   3. 自动重置完成的环境
    collector = SyncDataCollector(
        transformed_env,
        policy=policy, 
        frames_per_batch=cfg.env.num_envs * cfg.algo.training_frame_num,  # 每批数据量
        total_frames=cfg.max_frame_num,      # 总训练帧数（训练停止条件）
        device=cfg.device,
        return_same_td=True,  # 原地更新，节省内存
        exploration_type=ExplorationType.RANDOM,  # 训练时使用随机探索
    )

    # ============================================
    # 第 9 步：主训练循环 🔄
    # ============================================
    # collector 是一个迭代器，每次迭代：
    #   1. 与环境交互收集 frames_per_batch 帧数据
    #   2. 返回一个 TensorDict，包含 (state, action, reward, next_state)
    for i, data in enumerate(collector):
        # data 的结构：
        # {
        #   "agents": {
        #     "observation": {"lidar": [...], "state": [...], ...},
        #     "action": [...],
        #     "reward": [...]
        #   },
        #   "next": {...},  # 下一个状态
        #   "done": [...],
        #   "terminated": [...]
        # }
        
        # -------- 记录基本信息 --------
        info = {
            "env_frames": collector._frames,  # 已训练的总帧数
            "rollout_fps": collector._fps      # 数据收集速度（帧/秒）
        }

        # -------- 训练策略网络 --------
        # policy.train() 执行：
        #   1. 计算 GAE 优势函数
        #   2. 进行多轮（epochs）小批量（minibatch）更新
        #   3. 返回损失统计信息
        train_loss_stats = policy.train(data)
        info.update(train_loss_stats)  # 添加训练损失信息

        # -------- 统计训练 episode 信息 --------
        episode_stats.add(data)
        if len(episode_stats) >= transformed_env.num_envs:
            # 所有环境都至少完成一个 episode，计算平均统计
            stats = {
                "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item() 
                for k, v in episode_stats.pop().items(True, True)
            }
            info.update(stats)

        # -------- 周期性评估策略 --------
        if i % cfg.eval_interval == 0:
            print("[NavRL]: start evaluating policy at training step: ", i)
            
            # 开启渲染（用于录制视频）
            env.enable_render(True)
            env.eval()  # 设置为评估模式
            
            # 运行评估：使用确定性策略（MEAN），不随机探索
            eval_info = evaluate(
                env=transformed_env, 
                policy=policy,
                seed=cfg.seed, 
                cfg=cfg,
                exploration_type=ExplorationType.MEAN  # 确定性动作
            )
            
            # 恢复原来的渲染设置
            env.enable_render(not cfg.headless)
            env.train()  # 恢复训练模式
            env.reset()
            info.update(eval_info)
            print("\n[NavRL]: evaluation done.")
        
        # -------- 记录到 WandB --------
        run.log(info)

        # -------- 周期性保存模型 --------
        if i % cfg.save_interval == 0:
            ckpt_path = os.path.join(run.dir, f"checkpoint_{i}.pt")
            torch.save(policy.state_dict(), ckpt_path)
            print("[NavRL]: model saved at training step: ", i)

    # ============================================
    # 第 10 步：训练完成，保存最终模型
    # ============================================
    ckpt_path = os.path.join(run.dir, "checkpoint_final.pt")
    torch.save(policy.state_dict(), ckpt_path)
    print(f"[NavRL]: Training complete! Final model saved to {ckpt_path}")
    
    # 关闭 WandB 和仿真器
    wandb.finish()
    sim_app.close()

if __name__ == "__main__":
    main()
    