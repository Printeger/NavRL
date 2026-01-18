"""
评估脚本 (Evaluation Script)
================================
用于加载训练好的模型，在仿真环境中测试其性能。
与 train.py 的主要区别：
  1. 不进行模型训练（policy.train() 被注释掉）
  2. 加载已有的 checkpoint
  3. 可以开启图形界面观察无人机的实际飞行表现
"""

import argparse
import os
import hydra
import datetime
import wandb
import torch
from omegaconf import DictConfig, OmegaConf
from omni.isaac.kit import SimulationApp
from ppo import PPO
from omni_drones.controllers import LeePositionController
from omni_drones.utils.torchrl.transforms import VelController, ravel_composite
from omni_drones.utils.torchrl import SyncDataCollector, EpisodeStats
from torchrl.envs.transforms import TransformedEnv, Compose
from utils import evaluate
from torchrl.envs.utils import ExplorationType


FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")

@hydra.main(config_path=FILE_PATH, config_name="eval", version_base=None)
def main(cfg):
    """
    主函数：评估训练好的导航模型
    
    流程：
    1. 启动仿真环境
    2. 初始化日志记录器（WandB）
    3. 创建环境和策略
    4. 加载训练好的模型权重
    5. 运行评估循环（不训练，只测试）
    """
    
    # ============================================
    # 第 1 步：启动 Isaac Sim 仿真应用
    # ============================================
    # headless: True=无图形界面(快), False=显示3D场景(慢但可观察)
    # anti_aliasing: 抗锯齿等级，提高渲染质量
    sim_app = SimulationApp({"headless": cfg.headless, "anti_aliasing": 1})

    # ============================================
    # 第 2 步：初始化 WandB 日志记录器
    # ============================================
    # 将 Hydra 的 DictConfig 转换为普通字典，避免序列化错误
    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    
    if (cfg.wandb.run_id is None):
        # 新建一个评估运行（run）
        run = wandb.init(
            project=cfg.wandb.project,
            name=f"{cfg.wandb.name}/eval_{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
            entity=cfg.wandb.entity,
            config=wandb_config,
            mode=cfg.wandb.mode,  # offline 或 online
            id=wandb.util.generate_id(),
        )
    else:
        # 恢复之前的运行（用于继续中断的评估）
        run = wandb.init(
            project=cfg.wandb.project,
            name=f"{cfg.wandb.name}/eval_{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
            entity=cfg.wandb.entity,
            config=wandb_config,
            mode=cfg.wandb.mode,
            id=cfg.wandb.run_id,
            resume="must"
        )

    # ============================================
    # 第 3 步：创建导航环境
    # ============================================
    from env import NavigationEnv
    # NavigationEnv 包含：
    #   - 无人机模型（Hummingbird）
    #   - LiDAR 传感器
    #   - 静态/动态障碍物
    #   - 奖励函数和终止条件
    env = NavigationEnv(cfg)

    # ============================================
    # 第 4 步：包装环境（添加控制器）
    # ============================================
    # 将原始环境包装上一层速度控制器
    # 原因：策略网络输出速度指令，而非直接的电机推力
    transforms = []
    
    # Lee Position Controller: 一个经典的四旋翼姿态控制器
    # 作用：将速度指令转换为电机推力
    controller = LeePositionController(9.81, env.drone.params).to(cfg.device)
    vel_transform = VelController(controller, yaw_control=False)
    transforms.append(vel_transform)
    
    # 创建转换后的环境
    transformed_env = TransformedEnv(env, Compose(*transforms)).train()
    transformed_env.set_seed(cfg.seed)
    
    # ============================================
    # 第 5 步：创建策略网络（PPO）
    # ============================================
    # PPO 包含：
    #   - Actor（策略网络）：输入观测 → 输出动作
    #   - Critic（价值网络）：输入观测 → 输出状态价值
    #   - Feature Extractor：处理 LiDAR 数据和状态信息
    policy = PPO(
        cfg.algo, 
        transformed_env.observation_spec,  # 观测空间定义
        transformed_env.action_spec,       # 动作空间定义
        cfg.device
    )

    # ============================================
    # 第 6 步：加载训练好的模型权重 ⭐ 关键步骤
    # ============================================
    # 这是评估脚本最重要的部分：加载之前训练好的模型
    checkpoint = "./wandb/offline-run-20251209_201022-c9so0klx/files/checkpoint_final.pt"
    print(f"[NavRL]: Loading checkpoint from {checkpoint}")
    policy.load_state_dict(torch.load(checkpoint))
    print("[NavRL]: Checkpoint loaded successfully!")
    
    # ============================================
    # 第 7 步：创建统计收集器
    # ============================================
    # 用于收集每个 episode 的统计信息：
    #   - return: 累积奖励
    #   - reach_goal: 是否到达目标
    #   - collision: 是否碰撞
    #   - episode_len: episode 长度
    episode_stats_keys = [
        k for k in transformed_env.observation_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(episode_stats_keys)

    # ============================================
    # 第 8 步：创建数据收集器
    # ============================================
    # SyncDataCollector: 负责与环境交互，收集数据
    # 注意：这里 exploration_type=RANDOM，但后面 evaluate 函数会用 MEAN
    collector = SyncDataCollector(
        transformed_env,
        policy=policy, 
        frames_per_batch=cfg.env.num_envs * cfg.algo.training_frame_num, 
        total_frames=cfg.max_frame_num,  # 总共运行多少帧
        device=cfg.device,
        return_same_td=True,  # 原地更新，节省内存
        exploration_type=ExplorationType.RANDOM,  # 采样方式（实际评估时会被覆盖）
    )

    # ============================================
    # 第 9 步：评估循环（主循环）
    # ============================================
    # 与 train.py 的区别：
    #   - 不调用 policy.train()（不更新网络参数）
    #   - 每次迭代都进行评估（而非每 N 步）
    for i, data in enumerate(collector):
        # data 包含：observation, action, reward, done 等信息
        
        # 记录基本信息
        info = {
            "env_frames": collector._frames,  # 总帧数
            "rollout_fps": collector._fps      # 运行速度（帧/秒）
        }

        # ===== 训练部分（已注释掉） =====
        # 评估时不需要训练，所以这些代码被注释了
        # train_loss_stats = policy.train(data)
        # info.update(train_loss_stats)
        
        # ===== 评估部分 ⭐ 核心功能 =====
        print("[NavRL]: start evaluating policy at training step: ", i)
        
        # 设置环境为评估模式
        env.eval()
        
        # 运行评估函数（在 utils.py 中定义）
        # evaluate() 会：
        #   1. 运行完整的 episode
        #   2. 使用确定性策略（ExplorationType.MEAN，不随机）
        #   3. 统计成功率、碰撞率等指标
        eval_info = evaluate(
            env=transformed_env, 
            policy=policy,
            seed=cfg.seed, 
            cfg=cfg,
            exploration_type=ExplorationType.MEAN  # 使用平均动作（确定性）
        )
        
        # 恢复训练模式（虽然不训练，但保持一致性）
        env.train()
        env.reset()
        
        # 更新信息字典
        info.update(eval_info)
        print("\n[NavRL]: evaluation done.")
        
        # 将评估结果记录到 WandB
        run.log(info)

        # ===== 模型保存部分（已注释掉） =====
        # 评估时不保存新模型
        # if i % cfg.save_interval == 0:
        #     ckpt_path = os.path.join(run.dir, f"checkpoint_{i}.pt")
        #     torch.save(policy.state_dict(), ckpt_path)

    # ============================================
    # 第 10 步：清理和关闭
    # ============================================
    # 评估结束，关闭日志和仿真器
    wandb.finish()
    sim_app.close()

if __name__ == "__main__":
    main()
