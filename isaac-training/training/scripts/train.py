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
import sys
import hydra              # 配置管理框架
import datetime
import wandb              # 实验跟踪工具
import torch
from omegaconf import DictConfig, OmegaConf
# OmniDrones imports moved inside main() after SimulationApp startup
# to avoid the "loaded before SimulationApp" warning and init issues.


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
    instinct_enabled = bool(getattr(cfg, "instinctRL", None) and cfg.instinctRL.enabled)
    instinct_mode = getattr(cfg.instinctRL, "mode", "smoke") if instinct_enabled else "off"

    if instinct_enabled:
        from instinctRL.audit import check_platform_lock

        print("\n" + "=" * 60, flush=True)
        print(f"[instinctRL] Mode: {instinct_mode}", flush=True)
        print("[instinctRL-A/B] B0/B Observation Smoke Path", flush=True)
        print("=" * 60, flush=True)
        print("[instinctRL-A] Running pre-Isaac platform audit...", flush=True)

        platform_ok, platform_msg = check_platform_lock(cfg)
        print(f"[instinctRL audit] {platform_msg}", flush=True)
        if not platform_ok:
            raise RuntimeError(f"instinctRL platform audit FAILED: {platform_msg}")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "[instinctRL-A] CUDA preflight failed: no CUDA-capable device is "
                "visible. B0 smoke requires Isaac Sim GPU physics/MID360 ray casting."
            )

    # ============================================
    # 第 1 步：启动 Isaac Sim 仿真器
    # ============================================
    # SimulationApp 是 Isaac Sim 的核心，负责：
    #   - 创建 3D 仿真场景
    #   - 运行物理引擎（PhysX）
    #   - 渲染图形（如果 headless=False）
    from omni.isaac.kit import SimulationApp     # Isaac Sim 应用
    app_experience = ""
    if cfg.headless and os.environ.get("EXP_PATH"):
        app_experience = f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.gym.headless.kit'
    sim_app = SimulationApp(
        {"headless": cfg.headless, "anti_aliasing": 1},
        experience=app_experience,
    )
    if app_experience:
        from omni.isaac.core.utils.extensions import enable_extension

        for extension_name in (
            "omni.isaac.debug_draw",
            "omni.syntheticdata",
            "omni.replicator.core",
        ):
            enable_extension(extension_name)
        sim_app.update()

    # Import OmniDrones/TorchRL modules AFTER SimulationApp (Isaac Sim requirement)
    from ppo import PPO
    from omni_drones.controllers import LeePositionController
    from omni_drones.utils.torchrl.transforms import VelController, ravel_composite
    from omni_drones.utils.torchrl import SyncDataCollector, EpisodeStats
    from torchrl.envs.transforms import TransformedEnv, Compose
    from utils import evaluate
    from torchrl.envs.utils import ExplorationType

    # ============================================
    # 第 2 步：初始化 WandB 实验跟踪
    # ============================================
    # WandB 用于记录和可视化训练过程：
    #   - 损失曲线（actor_loss, critic_loss）
    #   - 训练指标（成功率、碰撞率、回报）
    #   - 视频录制（评估时的无人机飞行）
    
    # 将 Hydra 的 DictConfig 转换为普通字典，避免序列化错误
    run = None
    if not instinct_enabled:
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
    # instinctRL-A: Platform audit + B0 smoke test
    # ============================================
    if instinct_enabled:
        from instinctRL.audit import check_actor_input, check_actor_schema, check_action_type
        from instinctRL.governor import MinimalGovernor
        from instinctRL.command_adapter import BodyToWorldVelocityAdapter

        print("[instinctRL-A] Creating B0 governor and body-to-world adapter...", flush=True)

        # Create B0 governor and body→world adapter
        gov_cfg = cfg.algo.instinctRL.governor
        governor = MinimalGovernor(
            v_corr_limit=gov_cfg.v_corr_limit,
            velocity_limit=gov_cfg.velocity_limit,
        ).to(cfg.device)
        adapter = BodyToWorldVelocityAdapter().to(cfg.device)
        ics_attenuator = None
        ics_cfg = getattr(cfg.instinctRL, "ics", None)
        if ics_cfg is not None and getattr(ics_cfg, "enabled", False):
            from instinctRL.ics import ICSConfig, RangeHistoryICSAttenuator

            ics_attenuator = RangeHistoryICSAttenuator(
                ICSConfig.from_namespace(ics_cfg),
                device=cfg.device,
            )
            print("[instinctRL-E] ICS attenuation enabled (brake_mode=zero)", flush=True)
        print(f"[instinctRL-A] Governor: B0 (alpha={gov_cfg.alpha_fixed})", flush=True)
        print(f"[instinctRL-A] Baseline: {cfg.instinctRL.baseline.id}", flush=True)

    # ============================================
    # 第 4 步：包装环境（添加控制器）
    # ============================================
    transforms = []
    controller = LeePositionController(9.81, env.drone.params).to(cfg.device)
    vel_transform = VelController(controller, yaw_control=False)
    transforms.append(vel_transform)
    transformed_env = TransformedEnv(env, Compose(*transforms)).train()
    transformed_env.set_seed(cfg.seed)

    # ============================================
    # instinctRL-A: B0 Smoke Test
    # ============================================
    if instinct_enabled and instinct_mode == "smoke":
        smoke_steps = 500
        print(f"[instinctRL-B] Running {smoke_steps} physics steps...", flush=True)

        try:
            print("[instinctRL-A] Resetting transformed environment...", flush=True)
            td = transformed_env.reset()
            print("[instinctRL-A] Reset complete.", flush=True)
            policy_smoke = PPO(
                cfg.algo,
                transformed_env.observation_spec,
                transformed_env.action_spec,
                cfg.device,
            )
            with torch.no_grad():
                policy_smoke(td.clone())
            print("[instinctRL-B] PPO hybrid forward smoke PASSED.", flush=True)
            for step in range(smoke_steps):
                v_cmd_body = td["info", "v_cmd"]

                # Audit 2: Actor input check (first step only)
                if step == 0:
                    actor_ok, actor_msg = check_actor_input(td)
                    print(f"[instinctRL audit] {actor_msg}", flush=True)
                    if not actor_ok:
                        raise RuntimeError(f"instinctRL actor audit FAILED: {actor_msg}")
                    schema_ok, schema_msg = check_actor_schema(
                        td, cfg.instinctRL.observation.history_len
                    )
                    print(f"[instinctRL audit] {schema_msg}", flush=True)
                    if not schema_ok:
                        raise RuntimeError(f"instinctRL actor schema audit FAILED: {schema_msg}")

                # B0: v_cmd → governor (α=1, v_corr=0) → v_gov
                gov_out = governor(v_cmd_body)
                v_gov_body = gov_out.v_gov
                v_final_body = v_gov_body
                if ics_attenuator is not None:
                    histories = env.get_instinctrl_range_history(copy=False)
                    ics_out = ics_attenuator(
                        histories["range_history"],
                        histories["mask_history"],
                        histories["weight_history"],
                        env._mid360_ray_dirs_b,
                        v_gov_body,
                        dt=env.dt,
                    )
                    v_final_body = ics_out.v_final_b
                    env.record_instinctrl_ics_output(ics_out)
                env.set_prev_issued_action_body(v_final_body)

                # Body → world using privileged drone quaternion at controller boundary.
                # The actor never receives this attitude/pose state.
                drone_quat = td["info", "drone_state"][..., 3:7]
                v_final_world = adapter(v_final_body, drone_quat)

                # Audit 3: Action type check
                if step == 0:
                    action_ok, action_msg = check_action_type(v_final_world, expected_dim=3)
                    print(f"[instinctRL audit] {action_msg}", flush=True)
                    if not action_ok:
                        raise RuntimeError(f"instinctRL action audit FAILED: {action_msg}")

                action = v_final_world.squeeze(1)
                if torch.isnan(action).any():
                    raise RuntimeError(f"[instinctRL-A] NaN in action at step {step}!")

                td[("agents", "action")] = action
                step_td = transformed_env.step(td)

                reward = step_td["next", "agents", "reward"]
                if torch.isnan(reward).any():
                    raise RuntimeError(f"[instinctRL-A] NaN in reward at step {step}!")
                td = step_td["next"]
                if (step + 1) % 50 == 0:
                    print(f"[instinctRL-A] Completed {step + 1}/{smoke_steps} steps.", flush=True)

            # Verify LiDAR
            lidar_range = getattr(env, "lidar_raw_range", None)
            if lidar_range is None or lidar_range.numel() == 0:
                raise RuntimeError("[instinctRL-A] LiDAR raw range missing or empty.")
            valid_ratio = (
                torch.isfinite(lidar_range) & (lidar_range < env.lidar_range)
            ).float().mean().item()
            if valid_ratio <= 0.0:
                raise RuntimeError("[instinctRL-A] LiDAR has no valid returns.")
            print(f"[instinctRL-A] LiDAR raw range: shape={lidar_range.shape}, "
                  f"valid={valid_ratio:.2%}", flush=True)
            print(f"\n[instinctRL-A] B0 Smoke Test PASSED ({smoke_steps} steps).", flush=True)
            print("[instinctRL-B] Observation smoke path PASSED.", flush=True)
            print(
                "[instinctRL-B] Smoke validation complete. Exiting before "
                "SimulationApp.close() to avoid Isaac Kit shutdown segfault.",
                flush=True,
            )
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
        except Exception as exc:
            print(
                f"[instinctRL-A] B0 Smoke Test FAILED: {type(exc).__name__}: {exc}",
                flush=True,
            )
            sim_app.close()
            raise
    elif instinct_enabled and instinct_mode != "train":
        sim_app.close()
        raise ValueError(
            f"Unsupported instinctRL.mode={instinct_mode!r}; expected 'smoke' or 'train'."
        )

    # ============================================
    # 第 5 步：创建 PPO 策略网络（非 instinctRL 模式）
    # ============================================
    policy = PPO(cfg.algo, transformed_env.observation_spec, transformed_env.action_spec, cfg.device)
    if instinct_enabled:
        td = transformed_env.reset()
        actor_ok, actor_msg = check_actor_input(td)
        print(f"[instinctRL audit] {actor_msg}", flush=True)
        if not actor_ok:
            raise RuntimeError(f"instinctRL actor audit FAILED: {actor_msg}")
        schema_ok, schema_msg = check_actor_schema(td, cfg.instinctRL.observation.history_len)
        print(f"[instinctRL audit] {schema_msg}", flush=True)
        if not schema_ok:
            raise RuntimeError(f"instinctRL actor schema audit FAILED: {schema_msg}")
        with torch.no_grad():
            policy(td.clone())
        print("[instinctRL-B] PPO hybrid forward smoke PASSED.", flush=True)

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
    
