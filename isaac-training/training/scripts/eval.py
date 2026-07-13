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
import json
import os
import hydra
import datetime
import wandb
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omni.isaac.kit import SimulationApp


FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")


class InstinctRLEvalPolicy(torch.nn.Module):
    """Eval wrapper matching the learned-governor collector path used in training."""

    def __init__(self, policy, env, adapter, ics_attenuator=None):
        super().__init__()
        self.policy = policy
        self.env = env
        self.adapter = adapter
        self.ics_attenuator = ics_attenuator

    def forward(self, tensordict):
        self.policy(tensordict)
        if "governor_v_gov_b" not in tensordict.keys(True):
            return tensordict

        v_gov_body = tensordict["governor_v_gov_b"]
        v_final_body = v_gov_body
        if self.ics_attenuator is not None:
            histories = self.env.get_instinctrl_range_history(copy=False)
            ics_out = self.ics_attenuator(
                histories["range_history"],
                histories["mask_history"],
                histories["weight_history"],
                self.env._mid360_ray_dirs_b,
                v_gov_body,
                dt=self.env.dt,
            )
            v_final_body = ics_out.v_final_b
            self.env.record_instinctrl_ics_output(ics_out)

        self.env.set_prev_issued_action_body(v_final_body)
        drone_quat = tensordict["info", "drone_state"][..., 3:7]
        v_final_world = self.adapter(v_final_body, drone_quat)
        if v_final_world.dim() == 3 and v_final_world.shape[-2] == 1:
            v_final_world = v_final_world.squeeze(-2)
        tensordict["governor_v_final_b"] = v_final_body
        tensordict["governor_v_final_b_z"] = v_final_body[..., 2:3]
        tensordict["agents", "action"] = v_final_world
        return tensordict


def _resolve_required_checkpoint(cfg):
    checkpoint_path = str(getattr(cfg, "checkpoint_path", "")).strip()
    if not checkpoint_path:
        raise ValueError(
            "cfg.checkpoint_path is required for eval. Pass "
            "checkpoint_path=/absolute/or/repo-relative/checkpoint.pt"
        )
    resolved = to_absolute_path(os.path.expanduser(checkpoint_path))
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"checkpoint_path does not exist: {resolved}")
    return resolved


def _resolve_result_path(cfg, checkpoint_path):
    result_path = str(getattr(cfg, "result_path", "")).strip()
    if result_path:
        resolved = to_absolute_path(os.path.expanduser(result_path))
    else:
        checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        resolved = os.path.join(
            HydraConfig.get().runtime.output_dir,
            f"{checkpoint_name}_eval_summary.json",
        )
    os.makedirs(os.path.dirname(resolved), exist_ok=True)
    return resolved


def _preflight_instinctrl_eval(cfg):
    instinct_enabled = bool(getattr(cfg, "instinctRL", None) and cfg.instinctRL.enabled)
    if not instinct_enabled:
        return False, "off"
    instinct_mode = getattr(cfg.instinctRL, "mode", "smoke")

    from instinctRL.audit import check_platform_lock

    platform_ok, platform_msg = check_platform_lock(cfg)
    print(f"[instinctRL audit] {platform_msg}", flush=True)
    if not platform_ok:
        raise RuntimeError(f"instinctRL platform audit FAILED: {platform_msg}")

    eval_cfg = getattr(cfg.instinctRL, "eval", None)
    require_static = bool(getattr(eval_cfg, "require_static_mid360", False))
    if require_static and int(cfg.env_dyn.num_obstacles) != 0:
        raise ValueError(
            "short diagnostic eval requires env_dyn.num_obstacles=0 until "
            "dynamic obstacles are MID360 RayCaster-visible"
        )
    if getattr(cfg.instinctRL, "task", None) == "command_governor":
        reward_cfg = getattr(cfg.instinctRL, "reward", None)
        if reward_cfg is not None and not bool(getattr(reward_cfg, "enabled", False)):
            raise ValueError("command_governor eval requires instinctRL.reward.enabled=true")
    return True, instinct_mode


def _prefix_eval_summary(summary, pass_name):
    prefixed = {}
    for key, value in summary.items():
        if key == "passes":
            continue
        if key.startswith("eval/"):
            prefixed[f"eval/{pass_name}/{key[len('eval/'):]}"] = value
        else:
            prefixed[f"eval/{pass_name}/{key}"] = value
    return prefixed


def _copy_handbook_keys(target, source, *, prefixes):
    for key, value in source.items():
        if not key.startswith("eval/handbook."):
            continue
        metric_name = key[len("eval/handbook."):]
        if any(metric_name.startswith(prefix) for prefix in prefixes):
            target[key] = value


def _scalar_log_fields(summary):
    flat = {}
    for key, value in summary.items():
        if isinstance(value, (int, float, bool)):
            flat[key] = value
    return flat


def _run_eval_pass(
    *,
    pass_name,
    raw_env,
    transformed_env,
    policy,
    cfg,
    evaluate_fn,
    seed,
    exploration_type,
    command_source,
    command_frame_count,
    command_curriculum_profile,
    scenario_id_code,
):
    if hasattr(raw_env, "configure_instinctrl_eval_pass"):
        raw_env.configure_instinctrl_eval_pass(
            command_source=command_source,
            command_frame_count=command_frame_count,
            command_curriculum_profile=command_curriculum_profile,
            scenario_id_code=scenario_id_code,
        )
    print(f"[instinctRL eval] starting pass={pass_name}", flush=True)
    info, summary = evaluate_fn(
        env=transformed_env,
        policy=policy,
        seed=seed,
        cfg=cfg,
        exploration_type=exploration_type,
        return_summary=True,
        streaming=True,
        record_video=False,
    )
    summary["eval/pass_name"] = pass_name
    summary["eval/pass_command_source"] = command_source
    summary["eval/pass_command_frame_count"] = int(command_frame_count)
    summary["eval/pass_command_curriculum_profile"] = str(command_curriculum_profile)
    return info, summary


def _run_short_diagnostic_eval(*, raw_env, transformed_env, policy, cfg, evaluate_fn, exploration_type):
    eval_cfg = cfg.instinctRL.eval
    scenario_id_code = int(getattr(eval_cfg, "scenario_id_code", 1))
    tracking_frame = int(getattr(eval_cfg, "tracking_curriculum_frame", 600000))

    station_info, station_summary = _run_eval_pass(
        pass_name="station_static_mid360",
        raw_env=raw_env,
        transformed_env=transformed_env,
        policy=policy,
        cfg=cfg,
        evaluate_fn=evaluate_fn,
        seed=cfg.seed,
        exploration_type=exploration_type,
        command_source=str(getattr(eval_cfg, "station_command_source", "scripted_eval")),
        command_frame_count=0,
        command_curriculum_profile=str(getattr(eval_cfg, "station_curriculum_profile", "station_first")),
        scenario_id_code=scenario_id_code,
    )
    tracking_info, tracking_summary = _run_eval_pass(
        pass_name="tracking_static_mid360",
        raw_env=raw_env,
        transformed_env=transformed_env,
        policy=policy,
        cfg=cfg,
        evaluate_fn=evaluate_fn,
        seed=cfg.seed + 1,
        exploration_type=exploration_type,
        command_source=str(getattr(eval_cfg, "tracking_command_source", "curriculum_generator")),
        command_frame_count=tracking_frame,
        command_curriculum_profile=str(getattr(eval_cfg, "tracking_curriculum_profile", "diagnostic_mixed")),
        scenario_id_code=scenario_id_code,
    )

    combined_summary = {
        "eval/suite": "short_diagnostic",
        "eval/scenario_id": str(getattr(eval_cfg, "scenario_id", "static_mid360_short_diag")),
        "eval/scenario_id_code": scenario_id_code,
        "passes": {
            "station_static_mid360": station_summary,
            "tracking_static_mid360": tracking_summary,
        },
    }
    combined_summary.update(_prefix_eval_summary(station_summary, "station"))
    combined_summary.update(_prefix_eval_summary(tracking_summary, "tracking"))
    _copy_handbook_keys(
        combined_summary,
        station_summary,
        prefixes=("station_keeping", "anchor", "observability", "null_command"),
    )
    _copy_handbook_keys(
        combined_summary,
        tracking_summary,
        prefixes=("tracking", "command", "height", "safety", "ics", "termination"),
    )

    combined_info = {}
    combined_info.update(_scalar_log_fields(_prefix_eval_summary(station_summary, "station")))
    combined_info.update(_scalar_log_fields(_prefix_eval_summary(tracking_summary, "tracking")))
    combined_info.update(_scalar_log_fields(combined_summary))
    return combined_info, combined_summary


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

    checkpoint = _resolve_required_checkpoint(cfg)
    result_path = _resolve_result_path(cfg, checkpoint)
    instinct_enabled, instinct_mode = _preflight_instinctrl_eval(cfg)

    # ============================================
    # 第 1 步：启动 Isaac Sim 仿真应用
    # ============================================
    # headless: True=无图形界面(快), False=显示3D场景(慢但可观察)
    # anti_aliasing: 抗锯齿等级，提高渲染质量
    sim_app = SimulationApp({"headless": cfg.headless, "anti_aliasing": 1})

    from ppo import PPO
    from omni_drones.controllers import LeePositionController
    from omni_drones.utils.torchrl.transforms import VelController
    from torchrl.envs.transforms import TransformedEnv, Compose
    from utils import evaluate
    from torchrl.envs.utils import ExplorationType

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

    adapter = None
    ics_attenuator = None
    if instinct_enabled:
        from instinctRL.command_adapter import BodyToWorldVelocityAdapter

        adapter = BodyToWorldVelocityAdapter().to(cfg.device)
        ics_cfg = getattr(cfg.instinctRL, "ics", None)
        if ics_cfg is not None and getattr(ics_cfg, "enabled", False):
            from instinctRL.ics import ICSConfig, RangeHistoryICSAttenuator

            ics_attenuator = RangeHistoryICSAttenuator(
                ICSConfig.from_namespace(ics_cfg),
                device=cfg.device,
            )
            print("[instinctRL-E] ICS attenuation enabled (brake_mode=zero)", flush=True)
    
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
    # 这是评估脚本最重要的部分：加载明确指定的模型
    print(f"[NavRL]: Loading checkpoint from {checkpoint}", flush=True)
    policy.load_state_dict(torch.load(checkpoint, map_location=cfg.device))
    print("[NavRL]: Checkpoint loaded successfully!", flush=True)

    eval_policy = policy
    if instinct_enabled and instinct_mode == "train" and getattr(policy, "learned_governor", False):
        eval_policy = InstinctRLEvalPolicy(
            policy=policy,
            env=env,
            adapter=adapter,
            ics_attenuator=ics_attenuator,
        ).to(cfg.device)
        print("[instinctRL-A2] Learned governor eval wrapper enabled.", flush=True)

    if instinct_enabled:
        from instinctRL.audit import check_actor_input, check_actor_schema

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
            eval_policy(td.clone())
        env.reset()

    # ============================================
    # 第 9 步：评估循环（主循环）
    # ============================================
    # 与 train.py 的区别：
    #   - 不调用 policy.train()（不更新网络参数）
    #   - 只运行一次确定性评估
    print("[NavRL]: start deterministic evaluation", flush=True)
    env.eval()
    eval_cfg = getattr(getattr(cfg, "instinctRL", None), "eval", None)
    suite = str(getattr(eval_cfg, "suite", "single_rollout"))
    if instinct_enabled and suite == "short_diagnostic":
        eval_info, eval_summary = _run_short_diagnostic_eval(
            raw_env=env,
            transformed_env=transformed_env,
            policy=eval_policy,
            cfg=cfg,
            evaluate_fn=evaluate,
            exploration_type=ExplorationType.MEAN,
        )
    elif suite == "single_rollout":
        eval_info, eval_summary = evaluate(
            env=transformed_env,
            policy=eval_policy,
            seed=cfg.seed,
            cfg=cfg,
            exploration_type=ExplorationType.MEAN,
            return_summary=True,
            streaming=True,
            record_video=False,
        )
    else:
        raise ValueError(f"Unsupported instinctRL.eval.suite={suite!r}")
    eval_summary.update({
        "checkpoint_path": checkpoint,
        "result_path": result_path,
        "deterministic_exploration_type": "MEAN",
        "headless": bool(cfg.headless),
        "wandb_mode": cfg.wandb.mode,
        "env_num_envs": int(cfg.env.num_envs),
        "env_num_obstacles": int(cfg.env.num_obstacles),
        "env_dyn_num_obstacles": int(cfg.env_dyn.num_obstacles),
        "env_max_episode_length": int(cfg.env.max_episode_length),
        "max_frame_num": int(cfg.max_frame_num),
        "instinctRL_enabled": bool(instinct_enabled),
        "instinctRL_mode": str(instinct_mode),
        "instinctRL_eval_suite": suite,
        "learned_governor": bool(getattr(policy, "learned_governor", False)),
    })
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print("[NavRL]: evaluation done.", flush=True)
    print("[NavRL]: eval_summary_json=" + json.dumps(eval_summary, sort_keys=True), flush=True)
    print(f"[NavRL]: eval summary saved to {result_path}", flush=True)
    run.log(eval_info)

    # ============================================
    # 第 10 步：清理和关闭
    # ============================================
    # 评估结束，关闭日志和仿真器
    wandb.finish()
    sim_app.close()

if __name__ == "__main__":
    main()
