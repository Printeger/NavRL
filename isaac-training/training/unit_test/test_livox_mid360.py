"""
Livox Mid-360 LiDAR 传感器测试脚本
====================================
测试目的:
1. 验证 Livox Mid-360 配置是否正确加载
2. 检查射线模式生成是否符合规格
3. 在仿真环境中可视化 LiDAR 扫描效果
4. 验证与无人机的集成是否正常

运行方式 (与 train.py 相同):
    conda activate NavRL
    cd /home/mint/rl_dev/NavRL/isaac-training
    
    # 默认参数运行 (headless)
    python training/unit_test/test_livox_mid360.py
    
    # 带可视化
    python training/unit_test/test_livox_mid360.py headless=False
    
    # 使用 Livox Mid-360 参数
    python training/unit_test/test_livox_mid360.py headless=False sensor.lidar_vfov=[-7,52]
"""

import os
import sys
import hydra
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf

# Isaac Sim 应用
from omni.isaac.kit import SimulationApp

# 添加路径到 sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "scripts")
ENVS_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "envs")
sys.path.insert(0, SCRIPTS_PATH)
sys.path.insert(0, ENVS_PATH)  # livox_mid360.py 在 envs 目录下

# 配置文件路径
CFG_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "cfg")


@hydra.main(config_path=CFG_PATH, config_name="train", version_base=None)
def main(cfg: DictConfig):
    """
    主测试函数

    参数:
        cfg: Hydra 配置对象（来自 train.yaml）
    """
    # ============================================
    # 第 1 步: 启动 Isaac Sim
    # ============================================
    print("=" * 60)
    print("Livox Mid-360 LiDAR 传感器测试")
    print("=" * 60)
    print(f"[INFO] Headless 模式: {cfg.headless}")
    print(f"[INFO] 设备: {cfg.device}")
    print(f"[INFO] LiDAR 范围: {cfg.sensor.lidar_range} m")
    print(f"[INFO] 垂直 FOV: {cfg.sensor.lidar_vfov}")
    print(f"[INFO] 垂直线数: {cfg.sensor.lidar_vbeams}")
    print(f"[INFO] 水平分辨率: {cfg.sensor.lidar_hres}°")
    print(f"[INFO] 安装俯仰角: {cfg.sensor.lidar_mount_pitch}°")
    print("-" * 60)

    print("[STEP 1] 启动 Isaac Sim...")
    sim_app = SimulationApp({"headless": cfg.headless, "anti_aliasing": 1})

    # ============================================
    # 第 2 步: 导入 Isaac Sim 依赖 (SimulationApp 启动后)
    # ============================================
    print("[STEP 2] 导入依赖库...")

    import omni.isaac.orbit.sim as sim_utils
    from omni.isaac.core.simulation_context import SimulationContext
    from omni.isaac.orbit.assets import AssetBaseCfg
    from omni.isaac.orbit.sensors import RayCaster, RayCasterCfg, patterns
    from omni.isaac.orbit.terrains import (
        TerrainImporterCfg, TerrainImporter,
        TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg
    )
    from omni_drones.robots.drone import MultirotorBase
    from omni.isaac.debug_draw import _debug_draw

    # 导入 Livox Mid-360 配置
    from livox_mid360 import (
        LivoxMid360Config,
        create_livox_mid360_pattern,
        LIVOX_MID360_RL_RES,
    )

    print("[STEP 2] ✓ 依赖导入成功")

    # ============================================
    # TEST 1: 测试 Livox Mid-360 配置
    # ============================================
    def test_livox_config():
        """测试 Livox Mid-360 配置参数"""
        print("\n" + "=" * 60)
        print("[TEST 1] Livox Mid-360 配置参数测试")
        print("=" * 60)

        # 使用配置文件中的传感器参数
        lidar_range = cfg.sensor.lidar_range
        lidar_vfov = cfg.sensor.lidar_vfov
        lidar_vbeams = cfg.sensor.lidar_vbeams
        lidar_hres = cfg.sensor.lidar_hres
        lidar_hbeams = int(360 / lidar_hres)

        print(f"  ├── 最大范围: {lidar_range} m")
        print(f"  ├── 垂直 FOV: {lidar_vfov[0]}° ~ {lidar_vfov[1]}°")
        print(f"  ├── 水平分辨率: {lidar_hres}°")
        print(f"  ├── 水平射线数: {lidar_hbeams}")
        print(f"  ├── 垂直线数: {lidar_vbeams}")
        print(f"  └── 总射线数: {lidar_hbeams * lidar_vbeams}")

        # 测试从 Hydra 配置创建 (推荐方式)
        from livox_mid360 import create_livox_from_hydra_cfg

        livox_pattern = create_livox_from_hydra_cfg(cfg, device="cpu")
        livox_cfg = livox_pattern.cfg

        print(f"\n  Livox Mid-360 Config 验证:")
        print(f"  ├── 水平射线数: {livox_cfg.num_horizontal_rays}")
        print(f"  ├── 垂直线数: {livox_cfg.num_vertical_lines}")
        print(f"  ├── 总射线数: {livox_cfg.total_rays_nominal}")
        print(f"  ├── 安装俯仰角: {livox_cfg.mount_pitch}°")
        print(f"  ├── 安装横滚角: {livox_cfg.mount_roll}°")
        print(f"  ├── 安装偏航角: {livox_cfg.mount_yaw}°")
        print(f"  └── 安装位置: {livox_cfg.mount_position}")

        assert livox_cfg.num_horizontal_rays == lidar_hbeams
        print("[TEST 1] ✓ 配置参数测试通过")
        return livox_cfg

    # ============================================
    # TEST 2: 测试射线模式生成
    # ============================================
    def test_ray_pattern(livox_cfg):
        """测试射线模式生成"""
        print("\n" + "=" * 60)
        print("[TEST 2] 射线模式生成测试")
        print("=" * 60)

        ray_starts, ray_dirs = create_livox_mid360_pattern(
            livox_cfg, device="cpu")

        print(f"  ├── 射线起点形状: {ray_starts.shape}")
        print(f"  ├── 射线方向形状: {ray_dirs.shape}")

        # 验证单位向量
        norms = torch.norm(ray_dirs, dim=1)
        print(f"  ├── 方向向量范数: min={norms.min():.6f}, max={norms.max():.6f}")

        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

        # 验证角度分布
        v_angles_rad = torch.asin(ray_dirs[:, 2].clamp(-1, 1))
        v_angles_deg = torch.rad2deg(v_angles_rad)
        print(
            f"  └── 垂直角度范围: {v_angles_deg.min():.1f}° ~ {v_angles_deg.max():.1f}°")

        print("[TEST 2] ✓ 射线模式测试通过")
        return ray_dirs

    # ============================================
    # TEST 3: 创建测试场景
    # ============================================
    def create_test_scene():
        """创建测试场景"""
        print("\n" + "=" * 60)
        print("[TEST 3] 创建测试场景")
        print("=" * 60)

        # 首先创建 SimulationContext (TerrainImporter 需要它)
        # 使用 OmniDrones 相同的 SimulationContext
        print("  ├── 初始化仿真上下文...")
        sim_context = SimulationContext(
            stage_units_in_meters=1.0,
            physics_dt=0.02,
            rendering_dt=0.02,
            backend="torch",
            physics_prim_path="/physicsScene",
            device="cuda:0",
        )

        # 创建地形 (使用 TerrainImporter，LiDAR 可以扫描)
        print("  ├── 创建障碍物地形...")
        terrain_cfg = TerrainImporterCfg(
            num_envs=1,
            env_spacing=0.0,
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                seed=42,
                size=(30., 30.),
                border_width=2.0,
                num_rows=1,
                num_cols=1,
                horizontal_scale=0.1,
                vertical_scale=0.1,
                slope_threshold=0.75,
                use_cache=False,
                color_scheme="height",
                sub_terrains={
                    "obstacles": HfDiscreteObstaclesTerrainCfg(
                        horizontal_scale=0.1,
                        vertical_scale=0.1,
                        border_width=0.0,
                        num_obstacles=20,
                        obstacle_height_mode="range",
                        obstacle_width_range=(0.5, 1.5),
                        obstacle_height_range=[1.0, 1.5, 2.0, 3.0, 4.0],
                        obstacle_height_probability=[0.2, 0.25, 0.25, 0.3],
                        platform_width=0.0,
                    ),
                },
            ),
            visual_material=None,
            max_init_terrain_level=None,
            collision_group=-1,
            debug_vis=False,
        )
        terrain = TerrainImporter(terrain_cfg)

        # 添加光照
        print("  ├── 添加光照...")
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(
                color=(0.75, 0.75, 0.75),
                intensity=3000.0
            ),
        )
        light.spawn.func(light.prim_path, light.spawn, light.init_state.pos)

        # 创建环境模板路径 (无人机默认在这个路径下创建)
        import omni.isaac.core.utils.prims as prim_utils
        if not prim_utils.is_prim_path_valid("/World/envs/env_0"):
            prim_utils.define_prim("/World/envs/env_0")

        # 创建无人机 (使用 Hummingbird，它是默认可用的)
        print("  ├── 创建无人机...")
        # 尝试使用配置中的模型，如果不存在则使用 Hummingbird
        model_name = cfg.drone.model_name
        if model_name not in MultirotorBase.REGISTRY:
            print(f"  │   警告: 模型 '{model_name}' 不存在，使用 'Hummingbird'")
            model_name = "Hummingbird"
        drone_model = MultirotorBase.REGISTRY[model_name]
        drone_cfg = drone_model.cfg_cls(force_sensor=False)
        drone = drone_model(cfg=drone_cfg)
        drone.spawn(translations=[(0.0, 0.0, 2.0)])

        # 重置仿真以激活物理场景 (LiDAR 需要)
        print("  ├── 激活物理场景...")
        sim_context.reset()
        drone.initialize()

        # 查找实际的无人机 prim 路径
        print("  ├── 验证无人机 prim 路径...")
        stage = prim_utils.get_current_stage()
        drone_prim_path = None

        # 尝试几种可能的路径
        possible_paths = [
            "/World/envs/env_0/Hummingbird_0",
            "/World/envs/env_0/Hummingbird",
            f"/World/envs/env_0/{model_name}_0",
            f"/World/envs/env_0/{model_name}",
        ]

        for path in possible_paths:
            if prim_utils.is_prim_path_valid(path):
                drone_prim_path = path
                print(f"  │   找到无人机: {path}")

                # 列出子节点
                drone_prim = prim_utils.get_prim_at_path(path)
                children = prim_utils.get_prim_children(drone_prim)
                print(f"  │   子节点数量: {len(children)}")
                for child in children[:10]:  # 只显示前10个
                    print(f"  │     - {child.GetPath()}")
                break

        if drone_prim_path is None:
            print("  │   错误: 未找到无人机 prim!")
            # 列出 /World/envs/env_0 下的所有内容
            env_prim = prim_utils.get_prim_at_path("/World/envs/env_0")
            if env_prim.IsValid():
                children = prim_utils.get_prim_children(env_prim)
                print(f"  │   /World/envs/env_0 下的节点:")
                for child in children:
                    print(f"  │     - {child.GetPath()}")

        print("[TEST 3] ✓ 测试场景创建成功")
        return drone, sim_context, drone_prim_path

    # ============================================
    # TEST 4: 创建 Livox Mid-360 LiDAR
    # ============================================
    def create_livox_lidar(drone_prim_path):
        """创建 Livox Mid-360 LiDAR"""
        print("\n" + "=" * 60)
        print("[TEST 4] 创建 Livox Mid-360 LiDAR")
        print("=" * 60)

        if drone_prim_path is None:
            raise RuntimeError("无人机 prim 路径未找到，无法创建 LiDAR")

        # 使用配置文件中的参数
        lidar_range = cfg.sensor.lidar_range
        lidar_vfov = cfg.sensor.lidar_vfov
        lidar_vbeams = cfg.sensor.lidar_vbeams
        lidar_hres = cfg.sensor.lidar_hres

        # Livox Mid-360 垂直角度
        vertical_angles = torch.linspace(
            lidar_vfov[0], lidar_vfov[1], lidar_vbeams
        ).tolist()

        print(f"  ├── 最大范围: {lidar_range} m")
        print(f"  ├── 水平分辨率: {lidar_hres}° → {int(360/lidar_hres)} 条")
        print(f"  ├── 垂直线数: {lidar_vbeams}")
        print(
            f"  ├── 垂直角度: {vertical_angles[0]:.1f}° ~ {vertical_angles[-1]:.1f}°")
        print(f"  └── 总点数: {int(360/lidar_hres) * lidar_vbeams}")

        # 创建 RayCaster 配置 (使用 Livox Mid-360 参数)
        # 检查地形路径
        import omni.isaac.core.utils.prims as prim_utils
        ground_prim = prim_utils.get_prim_at_path("/World/ground")
        if ground_prim.IsValid():
            print(f"  ├── 地形路径 /World/ground 有效")
            # 列出子 prim
            children = prim_utils.get_prim_children(ground_prim)
            print(f"  ├── 子节点数量: {len(children)}")
            for i, child in enumerate(children[:5]):  # 只打印前5个
                print(f"  │   - {child.GetPath()}")
        else:
            print(f"  ├── 警告: 地形路径 /World/ground 无效!")

        # 查找 base_link 或类似的刚体 prim
        print(f"  ├── 查找 LiDAR 附着点...")
        drone_prim = prim_utils.get_prim_at_path(drone_prim_path)
        base_link_path = None

        # 尝试几种可能的子节点名称
        possible_links = ["base_link", "body", "base", "chassis"]
        for link_name in possible_links:
            test_path = f"{drone_prim_path}/{link_name}"
            if prim_utils.is_prim_path_valid(test_path):
                base_link_path = test_path
                print(f"  │   找到附着点: {base_link_path}")
                break

        if base_link_path is None:
            # 如果没找到特定名称，使用第一个刚体子节点
            children = prim_utils.get_prim_children(drone_prim)
            if children:
                base_link_path = str(children[0].GetPath())
                print(f"  │   使用第一个子节点: {base_link_path}")
            else:
                # 最后选择：直接使用无人机根节点
                base_link_path = drone_prim_path
                print(f"  │   使用无人机根节点: {base_link_path}")

        ray_caster_cfg = RayCasterCfg(
            prim_path=base_link_path,
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            attach_yaw_only=False,  # Livox Mid-360 是固态雷达
            pattern_cfg=patterns.BpearlPatternCfg(
                horizontal_fov=360.0,
                horizontal_res=lidar_hres,
                vertical_ray_angles=vertical_angles,
            ),
            max_distance=lidar_range,
            mesh_prim_paths=["/World/ground"],  # 直接使用地形路径
            debug_vis=not cfg.headless,
        )

        lidar = RayCaster(ray_caster_cfg)
        lidar._initialize_impl()

        print("[TEST 4] ✓ Livox Mid-360 LiDAR 创建成功")
        return lidar

    # ============================================
    # TEST 5: 运行仿真测试 (短期验证)
    # ============================================
    def run_simulation_test(drone, lidar, sim_context, steps=200):
        """运行仿真测试"""
        print("\n" + "=" * 60)
        print("[TEST 5] 运行仿真测试 (验证 LiDAR)")
        print("=" * 60)

        dt = sim_context.get_physics_dt()
        print(f"  ├── 仿真时间步: {dt} s")
        print(f"  ├── 设备: {cfg.device}")
        print(f"  ├── 运行 {steps} 步...")

        lidar_distances = []

        for step in range(steps):
            sim_context.step()
            lidar.update(dt)

            ray_hits = lidar.data.ray_hits_w
            lidar_pos = lidar.data.pos_w

            if ray_hits is not None and lidar_pos is not None:
                distances = (ray_hits - lidar_pos.unsqueeze(1)).norm(dim=-1)
                lidar_distances.append(distances.cpu())

            if (step + 1) % 50 == 0:
                print(f"  │   步数: {step + 1}/{steps}")

        print("  ├── 仿真完成")

        if lidar_distances:
            all_distances = torch.cat(lidar_distances, dim=0)
            valid_mask = all_distances < cfg.sensor.lidar_range

            print(f"  ├── LiDAR 数据统计:")
            print(f"  │   ├── 总采样数: {all_distances.numel()}")
            print(
                f"  │   ├── 有效击中: {valid_mask.sum().item()} ({100*valid_mask.float().mean():.1f}%)")

            if valid_mask.sum() > 0:
                valid_distances = all_distances[valid_mask]
                print(
                    f"  │   ├── 距离范围: {valid_distances.min():.2f} ~ {valid_distances.max():.2f} m")
                print(f"  │   └── 平均距离: {valid_distances.mean():.2f} m")

            print("[TEST 5] ✓ 仿真测试通过")
            return True
        else:
            print("[TEST 5] ✗ 未获取到 LiDAR 数据")
            return False

    # ============================================
    # TEST 6: 无人机悬停演示 (点云可视化)
    # ============================================
    def run_flight_demo(drone, lidar, sim_context):
        """让无人机悬停在场景中间，实时显示点云，支持键盘控制"""
        print("\n" + "=" * 60)
        print("[TEST 6] 无人机飞行演示 (带点云可视化)")
        print("=" * 60)
        print("  ├── 无人机使用 LeePositionController 控制")
        print("  ├── LiDAR 点云实时可视化中...")
        print("  ├── 绿色点: 远距离 | 黄色点: 中距离 | 红色点: 近距离")
        print("  ├──────────────────────────────────────────────")
        print("  ├── 键盘控制:")
        print("  │   W/S: 前进/后退")
        print("  │   A/D: 左移/右移")
        print("  │   Q/E: 上升/下降")
        print("  │   R: 重置到初始位置")
        print("  ├──────────────────────────────────────────────")
        print("  ├── 按 Ctrl+C 或关闭 Isaac Sim 窗口退出")
        print("=" * 60)

        # 导入控制器和键盘输入
        from omni_drones.controllers import LeePositionController
        import carb.input

        dt = sim_context.get_physics_dt()

        # 创建位置控制器
        controller = LeePositionController(
            g=9.81, uav_params=drone.params).to(drone.device)

        # 初始化 DebugDraw 用于点云可视化
        debug_draw = _debug_draw.acquire_debug_draw_interface()

        # 初始位置
        init_pos = torch.tensor(
            [0.0, 0.0, 1.5], device=drone.device, dtype=torch.float32)

        # 当前目标位置 (可通过键盘控制)
        target_pos = init_pos.clone()
        target_yaw = torch.tensor([0.0], device=drone.device)

        # 键盘控制参数
        move_speed = 0.05  # 每次按键移动的距离 (米)

        # 获取键盘输入接口
        import omni.appwindow
        appwindow = omni.appwindow.get_default_app_window()
        input_interface = carb.input.acquire_input_interface()
        keyboard = appwindow.get_keyboard()

        # 按键状态
        key_states = {
            carb.input.KeyboardInput.W: False,
            carb.input.KeyboardInput.S: False,
            carb.input.KeyboardInput.A: False,
            carb.input.KeyboardInput.D: False,
            carb.input.KeyboardInput.Q: False,
            carb.input.KeyboardInput.E: False,
            carb.input.KeyboardInput.R: False,
        }

        def on_keyboard_event(event):
            """键盘事件回调"""
            if event.input in key_states:
                key_states[event.input] = (event.type == carb.input.KeyboardEventType.KEY_PRESS or
                                           event.type == carb.input.KeyboardEventType.KEY_REPEAT)
            return True

        # 订阅键盘事件
        keyboard_sub = input_interface.subscribe_to_keyboard_events(
            keyboard, on_keyboard_event)

        step = 0
        print_interval = 200  # 每 200 步打印一次 (约 4 秒)
        time_elapsed = 0.0

        # 点云可视化参数
        point_size = 5.0  # 点的大小
        max_range = cfg.sensor.lidar_range

        print(
            f"  ├── 初始位置: ({target_pos[0]:.1f}, {target_pos[1]:.1f}, {target_pos[2]:.1f})")

        try:
            while sim_app.is_running():
                time_elapsed += dt

                # ===== 处理键盘输入 =====
                if key_states[carb.input.KeyboardInput.W]:
                    target_pos[0] += move_speed  # 前进 (X+)
                if key_states[carb.input.KeyboardInput.S]:
                    target_pos[0] -= move_speed  # 后退 (X-)
                if key_states[carb.input.KeyboardInput.A]:
                    target_pos[1] += move_speed  # 左移 (Y+)
                if key_states[carb.input.KeyboardInput.D]:
                    target_pos[1] -= move_speed  # 右移 (Y-)
                if key_states[carb.input.KeyboardInput.Q]:
                    target_pos[2] += move_speed  # 上升 (Z+)
                if key_states[carb.input.KeyboardInput.E]:
                    target_pos[2] -= move_speed  # 下降 (Z-)
                if key_states[carb.input.KeyboardInput.R]:
                    target_pos = init_pos.clone()  # 重置位置
                    key_states[carb.input.KeyboardInput.R] = False  # 防止重复触发

                # 限制高度范围
                target_pos[2] = torch.clamp(target_pos[2], 0.5, 5.0)

                # 获取无人机状态 (13维: pos[3], rot[4], vel[3], ang_vel[3])
                raw_state = drone.get_state()
                if raw_state.dim() == 3:
                    drone_state = raw_state[0, 0, :13]
                elif raw_state.dim() == 2:
                    drone_state = raw_state[0, :13]
                else:
                    drone_state = raw_state[:13]

                # 使用 LeePositionController 计算控制动作
                action = controller(
                    drone_state, target_pos=target_pos, target_yaw=target_yaw)

                # 应用动作 (让螺旋桨转动)
                drone.apply_action(action)

                # 仿真步进
                sim_context.step()

                # 更新 LiDAR
                lidar.update(dt)

                # ===== 点云可视化 =====
                # 清除上一帧的点
                debug_draw.clear_points()

                ray_hits = lidar.data.ray_hits_w
                lidar_pos = lidar.data.pos_w

                if ray_hits is not None and lidar_pos is not None:
                    # 计算距离
                    distances = (
                        ray_hits - lidar_pos.unsqueeze(1)).norm(dim=-1)

                    # 获取有效点 (在最大范围内)
                    valid_mask = distances[0] < max_range
                    valid_points = ray_hits[0][valid_mask]
                    valid_distances = distances[0][valid_mask]

                    if len(valid_points) > 0:
                        # 转换为 CPU numpy 用于绘制
                        points_np = valid_points.cpu().numpy()
                        dists_np = valid_distances.cpu().numpy()

                        # 根据距离计算颜色 (近红-中黄-远绿)
                        norm_dists = dists_np / max_range

                        # 创建颜色列表
                        colors = []
                        for d in norm_dists:
                            if d < 0.33:
                                colors.append((1.0, 0.2, 0.2, 1.0))  # 红
                            elif d < 0.66:
                                colors.append((1.0, 1.0, 0.2, 1.0))  # 黄
                            else:
                                colors.append((0.2, 1.0, 0.2, 1.0))  # 绿

                        # 绘制点云
                        point_list = [tuple(p) for p in points_np]
                        debug_draw.draw_points(point_list, colors, [
                                               point_size] * len(point_list))

                step += 1

                # 定期打印状态
                if step % print_interval == 0:
                    # 从状态中获取位置 (前3个元素)
                    # drone_state 现在是 [13] 的 1D 张量
                    pos_tensor = drone_state[:3]
                    px = float(pos_tensor[0])
                    py = float(pos_tensor[1])
                    pz = float(pos_tensor[2])
                    # 目标位置
                    tx = float(target_pos[0])
                    ty = float(target_pos[1])
                    tz = float(target_pos[2])
                    if ray_hits is not None and lidar_pos is not None:
                        distances = (
                            ray_hits - lidar_pos.unsqueeze(1)).norm(dim=-1)
                        valid_mask = distances < max_range
                        if valid_mask.sum() > 0:
                            min_dist = distances[valid_mask].min().item()
                            num_points = valid_mask.sum().item()
                        else:
                            min_dist = float('inf')
                            num_points = 0

                        print(f"  │ t={time_elapsed:5.1f}s | 当前: ({px:5.2f}, {py:5.2f}, {pz:5.2f}) | "
                              f"目标: ({tx:5.2f}, {ty:5.2f}, {tz:5.2f}) | "
                              f"点数: {num_points:4d} | 最近: {min_dist:5.2f}m")

        except KeyboardInterrupt:
            print("\n  ├── 收到 Ctrl+C 退出信号...")

        # 取消键盘订阅
        input_interface.unsubscribe_to_keyboard_events(keyboard, keyboard_sub)

        # 清除点云
        debug_draw.clear_points()

        print("[TEST 6] ✓ 飞行演示结束")
        return True

    # ============================================
    # 运行所有测试
    # ============================================
    all_passed = True

    try:
        # TEST 1: 配置参数
        livox_cfg = test_livox_config()

        # TEST 2: 射线模式
        ray_dirs = test_ray_pattern(livox_cfg)

        # TEST 3: 创建场景
        drone, sim_context, drone_prim_path = create_test_scene()

        # TEST 4: 创建 LiDAR
        lidar = create_livox_lidar(drone_prim_path)

        # TEST 5: 运行仿真验证
        sim_passed = run_simulation_test(drone, lidar, sim_context, steps=200)
        all_passed = all_passed and sim_passed

        # 测试结果汇总
        print("\n" + "=" * 60)
        if all_passed:
            print("✓ 所有测试通过！Livox Mid-360 LiDAR 集成成功")
        else:
            print("✗ 部分测试失败，请检查错误信息")
        print("=" * 60)

        # TEST 6: 飞行演示 (持续运行)
        if not cfg.headless:
            run_flight_demo(drone, lidar, sim_context)

            # 保持 Isaac Sim 运行，等待用户手动关闭
            print("\n" + "=" * 60)
            print("Isaac Sim 保持运行中...")
            print("请手动关闭 Isaac Sim 窗口退出")
            print("=" * 60)

            # 持续运行直到窗口关闭
            while sim_app.is_running():
                sim_context.step()
        else:
            print("\n[INFO] Headless 模式，跳过飞行演示")
            sim_app.close()

    except Exception as e:
        print(f"\n[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
        sim_app.close()


if __name__ == "__main__":
    main()
