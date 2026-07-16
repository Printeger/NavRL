"""
导航环境定义 (NavigationEnv)
==============================
这是强化学习训练的核心环境，定义了：
1. 场景设计：地形、障碍物、无人机
2. 传感器：LiDAR（激光雷达）
3. 观测空间：LiDAR 数据、无人机状态、动态障碍物信息
4. 动作空间：速度指令 [vx, vy, vz]
5. 奖励函数：安全性、速度、平滑性
6. 终止条件：碰撞、超出边界、到达目标

继承关系：
NavigationEnv -> IsaacEnv (OmniDrones) -> EnvBase (TorchRL)
"""

import torch
import einops
import numpy as np
import importlib.util
import os
import sys
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec
import omni.isaac.orbit.sim as sim_utils
import omni_drones
from omni_drones.robots.drone import MultirotorBase
from omni.isaac.orbit.assets import AssetBaseCfg
from omni.isaac.orbit.terrains import TerrainImporterCfg, TerrainImporter, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg
from omni_drones.utils.torch import euler_to_quaternion, quat_axis
from omni.isaac.orbit.sensors import RayCaster, RayCasterCfg
from omni.isaac.core.utils.viewports import set_camera_view
from utils import vec_to_new_frame, vec_to_world, construct_input
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.orbit.sim as sim_utils
import omni.isaac.orbit.utils.math as math_utils
from omni.isaac.orbit.assets import RigidObject, RigidObjectCfg
import time
from instinctRL.task_metrics import (
    COMMAND_MODE_NORMAL,
    COMMAND_MODE_RECOVERY,
    R5E3_DIAGNOSTIC_FIELDS,
    command_curriculum_probabilities,
    compute_handbook_step_metrics,
    compute_r5e2_collision_geometry_step_metrics,
    compute_r5e3_braking_residual_step_metrics,
    compute_termination_stats,
    nearest_obstacle_vector_from_scan,
    world_to_body_velocity,
)


def _load_isaac_env_base():
    """
    Load OmniDrones' isaac_env.py without executing omni_drones.envs.__init__.

    The package __init__ eagerly imports all sample environments, including
    camera-based tasks that require omni.replicator. NavigationEnv only needs
    IsaacEnv, so loading the file directly keeps B0 headless dependencies small.
    """
    module_name = "_navrl_omnidrones_isaac_env"
    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        module_path = os.path.join(
            os.path.dirname(omni_drones.__file__),
            "envs",
            "isaac_env.py",
        )
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module.IsaacEnv


IsaacEnv = _load_isaac_env_base()


class NavigationEnv(IsaacEnv):
    """
    导航环境类
    
    任务：无人机从起点飞到终点，避开静态和动态障碍物
    
    每一步的执行顺序：
    1. _pre_sim_step: 应用动作（设置电机推力）
    2. step isaac sim: 物理仿真更新（PhysX）
    3. _post_sim_step: 更新传感器（LiDAR）和动态障碍物
    4. increment progress_buf: 步数 +1
    5. _compute_state_and_obs: 计算观测和状态
    6. _compute_reward_and_done: 计算奖励和终止条件
    
    观测空间（输入给策略网络）：
    - LiDAR: [1, 36, 4] 点云数据
    - 无人机状态: [8] (距离、方向、速度)
    - 动态障碍物: [1, N, 10] N个最近障碍物的信息
    
    动作空间（策略网络输出）：
    - 速度指令: [3] (vx, vy, vz)
    """

    def __init__(self, cfg):
        """
        初始化环境
        
        参数:
            cfg: 配置对象（来自 train.yaml + drone.yaml 等）
                - cfg.sensor: LiDAR 配置
                - cfg.env: 环境配置（地图大小、障碍物数量）
                - cfg.env_dyn: 动态障碍物配置
        """
        print("[Navigation Environment]: Initializing Env...", flush=True)
        
        # ============================================
        # 第 1 步：配置 LiDAR 参数
        # ============================================
        # LiDAR（激光雷达）参数
        self.lidar_range = cfg.sensor.lidar_range  # 最大探测距离（米）
        self.lidar_vfov = (  # 垂直视场角（度）
            max(-89., cfg.sensor.lidar_vfov[0]), 
            min(89., cfg.sensor.lidar_vfov[1])
        )
        self.lidar_vbeams = cfg.sensor.lidar_vbeams  # 垂直线束数（例如4条）
        self.lidar_hres = cfg.sensor.lidar_hres  # 水平角分辨率（度，例如10°）
        self.lidar_hbeams = int(360/self.lidar_hres)  # 水平线束数（360°/10° = 36条）

        # ============================================
        # 第 2 步：调用父类初始化（创建仿真场景）
        # ============================================
        # IsaacEnv.__init__() 会：
        # 1. 初始化 Isaac Sim 上下文
        # 2. 调用 _design_scene() 创建场景
        # 3. 调用 _set_specs() 定义空间规范
        print("[Navigation Environment]: Calling IsaacEnv.__init__...", flush=True)
        super().__init__(cfg, cfg.headless)
        print("[Navigation Environment]: IsaacEnv.__init__ complete.", flush=True)
        
        # ============================================
        # 第 3 步：初始化无人机
        # ============================================
        print("[Navigation Environment]: Initializing TASLAB_UAV articulation...", flush=True)
        self.drone.initialize()  # 初始化无人机物理属性
        print("[Navigation Environment]: TASLAB_UAV articulation initialized.", flush=True)
        self.init_vels = torch.zeros_like(self.drone.get_velocities())  # 初始速度为 0

        # ============================================
        # 第 4 步：初始化 LiDAR 传感器 ⭐ 重要
        # ============================================
        # instinctRL-0: Dynamically resolve LiDAR prim path from spawned drone.
        # Uses discovered base_link name (not hardcoded Hummingbird).
        base_link = self._base_link_name
        if base_link:
            drone_prim_pattern = f"/World/envs/env_.*/{self.cfg.drone.model_name}_0/{base_link}"
        else:
            drone_prim_pattern = f"/World/envs/env_.*/{self.cfg.drone.model_name}_0"
        print(f"[instinctRL-0] LiDAR prim path pattern: {drone_prim_pattern}", flush=True)
        if getattr(cfg, "instinctRL", None) and cfg.instinctRL.enabled:
            from instinctRL.mid360_pattern import (
                create_mid360_pattern_cfg,
                mount_position,
                mount_quat_wxyz,
                ray_order_hash,
            )

            pattern_cfg = create_mid360_pattern_cfg(cfg.sensor)
            ray_starts, ray_dirs = pattern_cfg.func(pattern_cfg, self.device)
            offset_pos = mount_position(cfg.sensor)
            offset_rot = mount_quat_wxyz(cfg.sensor)
            self._mid360_ray_dirs_b = ray_dirs.clone()
            self._mid360_ray_order_hash = ray_order_hash(ray_dirs)
            print(
                "[instinctRL-B] Active sensor pattern: LivoxMid360Pattern "
                f"rays={pattern_cfg.num_rays} shape=({pattern_cfg.num_horizontal_rays}, "
                f"{pattern_cfg.num_vertical_lines}) hash={self._mid360_ray_order_hash}",
                flush=True,
            )
        else:
            from instinctRL.mid360_pattern import create_mid360_pattern_cfg

            pattern_cfg = create_mid360_pattern_cfg(cfg.sensor)
            _, ray_dirs = pattern_cfg.func(pattern_cfg, self.device)
            self._mid360_ray_dirs_b = ray_dirs.clone()
            offset_pos = (0.0, 0.0, 0.0)
            offset_rot = (1.0, 0.0, 0.0, 0.0)

        ray_caster_cfg = RayCasterCfg(
            # 绑定到无人机的 base_link（所有环境的所有无人机）
            prim_path=drone_prim_pattern,
            
            offset=RayCasterCfg.OffsetCfg(pos=offset_pos, rot=offset_rot),
            
            # instinctRL-0: attach_yaw_only=False for solid-state MID360.
            # MID360 is a non-repetitive scanning LiDAR, not a spinning mirror.
            # This changes LiDAR data distribution vs old yaw-only policies.
            attach_yaw_only=False,
            
            pattern_cfg=pattern_cfg,
            max_distance=self.lidar_range,
            
            debug_vis=False,  # 不可视化射线（提高性能）
            
            # 检测的对象：只检测地面（静态障碍物在地面上）
            mesh_prim_paths=["/World/ground"],
        )
        print("[instinctRL-0] Initializing MID360 RayCaster...", flush=True)
        self.lidar = RayCaster(ray_caster_cfg)
        self.lidar._initialize_impl()  # 初始化射线投射器
        print("[instinctRL-0] MID360 RayCaster initialized.", flush=True)
        self.lidar_resolution = (self.lidar_hbeams, self.lidar_vbeams)  # (36, 4)

        # instinctRL-B: Create MID360 observation builder (actor-clean pipeline)
        if getattr(cfg, "instinctRL", None) and cfg.instinctRL.enabled:
            from instinctRL.observation import ObservationConfig, MID360ObservationBuilder
            self._obs_builder = MID360ObservationBuilder(
                ObservationConfig(
                    history_len=cfg.instinctRL.observation.history_len,
                    enable_noise=cfg.instinctRL.observation.enable_noise,
                    enable_dropout=cfg.instinctRL.observation.enable_dropout,
                    tau_staleness=cfg.instinctRL.observation.tau_staleness,
                    lidar_hbeams=self.lidar_hbeams,
                    lidar_vbeams=self.lidar_vbeams,
                    lidar_range=cfg.sensor.lidar_range,
                ),
                device=self.device,
            )
            print("[instinctRL-B] MID360ObservationBuilder created "
                  f"(history={cfg.instinctRL.observation.history_len})")

            anchor_cfg = getattr(cfg.instinctRL, "anchor", None)
            if anchor_cfg is not None and getattr(anchor_cfg, "enabled", False):
                from instinctRL.anchor import AnchorConfig, MeasurementSpaceAnchorManager
                self._anchor_manager = MeasurementSpaceAnchorManager(
                    AnchorConfig.from_namespace(
                        anchor_cfg,
                        lidar_hbeams=self.lidar_hbeams,
                        lidar_vbeams=self.lidar_vbeams,
                    ),
                    num_envs=self.num_envs,
                    device=self.device,
                )
                self.anchor_outputs = {}
                print("[instinctRL-C] MeasurementSpaceAnchorManager created")

            observability_cfg = getattr(cfg.instinctRL, "observability", None)
            if observability_cfg is not None and getattr(observability_cfg, "enabled", False):
                from instinctRL.observability import (
                    ObservabilityConfig,
                    RangeJacobianObservabilityLogger,
                )
                self._observability_logger = RangeJacobianObservabilityLogger(
                    ObservabilityConfig.from_namespace(observability_cfg),
                    device=self.device,
                )
                self.observability_outputs = {}
                print("[instinctRL-D] RangeJacobianObservabilityLogger created")

            ics_cfg = getattr(cfg.instinctRL, "ics", None)
            if ics_cfg is not None and getattr(ics_cfg, "enabled", False):
                self.ics_outputs = {}

            # cfg.instinctRL.reward.enabled gates F reward integration/readiness.
            reward_cfg = getattr(cfg.instinctRL, "reward", None)
            if reward_cfg is not None and getattr(reward_cfg, "enabled", False):
                from instinctRL.rewards import RewardConfig, InstinctRLRewardComputer
                self._reward_computer = InstinctRLRewardComputer(
                    RewardConfig.from_namespace(reward_cfg),
                    device=self.device,
                )
                print("[instinctRL-F] Reward computer created")

            command_cfg = getattr(cfg.instinctRL, "command", None)
            self._instinctrl_task = getattr(cfg.instinctRL, "task", "command_governor")
            self._command_source = getattr(command_cfg, "source", "basic_random")
            self._command_max_vel = float(getattr(command_cfg, "max_vel", 1.0))
            self._command_curriculum_profile = str(
                getattr(command_cfg, "curriculum_profile", "default")
            )
            self._command_frame_count = 0
            eval_cfg = getattr(cfg.instinctRL, "eval", None)
            self._instinctrl_eval_scenario_id_code = int(
                getattr(eval_cfg, "scenario_id_code", 0)
            )
            self._v_cmd = torch.zeros(self.num_envs, 1, 3, device=self.device)
            self._nearest_obstacle_vector_b = torch.zeros(self.num_envs, 3, device=self.device)
            self._nearest_obstacle_vector_b[:, 0] = 1.0
            if self._command_source == "curriculum_generator":
                self._ensure_command_generator()
                print("[instinctRL] Command source: curriculum_generator")
            elif self._command_source not in {"basic_random", "scripted_eval"}:
                raise ValueError(f"Unsupported instinctRL.command.source={self._command_source!r}")
        
        # ============================================
        # 第 5 步：初始化目标和状态变量
        # ============================================
        with torch.device(self.device):
            # 目标位置（每个环境一个目标）
            self.target_pos = torch.zeros(self.num_envs, 1, 3)
            
            # 目标方向（用于坐标变换）
            self.target_dir = torch.zeros(self.num_envs, 1, 3)
            
            # 高度范围（用于惩罚过高/过低飞行）
            # [0]: 最小高度, [1]: 最大高度
            self.height_range = torch.zeros(self.num_envs, 1, 2)
            
            # 前一步的速度（用于计算平滑性奖励）
            self.prev_drone_vel_w = torch.zeros(self.num_envs, 1 , 3) 
            self._prev_issued_action_body = torch.zeros(self.num_envs, 3)
            self._has_prev_issued_action_body = torch.zeros(self.num_envs, 1, dtype=torch.bool)
            self._reward_prev_v_final_body = torch.zeros(self.num_envs, 3)
            self._station_origin_pos_w = torch.zeros(self.num_envs, 3)


    def _design_scene(self):
        """
        设计仿真场景
        
        场景包含：
        1. 无人机模型（Hummingbird）
        2. 光照（太阳光 + 天空光）
        3. 地面
        4. 静态障碍物（地形）
        5. 动态障碍物（移动的立方体和圆柱）
        
        这个方法会在环境初始化时被调用一次。
        """
        # ============================================
        # 1. 创建无人机模型
        # ============================================
        # 从注册表中获取无人机模型类（例如 "Hummingbird"）
        drone_model = MultirotorBase.REGISTRY[self.cfg.drone.model_name]
        cfg = drone_model.cfg_cls(force_sensor=False)  # 不使用力传感器
        self.drone = drone_model(cfg=cfg)
        # 生成无人机，初始位置在 z=2.0 米处
        drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 2.0)])[0]

        # instinctRL-0: Discover base_link child prim for LiDAR attachment.
        # Uses robust search strategy from MID360 integration test.
        # drone.spawn() returns a Usd.Prim; extract the path string.
        drone_prim_path_str = str(drone_prim.GetPath())
        self._drone_spawn_prim = drone_prim_path_str
        self._base_link_name = self._resolve_base_link(drone_prim_path_str)

        # ============================================
        # 2. 添加光照（让场景可见）
        # ============================================
        # 定向光（模拟太阳光）
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(
                color=(0.75, 0.75, 0.75), 
                intensity=3000.0
            ),
        )
        # 天空光（环境光）
        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                color=(0.2, 0.2, 0.3), 
                intensity=2000.0
            ),
        )
        light.spawn.func(light.prim_path, light.spawn, light.init_state.pos)
        sky_light.spawn.func(sky_light.prim_path, sky_light.spawn)
        
        # ============================================
        # 3. 创建地面
        # ============================================
        # The Orbit GroundPlaneCfg references a remote Isaac USD asset. Keep the
        # training path independent from Nucleus/S3; the generated terrain below
        # supplies the physical ground used by this environment.

        # ============================================
        # 4. 生成静态障碍物地形
        # ============================================
        # 地图范围：40m × 40m × 4.5m（x, y, z）
        self.map_range = [20.0, 20.0, 4.5]

        terrain_cfg = TerrainImporterCfg(
            num_envs=self.num_envs,  # 多少个并行环境
            env_spacing=0.0,  # 环境之间的间距（0表示共享地形）
            prim_path="/World/ground",
            terrain_type="generator",  # 使用生成器创建地形
            
            terrain_generator=TerrainGeneratorCfg(
                seed=0,  # 随机种子（保证可复现）
                size=(self.map_range[0]*2, self.map_range[1]*2),  # 40m × 40m
                border_width=5.0,  # 边界宽度
                num_rows=1,  # 地形块行数
                num_cols=1,  # 地形块列数
                horizontal_scale=0.1,  # 水平分辨率（10cm）
                vertical_scale=0.1,  # 垂直分辨率（10cm）
                slope_threshold=0.75,  # 坡度阈值
                use_cache=False,  # 不使用缓存（每次重新生成）
                color_scheme="height",  # 按高度着色
                
                # 子地形：离散障碍物
                sub_terrains={
                    "obstacles": HfDiscreteObstaclesTerrainCfg(
                        horizontal_scale=0.1,
                        vertical_scale=0.1,
                        border_width=0.0,
                        num_obstacles=self.cfg.env.num_obstacles,  # 障碍物数量
                        obstacle_height_mode="range",  # 高度模式：范围
                        obstacle_width_range=(0.4, 1.1),  # 宽度范围：0.4-1.1m
                        # 高度范围（米）：[1.0, 1.5, 2.0, 4.0, 6.0]
                        obstacle_height_range=[1.0, 1.5, 2.0, 4.0, 6.0],
                        # 每个高度的概率：[10%, 15%, 20%, 55%]
                        obstacle_height_probability=[0.1, 0.15, 0.20, 0.55],
                        platform_width=0.0,  # 平台宽度
                    ),
                },
            ),
            visual_material = None,
            max_init_terrain_level=None,
            collision_group=-1,  # 碰撞组（-1表示与所有物体碰撞）
            debug_vis=False,  # avoid remote marker USD assets during headless training
        )
        terrain_importer = TerrainImporter(terrain_cfg)  # 导入地形

        if (self.cfg.env_dyn.num_obstacles == 0):
            return
        # Dynamic Obstacles
        # NOTE: we use cuboid to represent 3D dynamic obstacles which can float in the air 
        # and the long cylinder to represent 2D dynamic obstacles for which the drone can only pass in 2D 
        # The width of the dynamic obstacles is divided into N_w=4 bins
        # [[0, 0.25], [0.25, 0.50], [0.50, 0.75], [0.75, 1.0]]
        # The height of the dynamic obstacles is divided into N_h=2 bins
        # [[0, 0.5], [0.5, inf]] we want to distinguish 3D obstacles and 2d obstacles
        N_w = 4 # number of width intervals between [0, 1]
        N_h = 2 # number of height: current only support binary
        max_obs_width = 1.0
        self.max_obs_3d_height = 1.0
        self.max_obs_2d_height = 5.0
        self.dyn_obs_width_res = max_obs_width/float(N_w)
        dyn_obs_category_num = N_w * N_h
        self.dyn_obs_num_of_each_category = int(self.cfg.env_dyn.num_obstacles / dyn_obs_category_num)
        self.cfg.env_dyn.num_obstacles = self.dyn_obs_num_of_each_category * dyn_obs_category_num # in case of the roundup error


        # Dynamic obstacle info
        self.dyn_obs_list = []
        self.dyn_obs_state = torch.zeros((self.cfg.env_dyn.num_obstacles, 13), dtype=torch.float, device=self.cfg.device) # 13 is based on the states from sim, we only care the first three which is position
        self.dyn_obs_state[:, 3] = 1. # Quaternion
        self.dyn_obs_goal = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.cfg.device)
        self.dyn_obs_origin = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.cfg.device)
        self.dyn_obs_vel = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.cfg.device)
        self.dyn_obs_step_count = 0 # dynamic obstacle motion step count
        self.dyn_obs_size = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.device) # size of dynamic obstacles


        # helper function to check pos validity for even distribution condition
        def check_pos_validity(prev_pos_list, curr_pos, adjusted_obs_dist):
            for prev_pos in prev_pos_list:
                if (np.linalg.norm(curr_pos - prev_pos) <= adjusted_obs_dist):
                    return False
            return True            
        
        obs_dist = 2 * np.sqrt(self.map_range[0] * self.map_range[1] / self.cfg.env_dyn.num_obstacles) # prefered distance between each dynamic obstacle
        curr_obs_dist = obs_dist
        prev_pos_list = [] # for distance check
        cuboid_category_num = cylinder_category_num = int(dyn_obs_category_num/N_h)
        for category_idx in range(cuboid_category_num + cylinder_category_num):
            # create all origins for 3D dynamic obstacles of this category (size)
            for origin_idx in range(self.dyn_obs_num_of_each_category):
                # random sample an origin until satisfy the evenly distributed condition
                start_time = time.time()
                while (True):
                    ox = np.random.uniform(low=-self.map_range[0], high=self.map_range[0])
                    oy = np.random.uniform(low=-self.map_range[1], high=self.map_range[1])
                    if (category_idx < cuboid_category_num):
                        oz = np.random.uniform(low=0.0, high=self.map_range[2]) 
                    else:
                        oz = self.max_obs_2d_height/2. # half of the height
                    curr_pos = np.array([ox, oy])
                    valid = check_pos_validity(prev_pos_list, curr_pos, curr_obs_dist)
                    curr_time = time.time()
                    if (curr_time - start_time > 0.1):
                        curr_obs_dist *= 0.8
                        start_time = time.time()
                    if (valid):
                        prev_pos_list.append(curr_pos)
                        break
                curr_obs_dist = obs_dist
                origin = [ox, oy, oz]
                self.dyn_obs_origin[origin_idx+category_idx*self.dyn_obs_num_of_each_category] = torch.tensor(origin, dtype=torch.float, device=self.cfg.device)     
                self.dyn_obs_state[origin_idx+category_idx*self.dyn_obs_num_of_each_category, :3] = torch.tensor(origin, dtype=torch.float, device=self.cfg.device)                        
                prim_utils.create_prim(f"/World/Origin{origin_idx+category_idx*self.dyn_obs_num_of_each_category}", "Xform", translation=origin)

            # Spawn various sizes of dynamic obstacles 
            if (category_idx < cuboid_category_num):
                # spawn for 3D dynamic obstacles
                obs_width = width = float(category_idx+1) * max_obs_width/float(N_w)
                obs_height = self.max_obs_3d_height
                cuboid_cfg = RigidObjectCfg(
                    prim_path=f"/World/Origin{construct_input(category_idx*self.dyn_obs_num_of_each_category, (category_idx+1)*self.dyn_obs_num_of_each_category)}/Cuboid",
                    spawn=sim_utils.CuboidCfg(
                        size=[width, width, self.max_obs_3d_height],
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(),
                )
                dynamic_obstacle = RigidObject(cfg=cuboid_cfg)
            else:
                radius = float(category_idx-cuboid_category_num+1) * max_obs_width/float(N_w) / 2.
                obs_width = radius * 2
                obs_height = self.max_obs_2d_height
                # spawn for 2D dynamic obstacles
                cylinder_cfg = RigidObjectCfg(
                    prim_path=f"/World/Origin{construct_input(category_idx*self.dyn_obs_num_of_each_category, (category_idx+1)*self.dyn_obs_num_of_each_category)}/Cylinder",
                    spawn=sim_utils.CylinderCfg(
                        radius = radius,
                        height = self.max_obs_2d_height, 
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(),
                )
                dynamic_obstacle = RigidObject(cfg=cylinder_cfg)
            self.dyn_obs_list.append(dynamic_obstacle)
            self.dyn_obs_size[category_idx*self.dyn_obs_num_of_each_category:(category_idx+1)*self.dyn_obs_num_of_each_category] \
                = torch.tensor([obs_width, obs_width, obs_height], dtype=torch.float, device=self.cfg.device)



    def move_dynamic_obstacle(self):
        # Step 1: Random sample new goals for required update dynamic obstacles
        # Check whether the current dynamic obstacles need new goals
        dyn_obs_goal_dist = torch.sqrt(torch.sum((self.dyn_obs_state[:, :3] - self.dyn_obs_goal)**2, dim=1)) if self.dyn_obs_step_count !=0 \
            else torch.zeros(self.dyn_obs_state.size(0), device=self.cfg.device)
        dyn_obs_new_goal_mask = dyn_obs_goal_dist < 0.5 # change to a new goal if less than the threshold
        
        # sample new goals in local range
        num_new_goal = torch.sum(dyn_obs_new_goal_mask)
        sample_x_local = -self.cfg.env_dyn.local_range[0] + 2. * self.cfg.env_dyn.local_range[0] * torch.rand(num_new_goal, 1, dtype=torch.float, device=self.cfg.device)
        sample_y_local = -self.cfg.env_dyn.local_range[1] + 2. * self.cfg.env_dyn.local_range[1] * torch.rand(num_new_goal, 1, dtype=torch.float, device=self.cfg.device)
        sample_z_local = -self.cfg.env_dyn.local_range[1] + 2. * self.cfg.env_dyn.local_range[2] * torch.rand(num_new_goal, 1, dtype=torch.float, device=self.cfg.device)
        sample_goal_local = torch.cat([sample_x_local, sample_y_local, sample_z_local], dim=1)
    
        # apply local goal to the global range
        self.dyn_obs_goal[dyn_obs_new_goal_mask] = self.dyn_obs_origin[dyn_obs_new_goal_mask] + sample_goal_local
        # clamp the range if out of the static env range
        self.dyn_obs_goal[:, 0] = torch.clamp(self.dyn_obs_goal[:, 0], min=-self.map_range[0], max=self.map_range[0])
        self.dyn_obs_goal[:, 1] = torch.clamp(self.dyn_obs_goal[:, 1], min=-self.map_range[1], max=self.map_range[1])
        self.dyn_obs_goal[:, 2] = torch.clamp(self.dyn_obs_goal[:, 2], min=0., max=self.map_range[2])
        self.dyn_obs_goal[int(self.dyn_obs_goal.size(0)/2):, 2] = self.max_obs_2d_height/2. # for 2d obstacles


        # Step 2: Random sample velocity for roughly every 2 seconds
        if (self.dyn_obs_step_count % int(2.0/self.cfg.sim.dt) == 0):
            self.dyn_obs_vel_norm = self.cfg.env_dyn.vel_range[0] + (self.cfg.env_dyn.vel_range[1] \
              - self.cfg.env_dyn.vel_range[0]) * torch.rand(self.dyn_obs_vel.size(0), 1, dtype=torch.float, device=self.cfg.device)
            self.dyn_obs_vel = self.dyn_obs_vel_norm * \
                (self.dyn_obs_goal - self.dyn_obs_state[:, :3])/torch.norm((self.dyn_obs_goal - self.dyn_obs_state[:, :3]), dim=1, keepdim=True)

        # Step 3: Calculate new position update for current timestep
        self.dyn_obs_state[:, :3] += self.dyn_obs_vel * self.cfg.sim.dt


        # Step 4: Update Visualized Location in Simulation
        for category_idx, dynamic_obstacle in enumerate(self.dyn_obs_list):
            dynamic_obstacle.write_root_state_to_sim(self.dyn_obs_state[category_idx*self.dyn_obs_num_of_each_category:(category_idx+1)*self.dyn_obs_num_of_each_category]) 
            dynamic_obstacle.write_data_to_sim()
            dynamic_obstacle.update(self.cfg.sim.dt)

        self.dyn_obs_step_count += 1

    def _resolve_base_link(self, drone_root_prim_path: str) -> str:
        """
        Find the base_link child prim name under the drone root prim.

        Uses the robust search strategy from the MID360 integration test
        (test_livox_mid360.py): tries candidate names in priority order,
        falls back to first rigid child, then to drone root.

        Args:
            drone_root_prim_path: Full prim path of spawned drone root
                                  (e.g., /World/envs/env_0/TaslabUAV_0)

        Returns:
            Name of the base_link child prim (e.g., "base_link"),
            or empty string if no suitable child found.
        """
        candidates = ["base_link", "body", "base", "chassis"]
        drone_prim = prim_utils.get_prim_at_path(drone_root_prim_path)
        for name in candidates:
            test_path = f"{drone_root_prim_path}/{name}"
            if prim_utils.is_prim_path_valid(test_path):
                print(f"[instinctRL-0] LiDAR attachment point: {test_path}")
                return name
        # Fallback: use first rigid child
        children = prim_utils.get_prim_children(drone_prim)
        if children:
            child_name = str(children[0].GetPath()).split("/")[-1]
            print(f"[instinctRL-0] LiDAR attachment point (fallback): "
                  f"{drone_root_prim_path}/{child_name}")
            return child_name
        print(f"[instinctRL-0] WARNING: No base_link found under {drone_root_prim_path}. "
              f"Attaching LiDAR to drone root.")
        return ""


    def _set_specs(self):
        # instinctRL-B: Actor input contract — hybrid observation format.
        #   lidar_grid: history-stacked range/mask/weight channels [N, L*3, H, V]
        #   state_vec:  history-stacked IMU+v_cmd+prev_action+frame_age [N, L*13]
        # No pose, odometry, explicit velocity, map, or privileged state.
        lidar_channels = self.cfg.instinctRL.observation.history_len * 3
        state_dim = self.cfg.instinctRL.observation.history_len * 13
        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": CompositeSpec({
                    "lidar_grid": UnboundedContinuousTensorSpec(
                        (lidar_channels, self.lidar_hbeams, self.lidar_vbeams),
                        device=self.device
                    ),
                    "state_vec": UnboundedContinuousTensorSpec(
                        (state_dim,), device=self.device
                    ),
                }),
            }).expand(self.num_envs)
        }, shape=[self.num_envs], device=self.device)
        
        # Action Spec
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": self.drone.action_spec, # number of motor
            })
        }).expand(self.num_envs).to(self.device)
        
        # Reward Spec
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1,))
            })
        }).expand(self.num_envs).to(self.device)

        # Done Spec
        self.done_spec = CompositeSpec({
            "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "terminated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "truncated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
        }).expand(self.num_envs).to(self.device) 


        stats_spec_fields = {
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "legacy_reach_goal": UnboundedContinuousTensorSpec(1),
            "collision": UnboundedContinuousTensorSpec(1),
            "truncated": UnboundedContinuousTensorSpec(1),
            "terminated_below_bound": UnboundedContinuousTensorSpec(1),
            "terminated_above_bound": UnboundedContinuousTensorSpec(1),
            "terminated_collision": UnboundedContinuousTensorSpec(1),
            "truncated_timeout": UnboundedContinuousTensorSpec(1),
            "termination_reason_code": UnboundedContinuousTensorSpec(1, dtype=torch.long),
        }
        reward_cfg = getattr(getattr(self.cfg, "instinctRL", None), "reward", None)
        if reward_cfg is not None and getattr(reward_cfg, "enabled", False):
            from instinctRL.rewards import REWARD_COMPONENT_KEYS

            stats_spec_fields.update({
                key: UnboundedContinuousTensorSpec(1, device=self.device)
                for key in REWARD_COMPONENT_KEYS
            })
        stats_spec = CompositeSpec(stats_spec_fields).expand(self.num_envs).to(self.device)

        # instinctRL-A: Add v_cmd to info spec (critic-accessible, not actor input).
        # Body-frame velocity command for B0 baseline.
        info_spec_fields = {
            "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13), device=self.device),
            # instinctRL-A: Body-frame velocity command (critic-accessible).
            "v_cmd": UnboundedContinuousTensorSpec((1, 3), device=self.device),
            # Command-governor task fields. These are reward/critic/eval-only;
            # the actor observation remains lidar_grid + state_vec.
            "actual_velocity_b": UnboundedContinuousTensorSpec((1, 3), device=self.device),
            "r5e1_controller_command_w": UnboundedContinuousTensorSpec((1, 3), device=self.device),
            "r5e1_actual_velocity_w": UnboundedContinuousTensorSpec((1, 3), device=self.device),
            "min_clearance": UnboundedContinuousTensorSpec((1,), device=self.device),
            "command_mode_code": UnboundedContinuousTensorSpec((1,), dtype=torch.long, device=self.device),
            "command_speed": UnboundedContinuousTensorSpec((1,), device=self.device),
            "tracking_actual_error_sq": UnboundedContinuousTensorSpec((1,), device=self.device),
            "tracking_proxy_error_sq": UnboundedContinuousTensorSpec((1,), device=self.device),
            "command_preservation_ratio": UnboundedContinuousTensorSpec((1,), device=self.device),
            "null_command_speed": UnboundedContinuousTensorSpec((1,), device=self.device),
            "null_command_output_speed": UnboundedContinuousTensorSpec((1,), device=self.device),
            "command_amplification": UnboundedContinuousTensorSpec((1,), device=self.device),
            "command_amplification_active": UnboundedContinuousTensorSpec((1,), device=self.device),
            "command_amplification_horizontal": UnboundedContinuousTensorSpec((1,), device=self.device),
            "command_amplification_horizontal_active": UnboundedContinuousTensorSpec((1,), device=self.device),
            "command_amplification_vertical": UnboundedContinuousTensorSpec((1,), device=self.device),
            "command_amplification_vertical_active": UnboundedContinuousTensorSpec((1,), device=self.device),
            "height_world_z": UnboundedContinuousTensorSpec((1,), device=self.device),
            "height_floor_violation": UnboundedContinuousTensorSpec((1,), device=self.device),
            "height_ceiling_violation": UnboundedContinuousTensorSpec((1,), device=self.device),
            "height_ceiling_margin": UnboundedContinuousTensorSpec((1,), device=self.device),
            "v_cmd_z": UnboundedContinuousTensorSpec((1,), device=self.device),
            "v_final_b_z": UnboundedContinuousTensorSpec((1,), device=self.device),
            "station_keeping_drift": UnboundedContinuousTensorSpec((1,), device=self.device),
            "safety_min_clearance": UnboundedContinuousTensorSpec((1,), device=self.device),
            "safety_collision": UnboundedContinuousTensorSpec((1,), device=self.device),
            "ics_intervention": UnboundedContinuousTensorSpec((1,), device=self.device),
            "ics_violation": UnboundedContinuousTensorSpec((1,), device=self.device),
            "r5e2_collision": UnboundedContinuousTensorSpec((1,), device=self.device),
            "r5e2_terminated_collision": UnboundedContinuousTensorSpec((1,), device=self.device),
            "r5e2_below_bound": UnboundedContinuousTensorSpec((1,), device=self.device),
            "r5e2_above_bound": UnboundedContinuousTensorSpec((1,), device=self.device),
            "r5e2_root_z": UnboundedContinuousTensorSpec((1,), device=self.device),
            "r5e2_below_bound_adjacent": UnboundedContinuousTensorSpec((1,), device=self.device),
            "r5e2_ceiling_adjacent": UnboundedContinuousTensorSpec((1,), device=self.device),
            "r5e2_height_adjacent": UnboundedContinuousTensorSpec((1,), device=self.device),
            "r5e2_min_clearance": UnboundedContinuousTensorSpec((1,), device=self.device),
            "r5e2_min_clearance_source_available": UnboundedContinuousTensorSpec((1,), device=self.device),
            "r5e2_missing_clearance_source": UnboundedContinuousTensorSpec((1,), device=self.device),
            "r5e2_lidar_collision_evidence": UnboundedContinuousTensorSpec((1,), device=self.device),
            "r5e2_contact_telemetry_available": UnboundedContinuousTensorSpec((1,), device=self.device),
            "r5e2_missing_contact_telemetry": UnboundedContinuousTensorSpec((1,), device=self.device),
            "r5e2_ground_contact": UnboundedContinuousTensorSpec((1,), device=self.device),
            "r5e2_collision_termination_same_step": UnboundedContinuousTensorSpec((1,), device=self.device),
            "r5e2_collision_without_termination": UnboundedContinuousTensorSpec((1,), device=self.device),
            "r5e2_termination_collision_without_collision": UnboundedContinuousTensorSpec((1,), device=self.device),
            "r5e2_reason_code": UnboundedContinuousTensorSpec((1,), dtype=torch.long, device=self.device),
            "r5e2_reason_below_bound_adjacent": UnboundedContinuousTensorSpec((1,), device=self.device),
            "r5e2_reason_ceiling": UnboundedContinuousTensorSpec((1,), device=self.device),
            "r5e2_reason_obstacle": UnboundedContinuousTensorSpec((1,), device=self.device),
            "r5e2_reason_ground": UnboundedContinuousTensorSpec((1,), device=self.device),
            "r5e2_reason_unknown": UnboundedContinuousTensorSpec((1,), device=self.device),
        }
        info_spec_fields.update({
            key: UnboundedContinuousTensorSpec((1,), device=self.device)
            for key in R5E3_DIAGNOSTIC_FIELDS
        })
        anchor_cfg = getattr(getattr(self.cfg, "instinctRL", None), "anchor", None)
        if anchor_cfg is not None and getattr(anchor_cfg, "enabled", False):
            info_spec_fields.update({
                # instinctRL-C: Scalar anchor diagnostics only. Dense anchor
                # tensors stay in self.anchor_outputs and never enter actor obs.
                "anchor_active": UnboundedContinuousTensorSpec((1,), device=self.device),
                "anchor_loss": UnboundedContinuousTensorSpec((1,), device=self.device),
                "anchor_valid_fraction": UnboundedContinuousTensorSpec((1,), device=self.device),
                "anchor_error_mean": UnboundedContinuousTensorSpec((1,), device=self.device),
                "anchor_error_max": UnboundedContinuousTensorSpec((1,), device=self.device),
                "anchor_hold_steps": UnboundedContinuousTensorSpec((1,), device=self.device),
                "anchor_activation_count": UnboundedContinuousTensorSpec((1,), device=self.device),
                "anchor_reset_reason": UnboundedContinuousTensorSpec(
                    (1,), dtype=torch.long, device=self.device
                ),
            })
        observability_cfg = getattr(getattr(self.cfg, "instinctRL", None), "observability", None)
        if observability_cfg is not None and getattr(observability_cfg, "enabled", False):
            info_spec_fields.update({
                # instinctRL-D: Scalar observability diagnostics only. Dense
                # Jacobians/SVD internals stay in self.observability_outputs.
                "observability_valid_fraction": UnboundedContinuousTensorSpec((1,), device=self.device),
                "observability_weighted_valid_fraction": UnboundedContinuousTensorSpec((1,), device=self.device),
                "observability_rank": UnboundedContinuousTensorSpec((1,), device=self.device),
                "observability_sigma_min": UnboundedContinuousTensorSpec((1,), device=self.device),
                "observability_sigma_max": UnboundedContinuousTensorSpec((1,), device=self.device),
                "observability_condition_number": UnboundedContinuousTensorSpec((1,), device=self.device),
                "observability_score": UnboundedContinuousTensorSpec((1,), device=self.device),
                "observability_drift_projection": UnboundedContinuousTensorSpec((1,), device=self.device),
                "observability_drift_norm": UnboundedContinuousTensorSpec((1,), device=self.device),
                "observability_is_proxy": UnboundedContinuousTensorSpec((1,), device=self.device),
                "observability_mode_code": UnboundedContinuousTensorSpec(
                    (1,), dtype=torch.long, device=self.device
                ),
                "observability_scenario_id": UnboundedContinuousTensorSpec(
                    (1,), dtype=torch.long, device=self.device
                ),
            })
        ics_cfg = getattr(getattr(self.cfg, "instinctRL", None), "ics", None)
        if ics_cfg is not None and getattr(ics_cfg, "enabled", False):
            info_spec_fields.update({
                # instinctRL-E: Scalar attenuation diagnostics only. Dense
                # per-beam internals stay in self.ics_outputs.
                "ics_beta": UnboundedContinuousTensorSpec((1,), device=self.device),
                "ics_active_beam_count": UnboundedContinuousTensorSpec((1,), device=self.device),
                "ics_min_clearance": UnboundedContinuousTensorSpec((1,), device=self.device),
                "ics_worst_margin": UnboundedContinuousTensorSpec((1,), device=self.device),
                "ics_worst_beam_index": UnboundedContinuousTensorSpec((1,), dtype=torch.long, device=self.device),
                "ics_emergency": UnboundedContinuousTensorSpec((1,), device=self.device),
                "ics_command_speed": UnboundedContinuousTensorSpec((1,), device=self.device),
                "ics_brake_speed": UnboundedContinuousTensorSpec((1,), device=self.device),
                "ics_final_speed": UnboundedContinuousTensorSpec((1,), device=self.device),
                "ics_clip_ratio": UnboundedContinuousTensorSpec((1,), device=self.device),
                "ics_downward_active": UnboundedContinuousTensorSpec((1,), device=self.device),
                "ics_downward_has_ray": UnboundedContinuousTensorSpec((1,), device=self.device),
                "ics_downward_beta": UnboundedContinuousTensorSpec((1,), device=self.device),
                "ics_downward_min_clearance": UnboundedContinuousTensorSpec((1,), device=self.device),
                "ics_downward_pre_z": UnboundedContinuousTensorSpec((1,), device=self.device),
                "ics_downward_post_z": UnboundedContinuousTensorSpec((1,), device=self.device),
                "ics_downward_z_delta_abs": UnboundedContinuousTensorSpec((1,), device=self.device),
                "ics_downward_attenuation_ratio": UnboundedContinuousTensorSpec((1,), device=self.device),
                "ics_residual_preemption_trigger": UnboundedContinuousTensorSpec((1,), device=self.device),
                "ics_residual_preemption_range_rate_available": UnboundedContinuousTensorSpec((1,), device=self.device),
            })
        info_spec = CompositeSpec(info_spec_fields).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec
        self.stats = stats_spec.zero()
        self.info = info_spec.zero()

    
    def reset_target(self, env_ids: torch.Tensor):
        if (self.training):
            # decide which side
            masks = torch.tensor([[1., 0., 1.], [1., 0., 1.], [0., 1., 1.], [0., 1., 1.]], dtype=torch.float, device=self.device)
            shifts = torch.tensor([[0., 24., 0.], [0., -24., 0.], [24., 0., 0.], [-24., 0., 0.]], dtype=torch.float, device=self.device)
            mask_indices = np.random.randint(0, masks.size(0), size=env_ids.size(0))
            selected_masks = masks[mask_indices].unsqueeze(1)
            selected_shifts = shifts[mask_indices].unsqueeze(1)


            # generate random positions
            target_pos = 48. * torch.rand(env_ids.size(0), 1, 3, dtype=torch.float, device=self.device) + (-24.)
            heights = 0.5 + torch.rand(env_ids.size(0), dtype=torch.float, device=self.device) * (2.5 - 0.5)
            target_pos[:, 0, 2] = heights# height
            target_pos = target_pos * selected_masks + selected_shifts
            
            # apply target pos
            self.target_pos[env_ids] = target_pos

            # self.target_pos[:, 0, 0] = torch.linspace(-0.5, 0.5, self.num_envs) * 32.
            # self.target_pos[:, 0, 1] = 24.
            # self.target_pos[:, 0, 2] = 2.    
        else:
            self.target_pos[:, 0, 0] = torch.linspace(-0.5, 0.5, self.num_envs) * 32.
            self.target_pos[:, 0, 1] = -24.
            self.target_pos[:, 0, 2] = 2.            


    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids, self.training)
        self.reset_target(env_ids)
        if (self.training):
            masks = torch.tensor([[1., 0., 1.], [1., 0., 1.], [0., 1., 1.], [0., 1., 1.]], dtype=torch.float, device=self.device)
            shifts = torch.tensor([[0., 24., 0.], [0., -24., 0.], [24., 0., 0.], [-24., 0., 0.]], dtype=torch.float, device=self.device)
            mask_indices = np.random.randint(0, masks.size(0), size=env_ids.size(0))
            selected_masks = masks[mask_indices].unsqueeze(1)
            selected_shifts = shifts[mask_indices].unsqueeze(1)

            # generate random positions
            pos = 48. * torch.rand(env_ids.size(0), 1, 3, dtype=torch.float, device=self.device) + (-24.)
            heights = 0.5 + torch.rand(env_ids.size(0), dtype=torch.float, device=self.device) * (2.5 - 0.5)
            pos[:, 0, 2] = heights# height
            pos = pos * selected_masks + selected_shifts
            
            # pos = torch.zeros(len(env_ids), 1, 3, device=self.device)
            # pos[:, 0, 0] = (env_ids / self.num_envs - 0.5) * 32.
            # pos[:, 0, 1] = -24.
            # pos[:, 0, 2] = 2.
        else:
            pos = torch.zeros(len(env_ids), 1, 3, device=self.device)
            pos[:, 0, 0] = (env_ids / self.num_envs - 0.5) * 32.
            pos[:, 0, 1] = 24.
            pos[:, 0, 2] = 2.
        
        # Coordinate change: after reset, the drone's target direction should be changed
        self.target_dir[env_ids] = self.target_pos[env_ids] - pos

        # Coordinate change: after reset, the drone's facing direction should face the current goal
        rpy = torch.zeros(len(env_ids), 1, 3, device=self.device)
        diff = self.target_pos[env_ids] - pos
        facing_yaw = torch.atan2(diff[..., 1], diff[..., 0])
        rpy[..., 2] = facing_yaw

        rot = euler_to_quaternion(rpy)
        self.drone.set_world_poses(pos, rot, env_ids)
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)
        self.prev_drone_vel_w[env_ids] = 0.
        if hasattr(self, "_prev_issued_action_body"):
            self._prev_issued_action_body[env_ids] = 0.0
        if hasattr(self, "_has_prev_issued_action_body"):
            self._has_prev_issued_action_body[env_ids] = False
        if hasattr(self, "_reward_prev_v_final_body"):
            self._reward_prev_v_final_body[env_ids] = 0.0
        if hasattr(self, "_station_origin_pos_w"):
            self._station_origin_pos_w[env_ids] = pos[:, 0, :].detach()
        if hasattr(self, "_v_cmd"):
            self._v_cmd[env_ids] = 0.0
        if hasattr(self, "_nearest_obstacle_vector_b"):
            self._nearest_obstacle_vector_b[env_ids] = torch.tensor(
                [1.0, 0.0, 0.0],
                dtype=self._nearest_obstacle_vector_b.dtype,
                device=self.device,
            )
        if hasattr(self, "_command_generator"):
            self._command_generator.timers[env_ids] = 0.0
            self._command_generator.current_modes[env_ids] = COMMAND_MODE_RECOVERY
        if hasattr(self, "_obs_builder"):
            self._obs_builder.reset_history(env_ids)
        if hasattr(self, "_anchor_manager"):
            from instinctRL.anchor import ANCHOR_RESET_EPISODE
            self._anchor_manager.reset(env_ids, reason=ANCHOR_RESET_EPISODE)
        self.height_range[env_ids, 0, 0] = torch.min(pos[:, 0, 2], self.target_pos[env_ids, 0, 2])
        self.height_range[env_ids, 0, 1] = torch.max(pos[:, 0, 2], self.target_pos[env_ids, 0, 2])

        for _, value in self.stats.items():
            value[env_ids] = 0
        
    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")] 
        self.drone.apply_action(actions) 

    def _post_sim_step(self, tensordict: TensorDictBase):
        if (self.cfg.env_dyn.num_obstacles != 0):
            self.move_dynamic_obstacle()
        self.lidar.update(self.dt)

    def _ensure_command_generator(self):
        """Create the adversarial command generator if this run needs it."""
        if hasattr(self, "_command_generator"):
            return
        from command_generator import AdversarialCommandGenerator

        self._command_generator = AdversarialCommandGenerator(
            self.num_envs,
            torch.device(self.device),
            max_vel=self._command_max_vel,
            dt=self.dt,
        )

    def configure_instinctrl_eval_pass(
        self,
        *,
        command_source: str | None = None,
        command_frame_count: int | None = None,
        command_curriculum_profile: str | None = None,
        scenario_id_code: int | None = None,
    ):
        """Configure eval-only command/scenario controls before a rollout reset."""
        if command_source is not None:
            if command_source not in {"curriculum_generator", "basic_random", "scripted_eval"}:
                raise ValueError(f"Unsupported eval command source={command_source!r}")
            self._command_source = command_source
            if command_source == "curriculum_generator":
                self._ensure_command_generator()
        if command_frame_count is not None:
            self._command_frame_count = int(command_frame_count)
        if command_curriculum_profile is not None:
            self._command_curriculum_profile = str(command_curriculum_profile)
        if scenario_id_code is not None:
            self._instinctrl_eval_scenario_id_code = int(scenario_id_code)

    def set_prev_issued_action_body(self, action_body: torch.Tensor):
        """Store the last body-frame command issued to the controller."""
        if action_body.dim() == 3 and action_body.shape[1] == 1:
            action_body = action_body.squeeze(1)
        self._prev_issued_action_body[:] = action_body.reshape(self.num_envs, 3).detach()
        self._has_prev_issued_action_body[:] = True

    def get_instinctrl_range_history(self, copy: bool = True):
        """Return MID360 range/mask/weight history from the observation builder."""
        if not hasattr(self, "_obs_builder"):
            raise RuntimeError("instinctRL observation builder is not available")
        return self._obs_builder.get_history(copy=copy)

    def record_instinctrl_ics_output(self, ics_out):
        """Store ICS scalar metrics in info and dense tensors internally."""
        self.ics_outputs = ics_out.cache
        for key, value in ics_out.metrics.items():
            if key in self.info.keys():
                self.info[key][:] = value
        if "ics_worst_beam_index" in self.info.keys():
            index = ics_out.cache.get("ics_worst_beam_index")
            if index is not None:
                self.info["ics_worst_beam_index"][:] = index.reshape(self.num_envs, 1)

    def _update_instinctrl_command(self):
        """Update body-frame command for the command-governor task."""
        if self._command_source == "curriculum_generator":
            self._command_frame_count += int(self.num_envs)
            probabilities = command_curriculum_probabilities(
                self._command_frame_count,
                profile=getattr(self, "_command_curriculum_profile", "default"),
            )
            v_cmd = self._command_generator.update_commands(
                self.root_state[..., :3].reshape(self.num_envs, 3),
                self.root_state[..., 7:10].reshape(self.num_envs, 3),
                self._nearest_obstacle_vector_b,
                probabilities=probabilities,
            )
            self._v_cmd = v_cmd.reshape(self.num_envs, 1, 3)
            if "command_mode_code" in self.info.keys():
                self.info["command_mode_code"][:] = self._command_generator.current_modes.reshape(
                    self.num_envs, 1
                )
        elif self._command_source == "scripted_eval":
            self._v_cmd.zero_()
            if "command_mode_code" in self.info.keys():
                self.info["command_mode_code"].fill_(COMMAND_MODE_RECOVERY)
        else:
            if not hasattr(self, "_v_cmd_step_count"):
                self._v_cmd_step_count = 0
            self._v_cmd_step_count += 1
            if self._v_cmd_step_count % 125 == 1:
                self._v_cmd = (
                    torch.rand(self.num_envs, 1, 3, device=self.device) * 2.0 - 1.0
                ) * 0.5
                self._v_cmd[..., 2] *= 0.3
            if "command_mode_code" in self.info.keys():
                self.info["command_mode_code"].fill_(COMMAND_MODE_NORMAL)
        if "command_speed" in self.info.keys():
            self.info["command_speed"][:] = self._v_cmd.norm(dim=-1)
    
    # ============================================
    # 计算观测和奖励（每步调用）
    # ============================================
    def _compute_state_and_obs(self):
        """
        计算当前状态、观测和奖励
        
        返回:
            TensorDict: 包含观测、统计信息、信息的字典
                - ("agents", "observation"): 策略网络的输入
                    - "lidar": [num_envs, 1, 36, 4] LiDAR 数据
                    - "state": [num_envs, 8] 无人机状态
                    - "dynamic_obstacle": [num_envs, 1, N, 10] 动态障碍物信息
                - "stats": 统计信息（return, collision, etc.）
                - "info": 额外信息（用于控制器）
        """
        # 获取无人机状态（世界坐标系）
        # 包含：位置、姿态（四元数）、速度、角速度、朝向、上方向、电机推力
        self.root_state = self.drone.get_state(env_frame=False)
        self.info["drone_state"][:] = self.root_state[..., :13]  # 保存状态信息
        if "r5e1_actual_velocity_w" in self.info.keys():
            self.info["r5e1_actual_velocity_w"][:] = self.root_state[..., 7:10]
        root_flat = self.root_state[..., :13].reshape(self.num_envs, 13)
        if hasattr(self, "_station_origin_pos_w"):
            station_drift_w = root_flat[:, :3] - self._station_origin_pos_w
            station_drift_b = world_to_body_velocity(station_drift_w, root_flat[:, 3:7])
            if "station_keeping_drift" in self.info.keys():
                self.info["station_keeping_drift"][:] = station_drift_w.norm(
                    dim=-1, keepdim=True
                )
        else:
            station_drift_b = None

        # ============================================
        # 网络输入 I：MID360 LiDAR → observation builder
        # ============================================
        # instinctRL-B: Use MID360ObservationBuilder for actor-clean pipeline.
        # Produces: raw range r_t, mask m_t, staleness-weighted reliability w_t,
        # IMU cues, v_cmd, prev_action, frame_age, sim_time, history buffer.
        if hasattr(self, "_obs_builder"):
            self._update_instinctrl_command()

            # Build observation
            obs_frame = self._obs_builder.build(
                ray_hits_w=self.lidar.data.ray_hits_w,
                lidar_pos_w=self.lidar.data.pos_w,
                drone_state=self.root_state[..., :13],
                v_cmd=self._v_cmd,
                dt=self.dt,
                num_envs=self.num_envs,
                prev_action=self._prev_issued_action_body,
            )

            if hasattr(self, "_anchor_manager"):
                anchor_out = self._anchor_manager.step(
                    obs_frame["range"],
                    obs_frame["mask"],
                    obs_frame["weight"],
                    self._v_cmd,
                )
                self.anchor_outputs = anchor_out.cache
                for key, value in anchor_out.metrics.items():
                    self.info[key][:] = value

            if hasattr(self, "_observability_logger"):
                scenario_id = torch.full(
                    (self.num_envs,),
                    int(getattr(self, "_instinctrl_eval_scenario_id_code", 0)),
                    dtype=torch.long,
                    device=self.device,
                )
                observability_out = self._observability_logger.compute(
                    ray_directions_b=self._mid360_ray_dirs_b,
                    valid_mask=obs_frame["mask"].reshape(self.num_envs, -1),
                    reliability_weight=obs_frame["weight"].reshape(self.num_envs, -1),
                    drift_b=station_drift_b,
                    scenario_id=scenario_id,
                )
                self.observability_outputs = observability_out.cache
                for key, value in observability_out.metrics.items():
                    if key in self.info.keys():
                        self.info[key][:] = value

            obs_hist = self._obs_builder.build_history(obs_frame)

            # Raw range for reward computation (keep self.lidar_scan for backward compat)
            self.lidar_scan = obs_frame["range"].unsqueeze(1)  # [N, 1, H, V]
            self._nearest_obstacle_vector_b = nearest_obstacle_vector_from_scan(
                ranges=obs_frame["range"],
                mask=obs_frame["mask"],
                ray_directions_b=self._mid360_ray_dirs_b,
            ).detach()

            # Store v_cmd in info for critic/debug
            self.info["v_cmd"] = self._v_cmd.clone()
        else:
            # Fallback: danger-coded LiDAR (non-instinctRL mode)
            self.lidar_scan = self.lidar_range - (
                (self.lidar.data.ray_hits_w - self.lidar.data.pos_w.unsqueeze(1))
                .norm(dim=-1)
                .clamp_max(self.lidar_range)
                .reshape(self.num_envs, 1, *self.lidar_resolution)
            )

        # Optional render for LiDAR
        if self._should_render(0):
            self.debug_draw.clear()
            x = self.lidar.data.pos_w[0]
            # set_camera_view(
            #     eye=x.cpu() + torch.as_tensor(self.cfg.viewer.eye),
            #     target=x.cpu() + torch.as_tensor(self.cfg.viewer.lookat)                        
            # )
            v = (self.lidar.data.ray_hits_w[0] - x).reshape(*self.lidar_resolution, 3)
            # self.debug_draw.vector(x.expand_as(v[:, 0]), v[:, 0])
            # self.debug_draw.vector(x.expand_as(v[:, -1]), v[:, -1])
            self.debug_draw.vector(x.expand_as(v[:, 0])[0], v[0, 0])

        # ============================================
        # 网络输入 II：无人机内部状态
        # ============================================
        # 这些状态描述无人机与目标的关系
        
        # a. 距离信息（水平和垂直分离）
        rpos = self.target_pos - self.root_state[..., :3]  # 相对位置向量
        distance = rpos.norm(dim=-1, keepdim=True)  # 3D 距离
        distance_2d = rpos[..., :2].norm(dim=-1, keepdim=True)  # 水平距离
        distance_z = rpos[..., 2].unsqueeze(-1)  # 垂直距离（高度差）

        # instinctRL-A: Compute raw MID360 range for backward compat (lidar_raw_range)
        self.lidar_raw_range = (
            (self.lidar.data.ray_hits_w - self.lidar.data.pos_w.unsqueeze(1))
            .norm(dim=-1)
            .clamp_max(self.lidar_range)
            .reshape(self.num_envs, 1, *self.lidar_resolution)
        )
        
        # b. 指向目标的单位方向向量（在目标坐标系下）
        # 为什么要坐标变换？
        # - 在目标坐标系下，"向前"总是朝向目标
        # - 策略网络更容易学习：只需学"向前飞"，而非"向北飞"或"向南飞"
        target_dir_2d = self.target_dir.clone()
        target_dir_2d[..., 2] = 0  # 只保留水平方向

        rpos_clipped = rpos / distance.clamp(1e-6)  # 单位方向向量（归一化）
        rpos_clipped_g = vec_to_new_frame(rpos_clipped, target_dir_2d)  # 转到目标坐标系
        
        # c. 速度（在目标坐标系下）
        vel_w = self.root_state[..., 7:10]  # 世界坐标系速度
        vel_g = vec_to_new_frame(vel_w, target_dir_2d)  # 转到目标坐标系

        # 拼接为无人机状态：[方向(3) + 水平距离(1) + 垂直距离(1) + 速度(3)] = 8维
        drone_state = torch.cat([rpos_clipped_g, distance_2d, distance_z, vel_g], dim=-1).squeeze(1)

        if (self.cfg.env_dyn.num_obstacles != 0):
            # ---------Network Input III: Dynamic obstacle states--------
            # ------------------------------------------------------------
            # a. Closest N obstacles relative position in the goal frame 
            # Find the N closest and within range obstacles for each drone
            dyn_obs_pos_expanded = self.dyn_obs_state[..., :3].unsqueeze(0).repeat(self.num_envs, 1, 1)
            dyn_obs_rpos_expanded = dyn_obs_pos_expanded[..., :3] - self.root_state[..., :3] 
            dyn_obs_rpos_expanded[:, int(self.dyn_obs_state.size(0)/2):, 2] = 0.
            dyn_obs_distance_2d = torch.norm(dyn_obs_rpos_expanded[..., :2], dim=2)  # Shape: (1000, 40). calculate 2d distance to each obstacle for all drones
            _, closest_dyn_obs_idx = torch.topk(dyn_obs_distance_2d, self.cfg.algo.feature_extractor.dyn_obs_num, dim=1, largest=False) # pick top N closest obstacle index
            dyn_obs_range_mask = dyn_obs_distance_2d.gather(1, closest_dyn_obs_idx) > self.lidar_range

            # relative distance of obstacles in the goal frame
            closest_dyn_obs_rpos = torch.gather(dyn_obs_rpos_expanded, 1, closest_dyn_obs_idx.unsqueeze(-1).expand(-1, -1, 3))
            closest_dyn_obs_rpos_g = vec_to_new_frame(closest_dyn_obs_rpos, target_dir_2d) 
            closest_dyn_obs_rpos_g[dyn_obs_range_mask] = 0. # exclude out of range obstacles
            closest_dyn_obs_distance = closest_dyn_obs_rpos.norm(dim=-1, keepdim=True)
            closest_dyn_obs_distance_2d = closest_dyn_obs_rpos_g[..., :2].norm(dim=-1, keepdim=True)
            closest_dyn_obs_distance_z = closest_dyn_obs_rpos_g[..., 2].unsqueeze(-1)
            closest_dyn_obs_rpos_gn = closest_dyn_obs_rpos_g / closest_dyn_obs_distance.clamp(1e-6)

            # b. Velocity in the goal frame for the dynamic obstacles
            closest_dyn_obs_vel = self.dyn_obs_vel[closest_dyn_obs_idx]
            closest_dyn_obs_vel[dyn_obs_range_mask] = 0.
            closest_dyn_obs_vel_g = vec_to_new_frame(closest_dyn_obs_vel, target_dir_2d) 

            # c. Size of dynamic obstacles in category
            closest_dyn_obs_size = self.dyn_obs_size[closest_dyn_obs_idx] # the acutal size

            closest_dyn_obs_width = closest_dyn_obs_size[..., 0].unsqueeze(-1)
            closest_dyn_obs_width_category = closest_dyn_obs_width / self.dyn_obs_width_res - 1. # convert to category: [0, 1, 2, 3]
            closest_dyn_obs_width_category[dyn_obs_range_mask] = 0.

            closest_dyn_obs_height = closest_dyn_obs_size[..., 2].unsqueeze(-1)
            closest_dyn_obs_height_category = torch.where(closest_dyn_obs_height > self.max_obs_3d_height, torch.tensor(0.0), closest_dyn_obs_height)
            closest_dyn_obs_height_category[dyn_obs_range_mask] = 0.

            # concatenate all for dynamic obstacles
            # dyn_obs_states = torch.cat([closest_dyn_obs_rpos_g, closest_dyn_obs_vel_g, closest_dyn_obs_width_category, closest_dyn_obs_height_category], dim=-1).unsqueeze(1)
            dyn_obs_states = torch.cat([closest_dyn_obs_rpos_gn, closest_dyn_obs_distance_2d, closest_dyn_obs_distance_z, closest_dyn_obs_vel_g, closest_dyn_obs_width_category, closest_dyn_obs_height_category], dim=-1).unsqueeze(1)

            # check dynamic obstacle collision for later reward
            closest_dyn_obs_distance_2d_collsion = closest_dyn_obs_rpos[..., :2].norm(dim=-1, keepdim=True)
            closest_dyn_obs_distance_2d_collsion[dyn_obs_range_mask] = float('inf')
            closest_dyn_obs_distance_zn_collision = closest_dyn_obs_rpos[..., 2].unsqueeze(-1).norm(dim=-1, keepdim=True)
            closest_dyn_obs_distance_zn_collision[dyn_obs_range_mask] = float('inf')
            dynamic_collision_2d = closest_dyn_obs_distance_2d_collsion <= (closest_dyn_obs_width/2. + 0.3)
            dynamic_collision_z = closest_dyn_obs_distance_zn_collision <= (closest_dyn_obs_height/2. + 0.3)
            dynamic_collision_each = dynamic_collision_2d & dynamic_collision_z
            dynamic_collision = torch.any(dynamic_collision_each, dim=1)

            # distance to dynamic obstacle for reward calculation (not 100% correct in math but should be good enough for approximation)
            closest_dyn_obs_distance_reward = closest_dyn_obs_rpos.norm(dim=-1) - closest_dyn_obs_size[..., 0]/2. # for those 2D obstacle, z distance will not be considered
            closest_dyn_obs_distance_reward[dyn_obs_range_mask] = self.cfg.sensor.lidar_range
            
        else:
            dyn_obs_states = torch.zeros(self.num_envs, 1, self.cfg.algo.feature_extractor.dyn_obs_num, 10, device=self.cfg.device)
            dynamic_collision = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.cfg.device)
            
        # -----------------Network Input Final--------------
        # instinctRL-B: Actor input contract — hybrid observation.
        # lidar_grid: history-stacked range/mask/weight channels
        # state_vec:  history-stacked IMU+v_cmd+prev_action+frame_age
        # No pose, odometry, explicit velocity, map, or privileged state.
        if hasattr(self, "_obs_builder"):
            obs = {
                "lidar_grid": obs_hist["lidar_grid"],
                "state_vec": obs_hist["state_vec"],
            }
        else:
            obs = {
                "lidar": self.lidar_scan,
            }


        # ============================================
        # 奖励函数设计 ⭐ 非常重要
        # ============================================
        # 奖励 = 速度奖励 + 安全奖励 - 平滑性惩罚 - 高度惩罚
        
        # a. 静态障碍物安全奖励
        # 原理：距离越远，奖励越高（使用对数，避免奖励过大）
        # log(distance) 保证：很近时惩罚大，较远时惩罚小
        reward_safety_static = torch.log(
            (self.lidar_range - self.lidar_scan).clamp(min=1e-6, max=self.lidar_range)
        ).mean(dim=(2, 3))

        # b. 动态障碍物安全奖励
        if (self.cfg.env_dyn.num_obstacles != 0):
            reward_safety_dynamic = torch.log(
                (closest_dyn_obs_distance_reward).clamp(min=1e-6, max=self.lidar_range)
            ).mean(dim=-1, keepdim=True)

        # c. 速度奖励（朝向目标方向的速度越快，奖励越高）
        # 计算：速度 · 目标方向（点积）
        # 效果：鼓励无人机快速飞向目标
        vel_direction = rpos / distance.clamp_min(1e-6)  # 目标方向（单位向量）
        reward_vel = (self.drone.vel_w[..., :3] * vel_direction).sum(-1)
        
        # d. 平滑性惩罚（避免剧烈加速/减速）
        # 计算：||v_t - v_{t-1}||
        # 效果：鼓励平滑飞行，提高真实性
        penalty_smooth = (self.drone.vel_w[..., :3] - self.prev_drone_vel_w).norm(dim=-1)
        
        # e. 高度惩罚（避免飞得过高或过低）
        # 原因：效率低、浪费能量
        # 计算：如果超出合理高度范围，惩罚 = (超出距离)²
        penalty_height = torch.zeros(self.num_envs, 1, device=self.cfg.device)
        # 飞得太高
        too_high = self.drone.pos[..., 2] > (self.height_range[..., 1] + 0.2)
        penalty_height[too_high] = ((self.drone.pos[..., 2] - self.height_range[..., 1] - 0.2)**2)[too_high]
        # 飞得太低
        too_low = self.drone.pos[..., 2] < (self.height_range[..., 0] - 0.2)
        penalty_height[too_low] = ((self.height_range[..., 0] - 0.2 - self.drone.pos[..., 2])**2)[too_low]

        # f. 碰撞检测
        # 静态碰撞：LiDAR 检测到距离 < 0.3m
        if hasattr(self, "_obs_builder"):
            static_collision = einops.reduce(self.lidar_scan, "n 1 w h -> n 1", "min") < 0.3
        else:
            static_collision = einops.reduce(self.lidar_scan, "n 1 w h -> n 1", "max") > (self.lidar_range - 0.3)
        collision = static_collision | dynamic_collision
        r5e2_min_clearance = None
        r5e2_min_clearance_source_available = None
        r5e3_actual_velocity_b = None
        r5e3_v_final_b = None
        r5e3_raw_min_clearance = None
        r5e3_raw_min_clearance_source_available = None
        r5e3_ics_min_clearance_source_available = None
        
        # ============================================
        # 最终奖励计算（权重调优）
        # ============================================
        # reward = vel_reward + 1.0 (基础奖励)
        #          + safety_static * 1.0
        #          + safety_dynamic * 1.0
        #          - smoothness * 0.1
        #          - height_penalty * 8.0
        if hasattr(self, "_reward_computer") and hasattr(self, "_obs_builder"):
            v_cmd_b = self._v_cmd.reshape(self.num_envs, 3)
            actual_velocity_b = world_to_body_velocity(
                self.root_state[..., 7:10].reshape(self.num_envs, 3),
                self.root_state[..., 3:7].reshape(self.num_envs, 4),
            )
            self.info["actual_velocity_b"][:] = actual_velocity_b.reshape(self.num_envs, 1, 3)
            issued_v_final_b = torch.where(
                self._has_prev_issued_action_body,
                self._prev_issued_action_body,
                v_cmd_b,
            )
            prev_v_final_b = torch.where(
                self._has_prev_issued_action_body,
                self._reward_prev_v_final_body,
                issued_v_final_b,
            )
            flat_range = obs_frame["range"].reshape(self.num_envs, -1)
            flat_mask = obs_frame["mask"].reshape(self.num_envs, -1) > 0
            flat_weight = obs_frame["weight"].reshape(self.num_envs, -1).clamp(0.0, 1.0)
            valid_range = flat_mask & torch.isfinite(flat_range)
            min_clearance_source_available = valid_range.any(dim=1, keepdim=True).float()
            raw_min_clearance = torch.where(
                valid_range,
                flat_range,
                torch.full_like(flat_range, float("inf")),
            ).min(dim=1, keepdim=True).values
            min_clearance = torch.where(
                min_clearance_source_available.bool(),
                raw_min_clearance,
                torch.full_like(raw_min_clearance, self.lidar_range),
            )
            r5e2_min_clearance = torch.where(
                min_clearance_source_available.bool(),
                raw_min_clearance,
                torch.full_like(raw_min_clearance, float("nan")),
            )
            r5e2_min_clearance_source_available = min_clearance_source_available
            ics_cfg = getattr(self.cfg.instinctRL, "ics", None)
            ics_min_reliability = float(getattr(ics_cfg, "min_reliability", 0.1))
            r5e3_ics_source = valid_range & (flat_weight >= ics_min_reliability)
            r5e3_actual_velocity_b = actual_velocity_b
            r5e3_v_final_b = issued_v_final_b
            r5e3_raw_min_clearance = r5e2_min_clearance
            r5e3_raw_min_clearance_source_available = min_clearance_source_available
            r5e3_ics_min_clearance_source_available = r5e3_ics_source.any(
                dim=1,
                keepdim=True,
            ).float()
            self.info["min_clearance"][:] = min_clearance

            reward_terms = self._reward_computer.compute(
                v_cmd_b=v_cmd_b,
                v_final_b=issued_v_final_b,
                prev_v_final_b=prev_v_final_b,
                actual_velocity_b=actual_velocity_b,
                height_w=self.root_state[..., 2].reshape(self.num_envs, 1),
                anchor_loss=self.info["anchor_loss"] if "anchor_loss" in self.info.keys() else None,
                anchor_active=self.info["anchor_active"] if "anchor_active" in self.info.keys() else None,
                anchor_valid_fraction=(
                    self.info["anchor_valid_fraction"]
                    if "anchor_valid_fraction" in self.info.keys()
                    else None
                ),
                ics_beta=self.info["ics_beta"] if "ics_beta" in self.info.keys() else None,
                ics_emergency=self.info["ics_emergency"] if "ics_emergency" in self.info.keys() else None,
                ics_active_beam_count=(
                    self.info["ics_active_beam_count"]
                    if "ics_active_beam_count" in self.info.keys()
                    else None
                ),
                min_clearance=min_clearance,
                collision=collision,
            )
            self.reward = reward_terms.total
            self.reward_outputs = reward_terms.cache
            for key, value in reward_terms.components.items():
                self.stats[key] += value
            self._reward_prev_v_final_body[:] = issued_v_final_b.detach()
            reward_cfg = getattr(self.cfg.instinctRL, "reward", None)
            handbook_metrics = compute_handbook_step_metrics(
                v_cmd_b=v_cmd_b,
                actual_velocity_b=actual_velocity_b,
                v_final_b=issued_v_final_b,
                min_clearance=min_clearance,
                height_w=self.root_state[..., 2].reshape(self.num_envs, 1),
                ics_beta=self.info["ics_beta"] if "ics_beta" in self.info.keys() else None,
                ics_emergency=self.info["ics_emergency"] if "ics_emergency" in self.info.keys() else None,
                anchor_active=self.info["anchor_active"] if "anchor_active" in self.info.keys() else None,
                anchor_error_mean=(
                    self.info["anchor_error_mean"] if "anchor_error_mean" in self.info.keys() else None
                ),
                anchor_error_max=(
                    self.info["anchor_error_max"] if "anchor_error_max" in self.info.keys() else None
                ),
                anchor_loss=self.info["anchor_loss"] if "anchor_loss" in self.info.keys() else None,
                collision=collision,
                d_safe=float(getattr(getattr(self.cfg.instinctRL, "ics", None), "d_safe", 0.8)),
                height_floor=float(getattr(reward_cfg, "height_floor", 0.5)),
                height_ceiling=float(getattr(reward_cfg, "height_ceiling", 4.0)),
                command_eps=float(
                    getattr(reward_cfg, "command_eps", 1e-3)
                ),
            )
            for key, value in handbook_metrics.items():
                if key in self.info.keys():
                    self.info[key][:] = value
        else:
            if (self.cfg.env_dyn.num_obstacles != 0):
                self.reward = reward_vel + 1. + reward_safety_static * 1.0 + reward_safety_dynamic * 1.0 - penalty_smooth * 0.1 - penalty_height * 8.0
            else:
                self.reward = reward_vel + 1. + reward_safety_static * 1.0 - penalty_smooth * 0.1 - penalty_height * 8.0

        # ============================================
        # 终止条件
        # ============================================
        # 成功：到达目标（距离 < 0.5m）
        reach_goal = (distance.squeeze(-1) < 0.5)
        
        # 失败：飞出边界或碰撞
        below_bound = self.drone.pos[..., 2] < 0.2  # 低于 0.2m
        above_bound = self.drone.pos[..., 2] > 4.  # 高于 4m
        self.terminated = below_bound | above_bound | collision
        
        # 截断：达到最大步数（500 步）
        self.truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)

        # 更新前一步速度（用于下一步的平滑性计算）
        self.prev_drone_vel_w = self.drone.vel_w[..., :3].clone()

        # # -----------------Training Stats-----------------
        self.stats["return"] += self.reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        self.stats["legacy_reach_goal"] = reach_goal.float()
        self.stats["collision"] = collision.float()
        self.stats["truncated"] = self.truncated.float()
        termination_stats = compute_termination_stats(
            below_bound=below_bound,
            above_bound=above_bound,
            collision=collision,
            truncated=self.truncated,
        )
        for key, value in termination_stats.items():
            self.stats[key] = value

        if r5e2_min_clearance is not None and "r5e2_collision" in self.info.keys():
            contact_available = torch.zeros_like(r5e2_min_clearance)
            clearance_source_available = (
                r5e2_min_clearance_source_available
                if r5e2_min_clearance_source_available is not None
                else torch.zeros_like(r5e2_min_clearance)
            )
            r5e2_metrics = compute_r5e2_collision_geometry_step_metrics(
                collision=collision,
                terminated_collision=termination_stats["terminated_collision"],
                below_bound=below_bound,
                above_bound=above_bound,
                root_z=self.root_state[..., 2].reshape(self.num_envs, 1),
                min_clearance=r5e2_min_clearance,
                min_clearance_source_available=clearance_source_available,
                contact_telemetry_available=contact_available,
                ground_contact=contact_available,
            )
            for key, value in r5e2_metrics.items():
                if key in self.info.keys():
                    self.info[key][:] = value

        if r5e3_v_final_b is not None and "r5e3_raw_min_clearance" in self.info.keys():
            reward_cfg = getattr(getattr(self.cfg, "instinctRL", None), "reward", None)
            ics_cfg = getattr(getattr(self.cfg, "instinctRL", None), "ics", None)
            contact_available = (
                torch.zeros_like(r5e3_raw_min_clearance)
                if r5e3_raw_min_clearance is not None
                else None
            )
            r5e3_metrics = compute_r5e3_braking_residual_step_metrics(
                v_final_b=r5e3_v_final_b,
                actual_velocity_b=r5e3_actual_velocity_b,
                raw_min_clearance=r5e3_raw_min_clearance,
                ics_min_clearance=(
                    self.info["ics_min_clearance"]
                    if "ics_min_clearance" in self.info.keys()
                    else None
                ),
                raw_min_clearance_source_available=r5e3_raw_min_clearance_source_available,
                ics_min_clearance_source_available=r5e3_ics_min_clearance_source_available,
                ics_beta=self.info["ics_beta"] if "ics_beta" in self.info.keys() else None,
                ics_emergency=(
                    self.info["ics_emergency"]
                    if "ics_emergency" in self.info.keys()
                    else None
                ),
                contact_telemetry_available=contact_available,
                ics_worst_beam_index=(
                    self.info["ics_worst_beam_index"]
                    if "ics_worst_beam_index" in self.info.keys()
                    else None
                ),
                ray_directions_b=(
                    self._mid360_ray_dirs_b
                    if hasattr(self, "_mid360_ray_dirs_b")
                    else None
                ),
                collision_clearance_threshold=0.3,
                emergency_clearance=float(getattr(ics_cfg, "emergency_clearance", 0.25)),
                d_safe=float(getattr(ics_cfg, "d_safe", 0.8)),
                a_max=float(getattr(ics_cfg, "a_max", 2.0)),
                latency_sec=float(getattr(ics_cfg, "latency_sec", 0.0)),
                command_eps=float(getattr(reward_cfg, "command_eps", 1e-3)),
                low_beta_threshold=float(getattr(ics_cfg, "low_beta_threshold", 0.999)),
            )
            for key, value in r5e3_metrics.items():
                if key in self.info.keys():
                    self.info[key][:] = value

        return TensorDict({
            "agents": TensorDict(
                {
                    "observation": obs,
                }, 
                [self.num_envs]
            ),
            "stats": self.stats.clone(),
            "info": self.info
        }, self.batch_size)

    def _compute_reward_and_done(self):
        reward = self.reward
        terminated = self.terminated
        truncated = self.truncated
        return TensorDict(
            {
                "agents": {
                    "reward": reward
                },
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )
