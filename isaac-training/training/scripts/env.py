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
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec
from omni_drones.envs.isaac_env import IsaacEnv, AgentSpec
import omni.isaac.orbit.sim as sim_utils
from omni_drones.robots.drone import MultirotorBase
from omni.isaac.orbit.assets import AssetBaseCfg
from omni.isaac.orbit.terrains import TerrainImporterCfg, TerrainImporter, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg
from omni_drones.utils.torch import euler_to_quaternion, quat_axis
from omni.isaac.orbit.sensors import RayCaster, RayCasterCfg, patterns
from omni.isaac.core.utils.viewports import set_camera_view
from utils import vec_to_new_frame, vec_to_world, construct_input
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.orbit.sim as sim_utils
import omni.isaac.orbit.utils.math as math_utils
from omni.isaac.orbit.assets import RigidObject, RigidObjectCfg
import time

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
        print("[Navigation Environment]: Initializing Env...")
        
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
        super().__init__(cfg, cfg.headless)
        
        # ============================================
        # 第 3 步：初始化无人机
        # ============================================
        self.drone.initialize()  # 初始化无人机物理属性
        self.init_vels = torch.zeros_like(self.drone.get_velocities())  # 初始速度为 0

        # ============================================
        # 第 4 步：初始化 LiDAR 传感器 ⭐ 重要
        # ============================================
        ray_caster_cfg = RayCasterCfg(
            # 绑定到无人机的 base_link（所有环境的所有无人机）
            prim_path="/World/envs/env_.*/Hummingbird_0/base_link",
            
            # 传感器相对于 base_link 的偏移（这里是原点）
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            
            # 只跟随偏航角（yaw），不跟随俯仰和滚转
            # 原因：让 LiDAR 保持水平，更稳定
            attach_yaw_only=True,
            
            # 使用 Bpearl 激光雷达的扫描模式
            pattern_cfg=patterns.BpearlPatternCfg(
                horizontal_res=self.lidar_hres,  # 水平分辨率：10°
                # 垂直角度：从 -10° 到 20°，均匀分布 4 条射线
                vertical_ray_angles=torch.linspace(*self.lidar_vfov, self.lidar_vbeams) 
            ),
            
            debug_vis=False,  # 不可视化射线（提高性能）
            
            # 检测的对象：只检测地面（静态障碍物在地面上）
            mesh_prim_paths=["/World/ground"],
        )
        self.lidar = RayCaster(ray_caster_cfg)
        self.lidar._initialize_impl()  # 初始化射线投射器
        self.lidar_resolution = (self.lidar_hbeams, self.lidar_vbeams)  # (36, 4)
        
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
        cfg_ground = sim_utils.GroundPlaneCfg(
            color=(0.1, 0.1, 0.1),  # 深灰色
            size=(300., 300.)  # 300m × 300m
        )
        cfg_ground.func("/World/defaultGroundPlane", cfg_ground, translation=(0, 0, 0.01))

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
            debug_vis=True,  # 显示调试可视化
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


    def _set_specs(self):
        observation_dim = 8
        num_dim_each_dyn_obs_state = 10

        # Observation Spec
        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": CompositeSpec({
                    "state": UnboundedContinuousTensorSpec((observation_dim,), device=self.device), 
                    "lidar": UnboundedContinuousTensorSpec((1, self.lidar_hbeams, self.lidar_vbeams), device=self.device),
                    "direction": UnboundedContinuousTensorSpec((1, 3), device=self.device),
                    "dynamic_obstacle": UnboundedContinuousTensorSpec((1, self.cfg.algo.feature_extractor.dyn_obs_num, num_dim_each_dyn_obs_state), device=self.device),
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


        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "reach_goal": UnboundedContinuousTensorSpec(1),
            "collision": UnboundedContinuousTensorSpec(1),
            "truncated": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)

        info_spec = CompositeSpec({
            "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13), device=self.device),
        }).expand(self.num_envs).to(self.device)
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
        self.height_range[env_ids, 0, 0] = torch.min(pos[:, 0, 2], self.target_pos[env_ids, 0, 2])
        self.height_range[env_ids, 0, 1] = torch.max(pos[:, 0, 2], self.target_pos[env_ids, 0, 2])

        self.stats[env_ids] = 0.  
        
    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")] 
        self.drone.apply_action(actions) 

    def _post_sim_step(self, tensordict: TensorDictBase):
        if (self.cfg.env_dyn.num_obstacles != 0):
            self.move_dynamic_obstacle()
        self.lidar.update(self.dt)
    
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

        # ============================================
        # 网络输入 I：LiDAR 数据 ⭐ 关键传感器
        # ============================================
        # LiDAR 原始数据：射线击中点的世界坐标
        # 我们需要计算：距离 = ||ray_hits - lidar_pos||
        # 然后转换为："剩余距离" = lidar_range - 实际距离
        # 作用：越近的障碍物值越大（便于网络学习）
        self.lidar_scan = self.lidar_range - (
            (self.lidar.data.ray_hits_w - self.lidar.data.pos_w.unsqueeze(1))
            .norm(dim=-1)  # 计算距离
            .clamp_max(self.lidar_range)  # 限制在最大范围内
            .reshape(self.num_envs, 1, *self.lidar_resolution)  # [num_envs, 1, 36, 4]
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
        obs = {
            "state": drone_state,
            "lidar": self.lidar_scan,
            "direction": target_dir_2d,
            "dynamic_obstacle": dyn_obs_states
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
        static_collision = einops.reduce(self.lidar_scan, "n 1 w h -> n 1", "max") > (self.lidar_range - 0.3)
        collision = static_collision | dynamic_collision
        
        # ============================================
        # 最终奖励计算（权重调优）
        # ============================================
        # reward = vel_reward + 1.0 (基础奖励)
        #          + safety_static * 1.0
        #          + safety_dynamic * 1.0
        #          - smoothness * 0.1
        #          - height_penalty * 8.0
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
        self.stats["reach_goal"] = reach_goal.float()
        self.stats["collision"] = collision.float()
        self.stats["truncated"] = self.truncated.float()

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
