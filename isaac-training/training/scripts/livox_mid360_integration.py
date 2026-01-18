"""
Livox Mid-360 LiDAR 传感器 - Isaac Sim 集成
=============================================
在 NavigationEnv 中使用 Livox Mid-360 LiDAR

使用方法:
在 env.py 中替换现有的 LiDAR 初始化代码
"""

import torch
from omni.isaac.orbit.sensors import RayCaster, RayCasterCfg, patterns
from omni.isaac.orbit.utils import configclass
from typing import Callable, Sequence


# ============================================
# 方法 1: 使用现有的 BpearlPatternCfg (最简单)
# ============================================
def get_livox_mid360_config_simple(lidar_range: float = 40.0):
    """
    使用 BpearlPatternCfg 模拟 Livox Mid-360

    Args:
        lidar_range: 最大探测距离 (m)

    Returns:
        RayCasterCfg 配置对象
    """
    # Livox Mid-360 垂直角度: -7° ~ 52°, 共59°
    # 这里使用 30 条垂直线
    vertical_angles = torch.linspace(-7.0, 52.0, 30).tolist()

    return RayCasterCfg(
        prim_path="/World/envs/env_.*/Hummingbird_0/base_link",
        offset=RayCasterCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),  # 传感器安装位置
        ),
        attach_yaw_only=False,  # 完全跟随姿态 (Mid-360 是固态雷达)
        pattern_cfg=patterns.BpearlPatternCfg(
            horizontal_fov=360.0,           # 水平 360°
            horizontal_res=5.0,             # 5° 分辨率 → 72 条 (RL训练用)
            vertical_ray_angles=vertical_angles,  # 自定义垂直角度
        ),
        max_distance=lidar_range,
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )


# ============================================
# 方法 2: 自定义模式函数 (更灵活)
# ============================================
def livox_mid360_pattern(
    horizontal_fov: float = 360.0,
    horizontal_res: float = 5.0,
    vertical_fov: tuple = (-7.0, 52.0),
    num_vertical_lines: int = 30,
    device: str = "cuda:0"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    生成 Livox Mid-360 射线模式

    默认配置 (RL 训练用):
    - 水平: 360° / 5° = 72 条
    - 垂直: 30 条
    - 总计: 72 × 30 = 2160 点
    """
    # 水平角度
    h_angles = torch.arange(
        -horizontal_fov / 2,
        horizontal_fov / 2,
        horizontal_res,
        device=device
    )

    # 垂直角度
    v_angles = torch.linspace(
        vertical_fov[0],
        vertical_fov[1],
        num_vertical_lines,
        device=device
    )

    # 创建网格
    yaw, pitch = torch.meshgrid(h_angles, v_angles, indexing='xy')
    yaw_rad = torch.deg2rad(yaw.flatten())
    pitch_rad = torch.deg2rad(pitch.flatten())

    # 方向向量 (X前, Y左, Z上)
    x = torch.cos(pitch_rad) * torch.cos(yaw_rad)
    y = torch.cos(pitch_rad) * torch.sin(yaw_rad)
    z = torch.sin(pitch_rad)

    ray_directions = torch.stack([x, y, z], dim=1)
    ray_starts = torch.zeros_like(ray_directions)

    return ray_starts, ray_directions


# ============================================
# 在 env.py 中使用的代码示例
# ============================================
"""
# === 在 env.py 的 __init__ 中替换 LiDAR 初始化代码 ===

# 导入
from livox_mid360_integration import get_livox_mid360_config_simple

# 方法 1: 使用简单配置 (推荐)
ray_caster_cfg = get_livox_mid360_config_simple(lidar_range=40.0)

# 或者使用自定义参数
ray_caster_cfg = RayCasterCfg(
    prim_path="/World/envs/env_.*/Hummingbird_0/base_link",
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
    attach_yaw_only=False,
    pattern_cfg=patterns.BpearlPatternCfg(
        horizontal_fov=360.0,
        horizontal_res=5.0,  # RL 训练: 5°, 高精度: 1°
        vertical_ray_angles=torch.linspace(-7.0, 52.0, 30).tolist(),
    ),
    max_distance=40.0,  # Livox Mid-360 量程
    mesh_prim_paths=["/World/ground"],
    debug_vis=False,
)

self.lidar = RayCaster(ray_caster_cfg)
self.lidar._initialize_impl()

# 更新分辨率
self.lidar_resolution = (72, 30)  # (水平, 垂直)
"""


# ============================================
# 配置模板
# ============================================
LIVOX_MID360_CONFIGS = {
    # 高精度 (可视化/测试)
    "high": {
        "horizontal_res": 1.0,      # 360 条
        "num_vertical_lines": 59,   # 59 条
        "total_rays": 21240,
    },
    # 中精度 (平衡)
    "medium": {
        "horizontal_res": 2.0,      # 180 条
        "num_vertical_lines": 30,   # 30 条
        "total_rays": 5400,
    },
    # 低精度 (RL 训练 - 推荐)
    "low": {
        "horizontal_res": 5.0,      # 72 条
        "num_vertical_lines": 15,   # 15 条
        "total_rays": 1080,
    },
    # 超低精度 (快速训练)
    "minimal": {
        "horizontal_res": 10.0,     # 36 条
        "num_vertical_lines": 8,    # 8 条
        "total_rays": 288,
    },
}


def print_config_info():
    """打印配置信息"""
    print("=" * 50)
    print("Livox Mid-360 LiDAR 配置选项")
    print("=" * 50)
    print(f"{'配置':<10} {'水平分辨率':<12} {'垂直线数':<10} {'总点数':<10}")
    print("-" * 50)
    for name, cfg in LIVOX_MID360_CONFIGS.items():
        print(
            f"{name:<10} {cfg['horizontal_res']}°{'':<9} {cfg['num_vertical_lines']:<10} {cfg['total_rays']:<10}")
    print("=" * 50)
    print("\nLivox Mid-360 官方规格:")
    print("  - 量程: 40m (10%反射率) / 70m (80%反射率)")
    print("  - FOV: 水平 360°, 垂直 -7° ~ 52° (共 59°)")
    print("  - 点云率: 200,000 点/秒")
    print("  - 精度: 距离 ≤2cm, 角度 <0.15°")


if __name__ == "__main__":
    print_config_info()
