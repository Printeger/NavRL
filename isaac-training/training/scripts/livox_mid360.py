"""
Livox Mid-360 LiDAR 传感器配置
================================
基于官方规格: https://www.livoxtech.com/cn/mid-360/specs

Livox Mid-360 技术参数:
========================
| 参数 | 值 |
|------|-----|
| 激光波长 | 905 nm |
| 人眼安全 | Class 1 (IEC60825-1:2014) |
| 量程 | 40m @ 10%反射率, 70m @ 80%反射率 |
| 近处盲区 | 0.1 m |
| FOV | 水平 360°, 垂直 -7° ~ 52° |
| 测距精度 | ≤ 2cm (1σ @ 10m) |
| 角度精度 | < 0.15° |
| 点云输出 | 200,000 点/秒 |
| 帧率 | 10 Hz |

注意: Livox 使用非重复扫描模式，这里简化为规则扫描模式
"""

import torch
from typing import Sequence
from dataclasses import dataclass


@dataclass
class LivoxMid360Config:
    """
    Livox Mid-360 LiDAR 配置参数

    可以根据仿真需求调整分辨率（降低计算量）
    """

    # ============================================
    # 基本参数 (来自官方规格)
    # ============================================
    max_range: float = 40.0
    """最大探测距离 (m) - 10%反射率下为40m, 80%反射率下为70m"""

    min_range: float = 0.1
    """最小探测距离/盲区 (m)"""

    # ============================================
    # FOV 参数
    # ============================================
    horizontal_fov: float = 360.0
    """水平视场角 (度) - Mid-360 为 360°"""

    vertical_fov_min: float = -7.0
    """垂直视场角下限 (度)"""

    vertical_fov_max: float = 52.0
    """垂直视场角上限 (度) - 总共 59° 垂直范围"""

    # ============================================
    # 分辨率参数 (可调整以平衡精度和性能)
    # ============================================
    horizontal_res: float = 1.0
    """水平角分辨率 (度) - 官方约 0.15°，仿真可降低"""

    vertical_res: float = 2.0
    """垂直角分辨率 (度) - 用于均匀分布线束"""

    num_vertical_lines: int = 30
    """垂直线束数 - 59°范围/2°分辨率 ≈ 30 线"""

    # ============================================
    # 精度参数 (可选，用于添加噪声)
    # ============================================
    range_accuracy: float = 0.02
    """测距精度 (m) - 1σ @ 10m"""

    angular_accuracy: float = 0.15
    """角度精度 (度) - 1σ"""

    # ============================================
    # 计算属性
    # ============================================
    @property
    def num_horizontal_rays(self) -> int:
        """水平射线数"""
        return int(self.horizontal_fov / self.horizontal_res)

    @property
    def total_rays(self) -> int:
        """总射线数"""
        return self.num_horizontal_rays * self.num_vertical_lines

    @property
    def vertical_angles(self) -> list:
        """垂直角度列表"""
        return torch.linspace(
            self.vertical_fov_min,
            self.vertical_fov_max,
            self.num_vertical_lines
        ).tolist()


def create_livox_mid360_pattern(
    cfg: LivoxMid360Config,
    device: str = "cuda:0"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    创建 Livox Mid-360 的射线扫描模式

    Args:
        cfg: LivoxMid360Config 配置对象
        device: 计算设备

    Returns:
        ray_starts: 射线起点 [N, 3]
        ray_directions: 射线方向 [N, 3] (单位向量)
    """
    # 水平角度 (360°)
    h_angles = torch.arange(
        -cfg.horizontal_fov / 2,
        cfg.horizontal_fov / 2,
        cfg.horizontal_res,
        device=device
    )

    # 垂直角度 (-7° ~ 52°)
    v_angles = torch.linspace(
        cfg.vertical_fov_min,
        cfg.vertical_fov_max,
        cfg.num_vertical_lines,
        device=device
    )

    # 创建角度网格
    yaw, pitch = torch.meshgrid(h_angles, v_angles, indexing='xy')
    yaw = torch.deg2rad(yaw.flatten())
    pitch = torch.deg2rad(pitch.flatten())

    # 转换为笛卡尔方向向量
    # X: 前, Y: 左, Z: 上
    x = torch.cos(pitch) * torch.cos(yaw)
    y = torch.cos(pitch) * torch.sin(yaw)
    z = torch.sin(pitch)

    ray_directions = torch.stack([x, y, z], dim=1)
    ray_starts = torch.zeros_like(ray_directions)

    return ray_starts, ray_directions


# ============================================
# 预定义配置
# ============================================

# 高精度配置 (接近真实规格)
LIVOX_MID360_HIGH_RES = LivoxMid360Config(
    horizontal_res=0.5,      # 720 条水平线
    num_vertical_lines=59,   # 59 条垂直线
    # 总计: 720 × 59 = 42,480 点
)

# 中等精度配置 (平衡性能)
LIVOX_MID360_MEDIUM_RES = LivoxMid360Config(
    horizontal_res=1.0,      # 360 条水平线
    num_vertical_lines=30,   # 30 条垂直线
    # 总计: 360 × 30 = 10,800 点
)

# 低精度配置 (快速仿真)
LIVOX_MID360_LOW_RES = LivoxMid360Config(
    horizontal_res=2.0,      # 180 条水平线
    num_vertical_lines=15,   # 15 条垂直线
    # 总计: 180 × 15 = 2,700 点
)

# 超低精度配置 (用于 RL 训练)
LIVOX_MID360_RL_RES = LivoxMid360Config(
    horizontal_res=5.0,      # 72 条水平线
    num_vertical_lines=12,   # 12 条垂直线
    # 总计: 72 × 12 = 864 点 (适合神经网络输入)
)


# ============================================
# 使用示例
# ============================================
if __name__ == "__main__":
    # 测试配置
    cfg = LIVOX_MID360_MEDIUM_RES
    print(f"Livox Mid-360 配置:")
    print(f"  最大范围: {cfg.max_range} m")
    print(f"  水平 FOV: {cfg.horizontal_fov}°")
    print(f"  垂直 FOV: {cfg.vertical_fov_min}° ~ {cfg.vertical_fov_max}°")
    print(f"  水平分辨率: {cfg.horizontal_res}°")
    print(f"  水平射线数: {cfg.num_horizontal_rays}")
    print(f"  垂直线数: {cfg.num_vertical_lines}")
    print(f"  总射线数: {cfg.total_rays}")

    # 生成射线
    ray_starts, ray_dirs = create_livox_mid360_pattern(cfg, device="cpu")
    print(f"\n射线形状: {ray_dirs.shape}")
