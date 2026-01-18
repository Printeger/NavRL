"""
测试从 Hydra 配置读取 LiDAR 参数
====================================
验证 lidar_mount_position 和其他参数能正确从 drone.yaml 读取
"""

import os
import sys
import hydra
from omegaconf import DictConfig

# 添加路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENVS_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "envs")
sys.path.insert(0, ENVS_PATH)

# 配置文件路径
CFG_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "cfg")


@hydra.main(config_path=CFG_PATH, config_name="drone", version_base=None)
def test_config(cfg: DictConfig):
    """测试配置读取"""

    print("=" * 60)
    print("LiDAR 配置参数读取测试")
    print("=" * 60)

    # 1. 直接读取配置
    print("\n[测试 1] 直接读取 cfg.sensor 参数:")
    print(f"  ├── lidar_range: {cfg.sensor.lidar_range} m")
    print(f"  ├── lidar_vfov: {cfg.sensor.lidar_vfov}")
    print(f"  ├── lidar_vbeams: {cfg.sensor.lidar_vbeams}")
    print(f"  ├── lidar_hres: {cfg.sensor.lidar_hres}°")
    print(f"  ├── lidar_mount_pitch: {cfg.sensor.lidar_mount_pitch}°")
    print(f"  ├── lidar_mount_roll: {cfg.sensor.lidar_mount_roll}°")
    print(f"  ├── lidar_mount_yaw: {cfg.sensor.lidar_mount_yaw}°")
    print(f"  └── lidar_mount_position: {cfg.sensor.lidar_mount_position}")

    # 2. 使用 create_livox_from_hydra_cfg
    print("\n[测试 2] 使用 create_livox_from_hydra_cfg 创建:")

    from livox_mid360 import create_livox_from_hydra_cfg

    livox_pattern = create_livox_from_hydra_cfg(cfg, device="cpu")
    livox_cfg = livox_pattern.cfg

    print(f"  ├── max_range: {livox_cfg.max_range} m")
    print(
        f"  ├── vertical_fov: [{livox_cfg.vertical_fov_min}, {livox_cfg.vertical_fov_max}]°")
    print(f"  ├── num_vertical_lines: {livox_cfg.num_vertical_lines}")
    print(f"  ├── horizontal_res: {livox_cfg.horizontal_res}°")
    print(f"  ├── num_horizontal_rays: {livox_cfg.num_horizontal_rays}")
    print(f"  ├── total_rays_nominal: {livox_cfg.total_rays_nominal}")
    print(f"  ├── mount_pitch: {livox_cfg.mount_pitch}°")
    print(f"  ├── mount_roll: {livox_cfg.mount_roll}°")
    print(f"  ├── mount_yaw: {livox_cfg.mount_yaw}°")
    print(f"  └── mount_position: {livox_cfg.mount_position}")

    # 3. 验证参数一致性
    print("\n[测试 3] 验证参数一致性:")

    assert livox_cfg.max_range == cfg.sensor.lidar_range
    print("  ✓ max_range 一致")

    assert livox_cfg.vertical_fov_min == cfg.sensor.lidar_vfov[0]
    assert livox_cfg.vertical_fov_max == cfg.sensor.lidar_vfov[1]
    print("  ✓ vertical_fov 一致")

    assert livox_cfg.num_vertical_lines == cfg.sensor.lidar_vbeams
    print("  ✓ num_vertical_lines 一致")

    assert livox_cfg.horizontal_res == cfg.sensor.lidar_hres
    print("  ✓ horizontal_res 一致")

    assert livox_cfg.mount_pitch == cfg.sensor.lidar_mount_pitch
    assert livox_cfg.mount_roll == cfg.sensor.lidar_mount_roll
    assert livox_cfg.mount_yaw == cfg.sensor.lidar_mount_yaw
    print("  ✓ mount angles 一致")

    expected_pos = tuple(cfg.sensor.lidar_mount_position)
    assert livox_cfg.mount_position == expected_pos
    print("  ✓ mount_position 一致")

    # 4. 生成射线测试
    print("\n[测试 4] 生成射线测试:")
    ray_origins, ray_dirs = livox_pattern.generate_rays(dt=0.0)
    print(f"  ├── ray_origins shape: {ray_origins.shape}")
    print(f"  ├── ray_directions shape: {ray_dirs.shape}")
    print(f"  ├── ray_origins[0]: {ray_origins[0].tolist()}")
    print(f"  └── 预期安装位置: {expected_pos}")

    # 验证射线起点等于安装位置
    import torch
    expected_origin = torch.tensor(expected_pos, dtype=torch.float32)
    assert torch.allclose(ray_origins[0], expected_origin, atol=1e-6)
    print("  ✓ 射线起点与安装位置一致")

    print("\n" + "=" * 60)
    print("✓ 所有测试通过！配置读取正常工作")
    print("=" * 60)


if __name__ == "__main__":
    test_config()
