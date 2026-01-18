# Livox Mid-360 LiDAR 单元测试

本目录包含 Livox Mid-360 LiDAR 传感器的测试脚本。

## 测试脚本

### 1. `test_livox_config.py` - 配置验证（无需 Isaac Sim）

快速验证 Livox Mid-360 配置参数，无需启动 Isaac Sim。

```bash
# 从 isaac-training 目录运行
cd /home/mint/rl_dev/NavRL/isaac-training
python training/unit_test/test_livox_config.py
```

**测试内容：**
- 配置参数正确性
- 射线模式生成
- 角度分布验证
- 可选的可视化输出

### 2. `test_livox_mid360.py` - 完整集成测试（需要 Isaac Sim）

在 Isaac Sim 中完整测试 LiDAR 集成。

**运行方式与 `train.py` 相同：**

```bash
# 激活 NavRL 环境
conda activate NavRL
cd /home/mint/rl_dev/NavRL/isaac-training

# 无头模式（快速测试）
python training/unit_test/test_livox_mid360.py

# 带可视化（查看效果）
python training/unit_test/test_livox_mid360.py headless=False

# 使用 Livox Mid-360 参数
python training/unit_test/test_livox_mid360.py headless=False sensor.lidar_vfov=[-7,52] sensor.lidar_vbeams=12
```

**测试内容：**
1. Livox Mid-360 配置参数验证
2. 射线模式生成验证
3. 创建测试场景（地面、障碍物、无人机）
4. 创建 Livox Mid-360 LiDAR
5. 运行仿真并收集 LiDAR 数据

**使用 Hydra 配置覆盖（与 train.py 相同）：**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `headless` | True | 是否无头模式 |
| `sensor.lidar_range` | 10.0 | LiDAR 最大范围 (m) |
| `sensor.lidar_vfov` | [-10, 20] | 垂直视场角 [下限, 上限] (°) |
| `sensor.lidar_vbeams` | 4 | 垂直线数 |
| `sensor.lidar_hres` | 10.0 | 水平分辨率 (°) |

**Livox Mid-360 推荐配置：**
```bash
python training/unit_test/test_livox_mid360.py \
    headless=False \
    sensor.lidar_range=40.0 \
    sensor.lidar_vfov=[-7,52] \
    sensor.lidar_vbeams=12
```

## Livox Mid-360 规格

| 参数 | 值 |
|------|-----|
| 激光波长 | 905 nm |
| 量程 | 40m @ 10%反射率, 70m @ 80%反射率 |
| 近处盲区 | 0.1 m |
| 水平 FOV | 360° |
| 垂直 FOV | -7° ~ 52° (共 59°) |
| 测距精度 | ≤ 2cm (1σ @ 10m) |
| 角度精度 | < 0.15° |
| 点云输出 | 200,000 点/秒 |

## 配置对比

| 配置 | 水平分辨率 | 垂直线数 | 总点数 | 用途 |
|------|------------|----------|--------|------|
| 高精度 | 1.0° | 59 | 21,240 | 精确仿真 |
| 中精度 | 2.0° | 30 | 5,400 | 平衡 |
| 低精度 | 5.0° | 15 | 1,080 | 快速仿真 |
| RL 训练 | 10.0° | 12 | 432 | 神经网络输入 |

## 相关文件

- `training/scripts/livox_mid360.py` - Livox Mid-360 配置类
- `training/scripts/livox_mid360_integration.py` - Isaac Sim 集成代码
- `training/cfg/drone_livox.yaml` - YAML 配置示例

## 集成到训练环境

要在训练中使用 Livox Mid-360，需要修改 `training/scripts/env.py`：

```python
# 在 __init__ 中修改 LiDAR 初始化
vertical_angles = torch.linspace(-7.0, 52.0, self.lidar_vbeams).tolist()

ray_caster_cfg = RayCasterCfg(
    prim_path="/World/envs/env_.*/Hummingbird_0/base_link",
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
    attach_yaw_only=False,  # Mid-360 固态雷达，跟随完整姿态
    pattern_cfg=patterns.BpearlPatternCfg(
        horizontal_fov=360.0,
        horizontal_res=self.lidar_hres,
        vertical_ray_angles=vertical_angles,
    ),
    max_distance=self.lidar_range,
    mesh_prim_paths=["/World/ground"],
    debug_vis=False,
)
```
