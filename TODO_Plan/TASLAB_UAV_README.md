# ============================================
# TASLAB UAV 无人机模型添加指南
# ============================================

## 📁 已创建的文件

```
NavRL/isaac-training/
├── third_party/OmniDrones/omni_drones/robots/
│   ├── assets/usd/
│   │   └── taslab_uav.yaml          # ✅ 参数配置文件
│   └── drone/
│       ├── taslab_uav.py            # ✅ 无人机类定义
│       └── __init__.py              # ✅ 已更新导入
└── training/cfg/
    └── drone_taslab.yaml            # ✅ 训练配置示例
```

## 🚀 使用方法

### 方法1: 修改默认配置
编辑 `training/cfg/drone.yaml`:
```yaml
drone:
  model_name: "taslab_uav"
```

### 方法2: 命令行指定
```bash
cd training/scripts
python train.py drone.model_name=taslab_uav
```

### 方法3: 使用单独配置文件
```bash
python train.py --config-name=train drone=drone_taslab
```

## ⚠️ 重要: 参数标定

请根据实际测量值修改 `taslab_uav.yaml` 中的以下参数:

### 必须修改的参数:
1. **mass**: 使用电子秤称量整机重量 (kg)
2. **inertia.xx/yy/zz**: CAD计算或摆动实验测量 (kg·m²)
3. **arm_lengths**: 卡尺测量电机到质心距离 (m)
4. **force_constants**: 推力台测试或悬停法标定
5. **max_rotation_velocities**: 查看电机规格或实测

### 可选调整的参数:
- **drag_coef**: 空气阻力系数 (典型值 0.1-0.4)
- **moment_constants**: 约等于 force_constants × 0.015
- **rotor_angles**: 根据电机布局调整 (X型或+型)

## 📐 快速参数估算

### 推力系数 KF (悬停法):
```
KF = (mass × 9.81) / (4 × ω_hover²)

例如: 1kg 无人机, 悬停转速 600 rad/s
KF = (1.0 × 9.81) / (4 × 600²) = 6.8e-06
```

### 力矩系数 KM:
```
KM ≈ KF × 0.015

例如: KF = 6.8e-06
KM ≈ 6.8e-06 × 0.015 = 1.02e-07
```

### 转速换算:
```
ω (rad/s) = RPM × 2π / 60

例如: 6000 RPM = 6000 × 2π / 60 ≈ 628 rad/s
```

## 🔧 添加自定义 3D 模型 (可选)

如果需要使用自己的 USD 模型:

1. 创建 USD 文件: `taslab_uav.usd`
2. 放置到: `third_party/OmniDrones/omni_drones/robots/assets/usd/`
3. 修改 `taslab_uav.py`:
   ```python
   usd_path: str = ASSET_PATH + "/usd/taslab_uav.usd"
   ```

## ✅ 验证安装

```bash
cd /home/mint/rl_dev/NavRL/isaac-training
source ../../setup_python_env.sh

# 测试导入
python -c "from omni_drones.robots.drone import TaslabUAV; print('TaslabUAV loaded successfully!')"
```

## 📝 注意事项

1. 当前使用 hummingbird.usd 作为 3D 模型 (仅影响视觉，不影响动力学)
2. 所有物理参数由 taslab_uav.yaml 控制
3. 确保参数单位正确 (SI 国际单位制)
