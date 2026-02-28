# 无人机振荡问题诊断与解决方案

## 🔍 症状描述
- **悬停振荡**：无人机在目标位置附近不停晃动
- **转圈抖动**：一旦移动就开始旋转、抖动、失控

---

## ⚠️ 主要原因分析

### 1. **惯性参数不准确** ⭐⭐⭐⭐⭐（最常见）

**问题**：惯性张量（Inertia Tensor）错误会导致控制器误判无人机的响应特性。

**检查方法**：
```yaml
# taslab_uav.yaml
inertia:
  xx: 0.0044966865  # 绕X轴转动惯量
  yy: 0.0050967255  # 绕Y轴转动惯量  
  zz: 0.0038767007  # 绕Z轴转动惯量
  xy, xz, yz: ~0    # 惯性积（通常很小）
```

**验证公式**（粗略估算）：
```
I_xx ≈ I_yy ≈ (1/12) × m × L²
对于 1.144kg, L=0.24m 的无人机：
I ≈ (1/12) × 1.144 × 0.24² ≈ 0.0055 kg·m²
```

**你的值**：
- `xx=0.0045`, `yy=0.0051`, `zz=0.0039`
- 看起来合理，但如果是复制其他模型的，需要用**CAD软件**或**实验测量**

**常见错误**：
- ❌ 直接复制其他无人机的惯性参数
- ❌ 忽略电池、LiDAR等载荷的影响
- ❌ 惯性积 xy, xz, yz 设置错误（应接近0）

**解决方案**：
```yaml
# 方法1: 使用经验公式（快速测试）
inertia:
  xx: 0.0055  # (1/12) × 1.144 × 0.24²
  yy: 0.0055
  zz: 0.0080  # Z轴通常 1.5x ~ 2x
  xy: 0.0
  xz: 0.0
  yz: 0.0

# 方法2: CAD软件精确计算（SolidWorks, Fusion360）
# 方法3: 物理实验测量（复摆法、双线摆法）
```

---

### 2. **推力/力矩系数不匹配** ⭐⭐⭐⭐

**问题**：`force_constants (KF)` 和 `moment_constants (KM)` 配置错误会导致姿态控制失效。

**检查方法**：
```yaml
rotor_configuration:
  force_constants:  [1.55e-06, ...]  # KF: 推力 = KF × ω²
  moment_constants: [2.39e-08, ...]  # KM: 力矩 = KM × ω²
```

**验证**：
```
KM / KF 比值应在 0.01 ~ 0.02 之间
你的比值: 2.39e-08 / 1.55e-06 = 0.0154 ✅ 合理
```

**常见错误**：
- ❌ KF 太小 → 无法产生足够推力 → 下坠、大幅振荡
- ❌ KM 太大 → 过度的偏航力矩 → 转圈
- ❌ 四个电机的 KF/KM 不一致 → 不对称响应

**验证方法（悬停法）**：
```python
# 理论悬停转速
ω_hover = sqrt((m × g) / (4 × KF))

# 对于你的无人机
m = 1.144 kg
g = 9.81 m/s²
KF = 1.55e-06 N/(rad/s)²

ω_hover = sqrt(1.144 × 9.81 / (4 × 1.55e-06))
       ≈ 1350 rad/s ≈ 12900 RPM

# 检查是否合理（通常悬停在 60%-70% 最大转速）
```

**解决方案**：
```yaml
# 选项1: 根据电机规格表调整
force_constants: [1.8e-06, 1.8e-06, 1.8e-06, 1.8e-06]
moment_constants: [2.7e-08, 2.7e-08, 2.7e-08, 2.7e-08]

# 选项2: 实验测量（推力台）
# 选项3: 参考同规格电机的数据手册
```

---

### 3. **电机布局配置错误** ⭐⭐⭐⭐

**问题**：`rotor_angles` 和 `directions` 配置错误会导致无法平衡力矩 → 转圈。

**检查方法**：
```yaml
rotor_configuration:
  rotor_angles:  # 电机角度（rad）
    - 5.49778714  # 电机0: 315° (-45°)
    - 0.78539816  # 电机1: 45°
    - 2.35619449  # 电机2: 135°
    - 3.92699082  # 电机3: 225°
  
  directions:  # 旋转方向（+1=CCW, -1=CW）
    - 1.0   # 电机0: CCW
    - -1.0  # 电机1: CW
    - 1.0   # 电机2: CCW
    - -1.0  # 电机3: CW
```

**X型布局规则**：
```
     前 (0°)
      ↑
  1●  |  ●0   电机布局
  CW  |  CCW
  ----+----→ 右
  CCW |  CW
  2●  |  ●3
```

**验证公式**：
```python
import numpy as np
angles = [5.49778714, 0.78539816, 2.35619449, 3.92699082]
degrees = [np.degrees(a) % 360 for a in angles]
# 应该输出: [315°, 45°, 135°, 225°] ✅

# 检查力矩平衡：相邻电机方向相反
directions = [1, -1, 1, -1]
sum(directions) == 0  # 必须为0
```

**常见错误**：
- ❌ 角度不是 45° 的倍数（X型）或 90° 的倍数（+型）
- ❌ 相邻电机旋转方向相同 → 无法平衡偏航力矩
- ❌ 角度与实际机架不符

**解决方案**：
```yaml
# X型布局（标准）
rotor_angles:
  - 0.78539816   # 45°  (右前)
  - 2.35619449   # 135° (左前)
  - 3.92699082   # 225° (左后)
  - 5.49778714   # 315° (右后)

directions:
  - 1.0   # CCW
  - -1.0  # CW
  - 1.0   # CCW
  - -1.0  # CW
```

---

### 4. **控制器增益不合适** ⭐⭐⭐

**问题**：Lee控制器的PID增益需要针对不同无人机调整。

**检查方法**：
```yaml
# lee_controller_taslab_uav.yaml
position_gain: [4, 4, 4]              # 位置比例增益
velocity_gain: [2.2, 2.2, 2.2]         # 速度比例增益
attitude_gain: [0.8, 0.8, 0.035]       # 姿态增益
angular_rate_gain: [0.25, 0.25, 0.03]  # 角速度阻尼增益 ⭐关键
```

**症状对应**：
- **高频振荡** → `angular_rate_gain` 太小，增加阻尼
- **响应缓慢** → `position_gain` 太小
- **超调严重** → `velocity_gain` 太小
- **低频晃动** → `attitude_gain` 不匹配惯性

**调参策略**：
```yaml
# 步骤1: 先降低增益到安全值（悬停测试）
position_gain: [2, 2, 2]
velocity_gain: [1.5, 1.5, 1.5]
attitude_gain: [0.5, 0.5, 0.02]
angular_rate_gain: [0.3, 0.3, 0.05]  # ⬆ 增加阻尼

# 步骤2: 逐步增加到目标响应速度
# 步骤3: 微调 z 轴单独参数
```

**Hummingbird vs TaslabUAV 对比**：
```yaml
# Hummingbird (0.716kg)
angular_rate_gain: [0.1, 0.1, 0.025]

# TaslabUAV (1.144kg, 重60%)
angular_rate_gain: [0.25, 0.25, 0.03]  # 需要更大阻尼
```

---

### 5. **物理仿真时间步长** ⭐⭐

**问题**：`physics_dt` 太大会导致数值不稳定。

**检查方法**：
```python
# test_hover.py
sim_context = SimulationContext(
    physics_dt=0.02,      # 物理仿真步长 (秒)
    rendering_dt=0.02,    # 渲染步长 (秒)
    ...
)
```

**推荐值**：
```
physics_dt = 0.01 ~ 0.02 (50-100 Hz)
太大 (>0.02) → 数值不稳定、振荡
太小 (<0.005) → 仿真太慢、资源浪费
```

**解决方案**：
```python
# 稳定性优先
physics_dt=0.01,  # 100 Hz

# 速度优先（训练）
physics_dt=0.02,  # 50 Hz
```

---

### 6. **质量/阻力系数** ⭐⭐

**检查方法**：
```yaml
mass: 1.144  # 总质量 (kg) - 包含电池、传感器
drag_coef: 0.2  # 空气阻力系数
```

**验证**：
- 用电子秤测量**实际总质量**（含电池、LiDAR）
- `drag_coef` 对悬停影响小，对高速飞行影响大

**常见错误**：
- ❌ 只计算机身质量，忽略电池、载荷
- ❌ 使用空载质量进行训练，实际飞行装载后失效

---

### 7. **最大转速限制** ⭐

**检查方法**：
```yaml
max_rotation_velocities: [2261, 2261, 2261, 2261]  # rad/s
```

**验证**：
```
最大转速 = 电机 KV 值 × 电池电压 × 2π / 60

例如: 2200 KV × 14.8V (4S) × 2π / 60 ≈ 3410 rad/s（理论上限）
实际使用: 60-80% ≈ 2000-2700 rad/s
```

**常见错误**：
- ❌ 最大转速设置太低 → 无法产生足够推力 → 下坠
- ❌ 悬停转速接近最大值 → 无控制余量 → 振荡

---

## 🛠️ 系统调试流程

### 第一步：验证基础物理参数

```bash
cd /home/mint/rl_dev/NavRL/isaac-training
python training/unit_test/test_hover.py drone.model_name=TaslabUAV headless=False
```

**测试项目**：
1. ✅ 无人机能否悬停（不下坠）
2. ✅ 按 Q/E 上下移动是否平稳
3. ✅ 按 Z/X 旋转是否稳定（不转圈）

### 第二步：调整控制器增益

```yaml
# 编辑: third_party/OmniDrones/omni_drones/controllers/cfg/lee_controller_taslab_uav.yaml

# 如果高频振荡：增加角速度阻尼
angular_rate_gain: [0.35, 0.35, 0.05]  # ⬆

# 如果低频晃动：降低姿态增益
attitude_gain: [0.6, 0.6, 0.03]  # ⬇

# 如果响应缓慢：增加位置/速度增益
position_gain: [5, 5, 5]
velocity_gain: [2.5, 2.5, 2.5]
```

### 第三步：记录日志分析

```python
# 在 test_hover.py 中添加日志
print(f"Target: {target_pos}, Current: {drone_state[:3]}")
print(f"Angular vel: {drone_state[10:13]}")
print(f"Action: {action}")
```

**观察指标**：
- 位置误差是否收敛
- 角速度是否发散
- 控制指令是否饱和

---

## 📋 快速检查清单

| 参数 | 位置 | 检查方法 | 常见值 |
|------|------|----------|--------|
| **惯性xx/yy** | `taslab_uav.yaml` | `≈ (1/12)×m×L²` | 0.004-0.006 |
| **惯性zz** | `taslab_uav.yaml` | `≈ 1.5-2× xx` | 0.006-0.010 |
| **质量** | `taslab_uav.yaml` | 实际称重 | 1.1-1.3 kg |
| **KF** | `taslab_uav.yaml` | 悬停测试 | 1.5e-06 ~ 2e-06 |
| **KM/KF** | `taslab_uav.yaml` | 比值 0.01-0.02 | 0.015 |
| **电机角度** | `taslab_uav.yaml` | X型: 45°倍数 | [45,135,225,315]° |
| **旋转方向** | `taslab_uav.yaml` | 相邻相反 | [+1,-1,+1,-1] |
| **angular_rate_gain** | `lee_controller_taslab_uav.yaml` | 阻尼 | 0.2-0.4 |
| **physics_dt** | `test_hover.py` | 50-100Hz | 0.01-0.02 s |

---

## 💡 推荐调试顺序

1. **先检查惯性参数** → 使用经验公式快速验证
2. **再检查电机布局** → 角度、方向必须正确
3. **降低控制增益** → 保证基础悬停稳定
4. **逐步调优** → 小步增加增益直到满意

---

## 🔗 参考资料

- [Lee Position Controller 论文](https://arxiv.org/abs/1003.2005)
- Hummingbird 参数: `lee_controller_hummingbird.yaml`
- 物理参数测量: CAD软件、实验法
