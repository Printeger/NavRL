#!/usr/bin/env python3
"""
无人机参数验证脚本
快速检查 taslab_uav.yaml 配置是否合理

运行:
    cd /home/mint/rl_dev/NavRL/isaac-training/training/unit_test
    python3 check_drone_params.py
"""

import yaml
import numpy as np
import os

def check_params():
    # 读取配置文件
    yaml_path = "../../third_party/OmniDrones/omni_drones/robots/assets/usd/taslab_uav.yaml"
    
    if not os.path.exists(yaml_path):
        print(f"❌ 文件不存在: {yaml_path}")
        return
    
    with open(yaml_path, 'r') as f:
        params = yaml.safe_load(f)
    
    print("=" * 70)
    print("无人机参数验证")
    print("=" * 70)
    
    # 1. 基础参数
    mass = params['mass']
    l = params['l']
    print(f"\n📊 基础参数:")
    print(f"  质量: {mass:.3f} kg")
    print(f"  轴距: {l:.3f} m")
    
    # 2. 惯性验证
    inertia = params['inertia']
    xx = inertia['xx']
    yy = inertia['yy']
    zz = inertia['zz']
    
    # 经验公式估算
    I_estimated = (1/12) * mass * l**2
    
    print(f"\n🔄 惯性参数:")
    print(f"  Ixx: {xx:.6f} kg·m²")
    print(f"  Iyy: {yy:.6f} kg·m²")
    print(f"  Izz: {zz:.6f} kg·m²")
    print(f"  理论估算值: {I_estimated:.6f} kg·m² (经验公式)")
    
    # 验证
    if abs(xx - I_estimated) / I_estimated < 0.3:
        print(f"  ✅ Ixx 在合理范围内")
    else:
        print(f"  ⚠️  Ixx 偏差较大，建议使用 {I_estimated:.6f}")
    
    if abs(yy - I_estimated) / I_estimated < 0.3:
        print(f"  ✅ Iyy 在合理范围内")
    else:
        print(f"  ⚠️  Iyy 偏差较大，建议使用 {I_estimated:.6f}")
    
    if 1.2 < zz/xx < 2.5:
        print(f"  ✅ Izz 比例合理 (Izz/Ixx = {zz/xx:.2f})")
    else:
        print(f"  ⚠️  Izz 比例异常，通常应为 1.5-2× Ixx")
    
    # 3. 电机配置
    rotor_config = params['rotor_configuration']
    num_rotors = rotor_config['num_rotors']
    angles = rotor_config['rotor_angles']
    directions = rotor_config['directions']
    force_constants = rotor_config['force_constants']
    moment_constants = rotor_config['moment_constants']
    max_vel = rotor_config['max_rotation_velocities']
    
    print(f"\n⚙️  电机配置:")
    print(f"  电机数量: {num_rotors}")
    
    # 角度验证
    angles_deg = [np.degrees(a) % 360 for a in angles]
    print(f"  电机角度: {[f'{d:.0f}°' for d in angles_deg]}")
    
    # X型标准角度
    x_config = [45, 135, 225, 315]
    is_x_config = all(abs(a - x) < 5 for a, x in zip(sorted(angles_deg), sorted(x_config)))
    if is_x_config:
        print(f"  ✅ X型布局配置正确")
    else:
        print(f"  ⚠️  角度配置异常，标准X型应为 [45°, 135°, 225°, 315°]")
    
    # 方向验证
    print(f"  旋转方向: {directions}")
    if sum(directions) == 0:
        print(f"  ✅ 力矩平衡 (方向之和为0)")
    else:
        print(f"  ❌ 力矩不平衡！方向之和应为0")
    
    # KF/KM 验证
    KF = force_constants[0]
    KM = moment_constants[0]
    ratio = KM / KF
    print(f"\n🚀 推力参数:")
    print(f"  KF: {KF:.2e} N/(rad/s)²")
    print(f"  KM: {KM:.2e} N·m/(rad/s)²")
    print(f"  KM/KF 比值: {ratio:.4f}")
    
    if 0.01 < ratio < 0.02:
        print(f"  ✅ KM/KF 比值合理 (0.01-0.02)")
    else:
        print(f"  ⚠️  KM/KF 比值异常，建议调整为 0.015")
    
    # 悬停转速
    g = 9.81
    omega_hover = np.sqrt((mass * g) / (4 * KF))
    omega_hover_rpm = omega_hover * 60 / (2 * np.pi)
    
    print(f"\n📈 悬停分析:")
    print(f"  理论悬停转速: {omega_hover:.0f} rad/s ({omega_hover_rpm:.0f} RPM)")
    print(f"  最大转速: {max_vel[0]:.0f} rad/s")
    print(f"  悬停占比: {omega_hover/max_vel[0]*100:.1f}%")
    
    if 50 < omega_hover/max_vel[0]*100 < 70:
        print(f"  ✅ 悬停占比合理 (50-70%)")
    elif omega_hover/max_vel[0]*100 < 50:
        print(f"  ⚠️  控制余量过大，可以降低 max_rotation_velocities")
    else:
        print(f"  ❌ 控制余量不足！悬停转速过高，需要:")
        print(f"     - 增加 force_constants (KF)")
        print(f"     - 或增加 max_rotation_velocities")
    
    # 总结
    print("\n" + "=" * 70)
    print("💡 问题排查建议:")
    print("=" * 70)
    
    issues = []
    
    if abs(xx - I_estimated) / I_estimated > 0.3:
        issues.append("⚠️  惯性参数 Ixx/Iyy 偏差较大")
    
    if not (1.2 < zz/xx < 2.5):
        issues.append("⚠️  Izz 比例异常")
    
    if not is_x_config:
        issues.append("⚠️  电机角度配置可能错误")
    
    if sum(directions) != 0:
        issues.append("❌ 电机旋转方向不平衡（会转圈）")
    
    if not (0.01 < ratio < 0.02):
        issues.append("⚠️  KM/KF 比值异常")
    
    if omega_hover/max_vel[0] > 0.7:
        issues.append("❌ 悬停转速过高，控制余量不足（会振荡）")
    
    if len(issues) == 0:
        print("✅ 所有参数检查通过！")
        print("\n如果仍然振荡，请调整控制器增益:")
        print("   编辑: third_party/OmniDrones/omni_drones/controllers/cfg/lee_controller_taslab_uav.yaml")
        print("   增加 angular_rate_gain 来增加阻尼")
    else:
        print("发现以下问题:")
        for issue in issues:
            print(f"  {issue}")
        print("\n建议:")
        print("  1. 备份当前配置")
        print("  2. 使用 taslab_uav_stable.yaml 中的推荐参数")
        print("  3. 重新测试悬停")
    
    print("=" * 70)

if __name__ == "__main__":
    check_params()
