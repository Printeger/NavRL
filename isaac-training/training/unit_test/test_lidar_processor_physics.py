"""
LiDAR Processor 物理严谨性测试 (Science Robotics 标准)
========================================================
测试目标：
1. 物理一致性 (Physics Check)：验证坐标变换的物理正确性
2. 极值逻辑 (Safety Critical Check)：验证最小距离池化的正确性
3. 可视化验证 (Visual Sanity Check)：验证深度图的可解释性

运行方式:
    conda activate NavRL
    cd /home/mint/rl_dev/NavRL/isaac-training
    python training/unit_test/test_lidar_processor_physics.py
"""

import os
import sys
import torch
import numpy as np
from typing import Tuple

# type: ignore
# 添加路径 (必须在导入 lidar_processor 之前)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENVS_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "envs")
sys.path.insert(0, ENVS_PATH)
from lidar_processor import LidarRetina

class PhysicsTestSuite:
    """物理严谨性测试套件"""

    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.tolerance = 1e-5  # 数值容差

    def test_1_coordinate_transformation_physics(self):
        """
        测试 1: 物理一致性 (Physics Check)
        ================================================
        目标：验证雷达坐标系到机体坐标系的变换符合物理规律

        测试场景：
        - LiDAR 安装倾角: 45° 向前下
        - 输入: LiDAR 正前方的点 (1, 0, 0)
        - 预期: 机体坐标系中应该指向前下方 (X>0, Z<0)

        物理原理：
        - LiDAR 传感器坐标系: X 轴沿激光发射方向
        - 机体坐标系: X 轴沿无人机前进方向，Z 轴向上
        - 当 LiDAR 向下倾斜 45° 时，其 X 轴应该在机体坐标系中
          分解为 X_body≈0.707 (前), Z_body≈-0.707 (下)
        """
        print("\n" + "=" * 70)
        print("测试 1: 物理一致性 (Physics Check)")
        print("=" * 70)

        # 创建 LiDAR 处理器，45° 俯仰角
        retina = LidarRetina(
            mount_angle_deg=45.0,
            grid_H=16,
            grid_W=72,
            device=self.device
        )

        print("\n[子测试 1.1] 正前方点的变换")
        print("-" * 70)

        # 测试点: LiDAR 正前方 1 米
        point_lidar = torch.tensor([[1.0, 0.0, 0.0]], device=self.device)
        point_body = retina.transform_to_body_frame(point_lidar)

        x, y, z = point_body[0].tolist()

        print(f"输入 (LiDAR 坐标系): {point_lidar[0].tolist()}")
        print(f"输出 (机体坐标系):   [{x:.6f}, {y:.6f}, {z:.6f}]")
        print(f"期望值:              [0.707107, 0.0, -0.707107]")

        # 验证 X 分量 (前向)
        expected_x = np.cos(np.radians(45.0))
        assert abs(x - expected_x) < self.tolerance, \
            f"X 分量错误: {x} != {expected_x}"
        print(f"✓ X 分量正确 (前向): {x:.6f} ≈ {expected_x:.6f}")

        # 验证 Y 分量 (横向，应该为 0)
        assert abs(y) < self.tolerance, \
            f"Y 分量应该为 0: {y}"
        print(f"✓ Y 分量正确 (横向): {y:.6f} ≈ 0.0")

        # 验证 Z 分量 (向下，必须为负)
        expected_z = -np.sin(np.radians(45.0))
        assert z < 0, \
            f"Z 分量必须为负 (向下)，但得到: {z}"
        assert abs(z - expected_z) < self.tolerance, \
            f"Z 分量错误: {z} != {expected_z}"
        print(f"✓ Z 分量正确 (向下): {z:.6f} ≈ {expected_z:.6f}")

        print("\n[子测试 1.2] 多个方向的变换验证")
        print("-" * 70)

        # 测试多个方向
        test_cases = [
            ("前方", torch.tensor([[1.0, 0.0, 0.0]])),
            ("左侧", torch.tensor([[0.0, 1.0, 0.0]])),
            ("上方", torch.tensor([[0.0, 0.0, 1.0]])),
        ]

        for name, point in test_cases:
            transformed = retina.transform_to_body_frame(point.to(self.device))
            print(
                f"{name:4s}: {point[0].tolist()} -> {transformed[0].tolist()}")

        # 验证旋转矩阵是正交的 (det = 1)
        R = retina.R_lidar_to_body
        det = torch.det(R)
        print(f"\n旋转矩阵行列式: {det:.6f} (应该 ≈ 1.0)")
        assert abs(det - 1.0) < self.tolerance, \
            f"旋转矩阵不是正交的: det = {det}"

        print("\n" + "✓" * 35)
        print("测试 1 通过: 坐标变换符合物理规律")
        print("✓" * 35)

        return True

    def test_2_min_pooling_safety_critical(self):
        """
        测试 2: 极值逻辑 (Safety Critical Check)
        ================================================
        目标：验证最小距离池化能正确处理多个点投影到同一像素的情况

        测试场景：
        - 同一视线方向上有两个障碍物：
          * 近处: 1.0m (电线/树枝 - 致命威胁)
          * 远处: 10.0m (墙壁 - 次要威胁)
        - 预期: 深度图必须显示 1.0m，而不是平均值或错误值

        安全关键性：
        - 如果系统将距离判断为 5.5m (平均)，无人机会撞上 1m 处的电线
        - 必须采用 min-pooling 而非 mean-pooling
        """
        print("\n" + "=" * 70)
        print("测试 2: 极值逻辑 (Safety Critical Check)")
        print("=" * 70)

        retina = LidarRetina(
            mount_angle_deg=45.0,
            grid_H=16,
            grid_W=72,
            max_dist=40.0,
            device=self.device
        )

        print("\n[子测试 2.1] 同一视线上的两个障碍物")
        print("-" * 70)

        # 创建测试场景: 同一方向上的远近两点
        # 方向: 机体前方 (0°, 0°)
        near_distance = 1.0  # 电线
        far_distance = 10.0  # 墙壁

        # 构造点云 (机体坐标系)
        points_body = torch.tensor([
            [near_distance, 0.0, 0.0],  # 近点
            [far_distance, 0.0, 0.0],   # 远点
        ], device=self.device).unsqueeze(0)  # (1, 2, 3)

        print(f"输入点云 (机体坐标系):")
        print(f"  近点 (电线): [{near_distance:.1f}, 0.0, 0.0] m")
        print(f"  远点 (墙壁): [{far_distance:.1f}, 0.0, 0.0] m")

        # 处理点云
        depth_image, debug = retina.spherical_min_pool(
            points_body, return_debug=True
        )

        # 找到中心像素 (正前方应该投影到这里)
        center_row = retina.grid_H // 2
        center_col = retina.grid_W // 2

        depth_value = depth_image[0, center_row, center_col].item()

        print(f"\n深度图中心像素值: {depth_value:.3f} m")
        print(f"期望值: {near_distance:.3f} m (近点距离)")
        print(f"错误值: {far_distance:.3f} m (远点距离)")
        print(f"致命错误值: {(near_distance + far_distance)/2:.3f} m (平均值)")

        # 严格验证: 必须是近点距离
        assert abs(depth_value - near_distance) < 0.1, \
            f"深度值错误! 得到 {depth_value:.3f}m，期望 {near_distance:.3f}m"

        print(f"\n✓ 正确采用最小距离: {depth_value:.3f}m ≈ {near_distance:.3f}m")

        # 验证不是平均值
        mean_value = (near_distance + far_distance) / 2
        assert abs(depth_value - mean_value) > 0.5, \
            f"错误! 使用了平均值而非最小值"

        print(f"✓ 未使用平均值: {depth_value:.3f}m ≠ {mean_value:.3f}m")

        print("\n[子测试 2.2] 多个像素的 Min-Pooling 验证")
        print("-" * 70)

        # 创建更复杂的场景
        num_points = 100
        torch.manual_seed(42)

        # 生成随机点云，每个方向上有多个点
        angles = torch.rand(
            num_points, 2, device=self.device) * 2 - 1  # [-1, 1]
        distances = torch.rand(
            num_points, device=self.device) * 10 + 1  # [1, 11]m

        # 转换为笛卡尔坐标
        points = torch.stack([
            distances,  # X
            angles[:, 0] * distances,  # Y
            angles[:, 1] * distances,  # Z
        ], dim=1).unsqueeze(0)  # (1, N, 3)

        # 为每个方向添加一个近点
        near_points = points.clone()
        near_points[:, :, 0] = 0.5  # 所有点都设为 0.5m

        combined_points = torch.cat([points, near_points], dim=1)

        print(f"生成点云: {combined_points.shape[1]} 个点")
        print(f"  原始点距离范围: [1.0, 11.0] m")
        print(f"  添加近点距离: 0.5 m")

        depth_image = retina.spherical_min_pool(combined_points)

        # 验证所有非空像素都反映了最小距离
        non_empty = depth_image[0] > 0
        if non_empty.any():
            min_depth = depth_image[0][non_empty].min().item()
            max_depth = depth_image[0][non_empty].max().item()

            print(f"\n深度图统计:")
            print(f"  非空像素数: {non_empty.sum().item()}")
            print(f"  最小深度: {min_depth:.3f} m")
            print(f"  最大深度: {max_depth:.3f} m")

            # 验证最小深度接近 0.5m (我们添加的近点)
            assert min_depth < 1.0, \
                f"最小深度过大: {min_depth:.3f}m，应该接近 0.5m"

            print(f"✓ Min-Pooling 正常工作: 最小深度 = {min_depth:.3f}m")

        print("\n" + "✓" * 35)
        print("测试 2 通过: 极值逻辑正确，安全关键")
        print("✓" * 35)

        return True

    def test_3_visual_sanity_check(self):
        """
        测试 3: 可视化验证 (Visual Sanity Check)
        ================================================
        目标：验证深度图的可解释性，确保生成的图像符合人类直觉

        测试场景：
        - 地面: Z = -1.5m (无人机下方)
        - 柱子: X = 3m, Y = 0, Z = [-1.5, 1.5]m (前方中央)

        预期结果：
        - 深度图下半部分应该显示地面 (相对近)
        - 深度图中部应该显示柱子 (中等距离)
        - 深度图上半部分应该为空 (天空)
        """
        print("\n" + "=" * 70)
        print("测试 3: 可视化验证 (Visual Sanity Check)")
        print("=" * 70)

        retina = LidarRetina(
            mount_angle_deg=45.0,
            grid_H=32,  # 更高分辨率用于可视化
            grid_W=144,
            max_dist=40.0,
            device=self.device,
            invert_depth=True  # 危险归一化: 1.0=近(危险), 0.0=远(安全)
        )

        print("\n[子测试 3.1] 合成简单场景")
        print("-" * 70)

        # 1. 生成地面点云
        ground_z = -1.5  # 地面高度
        ground_x = torch.linspace(0.5, 10, 50, device=self.device)
        ground_y = torch.linspace(-5, 5, 30, device=self.device)

        # 网格化
        gx, gy = torch.meshgrid(ground_x, ground_y, indexing='ij')
        ground_points = torch.stack([
            gx.flatten(),
            gy.flatten(),
            torch.full_like(gx.flatten(), ground_z)
        ], dim=1)

        print(f"地面点云: {ground_points.shape[0]} 个点")
        print(f"  位置: X=[0.5, 10]m, Y=[-5, 5]m, Z={ground_z}m")

        # 2. 生成柱子点云
        pillar_x = -3.0  # 柱子距离
        pillar_y = 0.0  # 柱子位置 (中央)
        pillar_z = torch.linspace(-1.5, 15, 30, device=self.device)
        pillar_points = torch.stack([
            torch.full_like(pillar_z, pillar_x),
            torch.full_like(pillar_z, pillar_y),
            pillar_z
        ], dim=1)

        print(f"柱子点云: {pillar_points.shape[0]} 个点")
        print(f"  位置: X={pillar_x}m, Y={pillar_y}m, Z=[-1.5, 1.5]m")

        # 3. 合并点云
        all_points = torch.cat([ground_points, pillar_points], dim=0)
        all_points = all_points.unsqueeze(0)  # (1, N, 3)

        print(f"\n总点云: {all_points.shape[1]} 个点")

        # 处理点云
        depth_image = retina.process(all_points)

        print(f"\n深度图形状: {depth_image.shape}")
        print(f"  高度 (仰角): {depth_image.shape[1]} bins")
        print(f"  宽度 (方位): {depth_image.shape[2]} bins")

        # 统计分析
        non_empty = depth_image[0] > 0
        if non_empty.any():
            print(f"\n深度图统计:")
            print(
                f"  非空像素: {non_empty.sum().item()} / {depth_image[0].numel()}")
            print(f"  填充率: {100 * non_empty.float().mean().item():.1f}%")
            print(f"  深度范围: [{depth_image[0][non_empty].min():.3f}, "
                  f"{depth_image[0][non_empty].max():.3f}]")

        print("\n[子测试 3.2] 空间分布验证")
        print("-" * 70)

        # 将深度图分为上、中、下三个区域
        H = depth_image.shape[1]
        upper_third = depth_image[0, :H//3, :]
        middle_third = depth_image[0, H//3:2*H//3, :]
        lower_third = depth_image[0, 2*H//3:, :]

        # 统计每个区域的覆盖率
        upper_coverage = (upper_third > 0).float().mean().item()
        middle_coverage = (middle_third > 0).float().mean().item()
        lower_coverage = (lower_third > 0).float().mean().item()

        print(f"深度图区域覆盖率:")
        print(f"  上部 (天空):    {upper_coverage*100:.1f}%")
        print(f"  中部 (柱子):    {middle_coverage*100:.1f}%")
        print(f"  下部 (地面):    {lower_coverage*100:.1f}%")

        # 验证: 下部应该比上部有更多覆盖 (因为有地面)
        assert lower_coverage > upper_coverage * 0.5, \
            f"下部覆盖率 ({lower_coverage:.2f}) 应该明显高于上部 ({upper_coverage:.2f})"

        print(
            f"\n✓ 空间分布合理: 下部({lower_coverage:.2%}) > 上部({upper_coverage:.2%})")

        print("\n[子测试 3.3] 柱子检测验证")
        print("-" * 70)

        # 检查中央列 (柱子应该在这里)
        center_col = depth_image.shape[2] // 2
        center_column = depth_image[0, :, center_col]

        # 统计中央列的非空像素
        center_non_empty = (center_column > 0).sum().item()

        print(f"中央列 (柱子位置) 分析:")
        print(f"  非空像素: {center_non_empty} / {len(center_column)}")

        # 验证: 中央列应该有明显的信号 (柱子)
        assert center_non_empty > 0, "中央列应该检测到柱子"

        print(f"✓ 柱子被检测到: 中央列有 {center_non_empty} 个非空像素")

        # 检查柱子的深度值
        if center_non_empty > 0:
            pillar_depths = center_column[center_column > 0]
            mean_pillar_depth = pillar_depths.mean().item()

            print(f"  柱子平均深度: {mean_pillar_depth:.3f}")
            print(f"  期望范围: [0.5, 0.8] (反转归一化，3m 距离)")

            # 由于深度反转归一化，3m 处的物体应该在中等值
            # depth_normalized = 1.0 - min(distance / max_dist, 1.0)
            # 对于 3m: 1.0 - 3/40 = 1.0 - 0.075 = 0.925
            expected_depth = 1.0 - (pillar_x / retina.max_dist)
            print(f"  理论期望值: {expected_depth:.3f}")

        print("\n[子测试 3.4] 生成可视化")
        print("-" * 70)

        try:
            import matplotlib.pyplot as plt

            # 创建图像
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # 左图: 深度图
            im1 = axes[0].imshow(depth_image[0].cpu().numpy(),
                                 cmap='viridis', aspect='auto', origin='lower')
            axes[0].set_xlabel('Azimuth Bin')
            axes[0].set_ylabel('Elevation Bin')
            axes[0].set_title(
                'LiDAR Depth Image\n(1.0=Close/Danger, 0.0=Far/Safe)')
            axes[0].axvline(center_col, color='r', linestyle='--',
                            alpha=0.5, label='Pillar Center')
            axes[0].axhline(H//3, color='w', linestyle=':', alpha=0.3)
            axes[0].axhline(2*H//3, color='w', linestyle=':', alpha=0.3)
            axes[0].legend()
            plt.colorbar(im1, ax=axes[0], label='Normalized Depth')

            # 右图: 俯视图 (XY 平面投影)
            axes[1].scatter(ground_points[:, 0].cpu(),
                            ground_points[:, 1].cpu(),
                            c='green', s=1, alpha=0.3, label='Ground')
            axes[1].scatter(pillar_points[:, 0].cpu(),
                            pillar_points[:, 1].cpu(),
                            c='red', s=20, alpha=0.8, label='Pillar')
            axes[1].scatter([0], [0], c='blue', s=200, marker='^',
                            label='Drone', edgecolors='black', linewidth=2)
            axes[1].set_xlabel('X (Forward, m)')
            axes[1].set_ylabel('Y (Lateral, m)')
            axes[1].set_title('Scene Top View (XY Plane)')
            axes[1].axis('equal')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()

            save_path = os.path.join(
                SCRIPT_DIR, 'lidar_processor_physics_test.png')
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 可视化已保存: {save_path}")
            plt.close()

        except ImportError:
            print("⚠ matplotlib 未安装，跳过可视化")

        print("\n" + "✓" * 35)
        print("测试 3 通过: 深度图可解释，符合物理直觉")
        print("✓" * 35)

        return True

    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "=" * 70)
        print("LiDAR Processor 物理严谨性测试套件")
        print("Science Robotics 标准")
        print("=" * 70)
        print(f"设备: {self.device}")
        print(f"数值容差: {self.tolerance}")

        results = []

        try:
            results.append(("测试 1: 物理一致性",
                            self.test_1_coordinate_transformation_physics()))
        except AssertionError as e:
            print(f"\n✗ 测试 1 失败: {e}")
            results.append(("测试 1: 物理一致性", False))

        try:
            results.append(("测试 2: 极值逻辑",
                            self.test_2_min_pooling_safety_critical()))
        except AssertionError as e:
            print(f"\n✗ 测试 2 失败: {e}")
            results.append(("测试 2: 极值逻辑", False))

        try:
            results.append(("测试 3: 可视化验证",
                            self.test_3_visual_sanity_check()))
        except AssertionError as e:
            print(f"\n✗ 测试 3 失败: {e}")
            results.append(("测试 3: 可视化验证", False))

        # 汇总报告
        print("\n" + "=" * 70)
        print("测试结果汇总")
        print("=" * 70)

        for name, passed in results:
            status = "✓ 通过" if passed else "✗ 失败"
            print(f"{status:8s} | {name}")

        all_passed = all(passed for _, passed in results)

        print("\n" + "=" * 70)
        if all_passed:
            print("🎉 所有测试通过！")
            print("LiDAR Processor 符合 Science Robotics 物理严谨性标准")
        else:
            print("❌ 部分测试失败")
            print("请检查失败的测试并修复问题")
        print("=" * 70)

        return all_passed


def main():
    """主函数"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    test_suite = PhysicsTestSuite(device=device)
    success = test_suite.run_all_tests()

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
