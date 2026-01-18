# MIT License
#
# Copyright (c) 2024 TASLAB
#
# TASLAB UAV 无人机模型定义
# 继承自 MultirotorBase，使用自定义参数配置

from omni_drones.robots.drone.multirotor import MultirotorBase
from omni_drones.robots.robot import ASSET_PATH


class TaslabUAV(MultirotorBase):
    """
    TASLAB 自定义无人机模型

    配置文件: taslab_uav.yaml
    3D模型文件: taslab_uav.usd (需要单独创建或使用现有模型)

    使用方法:
        在 drone.yaml 中设置:
        drone:
          model_name: "taslab_uav"
    """

    # USD 3D模型文件路径
    # 选项1: 使用现有的 hummingbird 模型 (快速测试)
    usd_path: str = ASSET_PATH + "/usd/hummingbird.usd"

    # 选项2: 使用自定义模型 (需要创建 taslab_uav.usd)
    # usd_path: str = ASSET_PATH + "/usd/taslab_uav.usd"

    # YAML 参数配置文件路径
    param_path: str = ASSET_PATH + "/usd/taslab_uav.yaml"
