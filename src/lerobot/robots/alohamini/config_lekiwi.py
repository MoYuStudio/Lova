# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from lerobot.cameras.configs import CameraConfig, Cv2Rotation
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

from ..config import RobotConfig


def lekiwi_cameras_config() -> dict[str, CameraConfig]:
    return {
        # "head_top": OpenCVCameraConfig(
        #     index_or_path="/dev/am_camera_head_top", fps=30, width=640, height=480, rotation=Cv2Rotation.NO_ROTATION
        # ),
        # "head_back": OpenCVCameraConfig(
        #     index_or_path="/dev/am_camera_head_back", fps=30, width=640, height=480, rotation=Cv2Rotation.NO_ROTATION
        # ),
        # "head_front": OpenCVCameraConfig(
        #     index_or_path="/dev/am_camera_head_front", fps=30, width=640, height=480, rotation=Cv2Rotation.NO_ROTATION
        # ),
        # "wrist_left": OpenCVCameraConfig(
        #     index_or_path="/dev/am_camera_wrist_left", fps=30, width=640, height=480, rotation=Cv2Rotation.NO_ROTATION
        # ),
        # "wrist_right": OpenCVCameraConfig(
        #     index_or_path="/dev/am_camera_wrist_right", fps=30, width=640, height=480, rotation=Cv2Rotation.NO_ROTATION
        # ),
    }


@RobotConfig.register_subclass("lekiwi")
@dataclass
class LeKiwiConfig(RobotConfig):
    left_port: str = "/dev/am_arm_follower_left"  # 连接到总线的端口
    right_port: str = "/dev/am_arm_follower_right"  # 连接到总线的端口
    disable_torque_on_disconnect: bool = True

    # `max_relative_target` 为了安全目的限制相对位置目标向量的大小。
    # 设置为正标量以为所有电机使用相同的值，或设置为与从控臂电机数量相同长度的列表。
    max_relative_target: int | None = None

    cameras: dict[str, CameraConfig] = field(default_factory=lekiwi_cameras_config)

    # 设置为 `True` 以与之前的策略/数据集向后兼容
    use_degrees: bool = False




@dataclass
class LeKiwiHostConfig:
    # 网络配置
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556

    # 应用程序持续时间
    connection_time_s: int = 6000

    # 看门狗：如果超过 1.5 秒未收到命令，则停止机器人。
    watchdog_timeout_ms: int = 1500

    # 如果机器人抖动，降低频率并使用 `top` 命令监控 CPU 负载
    max_loop_freq_hz: int = 30




@RobotConfig.register_subclass("lekiwi_client")
@dataclass
class LeKiwiClientConfig(RobotConfig):
    # 网络配置
    remote_ip: str
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556

    teleop_keys: dict[str, str] = field(
        default_factory=lambda: {
            # 移动
            "forward": "w",
            "backward": "s",
            "left": "z",
            "right": "x",
            "rotate_left": "a",
            "rotate_right": "d",
            # 速度控制
            "speed_up": "r",
            "speed_down": "f",
            # Z 轴
            "lift_up": "u",
            "lift_down": "j",
            # 退出遥操作
            "quit": "q",
        }
    )

    cameras: dict[str, CameraConfig] = field(default_factory=lekiwi_cameras_config)

    polling_timeout_ms: int = 15
    connect_timeout_s: int = 5

