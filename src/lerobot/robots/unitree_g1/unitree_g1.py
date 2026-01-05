#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import logging
import struct
import threading
import time
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
    LowCmd_ as hg_LowCmd,
    LowState_ as hg_LowState,
)
from unitree_sdk2py.utils.crc import CRC

from lerobot.robots.unitree_g1.g1_utils import G1_29_JointIndex
from lerobot.robots.unitree_g1.unitree_sdk2_socket import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)

from ..robot import Robot
from .config_unitree_g1 import UnitreeG1Config

logger = logging.getLogger(__name__)

# DDS 主题名称遵循 Unitree SDK 命名约定
# ruff: noqa: N816
kTopicLowCommand_Debug = "rt/lowcmd"
kTopicLowState = "rt/lowstate"

G1_29_Num_Motors = 35
G1_23_Num_Motors = 35
H1_2_Num_Motors = 35
H1_Num_Motors = 20


@dataclass
class MotorState:
    q: float | None = None  # 位置
    dq: float | None = None  # 速度
    tau_est: float | None = None  # 估计扭矩
    temperature: float | None = None  # 电机温度


@dataclass
class IMUState:
    quaternion: np.ndarray | None = None  # [w, x, y, z]
    gyroscope: np.ndarray | None = None  # [x, y, z] 角速度 (rad/s)
    accelerometer: np.ndarray | None = None  # [x, y, z] 线性加速度 (m/s²)
    rpy: np.ndarray | None = None  # [roll, pitch, yaw] (rad)
    temperature: float | None = None  # IMU 温度


# G1 观察类
@dataclass
class G1_29_LowState:  # noqa: N801
    motor_state: list[MotorState] = field(
        default_factory=lambda: [MotorState() for _ in range(G1_29_Num_Motors)]
    )
    imu_state: IMUState = field(default_factory=IMUState)
    wireless_remote: Any = None  # 原始无线遥控数据
    mode_machine: int = 0  # 机器人模式


class DataBuffer:
    def __init__(self):
        self.data = None
        self.lock = threading.Lock()

    def get_data(self):
        with self.lock:
            return self.data

    def set_data(self, data):
        with self.lock:
            self.data = data


class UnitreeG1(Robot):
    config_class = UnitreeG1Config
    name = "unitree_g1"

    # Unitree 遥控器
    class RemoteController:
        def __init__(self):
            self.lx = 0
            self.ly = 0
            self.rx = 0
            self.ry = 0
            self.button = [0] * 16

        def set(self, data):
            # 无线遥控
            keys = struct.unpack("H", data[2:4])[0]
            for i in range(16):
                self.button[i] = (keys & (1 << i)) >> i
            self.lx = struct.unpack("f", data[4:8])[0]
            self.rx = struct.unpack("f", data[8:12])[0]
            self.ry = struct.unpack("f", data[12:16])[0]
            self.ly = struct.unpack("f", data[20:24])[0]

    def __init__(self, config: UnitreeG1Config):
        super().__init__(config)

        logger.info("Initialize UnitreeG1...")

        self.config = config

        self.control_dt = config.control_dt

        # 连接机器人
        self.connect()

        # 初始化直接电机控制接口
        self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Debug, hg_LowCmd)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber(kTopicLowState, hg_LowState)
        self.lowstate_subscriber.Init()
        self.lowstate_buffer = DataBuffer()

        # 初始化订阅线程以读取机器人状态
        self.subscribe_thread = threading.Thread(target=self._subscribe_motor_state)
        self.subscribe_thread.daemon = True
        self.subscribe_thread.start()

        while not self.is_connected:
            time.sleep(0.1)

        # 初始化 hg 的 lowcmd 消息
        self.crc = CRC()
        self.msg = unitree_hg_msg_dds__LowCmd_()
        self.msg.mode_pr = 0

        # 等待第一个状态消息到达
        lowstate = None
        while lowstate is None:
            lowstate = self.lowstate_buffer.get_data()
            if lowstate is None:
                time.sleep(0.01)
            logger.warning("[UnitreeG1] Waiting for robot state...")
        logger.warning("[UnitreeG1] Connected to robot.")
        self.msg.mode_machine = lowstate.mode_machine

        # 使用配置中的统一 kp/kd 初始化所有电机
        self.kp = np.array(config.kp, dtype=np.float32)
        self.kd = np.array(config.kd, dtype=np.float32)

        for id in G1_29_JointIndex:
            self.msg.motor_cmd[id].mode = 1
            self.msg.motor_cmd[id].kp = self.kp[id.value]
            self.msg.motor_cmd[id].kd = self.kd[id.value]
            self.msg.motor_cmd[id].q = lowstate.motor_state[id.value].q

        # 初始化遥控器
        self.remote_controller = self.RemoteController()

    def _subscribe_motor_state(self):  # 以 250Hz 轮询机器人状态
        while True:
            start_time = time.time()
            msg = self.lowstate_subscriber.Read()
            if msg is not None:
                lowstate = G1_29_LowState()

                # 捕获电机状态
                for id in range(G1_29_Num_Motors):
                    lowstate.motor_state[id].q = msg.motor_state[id].q
                    lowstate.motor_state[id].dq = msg.motor_state[id].dq
                    lowstate.motor_state[id].tau_est = msg.motor_state[id].tau_est
                    lowstate.motor_state[id].temperature = msg.motor_state[id].temperature

                # 捕获 IMU 状态
                lowstate.imu_state.quaternion = list(msg.imu_state.quaternion)
                lowstate.imu_state.gyroscope = list(msg.imu_state.gyroscope)
                lowstate.imu_state.accelerometer = list(msg.imu_state.accelerometer)
                lowstate.imu_state.rpy = list(msg.imu_state.rpy)
                lowstate.imu_state.temperature = msg.imu_state.temperature

                # 捕获无线遥控数据
                lowstate.wireless_remote = msg.wireless_remote

                # 捕获 mode_machine
                lowstate.mode_machine = msg.mode_machine

                self.lowstate_buffer.set_data(lowstate)

            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, (self.control_dt - all_t_elapsed))  # 维持恒定的控制 dt
            time.sleep(sleep_time)

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{G1_29_JointIndex(motor).name}.pos": float for motor in G1_29_JointIndex}

    def calibrate(self) -> None:  # 机器人已经校准
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:  # 连接到 DDS
        ChannelFactoryInitialize(0)

    def disconnect(self):
        pass

    def get_observation(self) -> dict[str, Any]:
        return self.lowstate_buffer.get_data()

    @property
    def is_calibrated(self) -> bool:
        return True

    @property
    def is_connected(self) -> bool:
        return self.lowstate_buffer.get_data() is not None

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{G1_29_JointIndex(motor).name}.pos": float for motor in G1_29_JointIndex}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        self.msg.crc = self.crc.Crc(action)
        self.lowcmd_publisher.Write(action)
        return action

    def get_gravity_orientation(self, quaternion):  # 从四元数获取重力方向
        """从四元数获取重力方向。"""
        qw = quaternion[0]
        qx = quaternion[1]
        qy = quaternion[2]
        qz = quaternion[3]

        gravity_orientation = np.zeros(3)
        gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
        gravity_orientation[1] = -2 * (qz * qy + qw * qx)
        gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
        return gravity_orientation
