#!/usr/bin/env python

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

import inspect
import logging
import os
import time
from functools import cached_property
from itertools import chain
from typing import Any
import sys

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_lekiwi import LeKiwiConfig

logger = logging.getLogger(__name__)

from .lift_axis import LiftAxis, LiftAxisConfig


class LeKiwi(Robot):
    """
    该机器人包括一个三轮全向移动底盘和远程从控臂。
    主控臂本地连接（在笔记本电脑上），其关节位置被记录，然后转发到远程从控臂（在应用安全限制后）。
    同时，键盘遥操作用于生成轮子的原始速度命令。
    """

    config_class = LeKiwiConfig
    name = "lekiwi"

    def __init__(self, config: LeKiwiConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100


        self.left_bus = FeetechMotorsBus(
            port=self.config.left_port,
            motors={
                # arm
                "arm_left_shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "arm_left_shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "arm_left_elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "arm_left_wrist_flex": Motor(4, "sts3215", norm_mode_body),
                #"arm_left_wrist_yaw": Motor(5, "sts3215", norm_mode_body),
                "arm_left_wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "arm_left_gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
                # base
                "base_left_wheel": Motor(8, "sts3215", MotorNormMode.RANGE_M100_100),
                "base_back_wheel": Motor(9, "sts3215", MotorNormMode.RANGE_M100_100),
                "base_right_wheel": Motor(10, "sts3215", MotorNormMode.RANGE_M100_100),
                "lift_axis": Motor(11, "sts3215", MotorNormMode.DEGREES),
            },
            calibration=self.calibration,
        )

        self.right_bus = FeetechMotorsBus(
            port=self.config.right_port,
            motors={
                # arm
                "arm_right_shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "arm_right_shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "arm_right_elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "arm_right_wrist_flex": Motor(4, "sts3215", norm_mode_body),
                #"arm_right_wrist_yaw": Motor(5, "sts3215", norm_mode_body),
                "arm_right_wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "arm_right_gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
                #"lift_axis": Motor(12, "sts3215", MotorNormMode.DEGREES),
            },
            calibration=self.calibration,
        )


        self.left_arm_motors  = [m for m in self.left_bus.motors        if m.startswith("arm_left_")]
        self.base_motors      = [m for m in self.left_bus.motors        if m.startswith("base_")]
        #self.left_arm_motors  = [m for m in self.left_bus.motors        if m.startswith("right_arm_")]

        self.right_arm_motors = [m for m in (self.right_bus.motors if self.right_bus else []) if m.startswith("arm_right_")]

        # self.arm_motors = [motor for motor in self.left_bus.motors if motor.startswith("arm")]
        # self.base_motors = [motor for motor in self.left_bus.motors if motor.startswith("base")]

        self.cameras = make_cameras_from_configs(config.cameras)


        self.lift = LiftAxis(
        LiftAxisConfig(),        
        bus_left=self.left_bus,
        bus_right=self.right_bus,
)
        # 过流去抖：需要 N 次连续超限读取
        self._overcurrent_count: dict[str, int] = {}
        self._overcurrent_trip_n = 20


    @property
    def _state_ft(self) -> dict[str, type]:
        return dict.fromkeys(
            (
                "arm_left_shoulder_pan.pos",
                "arm_left_shoulder_lift.pos",
                "arm_left_elbow_flex.pos",
                "arm_left_wrist_flex.pos",
                #"left_wrist_yaw.pos",
                "arm_left_wrist_roll.pos",
                "arm_left_gripper.pos",
                "arm_right_shoulder_pan.pos",
                "arm_right_shoulder_lift.pos",
                "arm_right_elbow_flex.pos",
                "arm_right_wrist_flex.pos",
                #"right_wrist_yaw.pos",
                "arm_right_wrist_roll.pos",
                "arm_right_gripper.pos",
                "x.vel",
                "y.vel",
                "theta.vel",
                "lift_axis.height_mm",   # ← 新增
                #"lift_axis.vel",         # ← 新增（可选，做调试用）
            ),
            float,
        )

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._state_ft

    # @property
    # def is_connected(self) -> bool:
    #     return self.left_bus.is_connected and all(cam.is_connected for cam in self.cameras.values())
    
    @property
    def is_connected(self) -> bool:
        cams_ok = all(cam.is_connected for cam in self.cameras.values())
        return self.left_bus.is_connected and (self.right_bus.is_connected if self.right_bus else True) and cams_ok



    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.left_bus.connect()
        self.right_bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

        self.lift.home()
        print("升降轴已归零至 0mm。")

        

    @property
    def is_calibrated(self) -> bool:
        return self.left_bus.is_calibrated

    def calibrate(self) -> None:
        """
        双臂校准（左臂 + 底盘在 self.left_bus，右臂在 self.right_bus）：
        - 左臂：位置模式 → 半转归零 → 收集运动范围
        - 底盘：无需归零；运动范围固定为 0–4095
        - 右臂（如果存在）：位置模式 → 半转归零 → 收集运动范围
        - 合并为单个 self.calibration，按总线分割，写回两个总线，并保存
        """
        # 如果校准文件已存在：加载它并写回，为每个总线分别过滤
        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, "
                f"or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info("Writing existing calibration to both buses (trim per-bus caches)")

                calib_left = {k: v for k, v in self.calibration.items() if k in self.left_bus.motors}
                self.left_bus.write_calibration(calib_left, cache=False)
                self.left_bus.calibration = calib_left

                if getattr(self, "right_bus", None):
                    calib_right = {k: v for k, v in self.calibration.items() if k in self.right_bus.motors}
                    self.right_bus.write_calibration(calib_right, cache=False)
                    self.right_bus.calibration = calib_right

                return

        logger.info(f"\nRunning calibration of {self} (dual-bus if right_bus present)")

        if not getattr(self, "left_arm_motors", None):
            raise RuntimeError("left_arm_motors is empty; expected names starting with 'left_arm_'")

        self.left_bus.disable_torque(self.left_arm_motors)
        for name in self.left_arm_motors:
            self.left_bus.write("Operating_Mode", name, OperatingMode.POSITION.value)

        input("Move LEFT arm to the middle of its range of motion, then press ENTER...")
        left_homing = self.left_bus.set_half_turn_homings(self.left_arm_motors)  # 仅左臂条目

        for wheel in self.base_motors:
            left_homing[wheel] = 0

        motors_left_all = self.left_arm_motors + self.base_motors
        full_turn_left = [m for m in motors_left_all if m.startswith("base_")]  # 三个轮子
        unknown_left = [m for m in motors_left_all if m not in full_turn_left]

        print("按顺序移动左臂关节至完整运动范围。按 ENTER 键停止...")
        l_mins, l_maxs = self.left_bus.record_ranges_of_motion(unknown_left)
        for m in full_turn_left:
            l_mins[m] = 0
            l_maxs[m] = 4095

        right_homing = {}
        r_mins, r_maxs = {}, {}

        if getattr(self, "right_bus", None) and getattr(self, "right_arm_motors", None):
            self.right_bus.disable_torque(self.right_arm_motors)
            for name in self.right_arm_motors:
                self.right_bus.write("Operating_Mode", name, OperatingMode.POSITION.value)

            input("Move RIGHT arm to the middle of its range of motion, then press ENTER...")
            right_homing = self.right_bus.set_half_turn_homings(self.right_arm_motors)

            print("按顺序移动右臂关节至完整运动范围。按 ENTER 键停止...")
            r_mins, r_maxs = self.right_bus.record_ranges_of_motion(self.right_arm_motors)

        # 合并 → 按总线过滤并写回 → 保存为单个文件
        self.calibration = {}

        for name, motor in self.left_bus.motors.items():
            self.calibration[name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=left_homing.get(name, 0),
                range_min=l_mins.get(name, 0),
                range_max=l_maxs.get(name, 4095),
            )

        if getattr(self, "right_bus", None):
            for name, motor in self.right_bus.motors.items():
                self.calibration[name] = MotorCalibration(
                    id=motor.id,
                    drive_mode=0,
                    homing_offset=right_homing.get(name, 0),
                    range_min=r_mins.get(name, 0),
                    range_max=r_maxs.get(name, 4095),
                )

        # 写回：每个总线只写入自己的条目以避免 KeyError
        calib_left = {k: v for k, v in self.calibration.items() if k in self.left_bus.motors}
        self.left_bus.write_calibration(calib_left, cache=False)
        self.left_bus.calibration = calib_left

        if getattr(self, "right_bus", None):
            calib_right = {k: v for k, v in self.calibration.items() if k in self.right_bus.motors}
            self.right_bus.write_calibration(calib_right, cache=False)
            self.right_bus.calibration = calib_right

        self._save_calibration()
        print("校准已保存至", self.calibration_fpath)





    def configure(self):
        # 设置机械臂执行器（位置模式）
        # 我们假设在连接时，机械臂处于静止位置，
        # 可以安全地禁用扭矩以运行校准。
        self.left_bus.disable_torque()
        self.left_bus.configure_motors()
        for name in self.left_arm_motors:
            self.left_bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
            # 将 P_Coefficient 设置为较低值以避免抖动（默认值为 32）
            self.left_bus.write("P_Coefficient", name, 16)
            # 将 I_Coefficient 和 D_Coefficient 设置为默认值 0 和 32
            self.left_bus.write("I_Coefficient", name, 0)
            self.left_bus.write("D_Coefficient", name, 32)

        for name in self.base_motors:
            self.left_bus.write("Operating_Mode", name, OperatingMode.VELOCITY.value)

        #self.left_bus.enable_torque()

        self.right_bus.disable_torque()
        self.right_bus.configure_motors()
        for name in self.right_arm_motors:
            self.right_bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
            self.right_bus.write("P_Coefficient", name, 16)
            self.right_bus.write("I_Coefficient", name, 0)
            self.right_bus.write("D_Coefficient", name, 32)
        #self.right_bus.enable_torque()

        #self.lift.configure()




    def setup_motors(self) -> None:
        for motor in chain(reversed(self.arm_motors), reversed(self.base_motors)):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.left_bus.setup_motor(motor)
            print(f"'{motor}' 电机 ID 已设置为 {self.left_bus.motors[motor].id}")

    @staticmethod
    def _degps_to_raw(degps: float) -> int:
        steps_per_deg = 4096.0 / 360.0
        speed_in_steps = degps * steps_per_deg
        speed_int = int(round(speed_in_steps))
        # 限制值以适应有符号 16 位范围 (-32768 到 32767)
        if speed_int > 0x7FFF:
            speed_int = 0x7FFF  # 32767 -> 最大正值
        elif speed_int < -0x8000:
            speed_int = -0x8000  # -32768 -> 最小负值
        return speed_int

    @staticmethod
    def _raw_to_degps(raw_speed: int) -> float:
        steps_per_deg = 4096.0 / 360.0
        magnitude = raw_speed
        degps = magnitude / steps_per_deg
        return degps

    def _body_to_wheel_raw(
        self,
        x: float,
        y: float,
        theta: float,
        wheel_radius: float = 0.05,
        base_radius: float = 0.125,
        max_raw: int = 3000,
    ) -> dict:
        """
        将期望的机体坐标系速度转换为轮子原始命令。

        参数:
          x_cmd      : x 方向线速度 (m/s)。
          y_cmd      : y 方向线速度 (m/s)。
          theta_cmd  : 旋转速度 (deg/s)。
          wheel_radius: 每个轮子的半径（米）。
          base_radius : 从旋转中心到每个轮子的距离（米）。
          max_raw    : 每个轮子允许的最大原始命令（刻度）。

        返回:
          包含轮子原始命令的字典：
             {"base_left_wheel": value, "base_back_wheel": value, "base_right_wheel": value}。

        注意:
          - 内部，该方法将 theta_cmd 转换为 rad/s 用于运动学计算。
          - 原始命令从轮子角速度（deg/s）计算得出，使用 _degps_to_raw()。
            如果任何命令超过 max_raw，所有命令按比例缩小。
        """
        # 将旋转速度从 deg/s 转换为 rad/s。
        theta_rad = theta * (np.pi / 180.0)
        # 创建机体速度向量 [x, y, theta_rad]。
        velocity_vector = np.array([-x, -y, theta_rad])

        # 定义轮子安装角度，偏移 -90°。
        angles = np.radians(np.array([240, 0, 120]) - 90)
        # 构建运动学矩阵：每行将机体速度映射到轮子的线速度。
        # 第三列 (base_radius) 考虑了旋转的影响。
        m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

        # 计算每个轮子的线速度 (m/s)，然后计算其角速度 (rad/s)。
        wheel_linear_speeds = m.dot(velocity_vector)
        wheel_angular_speeds = wheel_linear_speeds / wheel_radius

        # 将轮子角速度从 rad/s 转换为 deg/s。
        wheel_degps = wheel_angular_speeds * (180.0 / np.pi)

        # 缩放
        steps_per_deg = 4096.0 / 360.0
        raw_floats = [abs(degps) * steps_per_deg for degps in wheel_degps]
        max_raw_computed = max(raw_floats)
        if max_raw_computed > max_raw:
            scale = max_raw / max_raw_computed
            wheel_degps = wheel_degps * scale

        # 将每个轮子的角速度 (deg/s) 转换为原始整数。
        wheel_raw = [self._degps_to_raw(deg) for deg in wheel_degps]

        return {
            "base_left_wheel": wheel_raw[0],
            "base_back_wheel": wheel_raw[1],
            "base_right_wheel": wheel_raw[2],
        }

    def _wheel_raw_to_body(
        self,
        left_wheel_speed,
        back_wheel_speed,
        right_wheel_speed,
        wheel_radius: float = 0.05,
        base_radius: float = 0.125,
    ) -> dict[str, Any]:
        """
        将轮子原始命令反馈转换回机体坐标系速度。

        参数:
          wheel_raw   : 包含原始轮子命令的向量 ("base_left_wheel", "base_back_wheel", "base_right_wheel")。
          wheel_radius: 每个轮子的半径（米）。
          base_radius : 从机器人中心到每个轮子的距离（米）。

        返回:
          字典 (x.vel, y.vel, theta.vel)，单位均为 m/s
        """

        # 将每个原始命令转换回角速度 (deg/s)。
        wheel_degps = np.array(
            [
                self._raw_to_degps(left_wheel_speed),
                self._raw_to_degps(back_wheel_speed),
                self._raw_to_degps(right_wheel_speed),
            ]
        )

        # 从 deg/s 转换为 rad/s。
        wheel_radps = wheel_degps * (np.pi / 180.0)
        # 从角速度计算每个轮子的线速度 (m/s)。
        wheel_linear_speeds = wheel_radps * wheel_radius

        # 定义轮子安装角度，偏移 -90°。
        angles = np.radians(np.array([240, 0, 120]) - 90)
        m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

        # 求解逆运动学：body_velocity = M⁻¹ · wheel_linear_speeds。
        m_inv = np.linalg.inv(m)
        velocity_vector = m_inv.dot(wheel_linear_speeds)
        x, y, theta_rad = velocity_vector
        
        theta = theta_rad * (180.0 / np.pi)
        return {
            "x.vel": x,
            "y.vel": y,
            "theta.vel": theta,
        }  # m/s and deg/s
    
    def _raw_to_ma(raw):
        try:
            return float(raw) * 6.5
        except Exception:
            return 0.0
        
    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # 读取机械臂执行器位置和底盘速度
        start = time.perf_counter()
        # arm_pos = self.left_bus.sync_read("Present_Position", self.arm_motors)

        #print(f"Left arm motors: {self.left_arm_motors}, Right arm motors: {self.right_arm_motors}")  # debug
        left_pos = self.left_bus.sync_read("Present_Position", self.left_arm_motors)   # left_arm_*


        base_wheel_vel = self.left_bus.sync_read("Present_Velocity", self.base_motors)

        base_vel = self._wheel_raw_to_body(
            base_wheel_vel["base_left_wheel"],
            base_wheel_vel["base_back_wheel"],
            base_wheel_vel["base_right_wheel"],
        )

        right_pos = self.right_bus.sync_read("Present_Position", self.right_arm_motors)  # right_arm_*


        left_arm_state = {f"{k}.pos": v for k, v in left_pos.items()}
        right_arm_state = {f"{k}.pos": v for k, v in right_pos.items()}

        obs_dict = {**left_arm_state, **right_arm_state,**base_vel}
        self.lift.contribute_observation(obs_dict)
        #print(f"Observation dict so far: {obs_dict}")  # debug

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # 电流保护
        self.read_and_check_currents(limit_ma=2000, print_currents=True)

        # 从相机捕获图像
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """命令 AlohaMini 移动到目标关节配置。

        相对动作幅度可能会根据配置参数 `max_relative_target` 被裁剪。
        在这种情况下，发送的动作与原始动作不同。
        因此，此函数总是返回实际发送的动作。

        抛出:
            RobotDeviceNotConnectedError: 如果机器人未连接。

        返回:
            np.ndarray: 发送到电机的动作，可能已被裁剪。
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # arm_goal_pos = {k: v for k, v in action.items() if k.endswith(".pos")}
        left_pos  = {k: v for k, v in action.items() if k.endswith(".pos") and k.startswith("arm_left_")}
        right_pos = {k: v for k, v in action.items() if k.endswith(".pos") and k.startswith("arm_right_")}


        base_goal_vel = {k: v for k, v in action.items() if k.endswith(".vel")}

        base_wheel_goal_vel = self._body_to_wheel_raw(
            base_goal_vel["x.vel"], base_goal_vel["y.vel"], base_goal_vel["theta.vel"]
        )

        # 当目标位置距离当前位置太远时限制目标位置。
        # /!\ 由于从从控臂读取数据，预期帧率会降低。
        # if self.config.max_relative_target is not None:
        #     present_pos = self.left_bus.sync_read("Present_Position", self.arm_motors)
        #     goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in arm_goal_pos.items()}
        #     arm_safe_goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)
        #     arm_goal_pos = arm_safe_goal_pos

        self.lift.apply_action(action)

        if left_pos and self.config.max_relative_target is not None:
            present_left = self.left_bus.sync_read("Present_Position", self.left_arm_motors)  # left_arm_*
            gp_left = {k: (v, present_left[k.replace(".pos", "")]) for k, v in left_pos.items()}
            left_pos = ensure_safe_goal_position(gp_left, self.config.max_relative_target)

        if self.right_bus and right_pos and self.config.max_relative_target is not None:
            present_right = self.right_bus.sync_read("Present_Position", self.right_arm_motors)
            gp_right = {k: (v, present_right[k.replace(".pos", "")]) for k, v in right_pos.items()}
            right_pos = ensure_safe_goal_position(gp_right, self.config.max_relative_target)


        # 发送目标位置到执行器
        # arm_goal_pos_raw = {k.replace(".pos", ""): v for k, v in arm_goal_pos.items()}
        # self.left_bus.sync_write("Goal_Position", arm_goal_pos_raw)
        # self.left_bus.sync_write("Goal_Velocity", base_wheel_goal_vel)

        # return {**arm_goal_pos, **base_goal_vel}

        #print(f"[{filename}:{lineno}]Sending left_pos:{left_pos}, right_pos:{right_pos}, base_wheel_goal_vel:{base_wheel_goal_vel}")  # debug
    
        if left_pos:
            self.left_bus.sync_write("Goal_Position", {k.replace(".pos", ""): v for k, v in left_pos.items()})
        if self.right_bus and right_pos:
            self.right_bus.sync_write("Goal_Position", {k.replace(".pos", ""): v for k, v in right_pos.items()})
        self.left_bus.sync_write("Goal_Velocity", base_wheel_goal_vel)

        lift_sent = {k: v for k, v in action.items() if k.startswith("lift_axis.")}
        return {**left_pos, **right_pos, **base_goal_vel, **lift_sent}


    def stop_base(self):
        self.left_bus.sync_write("Goal_Velocity", dict.fromkeys(self.base_motors, 0), num_retry=0)
        logger.info("Base motors stopped")

    def read_and_check_currents(self, limit_ma, print_currents):
        """读取左/右总线电流 (mA)，打印它们，并执行过流保护"""
        scale = 6.5  # sts3215 电流单位转换系数
        left_curr_raw = {}
        left_curr_raw = self.left_bus.sync_read("Present_Current", list(self.left_bus.motors.keys()))
        right_curr_raw = {}
        if getattr(self, "right_bus", None):
            right_curr_raw = self.right_bus.sync_read("Present_Current", list(self.right_bus.motors.keys()))

        if print_currents:
            left_line = "{" + ",".join(str(int(v * scale)) for v in left_curr_raw.values()) + "}"
            #print(f"Left Bus currents(ma): {left_line}")
            if right_curr_raw:
                right_line = "{" + ",".join(str(int(v * scale)) for v in right_curr_raw.values()) + "}"
                #print(f"Right Bus currents(ma): {right_line}")

        tripped = None
        for name, raw in {**left_curr_raw, **right_curr_raw}.items():
            current_ma = float(raw) * scale

            if current_ma > limit_ma:
                self._overcurrent_count[name] = self._overcurrent_count.get(name, 0) + 1
                print(f"[过流] {name}: {current_ma:.1f} mA > {limit_ma:.1f} mA ")
            else:
                # 当恢复正常时重置 -> "连续"语义
                self._overcurrent_count[name] = 0

            if self._overcurrent_count[name] >= self._overcurrent_trip_n:
                tripped = (name, current_ma, self._overcurrent_count[name])
                break

        if tripped is not None:
            name, current_ma, n = tripped
            print(
                f"[过流] {name}: {current_ma:.1f} mA > {limit_ma:.1f} mA "
                f"连续 {n} 次读取超过限制，正在断开连接！"
            )
            try:
                self.stop_base()
            except Exception:
                pass
            try:
                self.disconnect()
            except Exception as e:
                print(f"[过流] 断开连接错误: {e}")
            sys.exit(1)


        return {k: round(v * scale, 1) for k, v in {**left_curr_raw, **right_curr_raw}.items()}

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.stop_base()
        self.left_bus.disconnect(self.config.disable_torque_on_disconnect)
        self.right_bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")

