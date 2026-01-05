#!/usr/bin/env python3

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

"""
Unitree G1 机器人的 DDS 到 ZMQ 桥接服务器。

此服务器在机器人上运行并转发：
- 机器人状态 (LowState) 从 DDS 到 ZMQ（用于远程客户端）
- 机器人命令 (LowCmd) 从 ZMQ 到 DDS（来自远程客户端）

使用 JSON 进行安全序列化，而不是 pickle。
"""

import base64
import contextlib
import json
import threading
import time
from typing import Any

import zmq
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as hg_LowCmd, LowState_ as hg_LowState
from unitree_sdk2py.utils.crc import CRC

# DDS 主题名称遵循 Unitree SDK 命名约定
# ruff: noqa: N816
kTopicLowCommand_Debug = "rt/lowcmd"  # 发送到机器人的动作
kTopicLowState = "rt/lowstate"  # 来自机器人的观察

LOWCMD_PORT = 6000
LOWSTATE_PORT = 6001
NUM_MOTORS = 35


def lowstate_to_dict(msg: hg_LowState) -> dict[str, Any]:
    """将 LowState SDK 消息转换为可 JSON 序列化的字典。"""
    motor_states = []
    for i in range(NUM_MOTORS):
        temp = msg.motor_state[i].temperature
        avg_temp = float(sum(temp) / len(temp)) if isinstance(temp, list) else float(temp)
        motor_states.append(
            {
                "q": float(msg.motor_state[i].q),
                "dq": float(msg.motor_state[i].dq),
                "tau_est": float(msg.motor_state[i].tau_est),
                "temperature": avg_temp,
            }
        )

    return {
        "motor_state": motor_states,
        "imu_state": {
            "quaternion": [float(x) for x in msg.imu_state.quaternion],
            "gyroscope": [float(x) for x in msg.imu_state.gyroscope],
            "accelerometer": [float(x) for x in msg.imu_state.accelerometer],
            "rpy": [float(x) for x in msg.imu_state.rpy],
            "temperature": float(msg.imu_state.temperature),
        },
        # 将字节编码为 base64 以兼容 JSON
        "wireless_remote": base64.b64encode(bytes(msg.wireless_remote)).decode("ascii"),
        "mode_machine": int(msg.mode_machine),
    }


def dict_to_lowcmd(data: dict[str, Any]) -> hg_LowCmd:
    """将字典转换回 LowCmd SDK 消息。"""
    cmd = unitree_hg_msg_dds__LowCmd_()
    cmd.mode_pr = data.get("mode_pr", 0)
    cmd.mode_machine = data.get("mode_machine", 0)

    for i, motor_data in enumerate(data.get("motor_cmd", [])):
        cmd.motor_cmd[i].mode = motor_data.get("mode", 0)
        cmd.motor_cmd[i].q = motor_data.get("q", 0.0)
        cmd.motor_cmd[i].dq = motor_data.get("dq", 0.0)
        cmd.motor_cmd[i].kp = motor_data.get("kp", 0.0)
        cmd.motor_cmd[i].kd = motor_data.get("kd", 0.0)
        cmd.motor_cmd[i].tau = motor_data.get("tau", 0.0)

    return cmd


def state_forward_loop(
    lowstate_sub: ChannelSubscriber,
    lowstate_sock: zmq.Socket,
    state_period: float,
) -> None:
    """从 DDS 读取观察并转发到 ZMQ 客户端。"""
    last_state_time = 0.0

    while True:
        # 从 DDS 读取
        msg = lowstate_sub.Read()
        if msg is None:
            continue

        now = time.time()
        # 可选的降采样（如果机器人 DDS 速率 > state_period）
        if now - last_state_time >= state_period:
            # 转换为字典并使用 JSON 序列化
            state_dict = lowstate_to_dict(msg)
            payload = json.dumps({"topic": kTopicLowState, "data": state_dict}).encode("utf-8")
            # 如果没有订阅者/发送缓冲区满，则丢弃
            with contextlib.suppress(zmq.Again):
                lowstate_sock.send(payload, zmq.NOBLOCK)
            last_state_time = now


def cmd_forward_loop(
    lowcmd_sock: zmq.Socket,
    lowcmd_pub_debug: ChannelPublisher,
    crc: CRC,
) -> None:
    """从 ZMQ 接收命令并转发到 DDS。"""
    while True:
        payload = lowcmd_sock.recv()
        msg_dict = json.loads(payload.decode("utf-8"))

        topic = msg_dict.get("topic", "")
        cmd_data = msg_dict.get("data", {})

        # 从字典重构 LowCmd 对象
        cmd = dict_to_lowcmd(cmd_data)

        # 重新计算 crc
        cmd.crc = crc.Crc(cmd)

        if topic == kTopicLowCommand_Debug:
            lowcmd_pub_debug.Write(cmd)


def main() -> None:
    """机器人服务器桥接的主入口点。"""
    # 初始化 DDS
    ChannelFactoryInitialize(0)

    # 停止机器上所有活动的发布者
    msc = MotionSwitcherClient()
    msc.SetTimeout(5.0)
    msc.Init()

    status, result = msc.CheckMode()
    while result is not None and "name" in result and result["name"]:
        msc.ReleaseMode()
        status, result = msc.CheckMode()
        time.sleep(1.0)

    crc = CRC()

    # 初始化 DDS 发布者
    lowcmd_pub_debug = ChannelPublisher(kTopicLowCommand_Debug, hg_LowCmd)
    lowcmd_pub_debug.Init()

    # 初始化 DDS 订阅者
    lowstate_sub = ChannelSubscriber(kTopicLowState, hg_LowState)
    lowstate_sub.Init()

    # 初始化 ZMQ
    ctx = zmq.Context.instance()

    # 从远程客户端接收命令
    lowcmd_sock = ctx.socket(zmq.PULL)
    lowcmd_sock.bind(f"tcp://0.0.0.0:{LOWCMD_PORT}")

    # 向远程客户端发布状态
    lowstate_sock = ctx.socket(zmq.PUB)
    lowstate_sock.bind(f"tcp://0.0.0.0:{LOWSTATE_PORT}")

    state_period = 0.002  # ~500 Hz

    # 启动观察转发线程
    t_state = threading.Thread(
        target=state_forward_loop,
        args=(lowstate_sub, lowstate_sock, state_period),
        daemon=True,
    )
    t_state.start()

    # 启动动作转发线程
    t_cmd = threading.Thread(
        target=cmd_forward_loop,
        args=(lowcmd_sock, lowcmd_pub_debug, crc),
        daemon=True,
    )
    t_cmd.start()

    print("bridge running (lowstate -> zmq, lowcmd -> dds)")
    # 保持主线程存活，以便守护线程不会退出
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("shutting down bridge...")


if __name__ == "__main__":
    main()
