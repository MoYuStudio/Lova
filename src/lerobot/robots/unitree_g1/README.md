# Unitree G1 机器人模块开发文档

本文档为开发者提供了 Unitree G1 机器人模块的详细说明，包括架构、组件和使用方法。

## 目录

- [概述](#概述)
- [模块架构](#模块架构)
- [文件说明](#文件说明)
- [核心组件](#核心组件)
- [通信机制](#通信机制)
- [配置说明](#配置说明)
- [使用方法](#使用方法)
- [开发指南](#开发指南)

---

## 概述

Unitree G1 是一个人形机器人系统。该模块实现了机器人的控制接口，支持通过 DDS（Data Distribution Service）和 ZMQ（ZeroMQ）桥接进行远程控制和数据采集。

### 主要特性

- **35 自由度**：全身关节控制（双腿、双臂、腰部、手腕）
- **DDS/ZMQ 桥接**：支持远程控制，使用 JSON 安全序列化
- **实时控制**：250Hz 控制频率
- **状态监控**：电机状态、IMU 数据、温度监控
- **PID 控制**：可配置的 kp/kd 增益参数

---

## 模块架构

```
unitree_g1/
├── __init__.py                  # 模块导出
├── config_unitree_g1.py         # 配置类定义
├── unitree_g1.py                # 机器人主类实现
├── g1_utils.py                  # 关节索引定义
├── unitree_sdk2_socket.py       # ZMQ 客户端接口
└── run_g1_server.py             # DDS-ZMQ 桥接服务器
```

### 架构模式

系统采用 **服务器-客户端** 架构：

- **服务器端（Server）**：运行在机器人本体上，使用 DDS 与机器人通信，通过 ZMQ 桥接为远程客户端提供服务
- **客户端（Client）**：运行在控制计算机上，通过 ZMQ 与服务器通信

```
┌─────────────────┐         ZMQ          ┌─────────────────┐         DDS         ┌──────────┐
│                 │  ────────────────────>│                 │  ───────────────────>│          │
│  客户端 (PC)    │   命令 (Command)      │  桥接服务器     │   命令 (Command)     │  G1 机器人│
│                 │<────────────────────  │  (run_g1_server)│<────────────────────│          │
│ UnitreeG1       │   状态 (State)        │                 │   状态 (State)       │          │
└─────────────────┘                       └─────────────────┘                      └──────────┘
```

---

## 文件说明

### `__init__.py`

模块初始化文件，导出主要的类和配置：

- `UnitreeG1Config` - 机器人配置
- `UnitreeG1` - 机器人主类

### `config_unitree_g1.py`

配置类定义文件。

#### `UnitreeG1Config`

机器人配置类：

- `kp` - 比例增益列表（35 个关节）
- `kd` - 微分增益列表（35 个关节）
- `control_dt` - 控制周期（默认 1/250 秒，即 250Hz）
- `robot_ip` - 机器人 IP 地址（用于 ZMQ 连接）

**默认增益配置**：

- **左/右腿**：kp=[150,150,150,300,40,40], kd=[2,2,2,4,2,2]
- **腰部**：kp=[250,250,250], kd=[5,5,5]
- **左/右臂**：kp=[80,80,80,80], kd=[3,3,3,3]
- **左/右手腕**：kp=[40,40,40], kd=[1.5,1.5,1.5]

### `unitree_g1.py`

机器人主类实现 `UnitreeG1`，继承自 `Robot` 基类。

#### 主要功能

1. **硬件通信**
   - DDS/ZMQ 桥接连接
   - 状态订阅线程（250Hz）
   - 命令发布接口

2. **状态获取**
   - 35 个关节的位置、速度、扭矩、温度
   - IMU 数据（四元数、角速度、加速度、RPY）
   - 机器人模式信息

3. **动作执行**
   - 关节位置控制
   - PID 参数配置

#### 关键类

- `MotorState` - 电机状态数据类（位置、速度、扭矩、温度）
- `IMUState` - IMU 状态数据类（四元数、角速度、加速度、RPY）
- `G1_29_LowState` - G1 机器人完整状态类
- `DataBuffer` - 线程安全的数据缓冲区
- `RemoteController` - 遥控器数据解析类

#### 关键方法

- `connect()` - 连接到机器人（初始化 DDS/ZMQ）
- `get_observation()` - 获取当前状态观察
- `send_action()` - 发送动作命令
- `_subscribe_motor_state()` - 状态订阅线程（内部）
- `get_gravity_orientation()` - 从四元数计算重力方向

### `g1_utils.py`

关节索引定义文件。

#### `G1_29_JointIndex`

完整的 35 个关节索引枚举：

- **左腿** (0-5)：HipPitch, HipRoll, HipYaw, Knee, AnklePitch, AnkleRoll
- **右腿** (6-11)：HipPitch, HipRoll, HipYaw, Knee, AnklePitch, AnkleRoll
- **腰部** (12-14)：WaistYaw, WaistRoll, WaistPitch
- **左臂** (15-21)：ShoulderPitch, ShoulderRoll, ShoulderYaw, Elbow, WristRoll, WristPitch, WristYaw
- **右臂** (22-28)：ShoulderPitch, ShoulderRoll, ShoulderYaw, Elbow, WristRoll, WristPitch, WristYaw
- **未使用** (29-34)：6 个保留关节

#### `G1_29_JointArmIndex`

仅包含手臂关节的索引枚举。

### `unitree_sdk2_socket.py`

ZMQ 客户端接口实现，提供与服务器桥接的通信接口。

#### 主要类

- `LowStateMsg` - 状态消息包装类，模仿 SDK 消息结构
- `ChannelFactoryInitialize()` - 初始化 ZMQ 套接字
- `ChannelPublisher` - 命令发布者（ZMQ PUSH）
- `ChannelSubscriber` - 状态订阅者（ZMQ SUB）

#### 关键功能

- 将 SDK 消息格式转换为 JSON
- Base64 编码/解码无线遥控数据
- 线程安全的套接字管理

### `run_g1_server.py`

DDS 到 ZMQ 桥接服务器，运行在机器人本体上。

#### 主要功能

1. **DDS 初始化**
   - 初始化 DDS 通道
   - 停止所有活动的发布者
   - 创建命令发布者和状态订阅者

2. **ZMQ 服务器**
   - 绑定端口 6000（命令接收，PULL）
   - 绑定端口 6001（状态发布，PUB）

3. **双向转发**
   - **状态转发线程**：DDS → ZMQ（~500Hz）
   - **命令转发线程**：ZMQ → DDS

#### 关键函数

- `lowstate_to_dict()` - 将 SDK LowState 转换为字典
- `dict_to_lowcmd()` - 将字典转换为 SDK LowCmd
- `state_forward_loop()` - 状态转发循环
- `cmd_forward_loop()` - 命令转发循环
- `main()` - 服务器主入口

---

## 核心组件

### 关节配置

G1 机器人共有 35 个关节：

1. **左腿** (6 个关节)
2. **右腿** (6 个关节)
3. **腰部** (3 个关节)
4. **左臂** (7 个关节)
5. **右臂** (7 个关节)
6. **保留关节** (6 个，未使用)

### 状态数据结构

#### 电机状态 (`MotorState`)

- `q` - 位置（弧度）
- `dq` - 速度（弧度/秒）
- `tau_est` - 估计扭矩（Nm）
- `temperature` - 温度（°C）

#### IMU 状态 (`IMUState`)

- `quaternion` - 四元数 [w, x, y, z]
- `gyroscope` - 角速度 [x, y, z] (rad/s)
- `accelerometer` - 线性加速度 [x, y, z] (m/s²)
- `rpy` - 欧拉角 [roll, pitch, yaw] (rad)
- `temperature` - IMU 温度（°C）

---

## 通信机制

### DDS 通信

- **协议**：Unitree SDK2 DDS（Data Distribution Service）
- **命令主题**：`rt/lowcmd`
- **状态主题**：`rt/lowstate`
- **频率**：约 250Hz（机器人原生频率）

### ZMQ 桥接

- **协议**：ZeroMQ (ZMQ)
- **命令端口**：6000 (PULL，服务器接收)
- **状态端口**：6001 (PUB，服务器发布)
- **数据格式**：JSON（安全序列化）
- **状态频率**：~500Hz（可配置降采样）

### 消息格式

#### 命令消息 (LowCmd)

```json
{
  "topic": "rt/lowcmd",
  "data": {
    "mode_pr": 0,
    "mode_machine": 0,
    "motor_cmd": [
      {
        "mode": 1,
        "q": 0.0,
        "dq": 0.0,
        "kp": 150.0,
        "kd": 2.0,
        "tau": 0.0
      },
      // ... 35 个关节
    ]
  }
}
```

#### 状态消息 (LowState)

```json
{
  "topic": "rt/lowstate",
  "data": {
    "motor_state": [
      {
        "q": 0.0,
        "dq": 0.0,
        "tau_est": 0.0,
        "temperature": 25.0
      },
      // ... 35 个关节
    ],
    "imu_state": {
      "quaternion": [1.0, 0.0, 0.0, 0.0],
      "gyroscope": [0.0, 0.0, 0.0],
      "accelerometer": [0.0, 0.0, 9.8],
      "rpy": [0.0, 0.0, 0.0],
      "temperature": 25.0
    },
    "wireless_remote": "base64_encoded_bytes",
    "mode_machine": 0
  }
}
```

---

## 配置说明

### 基本配置

```python
from lerobot.robots.unitree_g1 import UnitreeG1Config, UnitreeG1

config = UnitreeG1Config(
    robot_ip="192.168.123.164",  # 机器人 IP 地址
    control_dt=1.0 / 250.0,      # 控制周期（250Hz）
)

robot = UnitreeG1(config)
```

### 自定义 PID 增益

```python
from lerobot.robots.unitree_g1.config_unitree_g1 import _DEFAULT_KP, _DEFAULT_KD

# 复制默认增益并修改
custom_kp = _DEFAULT_KP.copy()
custom_kd = _DEFAULT_KD.copy()

# 修改特定关节的增益（例如左肩）
from lerobot.robots.unitree_g1.g1_utils import G1_29_JointIndex
custom_kp[G1_29_JointIndex.kLeftShoulderPitch.value] = 100.0
custom_kd[G1_29_JointIndex.kLeftShoulderPitch.value] = 5.0

config = UnitreeG1Config(
    kp=custom_kp,
    kd=custom_kd,
)
```

---

## 使用方法

### 服务器端运行

在机器人本体上运行桥接服务器：

```bash
python -m lerobot.robots.unitree_g1.run_g1_server
```

或使用 Python：

```python
from lerobot.robots.unitree_g1.run_g1_server import main
main()
```

服务器启动后会：
1. 初始化 DDS 连接
2. 停止所有活动的发布者
3. 绑定 ZMQ 端口（6000, 6001）
4. 启动状态和命令转发线程

### 客户端使用

在控制计算机上使用：

```python
from lerobot.robots.unitree_g1 import UnitreeG1Config, UnitreeG1

# 创建配置
config = UnitreeG1Config(robot_ip="192.168.123.164")

# 创建机器人实例（会自动连接）
robot = UnitreeG1(config)

# 获取观察
obs = robot.get_observation()

# 访问电机状态
left_shoulder_pos = obs.motor_state[G1_29_JointIndex.kLeftShoulderPitch.value].q
print(f"Left shoulder position: {left_shoulder_pos} rad")

# 访问 IMU 数据
imu_quat = obs.imu_state.quaternion
print(f"IMU quaternion: {imu_quat}")

# 发送动作
from lerobot.robots.unitree_g1.g1_utils import G1_29_JointIndex

action = {
    f"{G1_29_JointIndex.kLeftShoulderPitch.name}.pos": 0.5,  # 弧度
    f"{G1_29_JointIndex.kLeftElbow.name}.pos": -0.3,
    # ... 其他关节
}

robot.send_action(action)
```

### 完整示例

```python
from lerobot.robots.unitree_g1 import UnitreeG1Config, UnitreeG1
from lerobot.robots.unitree_g1.g1_utils import G1_29_JointIndex
import time

# 配置
config = UnitreeG1Config(robot_ip="192.168.123.164")

# 创建机器人
robot = UnitreeG1(config)

try:
    # 控制循环
    for i in range(100):
        # 获取观察
        obs = robot.get_observation()
        
        # 读取关节位置
        left_shoulder = obs.motor_state[G1_29_JointIndex.kLeftShoulderPitch.value].q
        right_shoulder = obs.motor_state[G1_29_JointIndex.kRightShoulderPitch.value].q
        
        print(f"Iteration {i}: Left={left_shoulder:.3f}, Right={right_shoulder:.3f}")
        
        # 发送动作（示例：小幅摆动）
        import math
        action = {
            f"{G1_29_JointIndex.kLeftShoulderPitch.name}.pos": 0.3 * math.sin(i * 0.1),
            f"{G1_29_JointIndex.kRightShoulderPitch.name}.pos": -0.3 * math.sin(i * 0.1),
        }
        robot.send_action(action)
        
        time.sleep(0.004)  # ~250Hz

finally:
    # 断开连接（UnitreeG1 的 disconnect 是空操作，实际连接由桥接服务器管理）
    robot.disconnect()
```

---

## 开发指南

### 添加新功能

#### 1. 修改 PID 增益

编辑 `config_unitree_g1.py` 中的 `_GAINS` 字典，或在使用时通过配置传递自定义增益。

#### 2. 添加新的观察字段

在 `unitree_g1.py` 中修改 `G1_29_LowState` 类或 `_subscribe_motor_state()` 方法。

#### 3. 修改控制频率

修改配置中的 `control_dt` 参数，或修改 `_subscribe_motor_state()` 中的控制周期。

### 调试技巧

1. **启用日志**：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. **检查连接状态**：

```python
print(f"Connected: {robot.is_connected}")
print(f"Calibrated: {robot.is_calibrated}")
```

3. **查看状态数据**：

```python
obs = robot.get_observation()
print(f"Motor states: {len(obs.motor_state)}")
print(f"IMU quaternion: {obs.imu_state.quaternion}")
```

4. **监控温度**：

```python
obs = robot.get_observation()
for i, motor in enumerate(obs.motor_state):
    if motor.temperature > 60:  # 高温警告
        print(f"Motor {i} temperature high: {motor.temperature}°C")
```

### 常见问题

#### 1. 连接失败

- 检查机器人 IP 地址是否正确
- 确认桥接服务器正在运行
- 检查防火墙设置
- 确认端口 6000 和 6001 未被占用

#### 2. 状态更新缓慢

- 检查网络延迟
- 确认服务器状态转发频率设置（`state_period`）
- 检查 CPU 使用率

#### 3. 控制不稳定

- 检查 PID 增益设置
- 确认控制频率设置正确（250Hz）
- 检查网络延迟和丢包

#### 4. 服务器启动失败

- 确认 Unitree SDK 已正确安装
- 检查 DDS 配置
- 确认机器人已上电并处于正确模式
- 查看服务器日志输出

### 性能优化

1. **网络优化**：
   - 使用有线网络连接
   - 减少网络延迟
   - 使用专用网络

2. **控制频率**：
   - 默认 250Hz 通常足够
   - 可根据需求调整 `control_dt`

3. **状态频率**：
   - 服务器默认 ~500Hz
   - 可通过 `state_period` 参数调整

---

## API 参考

### UnitreeG1 类

主要方法：

- `connect(calibrate: bool = True) -> None` - 连接机器人
- `calibrate() -> None` - 校准（G1 已预校准，空操作）
- `configure() -> None` - 配置（空操作）
- `get_observation() -> G1_29_LowState` - 获取观察
- `send_action(action: dict[str, Any]) -> dict[str, Any]` - 发送动作
- `disconnect() -> None` - 断开连接（空操作）
- `get_gravity_orientation(quaternion: list) -> np.ndarray` - 计算重力方向

属性：

- `is_connected: bool` - 连接状态
- `is_calibrated: bool` - 校准状态（总是 True）
- `observation_features: dict` - 观察特征定义
- `action_features: dict` - 动作特征定义

### 数据类

#### `MotorState`

- `q: float` - 位置（弧度）
- `dq: float` - 速度（弧度/秒）
- `tau_est: float` - 估计扭矩（Nm）
- `temperature: float` - 温度（°C）

#### `IMUState`

- `quaternion: list[float]` - 四元数 [w, x, y, z]
- `gyroscope: list[float]` - 角速度 [x, y, z] (rad/s)
- `accelerometer: list[float]` - 线性加速度 [x, y, z] (m/s²)
- `rpy: list[float]` - 欧拉角 [roll, pitch, yaw] (rad)
- `temperature: float` - IMU 温度（°C）

#### `G1_29_LowState`

- `motor_state: list[MotorState]` - 35 个电机状态
- `imu_state: IMUState` - IMU 状态
- `wireless_remote: Any` - 无线遥控数据
- `mode_machine: int` - 机器人模式

---

## 技术细节

### 通信协议

- **DDS**：Unitree SDK2 Data Distribution Service
- **ZMQ**：ZeroMQ (PULL-PUB 模式)
- **序列化**：JSON（安全，非 pickle）
- **编码**：Base64（用于二进制数据，如无线遥控）

### 控制参数

- **控制频率**：250Hz（默认）
- **状态频率**：~500Hz（服务器端）
- **PID 模式**：位置控制（mode=1）
- **关节数量**：35 个

### 坐标系统

- **关节位置**：弧度（rad）
- **关节速度**：弧度/秒（rad/s）
- **扭矩**：牛顿米（Nm）
- **IMU 角速度**：弧度/秒（rad/s）
- **IMU 加速度**：米/秒²（m/s²）

---

## 示例代码

### 基本控制循环

```python
from lerobot.robots.unitree_g1 import UnitreeG1Config, UnitreeG1
from lerobot.robots.unitree_g1.g1_utils import G1_29_JointIndex
import time

config = UnitreeG1Config(robot_ip="192.168.123.164")
robot = UnitreeG1(config)

try:
    while True:
        # 获取观察
        obs = robot.get_observation()
        
        # 构建动作（所有关节位置）
        action = {}
        for joint in G1_29_JointIndex:
            if joint.value < 29:  # 只控制实际使用的关节
                # 示例：保持当前位置（或其他控制逻辑）
                current_pos = obs.motor_state[joint.value].q
                action[f"{joint.name}.pos"] = current_pos
        
        # 发送动作
        robot.send_action(action)
        
        # 控制频率
        time.sleep(1.0 / 250.0)  # 250Hz

except KeyboardInterrupt:
    print("Stopping...")
finally:
    robot.disconnect()
```

### 读取 IMU 数据

```python
obs = robot.get_observation()

# 四元数
quat = obs.imu_state.quaternion
print(f"Quaternion: w={quat[0]:.3f}, x={quat[1]:.3f}, y={quat[2]:.3f}, z={quat[3]:.3f}")

# 角速度
gyro = obs.imu_state.gyroscope
print(f"Gyroscope: x={gyro[0]:.3f}, y={gyro[1]:.3f}, z={gyro[2]:.3f} rad/s")

# 加速度
accel = obs.imu_state.accelerometer
print(f"Accelerometer: x={accel[0]:.3f}, y={accel[1]:.3f}, z={accel[2]:.3f} m/s²")

# RPY 角度
rpy = obs.imu_state.rpy
print(f"RPY: roll={rpy[0]:.3f}, pitch={rpy[1]:.3f}, yaw={rpy[2]:.3f} rad")

# 计算重力方向
gravity_dir = robot.get_gravity_orientation(quat)
print(f"Gravity direction: {gravity_dir}")
```

---

## 版本历史

- **初始版本**：基于 Unitree SDK2 实现
- **ZMQ 桥接**：添加 DDS 到 ZMQ 的桥接支持
- **JSON 序列化**：使用 JSON 替代 pickle 以提高安全性

---

## 贡献指南

1. 遵循代码风格和注释规范
2. 添加适当的文档字符串
3. 更新本 README 文档
4. 测试新功能
5. 提交 Pull Request

---

## 许可证

Apache License 2.0

---

*最后更新：基于当前代码库结构生成*

