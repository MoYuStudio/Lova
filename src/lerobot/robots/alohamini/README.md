# AlohaMini 机器人模块开发文档

本文档为开发者提供了 AlohaMini 机器人模块的详细说明，包括架构、组件和使用方法。

## 目录

- [概述](#概述)
- [模块架构](#模块架构)
- [文件说明](#文件说明)
- [核心组件](#核心组件)
- [配置说明](#配置说明)
- [使用方法](#使用方法)
- [开发指南](#开发指南)

---

## 概述

AlohaMini 是一个基于 LeKiwi 扩展的轮式双臂机器人系统。该模块实现了机器人的主机端（Host）和客户端（Client）架构，支持远程控制和数据采集。

### 主要特性

- **双臂系统**：左臂和右臂独立控制
- **移动底盘**：三轮全向移动底盘（omniwheel）
- **升降轴**：独立的 Z 轴升降控制系统
- **网络通信**：基于 ZMQ 的远程通信
- **相机支持**：可配置多相机系统

---

## 模块架构

```
alohamini/
├── __init__.py              # 模块导出
├── config_lekiwi.py         # 配置类定义
├── lekiwi.py                # 主机端机器人实现（LeKiwi）
├── lekiwi_client.py         # 客户端机器人实现（LeKiwiClient）
├── lekiwi_host.py           # 主机服务程序
└── lift_axis.py             # 升降轴控制模块
```

### 架构模式

系统采用 **主机-客户端（Host-Client）** 架构：

- **主机端（Host）**：运行在机器人本体上（如树莓派），直接控制硬件
- **客户端（Client）**：运行在控制计算机上（如笔记本电脑），通过 ZMQ 与主机通信

```
┌─────────────────┐         ZMQ          ┌─────────────────┐
│                 │  ────────────────────>│                 │
│  客户端 (PC)    │   命令 (Command)      │  主机 (Raspberry│
│                 │<────────────────────  │        Pi)      │
│  LeKiwiClient   │   观察 (Observation)  │   LeKiwiHost    │
└─────────────────┘                       └─────────────────┘
```

---

## 文件说明

### `__init__.py`

模块初始化文件，导出主要的类和配置：

- `LeKiwiConfig` - 主机端配置
- `LeKiwiClientConfig` - 客户端配置
- `LeKiwi` - 主机端机器人类
- `LeKiwiClient` - 客户端机器人类

### `config_lekiwi.py`

配置类定义文件，包含：

#### `LeKiwiConfig`

主机端机器人配置：

- `left_port` / `right_port` - 左/右机械臂串口路径
- `disable_torque_on_disconnect` - 断开连接时是否禁用扭矩
- `max_relative_target` - 相对目标位置的最大限制（安全）
- `cameras` - 相机配置字典
- `use_degrees` - 是否使用角度单位（向后兼容）

#### `LeKiwiHostConfig`

主机服务配置：

- `port_zmq_cmd` - ZMQ 命令端口（默认 5555）
- `port_zmq_observations` - ZMQ 观察端口（默认 5556）
- `connection_time_s` - 连接持续时间（秒）
- `watchdog_timeout_ms` - 看门狗超时时间（毫秒）
- `max_loop_freq_hz` - 最大循环频率（Hz）

#### `LeKiwiClientConfig`

客户端配置：

- `remote_ip` - 远程主机 IP 地址
- `port_zmq_cmd` / `port_zmq_observations` - ZMQ 端口
- `teleop_keys` - 键盘遥操作按键映射
- `polling_timeout_ms` - 轮询超时时间
- `connect_timeout_s` - 连接超时时间

### `lekiwi.py`

主机端机器人实现类 `LeKiwi`，继承自 `Robot` 基类。

#### 主要功能

1. **硬件控制**
   - 左/右机械臂控制（Feetech 伺服电机）
   - 三轮全向移动底盘控制
   - 升降轴控制

2. **状态获取**
   - 关节位置读取
   - 底盘速度读取
   - 相机图像捕获
   - 电流监控和过流保护

3. **动作执行**
   - 关节位置控制
   - 底盘速度控制
   - 升降轴高度控制

4. **校准功能**
   - 双臂校准流程
   - 运动范围记录
   - 校准数据保存

#### 关键方法

- `connect()` - 连接机器人硬件
- `calibrate()` - 执行校准流程
- `configure()` - 配置电机参数
- `get_observation()` - 获取当前状态观察
- `send_action()` - 发送动作命令
- `stop_base()` - 停止底盘移动
- `disconnect()` - 断开连接

#### 运动学计算

- `_body_to_wheel_raw()` - 机体坐标系速度 → 轮子原始命令
- `_wheel_raw_to_body()` - 轮子原始命令 → 机体坐标系速度
- `_degps_to_raw()` / `_raw_to_degps()` - 角度/秒 ↔ 原始值转换

### `lekiwi_client.py`

客户端机器人实现类 `LeKiwiClient`，继承自 `Robot` 基类。

#### 主要功能

1. **网络通信**
   - ZMQ 套接字管理
   - 命令发送（PUSH）
   - 观察接收（PULL）

2. **数据处理**
   - JSON 消息解析
   - Base64 图像解码
   - 状态向量构建

3. **键盘遥操作**
   - 速度级别控制
   - 底盘移动控制
   - 升降轴控制

#### 关键方法

- `connect()` - 连接到远程主机
- `get_observation()` - 获取远程观察数据
- `send_action()` - 发送动作到远程主机
- `disconnect()` - 断开连接

### `lekiwi_host.py`

主机服务程序，提供主循环逻辑。

#### 主要功能

1. **服务初始化**
   - 创建 ZMQ 套接字
   - 绑定端口
   - 连接机器人硬件

2. **主循环**
   - 接收客户端命令
   - 执行机器人动作
   - 获取观察数据
   - 发送观察到客户端
   - 看门狗监控

3. **安全机制**
   - 超时停止底盘
   - 异常处理
   - 优雅关闭

### `lift_axis.py`

升降轴控制模块，实现 Z 轴高度控制。

#### 主要特性

- **多转跟踪**：使用扩展角度计数器跟踪多圈旋转
- **闭环控制**：基于位置误差的 P 控制器
- **归零功能**：自动检测硬限位并设置零位
- **安全限制**：软限位保护

#### `LiftAxisConfig`

配置参数：

- `bus` - 使用哪个总线（"left" 或 "right"）
- `motor_id` - 电机 ID
- `lead_mm_per_rev` - 丝杠螺距（mm/转）
- `soft_min_mm` / `soft_max_mm` - 软限位范围
- `home_down_speed` - 归零下降速度
- `home_stall_current_ma` - 归零堵转电流阈值
- `kp_vel` - 速度比例系数
- `v_max` - 最大速度
- `step_mm` - 每次按键步进距离

#### 关键方法

- `home()` - 执行归零操作
- `set_height_target_mm()` - 设置目标高度
- `update()` - 更新控制循环（需高频调用）
- `get_height_mm()` - 获取当前高度
- `contribute_observation()` - 贡献观察数据
- `apply_action()` - 应用动作命令

---

## 核心组件

### 机械臂配置

左臂和右臂各包含 6 个关节：

1. `arm_*_shoulder_pan` - 肩部平移（ID: 1）
2. `arm_*_shoulder_lift` - 肩部升降（ID: 2）
3. `arm_*_elbow_flex` - 肘部弯曲（ID: 3）
4. `arm_*_wrist_flex` - 腕部弯曲（ID: 4）
5. `arm_*_wrist_roll` - 腕部旋转（ID: 5）
6. `arm_*_gripper` - 夹爪（ID: 6）

### 底盘配置

三轮全向底盘（安装在左总线）：

1. `base_left_wheel` - 左轮（ID: 8）
2. `base_back_wheel` - 后轮（ID: 9）
3. `base_right_wheel` - 右轮（ID: 10）

轮子安装角度：240°, 0°, 120°（相对于机器人坐标系，偏移 -90°）

### 升降轴配置

- `lift_axis` - 升降轴（ID: 11，安装在左总线）

---

## 配置说明

### 主机端配置示例

```python
from lerobot.robots.alohamini import LeKiwiConfig, LeKiwi

config = LeKiwiConfig(
    left_port="/dev/am_arm_follower_left",
    right_port="/dev/am_arm_follower_right",
    disable_torque_on_disconnect=True,
    max_relative_target=100,  # 安全限制
    use_degrees=False,
)

robot = LeKiwi(config)
robot.connect()
```

### 客户端配置示例

```python
from lerobot.robots.alohamini import LeKiwiClientConfig, LeKiwiClient

config = LeKiwiClientConfig(
    remote_ip="192.168.1.100",  # 主机 IP
    port_zmq_cmd=5555,
    port_zmq_observations=5556,
)

client = LeKiwiClient(config)
client.connect()
```

### 相机配置

相机配置通过 `cameras` 字典设置：

```python
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.configs import Cv2Rotation

config = LeKiwiConfig(
    cameras={
        "head_camera": OpenCVCameraConfig(
            index_or_path="/dev/video0",
            fps=30,
            width=640,
            height=480,
            rotation=Cv2Rotation.NO_ROTATION
        ),
    }
)
```

---

## 使用方法

### 主机端运行

在机器人本体（如树莓派）上运行：

```bash
python -m lerobot.robots.alohamini.lekiwi_host
```

或者使用配置：

```python
from lerobot.robots.alohamini.lekiwi_host import main
main()
```

### 客户端使用

在控制计算机上使用：

```python
from lerobot.robots.alohamini import LeKiwiClientConfig, LeKiwiClient

# 创建配置
config = LeKiwiClientConfig(remote_ip="192.168.1.100")

# 创建客户端
client = LeKiwiClient(config)

# 连接
client.connect()

# 获取观察
obs = client.get_observation()

# 发送动作
action = {
    "arm_left_shoulder_pan.pos": 0.0,
    "arm_left_shoulder_lift.pos": 0.0,
    # ... 其他关节
    "x.vel": 0.1,  # 前进速度
    "y.vel": 0.0,
    "theta.vel": 0.0,
    "lift_axis.height_mm": 100.0,  # 升降轴高度
}
client.send_action(action)

# 断开连接
client.disconnect()
```

### 校准流程

首次使用或更换硬件后需要校准：

```python
robot = LeKiwi(config)
robot.connect(calibrate=True)  # 会自动提示校准
```

校准步骤：

1. 将左臂移动到中间位置
2. 记录左臂关节的运动范围
3. 将右臂移动到中间位置（如果存在）
4. 记录右臂关节的运动范围
5. 校准数据自动保存

---

## 开发指南

### 添加新功能

#### 1. 添加新电机

在 `lekiwi.py` 的 `__init__` 方法中添加：

```python
self.left_bus = FeetechMotorsBus(
    port=self.config.left_port,
    motors={
        # ... 现有电机
        "new_motor": Motor(12, "sts3215", MotorNormMode.RANGE_M100_100),
    },
    calibration=self.calibration,
)
```

#### 2. 添加新观察

在 `_state_ft` 属性中添加：

```python
@property
def _state_ft(self) -> dict[str, type]:
    return dict.fromkeys(
        (
            # ... 现有观察
            "new_motor.pos",
        ),
        float,
    )
```

#### 3. 修改运动学

如需修改底盘运动学，编辑 `_body_to_wheel_raw()` 和 `_wheel_raw_to_body()` 方法。

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

3. **监控电流**：

电流监控在 `get_observation()` 中自动执行，过流保护会自动触发。

4. **查看观察数据**：

```python
obs = robot.get_observation()
print(obs.keys())  # 查看所有观察键
print(obs["arm_left_shoulder_pan.pos"])  # 查看特定观察
```

### 常见问题

#### 1. 串口权限错误

确保用户已添加到 `dialout` 组：

```bash
sudo usermod -a -G dialout $USER
# 然后重启或重新登录
```

#### 2. ZMQ 连接失败

- 检查防火墙设置
- 确认主机 IP 地址正确
- 检查端口是否被占用

#### 3. 电机不响应

- 检查串口连接
- 确认电机 ID 配置正确
- 检查校准文件是否存在

#### 4. 升降轴不工作

- 检查 `lift_axis` 是否在正确的总线上
- 确认电机 ID 为 11
- 检查 `LiftAxisConfig` 配置

### 性能优化

1. **循环频率**：调整 `max_loop_freq_hz` 以平衡性能和平滑度
2. **图像质量**：降低 JPEG 质量以减少网络带宽
3. **看门狗超时**：根据实际需求调整超时时间

---

## API 参考

### LeKiwi 类

主要方法：

- `connect(calibrate: bool = True) -> None` - 连接机器人
- `calibrate() -> None` - 执行校准
- `configure() -> None` - 配置电机
- `get_observation() -> dict[str, Any]` - 获取观察
- `send_action(action: dict[str, Any]) -> dict[str, Any]` - 发送动作
- `stop_base() -> None` - 停止底盘
- `disconnect() -> None` - 断开连接

属性：

- `is_connected: bool` - 连接状态
- `is_calibrated: bool` - 校准状态
- `observation_features: dict` - 观察特征定义
- `action_features: dict` - 动作特征定义

### LeKiwiClient 类

主要方法：

- `connect() -> None` - 连接远程主机
- `get_observation() -> dict[str, Any]` - 获取观察
- `send_action(action: dict[str, Any]) -> dict[str, Any]` - 发送动作
- `disconnect() -> None` - 断开连接

属性：

- `is_connected: bool` - 连接状态
- `observation_features: dict` - 观察特征定义
- `action_features: dict` - 动作特征定义

### LiftAxis 类

主要方法：

- `home(use_current: bool = True) -> None` - 归零
- `set_height_target_mm(height_mm: float) -> None` - 设置目标高度
- `update() -> None` - 更新控制循环
- `get_height_mm() -> float` - 获取当前高度
- `contribute_observation(obs: dict) -> None` - 贡献观察
- `apply_action(action: dict) -> None` - 应用动作

---

## 示例代码

### 完整使用示例

```python
from lerobot.robots.alohamini import LeKiwiConfig, LeKiwi

# 配置
config = LeKiwiConfig(
    left_port="/dev/ttyACM0",
    right_port="/dev/ttyACM1",
)

# 创建机器人
robot = LeKiwi(config)

# 连接（首次会提示校准）
robot.connect()

try:
    # 获取观察
    obs = robot.get_observation()
    print(f"Left shoulder position: {obs['arm_left_shoulder_pan.pos']}")
    print(f"Base velocity: x={obs['x.vel']}, y={obs['y.vel']}, theta={obs['theta.vel']}")
    print(f"Lift height: {obs['lift_axis.height_mm']} mm")
    
    # 发送动作
    action = {
        "arm_left_shoulder_pan.pos": 0.5,
        "arm_left_shoulder_lift.pos": 0.3,
        "arm_left_elbow_flex.pos": -0.2,
        "arm_left_wrist_flex.pos": 0.0,
        "arm_left_wrist_roll.pos": 0.0,
        "arm_left_gripper.pos": 50.0,
        "arm_right_shoulder_pan.pos": -0.5,
        "arm_right_shoulder_lift.pos": 0.3,
        "arm_right_elbow_flex.pos": -0.2,
        "arm_right_wrist_flex.pos": 0.0,
        "arm_right_wrist_roll.pos": 0.0,
        "arm_right_gripper.pos": 50.0,
        "x.vel": 0.1,
        "y.vel": 0.0,
        "theta.vel": 0.0,
        "lift_axis.height_mm": 150.0,
    }
    robot.send_action(action)
    
finally:
    # 断开连接
    robot.disconnect()
```

---

## 技术细节

### 通信协议

- **协议**：ZMQ (ZeroMQ)
- **命令通道**：PUSH-PULL（客户端 → 主机）
- **观察通道**：PUSH-PULL（主机 → 客户端）
- **数据格式**：JSON
- **图像编码**：Base64 JPEG

### 电机规格

- **型号**：STS3215（Feetech）
- **控制模式**：位置模式（机械臂）、速度模式（底盘、升降轴）
- **分辨率**：4096 刻度/转
- **通信协议**：RS485

### 坐标系统

- **关节空间**：归一化到 [-100, 100] 或角度（度）
- **机体坐标系**：x（前进）、y（横向）、theta（旋转，度/秒）
- **升降轴**：毫米（mm）

---

## 版本历史

- **初始版本**：基于 LeKiwi 扩展实现
- **升降轴支持**：添加 Z 轴升降控制
- **双臂支持**：添加右臂控制
- **网络通信**：实现主机-客户端架构

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

