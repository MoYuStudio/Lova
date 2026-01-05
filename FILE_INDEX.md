# 文件索引

本文档提供了项目的主要目录和文件索引，特别是 `/src` 和 `/examples` 目录的详细结构。

## 目录

- [源代码目录 (`/src`)](#源代码目录-src)
- [示例目录 (`/examples`)](#示例目录-examples)

---

## 源代码目录 (`/src`)

### `/src/lerobot/`

LeRobot 核心库的主要源代码目录。

#### 核心模块

- `__init__.py` - 包初始化文件
- `__version__.py` - 版本信息

#### 异步推理 (`async_inference/`)

- `configs.py` - 异步推理配置
- `constants.py` - 常量定义
- `helpers.py` - 辅助函数
- `policy_server.py` - 策略服务器
- `robot_client.py` - 机器人客户端

#### 相机模块 (`cameras/`)

- `__init__.py`
- `camera.py` - 相机基类
- `configs.py` - 相机配置
- `utils.py` - 工具函数

**子模块：**

- **OpenCV 相机** (`opencv/`)
  - `__init__.py`
  - `camera_opencv.py` - OpenCV 相机实现
  - `configuration_opencv.py` - OpenCV 配置

- **Reachy2 相机** (`reachy2_camera/`)
  - `__init__.py`
  - `reachy2_camera.py` - Reachy2 相机实现
  - `configuration_reachy2_camera.py` - Reachy2 配置

- **RealSense 相机** (`realsense/`)
  - `__init__.py`
  - `camera_realsense.py` - RealSense 相机实现
  - `configuration_realsense.py` - RealSense 配置

#### 配置模块 (`configs/`)

- `default.py` - 默认配置
- `eval.py` - 评估配置
- `parser.py` - 配置解析器
- `policies.py` - 策略配置
- `train.py` - 训练配置
- `types.py` - 类型定义

#### 数据集模块 (`datasets/`)

- `aggregate.py` - 数据集聚合
- `backward_compatibility.py` - 向后兼容性
- `card_template.md` - 数据集卡片模板
- `compute_stats.py` - 计算统计数据
- `dataset_tools.py` - 数据集工具
- `factory.py` - 数据集工厂
- `image_writer.py` - 图像写入器
- `lerobot_dataset.py` - LeRobot 数据集核心类
- `online_buffer.py` - 在线缓冲区
- `pipeline_features.py` - 管道特征
- `sampler.py` - 采样器
- `streaming_dataset.py` - 流式数据集
- `transforms.py` - 数据转换
- `utils.py` - 工具函数
- `video_utils.py` - 视频工具

**子模块：**

- **推送数据集到 Hub** (`push_dataset_to_hub/`)
  - `utils.py` - 工具函数

- **v30 版本** (`v30/`)
  - `augment_dataset_quantile_stats.py` - 增强数据集分位数统计
  - `convert_dataset_v21_to_v30.py` - 数据集版本转换 (v21 -> v30)

#### 环境模块 (`envs/`)

- `__init__.py`
- `configs.py` - 环境配置
- `factory.py` - 环境工厂
- `libero.py` - LIBERO 环境
- `metaworld.py` - MetaWorld 环境
- `metaworld_config.json` - MetaWorld 配置文件
- `utils.py` - 工具函数

#### 模型模块 (`model/`)

- `kinematics.py` - 运动学模型

#### 电机模块 (`motors/`)

- `__init__.py`
- `calibration_gui.py` - 校准图形界面
- `encoding_utils.py` - 编码工具
- `motors_bus.py` - 电机总线

**子模块：**

- **Dynamixel** (`dynamixel/`)
  - `__init__.py`
  - `dynamixel.py` - Dynamixel 电机实现
  - `tables.py` - 数据表

- **Feetech** (`feetech/`)
  - `__init__.py`
  - `feetech.py` - Feetech 电机实现
  - `tables.py` - 数据表

#### 优化器模块 (`optim/`)

- `__init__.py`
- `factory.py` - 优化器工厂
- `optimizers.py` - 优化器实现
- `schedulers.py` - 学习率调度器

#### 策略模块 (`policies/`)

- `__init__.py`
- `factory.py` - 策略工厂
- `pretrained.py` - 预训练模型
- `utils.py` - 工具函数

**策略实现：**

- **ACT** (`act/`)
  - `configuration_act.py` - ACT 配置
  - `modeling_act.py` - ACT 模型
  - `processor_act.py` - ACT 处理器
  - `README.md` - ACT 说明文档

- **Diffusion** (`diffusion/`)
  - `configuration_diffusion.py` - Diffusion 配置
  - `modeling_diffusion.py` - Diffusion 模型
  - `processor_diffusion.py` - Diffusion 处理器
  - `README.md` - Diffusion 说明文档

- **Gr00t** (`groot/`)
  - `__init__.py`
  - `configuration_groot.py` - Gr00t 配置
  - `modeling_groot.py` - Gr00t 模型
  - `processor_groot.py` - Gr00t 处理器
  - `groot_n1.py` - Gr00t N1 实现
  - `utils.py` - 工具函数
  - `README.md` - Gr00t 说明文档
  - **动作头** (`action_head/`)
    - `__init__.py`
    - `action_encoder.py` - 动作编码器
    - `cross_attention_dit.py` - 交叉注意力 DiT
    - `flow_matching_action_head.py` - 流匹配动作头
  - **Eagle2 模型** (`eagle2_hg_model/`)
    - `configuration_eagle2_5_vl.py` - Eagle2.5 VL 配置
    - `image_processing_eagle2_5_vl_fast.py` - 快速图像处理
    - `modeling_eagle2_5_vl.py` - Eagle2.5 VL 模型
    - `processing_eagle2_5_vl.py` - Eagle2.5 VL 处理

- **PI0** (`pi0/`)
  - `__init__.py`
  - `configuration_pi0.py` - PI0 配置
  - `modeling_pi0.py` - PI0 模型
  - `processor_pi0.py` - PI0 处理器
  - `README.md` - PI0 说明文档

- **PI05** (`pi05/`)
  - `__init__.py`
  - `configuration_pi05.py` - PI05 配置
  - `modeling_pi05.py` - PI05 模型
  - `processor_pi05.py` - PI05 处理器
  - `README.md` - PI05 说明文档

- **RTC** (`rtc/`)
  - `action_queue.py` - 动作队列
  - `configuration_rtc.py` - RTC 配置
  - `debug_tracker.py` - 调试追踪器
  - `debug_visualizer.py` - 调试可视化器
  - `latency_tracker.py` - 延迟追踪器
  - `modeling_rtc.py` - RTC 模型
  - `README.md` - RTC 说明文档

- **SAC** (`sac/`)
  - `configuration_sac.py` - SAC 配置
  - `modeling_sac.py` - SAC 模型
  - `processor_sac.py` - SAC 处理器
  - **奖励模型** (`reward_model/`)
    - `configuration_classifier.py` - 分类器配置
    - `modeling_classifier.py` - 分类器模型
    - `processor_classifier.py` - 分类器处理器

- **SmolVLA** (`smolvla/`)
  - `configuration_smolvla.py` - SmolVLA 配置
  - `modeling_smolvla.py` - SmolVLA 模型
  - `processor_smolvla.py` - SmolVLA 处理器
  - `smolvlm_with_expert.py` - 带专家的 SmolVLM
  - `README.md` - SmolVLA 说明文档

- **TDMPC** (`tdmpc/`)
  - `configuration_tdmpc.py` - TDMPC 配置
  - `modeling_tdmpc.py` - TDMPC 模型
  - `processor_tdmpc.py` - TDMPC 处理器
  - `README.md` - TDMPC 说明文档

- **VQBET** (`vqbet/`)
  - `configuration_vqbet.py` - VQBET 配置
  - `modeling_vqbet.py` - VQBET 模型
  - `processor_vqbet.py` - VQBET 处理器
  - `vqbet_utils.py` - VQBET 工具函数
  - `README.md` - VQBET 说明文档

- **XVLA** (`xvla/`)
  - `__init__.py`
  - `action_hub.py` - 动作中心
  - `configuration_florence2.py` - Florence2 配置
  - `configuration_xvla.py` - XVLA 配置
  - `modeling_florence2.py` - Florence2 模型
  - `modeling_xvla.py` - XVLA 模型
  - `processor_xvla.py` - XVLA 处理器
  - `soft_transformer.py` - Soft Transformer
  - `utils.py` - 工具函数

#### 处理器模块 (`processor/`)

- `__init__.py`
- `batch_processor.py` - 批处理器
- `converters.py` - 转换器
- `core.py` - 核心处理器
- `delta_action_processor.py` - 增量动作处理器
- `device_processor.py` - 设备处理器
- `env_processor.py` - 环境处理器
- `factory.py` - 处理器工厂
- `gym_action_processor.py` - Gym 动作处理器
- `hil_processor.py` - HIL 处理器
- `joint_observations_processor.py` - 关节观察处理器
- `migrate_policy_normalization.py` - 迁移策略归一化
- `normalize_processor.py` - 归一化处理器
- `observation_processor.py` - 观察处理器
- `pipeline.py` - 处理管道
- `policy_robot_bridge.py` - 策略-机器人桥接
- `rename_processor.py` - 重命名处理器
- `tokenizer_processor.py` - 分词器处理器

#### 强化学习模块 (`rl/`)

- `actor.py` - 执行器
- `buffer.py` - 缓冲区
- `crop_dataset_roi.py` - 裁剪数据集 ROI
- `eval_policy.py` - 评估策略
- `gym_manipulator.py` - Gym 机械手
- `learner_service.py` - 学习器服务
- `learner.py` - 学习器
- `process.py` - 进程管理
- `queue.py` - 队列
- `wandb_utils.py` - WandB 工具

#### 机器人模块 (`robots/`)

- `__init__.py`
- `config.py` - 机器人配置
- `robot.py` - 机器人基类
- `utils.py` - 工具函数

**机器人实现：**

- **AlohaMini** (`alohamini/`)
  - `__init__.py`
  - `config_lekiwi.py` - Lekiwi 配置
  - `lekiwi.py` - Lekiwi 实现
  - `lekiwi_client.py` - Lekiwi 客户端
  - `lekiwi_host.py` - Lekiwi 主机
  - `lift_axis.py` - 升降轴

- **Bi SO100 Follower** (`bi_so100_follower/`)
  - `__init__.py`
  - `bi_so100_follower.py` - Bi SO100 Follower 实现
  - `config_bi_so100_follower.py` - 配置

- **Earthrover Mini Plus** (`earthrover_mini_plus/`)
  - `__init__.py`
  - `config_earthrover_mini_plus.py` - 配置
  - `robot_earthrover_mini_plus.py` - 机器人实现
  - `*.mdx` - 文档文件

- **Hope Jr** (`hope_jr/`)
  - `__init__.py`
  - `config_hope_jr.py` - 配置
  - `hope_jr_arm.py` - 手臂实现
  - `hope_jr_hand.py` - 手部实现
  - `*.mdx` - 文档文件

- **Koch Follower** (`koch_follower/`)
  - `__init__.py`
  - `config_koch_follower.py` - 配置
  - `koch_follower.py` - Koch Follower 实现
  - `*.mdx` - 文档文件

- **Lekiwi** (`lekiwi/`)
  - `__init__.py`
  - `config_lekiwi.py` - Lekiwi 配置
  - `lekiwi.py` - Lekiwi 实现
  - `lekiwi_client.py` - Lekiwi 客户端
  - `lekiwi_host.py` - Lekiwi 主机
  - `*.mdx` - 文档文件

- **Reachy2** (`reachy2/`)
  - `__init__.py`
  - `configuration_reachy2.py` - Reachy2 配置
  - `robot_reachy2.py` - Reachy2 机器人实现

- **SO100 Follower** (`so100_follower/`)
  - `__init__.py`
  - `config_so100_follower.py` - 配置
  - `robot_kinematic_processor.py` - 机器人运动学处理器
  - `so100_follower.py` - SO100 Follower 实现
  - `*.mdx` - 文档文件

- **SO101 Follower** (`so101_follower/`)
  - `__init__.py`
  - `config_so101_follower.py` - 配置
  - `so101_follower.py` - SO101 Follower 实现
  - `*.mdx` - 文档文件

- **Unitree G1** (`unitree_g1/`)
  - `__init__.py`
  - `config_unitree_g1.py` - Unitree G1 配置
  - `g1_utils.py` - G1 工具函数
  - `run_g1_server.py` - G1 服务器运行
  - `unitree_g1.py` - Unitree G1 实现
  - `unitree_sdk2_socket.py` - Unitree SDK2 套接字

#### 脚本模块 (`scripts/`)

- `lerobot_calibrate.py` - 校准脚本
- `lerobot_dataset_viz.py` - 数据集可视化
- `lerobot_edit_dataset.py` - 编辑数据集
- `lerobot_eval.py` - 评估脚本
- `lerobot_find_cameras.py` - 查找相机
- `lerobot_find_joint_limits.py` - 查找关节限制
- `lerobot_find_port.py` - 查找端口
- `lerobot_imgtransform_viz.py` - 图像变换可视化
- `lerobot_info.py` - 信息脚本
- `lerobot_record.py` - 录制脚本
- `lerobot_replay.py` - 回放脚本
- `lerobot_setup_motors.py` - 设置电机
- `lerobot_teleoperate.py` - 遥操作脚本
- `lerobot_train.py` - 训练脚本

#### 遥操作模块 (`teleoperators/`)

- `__init__.py`
- `config.py` - 遥操作配置
- `teleoperator.py` - 遥操作基类
- `utils.py` - 工具函数

**遥操作实现：**

- **Bi SO100 Leader** (`bi_so100_leader/`)
  - `__init__.py`
  - `bi_so100_leader.py` - Bi SO100 Leader 实现
  - `config_bi_so100_leader.py` - 配置

- **Gamepad** (`gamepad/`)
  - `__init__.py`
  - `configuration_gamepad.py` - 游戏手柄配置
  - `gamepad_utils.py` - 游戏手柄工具
  - `teleop_gamepad.py` - 游戏手柄遥操作

- **Homunculus** (`homunculus/`)
  - `__init__.py`
  - `config_homunculus.py` - Homunculus 配置
  - `homunculus_arm.py` - Homunculus 手臂
  - `homunculus_glove.py` - Homunculus 手套
  - `joints_translation.py` - 关节转换

- **Keyboard** (`keyboard/`)
  - `__init__.py`
  - `configuration_keyboard.py` - 键盘配置
  - `teleop_keyboard.py` - 键盘遥操作

- **Koch Leader** (`koch_leader/`)
  - `__init__.py`
  - `config_koch_leader.py` - 配置
  - `koch_leader.py` - Koch Leader 实现

- **Phone** (`phone/`)
  - `__init__.py`
  - `config_phone.py` - 手机配置
  - `phone_processor.py` - 手机处理器
  - `teleop_phone.py` - 手机遥操作

- **Reachy2 Teleoperator** (`reachy2_teleoperator/`)
  - `__init__.py`
  - `config_reachy2_teleoperator.py` - 配置
  - `reachy2_teleoperator.py` - Reachy2 遥操作实现

- **SO100 Leader** (`so100_leader/`)
  - `__init__.py`
  - `config_so100_leader.py` - 配置
  - `so100_leader.py` - SO100 Leader 实现

- **SO101 Leader** (`so101_leader/`)
  - `__init__.py`
  - `config_so101_leader.py` - 配置
  - `so101_leader.py` - SO101 Leader 实现

#### 模板 (`templates/`)

- `lerobot_modelcard_template.md` - LeRobot 模型卡片模板

#### 传输模块 (`transport/`)

- `services.proto` - gRPC 服务定义（Protocol Buffers）
- `services_pb2.py` - 生成的 Protocol Buffers Python 代码
- `services_pb2_grpc.py` - 生成的 gRPC Python 代码
- `utils.py` - 工具函数

#### 工具模块 (`utils/`)

- `constants.py` - 常量定义
- `control_utils.py` - 控制工具
- `errors.py` - 错误定义
- `hub.py` - HuggingFace Hub 工具
- `import_utils.py` - 导入工具
- `io_utils.py` - I/O 工具
- `logging_utils.py` - 日志工具
- `random_utils.py` - 随机工具
- `robot_utils.py` - 机器人工具
- `rotation.py` - 旋转工具
- `train_utils.py` - 训练工具
- `transition.py` - 转换工具
- `utils.py` - 通用工具
- `visualization_utils.py` - 可视化工具

---

## 示例目录 (`/examples`)

### 机器人特定示例

#### AlohaMini (`alohamini/`)

- `evaluate_bi.py` - 双向评估
- `record_bi.py` - 双向录制
- `replay_bi.py` - 双向回放
- `teleoperate_bi.py` - 双向遥操作
- `teleoperate_bi_voice.py` - 带语音的双向遥操作
- `voice_engine_gummy.py` - Gummy 语音引擎
- `voice_exec.py` - 语音执行
- `media/` - 媒体文件
  - `alohamini3a.png` - AlohaMini 图片
  - `mid_position_so100.png` - SO100 中间位置图片

#### Lekiwi (`lekiwi/`)

- `evaluate.py` - 评估
- `record.py` - 录制
- `replay.py` - 回放
- `teleoperate.py` - 遥操作

#### Phone to SO100 (`phone_to_so100/`)

- `evaluate.py` - 评估
- `record.py` - 录制
- `replay.py` - 回放
- `teleoperate.py` - 遥操作

#### SO100 to SO100 EE (`so100_to_so100_EE/`)

- `evaluate.py` - 评估
- `record.py` - 录制
- `replay.py` - 回放
- `teleoperate.py` - 遥操作

#### Unitree G1 (`unitree_g1/`)

- `gr00t_locomotion.py` - Gr00t 运动控制

### 调试工具 (`debug/`)

- `README.md` - 调试命令说明文档
- `axis.py` - 轴控制
- `motors.py` - 电机控制
- `test_cuda.py` - CUDA 测试
- `test_cv.py` - 计算机视觉测试
- `test_dataset.py` - 数据集测试
- `test_input.py` - 输入测试
- `test_mic.py` - 麦克风测试
- `test_network.py` - 网络测试
- `wheels.py` - 轮子控制
- `action_scripts/` - 动作脚本
  - `go_to_midpoint.txt` - 移动到中点
  - `go_to_restposition.txt` - 移动到休息位置
  - `test_dance.txt` - 测试舞蹈

### 数据集示例 (`dataset/`)

- `load_lerobot_dataset.py` - 加载 LeRobot 数据集
- `use_dataset_image_transforms.py` - 使用数据集图像变换
- `use_dataset_tools.py` - 使用数据集工具

### 训练示例 (`training/`)

- `train_policy.py` - 训练策略
- `train_with_streaming.py` - 流式训练

### 教程示例 (`tutorial/`)

#### ACT (`tutorial/act/`)

- `act_training_example.py` - ACT 训练示例
- `act_using_example.py` - ACT 使用示例

#### 异步推理 (`tutorial/async-inf/`)

- `policy_server.py` - 策略服务器示例
- `robot_client.py` - 机器人客户端示例

#### Diffusion (`tutorial/diffusion/`)

- `diffusion_training_example.py` - Diffusion 训练示例
- `diffusion_using_example.py` - Diffusion 使用示例

#### PI0 (`tutorial/pi0/`)

- `using_pi0_example.py` - PI0 使用示例

#### 强化学习 (`tutorial/rl/`)

- `hilserl_example.py` - HILSERL 示例
- `reward_classifier_example.py` - 奖励分类器示例

#### SmolVLA (`tutorial/smolvla/`)

- `using_smolvla_example.py` - SmolVLA 使用示例

### 其他示例

#### 向后兼容 (`backward_compatibility/`)

- `replay.py` - 回放（向后兼容版本）

#### RTC (`rtc/`)

- `eval_dataset.py` - 评估数据集
- `eval_with_real_robot.py` - 使用真实机器人评估

#### 数据集迁移 (`port_datasets/`)

- `display_error_files.py` - 显示错误文件
- `port_droid.py` - 迁移 Droid 数据集
- `slurm_aggregate_shards.py` - Slurm 聚合分片
- `slurm_port_shards.py` - Slurm 迁移分片
- `slurm_upload.py` - Slurm 上传

---

## 文件统计

### 源代码 (`/src`)

- **总文件数**: 约 266 个 Python 文件
- **主要模块**: 14 个核心模块
- **策略实现**: 10 种不同的策略
- **机器人支持**: 10 种不同的机器人平台
- **遥操作支持**: 9 种不同的遥操作方法

### 示例 (`/examples`)

- **总文件数**: 约 52 个 Python 文件
- **机器人示例**: 5 个不同的机器人平台
- **教程示例**: 6 个不同的主题
- **调试工具**: 10 个调试脚本

---

## 快速导航

### 开始使用

1. **新用户**: 查看 `/examples/tutorial/` 中的教程示例
2. **特定机器人**: 查看 `/examples/` 中对应机器人的示例
3. **调试问题**: 查看 `/examples/debug/` 中的调试工具

### 核心功能

1. **训练策略**: `src/lerobot/scripts/lerobot_train.py`
2. **评估策略**: `src/lerobot/scripts/lerobot_eval.py`
3. **录制数据**: `src/lerobot/scripts/lerobot_record.py`
4. **回放数据**: `src/lerobot/scripts/lerobot_replay.py`
5. **遥操作**: `src/lerobot/scripts/lerobot_teleoperate.py`

### 扩展开发

1. **添加新策略**: 参考 `src/lerobot/policies/` 中的现有实现
2. **添加新机器人**: 参考 `src/lerobot/robots/` 中的现有实现
3. **添加新遥操作**: 参考 `src/lerobot/teleoperators/` 中的现有实现

---

*最后更新: 基于当前代码库结构生成*

