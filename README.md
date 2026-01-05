![Sign](media/lova_sign.png)

# Lova

Lova相比原始的 lerobot 及 lerobot_alohamini 优化了配置文档及软硬件改动，并修复了部分问题

lerobot_alohamini调试命令
[调试命令摘要](examples/debug/README.md)

## 构建

### Lerobot 构建

### AlohaMini 构建

开始构建和运行 AlohaMini：

1. **硬件采购 ** ——购买组件和3D打印部件
   参见 **[物料清单和3D打印 ](https://github.com/liyiteng/AlohaMini/blob/main/docs/BOM.md)**
2. **组装 ** ——大约60分钟即可组装完成（SO-ARM预组装）
   请参阅 **[组装指南 ](https://github.com/liyiteng/AlohaMini/blob/main/docs/hardware_assembly.md)**
3. **软件设置和远程操作 ** ——安装、连接和控制机器人
   请参阅 **[软件指南 ](https://github.com/liyiteng/AlohaMini/blob/main/docs/software_setup.md)**

### Lova 构建

Lova 硬件改动部分

TODO：

## 配置

系统：Ubuntu 24.04

### Linux 基础操作

终端指令

#### CD

选一复制

```
cd ~/lerobot
cd ~/lerobot_alohamini
```

#### 进入conda的lerobot空间

选一复制

```
conda activate lerobot
conda activate lerobot_alohamini
conda activate aloha
```

#### 寻找USB端口

```
lerobot-find-port
```

#### 串口给予权限（单次）

```
sudo chmod 666 /dev/ttyACM*
```

#### 串口给予权限（多次）

直接将当前用户添加到设备用户组，这是永久解决方案

1. 查看当前用户名
   `whoami`
2. 将用户名永久添加到设备用户组
   `sudo usermod -a -G dialout username`
3. 重启电脑以使权限生效

#### 机械臂电机标号（终端）

从6-1,从头到根

##### 主臂
```
lerobot-setup-motors 
    --teleop.type=so101_leader
    --teleop.port=/dev/ttyACM0
```

##### 从臂
```
lerobot-setup-motors 
    --robot.type=so101_follower
    --robot.port=/dev/ttyACM0
```

#### 机械臂电机标号（GUI软件）

#### 机械臂校准

如遇到ValueError: Magnitude 2753 exceeds 2047 (max for sign_bit_index=11)等
断电源即可修复

##### 主臂
```
lerobot-calibrate 
    --teleop.type=so101_leader
    --teleop.port=/dev/ttyACM0
    --teleop.id=moyu_leader_arm_1
```

##### 从臂
```
lerobot-calibrate 
    --robot.type=so101_follower
    --robot.port=/dev/ttyACM0
    --robot.id=moyu_follower_arm_1
```

#### 运行跟随遥操作
ttyACM*根据插入顺序标记
```
lerobot-teleoperate 
    --teleop.type=so101_leader
    --teleop.port=/dev/ttyACM0
    --teleop.id=moyu_leader_arm_0
    --robot.type=so101_follower
    --robot.port=/dev/ttyACM1
    --robot.id=moyu_follower_arm_0
```

```
lerobot-teleoperate 
    --teleop.type=so101_leader
    --teleop.port=/dev/ttyACM2
    --teleop.id=moyu_leader_arm_1
    --robot.type=so101_follower
    --robot.port=/dev/ttyACM3
    --robot.id=moyu_follower_arm_1
```

### 上位机 配置流程
Linux Ubuntu PC

主要参考：
https://github.com/liyiteng/lerobot_alohamini/tree/main

#### 分别校准左右Leader机械臂（如有需要，第一次遥操会强制校准）

##### 左Leader机械臂
```
lerobot-calibrate \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=left_leader
```

##### 右Leader机械臂
```
lerobot-calibrate \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=right_leader
```

#### PC遥操
```
// 普通遥操作 Normal teleoperation

python examples/alohamini/teleoperate_bi.py \
--remote_ip 192.168.0.13 \
--leader_id so101_leader_bi


// 带语音功能的遥操作 Teleoperation with voice functionality
python examples/alohamini/teleoperate_bi_voice.py \
--remote_ip 192.168.50.43 \
--leader_id so101_leader_bi

注：使用语音功能需安装依赖并设置 DASHSCOPE_API_KEY
Note: Voice functionality requires installing dependencies and setting DASHSCOPE_API_KEY

// 安装语音依赖 Install voice dependencies
conda install -c conda-forge python-sounddevice
pip install dashscope


// 前往阿里云百炼网站申请语音识别 API，然后执行以下命令将 API 加入环境变量 Go to Alibaba Cloud Bailian website, apply for speech recognition API, execute the following command to add the API to environment variables

export DASHSCOPE_API_KEY="sk-434f820ebaxxxxxxxxx"
```

##### 仅遥控
```
python examples/alohamini/teleoperate_bi.py --remote_ip {ip}
```


### 下位机 配置流程

Linux Ubuntu 树莓派

与PC侧环境安装存在巨大差异

#### 主流程

```
cd ~
git clone https://github.com/liyiteng/lerobot_alohamini.git
```

# 1. 建目录

```
mkdir -p ~/miniforge3
```

# 2. 下载 ARM 版安装脚本（以 Python 3.11 为例）
```
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh \
     -O ~/miniforge3/miniforge.sh
```

# 3. 静默安装
```
bash ~/miniforge3/miniforge.sh -b -u -p ~/miniforge3
```

# 4. 清理安装包
```
rm ~/miniforge3/miniforge.sh
```

# 5. 初始化 bash（或 zsh，看你自己用啥）
```
~/miniforge3/bin/conda init bash
source ~/.bashrc          # 重新加载环境变量

conda create -y -n lerobot_alohamini python=3.10
conda activate lerobot_alohamini

cd ~/lerobot_alohamini
pip install -e .[all]
conda install ffmpeg=7.1.1 -c conda-forge

cd ~/lerobot_alohamini      # 你的仓库根目录
```

追随者手臂:lerobot/robots/alohamini/config_lekiwi.py
ls /dev/ttyACM*

将SSH插入树莓派
在 Ubuntu 上打开 SSH 只需两步：
1. 安装服务器端
sudo apt update
sudo apt install openssh-server
 
2. 启动并设为开机自启
sudo systemctl enable --now ssh
 
3. 确认监听成功
sudo systemctl status ssh      # 看到 active (running) 即可
ss -ltn | grep :22             # 确认 22 端口已监听
 
##### PC端SSH连接
ssh robot@<树莓派IP>

ssh robot@192.168.31.141
ssh robot@192.168.8.186
 
（如果之前没改过密码，默认用户名 robot 就是你当前登录账号。）
防火墙若启用，再放行 22 端口：
 
sudo ufw allow 22/tcp
至此 Ubuntu 的 SSH 就“插”好了。

获取ip
ip -br -4 addr show

树莓派遥操
python -m lerobot.robots.alohamini.lekiwi_host

### 双端配置
摄像头连接
！！！双端都需要！！！
配置摄像头端口号 
lerobot/robots/alohamini/config_lekiwi.py 
```
def lekiwi_cameras_config() -> dict[str, CameraConfig]:
    return {
        "head_top": OpenCVCameraConfig(
            index_or_path="/dev/video0", fps=30, width=640, height=480, rotation=Cv2Rotation.NO_ROTATION
        ),
        "head_back": OpenCVCameraConfig(
            index_or_path="/dev/video2", fps=30, width=640, height=480, rotation=Cv2Rotation.NO_ROTATION
        ),
        "head_front": OpenCVCameraConfig(
            index_or_path="/dev/video4", fps=30, width=640, height=480, rotation=Cv2Rotation.NO_ROTATION
        ),
        # "wrist_left": OpenCVCameraConfig(
        #     index_or_path="/dev/am_camera_wrist_left", fps=30, width=640, height=480, rotation=Cv2Rotation.NO_ROTATION
        # ),
        # "wrist_right": OpenCVCameraConfig(
        #     index_or_path="/dev/am_camera_wrist_right", fps=30, width=640, height=480, rotation=Cv2Rotation.NO_ROTATION
        # ),
    }
```

##### 检查设备节点
插上摄像头后执行

为了实例化摄像头，您需要一个摄像头标识符。这个标识符可能会在您重启电脑或重新插拔摄像头时发生变化，这主要取决于您的操作系统。
要查找连接到您系统的摄像头的摄像头索引，请运行以下脚本：
lerobot-find-cameras opencv # or realsense for Intel Realsense cameras
终端会打印相关摄像头信息。
```
--- Detected Cameras ---
Camera #0:
  Name: OpenCV Camera @ 0
  Type: OpenCV
  Id: 0
  Backend api: AVFOUNDATION
  Default stream profile:
    Format: 16.0
    Width: 1920
    Height: 1080
    Fps: 15.0
--------------------
(more cameras ...)
```
您可以在 ~/lerobot/outputs/captured_images 目录中找到每台摄像头拍摄的图片。

### 主要问题修复
环境安装部分
# 1. 进仓库根目录
```
cd ~/lerobot_alohamini
```

# 2. 一次性的把 src/ 加入搜索路径
```
export PYTHONPATH=$HOME/lerobot_alohamini/src:$PYTHONPATH
```

# 3. 直接当模块启动（或者当脚本启动都可以）
```
python -m lerobot.robots.alohamini.lekiwi_host
```

pip install -e .

sudo apt update && sudo apt upgrade -y

sudo apt update
sudo apt install linux-headers-raspi

conda install conda-forge::evdev

pip install -e .

pip install pyzmq

在当前 PC 环境装 Feetech 官方 SDK（已打包成 pip）
```
pip install feetech-servo-sdk
```
检查
```
python -c "import scservo_sdk; print('OK')"
python examples/alohamini/teleoperate_bi.py --remote_ip {ip}
```

##### 相机连接配置相关


warning
在 macOS 中使用 Intel RealSense 摄像头时，您可能会遇到 “Error finding RealSense cameras: failed to set power state” 的错误。这可以通过使用 sudo 权限运行相同的命令来解决。请注意，在 macOS 中使用 RealSense 摄像头是不稳定的。
之后，您就可以在遥控操作时在电脑上显示摄像头画面了，只需运行以下代码即可。这对于在录制第一个数据集之前准备您的设置非常有用。
```
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true
```
tip
fourcc: "MJPG"格式图像是经过压缩后的图像，你可以尝试更高分辨率，当然你可以尝试YUYV格式图像，但是这会导致图像的分辨率和FPS降低导致机械臂运行卡顿。目前MJPG格式下可支持3个摄像头1920*1080分辨率并且保持30FPS, 但是依然不推荐2个摄像头通过同一个USB HUB接入电脑
如果您有更多摄像头，可以通过更改 --robot.cameras 参数来添加。您应该注意index_or_path 的格式，它由 python -m lerobot.find_cameras opencv 命令输出的摄像头 ID 的最后一位数字决定。
例如，如果你想添加摄像头:
```
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true
```
tip
fourcc: "MJPG"格式图像是经过压缩后的图像，你可以尝试更高分辨率，当然你可以尝试YUYV格式图像，但是这会导致图像的分辨率和FPS降低导致机械臂运行卡顿。目前MJPG格式下可支持3个摄像头1920*1080分辨率并且保持30FPS, 但是依然不推荐2个摄像头通过同一个USB HUB接入电脑

树莓派相机连接自查
在树莓派执行：
# 1) 相机是否就绪
lerobot-find-cameras opencv | grep -E "Name:|Fps: 30"# 必须能看到 /dev/video0、/dev/video2、/dev/video4 三路 Fps: 30.0（或 15.0）# 2) host 是否真正启动
python -m lerobot.robots.alohamini.lekiwi_host
 
robot = LeKiwi(LeKiwiConfig())
print(">>> cameras in robot:", list(robot.cameras.keys()) if hasattr(robot, 'cameras') else "NO CAMERAS")

# 原始AlohaMini教程

### 1. 准备工作

#### 网络环境测试

```
curl https://www.google.com
curl https://huggingface.co
```

首先确保网络连接正常

#### CUDA 环境测试

```
nvidia-smi
```

在终端输入后，应该能够看到 CUDA 版本号

### 2. 克隆 lerobot_alohamini 仓库

```
cd ~
git clone https://github.com/liyiteng/lerobot_alohamini.git
```

### 3. 串口授权

### 4. 安装 conda3 和环境依赖

安装 conda3

```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

初始化 conda3

```
conda create -y -n lerobot_alohamini python=3.10
conda activate lerobot_alohamini
```

安装环境依赖

```
cd ~/lerobot_alohamini
pip install -e .[all]
conda install ffmpeg=7.1.1 -c conda-forge
```

### 5. 配置机械臂端口号

AlohaMini 总共有 4 个机械臂：2 个主控臂连接到 PC，2 个从控臂连接到树莓派，共 4 个端口。

由于每次重新连接后端口号都会变化，你必须掌握查找端口号的操作。熟练后，可以使用硬链接来固定端口。

如果你购买了完整的 AlohaMini 机器，随附的树莓派已经固定了 2 个从控臂的端口号，因此不需要额外配置。

将机械臂连接到电源并通过 USB 连接到电脑，然后查找机械臂端口号。

方法一：
通过脚本查找端口：

```
cd ~/lerobot_alohamini

lerobot-find-port
```

方法二：
你可以直接在终端输入命令，通过观察每次插入后显示的不同端口号来确认插入的端口号

```
ls /dev/ttyACM*
```

**找到正确的端口后，请修改以下文件中的相应端口号：
从控臂：lerobot/robots/alohamini/config_lekiwi.py
主控臂：examples/alohamini/teleoperate_bi.py**

注意：每次重新连接机械臂或重启电脑后都必须执行此操作

### 6. 配置相机端口号

相机端口已内置到树莓派中，无需配置：
lerobot/robots/alohamini/config_lekiwi.py

注意：

- 多个相机不能插入一个 USB Hub；1 个 USB Hub 仅支持 1 个相机

### 7. 遥操作校准和测试

#### 7.1 设置机械臂到中间位置

主机端校准：
SSH 到树莓派，安装 conda 环境，然后执行以下操作：

```
python -m lerobot.robots.alohamini.lekiwi_host
```

如果是首次执行，系统会提示我们校准机械臂。按照图片所示位置放置机械臂，按 Enter，然后将每个关节向左旋转 90 度，再向右旋转 90 度，然后按 Enter
![Calibration](examples/alohamini/media/mid_position_so100.png)

客户端校准：
执行以下命令，将 IP 替换为主机树莓派的实际 IP，然后重复上述步骤

```
python examples/alohamini/teleoperate_bi.py \
--remote_ip 192.168.50.43 \
--leader_id so101_leader_bi

```

#### 7.2 遥操作命令摘要

树莓派端：

```
python -m lerobot.robots.alohamini.lekiwi_host
```

PC 端：

```
// 普通遥操作

python examples/alohamini/teleoperate_bi.py \
--remote_ip 192.168.50.43 \
--leader_id so101_leader_bi


// 带语音功能的遥操作
python examples/alohamini/teleoperate_bi_voice.py \
--remote_ip 192.168.50.43 \
--leader_id so101_leader_bi


注意：语音功能需要安装依赖并设置 DASHSCOPE_API_KEY

// 安装语音依赖
conda install -c conda-forge python-sounddevice
pip install dashscope


// 前往阿里云百炼网站，申请语音识别 API，执行以下命令将 API 添加到环境变量

export DASHSCOPE_API_KEY="sk-434f820ebaxxxxxxxxx"
```

### 8. 录制数据集

#### 1 在 HuggingFace 注册，获取并配置密钥

1. 前往 HuggingFace 网站 (huggingface.co)，申请 {Key}，记住要包含读写权限
2. 将 API token 添加到 Git 凭据

```
git config --global credential.helper store

huggingface-cli login --token {key} --add-to-git-credential

```

#### 2 运行脚本

修改 repo-id 参数，然后执行：

```
HF_USER=$(huggingface-cli whoami | head -n 1)
echo $HF_USER

```

//创建新数据集

```
python examples/alohamini/record_bi.py \
  --dataset $HF_USER/so100_bi_test \
  --num_episodes 1 \
  --fps 30 \
  --episode_time 45 \
  --reset_time 8 \
  --task_description "pickup1" \
  --remote_ip 127.0.0.1 \
  --leader_id so101_leader_bi

```

//恢复数据集

```
python examples/alohamini/record_bi.py \
  --dataset $HF_USER/so100_bi_test \
  --num_episodes 1 \
  --fps 30 \
  --episode_time 45 \
  --reset_time 8 \
  --task_description "pickup1" \
  --remote_ip 127.0.0.1 \
  --leader_id so101_leader_bi \
  --resume 

```

### 9. 回放数据集

```
python examples/alohamini/replay_bi.py  \
--dataset $HF_USER/so100_bi_test \
--episode 0 \
--remote_ip 127.0.0.1
```

### 10. 数据集可视化

```
  lerobot-dataset-viz \
  --repo-id $HF_USER/so100_bi_test \
  --episode-index 0
```

### 11. 本地训练

// ACT

```
lerobot-train \
  --dataset.repo_id=$HF_USER/so100_bi_test \
  --policy.type=act \
  --output_dir=outputs/train/act_your_dataset1 \
  --job_name=act_your_dataset \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.repo_id=liyitenga/act_policy
```

### 12. 远程训练

以 AutoDL 为例：
申请 RTX 4070 GPU，选择 Python 3.8 (Ubuntu 20.04) CUDA 11.8 或更高版本作为容器镜像，通过终端登录

```
// 进入远程终端，初始化 conda
conda init

// 重启终端，创建环境
conda create -y -n lerobot python=3.10
conda activate lerobot

// 学术加速
source /etc/network_turbo

// 获取 lerobot
git clone https://github.com/liyiteng/lerobot_alohamini.git

// 安装必要文件
cd ~/lerobot_alohamini
pip install -e ".[feetech]"
```

运行训练命令

最后安装 FileZilla 来获取训练好的文件

```
sudo apt install filezilla -y
```

### 13. 评估训练集

使用 FileZilla 将训练好的模型复制到本地机器，然后运行以下命令：

```
python examples/alohamini/evaluate_bi.py \
  --num_episodes 3 \
  --fps 20 \
  --episode_time 45 \
  --task_description "Pick and place task" \
  --hf_model_id liyitenga/act_policy \
  --hf_dataset_id liyitenga/eval_dataset \
  --remote_ip 127.0.0.1 \
  --robot_id my_alohamini \
  --hf_model_id ./outputs/train/act_your_dataset1/checkpoints/020000/pretrained_model
  
```
