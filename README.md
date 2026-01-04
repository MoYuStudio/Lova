## Lova

Lova相比原始的 lerobot 及 lerobot_alohamini 优化了配置文档及软硬件改动，并修复了部分问题

lerobot_alohamini调试命令
[调试命令摘要](examples/debug/README.md)

## 快速开始（Ubuntu 系统）

*** 强烈建议按照顺序进行 ***

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

默认情况下，串口无法访问。我们需要授权端口。lerobot 官方文档示例将串口权限修改为 666，但在实际使用中，每次电脑重启后都需要重新设置，非常麻烦。建议直接将当前用户添加到设备用户组，这是永久解决方案。

1. 在终端输入 `whoami`  // 查看当前用户名
2. 输入 `sudo usermod -a -G dialout username` // 将用户名永久添加到设备用户组
3. 重启电脑以使权限生效

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
