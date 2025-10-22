import inspect
import os
import time

from lerobot.robots.alohamini import LeKiwiClient, LeKiwiClientConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.bi_so100_leader import BiSO100Leader, BiSO100LeaderConfig 


from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data

#from voice_control import VoiceConfig, VoiceEngine
from voice_gummy import VoiceConfig, VoiceEngine


class DummyLeader:
    """在没连上主从臂硬件时的空实现，确保调用安全"""
    def __init__(self, id="dummy_leader"):
        self.id = id
        self.is_connected = False
    def connect(self):
        # 与真实 Leader 接口保持一致
        self.is_connected = False
        return False
    def get_action(self):
        # 不对机械臂下发任何动作
        return {}
    def calibrate(self):  # 可选：与真实类对齐
        pass
    def close(self):
        pass

FPS = 30

def _prefix_arm_action(prefix: str, action_dict: dict) -> dict:
    # 将 leader 输出的键（如 "shoulder_pan.pos"）前缀化为
    # "left_arm_shoulder_pan.pos" 或 "right_arm_shoulder_pan.pos"
    return {f"{prefix}_{k}": v for k, v in action_dict.items()}

# Create the robot and teleoperator configurations
robot_config = LeKiwiClientConfig(remote_ip="127.0.0.1", id="my_lekiwi")   #192.168.50.43
#teleop_arm_config = SO101LeaderConfig(port="/dev/am_arm_leader_left", id="am_arm_sam_leader_left")

# teleop_arm_left_cfg  = SO101LeaderConfig(port="/dev/am_arm_leader_left",  id="am_arm_sam_leader_left")
# teleop_arm_right_cfg = SO101LeaderConfig(port="/dev/am_arm_leader_right", id="am_arm_sam_leader_right")

bi_cfg = BiSO100LeaderConfig(
    left_arm_port="/dev/am_arm_leader_left",
    right_arm_port="/dev/am_arm_leader_right",
    id="so101_leader_bi",
    # 如果你原来给 SO100/SO101 leader 传了 calibration_dir，也可在这里加
    # calibration_dir="path/to/calib",
)
#leader = BiSO100Leader(bi_cfg)
leader = DummyLeader()


keyboard_config = KeyboardTeleopConfig(id="my_laptop_keyboard")

robot = LeKiwiClient(robot_config)
#leader_arm = SO101Leader(teleop_arm_config)

keyboard = KeyboardTeleop(keyboard_config)

# To connect you already should have this script running on LeKiwi: `python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_awesome_kiwi`
robot.connect()
leader.connect()
keyboard.connect()

# leader.left_arm.calibrate()   
# leader.right_arm.calibrate()

_init_rerun(session_name="lekiwi_teleop")

# if not robot.is_connected or not left_leader.is_connected or not keyboard.is_connected:
#     raise ValueError("Robot, leader arm of keyboard is not connected!")

if not robot.is_connected or not leader.is_connected or not keyboard.is_connected:
    print("⚠️ Warning: Some devices are not connected! Still running for debug.")


def set_height_mm(mm: float):
    """命令Z轴上升到指定高度（mm）"""
    action = {"lift_axis.height_mm": float(mm)}
    robot.send_action(action)
    print(f"tb.py Set lift height to {mm} mm")



voice = VoiceEngine(VoiceConfig())
voice.start()

VOICE_Z_EPS = 0.8          # 认为到位的误差阈值（mm）
voice_z_target_mm = None   # 语音设定的粘性 Z 目标（未到位时每帧持续写入）

last_print = 0.0

while True:
    t0 = time.perf_counter()

    observation = robot.get_observation()
    
    #voice.set_height_mm(float(observation.get("lift_axis.height_mm", 0.0)))
    cur_h = float(observation.get("lift_axis.height_mm", 0.0))
    voice.set_height_mm(cur_h)
    voice_act = voice.get_action_nowait()  # dict 或 {}

    now = time.monotonic()
    if now - last_print >= 1.0:
        print(f"lift_axis.height_mm = {cur_h:.2f}")
        last_print = now


    arm_actions = {}
    arm_actions = leader.get_action()

    keyboard_keys = keyboard.get_action()
    base_action = robot._from_keyboard_to_base_action(keyboard_keys)
    lift_action = robot._from_keyboard_to_lift_action(keyboard_keys)

    #lift_action = {"lift_axis.height_mm": 100}
    # Z轴 粘性控制
    # 若语音给了绝对高度，设为“粘性目标”，并从一次性动作里移除（避免只生效一帧）
    if "lift_axis.height_mm" in voice_act:
        voice_z_target_mm = float(voice_act.pop("lift_axis.height_mm"))
    #（可选）若语音指令里带取消标记，就关闭粘性跟踪
    if voice_act.get("__cancel_z"):
        voice_z_target_mm = None
        voice_act.pop("__cancel_z", None)
    # 未到位就每帧把目标写进 lift_action；到位后清掉粘性目标
    if voice_z_target_mm is not None:
        if abs(cur_h - voice_z_target_mm) <= VOICE_Z_EPS:
            voice_z_target_mm = None
        else:
            lift_action["lift_axis.height_mm"] = voice_z_target_mm


    action = {**arm_actions, **base_action, **lift_action, **voice_act}

    #print(f"teleoperate_bi_voice.action:{action}")
    

    log_rerun_data(observation, {**arm_actions, **base_action, **lift_action})
    robot.send_action(action)

    

    busy_wait(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

