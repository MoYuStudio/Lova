import inspect
import os
import time

from lerobot.robots.alohamini import LeKiwiClient, LeKiwiClientConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.bi_so100_leader import BiSO100Leader, BiSO100LeaderConfig 


from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data


FPS = 30
USE_DUMMY = True  # ←← 不连接 robot 但仍打印要发送的 action


def _prefix_arm_action(prefix: str, action_dict: dict) -> dict:
    # 将 leader 输出的键（如 "shoulder_pan.pos"）前缀化为
    # "left_arm_shoulder_pan.pos" 或 "right_arm_shoulder_pan.pos"
    return {f"{prefix}_{k}": v for k, v in action_dict.items()}

# Create the robot and teleoperator configurations
robot_config = LeKiwiClientConfig(remote_ip="192.168.50.43", id="my_alohamini")
#teleop_arm_config = SO101LeaderConfig(port="/dev/am_arm_leader_left", id="am_arm_sam_leader_left")

# teleop_arm_left_cfg  = SO101LeaderConfig(port="/dev/am_arm_leader_left",  id="am_arm_sam_leader_left")
# teleop_arm_right_cfg = SO101LeaderConfig(port="/dev/am_arm_leader_right", id="am_arm_sam_leader_right")

bi_cfg = BiSO100LeaderConfig(
    left_arm_port="/dev/am_arm_leader_left",
    right_arm_port="/dev/am_arm_leader_right",
    id="so101_leader_bi3",
    # 如果你原来给 SO100/SO101 leader 传了 calibration_dir，也可在这里加
    # calibration_dir=".cache/huggingface/lerobot/calibration/teleoperators/so100_leader/xxxx.json",
)
leader = BiSO100Leader(bi_cfg)



keyboard_config = KeyboardTeleopConfig(id="my_laptop_keyboard")

robot = LeKiwiClient(robot_config)
#leader_arm = SO101Leader(teleop_arm_config)

keyboard = KeyboardTeleop(keyboard_config)

# To connect you already should have this script running on LeKiwi: `python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_awesome_kiwi`
if not USE_DUMMY:
    robot.connect()
else:
    print("🧪 USE_DUMMY: robot.connect() 被跳过，仅打印 action。")

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

while True:
    t0 = time.perf_counter()

    if not USE_DUMMY:
        observation = robot.get_observation()
    else:
        observation = {}  # 占位：不需要观测，仅调试 action 数值
    
    
    # left_arm_action = left_leader.get_action()
    # right_arm_action = right_leader.get_action()
    # left_arm_action = {f"left_arm_{k}": v for k, v in left_arm_action.items()}
    # right_arm_action = {f"left_arm_{k}": v for k, v in arm_action.items()}


    arm_actions = {}
    arm_actions = leader.get_action()

    keyboard_keys = keyboard.get_action()
    base_action = robot._from_keyboard_to_base_action(keyboard_keys)
    lift_action = robot._from_keyboard_to_lift_action(keyboard_keys)

    #lift_action = {"lift_axis.height_mm": 100}
    log_rerun_data(observation, {**arm_actions, **base_action, **lift_action})

    action = {**arm_actions, **base_action, **lift_action} #if len(base_action) > 0 else arm_actions
    print(f"teleoperate_bi.action:{action}")
    


    if USE_DUMMY:
        # 仅打印，便于你观察各通道的实时数值
        print(f"[USE_DUMMY] action → {action}")
    else:
        print(f"teleoperate_bi.action: {action}")
        robot.send_action(action)
    

    busy_wait(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

