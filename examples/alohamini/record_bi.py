from datetime import datetime
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.record import record_loop
from lerobot.robots.alohamini.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.alohamini.lekiwi_client import LeKiwiClient
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import _init_rerun
from lerobot.teleoperators.bi_so100_leader import BiSO100Leader, BiSO100LeaderConfig  # 新增

NUM_EPISODES = 1
FPS = 30
EPISODE_TIME_SEC = 60
RESET_TIME_SEC = 10
TASK_DESCRIPTION = "My task description4"

# Create the robot and teleoperator configurations
robot_config = LeKiwiClientConfig(remote_ip="127.0.0.1", id="my_lekiwi")
bi_cfg = BiSO100LeaderConfig(
    left_arm_port="/dev/am_arm_leader_left",
    right_arm_port="/dev/am_arm_leader_right",
    id="so101_leader_bi",
    # 如果你原来给 SO100/SO101 leader 传了 calibration_dir，也可在这里加
    # calibration_dir="path/to/calib",
)
leader = BiSO100Leader(bi_cfg)


keyboard_config = KeyboardTeleopConfig()

robot = LeKiwiClient(robot_config)
#leader_arm = SO100Leader(leader_arm_config)

keyboard = KeyboardTeleop(keyboard_config)

# Configure the dataset features
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

# Create the dataset
dataset = LeRobotDataset.create(
    #repo_id=f"liyitenga/record100",
    repo_id=f"liyitenga/record_{datetime.now().strftime('%Y%m%d%H%M%S')}",
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4,
)

print(f"Dataset created with id: {dataset.repo_id}")

# To connect you already should have this script running on LeKiwi: `python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_awesome_kiwi`
robot.connect()
leader.connect()
# left_leader.connect()
# right_leader.connect()
keyboard.connect()

_init_rerun(session_name="lekiwi_record")

listener, events = init_keyboard_listener()

#if not robot.is_connected or not leader_arm.is_connected or not keyboard.is_connected:
if not robot.is_connected or not leader.is_connected or not keyboard.is_connected:

    raise ValueError("Robot, leader arm of keyboard is not connected!")

recorded_episodes = 0
while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
    log_say(f"Recording episode {recorded_episodes}")

    # Run the record loop
    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        dataset=dataset,
        #teleop=[leader_arm, keyboard],
        teleop=[leader, keyboard],
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
    )

    # Logic for reset env
    if not events["stop_recording"] and (
        (recorded_episodes < NUM_EPISODES - 1) or events["rerecord_episode"]
    ):
        log_say("Reset the environment")
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            teleop=[leader, keyboard],
            control_time_s=RESET_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
        )

    if events["rerecord_episode"]:
        log_say("Re-record episode")
        events["rerecord_episode"] = False
        events["exit_early"] = False
        dataset.clear_episode_buffer()
        continue

    dataset.save_episode()
    recorded_episodes += 1

# Upload to hub and clean up
dataset.push_to_hub()
print(f"Dataset created with id: {dataset.repo_id}")

robot.disconnect()
#leader_arm.disconnect()
leader.disconnect()
keyboard.disconnect()
listener.stop()
