"""Render a single frame at reset to visually inspect ARM_HOLD_POSE variants."""

import argparse
import importlib
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-Velocity-ArmHold")
parser.add_argument("--output", type=str, default="/home/jkamohara3/Patrick_temp/unitree_rope/armhold_render.png")
parser.add_argument(
    "--pose_preset",
    type=str,
    default="current",
    choices=["current", "t_pose", "forward_horizontal", "baseline_default"],
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from PIL import Image

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

_armhold_cfg = importlib.import_module(
    "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_armhold_env_cfg"
)

POSE_PRESETS = {
    "current": _armhold_cfg.ARM_HOLD_POSE,
    "t_pose": {
        "left_shoulder_pitch_joint": 0.0,
        "right_shoulder_pitch_joint": 0.0,
        "left_shoulder_roll_joint": 1.5708,
        "right_shoulder_roll_joint": -1.5708,
        "left_shoulder_yaw_joint": 0.0,
        "right_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": 0.0,
        "right_elbow_joint": 0.0,
        "left_wrist_roll_joint": 0.0,
        "right_wrist_roll_joint": 0.0,
        "left_wrist_pitch_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "left_wrist_yaw_joint": 0.0,
        "right_wrist_yaw_joint": 0.0,
    },
    "forward_horizontal": {
        "left_shoulder_pitch_joint": 1.5708,
        "right_shoulder_pitch_joint": 1.5708,
        "left_shoulder_roll_joint": 0.25,
        "right_shoulder_roll_joint": -0.25,
        "left_shoulder_yaw_joint": 0.0,
        "right_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": 0.0,
        "right_elbow_joint": 0.0,
        "left_wrist_roll_joint": 0.0,
        "right_wrist_roll_joint": 0.0,
        "left_wrist_pitch_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "left_wrist_yaw_joint": 0.0,
        "right_wrist_yaw_joint": 0.0,
    },
    "baseline_default": {
        "left_shoulder_pitch_joint": 0.3,
        "right_shoulder_pitch_joint": 0.3,
        "left_shoulder_roll_joint": 0.25,
        "right_shoulder_roll_joint": -0.25,
        "left_shoulder_yaw_joint": 0.0,
        "right_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": 0.97,
        "right_elbow_joint": 0.97,
        "left_wrist_roll_joint": 0.15,
        "right_wrist_roll_joint": -0.15,
        "left_wrist_pitch_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "left_wrist_yaw_joint": 0.0,
        "right_wrist_yaw_joint": 0.0,
    },
}


def _apply_pose(env, preset: dict[str, float]) -> None:
    robot = env.unwrapped.scene["robot"]
    for joint_name, target_pos in preset.items():
        joint_ids, _ = robot.find_joints(joint_name)
        robot.data.joint_pos[0, joint_ids] = target_pos
        robot.data.joint_vel[0, joint_ids] = 0.0
    robot.write_joint_state_to_sim(robot.data.joint_pos, robot.data.joint_vel)
    for joint_name, target_pos in preset.items():
        joint_ids, _ = robot.find_joints(joint_name)
        target = torch.full((1, len(joint_ids)), target_pos, device=env.unwrapped.device)
        robot.set_joint_position_target(target, joint_ids=joint_ids)


def main() -> None:
    preset = POSE_PRESETS[args_cli.pose_preset]

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    import torch

    action_dim = env.unwrapped.action_manager.get_term("JointPositionAction").action_dim
    env.reset()
    _apply_pose(env, preset)
    env.step(torch.zeros(1, action_dim, device=env.unwrapped.device))
    _apply_pose(env, preset)
    frame = env.render()
    if frame is not None:
        Image.fromarray(frame).save(args_cli.output)
        print(f"saved render: {args_cli.output} (preset={args_cli.pose_preset})")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
