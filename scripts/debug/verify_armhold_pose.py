"""Headless debug script: verify ArmHold action space and arm joint tracking (env 0 only)."""

import argparse
import importlib
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify ArmHold pose and action space.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-Velocity-ArmHold")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_steps", type=int, default=5)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

_armhold_cfg = importlib.import_module(
    "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_armhold_env_cfg"
)
ARM_HOLD_POSE = _armhold_cfg.ARM_HOLD_POSE
ARM_JOINT_NAMES = _armhold_cfg.ARM_JOINT_NAMES
LOCOMOTION_JOINT_NAMES = _armhold_cfg.LOCOMOTION_JOINT_NAMES


def _print_arm_state(robot, label: str) -> None:
    env_id = 0
    print(f"\n=== {label} (env {env_id}) ===")
    print(f"{'joint':<30} {'target':>10} {'actual':>10} {'|err|':>10} {'vel':>10}")
    print("-" * 74)
    max_err = 0.0
    for joint_name in ARM_JOINT_NAMES:
        joint_ids, _ = robot.find_joints(joint_name)
        jid = joint_ids[0]
        target = ARM_HOLD_POSE[joint_name]
        actual = robot.data.joint_pos[env_id, jid].item()
        vel = robot.data.joint_vel[env_id, jid].item()
        err = abs(actual - target)
        max_err = max(max_err, err)
        print(f"{joint_name:<30} {target:10.4f} {actual:10.4f} {err:10.4f} {vel:10.4f}")
    print(f"max |error| across arm joints: {max_err:.4f}")


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    robot = env.unwrapped.scene["robot"]

    action_term = env.unwrapped.action_manager.get_term("JointPositionAction")
    print("\n=== Action space ===")
    print(f"action_dim (policy-controlled): {action_term.action_dim}")
    print(f"controlled joint count: {action_term.action_dim}")
    print(f"controlled joint names: {action_term._joint_names}")
    print(f"locomotion joint patterns (config): {LOCOMOTION_JOINT_NAMES}")
    print(f"fixed arm joints (ARM_HOLD_POSE): {ARM_JOINT_NAMES}")
    print("action override location: velocity_armhold_env_cfg.py -> ArmHoldActionsCfg.JointPositionAction")
    print("arm hold action: velocity_armhold_env_cfg.py -> ArmHoldActionsCfg.ArmHoldPositionAction")
    print("arm hold reset event: velocity_armhold_env_cfg.py -> ArmHoldEventCfg.hold_arms_reset")
    print("arm hold function: tasks/locomotion/mdp/events.py -> hold_joint_position_targets")

    all_joint_names = robot.data.joint_names
    controlled = set(action_term._joint_names)
    excluded = [n for n in all_joint_names if n not in controlled]
    print(f"total robot joints: {len(all_joint_names)}")
    print(f"excluded from policy ({len(excluded)} joints): {excluded}")

    env.reset()
    _print_arm_state(robot, "after reset")

    zero_action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    for step in range(args_cli.num_steps):
        env.step(zero_action)
        _print_arm_state(robot, f"after step {step + 1} (zero leg action)")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
