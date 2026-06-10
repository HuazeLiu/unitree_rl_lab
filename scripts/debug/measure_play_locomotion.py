"""Headless: measure root displacement during play for a checkpoint."""

import argparse
import pathlib
import sys

_script_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_script_root))
sys.path.insert(0, str(_script_root / "rsl_rl"))

from isaaclab.app import AppLauncher

import cli_args

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_steps", type=int, default=300)
parser.add_argument("--no_arm_kinematic_lock", action="store_true", help="Disable write_joint_state on arms (debug).")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not args_cli.checkpoint:
    parser.error("--checkpoint required")
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=False,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    resume_path = retrieve_file_path(args_cli.checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    base = env.unwrapped
    while hasattr(base, "env"):
        base = base.env
    robot = base.scene["robot"]

    if args_cli.no_arm_kinematic_lock:
        arm_term = base.action_manager.get_term("ArmHoldPositionAction")
        orig_apply = arm_term.apply_actions

        def apply_targets_only() -> None:
            arm_term._asset.set_joint_position_target(arm_term._targets, joint_ids=arm_term._joint_ids)

        arm_term.apply_actions = apply_targets_only
        print("[DEBUG] ArmHold: targets only (no write_joint_state_to_sim)")

    obs, _ = env.reset()
    p0 = robot.data.root_pos_w[0, :2].clone()
    cmd_mgr = base.command_manager

    for step in range(1, args_cli.num_steps + 1):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        if step in (1, 50, 100, 200, 300):
            cmd = cmd_mgr.get_command("base_velocity")[0]
            vel = robot.data.root_lin_vel_b[0]
            disp = torch.norm(robot.data.root_pos_w[0, :2] - p0).item()
            print(
                f"step={step:4d} cmd_norm={torch.norm(cmd):.3f} "
                f"speed_xy={torch.norm(vel[:2]):.3f} disp_xy={disp:.3f} "
                f"act_mean={actions[0].abs().mean():.3f}"
            )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
