#!/usr/bin/env python3
"""Step Cable-Bend-U and report non-finite observations."""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import gymnasium as gym
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import unitree_rl_lab.tasks  # noqa: E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402


def main() -> None:
    env = gym.make("Unitree-G1-29dof-Cable-Bend-U", cfg=parse_env_cfg("Unitree-G1-29dof-Cable-Bend-U", num_envs=args.num_envs))
    obs, _ = env.reset()
    act_dim = env.unwrapped.action_manager.total_action_dim
    for step in range(30):
        action = torch.empty(env.unwrapped.num_envs, act_dim, device=env.unwrapped.device).uniform_(-0.5, 0.5)
        obs, _, _, _, _ = env.step(action)
        for k, v in obs.items():
            bad = ~torch.isfinite(v)
            if bad.any():
                print(f"step {step} group {k}: non-finite count={bad.sum().item()} max={v.abs().max().item()}")
    for k, v in obs.items():
        print(f"final {k}: shape={v.shape} max={v.abs().max().item():.3f} finite={torch.isfinite(v).all().item()}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
