#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Print actor/critic observation dimensions for Cable-Bend-U (requires Isaac Sim)."""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import gymnasium as gym
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-Cable-Bend-U")
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import unitree_rl_lab.tasks  # noqa: E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402


def main() -> None:
    env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)
    obs, _ = env.reset()
    unwrapped = env.unwrapped

    print(f"\n=== {args.task} observation space ===\n")
    for group_name, group_obs in obs.items():
        print(f"[{group_name}] shape={group_obs.shape} dtype={group_obs.dtype}")
        finite = torch.isfinite(group_obs).all().item() if hasattr(group_obs, "isfinite") else True
        print(f"  finite={finite} min={group_obs.min().item():.4f} max={group_obs.max().item():.4f}")

    om = unwrapped.observation_manager
    print("\n--- Per-term (single step, no history stack) ---")
    for group_name in om.active_terms:
        term_names = om.active_terms[group_name]
        term_dims = om.group_obs_term_dim[group_name]
        dim_sum = sum(int(np.prod(d)) for d in term_dims)
        group_cfg = getattr(unwrapped.cfg.observations, group_name)
        hist = getattr(group_cfg, "history_length", 1)
        print(f"\n{group_name}: terms={dim_sum}, history={hist}, stacked={dim_sum * hist}")
        for term_name, term_dim in zip(term_names, term_dims):
            flat = int(np.prod(term_dim))
            print(f"  {term_name}: {term_dim} -> {flat}")

    print(f"\naction_dim={unwrapped.action_manager.total_action_dim}")
    env.close()


if __name__ == "__main__":
    import numpy as np
    import torch

    main()
    simulation_app.close()
