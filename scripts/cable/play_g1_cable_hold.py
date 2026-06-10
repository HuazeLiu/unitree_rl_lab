#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive demo: G1 stands still with arms up, holding a thick cable at both hands."""

from __future__ import annotations

import argparse
import pathlib
import sys
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 cable-hold scene (static pose + welded cable endpoints).")
parser.add_argument(
    "--num_steps",
    type=int,
    default=0,
    help="Max simulation steps (0 = run until the Isaac Sim window is closed).",
)
parser.add_argument("--slowdown", type=float, default=0.0, help="Sleep seconds after each GUI step.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def main() -> None:
    headless = bool(getattr(args_cli, "headless", False))
    disable_fabric = bool(getattr(args_cli, "disable_fabric", False))

    print("[INFO] Building Unitree-G1-29dof-Cable-Hold...", flush=True)
    env_cfg = parse_env_cfg(
        "Unitree-G1-29dof-Cable-Hold",
        device=args_cli.device,
        num_envs=1,
        use_fabric=not disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    env = gym.make("Unitree-G1-29dof-Cable-Hold", cfg=env_cfg)
    print("[INFO] Resetting environment (robot pose + cable hand attach)...", flush=True)
    env.reset()

    action_dim = env.unwrapped.action_manager.total_action_dim
    zero_action = torch.zeros(env.unwrapped.num_envs, action_dim, device=env.unwrapped.device)

    print("[INFO] G1 cable-hold scene running.", flush=True)
    if headless:
        print("[INFO] Headless mode: close with Ctrl+C or set --num_steps > 0.", flush=True)
    else:
        print("[INFO] Close the Isaac Sim window to stop (or use --num_steps).", flush=True)

    step = 0
    max_steps = args_cli.num_steps if args_cli.num_steps > 0 else None

    if headless:
        if max_steps is None:
            raise ValueError("Headless mode requires --num_steps > 0 (no viewer to keep the app alive).")
        for step in range(max_steps):
            with torch.inference_mode():
                env.step(zero_action)
    else:
        while simulation_app.is_running():
            with torch.inference_mode():
                env.step(zero_action)
            step += 1
            if args_cli.slowdown > 0.0:
                import time

                time.sleep(args_cli.slowdown)
            if max_steps is not None and step >= max_steps:
                break

    env.close()
    total = max_steps if headless and max_steps is not None else step
    print(f"[INFO] Finished after {total} simulation steps.", flush=True)


def _run() -> None:
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


# Run when executed via `python play_g1_cable_hold.py` or `isaaclab.sh -p play_g1_cable_hold.py`.
# Do not guard with __name__: some Isaac Sim launchers do not set __name__ == "__main__".
_run()

