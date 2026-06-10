#!/usr/bin/env python3
"""Diagnose chamfer distance between cable and target U centerlines.

Steps env 10 times to check if cable converges to hands.
"""

import importlib
import os
import sys
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_UNITREE_SRC = os.path.join(_THIS_DIR, "..", "..", "source", "unitree_rl_lab")
_ISAAC_SRC = os.path.join(_THIS_DIR, "..", "..", "..", "IsaacLab", "source", "isaaclab")
sys.path.insert(0, _UNITREE_SRC)
sys.path.insert(0, _ISAAC_SRC)

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(dict(headless=True, hide_ui=True, enable_cameras=False, livestream=0))
simulation_app = app_launcher.app

import gymnasium as gym
import unitree_rl_lab.tasks  # noqa

G1CableBendUEnvCfg = getattr(
    importlib.import_module("unitree_rl_lab.tasks.locomotion.robots.g1.29dof.g1_cable_bend_u_env_cfg"),
    "G1CableBendUEnvCfg",
)

from unitree_rl_lab.assets.cable.thick_cable_utils import get_cable_centerline_points

env_cfg = G1CableBendUEnvCfg()
env_cfg.scene.num_envs = 1
env_cfg.sim.device = "cuda:0"
env = gym.make("Unitree-G1-29dof-Cable-Bend-U", cfg=env_cfg, render_mode=None)
obs, _ = env.reset()

robot = env.unwrapped.scene["robot"]
cable = env.unwrapped.scene["cable"]

left_ids, _ = robot.find_bodies("left_rubber_hand")
right_ids, _ = robot.find_bodies("right_rubber_hand")

print("Step | Left Hand Y       | Right Hand Y      | Cable Seg00 Y     | Cable Seg11 Y     | Cable span")
print("-" * 100)

for step in range(11):
    lh = robot.data.body_pos_w[0, left_ids[0]]
    rh = robot.data.body_pos_w[0, right_ids[0]]
    centerline = get_cable_centerline_points(cable)
    s0 = centerline[0, 0]
    s11 = centerline[0, -1]
    span = torch.norm(s0 - s11).item()  # centerline endpoint distance
    print(f"  {step:2d}  | {lh[1].item():.4f}            | {rh[1].item():.4f}            | {s0[1].item():.4f}            | {s11[1].item():.4f}            | {span:.4f}m")
    if step < 10:
        obs, _, _, _, _ = env.step(torch.zeros(1, 17, device="cuda:0"))

# Final check
cmd_term = env.unwrapped.command_manager.get_term("cable_bend_u")
target = cmd_term.target_centerline
print(f"\nTarget span: {torch.norm(target[0,0] - target[0,-1]).item():.4f}m")
print(f"Target endpoints: ({target[0,0,1].item():.4f}, {target[0,-1,1].item():.4f}) Y")

env.close()
simulation_app.close()
