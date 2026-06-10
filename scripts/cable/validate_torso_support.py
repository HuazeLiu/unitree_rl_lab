#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Torso-support validation: press cable against a static torso-like collider."""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from isaaclab.app import AppLauncher

from cable.cable_visual_cli import add_cable_physics_override_args, add_cable_preset_arg, add_cable_visual_args, apply_visual_launch_defaults

parser = argparse.ArgumentParser(description="Thick cable torso-support validation scene.")
parser.add_argument("--num_steps", type=int, default=900)
parser.add_argument("--output_dir", type=str, default="")
add_cable_visual_args(parser)
add_cable_preset_arg(parser)
add_cable_physics_override_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
apply_visual_launch_defaults(args_cli)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from cable.cable_scenes import run_torso_scene

if __name__ == "__main__":
    run_torso_scene(args_cli, simulation_app)
    simulation_app.close()
