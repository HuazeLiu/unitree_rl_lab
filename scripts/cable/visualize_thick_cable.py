#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Unified GUI / headless visual demo entry point for thick cable scenes."""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from isaaclab.app import AppLauncher

from cable.cable_visual_cli import add_cable_physics_override_args, add_cable_preset_arg, add_cable_visual_args, apply_visual_launch_defaults

parser = argparse.ArgumentParser(description="Visualize thick cable validation scenes.")
parser.add_argument(
    "--scene",
    type=str,
    choices=["sag", "push", "torso", "all"],
    default="sag",
    help="Which validation scene to run.",
)
parser.add_argument("--num_steps", type=int, default=600)
parser.add_argument("--anchor_height", type=float, default=1.2)
parser.add_argument("--table_height", type=float, default=0.75)
parser.add_argument("--output_dir", type=str, default="")
add_cable_visual_args(parser)
add_cable_preset_arg(parser)
add_cable_physics_override_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
apply_visual_launch_defaults(args_cli)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from cable.cable_scenes import run_push_scene, run_sag_scene, run_torso_scene

SCENE_RUNNERS = {
    "sag": run_sag_scene,
    "push": run_push_scene,
    "torso": run_torso_scene,
}


def main() -> None:
    if args_cli.scene == "all":
        if not args_cli.headless:
            print("[WARN] --scene all in GUI mode runs sag only (one viewer session).")
            print("[WARN] Run push/torso separately, or use --headless --scene all for batch recording.")
            run_sag_scene(args_cli, simulation_app)
            return

        import subprocess

        script = str(pathlib.Path(__file__).resolve())
        base_cmd = [sys.executable, script]
        for name in ("sag", "push", "torso"):
            cmd = base_cmd + [
                "--scene",
                name,
                "--cable_preset",
                args_cli.cable_preset,
                "--headless",
                "--num_steps",
                str(args_cli.num_steps),
                "--video_dir",
                args_cli.video_dir,
                "--slowdown",
                str(args_cli.slowdown),
            ]
            if args_cli.enable_cameras:
                cmd.append("--enable_cameras")
            if args_cli.record_video:
                cmd.append("--record_video")
            print(f"\n[INFO] === subprocess scene: {name} ===")
            subprocess.run(cmd, check=True)
        return

    SCENE_RUNNERS[args_cli.scene](args_cli, simulation_app)


if __name__ == "__main__":
    main()
    simulation_app.close()
