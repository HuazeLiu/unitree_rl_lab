# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""CLI helpers for thick cable visual workflows (no Isaac Sim imports)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class SceneCameraView:
    """Viewport and sensor camera settings for a validation scene."""

    eye: tuple[float, float, float]
    target: tuple[float, float, float]
    scene_name: str


DEFAULT_VIDEO_DIR = "demos/thick_cable/videos"


def add_cable_visual_args(parser: argparse.ArgumentParser) -> None:
    """Add shared visual / recording CLI flags."""
    parser.add_argument(
        "--record_video",
        action="store_true",
        help="Save PNG frames from the scene camera (implies --enable_cameras).",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default=DEFAULT_VIDEO_DIR,
        help="Directory for PNG frame sequences and mp4 outputs.",
    )
    parser.add_argument(
        "--slowdown",
        type=float,
        default=0.0,
        help="Optional sleep duration in seconds after each GUI step.",
    )
    parser.add_argument(
        "--video_width",
        type=int,
        default=1280,
        help="Recorded camera width in pixels.",
    )
    parser.add_argument(
        "--video_height",
        type=int,
        default=720,
        help="Recorded camera height in pixels.",
    )
    parser.add_argument(
        "--video_stride",
        type=int,
        default=2,
        help="Save one frame every N simulation steps when recording.",
    )


def add_cable_preset_arg(parser: argparse.ArgumentParser) -> None:
    """Add cable physics preset selector."""
    parser.add_argument(
        "--cable_preset",
        type=str,
        choices=["default", "cable", "rubber", "stiffer", "softer", "legacy"],
        default="default",
        help="Cable preset: default/cable (CABLE_LIKE_CFG) | rubber | stiffer | softer | legacy.",
    )


def add_cable_physics_override_args(parser: argparse.ArgumentParser) -> None:
    """Optional CLI overrides applied on top of the selected preset."""
    parser.add_argument("--num_segments", type=int, default=None, help="Override segment count.")
    parser.add_argument("--total_mass", type=float, default=None, help="Override total cable mass (kg).")
    parser.add_argument("--bend_stiffness", type=float, default=None, help="Override bend stiffness (N·m/rad).")
    parser.add_argument("--bend_damping", type=float, default=None, help="Override bend damping (N·m·s/rad).")
    parser.add_argument("--twist_stiffness", type=float, default=None, help="Override twist stiffness (N·m/rad).")
    parser.add_argument("--twist_damping", type=float, default=None, help="Override twist damping (N·m·s/rad).")
    parser.add_argument("--bend_limit_deg", type=float, default=None, help="Override per-joint bend limit (deg).")
    parser.add_argument("--max_joint_effort", type=float, default=None, help="Override joint effort limit (N·m).")
    parser.add_argument(
        "--physics_dt",
        type=float,
        default=None,
        help="Override simulation physics dt (e.g. 0.004166 for 240 Hz).",
    )


def apply_visual_launch_defaults(args: argparse.Namespace) -> None:
    """Ensure camera rendering is enabled when recording."""
    if getattr(args, "record_video", False):
        args.enable_cameras = True
