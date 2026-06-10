# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Thick cable validation scene builders shared by CLI and visual demo scripts."""

from __future__ import annotations

import argparse
import math
import pathlib
from typing import Any

import torch

from isaaclab.assets import RigidObject
from isaaclab.sim import SimulationContext

from unitree_rl_lab.assets.cable.thick_cable_builder import build_thick_cable_metadata, create_fixed_anchor
from unitree_rl_lab.assets.cable.thick_cable_cfg import ThickCableCfg
from unitree_rl_lab.assets.cable.thick_cable_utils import (
    compute_bending_energy,
    compute_smoothness_reward,
    get_cable_centerline_points,
)

from cable.cable_validation_common import (
    DEFAULT_OUTPUT_ROOT,
    make_simulation_cfg,
    make_visualizer,
    move_kinematic_object,
    reset_cable_state,
    save_centerline_log,
    setup_base_scene,
    spawn_endpoint_marker,
    spawn_kinematic_pusher,
    spawn_table,
    spawn_torso_collider,
)
from cable.cable_preset_utils import finalize_cable_cfg, log_cable_cfg
from cable.cable_visual_cli import SceneCameraView
from cable.cable_visual_runtime import CableFrameRecorder, configure_viewport, run_simulation_loop

SAG_CAMERA = SceneCameraView(eye=(1.0, -1.2, 1.25), target=(0.0, 0.0, 1.12), scene_name="sag")
PUSH_CAMERA = SceneCameraView(eye=(0.95, -1.05, 0.98), target=(0.40, 0.0, 0.79), scene_name="push")
TORSO_CAMERA = SceneCameraView(eye=(0.75, -0.95, 1.05), target=(0.0, -0.20, 0.88), scene_name="torso")


def _debug_viz_flags(args: argparse.Namespace) -> dict[str, bool]:
    """Skip bead-like debug markers when recording video."""
    recording = getattr(args, "record_video", False)
    return {
        "draw_centerline": not recording,
        "draw_endpoints": not recording,
        "color_by_curvature": not recording,
    }


def _make_sag_cfg(args: argparse.Namespace, anchor_height: float) -> ThickCableCfg:
    base = finalize_cable_cfg(args)
    return base.replace(
        prim_path="/World/Cable",
        init_pos=(-0.5 * base.total_length, 0.0, anchor_height),
    )


def _recorder_from_args(args: argparse.Namespace, sim: SimulationContext, view: SceneCameraView) -> CableFrameRecorder:
    video_root = pathlib.Path(args.video_dir)
    if not video_root.is_absolute():
        video_root = DEFAULT_OUTPUT_ROOT.parent.parent / video_root
    return CableFrameRecorder(
        sim=sim,
        view=view,
        enabled=getattr(args, "enable_cameras", False),
        record=getattr(args, "record_video", False),
        frame_dir=video_root / view.scene_name,
        width=getattr(args, "video_width", 1280),
        height=getattr(args, "video_height", 720),
    )


def run_sag_scene(args: argparse.Namespace, simulation_app) -> dict[str, Any]:
    anchor_height = getattr(args, "anchor_height", 1.2)
    cable_cfg = _make_sag_cfg(args, anchor_height)
    output_dir = pathlib.Path(args.output_dir) if getattr(args, "output_dir", "") else DEFAULT_OUTPUT_ROOT / "sag_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "sag_centerline.jsonl"
    if log_path.exists():
        log_path.unlink()

    sim = SimulationContext(make_simulation_cfg(cable_cfg))
    configure_viewport(sim, SAG_CAMERA, headless=args.headless)
    cable = setup_base_scene(sim, cable_cfg)
    metadata = build_thick_cable_metadata(
        cable_cfg.prim_path,
        cable_cfg.num_segments,
        cable_cfg.total_length,
        cable_cfg.total_mass,
    )

    start_world = cable_cfg.init_pos
    end_world = (
        cable_cfg.init_pos[0] + (cable_cfg.num_segments - 1) * metadata.segment_length,
        cable_cfg.init_pos[1],
        cable_cfg.init_pos[2],
    )
    create_fixed_anchor("/World/Anchors/start", metadata.link_paths[0], start_world)
    create_fixed_anchor("/World/Anchors/end", metadata.link_paths[-1], end_world)
    spawn_endpoint_marker("/World/Visuals/AnchorStart", start_world)
    spawn_endpoint_marker("/World/Visuals/AnchorEnd", end_world)

    recorder = _recorder_from_args(args, sim, SAG_CAMERA)
    sim.reset()
    reset_cable_state(cable, cable_cfg)
    log_cable_cfg(cable_cfg, args, cable)
    visualizer = make_visualizer()
    recorder.setup_view()
    viz_flags = _debug_viz_flags(args)
    sim_dt = sim.get_physics_dt()

    print("[INFO] Running sag validation...")

    def _step(step: int) -> None:
        cable.write_data_to_sim()
        sim.step()
        cable.update(sim_dt)
        if step % 10 == 0:
            visualizer.update_all(cable, env_id=0, **viz_flags)

    run_simulation_loop(
        sim,
        simulation_app,
        args.num_steps,
        _step,
        headless=args.headless,
        slowdown=args.slowdown,
        recorder=recorder,
        video_stride=getattr(args, "video_stride", 2),
        on_log_step=lambda s: save_centerline_log(cable, log_path, s),
        log_interval=30,
    )

    centerline = get_cable_centerline_points(cable)[0]
    lowest_z = centerline[:, 2].min().item()
    results = {"lowest_z": lowest_z, "log_path": str(log_path), "frame_dir": str(recorder.frame_dir)}
    print(f"[RESULT] lowest centerline point z = {lowest_z:.4f} m")
    print(f"[RESULT] logs saved to {log_path}")
    if recorder.record:
        print(f"[RESULT] frames saved to {recorder.frame_dir}")
    return results


def run_push_scene(args: argparse.Namespace, simulation_app) -> dict[str, Any]:
    table_height = getattr(args, "table_height", 0.75)
    base = finalize_cable_cfg(args)
    cable_cfg = base.replace(
        prim_path="/World/Cable",
        init_pos=(0.0, 0.0, table_height + base.radius + 0.01),
    )
    output_dir = pathlib.Path(args.output_dir) if getattr(args, "output_dir", "") else DEFAULT_OUTPUT_ROOT / "push_on_table"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "push_centerline.jsonl"
    if log_path.exists():
        log_path.unlink()

    sim = SimulationContext(make_simulation_cfg(cable_cfg))
    configure_viewport(sim, PUSH_CAMERA, headless=args.headless)
    cable = setup_base_scene(sim, cable_cfg)
    spawn_table(height=table_height)
    pusher_cfg = spawn_kinematic_pusher(
        "/World/Pusher",
        radius=0.06,
        position=(0.5 * cable_cfg.total_length, 0.25, table_height + 0.08),
    )
    pusher = RigidObject(pusher_cfg)

    recorder = _recorder_from_args(args, sim, PUSH_CAMERA)
    sim.reset()
    reset_cable_state(cable, cable_cfg)
    pusher.reset()
    log_cable_cfg(cable_cfg, args, cable)
    visualizer = make_visualizer()
    recorder.setup_view()
    viz_flags = _debug_viz_flags(args)
    sim_dt = sim.get_physics_dt()
    max_bend = 0.0

    print("[INFO] Running push-on-table validation...")

    def _step(step: int) -> None:
        nonlocal max_bend
        phase = step / max(args.num_steps, 1)
        y_offset = 0.25 * math.sin(2.0 * math.pi * phase)
        move_kinematic_object(
            pusher,
            (0.5 * cable_cfg.total_length, y_offset, table_height + 0.08),
        )
        pusher.write_data_to_sim()
        cable.write_data_to_sim()
        sim.step()
        cable.update(sim_dt)
        pusher.update(sim_dt)
        max_bend = max(max_bend, compute_bending_energy(cable)[0].item())
        if step % 10 == 0:
            visualizer.update_all(cable, env_id=0, **viz_flags)
        if recorder.camera is not None and getattr(args, "record_video", False):
            centerline = get_cable_centerline_points(cable)[0]
            target = centerline.mean(dim=0)
            eye = target + torch.tensor([0.55, -0.85, 0.22], device=target.device)
            recorder.camera.set_world_poses_from_view(eye.unsqueeze(0), target.unsqueeze(0))

    run_simulation_loop(
        sim,
        simulation_app,
        args.num_steps,
        _step,
        headless=args.headless,
        slowdown=args.slowdown,
        recorder=recorder,
        video_stride=getattr(args, "video_stride", 2),
        on_log_step=lambda s: save_centerline_log(
            cable, log_path, s, extra={"bending_energy": compute_bending_energy(cable)[0].item()}
        ),
        log_interval=30,
    )

    centerline = get_cable_centerline_points(cable)[0]
    y_spread = (centerline[:, 1].max() - centerline[:, 1].min()).item()
    results = {
        "max_bending_energy": max_bend,
        "y_spread": y_spread,
        "log_path": str(log_path),
        "frame_dir": str(recorder.frame_dir),
    }
    print(f"[RESULT] max bending energy proxy = {max_bend:.4f}")
    print(f"[RESULT] final centerline y spread = {y_spread:.4f} m")
    print(f"[RESULT] logs saved to {log_path}")
    if recorder.record:
        print(f"[RESULT] frames saved to {recorder.frame_dir}")
    return results


def run_torso_scene(args: argparse.Namespace, simulation_app) -> dict[str, Any]:
    cable_cfg = finalize_cable_cfg(
        args,
        prim_path="/World/Cable",
        init_pos=(-0.15, -0.25, 1.02),
    )
    output_dir = pathlib.Path(args.output_dir) if getattr(args, "output_dir", "") else DEFAULT_OUTPUT_ROOT / "torso_support"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "torso_centerline.jsonl"
    if log_path.exists():
        log_path.unlink()

    sim = SimulationContext(make_simulation_cfg(cable_cfg))
    configure_viewport(sim, TORSO_CAMERA, headless=args.headless)
    cable = setup_base_scene(sim, cable_cfg)
    torso_cfg = spawn_torso_collider("/World/Torso", position=(0.15, 0.0, 0.95))
    torso = RigidObject(torso_cfg)

    recorder = _recorder_from_args(args, sim, TORSO_CAMERA)
    sim.reset()
    reset_cable_state(cable, cable_cfg)
    torso.reset()
    log_cable_cfg(cable_cfg, args, cable)
    visualizer = make_visualizer()
    recorder.setup_view()
    viz_flags = _debug_viz_flags(args)
    sim_dt = sim.get_physics_dt()
    min_smoothness = 1.0

    print("[INFO] Running torso-support validation...")

    def _step(step: int) -> None:
        nonlocal min_smoothness
        press = 0.04 * math.sin(0.5 * math.pi * min(step / (0.35 * max(args.num_steps, 1)), 1.0))
        move_kinematic_object(torso, (0.15 + press, 0.0, 0.95))
        torso.write_data_to_sim()
        cable.write_data_to_sim()
        sim.step()
        cable.update(sim_dt)
        torso.update(sim_dt)
        min_smoothness = min(min_smoothness, compute_smoothness_reward(cable)[0].item())
        if step % 10 == 0:
            visualizer.update_all(cable, env_id=0, **viz_flags)
        if recorder.camera is not None and getattr(args, "record_video", False):
            # Keep the recorded view centered on the cable as it settles.
            centerline = get_cable_centerline_points(cable)[0]
            target = centerline.mean(dim=0)
            eye = target + torch.tensor([0.75, -0.95, 0.18], device=target.device)
            recorder.camera.set_world_poses_from_view(eye.unsqueeze(0), target.unsqueeze(0))

    run_simulation_loop(
        sim,
        simulation_app,
        args.num_steps,
        _step,
        headless=args.headless,
        slowdown=args.slowdown,
        recorder=recorder,
        video_stride=getattr(args, "video_stride", 2),
        on_log_step=lambda s: save_centerline_log(
            cable, log_path, s, extra={"smoothness_reward": compute_smoothness_reward(cable)[0].item()}
        ),
        log_interval=30,
    )

    centerline = get_cable_centerline_points(cable)[0]
    contact_extent = (centerline[:, 0].max() - centerline[:, 0].min()).item()
    results = {
        "min_smoothness": min_smoothness,
        "contact_extent": contact_extent,
        "log_path": str(log_path),
        "frame_dir": str(recorder.frame_dir),
    }
    print(f"[RESULT] min smoothness reward = {min_smoothness:.4f}")
    print(f"[RESULT] final centerline x extent = {contact_extent:.4f} m")
    print(f"[RESULT] logs saved to {log_path}")
    if recorder.record:
        print(f"[RESULT] frames saved to {recorder.frame_dir}")
    return results
