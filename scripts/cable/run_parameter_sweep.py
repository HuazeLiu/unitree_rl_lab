#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Parameter sweep for thick cable tuning (sag test metrics)."""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Thick cable parameter sweep using sag validation metrics.")
parser.add_argument("--max_runs", type=int, default=0, help="Limit number of grid points (0 = full grid).")
parser.add_argument("--num_steps", type=int, default=480)
parser.add_argument("--output_dir", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim import SimulationContext

from unitree_rl_lab.assets.cable.thick_cable_builder import build_thick_cable_metadata, create_fixed_anchor
from unitree_rl_lab.assets.cable.thick_cable_cfg import THICK_CABLE_DEFAULT_CFG, ThickCableCfg
from unitree_rl_lab.assets.cable.thick_cable_utils import compute_bending_energy, get_cable_centerline_points
from cable.cable_validation_common import (
    DEFAULT_OUTPUT_ROOT,
    make_simulation_cfg,
    reset_cable_state,
    setup_base_scene,
)

BEND_STIFFNESS = [0.03, 0.08, 0.15, 0.30]
BEND_DAMPING = [0.006, 0.012, 0.025, 0.05]
BEND_LIMIT_DEG = [8, 12, 18, 25]
NUM_SEGMENTS = [16, 24, 32]
TOTAL_MASS = [0.4, 0.8, 1.2]


def _run_single(cfg: ThickCableCfg, num_steps: int) -> dict:
    sim = SimulationContext(make_simulation_cfg(cfg))
    cable = setup_base_scene(sim, cfg)
    metadata = build_thick_cable_metadata(cfg.prim_path, cfg.num_segments, cfg.total_length, cfg.total_mass)
    create_fixed_anchor("/World/Anchors/start", metadata.link_paths[0], cfg.init_pos)
    end_world = (
        cfg.init_pos[0] + (cfg.num_segments - 1) * metadata.segment_length,
        cfg.init_pos[1],
        cfg.init_pos[2],
    )
    create_fixed_anchor("/World/Anchors/end", metadata.link_paths[-1], end_world)
    sim.reset()
    reset_cable_state(cable, cfg)

    sim_dt = sim.get_physics_dt()
    for _ in range(num_steps):
        cable.write_data_to_sim()
        sim.step()
        cable.update(sim_dt)

    centerline = get_cable_centerline_points(cable)[0]
    lowest_z = centerline[:, 2].min().item()
    sag_depth = cfg.init_pos[2] - lowest_z
    bend_energy = compute_bending_energy(cable)[0].item()
    stretch_proxy = abs(centerline[-1, 0].item() - end_world[0])

    return {
        "lowest_z": lowest_z,
        "sag_depth": sag_depth,
        "bending_energy": bend_energy,
        "stretch_proxy": stretch_proxy,
    }


def main() -> None:
    output_dir = pathlib.Path(args_cli.output_dir) if args_cli.output_dir else DEFAULT_OUTPUT_ROOT / "parameter_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "sweep_results.jsonl"

    grid = list(
        itertools.product(BEND_STIFFNESS, BEND_DAMPING, BEND_LIMIT_DEG, NUM_SEGMENTS, TOTAL_MASS)
    )
    if args_cli.max_runs > 0:
        grid = grid[: args_cli.max_runs]

    print(f"[INFO] Running {len(grid)} sweep configurations...")
    for idx, (stiffness, damping, bend_limit, num_segments, total_mass) in enumerate(grid):
        base = THICK_CABLE_DEFAULT_CFG
        cfg = ThickCableCfg(
            prim_path=f"/World/Cable_{idx}",
            total_length=base.total_length,
            radius=base.radius,
            num_segments=num_segments,
            total_mass=total_mass,
            bend_stiffness=stiffness,
            bend_damping=damping,
            twist_stiffness=base.twist_stiffness,
            twist_damping=base.twist_damping,
            bend_limit_deg=float(bend_limit),
            twist_limit_deg=base.twist_limit_deg,
            static_friction=base.static_friction,
            dynamic_friction=base.dynamic_friction,
            self_collision=base.self_collision,
            physics_dt=base.physics_dt,
            position_solver_iterations=base.position_solver_iterations,
            velocity_solver_iterations=base.velocity_solver_iterations,
            init_pos=(-0.5 * base.total_length, 0.0, 1.2),
        )

        metrics = _run_single(cfg, args_cli.num_steps)
        record = {
            "run_id": idx,
            "bend_stiffness": stiffness,
            "bend_damping": damping,
            "bend_limit_deg": bend_limit,
            "num_segments": num_segments,
            "total_mass": total_mass,
            **metrics,
        }
        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[{idx + 1}/{len(grid)}] sag_depth={metrics['sag_depth']:.4f} bend={metrics['bending_energy']:.4f}")

    print(f"[RESULT] sweep logs saved to {results_path}")
    simulation_app.close()


if __name__ == "__main__":
    main()
