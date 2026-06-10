# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for thick cable validation scripts."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import isaacsim.core.utils.prims as prim_utils
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext

from unitree_rl_lab.assets.cable.thick_cable_cfg import THICK_CABLE_DEFAULT_CFG, ThickCableCfg, make_thick_cable_articulation_cfg
from unitree_rl_lab.assets.cable.thick_cable_utils import get_cable_centerline_points
from unitree_rl_lab.assets.cable.thick_cable_viz import ThickCableVisualizer


# Visual-only materials (physics unchanged).
CABLE_VISUAL_MATERIAL = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.22, 0.24, 0.28), roughness=0.28, metallic=0.35)
PUSHER_VISUAL_MATERIAL = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.45, 0.08), roughness=0.35)
TORSO_VISUAL_MATERIAL = sim_utils.PreviewSurfaceCfg(
    diffuse_color=(0.55, 0.72, 0.92), roughness=0.4, opacity=0.45
)
ANCHOR_VISUAL_MATERIAL = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.12, 0.12), roughness=0.35)
TABLE_VISUAL_MATERIAL = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.55, 0.58), roughness=0.55)


def make_simulation_cfg(cable_cfg: ThickCableCfg) -> sim_utils.SimulationCfg:
    return sim_utils.SimulationCfg(
        dt=cable_cfg.physics_dt,
        render_interval=max(1, int(round((1.0 / 30.0) / cable_cfg.physics_dt))),
        physx=sim_utils.PhysxCfg(
            solver_type=1,
            enable_ccd=False,
            enable_stabilization=False,
        ),
    )


def setup_base_scene(sim: SimulationContext, cable_cfg: ThickCableCfg) -> Articulation:
    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=4200.0, color=(0.92, 0.94, 1.0)).func("/World/Light", sim_utils.DomeLightCfg())
    sim_utils.DistantLightCfg(intensity=3200.0, angle=0.45, color=(1.0, 0.98, 0.95)).func(
        "/World/KeyLight", sim_utils.DistantLightCfg(intensity=3200.0, angle=0.45)
    )

    articulation_cfg = make_thick_cable_articulation_cfg(cable_cfg)
    cable = Articulation(articulation_cfg)
    return cable


def spawn_table(table_path: str = "/World/Table", height: float = 0.75, size: tuple[float, float, float] = (1.2, 0.8, 0.05)) -> str:
    prim_utils.create_prim(table_path, "Xform", translation=(0.0, 0.0, height))
    sim_utils.CuboidCfg(
        size=size,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=TABLE_VISUAL_MATERIAL,
    ).func(f"{table_path}/top", sim_utils.CuboidCfg(size=size))
    return table_path


def spawn_kinematic_pusher(path: str, radius: float, position: tuple[float, float, float]) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=path,
        spawn=sim_utils.SphereCfg(
            radius=radius,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=PUSHER_VISUAL_MATERIAL,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=position),
    )


def spawn_torso_collider(path: str, position: tuple[float, float, float], radius: float = 0.18, height: float = 0.45) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=path,
        spawn=sim_utils.CylinderCfg(
            radius=radius,
            height=height,
            axis="Y",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=TORSO_VISUAL_MATERIAL,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=position),
    )


def move_kinematic_object(obj: RigidObject, position: tuple[float, float, float]) -> None:
    pose = obj.data.default_root_state.clone()
    pose[:, :3] = torch.tensor(position, device=obj.device, dtype=pose.dtype)
    pose[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=obj.device, dtype=pose.dtype)
    obj.write_root_pose_to_sim(pose[:, :7])


def reset_cable_state(cable: Articulation, cable_cfg: ThickCableCfg) -> None:
    root_state = cable.data.default_root_state.clone()
    root_state[:, :3] = torch.tensor(cable_cfg.init_pos, device=cable.device)
    root_state[:, 3:7] = torch.tensor(cable_cfg.init_rot, device=cable.device)
    cable.write_root_pose_to_sim(root_state[:, :7])
    cable.write_root_velocity_to_sim(root_state[:, 7:])
    joint_pos = cable.data.default_joint_pos.clone()
    joint_vel = cable.data.default_joint_vel.clone()
    cable.write_joint_state_to_sim(joint_pos, joint_vel)
    cable.reset()


def save_centerline_log(
    cable: Articulation,
    output_path: pathlib.Path,
    step: int,
    extra: dict[str, Any] | None = None,
) -> None:
    centerline = get_cable_centerline_points(cable)[0].detach().cpu().tolist()
    lowest_z = min(p[2] for p in centerline)
    payload = {"step": step, "centerline": centerline, "lowest_z": lowest_z}
    if extra:
        payload.update(extra)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def spawn_endpoint_marker(path: str, position: tuple[float, float, float], radius: float = 0.03) -> None:
    """Spawn a red kinematic marker sphere for fixed cable endpoints (visual only)."""
    sim_utils.SphereCfg(
        radius=radius,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        visual_material=ANCHOR_VISUAL_MATERIAL,
    ).func(path, sim_utils.SphereCfg(radius=radius), translation=position)


def make_visualizer() -> ThickCableVisualizer:
    return ThickCableVisualizer()


def set_prim_translation(prim_path: str, translation: tuple[float, float, float]) -> None:
    from pxr import Gf, UsdGeom

    from isaaclab.sim.utils import get_current_stage

    prim = get_current_stage().GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    xformable = UsdGeom.Xformable(prim)
    translate_op = xformable.GetTranslateOp()
    if translate_op is None:
        translate_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    translate_op.Set(Gf.Vec3d(*translation))


DEFAULT_OUTPUT_ROOT = pathlib.Path(__file__).resolve().parents[2] / "demos" / "thick_cable"
