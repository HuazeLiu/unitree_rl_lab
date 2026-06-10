# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Procedural USD builder for a thick cable as a capsule chain with D6 joints."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import isaacsim.core.utils.prims as prim_utils
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sim import schemas
from isaaclab.sim.utils import bind_physics_material, clone, get_current_stage

if TYPE_CHECKING:
    from .thick_cable_cfg import ThickCableSpawnerCfg


@dataclass
class ThickCableBuildResult:
    """Metadata returned after building a cable on the USD stage."""

    prim_path: str
    num_segments: int
    segment_length: float
    segment_mass: float
    link_paths: list[str]
    joint_paths: list[str]


@clone
def create_thick_cable(
    prim_path: str,
    cfg: ThickCableSpawnerCfg | None = None,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    total_length: float = 0.80,
    radius: float = 0.025,
    num_segments: int = 24,
    total_mass: float = 0.8,
    bend_stiffness: float = 0.08,
    bend_damping: float = 0.012,
    twist_stiffness: float = 0.02,
    twist_damping: float = 0.006,
    bend_limit_deg: float = 12.0,
    twist_limit_deg: float = 5.0,
    static_friction: float = 0.9,
    dynamic_friction: float = 0.8,
    self_collision: bool = False,
    **kwargs,
) -> Usd.Prim:
    """Build a thick cable articulation as a chain of capsule links connected by passive D6 joints.

    This is an articulated rigid-body approximation for humanoid manipulation / RL, not a
    material-accurate cable model.

    Args:
        prim_path: USD prim path for the cable root link.
        cfg: Optional spawner config. When provided, its fields override the keyword defaults.
        translation: Root link translation w.r.t. parent.
        orientation: Root link orientation (w, x, y, z) w.r.t. parent.
        total_length: Total cable length in meters.
        radius: Capsule radius for visual and collision (meters).
        num_segments: Number of rigid capsule segments.
        total_mass: Total cable mass in kilograms.
        bend_stiffness: Passive angular stiffness for bend axes (N*m/rad).
        bend_damping: Passive angular damping for bend axes (N*m*s/rad).
        twist_stiffness: Passive angular stiffness for twist axis (N*m/rad).
        twist_damping: Passive angular damping for twist axis (N*m*s/rad).
        bend_limit_deg: Per-joint bend limit in degrees (+/-).
        twist_limit_deg: Per-joint twist limit in degrees (+/-).
        static_friction: Contact static friction coefficient.
        dynamic_friction: Contact dynamic friction coefficient.
        self_collision: Enable intra-cable self collision (off by default for stability).
        **kwargs: Additional spawner fields (restitution, contact_offset, solver iterations, etc.).

    Returns:
        The root link prim of the cable articulation.
    """
    if cfg is not None:
        total_length = cfg.total_length
        radius = cfg.radius
        num_segments = cfg.num_segments
        total_mass = cfg.total_mass
        bend_stiffness = cfg.bend_stiffness
        bend_damping = cfg.bend_damping
        twist_stiffness = cfg.twist_stiffness
        twist_damping = cfg.twist_damping
        bend_limit_deg = cfg.bend_limit_deg
        twist_limit_deg = cfg.twist_limit_deg
        static_friction = cfg.static_friction
        dynamic_friction = cfg.dynamic_friction
        self_collision = cfg.self_collision
        max_joint_effort = cfg.max_joint_effort
        max_joint_velocity = cfg.max_joint_velocity
        restitution = cfg.restitution
        contact_offset = cfg.contact_offset
        rest_offset = cfg.rest_offset
        position_solver_iterations = cfg.position_solver_iterations
        velocity_solver_iterations = cfg.velocity_solver_iterations
        disable_gravity = cfg.disable_gravity
        kinematic_enabled = cfg.kinematic_enabled
        collision_enabled = cfg.collision_enabled
    else:
        max_joint_effort = kwargs.get("max_joint_effort", 1.0)
        max_joint_velocity = kwargs.get("max_joint_velocity", 20.0)
        restitution = kwargs.get("restitution", 0.0)
        contact_offset = kwargs.get("contact_offset", 0.003)
        rest_offset = kwargs.get("rest_offset", 0.0)
        position_solver_iterations = kwargs.get("position_solver_iterations", 16)
        velocity_solver_iterations = kwargs.get("velocity_solver_iterations", 4)
        disable_gravity = kwargs.get("disable_gravity", False)
        kinematic_enabled = kwargs.get("kinematic_enabled", False)
        collision_enabled = kwargs.get("collision_enabled", True)

    if num_segments < 2:
        raise ValueError("num_segments must be >= 2 for a cable chain.")

    stage = get_current_stage()
    segment_length = total_length / num_segments
    segment_mass = total_mass / num_segments
    half_len = 0.5 * segment_length
    capsule_height = max(segment_length - 2.0 * radius, 1e-4)

    parent_path, asset_name = prim_path.rsplit("/", 1)
    physics_material_path = f"{parent_path}/Looks/{asset_name}_PhysicsMaterial"
    _spawn_physics_material(
        physics_material_path,
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        restitution=restitution,
    )

    link_paths: list[str] = []
    joint_paths: list[str] = []

    if prim_utils.is_prim_path_valid(prim_path):
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    prim_utils.create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation)
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(prim_path))
    schemas.modify_articulation_root_properties(
        prim_path,
        sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=self_collision,
            solver_position_iteration_count=position_solver_iterations,
            solver_velocity_iteration_count=velocity_solver_iterations,
        ),
        stage=stage,
    )

    for seg_idx in range(num_segments):
        link_path = f"{prim_path}/segments/seg_{seg_idx:02d}"
        _create_cable_link(
            stage=stage,
            link_path=link_path,
            radius=radius,
            segment_length=segment_length,
            capsule_height=capsule_height,
            segment_mass=segment_mass,
            physics_material_path=physics_material_path,
            contact_offset=contact_offset,
            rest_offset=rest_offset,
            translation=(seg_idx * segment_length, 0.0, 0.0),
            orientation=None,
            disable_gravity=disable_gravity,
            kinematic_enabled=kinematic_enabled,
            collision_enabled=collision_enabled,
        )
        link_paths.append(link_path)

    for seg_idx in range(num_segments - 1):
        joint_path = f"{prim_path}/joints/joint_{seg_idx:02d}"
        _create_d6_joint(
            stage=stage,
            joint_path=joint_path,
            body0_path=link_paths[seg_idx],
            body1_path=link_paths[seg_idx + 1],
            anchor_body0=(half_len, 0.0, 0.0),
            anchor_body1=(-half_len, 0.0, 0.0),
            bend_limit_deg=bend_limit_deg,
            twist_limit_deg=twist_limit_deg,
            bend_stiffness=bend_stiffness,
            bend_damping=bend_damping,
            twist_stiffness=twist_stiffness,
            twist_damping=twist_damping,
            max_joint_effort=max_joint_effort,
            max_joint_velocity=max_joint_velocity,
        )
        joint_paths.append(joint_path)

    return stage.GetPrimAtPath(prim_path)


def build_thick_cable_metadata(prim_path: str, num_segments: int, total_length: float, total_mass: float) -> ThickCableBuildResult:
    """Collect link/joint paths for a cable rooted at ``prim_path``."""
    segment_length = total_length / num_segments
    segment_mass = total_mass / num_segments
    link_paths = [f"{prim_path}/segments/seg_{i:02d}" for i in range(num_segments)]
    joint_paths = [f"{prim_path}/joints/joint_{i:02d}" for i in range(num_segments - 1)]
    return ThickCableBuildResult(
        prim_path=prim_path,
        num_segments=num_segments,
        segment_length=segment_length,
        segment_mass=segment_mass,
        link_paths=link_paths,
        joint_paths=joint_paths,
    )


def create_fixed_anchor(
    anchor_path: str,
    body_path: str,
    world_pos: tuple[float, float, float],
    local_body_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Usd.Prim:
    """Create a fixed joint from world to a cable link endpoint."""
    stage = get_current_stage()
    if not prim_utils.is_prim_path_valid(anchor_path):
        prim_utils.create_prim(anchor_path, "Xform", translation=world_pos)

    joint = UsdPhysics.FixedJoint.Define(stage, f"{anchor_path}/fixed_joint")
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body_path)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*world_pos))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*local_body_pos))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    return joint.GetPrim()


def create_body_fixed_joint(
    joint_path: str,
    body0_path: str,
    body1_path: str,
    local_pos0: tuple[float, float, float],
    local_pos1: tuple[float, float, float],
    local_rot0: Gf.Quatf | None = None,
    local_rot1: Gf.Quatf | None = None,
) -> Usd.Prim:
    """Create a fixed joint between two rigid bodies (e.g. G1 hand and cable segment).

    NOTE: a plain ``UsdPhysics.FixedJoint`` added at runtime (after ``sim.reset()`` builds the
    GPU physics view) is NOT picked up by the GPU solver, so the welded body free-falls. For a
    runtime weld that works on the GPU pipeline use :func:`create_rigid_body_attachment`
    (PhysX AutoAttachment). This function is kept for build-time (pre-sim) joint creation.
    """
    stage = get_current_stage()
    if prim_utils.is_prim_path_valid(joint_path):
        raise ValueError(f"A prim already exists at path: '{joint_path}'.")

    rot0 = local_rot0 if local_rot0 is not None else Gf.Quatf(1.0, 0.0, 0.0, 0.0)
    rot1 = local_rot1 if local_rot1 is not None else Gf.Quatf(1.0, 0.0, 0.0, 0.0)
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*local_pos0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*local_pos1))
    joint.CreateLocalRot0Attr().Set(rot0)
    joint.CreateLocalRot1Attr().Set(rot1)
    return joint.GetPrim()


def create_body_point_weld(
    joint_path: str,
    body0_path: str,
    body1_path: str,
    local_pos0: tuple[float, float, float],
    local_pos1: tuple[float, float, float],
) -> Usd.Prim:
    """Pin a point on ``body1`` to a point on ``body0`` (translation locked, rotation free).

    This is a ball/point weld: the two anchor points are forced to coincide while the bodies may
    still rotate relative to each other. It is the right model for a cable end held at a hand
    point, and (unlike a 6-DOF FixedJoint) it does not fight the runtime cable-orientation
    alignment. MUST be created before ``sim.reset()`` (e.g. in a ``prestartup`` event) so the GPU
    physics view includes the joint.
    """
    stage = get_current_stage()
    if prim_utils.is_prim_path_valid(joint_path):
        return prim_utils.get_prim_at_path(joint_path)

    joint = UsdPhysics.Joint.Define(stage, joint_path)
    joint_prim = joint.GetPrim()
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*local_pos0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*local_pos1))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateExcludeFromArticulationAttr().Set(True)
    for axis in ("transX", "transY", "transZ"):
        _lock_joint_axis(joint_prim, axis)
    return joint_prim


def create_rigid_body_attachment(
    attachment_path: str,
    actor0_path: str,
    actor1_path: str,
) -> Usd.Prim:
    """Weld two rigid bodies at runtime via the PhysX AutoAttachment API (GPU-compatible).

    Unlike a runtime ``UsdPhysics.FixedJoint`` (ignored by the GPU solver when created after the
    physics view is built), a ``PhysxPhysicsAttachment`` with ``PhysxAutoAttachmentAPI`` is
    parsed by PhysX on the next step and computes its constraint frames from the bodies' current
    simulated poses, so the bodies lock together at their present relative transform without
    snapping. Both actors must have a RigidBodyAPI.
    """
    stage = get_current_stage()
    if prim_utils.is_prim_path_valid(attachment_path):
        return prim_utils.get_prim_at_path(attachment_path)

    attachment = PhysxSchema.PhysxPhysicsAttachment.Define(stage, Sdf.Path(attachment_path))
    attachment.GetActor0Rel().SetTargets([Sdf.Path(actor0_path)])
    attachment.GetActor1Rel().SetTargets([Sdf.Path(actor1_path)])
    PhysxSchema.PhysxAutoAttachmentAPI.Apply(attachment.GetPrim())
    return attachment.GetPrim()


def _spawn_physics_material(
    material_path: str,
    static_friction: float,
    dynamic_friction: float,
    restitution: float,
) -> None:
    material_cfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        restitution=restitution,
    )
    material_cfg.func(material_path, material_cfg)


def _create_cable_link(
    stage: Usd.Stage,
    link_path: str,
    radius: float,
    segment_length: float,
    capsule_height: float,
    segment_mass: float,
    physics_material_path: str,
    contact_offset: float,
    rest_offset: float,
    translation: tuple[float, float, float] | None,
    orientation: tuple[float, float, float, float] | None,
    disable_gravity: bool = False,
    kinematic_enabled: bool = False,
    collision_enabled: bool = True,
) -> None:
    if prim_utils.is_prim_path_valid(link_path):
        raise ValueError(f"A prim already exists at path: '{link_path}'.")

    # Physics collision stays on the (possibly sphere-like) capsule; visual uses a smooth cylinder.
    visual_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.22, 0.24, 0.28), roughness=0.28, metallic=0.35)
    capsule_cfg = sim_utils.CapsuleCfg(
        radius=radius,
        height=capsule_height,
        axis="X",
        visible=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=disable_gravity,
            kinematic_enabled=kinematic_enabled,
            linear_damping=0.02,
            angular_damping=0.04,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=1.0,
        ),
        mass_props=schemas.MassPropertiesCfg(mass=segment_mass),
        collision_props=schemas.CollisionPropertiesCfg(
            collision_enabled=collision_enabled,
            contact_offset=contact_offset,
            rest_offset=rest_offset,
        ),
    )
    capsule_cfg.func(link_path, capsule_cfg, translation=translation, orientation=orientation)
    bind_physics_material(f"{link_path}/geometry/mesh", physics_material_path)
    geom_prim = prim_utils.get_prim_at_path(f"{link_path}/geometry")
    if geom_prim is not None:
        prim_utils.set_prim_visibility(geom_prim, False)

    # Visual-only capsule with heavy overlap and round caps for a smooth hose look.
    vis_capsule_height = max(segment_length * 1.65, radius * 0.15)
    sim_utils.CapsuleCfg(
        radius=radius * 0.96,
        height=vis_capsule_height,
        axis="X",
        visual_material=visual_cfg,
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
    ).func(
        f"{link_path}/visual",
        sim_utils.CapsuleCfg(radius=radius * 0.96, height=vis_capsule_height, axis="X"),
    )


def _create_d6_joint(
    stage: Usd.Stage,
    joint_path: str,
    body0_path: str,
    body1_path: str,
    anchor_body0: tuple[float, float, float],
    anchor_body1: tuple[float, float, float],
    bend_limit_deg: float,
    twist_limit_deg: float,
    bend_stiffness: float,
    bend_damping: float,
    twist_stiffness: float,
    twist_damping: float,
    max_joint_effort: float,
    max_joint_velocity: float,
) -> None:
    """Create a passive D6 joint with locked translation and limited rotation."""
    if prim_utils.is_prim_path_valid(joint_path):
        raise ValueError(f"A prim already exists at path: '{joint_path}'.")

    joint = UsdPhysics.Joint.Define(stage, joint_path)
    joint_prim = joint.GetPrim()
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*anchor_body0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*anchor_body1))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    for axis in ("transX", "transY", "transZ"):
        _lock_joint_axis(joint_prim, axis)

    _limit_joint_axis(joint_prim, "rotX", -twist_limit_deg, twist_limit_deg)
    _limit_joint_axis(joint_prim, "rotY", -bend_limit_deg, bend_limit_deg)
    _limit_joint_axis(joint_prim, "rotZ", -bend_limit_deg, bend_limit_deg)

    # Passive bend/twist resistance is applied via Isaac Lab ImplicitActuator (see thick_cable_cfg).
    # USD DriveAPI stiffness would be zeroed on articulation init if actuator stiffness=0.

    physx_joint_api = PhysxSchema.PhysxJointAPI.Apply(joint_prim)
    physx_joint_api.CreateMaxJointVelocityAttr().Set(max_joint_velocity * 180.0 / math.pi)


def _lock_joint_axis(joint_prim: Usd.Prim, axis: str) -> None:
    limit_api = UsdPhysics.LimitAPI.Apply(joint_prim, axis)
    limit_api.CreateLowAttr(1.0)
    limit_api.CreateHighAttr(-1.0)


def _limit_joint_axis(joint_prim: Usd.Prim, axis: str, low_deg: float, high_deg: float) -> None:
    limit_api = UsdPhysics.LimitAPI.Apply(joint_prim, axis)
    limit_api.CreateLowAttr(float(low_deg))
    limit_api.CreateHighAttr(float(high_deg))
