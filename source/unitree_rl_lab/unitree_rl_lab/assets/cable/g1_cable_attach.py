# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Attach thick-cable endpoints to Unitree G1 hand links via USD fixed joints."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import isaacsim.core.utils.prims as prim_utils
from pxr import Gf, UsdGeom

import isaaclab.utils.math as math_utils

from .thick_cable_builder import build_thick_cable_metadata, create_body_point_weld
from .thick_cable_cfg import ThickCableCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv

LEFT_HAND_BODY_NAME = "left_rubber_hand"
RIGHT_HAND_BODY_NAME = "right_rubber_hand"

# Per-env attachment joint paths already created (avoid duplicate FixedJoint prims).
_ATTACHED_ENV_IDS: set[int] = set()


def _env_root(env_id: int) -> str:
    return f"/World/envs/env_{env_id}"


def _resolve_hand_body_paths(robot: Articulation, env_id: int) -> tuple[str, str]:
    env_root = _env_root(env_id)
    left_ids, left_names = robot.find_bodies(LEFT_HAND_BODY_NAME)
    right_ids, right_names = robot.find_bodies(RIGHT_HAND_BODY_NAME)
    if not left_ids or not right_ids:
        raise RuntimeError(
            f"Could not find hand bodies on G1 (env {env_id}). "
            f"Found left={left_names}, right={right_names}. "
            "Expected left_rubber_hand and right_rubber_hand in URDF."
        )
    # body_names entries are short names; full prim paths live under the articulation root.
    robot_root = env_root + "/Robot"
    return f"{robot_root}/{left_names[0]}", f"{robot_root}/{right_names[0]}"


def _cable_endpoint_local_offsets(segment_length: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    half = 0.5 * segment_length
    return (-half, 0.0, 0.0), (half, 0.0, 0.0)


def _quat_wxyz_align_x_to_direction(direction: torch.Tensor) -> torch.Tensor:
    """Return (w, x, y, z) quaternion rotating +X to ``direction`` (batched N,3)."""
    ref = torch.tensor([1.0, 0.0, 0.0], device=direction.device, dtype=direction.dtype).expand(direction.shape[0], -1)
    dir_n = math_utils.normalize(direction)
    dot = torch.sum(ref * dir_n, dim=-1).clamp(-1.0, 1.0)
    cross = torch.cross(ref, dir_n, dim=-1)
    cross_norm = torch.linalg.norm(cross, dim=-1)
    quat = torch.zeros(direction.shape[0], 4, device=direction.device, dtype=direction.dtype)
    quat[:, 0] = 1.0
    general = cross_norm > 1e-6
    if torch.any(general):
        axis = cross[general] / cross_norm[general].unsqueeze(-1)
        angle = torch.acos(dot[general])
        quat[general] = math_utils.quat_from_angle_axis(angle, axis)
    opposite = dot < -0.999
    if torch.any(opposite):
        axis_y = torch.tensor([0.0, 1.0, 0.0], device=direction.device, dtype=direction.dtype).expand(int(opposite.sum()), -1)
        quat[opposite] = math_utils.quat_from_angle_axis(
            torch.full((int(opposite.sum()),), 3.14159265, device=direction.device, dtype=direction.dtype),
            axis_y,
        )
    return quat


def _find_root_idx_in_chain(body_names: list[str]) -> int:
    """Return the 0-based index of the articulation root body within the sequential
    seg_00..seg_N chain, inferred from the body name (e.g. 'seg_05' → 5)."""
    root_name = body_names[0]  # articulation root is always first entry
    try:
        return int(root_name.split("_")[-1])
    except (ValueError, IndexError):
        raise RuntimeError(
            f"Cannot parse root index from articulation root body name: '{root_name}'. "
            f"Expected format 'seg_NN'. Body names: {body_names}"
        )


def align_cable_between_hands(
    robot: Articulation,
    cable: Articulation,
    cable_cfg: ThickCableCfg,
    env_ids: torch.Tensor | slice,
) -> None:
    """Place the cable root pose along the line between G1 hand bodies.

    The cable is built as a linear chain seg_00–seg_01–…–seg_N.  Because the GPU
    articulation solver may re-root the tree (e.g. to seg_05 for a 12-segment cable),
    we look up the *actual* root body index and compute the root world pose so that
    both cable endpoints land as close as possible to their corresponding hands.
    """
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(robot.num_instances, device=robot.device, dtype=torch.int64)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, device=robot.device, dtype=torch.int64)

    left_ids, _ = robot.find_bodies(LEFT_HAND_BODY_NAME)
    right_ids, _ = robot.find_bodies(RIGHT_HAND_BODY_NAME)
    left_pos = robot.data.body_pos_w[env_ids, left_ids[0]]
    right_pos = robot.data.body_pos_w[env_ids, right_ids[0]]
    direction = right_pos - left_pos
    span = torch.linalg.norm(direction, dim=-1, keepdim=True).clamp(min=1e-4)
    dir_n = direction / span

    seg_len = cable_cfg.segment_length
    num_seg = cable_cfg.num_segments
    root_idx = _find_root_idx_in_chain(list(cable.body_names))

    # Local offsets from root to the two cable endpoints (straight, joints zeroed).
    # seg_00 is at local  -root_idx * seg_len  along the chain (+X direction).
    # seg_last is at local  (N-1-root_idx) * seg_len.
    left_end_local  = -root_idx * seg_len
    right_end_local = (num_seg - 1 - root_idx) * seg_len

    # World root position that makes seg_00 sit on the left hand:
    #   root = left_hand  -  dir_n  * left_end_local    (because +X → dir_n)
    root_from_left  = left_pos  - dir_n * left_end_local   # left_end_local is negative → moves root right
    # World root position that makes seg_last sit on the right hand:
    root_from_right = right_pos - dir_n * right_end_local

    # Least-squares compromise when the tree is asymmetric.
    root_pos = 0.5 * (root_from_left + root_from_right)

    quat = _quat_wxyz_align_x_to_direction(dir_n)
    root_state = cable.data.default_root_state[env_ids].clone()
    root_state[:, :3] = root_pos
    root_state[:, 3:7] = quat
    root_state[:, 7:] = 0.0
    cable.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    cable.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

    joint_pos = cable.data.default_joint_pos[env_ids].clone().zero_()
    joint_vel = cable.data.default_joint_vel[env_ids].clone().zero_()
    cable.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    # Also update the cached data buffer so write_data_to_sim won't overwrite.
    cable.data.joint_pos[env_ids] = joint_pos
    cable.data.joint_vel[env_ids] = joint_vel


def _world_point_in_body_frame(
    body_pos: torch.Tensor, body_quat: torch.Tensor, world_pos: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    local_pos, local_quat = math_utils.subtract_frame_transforms(
        body_pos, body_quat, world_pos, torch.tensor([1.0, 0.0, 0.0, 0.0], device=body_pos.device).expand(body_pos.shape[0], -1)
    )
    return local_pos, local_quat


def _attachment_exists(env_id: int) -> bool:
    anchor_root = f"{_env_root(env_id)}/CableAnchors/left_hand_to_cable_start"
    return prim_utils.is_prim_path_valid(anchor_root)


def _remove_attachment_for_env(env_id: int) -> None:
    """Delete welded joints so the next reset can re-attach at the current hand pose."""
    anchor_root = f"{_env_root(env_id)}/CableAnchors"
    if prim_utils.is_prim_path_valid(anchor_root):
        prim_utils.delete_prim(anchor_root)
    _ATTACHED_ENV_IDS.discard(env_id)


def _create_attachment_for_env(
    env_id: int,
    cable_cfg: ThickCableCfg,
) -> None:
    """Build-time weld: pin cable seg_00 start-tip to the left hand and seg_last end-tip to the
    right hand using point welds (translation locked, rotation free). MUST run before
    ``sim.reset()`` (prestartup) so the GPU physics view includes these joints.
    """
    env_root = _env_root(env_id)
    num_seg = cable_cfg.num_segments
    half_seg = 0.5 * cable_cfg.segment_length
    left_hand = f"{env_root}/Robot/{LEFT_HAND_BODY_NAME}"
    right_hand = f"{env_root}/Robot/{RIGHT_HAND_BODY_NAME}"
    start_link = f"{env_root}/Cable/segments/seg_00"
    end_link = f"{env_root}/Cable/segments/seg_{num_seg - 1:02d}"

    anchor_root = f"{env_root}/CableAnchors"
    create_body_point_weld(
        f"{anchor_root}/left_hand_to_cable_start",
        left_hand,
        start_link,
        (0.0, 0.0, 0.0),
        (-half_seg, 0.0, 0.0),
    )
    create_body_point_weld(
        f"{anchor_root}/right_hand_to_cable_end",
        right_hand,
        end_link,
        (0.0, 0.0, 0.0),
        (half_seg, 0.0, 0.0),
    )


def _prestartup_align_cable_usd(env_id: int, cable_cfg: ThickCableCfg) -> None:
    """Position the cable USD prim to span between the robot's CURRENT (rest-pose) hands.

    At prestartup the robot is still at its spawned URDF pose, so we read the hand world
    transforms straight from USD and lay the straight cable between them. This makes the point
    welds (created right after) start satisfied, so PhysX does not explode at ``sim.reset()``.
    """
    env_root = _env_root(env_id)
    stage = prim_utils.get_current_stage()
    cache = UsdGeom.XformCache()

    def _world_pos(path: str) -> Gf.Vec3d:
        return cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path)).ExtractTranslation()

    left = _world_pos(f"{env_root}/Robot/{LEFT_HAND_BODY_NAME}")
    right = _world_pos(f"{env_root}/Robot/{RIGHT_HAND_BODY_NAME}")
    cable_prim_path = f"{env_root}/Cable"
    env_origin = _world_pos(env_root)

    direction = Gf.Vec3d(right[0] - left[0], right[1] - left[1], right[2] - left[2])
    span = direction.GetLength()
    if span < 1e-4:
        return
    dir_n = direction / span
    half_seg = 0.5 * cable_cfg.segment_length
    # seg_00 center sits half a segment in from the left hand along the hand axis.
    root_world = Gf.Vec3d(left[0], left[1], left[2]) + dir_n * half_seg
    root_local = root_world - env_origin

    # Quaternion mapping +X to dir_n (rotation about the axis X x dir).
    x_axis = Gf.Vec3d(1.0, 0.0, 0.0)
    dot = max(-1.0, min(1.0, x_axis * dir_n))
    if dot > 0.999999:
        rot = Gf.Quatd(1.0, 0.0, 0.0, 0.0)
    elif dot < -0.999999:
        rot = Gf.Quatd(0.0, 0.0, 0.0, 1.0)
    else:
        axis = Gf.Cross(x_axis, dir_n).GetNormalized()
        angle = math.acos(dot)
        s = math.sin(0.5 * angle)
        rot = Gf.Quatd(math.cos(0.5 * angle), axis[0] * s, axis[1] * s, axis[2] * s)

    cable_prim = stage.GetPrimAtPath(cable_prim_path)
    xformable = UsdGeom.Xformable(cable_prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(root_local)
    xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(rot)


def weld_cable_endpoints_buildtime(
    env: ManagerBasedRLEnv,
    cable_cfg: ThickCableCfg,
) -> None:
    """Align each cable to its env's current hand positions, then create the hand<->cable point
    welds for all envs. Intended to run in a ``prestartup`` event (before the GPU physics view).
    """
    for env_id in range(env.num_envs):
        if _attachment_exists(env_id):
            continue
        _prestartup_align_cable_usd(env_id, cable_cfg)
        _create_attachment_for_env(env_id, cable_cfg)
        _ATTACHED_ENV_IDS.add(env_id)


def _sync_scene_kinematics(env: ManagerBasedRLEnv) -> None:
    """Refresh body poses before aligning or welding the cable."""
    env.scene.write_data_to_sim()
    env.sim.forward()


def attach_cable_endpoints_to_g1(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    robot_cfg_name: str = "robot",
    cable_cfg_name: str = "cable",
    cable_hold_cfg: ThickCableCfg | None = None,
) -> None:
    """Align cable between hands and weld endpoints to ``left_rubber_hand`` / ``right_rubber_hand``."""
    robot_entity = robot_cfg_name
    cable_entity = cable_cfg_name
    robot: Articulation = env.scene[robot_entity]
    cable: Articulation = env.scene[cable_entity]

    if cable_hold_cfg is None:
        from .thick_cable_cfg import STIFF_FLAT_TUBE_CFG

        cable_hold_cfg = STIFF_FLAT_TUBE_CFG

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int64)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, device=env.device, dtype=torch.int64)

    # Align cable root pose and zero joints so that the prestartup point welds are satisfied.
    # No extra sim.forward() here — the normal post-reset write_data_to_sim path will sync.
    align_cable_between_hands(robot, cable, cable_hold_cfg, env_ids)


def reset_attachment_cache() -> None:
    """Clear attachment cache (for repeated scene rebuilds in one process)."""
    _ATTACHED_ENV_IDS.clear()
