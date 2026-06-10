# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for cable U-bend manipulation.

All terms return tensors on the environment device.  Non-finite values are
clamped for RL stability.
"""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from unitree_rl_lab.assets.cable.thick_cable_utils import (
    compute_centerline_chamfer_reward,
    get_cable_centerline_points,
    get_cable_endpoint_poses,
)

from .commands import get_cable_bend_command
from .state import get_cable_bend_state

if True:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

# 17-DOF policy joints (14 arms + 3 waist) — must match the action space.
_POLICY_JOINT_NAMES: list[str] = [
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_shoulder_roll_joint",  "right_shoulder_roll_joint",
    "left_shoulder_yaw_joint",   "right_shoulder_yaw_joint",
    "left_elbow_joint",          "right_elbow_joint",
    "left_wrist_roll_joint",     "right_wrist_roll_joint",
    "left_wrist_pitch_joint",    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",      "right_wrist_yaw_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
]


def _sanitize(tensor: torch.Tensor, clip: float = 100.0) -> torch.Tensor:
    out = torch.nan_to_num(tensor, nan=0.0, posinf=clip, neginf=-clip)
    return torch.clamp(out, min=-clip, max=clip)


def _resolve_policy_joint_ids(robot: Articulation) -> torch.Tensor:
    ids: list[int] = []
    for name in _POLICY_JOINT_NAMES:
        jid, _ = robot.find_joints(name)
        if jid:
            ids.append(jid[0])
    return torch.tensor(ids, device=robot.device, dtype=torch.long)


def _resolve_torso_body_name(robot_cfg: SceneEntityCfg) -> str:
    if robot_cfg.body_names is None:
        return "torso_link"
    if isinstance(robot_cfg.body_names, str):
        return robot_cfg.body_names
    return robot_cfg.body_names[0]


def _points_in_body_frame(
    points_w: torch.Tensor, body_pos_w: torch.Tensor, body_quat_w: torch.Tensor
) -> torch.Tensor:
    """Transform world points to body frame.  ``points_w``: (N, K, 3)."""
    num_envs, num_pts, _ = points_w.shape
    rel = points_w - body_pos_w.unsqueeze(1)
    quat_norm = math_utils.normalize(body_quat_w)
    quat_inv = math_utils.quat_inv(quat_norm)
    quat_rep = quat_inv.unsqueeze(1).expand(-1, num_pts, -1).reshape(num_envs * num_pts, 4)
    rel_flat = rel.reshape(num_envs * num_pts, 3)
    body_flat = math_utils.quat_apply(quat_rep, rel_flat)
    return _sanitize(body_flat.reshape(num_envs, num_pts, 3))


def _get_robot_and_cable(
    env: "ManagerBasedRLEnv", robot_cfg: SceneEntityCfg, cable_cfg: SceneEntityCfg
) -> tuple[Articulation, Articulation, int]:
    robot: Articulation = env.scene[robot_cfg.name]
    cable: Articulation = env.scene[cable_cfg.name]
    torso_ids, _ = robot.find_bodies(_resolve_torso_body_name(robot_cfg))
    return robot, cable, torso_ids[0]


# ---------------------------------------------------------------------------
# Policy observations  (used by the actor)
# ---------------------------------------------------------------------------


def joint_pos_policy(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Policy-controlled joint positions (17-DOF arms + waist)."""
    robot: Articulation = env.scene[robot_cfg.name]
    if not hasattr(env, "_policy_joint_ids"):
        env._policy_joint_ids = _resolve_policy_joint_ids(robot)
    return robot.data.joint_pos[:, env._policy_joint_ids]


def joint_vel_policy(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Policy-controlled joint velocities (17-DOF arms + waist)."""
    robot: Articulation = env.scene[robot_cfg.name]
    if not hasattr(env, "_policy_joint_ids"):
        env._policy_joint_ids = _resolve_policy_joint_ids(robot)
    return robot.data.joint_vel[:, env._policy_joint_ids]


def cable_bend_command(
    env: "ManagerBasedRLEnv",
    command_name: str = "cable_bend_u",
) -> torch.Tensor:
    """One-hot U-depth command, shape ``(num_envs, NUM_PRESETS)``."""
    return get_cable_bend_command(env, command_name).command


def cable_endpoints_body(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
) -> torch.Tensor:
    """Cable start/end positions in torso frame, shape ``(num_envs, 6)``."""
    robot, cable, torso_id = _get_robot_and_cable(env, robot_cfg, cable_cfg)
    centerline = _sanitize(get_cable_centerline_points(cable))
    pts = torch.stack([centerline[:, 0], centerline[:, -1]], dim=1)
    torso_pos = robot.data.body_pos_w[:, torso_id]
    torso_quat = robot.data.body_quat_w[:, torso_id]
    body_pts = _points_in_body_frame(pts, torso_pos, torso_quat)
    return body_pts.reshape(env.num_envs, 6)


def cable_centerline_sparse(
    env: "ManagerBasedRLEnv",
    num_points: int = 8,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
) -> torch.Tensor:
    """Uniformly subsampled centreline in torso frame, shape ``(num_envs, num_points * 3)``."""
    robot, cable, torso_id = _get_robot_and_cable(env, robot_cfg, cable_cfg)
    centerline = _sanitize(get_cable_centerline_points(cable))
    n_seg = centerline.shape[1]
    indices = (
        torch.arange(n_seg, device=centerline.device)
        if n_seg <= num_points
        else torch.linspace(0, n_seg - 1, num_points, device=centerline.device).long()
    )
    sampled = centerline[:, indices, :]
    torso_pos = robot.data.body_pos_w[:, torso_id]
    torso_quat = robot.data.body_quat_w[:, torso_id]
    body_pts = _points_in_body_frame(sampled, torso_pos, torso_quat)
    return body_pts.reshape(env.num_envs, -1)


def target_centerline_flat(
    env: "ManagerBasedRLEnv",
    command_name: str = "cable_bend_u",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
) -> torch.Tensor:
    """Target U polyline in torso frame, flattened ``(num_envs, num_target * 3)``."""
    cmd = get_cable_bend_command(env, command_name)
    robot: Articulation = env.scene[robot_cfg.name]
    torso_ids, _ = robot.find_bodies(_resolve_torso_body_name(robot_cfg))
    torso_pos = robot.data.body_pos_w[:, torso_ids[0]]
    torso_quat = robot.data.body_quat_w[:, torso_ids[0]]
    body_pts = _points_in_body_frame(cmd.target_centerline, torso_pos, torso_quat)
    return body_pts.reshape(env.num_envs, -1)


def hand_pos_body(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
) -> torch.Tensor:
    """Left/right hand positions in torso frame, shape ``(num_envs, 6)``."""
    robot: Articulation = env.scene[robot_cfg.name]
    torso_ids, _ = robot.find_bodies(_resolve_torso_body_name(robot_cfg))
    left_ids, _ = robot.find_bodies("left_rubber_hand")
    right_ids, _ = robot.find_bodies("right_rubber_hand")
    pts = torch.stack(
        [robot.data.body_pos_w[:, left_ids[0]], robot.data.body_pos_w[:, right_ids[0]]], dim=1
    )
    torso_pos = robot.data.body_pos_w[:, torso_ids[0]]
    torso_quat = robot.data.body_quat_w[:, torso_ids[0]]
    body_pts = _points_in_body_frame(pts, torso_pos, torso_quat)
    return body_pts.reshape(env.num_envs, 6)


def hand_contact_force(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*rubber_hand"),
) -> torch.Tensor:
    """Contact force magnitude per hand, shape ``(num_envs, 2)``."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    body_ids = sensor_cfg.body_ids
    if body_ids is None or len(body_ids) == 0:
        return torch.zeros(env.num_envs, 2, device=env.device)
    forces = contact_sensor.data.net_forces_w[:, body_ids]
    mag = torch.norm(forces, dim=-1)
    if mag.shape[1] < 2:
        pad = torch.zeros(env.num_envs, 2 - mag.shape[1], device=env.device)
        mag = torch.cat([mag, pad], dim=-1)
    elif mag.shape[1] > 2:
        mag = mag[:, :2]
    return _sanitize(mag)


def cable_centerline_full_body(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
) -> torch.Tensor:
    """Full centreline in torso frame (critic-privileged), flattened."""
    robot, cable, torso_id = _get_robot_and_cable(env, robot_cfg, cable_cfg)
    centerline = _sanitize(get_cable_centerline_points(cable))
    torso_pos = robot.data.body_pos_w[:, torso_id]
    torso_quat = robot.data.body_quat_w[:, torso_id]
    body_pts = _points_in_body_frame(centerline, torso_pos, torso_quat)
    return body_pts.reshape(env.num_envs, -1)


def cable_shape_error(
    env: "ManagerBasedRLEnv",
    command_name: str = "cable_bend_u",
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
) -> torch.Tensor:
    """Scalar shape error ``1.0 - chamfer_score`` in ``[0, 1]``."""
    cable: Articulation = env.scene[cable_cfg.name]
    cmd = get_cable_bend_command(env, command_name)
    chamfer = compute_centerline_chamfer_reward(cable, cmd.target_centerline)
    return (1.0 - torch.clamp(chamfer, 0.0, 1.0)).unsqueeze(-1)


def phase_progress(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Normalised episode progress ``[0, 1]``, shape ``(num_envs, 1)``."""
    max_len = max(float(env.max_episode_length), 1.0)
    return (env.episode_length_buf.float() / max_len).unsqueeze(1)


def grasp_progress(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
    grasp_threshold: float = 0.08,
) -> torch.Tensor:
    """Per-hand grasp proxy ``[0, 1]``, shape ``(num_envs, 2)``."""
    state = get_cable_bend_state(env)
    if state.curriculum_stage == "bend":
        return torch.ones(env.num_envs, 2, device=env.device)

    robot: Articulation = env.scene[robot_cfg.name]
    cable: Articulation = env.scene[cable_cfg.name]
    left_ids, _ = robot.find_bodies("left_rubber_hand")
    right_ids, _ = robot.find_bodies("right_rubber_hand")
    left_pos = robot.data.body_pos_w[:, left_ids[0]]
    right_pos = robot.data.body_pos_w[:, right_ids[0]]
    centerline = _sanitize(get_cable_centerline_points(cable))
    start_pos, end_pos = centerline[:, 0], centerline[:, -1]
    left_dist = torch.norm(left_pos - start_pos, dim=-1)
    right_dist = torch.norm(right_pos - end_pos, dim=-1)
    left_grasp = torch.exp(-left_dist / grasp_threshold)
    right_grasp = torch.exp(-right_dist / grasp_threshold)
    return torch.stack([left_grasp, right_grasp], dim=-1)
