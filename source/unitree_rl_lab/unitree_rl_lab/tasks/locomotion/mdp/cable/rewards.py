# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for cable U-bend manipulation.

All reward functions accept ``(env, ...)`` and return a 1-D tensor of shape
``(num_envs,)``.  Non-finite values are clamped via :func:`_safe_reward`.
"""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from unitree_rl_lab.assets.cable.thick_cable_utils import (
    compute_centerline_chamfer_reward,
    compute_endpoint_reward,
    compute_smoothness_reward,
    get_cable_centerline_points,
    get_cable_endpoint_poses,
)
from unitree_rl_lab.assets.cable.u_shape_targets import U_SHAPE_DEPTH_PRESETS

from .commands import get_cable_bend_command
from .state import get_cable_bend_state

if True:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _safe_reward(reward: torch.Tensor, max_abs: float = 1e6) -> torch.Tensor:
    """Clamp NaN / Inf and extreme finite values for RL stability."""
    out = torch.nan_to_num(reward, nan=0.0, posinf=max_abs, neginf=-max_abs)
    return torch.clamp(out, min=-max_abs, max=max_abs)


def _safe_centerline(cable: Articulation) -> torch.Tensor:
    centerline = get_cable_centerline_points(cable)
    return torch.nan_to_num(centerline, nan=0.0, posinf=0.0, neginf=0.0)


def _sag_depth(points: torch.Tensor) -> torch.Tensor:
    """Sag depth of a polyline: mean endpoint height minus the lowest point."""
    endpoint_mean_z = 0.5 * (points[:, 0, 2] + points[:, -1, 2])
    min_z = points[:, :, 2].min(dim=1).values
    return endpoint_mean_z - min_z


# ---------------------------------------------------------------------------
# Policy-scoped penalty helpers  (exclude locked leg joints)
# ---------------------------------------------------------------------------

_POLICY_JOINT_PATTERNS = [
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw",
    "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw",
    "waist_yaw", "waist_roll", "waist_pitch",
]


def _resolve_policy_joint_ids(env: "ManagerBasedRLEnv", robot: Articulation) -> torch.Tensor:
    if not hasattr(env, "_policy_joint_ids_cache"):
        ids = [
            i for i, name in enumerate(robot.joint_names)
            if any(p in name for p in _POLICY_JOINT_PATTERNS)
        ]
        env._policy_joint_ids_cache = torch.tensor(ids, device=env.device, dtype=torch.long)
    return env._policy_joint_ids_cache


def _safe_clamp(t: torch.Tensor, limit: float = 100.0) -> torch.Tensor:
    t = torch.nan_to_num(t, nan=0.0, posinf=limit, neginf=-limit)
    return torch.clamp(t, min=-limit, max=limit)


# ---------------------------------------------------------------------------
# Shaping rewards
# ---------------------------------------------------------------------------


def bend_shape_reward(
    env: "ManagerBasedRLEnv",
    command_name: str = "cable_bend_u",
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
    temperature: float = 0.10,
) -> torch.Tensor:
    """Chamfer reward toward the commanded U centreline.

    The dominant reward term (30× weight).  With ``temperature = 0.10`` the
    gradient is broad enough to guide the policy from ≈ 35 cm chamfer down to
    < 10 cm, at which point the exponential starts to saturate toward 1.0.
    """
    cable: Articulation = env.scene[cable_cfg.name]
    cmd = get_cable_bend_command(env, command_name)
    return _safe_reward(
        compute_centerline_chamfer_reward(cable, cmd.target_centerline, temperature=temperature)
    )


def bend_endpoints_reward(
    env: "ManagerBasedRLEnv",
    command_name: str = "cable_bend_u",
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
    temperature: float = 0.10,
) -> torch.Tensor:
    """Position-only endpoint alignment reward.

    ``rot_weight=0.0`` because target quaternions are identity while the cable
    endpoint bodies have ≈ 90° rotation from +X to the span direction.
    """
    cable: Articulation = env.scene[cable_cfg.name]
    cmd = get_cable_bend_command(env, command_name)
    return _safe_reward(
        compute_endpoint_reward(cable, cmd.target_endpoint_poses, rot_weight=0.0, temperature=temperature)
    )


def bend_depth_reward(
    env: "ManagerBasedRLEnv",
    command_name: str = "cable_bend_u",
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
    temperature: float = 0.03,
) -> torch.Tensor:
    """Reward matching the commanded U sag depth (dense apex-depth shaping)."""
    cable: Articulation = env.scene[cable_cfg.name]
    cmd = get_cable_bend_command(env, command_name)
    actual = _sag_depth(_safe_centerline(cable))
    target = _sag_depth(cmd.target_centerline)
    return _safe_reward(torch.exp(-torch.abs(actual - target) / temperature))


def bend_smooth_reward(
    env: "ManagerBasedRLEnv",
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
    temperature: float = 0.06,
) -> torch.Tensor:
    """Reward smooth curvature (penalises sharp bends)."""
    cable: Articulation = env.scene[cable_cfg.name]
    return _safe_reward(compute_smoothness_reward(cable, temperature=temperature))


def bend_progress_reward(
    env: "ManagerBasedRLEnv",
    command_name: str = "cable_bend_u",
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
    normalization: float = 0.0001,
) -> torch.Tensor:
    """Reward per-step chamfer improvement since the previous control step."""
    state = get_cable_bend_state(env)
    cable: Articulation = env.scene[cable_cfg.name]
    cmd = get_cable_bend_command(env, command_name)
    current = _safe_reward(
        compute_centerline_chamfer_reward(cable, cmd.target_centerline)
    )

    if not hasattr(state, "_prev_chamfer"):
        state._prev_chamfer = current.detach().clone()
        return torch.zeros(env.num_envs, device=env.device)

    progress = torch.clamp(current - state._prev_chamfer, min=0.0)
    state._prev_chamfer = current.detach().clone()
    return progress / max(normalization, 1e-6)


def bend_stop_reward(
    env: "ManagerBasedRLEnv",
    command_name: str = "cable_bend_u",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
    vel_threshold: float = 0.5,
    chamfer_threshold: float = 0.55,
) -> torch.Tensor:
    """Encourage low joint velocity when shape error is already small."""
    robot: Articulation = env.scene[robot_cfg.name]
    cable: Articulation = env.scene[cable_cfg.name]
    cmd = get_cable_bend_command(env, command_name)

    chamfer = _safe_reward(
        compute_centerline_chamfer_reward(cable, cmd.target_centerline)
    )
    joint_vel = torch.nan_to_num(robot.data.joint_vel, nan=0.0, posinf=0.0, neginf=0.0)
    joint_vel_norm = torch.norm(joint_vel, dim=-1)

    near_target = torch.sigmoid((chamfer - chamfer_threshold) * 20.0)
    low_vel = torch.exp(-joint_vel_norm / vel_threshold)
    return _safe_reward(near_target * low_vel)


def bend_settle_reward(
    env: "ManagerBasedRLEnv",
    command_name: str = "cable_bend_u",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
    chamfer_threshold: float = 0.70,
    vel_threshold: float = 0.3,
) -> torch.Tensor:
    """Binary bonus when the hose matches the target AND arms are nearly still."""
    robot: Articulation = env.scene[robot_cfg.name]
    cable: Articulation = env.scene[cable_cfg.name]
    cmd = get_cable_bend_command(env, command_name)

    chamfer = compute_centerline_chamfer_reward(cable, cmd.target_centerline)
    joint_vel = torch.nan_to_num(robot.data.joint_vel, nan=0.0, posinf=0.0, neginf=0.0)
    joint_vel_norm = torch.norm(joint_vel, dim=-1)

    good_shape = chamfer >= chamfer_threshold
    still = joint_vel_norm < vel_threshold
    return (good_shape & still).float()


def bend_contact_stability(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*rubber_hand"),
) -> torch.Tensor:
    """Reward stable bilateral hand contact force."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    body_ids = sensor_cfg.body_ids
    if body_ids is None:
        return torch.zeros(env.num_envs, device=env.device)

    forces = contact_sensor.data.net_forces_w[:, body_ids]
    mag = torch.norm(forces, dim=-1)
    if mag.shape[1] < 2:
        return torch.zeros(env.num_envs, device=env.device)

    left_ok = torch.sigmoid(mag[:, 0] - 1.0)
    right_ok = torch.sigmoid(mag[:, 1] - 1.0)
    return left_ok * right_ok


def hand_proximity_reward(
    env: "ManagerBasedRLEnv",
    command_name: str = "cable_bend_u",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    temperature: float = 0.05,
) -> torch.Tensor:
    """Reward hands being at the correct span for the commanded U-depth."""
    robot: Articulation = env.scene[robot_cfg.name]
    cmd = get_cable_bend_command(env, command_name)

    left_ids, _ = robot.find_bodies("left_rubber_hand")
    right_ids, _ = robot.find_bodies("right_rubber_hand")
    left_pos = robot.data.body_pos_w[:, left_ids[0]]
    right_pos = robot.data.body_pos_w[:, right_ids[0]]
    hand_dist = torch.norm(right_pos - left_pos, dim=-1)

    depth_ids = cmd.depth_ids
    target_spans = torch.zeros(env.num_envs, device=env.device)
    for pid in range(len(U_SHAPE_DEPTH_PRESETS)):
        mask = depth_ids == pid
        if mask.any():
            target_spans[mask] = U_SHAPE_DEPTH_PRESETS[pid].u_span_m

    span_error = torch.abs(hand_dist - target_spans)
    return _safe_reward(torch.exp(-span_error / temperature))


def arm_symmetry_reward(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward left/right arm mirror symmetry.

    Pairs (e.g. left_shoulder_pitch ↔ right_shoulder_pitch) should have equal
    magnitude but opposite sign for roll/yaw axes and equal sign for pitch axes.
    This effectively halves the search space from 17D to ≈ 9D mirror pairs.
    """
    robot: Articulation = env.scene[robot_cfg.name]

    pairs = [
        ("left_shoulder_pitch_joint", "right_shoulder_pitch_joint", 1.0),
        ("left_shoulder_roll_joint",  "right_shoulder_roll_joint", -1.0),
        ("left_shoulder_yaw_joint",   "right_shoulder_yaw_joint", -1.0),
        ("left_elbow_joint",          "right_elbow_joint", 1.0),
        ("left_wrist_roll_joint",     "right_wrist_roll_joint", -1.0),
        ("left_wrist_pitch_joint",    "right_wrist_pitch_joint", 1.0),
        ("left_wrist_yaw_joint",      "right_wrist_yaw_joint", -1.0),
    ]

    total_err = torch.zeros(env.num_envs, device=env.device)
    for left_name, right_name, sign in pairs:
        left_ids, _ = robot.find_joints(left_name)
        right_ids, _ = robot.find_joints(right_name)
        if left_ids and right_ids:
            left_pos = robot.data.joint_pos[:, left_ids[0]]
            right_pos = robot.data.joint_pos[:, right_ids[0]]
            err = left_pos - sign * right_pos
            total_err = total_err + err * err

    return _safe_reward(torch.exp(-total_err / 0.1))


# ---------------------------------------------------------------------------
# Stage-1 rewards  (zeroed in bend-only curriculum)
# ---------------------------------------------------------------------------


def reach_hand_to_cable_end(
    env: "ManagerBasedRLEnv",
    side: str = "left",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
    temperature: float = 0.05,
) -> torch.Tensor:
    """Exponential reward for hand proximity to cable endpoint."""
    robot: Articulation = env.scene[robot_cfg.name]
    cable: Articulation = env.scene[cable_cfg.name]
    hand_name = "left_rubber_hand" if side == "left" else "right_rubber_hand"
    hand_ids, _ = robot.find_bodies(hand_name)
    hand_pos = robot.data.body_pos_w[:, hand_ids[0]]
    start_pos, _, end_pos, _ = get_cable_endpoint_poses(cable)
    target = start_pos if side == "left" else end_pos
    dist = torch.norm(hand_pos - target, dim=-1)
    return torch.exp(-dist / temperature)


def lift_cable_reward(
    env: "ManagerBasedRLEnv",
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
    lift_margin: float = 0.02,
) -> torch.Tensor:
    """Reward when cable centreline clears the table."""
    state = get_cable_bend_state(env)
    cable: Articulation = env.scene[cable_cfg.name]
    centerline = get_cable_centerline_points(cable)
    min_z = centerline[:, :, 2].min(dim=-1).values
    target_z = state.table_height + lift_margin
    return torch.clamp((min_z - target_z) / 0.15, 0.0, 1.0)


def dual_grasp_reward(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
    grasp_threshold: float = 0.08,
) -> torch.Tensor:
    """Reward when both hands are near cable endpoints."""
    left = reach_hand_to_cable_end(
        env, side="left", robot_cfg=robot_cfg, cable_cfg=cable_cfg, temperature=grasp_threshold
    )
    right = reach_hand_to_cable_end(
        env, side="right", robot_cfg=robot_cfg, cable_cfg=cable_cfg, temperature=grasp_threshold
    )
    return left * right


def update_success_counters(
    env: "ManagerBasedRLEnv",
    command_name: str = "cable_bend_u",
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
    success_threshold: float = 0.85,
) -> None:
    """Update consecutive success-step counters (called from termination)."""
    state = get_cable_bend_state(env)
    cable: Articulation = env.scene[cable_cfg.name]
    cmd = get_cable_bend_command(env, command_name)
    chamfer = compute_centerline_chamfer_reward(cable, cmd.target_centerline)
    good = chamfer >= success_threshold
    if state.curriculum_stage == "full":
        good = good & state.cable_attached
    state.success_steps = torch.where(
        good, state.success_steps + 1, torch.zeros_like(state.success_steps)
    )


# ---------------------------------------------------------------------------
# Policy-scoped penalties  (exclude locked leg joints)
# ---------------------------------------------------------------------------


def policy_joint_vel_l2(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L2 penalty on policy-controlled joint velocities only."""
    robot: Articulation = env.scene[robot_cfg.name]
    ids = _resolve_policy_joint_ids(env, robot)
    vel = _safe_clamp(robot.data.joint_vel[:, ids], limit=50.0)
    return torch.sum(vel * vel, dim=-1)


def policy_joint_acc_l2(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L2 penalty on policy-controlled joint accelerations only."""
    robot: Articulation = env.scene[robot_cfg.name]
    ids = _resolve_policy_joint_ids(env, robot)
    acc = _safe_clamp(robot.data.joint_acc[:, ids], limit=500.0)
    return torch.sum(acc * acc, dim=-1)


def policy_dof_pos_limits(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise policy joints exceeding their soft position limits."""
    robot: Articulation = env.scene[robot_cfg.name]
    ids = _resolve_policy_joint_ids(env, robot)
    pos = _safe_clamp(robot.data.joint_pos[:, ids], limit=10.0)
    lower = robot.data.soft_joint_pos_limits[0, ids, 0]
    upper = robot.data.soft_joint_pos_limits[0, ids, 1]
    out_of_lower = torch.clamp(lower - pos, min=0.0)
    out_of_upper = torch.clamp(pos - upper, min=0.0)
    return torch.sum(out_of_lower + out_of_upper, dim=-1)


def policy_energy(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Energy penalty on policy joints only (|torque × vel|)."""
    robot: Articulation = env.scene[robot_cfg.name]
    ids = _resolve_policy_joint_ids(env, robot)
    torque = _safe_clamp(robot.data.applied_torque[:, ids], limit=100.0)
    vel = _safe_clamp(robot.data.joint_vel[:, ids], limit=50.0)
    return torch.sum(torch.abs(torque * vel), dim=-1)


def lower_body_deviation(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise leg joints drifting from nominal standing offset.

    Currently zero-weighted in the production config; retained for potential
    future curriculum stages.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    leg_patterns = ["hip", "knee", "ankle"]
    leg_ids = [i for i, name in enumerate(robot.joint_names) if any(p in name for p in leg_patterns)]
    if not leg_ids:
        return torch.zeros(env.num_envs, device=env.device)
    leg_ids_t = torch.tensor(leg_ids, device=env.device)
    pos = robot.data.joint_pos[:, leg_ids_t]
    pos = torch.nan_to_num(pos, nan=0.0, posinf=10.0, neginf=-10.0)
    pos = torch.clamp(pos, min=-10.0, max=10.0)
    return torch.sum(pos * pos, dim=-1)
