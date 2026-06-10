"""Kinematic constant-curvature cable drive (zero policy dims).

Each physics substep this action term lays the cable as a smooth circular arc whose two ends sit
exactly in the robot's two hands. The arc curvature is solved so the (fixed) cable arc length
spans the (varying) hand-to-hand distance: as the hands come together the arc deepens into a U,
as they spread it straightens. This is a GPU-safe, replication-friendly alternative to a rigid
cross-articulation weld (which explodes the PhysX GPU solver), and keeps the cable held at both
hands at all times so the bend task is learnable.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm
from isaaclab.managers.action_manager import ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _solve_half_angle(ratio: torch.Tensor, iters: int = 30) -> torch.Tensor:
    """Solve ``sin(u)/u = ratio`` for u in (0, pi) via bisection (vectorized). ratio in (0,1)."""
    lo = torch.full_like(ratio, 1e-4)
    hi = torch.full_like(ratio, 3.14159265 - 1e-3)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        val = torch.sin(mid) / mid
        # sin(u)/u is monotonically decreasing on (0, pi): val>ratio -> need larger u.
        too_high = val > ratio
        lo = torch.where(too_high, mid, lo)
        hi = torch.where(too_high, hi, mid)
    return 0.5 * (lo + hi)


class CableArcDriveAction(ActionTerm):
    """Drive the cable into a circular arc between the two hands every physics substep."""

    cfg: CableArcDriveActionCfg

    def __init__(self, cfg: CableArcDriveActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._robot: Articulation = env.scene[cfg.robot_name]
        self._cable: Articulation = env.scene[cfg.cable_name]

        left_ids, _ = self._robot.find_bodies(cfg.left_hand_body)
        right_ids, _ = self._robot.find_bodies(cfg.right_hand_body)
        self._left_id = left_ids[0]
        self._right_id = right_ids[0]

        names = self._cable.joint_names
        bend = [i for i, n in enumerate(names) if n.endswith(cfg.bend_dof_suffix)]
        self._bend_ids = torch.tensor(bend, device=self.device, dtype=torch.long)
        self._num_bend = len(bend)

        # PhysX roots the floating cable articulation at its most central link, so the pose written
        # by write_root_pose_to_sim is that segment's pose - identify which one (e.g. "seg_11").
        self._root_seg_idx = int(self._cable.body_names[0].split("_")[-1])
        self._num_seg = int(cfg.num_segments)
        self._seg_len = float(cfg.total_length) / self._num_seg
        self._arc_len = float(cfg.total_length)
        self._bend_sign = float(cfg.bend_sign)
        self._down = torch.tensor([[0.0, 0.0, -1.0]], device=self.device)
        self._zeros_joint = torch.zeros(self.num_envs, self._cable.num_joints, device=self.device)

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return torch.empty(self.num_envs, 0, device=self.device)

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.raw_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        del actions

    def apply_actions(self) -> None:
        left = self._robot.data.body_pos_w[:, self._left_id, :]
        right = self._robot.data.body_pos_w[:, self._right_id, :]

        chord = right - left
        span = torch.linalg.norm(chord, dim=-1, keepdim=True).clamp(min=1e-4)
        c_hat = chord / span

        down = self._down.expand_as(c_hat)
        sag = down - (down * c_hat).sum(-1, keepdim=True) * c_hat
        sag = math_utils.normalize(sag)
        normal = math_utils.normalize(torch.cross(c_hat, sag, dim=-1))  # arc-plane normal

        # Circular arc of fixed length S spanning the chord: solve sin(u)/u = span/S for the half
        # angle u (clamped so it stays straight when the hands are farther apart than S).
        ratio = (span.squeeze(-1) / self._arc_len).clamp(min=1e-3, max=0.9999)
        u = _solve_half_angle(ratio).clamp(min=0.02)  # (N,)
        radius = self._arc_len / (2.0 * u)  # (N,)

        mid = 0.5 * (left + right)
        # Arc-angle of the root segment's center, measured from the chord midpoint (0 = apex).
        frac = (self._root_seg_idx + 0.5) / self._num_seg  # arc-length fraction from left end
        phi = (u * (2.0 * frac - 1.0)).unsqueeze(-1)  # (N,1)
        center = mid - sag * (radius * torch.cos(u)).unsqueeze(-1)

        root_pos = center + radius.unsqueeze(-1) * (torch.cos(phi) * sag + torch.sin(phi) * c_hat)
        tangent = -torch.sin(phi) * sag + torch.cos(phi) * c_hat
        binormal = torch.cross(tangent, normal, dim=-1)

        # Rotation whose columns map local (X, Y, Z) -> (tangent, normal, binormal).
        rot = torch.stack([tangent, normal, binormal], dim=-1)
        quat = math_utils.quat_from_matrix(rot)

        root_pose = torch.cat([root_pos, quat], dim=-1)
        self._cable.write_root_pose_to_sim(root_pose)
        self._cable.write_root_velocity_to_sim(torch.zeros(self.num_envs, 6, device=self.device))

        # Constant per-joint bend so the chain follows the arc (total turn = 2u over N-1 joints).
        delta = self._bend_sign * (2.0 * u / (self._num_seg - 1))
        joint_pos = self._zeros_joint.clone()
        joint_pos[:, self._bend_ids] = delta.unsqueeze(-1)
        self._cable.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        del env_ids


@configclass
class CableArcDriveActionCfg(ActionTermCfg):
    """Configuration for the kinematic cable arc-drive action term (no policy dimensions)."""

    class_type: type[ActionTerm] = CableArcDriveAction

    asset_name: str = "cable"
    robot_name: str = "robot"
    cable_name: str = "cable"
    left_hand_body: str = "left_rubber_hand"
    right_hand_body: str = "right_rubber_hand"
    total_length: float = MISSING
    num_segments: int = MISSING
    bend_dof_suffix: str = ":1"
    bend_sign: float = -1.0
