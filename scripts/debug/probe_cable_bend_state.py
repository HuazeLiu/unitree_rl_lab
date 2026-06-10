#!/usr/bin/env python3
"""Probe Stage-2 reset state: arm joint targets vs hold pose, hand/cable positions, weld prim."""

from __future__ import annotations

import argparse
import pathlib
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=20)
args = parser.parse_args()
args.headless = True

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
app = AppLauncher(args).app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
import isaacsim.core.utils.prims as prim_utils  # noqa: E402

import unitree_rl_lab.tasks  # noqa: E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402

TASK = "Unitree-G1-29dof-Cable-Bend-U"


def main() -> None:
    cfg = parse_env_cfg(TASK, num_envs=1)
    env = gym.make(TASK, cfg=cfg, render_mode=None)
    base = env.unwrapped

    env.reset()
    robot = base.scene["robot"]
    cable = base.scene["cable"]
    action = torch.zeros(1, base.action_manager.total_action_dim, device=base.device)
    for _ in range(args.steps):
        env.step(action)

    print("=== cable dof structure ===", flush=True)
    print("cable.num_joints:", cable.num_joints, flush=True)
    print("cable.joint_names[:12]:", cable.joint_names[:12], flush=True)
    print("cable.body_names[:6]:", cable.body_names[:6], flush=True)

    names = robot.joint_names
    jp = robot.data.joint_pos[0]
    print("=== arm joint pos after step (rad) ===", flush=True)
    for n, v in zip(names, jp.tolist()):
        if any(t in n for t in ("shoulder", "elbow", "wrist")):
            print(f"  {n:32s} {v:+.3f}", flush=True)

    import importlib

    import isaaclab.utils.math as m

    ccfg = importlib.import_module(
        "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.g1_cable_bend_u_env_cfg"
    ).G1_CABLE_BEND_SCENE_CABLE_CFG
    lid, _ = robot.find_bodies("left_rubber_hand")
    rid, _ = robot.find_bodies("right_rubber_hand")
    lp = robot.data.body_pos_w[0, lid[0]]
    rp = robot.data.body_pos_w[0, rid[0]]
    print("left_hand_w :", [round(x, 3) for x in lp.tolist()], flush=True)
    print("right_hand_w:", [round(x, 3) for x in rp.tolist()], flush=True)

    sid, _ = cable.find_bodies("seg_00")
    eid, _ = cable.find_bodies(f"seg_{ccfg.num_segments - 1:02d}")
    hs = 0.5 * ccfg.segment_length
    dev = base.device
    s_tip = cable.data.body_pos_w[0, sid[0]] + m.quat_apply(
        cable.data.body_quat_w[0:1, sid[0]], torch.tensor([[-hs, 0, 0]], device=dev)
    )[0]
    e_tip = cable.data.body_pos_w[0, eid[0]] + m.quat_apply(
        cable.data.body_quat_w[0:1, eid[0]], torch.tensor([[hs, 0, 0]], device=dev)
    )[0]
    print("seg00 start-tip:", [round(x, 3) for x in s_tip.tolist()], flush=True)
    print("seglast end-tip:", [round(x, 3) for x in e_tip.tolist()], flush=True)
    print(f"start_err_mm={float(torch.linalg.norm(s_tip - lp)) * 1000:.1f}", flush=True)
    print(f"end_err_mm  ={float(torch.linalg.norm(e_tip - rp)) * 1000:.1f}", flush=True)
    span = float(torch.linalg.norm(rp - lp))
    print(f"hand_span_m ={span:.3f} arc_len={ccfg.total_length}", flush=True)
    bend = [i for i, n in enumerate(cable.joint_names) if n.endswith(":1")]
    jpc = cable.data.joint_pos[0]
    print(f"bend dof (':1') mean/min/max rad: {jpc[bend].mean():.4f}/{jpc[bend].min():.4f}/{jpc[bend].max():.4f}", flush=True)
    # arm symmetry: left vs right shoulder/elbow
    jp_named = dict(zip(robot.joint_names, robot.data.joint_pos[0].tolist()))
    for j in ("shoulder_pitch_joint", "shoulder_roll_joint", "elbow_joint"):
        print(f"  L/R {j}: {jp_named.get('left_' + j):+.3f} / {jp_named.get('right_' + j):+.3f}", flush=True)
    # lowest cable point (sag depth indicator)
    cz = cable.data.body_pos_w[0, :, 2]
    print(f"cable z range: min={float(cz.min()):.3f} max={float(cz.max()):.3f}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    app.close()
