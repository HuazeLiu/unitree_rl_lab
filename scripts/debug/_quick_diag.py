#!/usr/bin/env python3
"""Print sorted cable centerline positions with error handling."""
import importlib, os, sys, torch
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "..", "source", "unitree_rl_lab"))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "..", "..", "IsaacLab", "source", "isaaclab"))

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(dict(headless=True, hide_ui=True, enable_cameras=False, livestream=0))
simulation_app = app_launcher.app

import gymnasium as gym
import unitree_rl_lab.tasks  # noqa

from unitree_rl_lab.assets.cable.thick_cable_utils import get_cable_centerline_points, get_cable_segment_poses, compute_centerline_chamfer_reward

G1CableBendUEnvCfg = getattr(
    importlib.import_module("unitree_rl_lab.tasks.locomotion.robots.g1.29dof.g1_cable_bend_u_env_cfg"),
    "G1CableBendUEnvCfg",
)
env_cfg = G1CableBendUEnvCfg(); env_cfg.scene.num_envs = 1; env_cfg.sim.device = "cuda:0"
env = gym.make("Unitree-G1-29dof-Cable-Bend-U", cfg=env_cfg, render_mode=None)
obs, _ = env.reset()

cable = env.unwrapped.scene["cable"]
robot = env.unwrapped.scene["robot"]
cmd_term = env.unwrapped.command_manager.get_term("cable_bend_u")

try:
    print("=== RAW body_pos_w ===")
    print(f"Shape: {cable.data.body_pos_w.shape}")
    print(f"Device: {cable.data.body_pos_w.device}")
    for i in range(cable.data.body_pos_w.shape[1]):
        pos = cable.data.body_pos_w[0, i]
        name = cable.body_names[i] if i < len(cable.body_names) else f"body_{i}"
        print(f"  idx={i} {name}: {pos.tolist()}")
except Exception as e:
    print(f"ERROR reading body_pos_w: {e}")

try:
    print("\n=== SORTED centerline (get_cable_centerline_points) ===")
    cl = get_cable_centerline_points(cable)
    print(f"Shape: {cl.shape}")
    for i in range(cl.shape[1]):
        print(f"  seg_{i:02d}: {cl[0, i].tolist()}")
except Exception as e:
    print(f"ERROR centerline: {e}")

try:
    print("\n=== get_cable_segment_poses ===")
    poses, _ = get_cable_segment_poses(cable)
    print(f"Shape: {poses.shape}")
    for i in range(poses.shape[1]):
        print(f"  seg_{i:02d}: {poses[0, i].tolist()}")
except Exception as e:
    print(f"ERROR poses: {e}")

try:
    left_ids, _ = robot.find_bodies("left_rubber_hand")
    right_ids, _ = robot.find_bodies("right_rubber_hand")
    lh = robot.data.body_pos_w[0, left_ids[0]]
    rh = robot.data.body_pos_w[0, right_ids[0]]
    print(f"\n=== HANDS ===")
    print(f"Left: {lh.tolist()}")
    print(f"Right: {rh.tolist()}")
except Exception as e:
    print(f"ERROR hands: {e}")

try:
    target = cmd_term.target_centerline[0]
    depth_id = cmd_term.depth_ids[0].item()
    print(f"\n=== TARGET (depth_id={depth_id}) ===")
    print(f"Start: {target[0].tolist()}")
    print(f"End: {target[-1].tolist()}")
    print(f"Samples:")
    for i in range(0, 20, 4):
        print(f"  pt_{i:02d}: {target[i].tolist()}")
except Exception as e:
    print(f"ERROR target: {e}")

try:
    print("\n=== KEY DISTANCES ===")
    cl = get_cable_centerline_points(cable)
    target = cmd_term.target_centerline[0]
    cs, ce = cl[0, 0], cl[0, -1]
    ts, te = target[0], target[-1]
    print(f"Cable seg_00 → Target start: {torch.norm(cs-ts).item():.4f}m")
    print(f"Cable seg_11 → Target end:   {torch.norm(ce-te).item():.4f}m")
    root = cable.data.body_pos_w[0, 0]
    print(f"Root ({cable.body_names[0]}) → Left:  {torch.norm(root-lh).item():.4f}m")
    print(f"Root ({cable.body_names[0]}) → Right: {torch.norm(root-rh).item():.4f}m")
except Exception as e:
    print(f"ERROR distances: {e}")

try:
    print("\n=== CHAMFER ===")
    cl = get_cable_centerline_points(cable)
    target = cmd_term.target_centerline[0]
    _T = 0.06
    # cable→target
    dists_ct = torch.cdist(cl[0].unsqueeze(0), target.unsqueeze(0)).squeeze(0)
    min_ct = dists_ct.min(dim=1).values  # for each cable point, min dist to target
    soft_ct = -_T * torch.logsumexp(-min_ct / _T, dim=0)
    # target→cable
    dists_tc = dists_ct.T
    min_tc = dists_tc.min(dim=1).values
    soft_tc = -_T * torch.logsumexp(-min_tc / _T, dim=0)
    chamfer = 0.5 * (soft_ct + soft_tc)
    reward = torch.exp(-chamfer / _T)
    print(f"Soft chamfer distance: {chamfer.item():.6f}m")
    print(f"Reward: {reward.item():.6f}")
    print(f"Mean min CT dist: {min_ct.mean().item():.4f}m")
    print(f"Mean min TC dist: {min_tc.mean().item():.4f}m")
except Exception as e:
    print(f"ERROR chamfer: {e}")

try:
    # Using raw body_pos_w directly as centerline (NO sorting)
    print("\n=== USING UNSORTED body_pos_w AS CENTERLINE ===")
    raw_debug = cable.data.body_pos_w[0:1]  # (1, N, 3)
    dists_ct2 = torch.cdist(raw_debug[0].unsqueeze(0), target.unsqueeze(0)).squeeze(0)
    min_ct2 = dists_ct2.min(dim=1).values
    soft_ct2 = -_T * torch.logsumexp(-min_ct2 / _T, dim=0)
    dists_tc2 = dists_ct2.T
    min_tc2 = dists_tc2.min(dim=1).values
    soft_tc2 = -_T * torch.logsumexp(-min_tc2 / _T, dim=0)
    chamfer2 = 0.5 * (soft_ct2 + soft_tc2)
    reward2 = torch.exp(-chamfer2 / _T)
    print(f"Soft chamfer distance: {chamfer2.item():.6f}m")
    print(f"Reward: {reward2.item():.6f}")
    print(f"Mean min CT dist: {min_ct2.mean().item():.4f}m")
    print(f"Mean min TC dist: {min_tc2.mean().item():.4f}m")
except Exception as e:
    print(f"ERROR unsorted chamfer: {e}")

env.close()
simulation_app.close()
print("\n[DONE]")
