#!/usr/bin/env python3
"""Dump cable diagnostic data to a file."""
import importlib, os, sys, torch, json
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "..", "source", "unitree_rl_lab"))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "..", "..", "IsaacLab", "source", "isaaclab"))

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(dict(headless=True, hide_ui=True, enable_cameras=False, livestream=0))
simulation_app = app_launcher.app

import gymnasium as gym
import unitree_rl_lab.tasks  # noqa

from unitree_rl_lab.assets.cable.thick_cable_utils import get_cable_centerline_points, get_cable_segment_poses

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

results = {}
results["body_names"] = list(cable.body_names)
results["num_bodies"] = len(cable.body_names)

# Raw body positions
bpw = cable.data.body_pos_w[0].cpu()
results["body_pos_w"] = []
for i in range(bpw.shape[0]):
    results["body_pos_w"].append({
        "idx": i,
        "name": cable.body_names[i],
        "pos": bpw[i].tolist()
    })

# Sorted centerline
cl = get_cable_centerline_points(cable).cpu()
results["sorted_centerline"] = []
for i in range(cl.shape[1]):
    results["sorted_centerline"].append({
        "idx": i,
        "pos": cl[0, i].tolist()
    })

# Target centerline
target = cmd_term.target_centerline[0].cpu()
results["target_centerline"] = []
for i in range(0, target.shape[0], 4):  # every 4th point
    results["target_centerline"].append({
        "idx": i,
        "pos": target[i].tolist()
    })
results["target_start"] = target[0].tolist()
results["target_end"] = target[-1].tolist()

# Hands
left_ids, _ = robot.find_bodies("left_rubber_hand")
right_ids, _ = robot.find_bodies("right_rubber_hand")
lh = robot.data.body_pos_w[0, left_ids[0]].cpu()
rh = robot.data.body_pos_w[0, right_ids[0]].cpu()
results["left_hand"] = lh.tolist()
results["right_hand"] = rh.tolist()

# Key distances
cs = torch.tensor(results["sorted_centerline"][0]["pos"])
ce = torch.tensor(results["sorted_centerline"][-1]["pos"])
ts = torch.tensor(results["target_start"])
te = torch.tensor(results["target_end"])
root = bpw[0]
results["dist_start_target"] = float(torch.norm(cs - ts))
results["dist_end_target"] = float(torch.norm(ce - te))
results["dist_root_left"] = float(torch.norm(root - lh))
results["dist_root_right"] = float(torch.norm(root - rh))

# Chamfer
_T = 0.06
dists_ct = torch.cdist(cl[0].unsqueeze(0), target.unsqueeze(0)).squeeze(0)
min_ct = dists_ct.min(dim=1).values
soft_ct = -_T * torch.logsumexp(-min_ct / _T, dim=0)
dists_tc = dists_ct.T
min_tc = dists_tc.min(dim=1).values
soft_tc = -_T * torch.logsumexp(-min_tc / _T, dim=0)
chamfer = 0.5 * (soft_ct + soft_tc)
reward = torch.exp(-chamfer / _T)
results["chamfer_dist"] = float(chamfer)
results["chamfer_reward"] = float(reward)
results["mean_min_ct"] = float(min_ct.mean())
results["mean_min_tc"] = float(min_tc.mean())

# Chamfer using UNSORTED body_pos_w
dists_ct2 = torch.cdist(bpw.unsqueeze(0), target.unsqueeze(0)).squeeze(0)
min_ct2 = dists_ct2.min(dim=1).values
soft_ct2 = -_T * torch.logsumexp(-min_ct2 / _T, dim=0)
dists_tc2 = dists_ct2.T
min_tc2 = dists_tc2.min(dim=1).values
soft_tc2 = -_T * torch.logsumexp(-min_tc2 / _T, dim=0)
chamfer2 = 0.5 * (soft_ct2 + soft_tc2)
reward2 = torch.exp(-chamfer2 / _T)
results["unsorted_chamfer_dist"] = float(chamfer2)
results["unsorted_chamfer_reward"] = float(reward2)
results["unsorted_mean_min_ct"] = float(min_ct2.mean())
results["unsorted_mean_min_tc"] = float(min_tc2.mean())

with open("/tmp/cable_diag.json", "w") as f:
    json.dump(results, f, indent=2)

env.close()
simulation_app.close()

# Write a separate summary file that definitely makes it
with open("/tmp/cable_diag_summary.txt", "w") as f:
    for b in results["body_pos_w"]:
        f.write(f"idx={b['idx']} {b['name']}: {b['pos']}\n")
    f.write(f"\n")
    for s in results["sorted_centerline"]:
        f.write(f"sorted_seg_{s['idx']:02d}: {s['pos']}\n")
    f.write(f"\nLeft: {results['left_hand']}\n")
    f.write(f"Right: {results['right_hand']}\n")
    f.write(f"\nTarget start: {results['target_start']}\n")
    f.write(f"Target end: {results['target_end']}\n")
    f.write(f"\nChamfer (sorted): dist={results['chamfer_dist']:.6f} reward={results['chamfer_reward']:.6f}\n")
    f.write(f"  mean_min_ct={results['mean_min_ct']:.4f} mean_min_tc={results['mean_min_tc']:.4f}\n")
    f.write(f"Chamfer (unsorted): dist={results['unsorted_chamfer_dist']:.6f} reward={results['unsorted_chamfer_reward']:.6f}\n")
    f.write(f"  mean_min_ct={results['unsorted_mean_min_ct']:.4f} mean_min_tc={results['unsorted_mean_min_tc']:.4f}\n")
    f.write(f"\nCable seg0->target start: {results['dist_start_target']:.4f}\n")
    f.write(f"Cable seg11->target end: {results['dist_end_target']:.4f}\n")
    f.write(f"Root->left: {results['dist_root_left']:.4f} Root->right: {results['dist_root_right']:.4f}\n")
