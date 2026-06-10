#!/usr/bin/env python3
"""Run Round 1 experiments sequentially, each 310 iters with seed 42."""
import subprocess, sys, time, shutil
from pathlib import Path

PROJECT = Path("/home/jkamohara3/Patrick_temp/unitree_rope/unitree_rl_lab")
CONDA_SH = "/home/jkamohara3/miniconda3/etc/profile.d/conda.sh"
ENV_NAME = "unitree_g1_lab"
TRAIN_SCRIPT = "scripts/rsl_rl/train.py"
RESULTS = PROJECT / "logs/rsl_rl/experiments/round1"
RESULTS.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = [
    ("Unitree-G1-29dof-Cable-Bend-U-Exp-Baseline", "exp_baseline"),
    ("Unitree-G1-29dof-Cable-Bend-U-Exp-1A",       "exp_1A_progress"),
    ("Unitree-G1-29dof-Cable-Bend-U-Exp-1B",       "exp_1B_high_entropy"),
    ("Unitree-G1-29dof-Cable-Bend-U-Exp-1C",       "exp_1C_small_scale"),
    ("Unitree-G1-29dof-Cable-Bend-U-Exp-1D",       "exp_1D_settle_shallow"),
]

def run_exp(task_id, name):
    log_file = RESULTS / f"{name}.log"
    print(f"\n{'='*60}")
    print(f"STARTING {name}  task={task_id}  {time.strftime('%H:%M:%S')}")
    print(f"{'='*60}\n", flush=True)

    cmd = (
        f"source {CONDA_SH} && conda activate {ENV_NAME} && "
        f"cd {PROJECT} && "
        f"python {TRAIN_SCRIPT} --task {task_id} --headless --seed 42"
    )
    with open(log_file, "w") as lf:
        proc = subprocess.run(
            ["bash", "-c", cmd],
            stdout=lf, stderr=subprocess.STDOUT,
            cwd=str(PROJECT),
        )
    rc = proc.returncode
    print(f"FINISHED {name}  exit={rc}  {time.strftime('%H:%M:%S')}\n", flush=True)

    # Copy tfevents for analysis
    log_dirs = sorted(
        PROJECT.glob("logs/rsl_rl/unitree_g1_29dof_cable_bend_u/*/"),
        key=lambda p: p.stat().st_mtime, reverse=True
    )
    for ld in log_dirs[:1]:
        for tf in ld.glob("events.out.tfevents.*"):
            dest = RESULTS / f"{name}.tfevents"
            shutil.copy2(tf, dest)
            print(f"  Saved: {dest}", flush=True)
        print(f"  log_dir: {ld}", flush=True)
    return rc

def main():
    for task_id, name in EXPERIMENTS:
        # Skip already-completed experiments
        done_marker = RESULTS / f"{name}.done"
        if done_marker.exists():
            print(f"SKIP {name} (already done)", flush=True)
            continue

        run_exp(task_id, name)
        done_marker.touch()

    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS DONE  {time.strftime('%H:%M:%S')}")
    print(f"Results at: {RESULTS}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
