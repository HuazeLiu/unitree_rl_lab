#!/bin/bash
# Round 1 experiment dispatch — 5 configs, 310 iters each, seed 42
set -euo pipefail

cd /home/jkamohara3/Patrick_temp/unitree_rope/unitree_rl_lab
source /home/jkamohara3/miniconda3/etc/profile.d/conda.sh
conda activate unitree_g1_lab

EXPS=(
  "Unitree-G1-29dof-Cable-Bend-U-Exp-Baseline:exp_1_baseline"
  "Unitree-G1-29dof-Cable-Bend-U-Exp-1A:exp_1A_progress"
  "Unitree-G1-29dof-Cable-Bend-U-Exp-1B:exp_1B_high_entropy"
  "Unitree-G1-29dof-Cable-Bend-U-Exp-1C:exp_1C_small_scale"
  "Unitree-G1-29dof-Cable-Bend-U-Exp-1D:exp_1D_settle_shallow"
)

RESULTS_DIR="logs/rsl_rl/experiments/round1"
mkdir -p "$RESULTS_DIR"

for entry in "${EXPS[@]}"; do
  TASK="${entry%%:*}"
  NAME="${entry##*:}"

  echo "============================================================"
  echo "=== STARTING $NAME  ($(date)) ==="
  echo "=== Task: $TASK"
  echo "============================================================"

  python scripts/rsl_rl/train.py \
    --task "$TASK" \
    --headless \
    --experiment_name "$NAME" \
    --seed 42 \
    > "$RESULTS_DIR/${NAME}.log" 2>&1

  RC=$?
  echo "=== FINISHED $NAME  exit=$RC  ($(date)) ==="

  # Find the log dir created by this run
  LOG_DIR=$(ls -dt logs/rsl_rl/unitree_g1_29dof_cable_bend_u/*/ 2>/dev/null | head -1)
  if [ -n "$LOG_DIR" ]; then
    # Copy the tfevents for analysis
    cp -v "$LOG_DIR"/events.out.tfevents.* "$RESULTS_DIR/${NAME}.tfevents" 2>/dev/null || true
    echo "  log_dir=$LOG_DIR"
  fi
done

echo "============================================================"
echo "=== ALL EXPERIMENTS COMPLETE ($(date)) ==="
echo "============================================================"
