#!/usr/bin/env bash
# Record high-res demo videos for ArmHold T-pose and/or Guard (normal locomotion).
# Default: 4 envs, 3840x2160 (4K), multi-env grid camera.
# Guard uses Unitree-G1-29dof-Velocity-ArmHold-Guard (same walk commands as T-pose play).
# Optional: RECORD_ONLY=guard or RECORD_ONLY=tpose to record one variant.
set -euo pipefail

export UNITREE_RL_LAB_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${UNITREE_RL_LAB_PATH}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    for _conda_sh in \
        "${HOME}/miniconda3/etc/profile.d/conda.sh" \
        "${HOME}/anaconda3/etc/profile.d/conda.sh" \
        "${HOME}/mambaforge/etc/profile.d/conda.sh"; do
        if [[ -f "${_conda_sh}" ]]; then
            # shellcheck source=/dev/null
            source "${_conda_sh}"
            conda activate unitree_g1_lab
            break
        fi
    done
fi
if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "[Error] conda env unitree_g1_lab not active. Run:"
    echo "  source ~/miniconda3/etc/profile.d/conda.sh && conda activate unitree_g1_lab"
    exit 1
fi

PYTHON="${CONDA_PREFIX}/bin/python"
DEMO_ROOT="${UNITREE_RL_LAB_PATH}/demos/armhold_locomotion"
VIDEO_LENGTH="${VIDEO_LENGTH:-1000}"   # ~20 s at 0.02 s/step
VIDEO_WIDTH="${VIDEO_WIDTH:-3840}"
VIDEO_HEIGHT="${VIDEO_HEIGHT:-2160}"
NUM_ENVS="${NUM_ENVS:-4}"

TPOSE_CKPT="${TPOSE_CKPT:-logs/rsl_rl/unitree_g1_29dof_velocity_armhold_tpose/2026-06-01_17-46-10_armhold_tpose_4096x1M/model_20600.pt}"
# Default: 4096x5k run — best velocity tracking near model_2300 (see Train/mean_reward peak at model_800).
GUARD_CKPT="${GUARD_CKPT:-logs/rsl_rl/unitree_g1_29dof_velocity_armhold_guard/2026-06-02_17-16-30_armhold_guard_8192x10k/model_3100.pt}"

COMMON_ARGS=(
    --headless
    --video
    --skip_export
    --num_envs "${NUM_ENVS}"
    --video_length "${VIDEO_LENGTH}"
    --video_width "${VIDEO_WIDTH}"
    --video_height "${VIDEO_HEIGHT}"
)

record_one() {
    local task="$1"
    local ckpt="$2"
    local out_dir="$3"
    local prefix="$4"

    echo "============================================================"
    echo "[INFO] Recording: ${task}"
    echo "[INFO] Checkpoint: ${ckpt}"
    echo "[INFO] Output dir: ${out_dir}"
    echo "============================================================"

    local extra_args=()
    if [[ "${NUM_ENVS}" -gt 1 ]]; then
        extra_args+=(--demo_multi_env)
    fi
    "${PYTHON}" scripts/rsl_rl/play.py \
        "${COMMON_ARGS[@]}" \
        "${extra_args[@]}" \
        --task "${task}" \
        --checkpoint "${ckpt}" \
        --video_dir "${out_dir}" \
        --video_name_prefix "${prefix}"
}

mkdir -p "${DEMO_ROOT}"

RECORD_ONLY="${RECORD_ONLY:-all}"
case "${RECORD_ONLY}" in
    all|tpose)
        record_one \
            "Unitree-G1-29dof-Velocity-ArmHold-Tpose" \
            "${TPOSE_CKPT}" \
            "${DEMO_ROOT}/tpose_4env" \
            "armhold_tpose_demo"
        ;;
esac
case "${RECORD_ONLY}" in
    all|guard)
        record_one \
            "Unitree-G1-29dof-Velocity-ArmHold-Guard" \
            "${GUARD_CKPT}" \
            "${DEMO_ROOT}/guard_4env" \
            "armhold_guard_demo"
        ;;
esac

echo ""
echo "[INFO] Done. Videos saved under: ${DEMO_ROOT}"
find "${DEMO_ROOT}" -name "*.mp4" -type f | sort
