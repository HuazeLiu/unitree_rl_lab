#!/usr/bin/env bash
# Record thick cable validation scenes (requires unitree_g1_lab conda env + Isaac Sim).
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
    echo "[Error] conda env unitree_g1_lab not active."
    exit 1
fi

PYTHON="${CONDA_PREFIX}/bin/python"
OUTPUT_ROOT="${OUTPUT_ROOT:-$UNITREE_RL_LAB_PATH/demos/thick_cable}"
NUM_STEPS="${NUM_STEPS:-720}"

mkdir -p "$OUTPUT_ROOT/sag_test" "$OUTPUT_ROOT/push_on_table" "$OUTPUT_ROOT/torso_support"

run_scene() {
    local script_name="$1"
    local out_dir="$2"
    echo "[record] $script_name -> $out_dir"
    PYTHONUNBUFFERED=1 "$PYTHON" "$UNITREE_RL_LAB_PATH/scripts/cable/${script_name}.py" \
        --headless \
        --num_steps "$NUM_STEPS" \
        --output_dir "$out_dir"
}

run_scene validate_sag_test "$OUTPUT_ROOT/sag_test"
run_scene validate_push_on_table "$OUTPUT_ROOT/push_on_table"
run_scene validate_torso_support "$OUTPUT_ROOT/torso_support"

echo "[done] Validation logs under $OUTPUT_ROOT"
