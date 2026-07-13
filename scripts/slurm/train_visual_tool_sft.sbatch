#!/usr/bin/env bash
#SBATCH --job-name=visual-tool-sft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=visual-tool-sft-%j.out
#SBATCH --error=visual-tool-sft-%j.err

set -Eeuo pipefail

[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "error: submit this script with sbatch" >&2; exit 2; }
[[ -n "${SLURM_NNODES:-}" ]] || { echo "error: SLURM_NNODES is missing" >&2; exit 2; }
[[ -n "${SLURM_JOB_NODELIST:-}" ]] || { echo "error: SLURM_JOB_NODELIST is missing" >&2; exit 2; }

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[[ -f "$REPO_ROOT/scripts/train_visual_tool_sft_full.sh" ]] || {
  echo "error: REPO_ROOT does not contain the training launcher: $REPO_ROOT" >&2
  exit 2
}
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-29500}"
NNODES="$SLURM_NNODES"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

export REPO_ROOT MASTER_ADDR MASTER_PORT NNODES NPROC_PER_NODE
mkdir -p "$REPO_ROOT/logs"

echo "Launching $NNODES node(s), $NPROC_PER_NODE GPU process(es) per node"
echo "Rendezvous: $MASTER_ADDR:$MASTER_PORT"

srun --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 --kill-on-bad-exit=1 \
  bash -c 'export NODE_RANK="${SLURM_PROCID}"; exec bash "${REPO_ROOT}/scripts/train_visual_tool_sft_full.sh"'
