#!/usr/bin/env bash
set -Eeuo pipefail

# ModelArts entrypoint for the ZWZ multi-node RL run. Platform jobs execute
# this file on every node; the repository launcher owns the shared training,
# Ray, checkpoint, and visual-tool configuration.

mkdir -p /opt/huawei/explorer-env /home/ma-user/work/algorithm /home/ma-user/work/model

ensure_symlink() {
  local target="$1"
  local link_path="$2"
  if [[ ! -e "$link_path" && ! -L "$link_path" ]]; then
    ln -s "$target" "$link_path"
  fi
}

ensure_symlink /opt/huawei/dataset /opt/huawei/explorer-env/dataset
ensure_symlink /opt/huawei/dataset /home/ma-user/work/dataset
ensure_symlink /opt/huawei/schedule-train/algorithm/algorithmrefs/synaflow_wl /home/ma-user/work/algorithm/synaflow_wl
ensure_symlink /opt/huawei/quoteModel/xiaoyi_tmpstorage /home/ma-user/work/model/xiaoyi_tmpstorage

export BASE="${BASE:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx}"
export REPO_ROOT="${REPO_ROOT:-$BASE/visual-agent}"
export RL_ROOT="${RL_ROOT:-$REPO_ROOT/reinforcement_learning}"
export RL_ENV_DIR="${RL_ENV_DIR:-/opt/huawei/explorer-env/dataset/Common_wl/miniconda3/envs/visual-agent-qwen3vl-rl}"
export TOOL_ENV_DIR="${TOOL_ENV_DIR:-$BASE/conda_envs/visual-tools}"
export CUDA_HOME="${CUDA_HOME:-$BASE/conda_envs/spacetools-rl}"
export CUDA_LIBRARY_DIR="${CUDA_LIBRARY_DIR:-$CUDA_HOME/targets/x86_64-linux/lib}"
export CC="${CC:-$CUDA_HOME/bin/x86_64-conda-linux-gnu-gcc}"
export CXX="${CXX:-$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-$CXX}"
export TOOL_CUDA_HOME="${TOOL_CUDA_HOME:-/opt/huawei/explorer-env/dataset/trellis_ckpt/cuda/cuda118}"
export TOOL_CUDA_LIBRARY_DIR="${TOOL_CUDA_LIBRARY_DIR:-$TOOL_CUDA_HOME/lib64}"

exec bash "$REPO_ROOT/scripts/run_visual_agent_zwz_rl_2node_16gpu.sh"
