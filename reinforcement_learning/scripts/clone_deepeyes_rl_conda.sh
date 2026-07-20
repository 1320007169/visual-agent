#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$RL_ROOT/../.." && pwd)"

SOURCE_PREFIX="${DEEPEYES_RL_SOURCE_PREFIX:-$WORKSPACE_ROOT/conda_envs/deepeyes-sft-conda}"
TARGET_PREFIX="${DEEPEYES_RL_TARGET_PREFIX:-$WORKSPACE_ROOT/conda_envs/deepeyes-rl-conda}"
CONDA_EXE="${DEEPEYES_RL_CONDA_EXE:-/opt/huawei/explorer-env/dataset/Common_wl/miniconda3/bin/conda}"
CONDA_PKGS_DIRS="${DEEPEYES_RL_CONDA_PKGS_DIRS:-$WORKSPACE_ROOT/conda_pkgs}"
RL_INSTALLER="$SCRIPT_DIR/install_visual_agent_rl_cuda118.sh"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--execute] [--with-rl]

Clone the Qwen2.5-VL SFT Conda environment into an isolated RL environment.
Without --execute, the script only prints the resolved plan.

Options:
  --execute   Perform the clone.
  --with-rl   After cloning, install the CUDA 11.8 veRL/vLLM stack.
  -h, --help  Show this help.

Environment overrides:
  DEEPEYES_RL_SOURCE_PREFIX   Source environment (default: $SOURCE_PREFIX)
  DEEPEYES_RL_TARGET_PREFIX   Target environment (default: $TARGET_PREFIX)
  DEEPEYES_RL_CONDA_EXE       Conda executable (default: $CONDA_EXE)
  DEEPEYES_RL_CONDA_PKGS_DIRS Conda package cache (default: $CONDA_PKGS_DIRS)
EOF
}

EXECUTE=0
WITH_RL=0
while (($#)); do
  case "$1" in
    --execute)
      EXECUTE=1
      ;;
    --with-rl)
      WITH_RL=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if ((WITH_RL == 1 && EXECUTE == 0)); then
  echo "error: --with-rl requires --execute" >&2
  exit 2
fi

cat <<EOF
DeepEyes RL Conda clone plan
  source: $SOURCE_PREFIX
  target: $TARGET_PREFIX
  conda: $CONDA_EXE
  package cache: $CONDA_PKGS_DIRS
  install RL dependencies: $([[ $WITH_RL == 1 ]] && echo yes || echo no)
EOF

[[ -x "$CONDA_EXE" ]] || {
  echo "error: conda executable not found: $CONDA_EXE" >&2
  exit 1
}
[[ -d "$SOURCE_PREFIX/conda-meta" ]] || {
  echo "error: source is not a Conda environment: $SOURCE_PREFIX" >&2
  exit 1
}
[[ -x "$SOURCE_PREFIX/bin/python" ]] || {
  echo "error: source Python not found: $SOURCE_PREFIX/bin/python" >&2
  exit 1
}

if [[ -e "$TARGET_PREFIX" ]]; then
  echo "error: target already exists; refusing to overwrite it: $TARGET_PREFIX" >&2
  exit 1
fi

if ((EXECUTE == 0)); then
  echo
  echo "No changes made. Run again with --execute to clone the environment."
  exit 0
fi

mkdir -p "$(dirname "$TARGET_PREFIX")" "$CONDA_PKGS_DIRS"
export CONDA_PKGS_DIRS

"$CONDA_EXE" create \
  --yes \
  --prefix "$TARGET_PREFIX" \
  --clone "$SOURCE_PREFIX"

"$CONDA_EXE" run --no-capture-output --prefix "$TARGET_PREFIX" python - <<'PY'
import importlib.metadata as metadata
import torch

packages = (
    "torch",
    "torchvision",
    "torchaudio",
    "transformers",
    "flash-attn",
    "deepspeed",
)

print(f"Python clone validation: torch={torch.__version__}, CUDA={torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
for package in packages:
    print(f"{package}: {metadata.version(package)}")

if not torch.__version__.startswith("2.6.0"):
    raise RuntimeError(f"expected cloned torch 2.6.0, found {torch.__version__}")
if torch.version.cuda != "11.8":
    raise RuntimeError(f"expected cloned PyTorch cu118, found CUDA {torch.version.cuda}")
PY

if ((WITH_RL == 1)); then
  [[ -x "$RL_INSTALLER" ]] || {
    echo "error: RL installer is missing or not executable: $RL_INSTALLER" >&2
    exit 1
  }
  VA_RL_ENV_PREFIX="$TARGET_PREFIX" \
    VA_RL_ENV_NAME="$(basename "$TARGET_PREFIX")" \
    "$RL_INSTALLER" --execute
else
  cat <<EOF

Clone completed. To add the CUDA 11.8 RL stack later, run:
  VA_RL_ENV_PREFIX="$TARGET_PREFIX" "$RL_INSTALLER" --execute
EOF
fi

echo
echo "Environment ready at: $TARGET_PREFIX"
echo "Activate with: conda activate $TARGET_PREFIX"
