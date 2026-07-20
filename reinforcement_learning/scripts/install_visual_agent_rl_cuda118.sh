#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$RL_ROOT/../.." && pwd)"

ENV_NAME="${VA_RL_ENV_NAME:-visual-agent-rl}"
ENV_PREFIX="${VA_RL_ENV_PREFIX:-$WORKSPACE_ROOT/conda_envs/$ENV_NAME}"
CONDA_EXE="${VA_RL_CONDA_EXE:-/opt/huawei/explorer-env/dataset/Common_wl/miniconda3/bin/conda}"
CUDA_HOME="${VA_RL_CUDA_HOME:-/home/ma-user/work/dataset/trellis_ckpt/cuda/cuda118}"
CONDA_PKGS_DIRS="${VA_RL_CONDA_PKGS_DIRS:-$WORKSPACE_ROOT/conda_pkgs}"
PIP_CACHE_DIR="${VA_RL_PIP_CACHE_DIR:-$WORKSPACE_ROOT/cache/pip}"
MAX_JOBS="${VA_RL_MAX_JOBS:-8}"
BUILD_TMP_DIR="${VA_RL_BUILD_TMP_DIR:-$WORKSPACE_ROOT/tmp/visual-agent-rl-build}"
TORCH_EXTENSIONS_DIR="${VA_RL_TORCH_EXTENSIONS_DIR:-$WORKSPACE_ROOT/cache/torch-extensions}"

PYTHON_VERSION="${VA_RL_PYTHON_VERSION:-3.10}"
TORCH_VERSION="${VA_RL_TORCH_VERSION:-2.6.0}"
TORCHVISION_VERSION="${VA_RL_TORCHVISION_VERSION:-0.21.0}"
TORCHAUDIO_VERSION="${VA_RL_TORCHAUDIO_VERSION:-2.6.0}"
VLLM_VERSION="${VA_RL_VLLM_VERSION:-0.8.2}"
XFORMERS_VERSION="${VA_RL_XFORMERS_VERSION:-0.0.29.post2}"
FLASH_ATTN_VERSION="${VA_RL_FLASH_ATTN_VERSION:-2.7.0.post2}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--execute]

Without --execute, print the resolved installation plan and make no changes.

Environment overrides:
  VA_RL_ENV_NAME       Conda environment name (default: $ENV_NAME)
  VA_RL_ENV_PREFIX     Installation path (default: $ENV_PREFIX)
  VA_RL_CONDA_EXE      Conda executable (default: $CONDA_EXE)
  VA_RL_CUDA_HOME      CUDA 11.8 toolkit path (default: $CUDA_HOME)
  VA_RL_MAX_JOBS       Parallel source-build jobs (default: $MAX_JOBS)
  VA_RL_BUILD_TMP_DIR  Source-build temporary directory (default: $BUILD_TMP_DIR)
  VA_RL_TORCH_EXTENSIONS_DIR Torch extension cache (default: $TORCH_EXTENSIONS_DIR)
EOF
}

EXECUTE=0
case "${1:-}" in
  "") ;;
  --execute) EXECUTE=1 ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

cat <<EOF
Visual-Agent RL environment installation plan
  environment name: $ENV_NAME
  environment path: $ENV_PREFIX
  conda package cache: $CONDA_PKGS_DIRS
  pip cache: $PIP_CACHE_DIR
  CUDA toolkit: $CUDA_HOME
  Python: $PYTHON_VERSION
  PyTorch: $TORCH_VERSION (cu118)
  xFormers: $XFORMERS_VERSION (source build)
  vLLM: $VLLM_VERSION (source build)
  FlashAttention: $FLASH_ATTN_VERSION (source build)
  backend: FSDP + vLLM; SGLang and Megatron are not installed
  build temporary directory: $BUILD_TMP_DIR
  Torch extension cache: $TORCH_EXTENSIONS_DIR
EOF

if ((EXECUTE == 0)); then
  echo
  echo "Review this script, then run it again with --execute."
  exit 0
fi

[[ -x "$CONDA_EXE" ]] || { echo "error: conda not found: $CONDA_EXE" >&2; exit 1; }
[[ -x "$CUDA_HOME/bin/nvcc" ]] || { echo "error: nvcc not found under CUDA_HOME: $CUDA_HOME" >&2; exit 1; }
[[ -f "$CUDA_HOME/lib64/libcudart.so" ]] || { echo "error: CUDA runtime not found under CUDA_HOME: $CUDA_HOME" >&2; exit 1; }

cuda_release="$($CUDA_HOME/bin/nvcc --version | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p')"
[[ "$cuda_release" == 11.8 ]] || {
  echo "error: this installer requires CUDA 11.8, found ${cuda_release:-unknown}" >&2
  exit 1
}

mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$BUILD_TMP_DIR" "$TORCH_EXTENSIONS_DIR" "$(dirname "$ENV_PREFIX")"
export CONDA_PKGS_DIRS PIP_CACHE_DIR CUDA_HOME MAX_JOBS
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-80}"
export VLLM_TARGET_DEVICE=cuda
export PIP_DISABLE_PIP_VERSION_CHECK=1

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  "$CONDA_EXE" create --yes --prefix "$ENV_PREFIX" "python=$PYTHON_VERSION" pip
else
  echo "Reusing existing environment: $ENV_PREFIX"
fi

run_in_env() {
  # conda run restores CUDA_HOME from the cloned SFT environment.
  # Inject the build toolchain after activation for CUDA 11.8 extensions.
  "$CONDA_EXE" run --no-capture-output --prefix "$ENV_PREFIX" \
    env \
    "CUDA_HOME=$CUDA_HOME" \
    "PATH=$CUDA_HOME/bin:$ENV_PREFIX/bin:$PATH" \
    "LD_LIBRARY_PATH=$CUDA_HOME/lib64:$ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "CC=$ENV_PREFIX/bin/gcc" \
    "CXX=$ENV_PREFIX/bin/g++" \
    "MAX_JOBS=$MAX_JOBS" \
    "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST" \
    "CMAKE_CUDA_ARCHITECTURES=$CMAKE_CUDA_ARCHITECTURES" \
    "VLLM_TARGET_DEVICE=$VLLM_TARGET_DEVICE" \
    "PIP_CACHE_DIR=$PIP_CACHE_DIR" \
    "PIP_DISABLE_PIP_VERSION_CHECK=$PIP_DISABLE_PIP_VERSION_CHECK" \
    "TMPDIR=$BUILD_TMP_DIR" \
    "TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR" \
    "$@"
}

run_in_env python -m pip install --upgrade pip setuptools-scm wheel packaging ninja cmake jinja2
run_in_env python -m pip install --ignore-installed --no-deps "setuptools==80.9.0"

# Install cu118 wheels first. Their local version satisfies packages requiring torch==2.6.0.
run_in_env python -m pip install \
  --index-url https://download.pytorch.org/whl/cu118 \
  "torch==$TORCH_VERSION" \
  "torchvision==$TORCHVISION_VERSION" \
  "torchaudio==$TORCHAUDIO_VERSION"

run_in_env python - <<'PY'
import os
import torch
from torch.utils.cpp_extension import CUDA_HOME
expected = os.path.realpath(os.environ["CUDA_HOME"])
detected = os.path.realpath(CUDA_HOME or "")
print(f"PyTorch CUDA_HOME: {detected}")
if detected != expected:
    raise RuntimeError(f"PyTorch detected CUDA_HOME={detected}, expected {expected}")
if torch.version.cuda != "11.8":
    raise RuntimeError(f"expected PyTorch cu118, found CUDA {torch.version.cuda}")
PY

# PyPI wheels for these packages target newer CUDA runtimes. Build against CUDA 11.8.
run_in_env python -m pip install \
  --no-build-isolation \
  --no-binary=xformers \
  "xformers==$XFORMERS_VERSION"

run_in_env python -m pip install \
  --no-build-isolation \
  --no-deps \
  "vllm @ git+https://github.com/vllm-project/vllm.git@v$VLLM_VERSION"

run_in_env python -m pip install \
  --no-build-isolation \
  "flash-attn==$FLASH_ATTN_VERSION"

# A cloned environment can contain a large Transformers tree. On some shared
# filesystems pip's uninstall-by-rename can stop halfway with "Directory not
# empty". Overlay the pinned wheel without running the old uninstall step.
run_in_env python -m pip install \
  --ignore-installed \
  --no-deps \
  "transformers==4.52.4"

# Pin the versions recorded by DeepEyes while keeping the CUDA stack installed above.
run_in_env python -m pip install \
  "ray[default]==2.46.0" \
  "tensordict==0.6.2" \
  "numpy==1.26.3" \
  "pyarrow==19.0.1" \
  "evaluate==0.4.3" \
  "pynvml==12.0.0" \
  "mathruler==0.1.0" \
  "pydantic==2.11.4" \
  "openai==1.79.0" \
  "qwen_vl_utils==0.0.11" \
  "math_verify==0.7.0" \
  eas_prediction \
  accelerate codetiming datasets dill hydra-core pandas peft pybind11 \
  pylatexenc torchdata wandb uvicorn fastapi liger-kernel

# vLLM is installed with --no-deps to avoid its ray[cgraph] extra, which pulls
# CUDA 12 CuPy wheels. Install the CUDA-neutral runtime dependencies explicitly.
run_in_env python -m pip install \
  evaluate==0.4.3 openai==1.79.0 \
  blake3 cachetools "compressed-tensors==0.9.2" "depyf==0.18.0" \
  "gguf==0.10.0" "lark==1.2.2" "llguidance>=0.7.9,<0.8.0" \
  "lm-format-enforcer>=0.10.11,<0.11" mistral-common msgspec \
  "outlines==0.1.11" partial-json-parser \
  "prometheus-fastapi-instrumentator==7.1.0" "starlette==0.47.1" \
  python-json-logger pyzmq watchfiles uvloop httptools "xgrammar==0.1.16" "numba==0.60.0"

# Keep OpenCV compatible with NumPy 1.26 and make the headless files win over
# opencv-python so compute nodes do not require libGL.so.1.
run_in_env python -m pip install \
  "opencv-python==4.11.0.86" "opencv-python-headless==4.11.0.86"
run_in_env python -m pip install \
  --ignore-installed --no-deps "opencv-python-headless==4.11.0.86"

# Install this repository's modified verl code without allowing pip to replace CUDA packages.
run_in_env python -m pip install --no-deps --editable "$RL_ROOT"

run_in_env python - <<'PY'
import importlib.metadata as metadata
import torch

expected = {
    "torch": "2.6.0",
    "vllm": "0.8.2",
    "xformers": "0.0.29.post2",
    "flash-attn": "2.7.0.post2",
    "ray": "2.46.0",
    "tensordict": "0.6.2",
    "transformers": "4.52.4",
    "verl": "0.4.0.dev0",
}

print(f"torch CUDA runtime: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
for package, wanted in expected.items():
    installed = metadata.version(package)
    print(f"{package}: {installed}")
    if not installed.startswith(wanted):
        raise RuntimeError(f"{package}: expected {wanted}, found {installed}")

if torch.version.cuda != "11.8":
    raise RuntimeError(f"expected PyTorch cu118, found cu{torch.version.cuda}")
PY

echo
echo "Installed $ENV_NAME at $ENV_PREFIX"
echo "Activate with: conda activate $ENV_PREFIX"
