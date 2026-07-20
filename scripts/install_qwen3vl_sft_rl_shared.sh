#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RL_ROOT="$REPO_ROOT/reinforcement_learning"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"

CONDA_ROOT="${VA_CONDA_ROOT:-/opt/huawei/explorer-env/dataset/Common_wl/miniconda3}"
CONDA_EXE="$CONDA_ROOT/bin/conda"
CUDA_BUILD_ROOT="${VA_CUDA_BUILD_ROOT:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx/conda_envs/spacetools-rl}"
CUDA_HOST_CC="${VA_CUDA_HOST_CC:-$CUDA_BUILD_ROOT/bin/x86_64-conda-linux-gnu-gcc}"
CUDA_HOST_CXX="${VA_CUDA_HOST_CXX:-$CUDA_BUILD_ROOT/bin/x86_64-conda-linux-gnu-g++}"
CUDA_TARGET_ROOT="${VA_CUDA_TARGET_ROOT:-$CUDA_BUILD_ROOT/targets/x86_64-linux}"
CUDA_INCLUDE_DIR="$CUDA_TARGET_ROOT/include"
CUDA_LIBRARY_DIR="$CUDA_TARGET_ROOT/lib"
PIP_CACHE_DIR="${VA_PIP_CACHE_DIR:-/cache/visual-agent-qwen3vl/pip}"
TMPDIR="${VA_TMPDIR:-/cache/visual-agent-qwen3vl/tmp}"
XDG_CACHE_HOME="${VA_XDG_CACHE_HOME:-/cache/visual-agent-qwen3vl/xdg}"
SFT_ENV_NAME="${VA_SFT_ENV_NAME:-visual-agent-qwen3vl-sft}"
RL_ENV_NAME="${VA_RL_ENV_NAME:-visual-agent-qwen3vl-rl}"
SFT_ENV_PREFIX="$CONDA_ROOT/envs/$SFT_ENV_NAME"
RL_ENV_PREFIX="$CONDA_ROOT/envs/$RL_ENV_NAME"

PYTHON_VERSION="3.11"
TORCH_VERSION="2.8.0"
TORCHVISION_VERSION="0.23.0"
TORCHAUDIO_VERSION="2.8.0"
TRANSFORMERS_VERSION="4.57.1"
LLAMA_FACTORY_TAG="v0.9.4"
VLLM_VERSION="0.11.0"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--execute]

Without --execute, this script only prints the installation plan.

Optional environment variables:
  VA_CONDA_ROOT    Shared Miniconda root (default: $CONDA_ROOT)
  VA_CUDA_BUILD_ROOT  CUDA 12.8 build toolkit (default: $CUDA_BUILD_ROOT)
  VA_CUDA_HOST_CC     CUDA host C compiler (default: $CUDA_HOST_CC)
  VA_CUDA_HOST_CXX    CUDA host C++ compiler (default: $CUDA_HOST_CXX)
  VA_CUDA_TARGET_ROOT CUDA target root (default: $CUDA_TARGET_ROOT)
  VA_PIP_CACHE_DIR  pip cache (default: $PIP_CACHE_DIR)
  VA_TMPDIR         build temp directory (default: $TMPDIR)
  VA_XDG_CACHE_HOME XDG cache (default: $XDG_CACHE_HOME)
  VA_SFT_ENV_NAME  SFT environment name (default: $SFT_ENV_NAME)
  VA_RL_ENV_NAME   RL environment name (default: $RL_ENV_NAME)

The script inherits Conda channels and pip indexes; caches and temporary files use the configured paths above.
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
Qwen3-VL SFT/RL environment plan
  Conda root: $CONDA_ROOT
  SFT environment: $SFT_ENV_NAME
  SFT path: $SFT_ENV_PREFIX
  RL environment: $RL_ENV_NAME
  RL path: $RL_ENV_PREFIX
  RL clone source: $SFT_ENV_PREFIX

  Python: $PYTHON_VERSION
  PyTorch: $TORCH_VERSION
  Transformers: $TRANSFORMERS_VERSION
  LLaMA-Factory: $LLAMA_FACTORY_TAG
  RL stack: Python 3.11, PyTorch 2.8 cu128, vLLM $VLLM_VERSION, FlashAttention 2.8.3
  FlashAttention build toolkit: $CUDA_BUILD_ROOT
  FlashAttention host compiler: $CUDA_HOST_CXX
  FlashAttention CUDA headers: $CUDA_INCLUDE_DIR

  Conda channels: inherited from the current default configuration
  pip index: inherited from the current default configuration
  pip cache: $PIP_CACHE_DIR
  temporary files: $TMPDIR
  XDG cache: $XDG_CACHE_HOME
  Conda package cache: inherited from the current default configuration
EOF

  
if ((EXECUTE == 0)); then
  echo
  echo "Review the script, then run it again with --execute."
  exit 0
fi

mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$XDG_CACHE_HOME"
export PIP_CACHE_DIR TMPDIR XDG_CACHE_HOME

[[ -x "$CONDA_EXE" ]] || { echo "error: conda not found: $CONDA_EXE" >&2; exit 1; }
[[ -d "$RL_ROOT/verl" ]] || { echo "error: Visual-Agent RL source not found: $RL_ROOT" >&2; exit 1; }
[[ -w "$CONDA_ROOT/envs" ]] || { echo "error: shared env directory is not writable: $CONDA_ROOT/envs" >&2; exit 1; }
[[ -x "$CUDA_BUILD_ROOT/bin/nvcc" ]] || { echo "error: CUDA build toolkit not found: $CUDA_BUILD_ROOT" >&2; exit 1; }
[[ -x "$CUDA_HOST_CC" ]] || { echo "error: CUDA host C compiler not found: $CUDA_HOST_CC" >&2; exit 1; }
[[ -x "$CUDA_HOST_CXX" ]] || { echo "error: CUDA host C++ compiler not found: $CUDA_HOST_CXX" >&2; exit 1; }
[[ -f "$CUDA_INCLUDE_DIR/cuda_runtime_api.h" ]] || { echo "error: CUDA runtime headers not found: $CUDA_INCLUDE_DIR" >&2; exit 1; }
[[ -e "$CUDA_LIBRARY_DIR/libcudart.so" ]] || { echo "error: CUDA runtime library not found: $CUDA_LIBRARY_DIR" >&2; exit 1; }

run_sft() {
  "$CONDA_EXE" run --no-capture-output --name "$SFT_ENV_NAME" "$@"
}

run_rl() {
  "$CONDA_EXE" run --no-capture-output --name "$RL_ENV_NAME" "$@"
}

if [[ ! -x "$SFT_ENV_PREFIX/bin/python" ]]; then
  "$CONDA_EXE" create --yes --name "$SFT_ENV_NAME" "python=$PYTHON_VERSION" pip
else
  echo "Reusing existing SFT environment: $SFT_ENV_PREFIX"
fi

run_sft python -m pip install --upgrade pip "setuptools<81" wheel packaging ninja
run_sft python -m pip install \
  "torch==$TORCH_VERSION" \
  "torchvision==$TORCHVISION_VERSION" \
  "torchaudio==$TORCHAUDIO_VERSION"
run_sft python -m pip install \
  "transformers==$TRANSFORMERS_VERSION" \
  "datasets==4.0.0" \
  "accelerate==1.11.0" \
  "peft==0.17.1" \
  "trl==0.24.0" \
  "torchdata==0.11.0" \
  "deepspeed==0.16.9" \
  "qwen-vl-utils==0.0.14"
run_sft python -m pip install \
  "git+https://github.com/hiyouga/LLaMA-Factory.git@$LLAMA_FACTORY_TAG"

run_sft python - <<'PY'
import importlib.metadata as metadata
import torch
from transformers import AutoConfig

assert metadata.version("llamafactory").startswith("0.9.4")
assert metadata.version("transformers") == "4.57.1"
assert metadata.version("torch").startswith("2.8.0")
assert torch.cuda.is_available(), "PyTorch cannot access a CUDA GPU"

config = AutoConfig.from_pretrained(
    "/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx/DeepEyesV2/models/Qwen3-VL-8B-Instruct",
    trust_remote_code=True,
)
assert config.model_type == "qwen3_vl"
print("SFT verification passed")
print("torch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
PY

if [[ ! -x "$RL_ENV_PREFIX/bin/python" ]]; then
  "$CONDA_EXE" create --yes --name "$RL_ENV_NAME" --clone "$SFT_ENV_NAME"
else
  echo "Reusing existing RL environment: $RL_ENV_PREFIX"
fi

# The RL clone does not need LLaMA-Factory's Gradio UI, whose Starlette<1 constraint conflicts with vLLM 0.11.
run_rl python -m pip uninstall --yes llamafactory gradio gradio-client || true

run_rl python -m pip install \
  "vllm==$VLLM_VERSION" \
  "transformers==$TRANSFORMERS_VERSION" \
  "ray[default]>=2.48.0" \
  "tensordict==0.10.0" \
  "setuptools<81" \
  pybind11 \
  autopep8 evaluate mathruler math-verify eas-prediction \
  codetiming hydra-core pylatexenc wandb liger-kernel

run_rl env \
  CUDA_HOME="$CUDA_BUILD_ROOT" \
  CUDACXX="$CUDA_BUILD_ROOT/bin/nvcc" \
  CC="$CUDA_HOST_CC" \
  CXX="$CUDA_HOST_CXX" \
  PATH="$CUDA_BUILD_ROOT/bin:$RL_ENV_PREFIX/bin:$PATH" \
  CPATH="$CUDA_INCLUDE_DIR:${CPATH:-}" \
  LIBRARY_PATH="$CUDA_LIBRARY_DIR:${LIBRARY_PATH:-}" \
  LD_LIBRARY_PATH="$CUDA_LIBRARY_DIR:$CUDA_BUILD_ROOT/lib:${LD_LIBRARY_PATH:-}" \
  TORCH_CUDA_ARCH_LIST="8.0" \
  MAX_JOBS="${VA_MAX_JOBS:-8}" \
  FLASH_ATTN_CUDA_ARCHS="80" \
  FLASH_ATTENTION_FORCE_BUILD="TRUE" \
  "$RL_ENV_PREFIX/bin/python" -m pip install --no-build-isolation --no-binary=flash-attn "flash-attn==2.8.3"
# Install the repository's DeepEyes/verl fork without its old vLLM<=0.8.5 constraint.
run_rl python -m pip install --no-deps --editable "$RL_ROOT"

run_rl python - <<'PY'
import importlib.metadata as metadata
import torch
import vllm
import vllm._C
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
from transformers import AutoConfig

assert metadata.version("vllm").startswith("0.11.0")
assert metadata.version("flash-attn") == "2.8.3"
assert metadata.version("transformers") == "4.57.1"
assert torch.cuda.is_available(), "PyTorch cannot access a CUDA GPU"

config = AutoConfig.from_pretrained(
    "/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx/DeepEyesV2/models/Qwen3-VL-8B-Instruct",
    trust_remote_code=True,
)
assert config.model_type == "qwen3_vl"
print("RL package verification passed")
print("vLLM:", vllm.__version__)
print("torch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
PY

echo
echo "Installed Qwen3-VL environments:"
echo "  conda activate $SFT_ENV_NAME"
echo "  conda activate $RL_ENV_NAME"
echo
echo "The package stack is installed. Visual-Agent's vLLM 0.11 rollout still requires a focused runtime smoke test."
