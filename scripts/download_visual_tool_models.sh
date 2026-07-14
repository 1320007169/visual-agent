#!/usr/bin/env bash
set -euo pipefail

# Download the open-source visual-tool models and their official inference code.
#
# Before downloading gated SAM checkpoints:
#   1. Accept the terms at https://huggingface.co/facebook/sam3
#   2. Accept the terms at https://huggingface.co/facebook/sam3.1
#   3. Export a read token without writing it into this file:
#        export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
#
# Example on the Huawei platform:
#   MODEL_ROOT=/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx/models/visual-tools \
#   HF_TOKEN="hf_xxx" \
#   bash scripts/download_visual_tool_models.sh
#
# Optional Hugging Face mirror:
#   HF_ENDPOINT=https://hf-mirror.com bash scripts/download_visual_tool_models.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

MODEL_ROOT="${MODEL_ROOT:-$ROOT_DIR/models/visual-tools}"
SOURCE_ROOT="${SOURCE_ROOT:-$MODEL_ROOT/src}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
HF_TOKEN="${HF_TOKEN:-}"

SAM3_REPO="${SAM3_REPO:-facebook/sam3}"
SAM31_REPO="${SAM31_REPO:-facebook/sam3.1}"
GROUNDING_DINO_REPO="${GROUNDING_DINO_REPO:-ShilongLiu/GroundingDINO}"
GROUNDING_DINO_TRANSFORMERS_REPO="${GROUNDING_DINO_TRANSFORMERS_REPO:-IDEA-Research/grounding-dino-base}"

SAM3_DIR="${SAM3_DIR:-$MODEL_ROOT/sam3}"
SAM31_DIR="${SAM31_DIR:-$MODEL_ROOT/sam3.1}"
GROUNDING_DINO_DIR="${GROUNDING_DINO_DIR:-$MODEL_ROOT/groundingdino-swinb}"
GROUNDING_DINO_TRANSFORMERS_DIR="${GROUNDING_DINO_TRANSFORMERS_DIR:-$MODEL_ROOT/grounding-dino-base-transformers}"

DOWNLOAD_SAM3="${DOWNLOAD_SAM3:-1}"
DOWNLOAD_SAM31="${DOWNLOAD_SAM31:-1}"
DOWNLOAD_GROUNDING_DINO="${DOWNLOAD_GROUNDING_DINO:-1}"
DOWNLOAD_TRANSFORMERS_GROUNDING_DINO="${DOWNLOAD_TRANSFORMERS_GROUNDING_DINO:-1}"
DOWNLOAD_SOURCE="${DOWNLOAD_SOURCE:-1}"

log() {
  printf '[visual-model-download] %s\n' "$*"
}

fail() {
  printf '[visual-model-download] ERROR: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "Missing command '$1'. $2"
}

clone_or_update() {
  local repository="$1"
  local destination="$2"

  if [[ -d "$destination/.git" ]]; then
    log "Updating source: $destination"
    git -C "$destination" pull --ff-only
  elif [[ -e "$destination" ]]; then
    fail "$destination exists but is not a Git repository"
  else
    log "Cloning source: $repository -> $destination"
    git clone --depth 1 "$repository" "$destination"
  fi
}

hf_snapshot() {
  local repository="$1"
  local destination="$2"
  shift 2

  local token_args=()
  if [[ -n "$HF_TOKEN" ]]; then
    token_args=(--token "$HF_TOKEN")
  fi

  mkdir -p "$destination"
  log "Downloading $repository -> $destination"
  HF_ENDPOINT="$HF_ENDPOINT" hf download "$repository" \
    --repo-type model \
    --local-dir "$destination" \
    "${token_args[@]}" \
    "$@"
}

require_command git "Install Git first."
mkdir -p "$MODEL_ROOT" "$SOURCE_ROOT"

if [[ "$DOWNLOAD_SAM3" == "1" || "$DOWNLOAD_SAM31" == "1" ]]; then
  if [[ -z "$HF_TOKEN" ]]; then
    fail "SAM3 checkpoints are gated. Accept both model licenses and export HF_TOKEN before running this script."
  fi
fi

if [[ "$DOWNLOAD_SAM3" == "1" || "$DOWNLOAD_SAM31" == "1" || "$DOWNLOAD_GROUNDING_DINO" == "1" || "$DOWNLOAD_TRANSFORMERS_GROUNDING_DINO" == "1" ]]; then
  require_command hf "Install the current Hugging Face CLI with: pip install -U huggingface_hub"
fi

if [[ "$DOWNLOAD_SOURCE" == "1" ]]; then
  # SAM 3 and SAM 3.1 use the same latest official code repository.
  clone_or_update "https://github.com/facebookresearch/sam3.git" "$SOURCE_ROOT/sam3"
  clone_or_update "https://github.com/IDEA-Research/GroundingDINO.git" "$SOURCE_ROOT/GroundingDINO"
fi

if [[ "$DOWNLOAD_SAM3" == "1" ]]; then
  hf_snapshot "$SAM3_REPO" "$SAM3_DIR"
fi

if [[ "$DOWNLOAD_SAM31" == "1" ]]; then
  hf_snapshot "$SAM31_REPO" "$SAM31_DIR"
fi

if [[ "$DOWNLOAD_GROUNDING_DINO" == "1" ]]; then
  # GroundingDINO-B (Swin-B) is the strongest checkpoint in the fully open
  # official GroundingDINO repository. GroundingDINO 1.5 is API-only and is
  # therefore intentionally not treated as an open checkpoint here.
  hf_snapshot "$GROUNDING_DINO_REPO" "$GROUNDING_DINO_DIR" \
    --include "groundingdino_swinb_cogcoor.pth" "README.md"
fi

if [[ "$DOWNLOAD_TRANSFORMERS_GROUNDING_DINO" == "1" ]]; then
  # This converted Transformers snapshot is directly compatible with
  # scripts/visual_tool_server.py and AutoModelForZeroShotObjectDetection.
  hf_snapshot "$GROUNDING_DINO_TRANSFORMERS_REPO" "$GROUNDING_DINO_TRANSFORMERS_DIR"
fi

log "All requested downloads completed."
log "MODEL_ROOT=$MODEL_ROOT"
log "SAM3_MODEL_PATH=$SAM3_DIR"
log "SAM31_MODEL_PATH=$SAM31_DIR"
log "GroundingDINO Swin-B weights=$GROUNDING_DINO_DIR/groundingdino_swinb_cogcoor.pth"
log "GROUNDING_DINO_MODEL_PATH=$GROUNDING_DINO_TRANSFORMERS_DIR"
log "Official SAM source=$SOURCE_ROOT/sam3"
log "Official GroundingDINO source=$SOURCE_ROOT/GroundingDINO"
