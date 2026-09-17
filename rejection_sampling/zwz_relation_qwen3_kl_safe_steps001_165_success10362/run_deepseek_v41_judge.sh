#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${BASE:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx}"
KEY_FILE="${DEEPSEEK_API_KEY_FILE:-$BASE/secrets/deepseek_api_key.txt}"

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  [[ -r "$KEY_FILE" ]] || { echo "error: DeepSeek API key file is not readable: $KEY_FILE" >&2; exit 2; }
  IFS= read -r DEEPSEEK_API_KEY < "$KEY_FILE" || true
  DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY%$'\r'}"
  [[ -n "$DEEPSEEK_API_KEY" ]] || { echo "error: DeepSeek API key file is empty" >&2; exit 2; }
  export DEEPSEEK_API_KEY
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
LIMIT="${LIMIT:-300}"
MAX_WORKERS="${MAX_WORKERS:-4}"
REQUESTS_PER_MINUTE="${REQUESTS_PER_MINUTE:-30}"

exec "$PYTHON_BIN" "$SCRIPT_DIR/judge_with_deepseek_v41.py" \
  --input "$BASE/visual-agent/data/rl_distill/qwen3_groundingdino_kl_safe_success_sft.jsonl" \
  --output-dir "$SCRIPT_DIR/outputs/deepseek_v41" \
  --base-url "https://api.deepseek.com" \
  --model "deepseek-flash" \
  --limit "$LIMIT" \
  --max-workers "$MAX_WORKERS" \
  --requests-per-minute "$REQUESTS_PER_MINUTE" \
  "$@"
