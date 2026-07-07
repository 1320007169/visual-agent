#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG_PATH="${CONFIG_PATH:-configs/train_visual_tool_sft_smoke.yaml}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-}"
LLAMAFACTORY_CLI="${LLAMAFACTORY_CLI:-}"

if [[ -z "$LLAMAFACTORY_CLI" ]] && command -v llamafactory-cli >/dev/null 2>&1; then
  LLAMAFACTORY_CLI="$(command -v llamafactory-cli)"
fi

if [[ -z "$LLAMAFACTORY_CLI" && -x ".venv-llamafactory/bin/llamafactory-cli" ]]; then
  LLAMAFACTORY_CLI=".venv-llamafactory/bin/llamafactory-cli"
fi

if [[ -z "$LLAMAFACTORY_CLI" ]]; then
  LLAMAFACTORY_CLI="llamafactory-cli"
fi

if [[ -n "$MODEL_NAME_OR_PATH" ]]; then
  tmp_config="$(mktemp /tmp/visual_tool_sft_smoke.XXXXXX.yaml)"
  python - "$CONFIG_PATH" "$tmp_config" "$MODEL_NAME_OR_PATH" <<'PY'
from pathlib import Path
import sys

src, dst, model_path = sys.argv[1:]
text = Path(src).read_text()
lines = []
for line in text.splitlines():
    if line.startswith("model_name_or_path:"):
        lines.append(f"model_name_or_path: {model_path}")
    else:
        lines.append(line)
Path(dst).write_text("\n".join(lines) + "\n")
PY
  CONFIG_PATH="$tmp_config"
fi

"$LLAMAFACTORY_CLI" train "$CONFIG_PATH"
