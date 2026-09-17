#!/usr/bin/env bash
set -Eeuo pipefail

# Run the same command on both 8-GPU ModelArts nodes. One scenario per job.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${CONFIG_PYTHON:-python}" "$SCRIPT_DIR/visual_agent_rl_resource_benchmark.py" "$@"
