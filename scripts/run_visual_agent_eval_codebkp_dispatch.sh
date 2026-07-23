#!/usr/bin/env bash
set -Eeuo pipefail

name="$(basename -- "$0")"
case "$name" in
  run_visual_agent_eval.sh|run_visual_agent_eval_qwen3.sh)
    target=run_visual_agent_eval_qwen3_compare.sh
    ;;
  run_visual_agent_zwz_rl_2node_16gpu.sh)
    target=run_visual_agent_zwz_rl_modelarts.sh
    ;;
  run_visual_agent_zwz_rl_4node_32gpu.sh)
    target=run_visual_agent_zwz_rl_4node_32gpu.sh
    ;;
  *)
    echo "Unsupported ModelArts entrypoint: $name" >&2
    exit 2
    ;;
esac

script="/opt/huawei/quoteModel/xiaoyi_tmpstorage/haohang/min/gx/visual-agent/scripts/$target"
if [[ ! -f "$script" ]]; then
  script="/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx/visual-agent/scripts/$target"
fi

exec bash "$script" "$@"
