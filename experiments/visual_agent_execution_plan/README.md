# Visual Agent 实验执行包

这个目录把《Visual Agent 实验执行计划》的前四步落为可审计的实验工具，并通过仓库脚本接入现有训练与评测链路。

## 包含内容

- `init` 生成 B0–B4 checkpoint 登记表和冻结的评测协议模板。
- `make-split` 按源图像稳定分组，输出训练、开发和测试清单；同一源图不会跨 split。
- `validate-log` 校验逐题 JSONL 日志。Native 模式的工具调用会被拒绝，缺少坐标映射、结束原因或 token/耗时字段也会被报告。
- `summarize-utility` 对同一题目的 ON/OFF 结果做配对聚合，输出 `delta_tool`、工具率、有效调用率及按题 bootstrap 置信区间。
- `check-fairness` 比较单流 A 与双流 D 配置，阻止它们使用不同父 checkpoint、奖励、数据池、冻结范围、KL 或预算口径。
- `dual_stream.py` 提供按 `(sample_id, stream_id, batch_id)` 计算 GRPO outcome advantage，以及 `L_A`/`L_N` 的等权或 alpha 加权参考实现。
- `scripts/run_visual_agent_step130_stream_ablation_32gpu.sh` 已接入 trainer：两路都基于同一旧策略生成，分别分组计算 advantage，并在每个 PPO mini-batch 中交错合并；Native 分支使用独立 prompt 且不会携带或执行工具调用。
- Native prompt 必须输出严格的 `<answer>relation label</answer>`；训练器会分别记录两路 reward 指标，并在整批 Native format reward 为零时立即终止。
- A/D 训练成功后由 0 号节点自动释放训练服务，并用最终 HuggingFace checkpoint 以 Agent/tool-on 口径评测 `VStarBench`、`HRBench4K`、`HRBench8K`。

## 实验结果（截至 2026-09-15）

以下均为 Agent/tool-on 评测；分数单位为 %，HRBench 取 `Average / all`。
API 失败数依次为 VStar / HR4K / HR8K，样本总数为 191 / 800 / 800。

| 权重 / 实验 | VStar | HR4K | HR8K | API 失败数 | 状态 |
|---|---:|---:|---:|---:|---|
| 起点 DINO + KL step130（补测后） | 91.10 | 83.00 | 79.88 | 0/0/0 | 旧成功预测与失败补测合并 |
| D：Agent4 + Native4，100 step | 90.58 | 81.75 | 78.13 | 0/0/0 | Native 奖励链路失效的异常运行，不作为有效双流对照 |
| D：Vision-OPD，54 step（9 月 15 日补测后） | 87.96 | 79.13 | 78.88 | 0/50/22 | HRBench 仍有失败，比较尚不完整 |
| A：Agent-only Vision-OPD，54 step | 无效 | 无效 | 无效 | 189/796/792 | 几乎全部 API failed，日志有 vLLM OOM |

`nativefmtfix` 续训到 step100 的运行仅有 VStar 结果文件：3.14%，其中
183/191 条 API failed，HRBench 无完整成绩，暂不纳入三榜比较。

最新 Vision-OPD A/D 评测使用旧 12 轮配置；这些成绩不是本次 loss mask、图片位置编码
修复及默认 6 轮设置后的结果。历史失败样本补测应保留原运行参数；若改用 6 轮，需要
全量重评对应权重，不能只替换失败行。当前没有有效结果证明 A 或 D 更优。

完整历史表和结果目录见 [三项 Benchmark 评测汇总](../../docs/three_benchmark_evaluation_summary.md)。

## 快速开始

```bash
PLAN_ROOT=experiments/visual_agent_execution_plan
PYTHONPATH="$PLAN_ROOT" python -m visual_agent_experiments init --output-dir "$PLAN_ROOT/runs/bootstrap"
PYTHONPATH="$PLAN_ROOT" python -m visual_agent_experiments make-split \
  --input data/train.jsonl --output "$PLAN_ROOT/runs/split_manifest.json"
PYTHONPATH="$PLAN_ROOT" python -m visual_agent_experiments validate-log \
  --input "$PLAN_ROOT/runs/dev_predictions.jsonl"
PYTHONPATH="$PLAN_ROOT" python -m visual_agent_experiments summarize-utility \
  --input "$PLAN_ROOT/runs/dev_on_off.jsonl" --output "$PLAN_ROOT/runs/utility.json"
PYTHONPATH="$PLAN_ROOT" python -m visual_agent_experiments check-fairness \
  --agent-config "$PLAN_ROOT/configs/train_A.example.json" \
  --dual-config "$PLAN_ROOT/configs/train_D.example.json"
```

`make-split` 只检查精确的 source image 分组；近重复图像仍需通过已有数据审查流程写入同一个 `source_group_id`。`summarize-utility` 要求每个预定题都有 ON 与 OFF 结果，避免只统计成功调用子集。

## 日志协议

每行必须至少包含 `run_id`、`checkpoint`、`split`、`sample_id`、`source_image_id`、`mode`、`seed`、`correct`、`tool_calls`、`finish_reason`、`generated_tokens`、`visual_tokens` 与 `elapsed_seconds`。工具调用需附带 `name`、`status`、`image_id`、`bbox`、`coordinate_system`；`bbox` 可为 `null`，但此时应说明失败状态。

`mode` 使用 `native` 或 `agent`。工具收益分析额外使用 `condition` 为 `on` 或 `off`；Native 模式不能有工具调用。所有输出按 `run_id` 放到本目录的 `runs/`，该目录已被忽略。
