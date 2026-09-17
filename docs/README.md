# 文档索引

本索引区分工作流说明、实验成绩和历史记录。文档存在不代表功能已经通过完整训练验证；
运行配置以实际启动参数和日志为准。整理日期：2026-09-15。

## 工作流

| 主题 | 文档 | 代码入口 |
|---|---|---|
| SFT 工具数据与训练 | [SFT 指南](visual_tool_sft.md) | [Qwen3 SFT](../scripts/run_visual_tool_sft_qwen3.sh) |
| Parquet 数据同步 | [同步说明](visual_agent_parquet_sync.md) | [更新脚本](../scripts/update_visual_agent_parquet.py) |
| GroundingDINO RL | [实验执行包](../experiments/visual_agent_execution_plan/README.md) | [RL 入口](../scripts/run_visual_agent_zwz_rl_groundingdino_2node_16gpu.sh) |
| 工具 CPU / GPU 配比测试 | [压测与判读](visual_tool_resource_benchmark.md) | [独立测试入口](../scripts/benchmark_visual_tool_resources.py) |
| ModelArts 16 卡 RL 配比测试 | [三组短训练协议](rl_resource_benchmark_modelarts.md) | [任务入口](../scripts/run_visual_agent_rl_resource_benchmark_modelarts.sh) |
| A/D 双流实验 | [实验配置与协议](../experiments/visual_agent_execution_plan/README.md) | [双流入口](../scripts/run_visual_agent_step130_stream_ablation_32gpu.sh) |
| 三榜评测 | [成绩及统计口径](three_benchmark_evaluation_summary.md) | [通用评测](../scripts/run_visual_agent_eval_qwen3.sh)、[两卡入口](../scripts/run_visual_agent_eval_qwen3_local_2gpu.sh) |
| RL 轨迹审核与 SFT 恢复 | [恢复流程](rl_rollout_recovery.md) | [轨迹转换](../scripts/convert_rl_rollouts_to_sft.py) |
| SLIME 兼容性实验 | [PoC 范围与限制](slime_visual_agent_poc.md) | [PoC 入口](../scripts/run_visual_agent_slime_poc.sh) |
| Perceive2Reason | [使用说明](../PERCEIVE2REASON_README.md)、[实现说明](../PERCEIVE2REASON_IMPLEMENTATION.md)、[设计](../perceive2reason_rl_design.md) | [P2R 入口](../scripts/run_visual_agent_perceive2reason_2node_16gpu.sh) |

P2R 文档含历史分析与预期效果，不能作为最新实验结果；其状态差异见[旧文档审查](archive/README.md)。

## 成绩与历史记录

- [三项 Benchmark 汇总](three_benchmark_evaluation_summary.md) 是成绩主表，包含失败数和异常运行说明。
- [实验执行包](../experiments/visual_agent_execution_plan/README.md) 说明实验目的、配置与状态；其中的成绩摘要需与主表同步。
- [旧 GroundingDINO SFT 统计](qwen3_groundingdino_experiment_summary.md) 保留当时的数据与 loss 分析，不代表最新三榜协议。
- [原始执行计划 DOCX](../Visual_Agent_实验执行计划_修订执行版.docx) 作为计划来源保留；执行状态看实验 README。
- [旧文档审查](archive/README.md) 列出保留、待归档和待核实文档。
- [产物清单](artifact_inventory.md) 记录本轮实测占用及保留建议，不是删除授权清单。

## 代码位置

`scripts/` 是项目启动与工具实现入口；`configs/`、`prompts/` 存放配置和提示词。
`reinforcement_learning/verl/` 是本地修改过的 VERL；`evaluation/VLMEvalKit/` 是本地评测实现。
这两部分的内部文档与测试仍保留在各自目录。项目工具测试在 `tests/`，实验协议测试在
`experiments/visual_agent_execution_plan/tests/`，不为统一外观而搬动它们。

[原始数据说明](../data/README.md) 描述的是上游 DeepEyesV2 数据布局，并未覆盖本地所有数据集。
