# FSC 计数后端与 MME 提示词 Agent 对照

执行顺序：统一纠正 FSC 评分基线 → 自动示例框 Agent 全量对照 → 验证分块计数 → 根据结果决定 observation 格式及 RL 训练。前两步完成前，不调整计数数据配比或采信奖励。

评分基线已完成：9 组已有预测统一按官方 `annotation_FSC147_384.json` 的 `points` 数重评分。历史预测及评分文件保留，实验总表已经更新；无效输出处理不变。详见 [评分基线报告](../outputs/vlmeval/fsc_baseline_corrected_20261009/report.md)。

## ModelArts 启动

- 资源：单节点 8×A100，0–6 卡运行模型副本，7 卡运行工具。
- 启动文件：`run_visual_agent/eval/run_visual_agent_eval_fsc_count_ab_8gpu_modelarts.sh`。
- 本地部署位置：`/home/ma-user/work/algorithm/codebkp/run_visual_agent/eval/run_visual_agent_eval_fsc_count_ab_8gpu_modelarts.sh`。
- 仓库入口：[run_visual_agent_eval_fsc_count_ab_8gpu_modelarts.sh](../scripts/run_visual_agent_eval_fsc_count_ab_8gpu_modelarts.sh)。
- 使用之前评测任务相同的镜像及共享存储挂载；不设评测超时。

固定新数据 64 卡 step80：`qwen3base_multitool_vlocr_hme_chartqa_tallyhalf_fsc3000_n16_8node_20261003T202135115738_f9c41af7/global_step_80/actor/huggingface`。

两组均重新推理全部 1190 题，依次运行 `text_only`、`auto_exemplar`。只切换 CountGD++ 后端；权重、提示词、原生工具协议、8 个回合、每回合 512 token、temperature=0、API 并发 7、只返回 count 的格式均相同。本阶段不启用自适应分块。参数和输入文件哈希记录于 `protocol.json`。

## 结果与验收

结果目录：`outputs/vlmeval/fsc_count_ab_8gpu/<run_id>/`。

- `text_only/`、`auto_exemplar/`：两组预测、评分与工具服务日志。
- `status.tsv`：退出状态及端到端耗时，包含数据校验、服务启动、推理、评分和进程退出。
- `comparison.json`、`report.md`：两组完成后自动汇总 MAE、RMSE、MAE_valid、RMSE_valid、无效输出数和耗时。
- `paired_examples.csv`：逐题预测、计数和绝对误差变化；另统计改善、退化、误差不变、有效转无效、无效转有效及两组均无效。

存在无效输出时，不将 MAE_valid 当成全量 MAE，也不只凭平均分判断收益。后续分块实验使用同样指标，并与这次结果分开记录。

## 启动状态

2026-10-09（提交前）：入口及汇总程序已通过配置预检和评分测试，尚未提交 ModelArts。当前会话未配置可用的 ModelArts 提交客户端及认证。

此前本地 GPU 1 尝试因其他进程占用显存，在 vLLM 初始化时退出，没有产生新预测。失败记录保存在 `outputs/vlmeval/fsc_agent_count_ab_20261009/`；ModelArts 入口使用全新的结果目录。

## ModelArts 运行结果（2026-10-09）

运行 ID：`fsc_count_ab_step80_20261009T173025567789537_353`。完整产物位于 [本次结果目录](../outputs/vlmeval/fsc_count_ab_8gpu/fsc_count_ab_step80_20261009T173025567789537_353/)；FSC 和 MME 的四组子任务退出码均为 0。以下是同一 step80 权重的全量 Agent 重新推理结果，不与前期[直接调用计数工具的诊断](diagnostic_results_20261009/count_mme_diagnostic_20261009/report.md)混作同一对照。

### FSC147_TEST：计数后端 A/B

两组各覆盖官方测试集 1190 题，均无无效答案或 API 失败。只切换 CountGD++ 的纯文本与自动示例框后端；逐题结果见 `paired_examples.csv`，完整指标见 `comparison.json`。

| 后端 | MAE ↓ | RMSE ↓ | 完全正确率 | 无效答案 | 端到端耗时 |
|---|---:|---:|---:|---:|---:|
| text_only | 15.2067 | 129.3733 | 32.35% | 0 | 877 秒 |
| auto_exemplar | 12.4193 | 98.8672 | 33.03% | 0 | 859 秒 |

自动示例框使 MAE 下降 2.7874（18.33%）、RMSE 下降 23.58%。逐题绝对误差改善 306 题、变差 308 题、持平 576 题，两组误差中位数均为 2。误差净减少 3317，其中改善最大的 5 题贡献 2200（66.33%）；平均收益主要来自少数大误差题，不能解释为逐题普遍改善。

### MME-RealWorld-Lite：提示词 A/B

FSC 汇总后，脚本将计数后端恢复为原配置，在相同权重上分别用原提示词 `baseline` 和证据提示词 `evidence` 全量重跑 1919 题。评分文件的原始 Overall 分别为 54.4554%（1045/1919）和 54.5597%（1047/1919）：evidence 多对 2 题，配对变化为 78 题错变对、76 题对变错。

但原始分数包含误判。baseline 有 11 个 API 失败和 3 个回合耗尽输出；evidence 分别有 12 个和 7 个。两组 API 失败均报输入超过模型 32768 token 上限。MME 评测器从预测文本中提取首个大写 A–E，因此将 `Failed to obtain answer via API.` 和 `Agent exceeded the maximum of 8 turns` 中的首字母 A 当作答案，分别把 baseline 的 1 个、evidence 的 4 个失败输出计为正确。仅将这些明确失败的输出改计为 0 后，baseline 为 **54.4033%（1044/1919）**，evidence 为 **54.3512%（1043/1919）**；这一步只是核算，原评分文件未改写。

这次对照不能证明证据提示词优于原提示词。MME 的原始评分文件位于 `mme_baseline/dino_latest/` 和 `mme_evidence/dino_latest/`，失败详情位于各组的 `failure_traces/`；评分逻辑见 `evaluation/VLMEvalKit/vlmeval/dataset/image_mcq.py` 的 `MMERealWorld.evaluate` 与 `evaluation/VLMEvalKit/vlmeval/dataset/utils/multiple_choice.py` 的 `extract_characters_regex`。
