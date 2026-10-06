# 16 卡反事实 RL：组构成、重调审计与 v2-lite 数据准备

审计快照：2026-10-06T17:58:59.286464+08:00。使用 `crt16_group_audit.json` 的 step1–14，共 28,224 条轨迹；每组 16 条。

## 组构成

“全对/全错/混合”按答案 acc 分类；“reward 有差异”按实际 score 分类。格式分变化也能造成 score 差异，不能视为答题能力的学习信号。

| 来源 | 分支 | 前缀工具 | 组数 | 全对 | 全错 | 答题混合 | 混合比例 | reward 有差异 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ChartQA | counterfactual | grounding_detect | 1 | 0 | 1 | 0 | 0.00% | 0 |
| ChartQA | counterfactual | ocr_read | 12 | 3 | 7 | 2 | 16.67% | 4 |
| ChartQA | factual | grounding_detect | 4 | 3 | 1 | 0 | 0.00% | 0 |
| ChartQA | factual | object_count | 1 | 0 | 1 | 0 | 0.00% | 0 |
| ChartQA | factual | ocr_read | 16 | 4 | 7 | 5 | 31.25% | 7 |
| DeepEyesV2 | counterfactual | grounding_detect | 11 | 3 | 4 | 4 | 36.36% | 4 |
| DeepEyesV2 | factual | grounding_detect | 10 | 3 | 5 | 2 | 20.00% | 3 |
| DeepEyesV2 | factual | ocr_read | 6 | 2 | 1 | 3 | 50.00% | 3 |
| depth | counterfactual | grounding_detect | 26 | 3 | 10 | 13 | 50.00% | 13 |
| depth | factual | grounding_detect | 18 | 4 | 9 | 5 | 27.78% | 5 |
| FSC147 | counterfactual | object_count | 15 | 0 | 14 | 1 | 6.67% | 1 |
| FSC147 | factual | object_count | 25 | 8 | 17 | 0 | 0.00% | 0 |
| OCR | counterfactual | grounding_detect | 3 | 2 | 0 | 1 | 33.33% | 1 |
| OCR | counterfactual | ocr_read | 10 | 2 | 5 | 3 | 30.00% | 3 |
| OCR | factual | grounding_detect | 5 | 3 | 0 | 2 | 40.00% | 2 |
| OCR | factual | ocr_read | 16 | 13 | 2 | 1 | 6.25% | 1 |
| TallyQA | counterfactual | grounding_detect | 10 | 6 | 0 | 4 | 40.00% | 4 |
| TallyQA | counterfactual | object_count | 2 | 0 | 1 | 1 | 50.00% | 1 |
| TallyQA | factual | grounding_detect | 19 | 14 | 3 | 2 | 10.53% | 2 |
| TallyQA | factual | object_count | 4 | 3 | 1 | 0 | 0.00% | 0 |
| ZWZ | counterfactual | grounding_detect | 78 | 14 | 25 | 39 | 50.00% | 40 |
| ZWZ | factual | grounding_detect | 73 | 16 | 28 | 29 | 39.73% | 31 |

OCR/ocr_read 的反事实混合组为 3/10（30%）；ChartQA/ocr_read 为 2/12（16.67%）。ChartQA 的 reward 有差异组为 4/12，但其中两组全部答错，仅存在格式分差异。TallyQA/object_count 为 1/2，样本很少。现有数据支持先做范围更窄的方案 B 试验，不能认定 OCR 和 ChartQA 均已有稳定的强信号。

## 同工具换参数重调

以下按反事实轨迹计数，分母包含该来源全部反事实轨迹。“正常响应”指调用成功且未标记 `replayed_fault`，不代表工具输出等于真值。

| 来源 | 轨迹数 | 换参数获得正常响应 | 占比 | 换参数且最终答对 | 同图 query 词集合包含关系 | 恢复原 factual 框（精确匹配） |
|---|---:|---:|---:|---:|---:|---:|
| DeepEyesV2 | 176 | 29 | 16.48% | 11 | 5 | 0 |
| depth | 416 | 379 | 91.11% | 122 | 47 | 0 |
| ZWZ | 1248 | 1238 | 99.20% | 534 | 30 | 0 |

重调频率很高，但常见变化是在查询另一个对象，例如 depth 的 box → cabinets、ZWZ 的 tie → microphone，可能属于正常任务分解。确有同一对象改写的候选，例如 ivy-covered wall → wall with ivy、jacket → blue jacket。词集合包含关系只是复核线索，既不能覆盖所有同义改写，也不保证查询对象相同。

本次未发现重调响应的非空 boxes 与原 factual boxes 精确一致；该严格条件不排除近似框恢复，且原 factual 输出也不保证正确。因此无法从日志给出可靠的“恢复真值比例”，不能把全部重调率称为已证实绕过，也不能将零精确匹配解释为没有绕过。当前代码仅对工具名和参数字典完全一致的请求重放错误观察，这是需要纳入结果解释的限制。

## v2-lite 数据

用户确认严格只保留三种来源与前缀工具组合，并将替换比例设为 17%。未启动新训练。

- Profile：[configs/tool_reliance_v2_lite.json](configs/tool_reliance_v2_lite.json)。
- 数据目录：`data/vlocr_reliance_pairs_v2_lite_20261006`。
- 基础数据：`data/zwz_multitool_relation20_hme_chartqa_tallyhalf_fsc3000_20261004`。
- 前缀来源：原 64 卡 `qwen3base_multitool_vlocr_hme_chartqa_tallyhalf_fsc3000_n16_8node_20261003T202135115738_f9c41af7` 的 step1–53。
- 构造脚本：`scripts/prepare_reliance_pairs.py`；`--fraction 0.17 --seed 20261005`。

严格筛选、沿用现有选择算法与随机种子时，最多有 2,936 对可用前缀，低于原 20% 替换需要的 3,392 对。本次选用 2,883 对。

| 来源 × 前缀工具 | 对数 | factual 行 | counterfactual 行 |
|---|---:|---:|---:|
| ChartQA × ocr_read | 1,367 | 1,367 | 1,367 |
| OCR × ocr_read | 1,250 | 1,250 | 1,250 |
| TallyQA × object_count | 266 | 266 | 266 |

训练共 33,921 行：普通 28,155，factual 2,883，counterfactual 2,883，比例约为 83% / 8.5% / 8.5%。验证 1,312 行，训练加验证 35,233 行。只限制成对数据的来源，普通数据仍包含原有七类任务。

已核验全部 2,883 对的来源、前缀工具、同题配对、扰动内容和重放参数；验证文件与基础数据逐字节一致，旧训练数据哈希保持一致，Parquet 和 JSONL 行数正确。无需修改训练代码。

## 工作量与预算

CPU 审计和数据构造已完成，成本为分钟量级。若后续沿用现有 16 卡、10 小时训练配置，一组训练预算约 160 卡时，另加初始化和收尾。与同预算普通 RL 做对照共约 320 卡时；单组 v2-lite 与已有混合反事实运行的比较，不能替代普通 RL 对照。

方案 A 的强制核查分支不在本次改动范围内，尚未实现或启动。数据已准备好不代表方法效果已得到验证。

机器可读统计：[crt16_v2_lite_audit.json](crt16_v2_lite_audit.json)。

独立训练入口已准备：[v2-lite 16 卡启动脚本](scripts/run_visual_agent_multitool_vlocr_reliance_v2_lite_2node_16gpu_modelarts.sh)。默认从 Qwen3-VL 原始权重开始，禁用断点恢复，使用 v2-lite 数据与独立运行名，每 5 步验证和保存，训练计时 10 小时后完成当前步并保存退出。
ModelArts 提交文件：`/home/ma-user/work/algorithm/codebkp/run_visual_agent/run_visual_agent_multitool_vlocr_reliance_v2_lite_2node_16gpu_modelarts.sh`。沿用 v1 的软链接与运行环境初始化；尚未启动训练。
