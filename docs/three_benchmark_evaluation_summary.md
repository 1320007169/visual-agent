# 三项 Benchmark 评测汇总

## 统计口径

只整理同时生成 VStarBench、HRBench4K、HRBench8K 三项结果的权重。分数均为准确率
（%）：VStar 取 `Overall`，HRBench 取 `Average / all`。

“跑完三榜”只表示结果文件和样本行数完整，不代表所有 API 请求成功。表中最后一列依次
记录 VStar / HR4K / HR8K 的 API 失败数；`0/0/0` 才是无 API 失败的完整结果。同一权重
的连续补测只保留最后或最完整的一次。

## 基线与 SFT

| 权重 | 评测时间 | VStar | HR4K | HR8K | API 失败数 |
|---|---:|---:|---:|---:|---:|
| Qwen3 base（补测完成） | 2026-07-28 | 84.29 | 76.38 | 70.62 | 0/0/0 |
| Combined-8679 SFT | 2026-07-22 | 68.06 | 67.38 | 64.25 | 0/0/0 |
| RFT-SFT，纯工具 10,953 | 2026-09-18 | 85.34 | **79.13** | **75.00** | 0/11/30 |
| RFT-SFT，工具/直答 1:1 Mixed 21,906 | 2026-09-18 | **86.91** | 78.50 | 74.13 | 0/3/12 |

Qwen3 base 的 7 月 22 日旧结果有 `8/49/86` 个 API 失败，已由 7 月 28 日零失败结果
替代。Combined-8679 SFT 在三榜上均低于补测后的 base。

两组 RFT-SFT 均从 Qwen3-VL-8B-Instruct 出发，使用 `2e-6` 学习率训练 1 epoch。
纯工具版使用 10,953 条经过视觉 judge 和泄漏复查的 RL 成功轨迹；1:1 Mixed 版再混入
10,953 条直接回答样本。表中记录的是当前 Agent/tool-on 协议的原始分数，模型在几乎所有
成功请求中都调用了 GroundingDINO/crop 工具；Qwen3 base 则是 `use_tools=false` 的直答
协议，因此两者不能视为严格同口径的权重 A/B。RFT-SFT 的 HRBench API 失败行仍按错误
计入准确率，补测前应将其视为略低估的系统分数。

RFT-SFT 结果目录：

- `outputs/vlmeval/qwen3_rl_rejection_sft_1to1_comparison/rl_rejection_clean10953_direct10953_mixed21906_lr2e6_1epoch_20260918_005022/`

## 原始关系 RL

使用原始空间关系数据训练。两个 step 40 来自不同训练运行，因此分别保留。

| 权重 | 评测时间 | VStar | HR4K | HR8K | API 失败数 |
|---|---:|---:|---:|---:|---:|
| RL step 20 | 2026-07-22 | 77.49 | 71.25 | 62.38 | 0/0/0 |
| RL step 40（早期 2 节点） | 2026-07-23 | 81.68 | 73.75 | 67.25 | 0/0/0 |
| RL step 64（4 节点续训） | 2026-07-24 | 77.49 | 72.12 | 67.50 | 0/0/0 |
| RL step 40（后期版本） | 2026-07-27 | 80.10 | 72.88 | 67.38 | 3/49/80 |
| RL step 50 | 2026-07-27 | **82.20** | 75.12 | 69.12 | 4/16/47 |
| RL step 55 | 2026-07-27 | 81.15 | **76.50** | 68.12 | 0/0/0 |
| RL step 60 | 2026-07-28 | 81.68 | 74.38 | 69.87 | 0/0/0 |
| RL step 65 | 2026-07-28 | 79.58 | 75.37 | 69.50 | 0/0/0 |
| RL step 80 | 2026-07-28 | 79.06 | 74.75 | **71.62** | 0/0/0 |
| RL step 90 | 2026-07-28 | 74.35 | 75.25 | 69.12 | 0/0/0 |

零失败结果中，step 55 的 HR4K 最好，step 80 的 HR8K 最好；没有一个 checkpoint
在三榜上同时最优。step 50 的 VStar 最高，但该次仍有 API 失败。

## GroundingDINO RL

`DINO-only` 只引入 GroundingDINO；`DINO + KL` 额外加入 KL 约束。

| 权重 | 评测时间 | VStar | HR4K | HR8K | API 失败数 |
|---|---:|---:|---:|---:|---:|
| DINO-only RL step 120 | 2026-07-31 | 86.39 | 80.25 | 71.37 | 1/13/91 |
| DINO + KL RL step 30 | 2026-08-04 | 81.15 | 77.25 | 69.25 | 0/24/71 |
| DINO + KL RL step 50 | 2026-08-04 | 85.86 | 78.88 | 73.37 | 0/1/12 |
| DINO + KL RL step 60 | 2026-08-05 02:31 | 87.96 | 79.75 | 76.12 | 0/1/32 |
| DINO + KL RL step 80 | 2026-08-05 14:03 | 87.43 | 81.62 | 76.00 | 0/1/44 |
| DINO + KL RL step 90 | 2026-08-05 20:50 | 89.01 | 79.12 | 75.63 | 0/0/38 |
| DINO + KL RL step 100 | 2026-08-06 02:00 | 88.48 | 80.88 | 74.50 | 0/1/49 |
| DINO + KL RL step 130（旧评测） | 2026-08-06 16:37 | **91.10** | 82.50 | 71.62 | 0/5/91 |
| DINO + KL RL step 130（失败样本补测） | 2026-09-12 | **91.10** | **83.00** | **79.88** | 0/0/0 |
| DINO + KL RL step 165 | 2026-08-07 13:33 | 89.53 | 81.75 | 76.12 | 0/0/27 |

step 130 就是此前表现突出的 KL 权重。补测后 VStar、HR4K、HR8K 分别为 91.10、
83.00、79.88，结果文件包含完整的 191/800/800 条样本，未发现空预测或 API 失败标记。
这次补测复用了旧评测中的成功预测，只重新生成失败样本，并非一次全量重跑；因此适合修正
旧评测的 API 失败，但与采用当前代码全量重跑的新实验比较时仍需注明这一差异。

这些结果曾被统一写进 `qwen3_dino_kl_step80/.../dino_kl_step50` 目录。这里根据训练日志
中的 checkpoint 导出时间重新映射：step 130 于 2026-08-06 15:44 导出，对应 16:37
开始的评测。2026-08-11 的一组结果无法从现有日志确认实际 checkpoint，暂不纳入主表。

## Mixed SFT + RL

以 mixed SFT 为起点继续 GroundingDINO + KL RL；后期版本加入 Thyme、slack 混合数据
和较低学习率。

| 权重 | 评测时间 | VStar | HR4K | HR8K | API 失败数 |
|---|---:|---:|---:|---:|---:|
| Mixed-56874 + RL step 100 | 2026-09-03 | 76.96 | 63.12 | 58.00 | 6/120/118 |
| Thyme/slack mixed + RL step 150（补测） | 2026-09-11 | 80.10 | 68.88 | 55.50 | 2/54/212 |
| Thyme/slack mixed + RL step 165 | 2026-09-10 | **80.10** | **74.13** | **69.75** | 0/0/0 |

step 150 的补测只比原结果少 `1/1/2` 条 API 失败，仍不完整。step 165 是该系列目前唯一
三榜零失败的结果，也在 HR4K、HR8K 上明显优于 step 150。

## RFT-SFT 1:1 Mixed + RL（2026-09-20）

从 RFT-SFT 工具/直答 1:1 Mixed 21,906 权重继续进行 GroundingDINO + KL RL。以下为
当前 6 回合、每回合最多 6,144 tokens 的 Agent/tool-on 全量评测。

| 权重 | 评测时间 | VStar | HR4K | HR8K | API 失败数 |
|---|---:|---:|---:|---:|---:|
| Mixed-21906 + RL best step40 | 2026-09-20 | 89.53 | **82.88** | 78.50 | 0/0/0 |
| Mixed-21906 + RL final step96 | 2026-09-20 | **90.05** | 82.50 | **79.38** | 0/0/0 |

step96 相比 step40 的 VStar、HR4K、HR8K 分别变化 `+0.52 / -0.38 / +0.88` 个百分点。
两份结果均包含完整的 191/800/800 条样本，最终预测中未发现 API 失败标记。

同一任务还在 Qwen3 base 上评测四个扩展 benchmark，当前已完成结果如下。这里的 base
同样使用 Agent/tool-on 协议，不等同于“基线与 SFT”表中的直答 base。

| 权重 | OCRBench | ChartQA_TEST | MME-RealWorld-Lite | MME-RealWorld-CN |
|---|---:|---:|---:|---:|
| Qwen3 base（Agent/tool-on） | 83.70 | 预测完成，评分待补 | 41.53 | 57.34 |

ChartQA 已生成完整预测文件，但本次评分因缺少 Judge API key 未生成 `_acc.csv`；不将其
记作零分。最终预测中，OCRBench 残留 4 条 API failed；ChartQA 没有 API failed，但有
2 条空预测；MME-RealWorld-Lite 没有 API failed，但有 1 条无法评分并按 0 计；
MME-RealWorld-CN 残留 8 条 API failed，另有 1 条无法评分并按 0 计。因此这些扩展榜分数
是当前故障口径结果，补测前应视为略低估。

结果目录：

- `outputs/vlmeval/mixedsft_rl_best40_step96/mixedsft_rl_3bench_qwen3base_new4_20260920_174347/`

### Mixed-21906 + RL：16 卡与 64 卡对比（2026-09-22）

两轮均从 Mixed-21906 SFT 出发。VStar/HR8K 使用同一套 6 回合、每回合最多
6,144 tokens 的 GroundingDINO Agent 独立评测；16 卡的 HR4K 来自训练内验证，
64 卡的 HR4K 来自独立评测，不应将 HR4K 的小幅分差视为严格同口径比较。

| 训练任务 / 权重 | VStar | HR4K | HR8K | API 失败数（VStar/HR4K/HR8K） |
|---|---:|---:|---:|---|
| 64 卡 best step40 | 89.53 | 82.88 | 78.50 | 0/0/0 |
| 64 卡 final step96 | 90.05 | 82.50 | **79.38** | 0/0/0 |
| 16 卡 step160 | 89.53 | **83.13（训练内）** | 78.00 | 0/不适用/0 |
| 16 卡 final step192 | **92.15** | 82.75（训练内） | 77.38 | 0/不适用/0 |

16 卡两次独立评测分别包含完整的 191 条 VStar 和 800 条 HR8K 样本，均无 API
失败或空预测。step192 相比 step160，VStar 提升 2.62 个百分点，HR8K 下降
0.63 个百分点；相比 64 卡 final step96，VStar 高 2.09 个百分点，HR8K 低
2.00 个百分点。16 卡 HR4K 若要与 64 卡严格比较，仍需按同一独立评测流程补测。

16 卡结果目录：

- step160：`outputs/vlmeval/deepeyesv2_n8_2node_step160_other2/deepeyesv2_n8_2node_step160_other2_20260921_225614/`
- step192：`outputs/vlmeval/deepeyesv2_n8_2node_step192_other2/deepeyesv2_n8_2node_step192_other2_20260922_140054/`

### Qwen3 base 五工具提示词（2026-09-21）

同一 Qwen3-VL-8B-Instruct 起点，以五工具提示词运行 6 回合 Agent 评测。VStar、HR4K、
HR8K 分别为 **73.30 / 76.88 / 70.00**；按同一顺序 API 失败数为 `0/0/0`。
OCRBench 为 **835/1000**（0 条 API 失败），MME-RealWorld-Lite 为 **44.29%**
（0 条），MME-RealWorld-CN 为 **58.68%**（1 条）。这不是表中 Qwen3 base
`use_tools=false` 的直答协议；与 84.29 / 76.38 / 70.62 的直答三榜对照仅作参考，
不能将分差直接归因于工具本身。

OCR 使用 PP-OCRv5 Server，而非 PaddleOCR-VL。唯一六榜工作簿记录了 60 次可恢复
`ocr_read` 的 `no_text` 返回，另有 42 次工具请求错误（全部在 CN：41 次图片索引越界、
1 次错误传递 `slack_ratio` 给检测工具）；CN 的 1 条 `api_failed` 与工具错误分开计数。
详细的样本数、工具调用率和结果路径见
`docs/qwen3_vl_8b_four_tool_prompt_experiment.md`。

### SFT/RL 轨迹与 query 临时诊断（2026-09-21）

对 RFT-SFT 工具/直答 1:1 Mixed 21,906 起点及其 RL final step96 的三榜轨迹使用同一
规则统计。SFT 的 VStar 使用正式结果；HR4K/HR8K 的补测合并文件未保留 `raw_response`，
因此 SFT 的 HR 轨迹统计来自补测前 `mixed_1to1_21906_retry_backup_20260918_143238`。
它适合分析模型行为，但其中的原始对错数不能替代上表的补测后分数。

| 行为指标 | Mixed SFT 21,906 | RL final step96 |
|---|---:|---:|
| 调用任意工具的样本 | 1,772 / 1,791 | 1,774 / 1,791 |
| 调用 GroundingDINO 的样本 | 1,755 / 1,791 | 1,755 / 1,791 |
| GroundingDINO 调用次数 | 2,778 | 2,752 |
| 正常响应但空框的调用 | 429 | 406 |
| 空框后继续/改写检测 | 133 | 2 |
| 空框后立即使用估计坐标 crop | 290 | 379 |
| 颜色题将待求颜色写进 query | 56 / 416 | 73 / 416 |
| 使用 `location N`、`item N` 等抽象 query 的样本 | 161 | 159 |
| 超出 6 回合且没有最终答案 | 0 | 4 |

两版均未发现 GroundingDINO 服务异常；这里的失败是工具正常响应但返回空框。两版工具
调用率都接近 99%，说明“几乎每题先调用工具”的策略在 SFT 后已经形成，1:1 混入直接
回答数据并未恢复按需调用。RL 没有显著增加 DINO 空框总量，但将空框后的行为从改写 query
重试明显推向“模型自行估计坐标后直接 crop”，并增加了把待求颜色写入 query 的确认偏差。

典型问题包括：颜色题查询 `red bike`、`woman in white dress` 等含候选答案的短语；地图
题查询 `location 1`、`item 6` 等非具体物体；关系题查询
`object directly to the right of ...` 等完整关系描述。还存在空框后使用极小估计框 crop、
连续调用工具耗尽回合预算的问题。现有 prompt 已要求按需调用、具体名词、关系目标分别
定位以及空框后改写重试，因此这不是单靠追加 prompt 规则即可解决的问题。

当前临时结论是：问题主要源于 SFT/RFT 轨迹中已经存在的工具强制化和 query 捷径，RL
在仅重视最终正确率时进一步强化了这些行为。后续调整 SFT 数据时，应优先过滤“待求属性
进入 query”、抽象编号、关系整句、空框后盲目小框 crop 和超回合轨迹，同时保留足够的
直接回答样本。空框与答错目前只可视为相关性；在完成低阈值或改写 query 的反事实重跑前，
不把它们全部归因为 GroundingDINO 本身。

## Step130 双流续训

实验 D 从保留的 DINO + KL step130 权重继续训练 100 个 outer step。每个输入分别生成
Agent K=4 和不调用视觉工具的 Native K=4 轨迹，再合并更新同一策略。

| 权重 | 评测时间 | VStar | HR4K | HR8K | API 失败数 |
|---|---:|---:|---:|---:|---:|
| DINO + KL RL step 130（失败样本补测） | 2026-09-12 | 91.10 | 83.00 | 79.88 | 0/0/0 |
| D：Agent+Native 双流续训 step 100（异常运行） | 2026-09-13 | 90.58 | 81.75 | 78.13 | 0/0/0 |

D 的评测是当前代码下的全量评测，包含完整的 191/800/800 条样本，未发现空预测或 API
失败标记。与 step130 补测结果相比，三项分别变化 -0.52、-1.25、-1.75 个百分点，未观察
到提升。

训练后检查全部 89,600 条 rollout，发现旧 Native prompt 未要求 `<answer>...</answer>`，
导致 44,800 条 Native 轨迹的 `score/acc/format` 全为 0，11,200 个 Native GRPO 组的
advantage 也全为 0。该运行实际相当于 Agent RL 加 Native 文本上的 KL 约束，不是有效的
双流 RL；三榜分数可以描述这个异常权重，但不能用于判断双流方案是否有效。修复后必须从
原 KL step130 权重重新运行 D，不能从这个异常 step100 继续训练。

需要注意：D 使用的仍是原 KL 训练使用过的
`data/zwz_rl_vqa/rl_original_relation/train.parquet`（18,518 条），并不是一批新训练数据。
因此该实验衡量的是“在相同训练集上继续加入双流目标”的效果，结果下降可能包含重复训练或
过拟合影响。并且 step130 对照是“旧成功结果 + 失败样本补测”的合并结果；若要做严格 A/B
结论，应使用当前代码对 step130 起始权重进行一次全量三榜重跑。

上述旧关系数据实验 A（Agent-only）在 5 小时时限内训练到 step 77 后退出，没有 step100 或最终三榜结果，
暂不写入结果表。

## Vision-OPD 续训（截至 2026-09-15）

以下 A/D 从 step130 出发，均训练到 step54，并使用 Agent/tool-on 口径评测。

| 权重 | 评测时间 | VStar | HR4K | HR8K | API 失败数 |
|---|---:|---:|---:|---:|---:|
| DINO + KL step130（对照，失败样本补测） | 2026-09-12 | 91.10 | 83.00 | 79.88 | 0/0/0 |
| D：Agent4 + Native4，Vision-OPD step54（补测后） | 2026-09-15 | 87.96 | 79.13 | 78.88 | 0/50/22 |
| A：Agent-only，Vision-OPD step54 | 2026-09-15 | 无效 | 无效 | 无效 | 189/796/792 |

D 的 VStar 已无 API 失败，HR4K、HR8K 仍有 50/800、22/800 条失败；分数包含这些失败行，
不能视作无故障评测。A 的结果文件虽为全零，但几乎所有预测都是 API failed，且日志中有
视觉编码阶段的 vLLM CUDA OOM，因此不将全零当作模型准确率。D 原评测也出现过 OOM，
不能把故障解释为 Agent-only 训练特有的问题。两组尚不足以支持有效的 A/D 优劣结论。

另外，`nativefmtfix_100step_16gpu_from_64gpu_step80` 的 step100 仅生成 VStar 结果：
3.14%，其中 183/191 条 API failed；未获得完整三榜结果，不作为模型退化证据。

结果目录（相对于仓库根目录）：

- D 补测：`outputs/vlmeval/visual_agent_execution_plan/step130_D_vision_opd_step54_failed_retry_20260915_102235/`
- A 原评测：`outputs/vlmeval/visual_agent_execution_plan/step130_agent4_vision_opd_54step_64gpu_final_step54_A_3bench/`
- nativefmtfix：`outputs/vlmeval/visual_agent_execution_plan/step130_dual_agent4_native4_nativefmtfix_100step_16gpu_from_64gpu_step80_final_step100_D_3bench/`

这批 Vision-OPD A/D 成绩来自旧 12 轮协议，不是本次工具返回 loss mask、图片位置编码修复
以及默认 6 轮设置后的结果。历史补测需保持原配置；使用 6 轮比较时应全量重评，不能混用
旧 12 轮成功预测和新 6 轮失败补测。

## 16 卡混合数据 RL step160：工具使用诊断（2026-09-17）

三组均使用 `zwz_deepeyesv2_3k_nocount_v1_n8_2node/global_step_160/actor/huggingface`。
分数来自各组评分 CSV；三组最终进程退出码均为 0，但尚未逐条核验 API 失败及空预测，
因此不将正常退出等同于 `0/0/0` API 失败。

| 模式 | 最大 assistant 回合 | 每回合 tokens | VStar | HR4K | HR8K | API 失败数 | 状态 |
|---|---:|---:|---:|---:|---:|---|---|
| direct | 1 | 6144 | 84.82 | 79.63 | **75.88** | 待逐条核验 | 补测完成，退出码 0 |
| auto | 6 | 6144 | 86.91 | 78.88 | 75.13 | 待逐条核验 | 完成，退出码 0 |
| tool_first | 6 | 6144 | **87.43** | **80.00** | 75.50 | 待逐条核验 | 完成，退出码 0 |

direct 使用原无工具简洁回答提示词，只请求一次回答。auto 使用
`prompts/visual_agent_rl_system_groundingdino.txt`；与训练 step160 保存的 896 条 rollout
中的 system prompt 逐条比较，文本全部一致。tool_first 在 auto 提示词末尾追加首轮先调用
`grounding_detect` 或 `crop_zoom` 的指令，没有在代码中强制插入调用。
本次未运行旧提示词、12 回合、每回合 512 tokens 的 legacy 组。

direct 首次启动因 `EVAL_TIMEOUT_SECONDS=0` 被错误判断为立即超时而失败，修复后单独
补测，2026-09-17 20:59:33 正常结束，耗时 1465 秒。表中只使用补测结果。

结果目录（相对于仓库根目录）：

- direct：`outputs/vlmeval/tool_diagnostic/tool_diagnostic_step160_20260917_203508/direct/qwen3_base/`
- auto：`outputs/vlmeval/tool_diagnostic/tool_diagnostic_step160_20260917_172929/auto/dino_latest/`
- tool_first：`outputs/vlmeval/tool_diagnostic/tool_diagnostic_step160_20260917_172929/tool_first/dino_latest/`

tool_first 相比 auto 的三榜提升约为 0.52 / 1.12 / 0.38 个百分点（按未舍入分数计算）。
尚未核验首轮工具调用率及工具成功率，不能据此认定收益来自实际工具调用。
direct 与两组 agent 的提示词及回合预算不同；这些结果也不能与历史 KL-step130 旧口径
直接作为严格控制变量的比较。

## VL-OCR 最新十榜评测：64 卡 step80、16 卡 step110 / step150（截至 2026-10-05）

三组均已生成全部十榜评分与最终预测。以下成绩直接取评分文件，API failed 和空预测
逐条检查最终预测 XLSX；每组共 16,955 条预测。64 卡、16 卡指训练规模，评测使用
8 卡入口。这里的 OCRBench 成绩属于原 observation 协议，与下一节两卡 A/B 重跑分别记录。

64 卡权重来自
`qwen3base_multitool_vlocr_ocr_chart1600_n16_8node_20261001T072342925318_021ea85c/global_step_80`；
16 卡两份权重来自
`qwen3base_multitool_vlocr_ocr_chart_n16_2node_from_step60_20261002T170651989009_040bfc67/global_step_110`
及同一训练目录的 `global_step_150`。上述路径均以
`saves/visual_agent_zwz_rl/qwen3/` 为前缀，以 `/actor/huggingface` 为后缀。
step150 是本次查到最新的完整十榜评测，不表示当前训练最新保存步数。

### 成绩

除手写公式正确数和 FSC147 误差外，表内均为百分制。HRBench 取 `Average / all`，
MME-RealWorld 取 `Overall`，CV-Bench 取各自 `Overall`，ChartQA 取规则评分 `Overall`；
OCRBench 原始总分分别为 854、850、850（满分 1000）。

| 指标 | 样本数 | 64 卡 step80 | 16 卡 step110 | 16 卡 step150 |
|---|---:|---:|---:|---:|
| VStarBench | 191 | 87.96 | 86.39 | 87.96 |
| HRBench4K | 800 | 83.63 | 79.75 | 80.75 |
| HRBench8K | 800 | 79.13 | 78.00 | 77.00 |
| OCRBench | 1000 | 85.40 | 85.00 | 85.00 |
| MME-RealWorld-Lite | 1919 | 55.50 | 55.08 | 55.81 |
| MME-RealWorld-CN | 5917 | 68.19 | 67.21 | 68.38 |
| CV-Bench-2D | 1438 | 81.32 | 82.30 | 82.35 |
| CV-Bench-3D | 1200 | 91.08 | 90.33 | 90.42 |
| ChartQA_TEST | 2500 | 76.96 | 77.36 | 74.92 |
| OCRBench 手写公式（子项） | 100 | 45/100 | 44/100 | 43/100 |

FSC147_TEST 共 1190 题，三组都有无法解析为有效计数的输出，因此全量 `MAE`、`RMSE`
字段为空。下列误差仅针对各组有效输出，不能当作全量误差或忽略有效率直接比较。

| FSC147 指标 | 64 卡 step80 | 16 卡 step110 | 16 卡 step150 |
|---|---:|---:|---:|
| MAE_valid（越低越好） | 17.04 | 16.53 | 17.15 |
| RMSE_valid（越低越好） | 120.12 | 116.42 | 134.72 |
| 有效输出数 / 1190 | 1188 | 1189 | 1102 |
| 有效输出率 | 99.83% | 99.92% | 92.61% |
| 无效输出数 | 2 | 1 | 88 |
| exact_accuracy | 4.03% | 4.12% | 3.61% |

### 失败数与比较限制

| 残留 API failed | 64 卡 step80 | 16 卡 step110 | 16 卡 step150 |
|---|---:|---:|---:|
| HRBench4K | 10 | 0 | 0 |
| HRBench8K | 22 | 6 | 4 |
| MME-RealWorld-Lite | 7 | 5 | 6 |
| MME-RealWorld-CN | 16 | 8 | 9 |
| 其余六榜 | 0 | 0 | 0 |
| 合计 / 16955 | 55（0.32%） | 19（0.11%） | 19（0.11%） |

step110 的 OCRBench 另有 1 条空预测，不计入 API failed；另外两组没有空预测。
FSC147 的无效计数也不等于 API failed。step150 目录保留 39 个失败轨迹文件，包含调用
尝试的失败记录，不能将文件数作为最终失败样本数。

64 卡 step80 的十榜运行包含历史成功预测复用和失败补测，不是全部样本重新推理。
2026-10-04 16:26 启动的最新四榜补测（HR4K、HR8K、MME-Lite、MME-CN）退出码为 0，
但四榜分数及 55 条残留 API failed 均未改变；正常退出不表示失败样本已解决。
本节分数包含这些失败预测，不能作为无故障的严格优劣结论。

同一 16 卡训练从 step110 到 step150，VStar 上升约 1.57 个百分点，HR4K 上升 1.00，
HR8K 下降 1.00，ChartQA 下降 2.44，OCRBench 总分不变，手写公式 44→43。
FSC147 无效输出从 1 条增加到 88 条，原因尚未逐例归因。因此，现有结果不支持
“继续训练后各项能力都提升”的结论，也不能仅凭保存步数选择 step150。

### 产物位置

- [64 卡 step80 与 16 卡 step110 十榜运行](../outputs/vlmeval/multitool_vlocr_64gpu_16gpu_latest_8gpu/multitool_vlocr_64gpu_16gpu_latest_20261003T194114860354878_343/)：
  分别位于 `64gpu_step80/`、`16gpu_step110/`；`checkpoints.tsv` 记录权重，
  `status.tsv` 中两组退出码均为 0，耗时分别为 6086、20368 秒。
  其中 6086 秒包含缓存复用，不能用来估计十榜全量重跑耗时。
- [64 卡 step80 最新四榜补测](../outputs/vlmeval/multitool_vlocr_64gpu_16gpu_latest_8gpu/multitool_vlocr_64gpu_16gpu_latest_20261004T162605972167401_313/)：
  `status.tsv` 记录耗时 707 秒、退出码 0。
- [16 卡 step150 最新十榜运行](../outputs/vlmeval/multitool_vlocr_16gpu_latest_8gpu/multitool_vlocr_16gpu_step150_20261004T170810303152388_335/)：
  最终预测与评分位于 `dino_latest/VisualAgent-vllm/T20261004_G/`；
  最后一项 FSC147 评分文件修改时间为 2026-10-04 23:29（北京时间）。

### 三份十榜运行的 OCRBench 全量规则错题 LLM 复判（2026-10-05）

已针对上面三份运行各自保存的 1000 条 OCRBench 预测执行复判，覆盖全部 OCRBench
分项的规则错题，不仅是 HME。使用 DeepEyesV2 原始 `verify.md` 评测模板，不附加
system 约束；API 为 `deepseek-v4-flash`，temperature 0，最多 1024 输出 tokens，
关闭 thinking 参数。多个参考变体全部列在参考答案字段，预测内容保持原样。

| 权重 | 规则正确数 / 1000 | judge 额外接受 | 合并正确数 / 1000 | HME：规则→合并 / 100 |
|---|---:|---:|---:|---:|
| 64 卡 step80 | 854 | 61 | **915（91.5）** | 45→78 |
| 16 卡 step110 | 850 | 59 | **909（90.9）** | 44→80 |
| 16 卡 step150 | 850 | 61 | **911（91.1）** | 43→78 |

合计 446 条规则错题，其中 step110 有 1 条空预测，保持错误且不送 judge。
其余 445 条候选去重为 230 次 API 请求，全部返回有效判定，无 API 失败和未决项。
相同题目、参考答案、预测及判分协议共用一次 judge 结果；规则已判对的样本保留。
已逐条核对 3000 条合并结果与汇总一致，并确认再次运行时 pending 为 0，不重复调用 API。

分数以新增“规则 + LLM 回退”口径记录，不覆盖三份原始 `_score.json`，不改变其余九榜。
同一 16 卡训练 step110→step150 的合并总分为 909→911，HME 为 80→78；这次没有运行
额外转录约束版。这些数值与后面的两卡 A/B 基于不同预测文件，应分别引用。

可复用脚本：[rejudge_ocrbench.py](../scripts/rejudge_ocrbench.py)。在仓库根目录、已有
`qwenvl3_xmx_vLLM` Python 环境中运行下面命令即可复用本次配置；只需 CPU 和 API：

```bash
python3 scripts/rejudge_ocrbench.py \
    --manifest outputs/vlmeval/ocrbench_llm_rejudge/latest_checkpoints_20261005T123411/manifest.json \
    --output-dir outputs/vlmeval/ocrbench_llm_rejudge/latest_checkpoints_20261005T123411
```

重跑会复用已有成功判定，只请求未决候选。换一批预测时，准备新的 manifest JSON
（运行名称到预测 XLSX/JSON 路径的映射）及新的输出目录即可。

- [分项汇总 summary.json](../outputs/vlmeval/ocrbench_llm_rejudge/latest_checkpoints_20261005T123411/summary.json)
- [3000 条逐题结果 predictions.jsonl](../outputs/vlmeval/ocrbench_llm_rejudge/latest_checkpoints_20261005T123411/predictions.jsonl)
- [API 请求与原始响应 requests.jsonl](../outputs/vlmeval/ocrbench_llm_rejudge/latest_checkpoints_20261005T123411/requests.jsonl)
- [完整协议 protocol.json](../outputs/vlmeval/ocrbench_llm_rejudge/latest_checkpoints_20261005T123411/protocol.json)
- [预测来源 manifest.json](../outputs/vlmeval/ocrbench_llm_rejudge/latest_checkpoints_20261005T123411/manifest.json)

### 最终答案反斜杠后处理 + 官方 OCRBench 评分（2026-10-05）

对三份历史权重结果和两组 observation 消融的已保存最终答案进行离线后处理，
再直接调用仓库中的 `OCRBench.evaluate`。没有重新推理，也没有使用 LLM judge。
处理规则固定应用于每条预测，不参考标准答案或原始对错：仅把恰好连续两个
反斜杠且后接英文字母的情况折叠成一个，例如 `\\frac` → `\frac`。
正则为 `r'(?<!\\)\\\\(?=[A-Za-z])'`，三个及以上连续反斜杠不匹配。

| 预测来源 | 总分：原始→后处理 / 1000 | HME：原始→后处理 / 100 | 修改答案数 | 错→对 | 对→错 |
|---|---:|---:|---:|---:|---:|
| 历史 64 卡 step80 | 854→**874** | 45→65 | 24 | 20 | 0 |
| 历史 16 卡 step110 | 850→**872** | 44→66 | 28 | 22 | 0 |
| 历史 16 卡 step150 | 850→**872** | 43→65 | 26 | 22 | 0 |
| 两卡消融：原 observation，64 卡 step80 权重 | 856→**875** | 47→66 | 23 | 19 | 0 |
| 两卡消融：只还原 observation 反斜杠，同一权重 | 860→**874** | 51→65 | 17 | 14 | 0 |

五组中其他 900 题的预测文本及对错均未变化。收益来自最终答案的格式修正；
两组消融在后处理后的 HME 分数为 66 和 65，不能据此认为修改 observation
提升了公式识别能力。该处理仍可能影响合法的 LaTeX 换行后紧接字母的文本，
本批没有观察到规则正确样本变错，不代表对所有未来样本都无损。

本结果应标注为“最终答案经反斜杠后处理，使用官方 OCRBench 评分”，与原始分数、
LLM 复判分数分别记录。本次仅保存独立预测副本及评分，未将后处理默认接入
`visual_agent_api.py`。原始预测文件通过 SHA-256 核对未变，原始评分未覆盖。
已核对 5000 条逐题结果与官方总分、HME 分数一致，并检查单反斜杠、数字、
方括号、三/四个连续反斜杠和真实换行等边界情形。

- [分项汇总 summary.json](../outputs/vlmeval/ocrbench_answer_backslash_20261005/summary.json)
- [逐题原始/后处理答案及对错 paired_examples.jsonl](../outputs/vlmeval/ocrbench_answer_backslash_20261005/paired_examples.jsonl)
- [离线评分脚本 rescore.py](../outputs/vlmeval/ocrbench_answer_backslash_20261005/rescore.py)
- [固定处理协议 protocol.json](../outputs/vlmeval/ocrbench_answer_backslash_20261005/protocol.json)

## 64 卡训练 step80：OCR observation 反斜杠消融（2026-10-04）

本实验在本地两张 A800 80GB 上评测同一份 64 卡训练 step80 权重，验证 OCR observation
中的反斜杠转义是否影响公式识别。结果于 2026-10-05 记录，两组均完整评测 OCRBench
1000 题，其中手写公式 100 题（原始 index 900–999），其余分项共 900 题。

### 配置与唯一改动

- 权重（相对于仓库根目录）：
  `saves/visual_agent_zwz_rl/qwen3/qwen3base_multitool_vlocr_ocr_chart1600_n16_8node_20261001T072342925318_021ea85c/global_step_80/actor/huggingface`。
- 基线 `json_observation`：`VISUAL_AGENT_OCR_RAW_BACKSLASH=0`，保留原有 JSON 序列化。
- 实验组 `raw_ocr_backslash`：`VISUAL_AGENT_OCR_RAW_BACKSLASH=1`，仍使用 `json.dumps`，
  只在 `ocr_read` 返回的 `text` 字段中把双反斜杠 `\\` 还原为单反斜杠 `\`；
  引号转义 `\"`、换行转义 `\n`、其他字段及外层格式保持原有呈现。改动位于
  `scripts/visual_agent_inference.py`，不改变工具服务和训练侧。
- 两组均使用 PaddleOCR-VL-1.6、native tools / Hermes parser、temperature 0、
  最多 8 个 assistant 回合、每回合最多 512 tokens、上下文上限 32768。
- GPU 0 跑模型，GPU 1 跑工具；评测并发 1，模型显存比例 0.80。
  两组顺序执行，不复用历史预测，不限时（`EVAL_TIMEOUT_SECONDS=0`）。
- 本地入口：[两卡 A/B 脚本](../scripts/run_visual_agent_eval_ocr_ab_2gpu.sh)；
  汇总入口：[逐题对比脚本](../scripts/summarize_ocr_hme_ab.py)。

### 结果与逐题变化

| 指标 | 基线 | 仅还原反斜杠 | 变化 |
|---|---:|---:|---:|
| OCRBench 正确数 | 856/1000 | 860/1000 | +4 题 |
| OCRBench 归一化分数 | 85.6 | 86.0 | +0.4 分 |
| 手写公式正确数 | 47/100 | 51/100 | +4 题 |
| 其他 900 题正确数 | 809/900 | 809/900 | 0 |
| 最终 API failed | 0 | 0 | 0 |
| 失败轨迹文件数 | 0 | 0 | 0 |
| 退出码 | 0 | 0 | — |
| 单组启动至退出耗时 | 3255 秒（54 分 15 秒） | 2905 秒（48 分 25 秒） | — |

两组总耗时 6160 秒（1 小时 42 分 40 秒，包含服务启动及退出清理）。
按 OCRBench 原始 index 比较，变化均发生在手写公式题中：

- 错→对：`926`、`951`、`953`、`959`、`989`，共 5 题。
- 对→错：`980`，共 1 题。
- 其他 900 题没有正确性变化；这不等同于所有回答文本完全一致。

本次同场基线为 47/100，不能用先前讨论的 44 分或其他历史评测分数替代，计算修复收益。
实验组取得小幅净收益，但没有达到预期的 60 分以上。这只能说明当前 observation 的改动
不足以解决失分，不能推导出“最终答案的双反斜杠不是重要失分原因”。下面的进一步核验
修正了此前过早的归因。这里记录的是单次 A/B 观测，尚未通过重复运行量化波动。

### 最终答案反斜杠诊断与训练奖励核验（2026-10-05）

对 `paired_examples.jsonl` 中解码后的 `prediction` 字符串计数，区别真实的双反斜杠和
JSON 文件用于存储单反斜杠的转义形式。仅将最终答案中的双反斜杠替换成单反斜杠，再按
原 OCRBench HME 规则处理空白并匹配答案，得到以下诊断结果；不覆盖官方评分。

| HME 100 题 | 原 observation | 仅还原 observation 反斜杠 |
|---|---:|---:|
| 原始正确数 | 47 | 51 |
| 最终答案含真实双反斜杠 | 23 | 17 |
| 仅归一化最终答案即可恢复的错题 | 19 | 14 |
| 最终答案归一化后正确数（诊断值） | 66 | 65 |
| 归一化导致对→错 | 0 | 0 |

实验组可恢复的 14 题 index 为 `906, 909, 928, 933, 935, 943, 947, 957, 958, 965,
969, 976, 982, 994`。例如第 906 题，OCR 原文和实验组 observation 中的 `\frac` 都是
单反斜杠，但模型最终答案仍输出 `\\frac`；这不是查看 JSON 时的显示转义。
两组 HME 的 OCR 工具原始 `text` 均未发现真实双反斜杠。

因此，修改 observation 后，最终答案的重复反斜杠仍是可直接确认的重要失分来源：
实验组 49 道错题中有 14 道只需这一项格式归一化即可恢复。该现象支持“模型输出格式
倾向并不完全由当前 observation 决定”，但尚不能仅凭此实验确定该倾向由哪一阶段训练形成。
剩余 35 道错误不能全部归因于 OCR，仍需区分识别、表达格式和模型改写等原因。
其他 900 题正确性未变，只能说明本次样本未观察到正确性副作用，不能证明普遍无副作用。

历史产物也已复核：“44→66”来自 **16 卡 step110** 的预测；同批 **64 卡 step80**
是“45→65”。它们与本次两卡重跑的“47→66 / 51→65”趋势一致，但不能混作同一次评测。

复核时需对参考答案和预测都执行 `strip()`、真实换行替换及去空格；参考答案含末尾换行，
仅 `replace(' ', '')` 会错误地把可恢复题数算成 0。可在结果目录执行以下 CPU 命令：

```bash
python3 - <<'PY'
import json

with open('paired_examples.jsonl') as stream:
    rows = [json.loads(line) for line in stream]
hme = [row for row in rows if row['json_observation']['category'] ==
       'Handwritten Mathematical Expression Recognition']
backslash = chr(92)

def normalize(text):
    return text.strip().replace('\n', ' ').replace(' ', '')

for arm in ('json_observation', 'raw_ocr_backslash'):
    predictions = [row[arm] for row in hme]
    doubled = sum(backslash * 2 in row['prediction'] for row in predictions)
    recovered = sum(
        not row['correct'] and any(
            normalize(answer) in normalize(row['prediction'].replace(backslash * 2, backslash))
            for answer in row['answers']
        )
        for row in predictions
    )
    print(arm, 'double_backslash:', doubled, 'recoverable_errors:', recovered)
PY
```

训练侧 `reinforcement_learning/verl/tools/visual_tool.py` 仍使用 `json.dumps` 生成 OCR
observation；评测侧的开关不会改变训练侧。HME 的 `formula_match` 确实保留 LaTeX 命令
边界，单反斜杠参考答案不会与双反斜杠预测匹配。但 `compute_score` 在规则判错后仍会
进入 LLM judge，因此不能说当前完整奖励链路已严格拒绝双反斜杠。

核验新数据训练
`qwen3base_multitool_vlocr_hme_chartqa_tallyhalf_fsc3000_n16_8node_20261003T202135115738_f9c41af7`
保存的 HME100K 轨迹（按 `source_image` 的 `hme100k` 来源筛选）：

| 训练 step | HME 轨迹数 | 已记录 acc=1 | 公式规则匹配数 | 双反斜杠答案数 | 双反斜杠且规则判错但 acc=1 |
|---|---:|---:|---:|---:|---:|
| 1 | 304 | 133 | 95 | 46 | 38 |
| 10 | 256 | 99 | 57 | 109 | 26 |
| 20 | 208 | 125 | 109 | 33 | 13 |
| 30 | 416 | 287 | 270 | 26 | 0 |
| 40 | 304 | 216 | 154 | 0 | 0 |
| 42 | 416 | 240 | 203 | 0 | 0 |

这里的轨迹数包含每题 16 次 rollout，不是独立题目数；不同 step 的题目不同，不能把
正确率差异直接解释为能力趋势。step40 / step42 没有双反斜杠答案，是输出改善的迹象，
仍需固定 HME 验证集确认。现有 `validation/0.jsonl` 和 `validation/40.jsonl` 没有
HME100K 样本，不能将其中混合 OCR 验证正确率视为 HME 正确率。

后续训练修复应同时评估 OCR observation 的呈现和 HME 规则判错后的 judge 回退，
不能只修改公式匹配函数，也不应禁止 LaTeX 中合法的 `\\` 换行。这次仅诊断和记录，
未改动正在运行的训练配置或奖励代码。

逐题 index 与训练统计见
[backslash_diagnosis.json](../outputs/vlmeval/ocrbench_backslash_ab_64gpu_step80/ocrbench_backslash_ab_64gpu_step80_local2gpu_20261004T184241/backslash_diagnosis.json)。

### 对规则错题进行 LLM 复判：两版提示词（2026-10-05）

保留每组规则判对的答案，仅对基线 53 条、反斜杠实验组 49 条 HME 错题调用 API。
两组答案与全部参考变体相同的请求共用一次判定，每版去重后为 64 个候选答案。
多个参考变体作为 `Alternative 1/2/...` 列在参考答案字段；预测文本保持原样。
最终分数为“规则正确数 + LLM 从规则错题中接受的数量”。其余 900 题未做 LLM 复判。

两版使用同一个 `deepseek-v4-flash` API、temperature 0、最多 1024 输出 tokens、
关闭 thinking 参数。模板来自本地 DeepEyesV2 的
`evaluation/VLMEvalKit/vlmeval/dataset/utils/judge_prompt/verify.md`，并通过其
`judge_utils.py::build_prompt_judge` 构造用户消息：

- **DeepEyesV2 原始模板版**：不附加 system message，沿用模板原文。模板允许忽略
  客观题答案的排版、LaTeX 表达和大小写等差异。
- **转录约束版**：用户消息与原始版完全一致，另外添加 system message，允许空白、
  排版、数学定界符及命令前多余反斜杠等差异，但要求忠实转录，拒绝增删符号、改数字、
  改变量或只在数学意义上等价的改写。

DeepEyesV2 本地 ChartQA 路径确有规则判错后的 LLM 回退；其 OCRBench 类仍使用规则。
这里复用的是评测模板，API 模型使用本项目现有配置，不是复现 DeepEyesV2 README 中的
Qwen judge 部署。最终判定读取完整的 `\boxed{Yes/No}`；模板解析辅助函数对不规范结束
标签的一次误解析已依据保存的完整 boxed 结论修复。原始响应保留可查。

| HME 指标 | 原 observation | 只还原 observation 反斜杠 |
|---|---:|---:|
| 原始规则正确数 | 47/100 | 51/100 |
| DeepEyesV2 原始模板额外接受 | 30 | 28 |
| 规则 + 原始模板复判 | **77/100** | **79/100** |
| 转录约束版额外接受 | 26 | 23 |
| 规则 + 转录约束复判 | **73/100** | **74/100** |

将两组其余 900 题原有的 809 道正确答案计入，可并列更新本次 A/B 的 OCRBench 汇总：

| 评分口径 | 原 observation | 只还原 observation 反斜杠 |
|---|---:|---:|
| 标准规则分数 | 856/1000（85.6） | 860/1000（86.0） |
| 仅 HME 错题回退到 DeepEyesV2 原始提示词 judge | 886/1000（88.6） | 888/1000（88.8） |
| 仅 HME 错题回退到转录约束版 judge | 882/1000（88.2） | 883/1000（88.3） |

更新方式是在报告中新增上述两种复判口径，保留 `comparison.json` 和官方 `_score.json`
中的标准分数。`api_judge_comparison.json` 的 `ocrbench_hme_rejudged_summary` 保存合并值。
这些合计仅对应本次两卡 A/B，不代表对全部 1000 题错题进行了 LLM 复判，也不能直接替换
前述十榜运行的 64 卡 step80 分数 854 或 16 卡 step110 / step150 分数 850；那些运行
需要基于各自的最终预测另行复判。

每版最终 64 个候选答案均有有效判定，无未决项；转录约束版出现的 4 次超时已补判完成。
横向比较衡量 observation 改动，纵向比较衡量 judge 提示词差异，不能混为一组对照。
原始模板版比转录约束版多接受基线 4 题、实验组 5 题；反向分歧为 0：

- 两组均有分歧：`904, 936, 940, 952`。
- 仅反斜杠实验组有分歧：`937`。
- 差异包括将参考变量 `u` 写成 `U`、`P` 写成 `p`，以及补写乘号等；原始模板版更宽松。

前面单靠反斜杠归一化能恢复的 19 / 14 题，两版 LLM 均接受。额外恢复说明还存在
其他表达差异；这些是 judge 判定，尚未逐题看图确认为忠实识别，不能直接改写原始
OCRBench 分数。本次 observation 改动在两种复判口径下的净收益分别为 2 / 1 题。

两版完整提示词、请求、原始 API 响应和逐题判定均已保存：

- [两版对比及分歧 index](../outputs/vlmeval/ocrbench_backslash_ab_64gpu_step80/ocrbench_backslash_ab_64gpu_step80_local2gpu_20261004T184241/api_judge_comparison.json)
- [DeepEyesV2 原始模板版](../outputs/vlmeval/ocrbench_backslash_ab_64gpu_step80/ocrbench_backslash_ab_64gpu_step80_local2gpu_20261004T184241/api_judge_deepeyes_original_20261005T122128/)
- [转录约束版](../outputs/vlmeval/ocrbench_backslash_ab_64gpu_step80/ocrbench_backslash_ab_64gpu_step80_local2gpu_20261004T184241/api_judge_transcription_20261005T121007/)

### 结果文件

本次运行 ID：`ocrbench_backslash_ab_64gpu_step80_local2gpu_20261004T184241`。
以下链接相对于本文件：

- [分数与分项对比 comparison.json](../outputs/vlmeval/ocrbench_backslash_ab_64gpu_step80/ocrbench_backslash_ab_64gpu_step80_local2gpu_20261004T184241/comparison.json)
- [1000 题配对记录 paired_examples.jsonl](../outputs/vlmeval/ocrbench_backslash_ab_64gpu_step80/ocrbench_backslash_ab_64gpu_step80_local2gpu_20261004T184241/paired_examples.jsonl)
- [退出状态及耗时 status.tsv](../outputs/vlmeval/ocrbench_backslash_ab_64gpu_step80/ocrbench_backslash_ab_64gpu_step80_local2gpu_20261004T184241/status.tsv)
- [基线组产物](../outputs/vlmeval/ocrbench_backslash_ab_64gpu_step80/ocrbench_backslash_ab_64gpu_step80_local2gpu_20261004T184241/json_observation/)
- [实验组产物](../outputs/vlmeval/ocrbench_backslash_ab_64gpu_step80/ocrbench_backslash_ab_64gpu_step80_local2gpu_20261004T184241/raw_ocr_backslash/)

两组最终预测 JSON 的具体路径见 `comparison.json` 中的 `prediction_file`，
其中保留成功样本的完整文本轨迹及工具结果，消息中的图片数据以占位符代替。
后台主日志位于仓库上级目录的
`logs/visual-agent-eval/ocrbench_backslash_ab_64gpu_step80_local2gpu_20261004T184241-driver.log`。

## 当前结论

- VL-OCR 的 64 卡 step80、16 卡 step110 / step150 十榜成绩已补齐，API failed 分别为
  55 / 19 / 19；step150 相比 step110 有升有降，且 FSC147 无效输出增至 88/1190。
- 上述三份 OCRBench 的规则错题已用 DeepEyesV2 原始模板复判，合并分数分别为
  915 / 909 / 911，原始规则分数仍为 854 / 850 / 850；复判无未决项。
- 64 卡训练 step80 的 OCR 反斜杠消融已完成：OCRBench 856→860，手写公式 47→51，
  其他 900 题正确性无变化，两组 API failed 均为 0。进一步归一化最终答案可得 66 / 65；
  最终答案重复反斜杠仍是重要失分来源，单改 observation 不足以消除。训练 HME 规则
  判错后仍有 judge 回退，不能仅凭规则代码认定格式错误不会获奖。

- KL step 130 的失败样本已经补测完成，当前记录为 91.10 / 83.00 / 79.88。
- RFT-SFT 纯工具版为 85.34 / 79.13 / 75.00，1:1 Mixed 版为
  86.91 / 78.50 / 74.13；二者 HRBench 均有少量 API 失败，且与 base 的评测模式不同。
- 64 卡 Mixed-21906 + RL 的 best step40 为 89.53 / 82.88 / 78.50，final step96 为
  90.05 / 82.50 / 79.38；两组均为 `0/0/0` API 失败。16 卡 final step192 的
  VStar/HR8K 为 92.15/77.38；其 HR4K 为训练内验证 82.75，不与 64 卡独立评测
  直接比较。
- Qwen3 base 扩展榜当前为 OCRBench 83.70、MME-RealWorld-Lite 41.53、
  MME-RealWorld-CN 57.34；OCR/CN 分别残留 4/8 条 API failed，ChartQA 待补评分。
- step160 三组诊断评测已完成，分数见上表；API 失败数及工具调用遵从率尚待核验。
- 旧 D step100 的 Native 奖励链路失效，不能作为双流方案的有效 A/B 结果。
- 最新 Vision-OPD A step54 评测无效，D step54 补测后仍有 72 条 HRBench API 失败。
- 若继续验证双流方案，应先全量重跑 step130 起始权重，再补齐 A 的 step100 结果。
- 不应把包含大量 API 失败的准确率直接当作模型能力上限或下限。

## 新实验追加模板

```markdown
## 实验名称

一句话说明初始权重、训练改动和实验目的。

| 权重 | 评测时间 | VStar | HR4K | HR8K | API 失败数 |
|---|---:|---:|---:|---:|---:|
| checkpoint / step | YYYY-MM-DD | 0.00 | 0.00 | 0.00 | 0/0/0 |

一句话写结论，并说明与哪组结果使用相同评测协议。
```
