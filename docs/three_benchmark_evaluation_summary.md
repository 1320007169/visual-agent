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
| Qwen3 base（补测完成） | 2026-07-28 | **84.29** | **76.38** | **70.62** | 0/0/0 |
| Combined-8679 SFT | 2026-07-22 | 68.06 | 67.38 | 64.25 | 0/0/0 |

Qwen3 base 的 7 月 22 日旧结果有 `8/49/86` 个 API 失败，已由 7 月 28 日零失败结果
替代。Combined-8679 SFT 在三榜上均低于补测后的 base。

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

## 当前结论

- KL step 130 的失败样本已经补测完成，当前记录为 91.10 / 83.00 / 79.88。
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
