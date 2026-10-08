# v2-lite OCR 机制检查与 v3 准备（2026-10-08）

截至北京时间 10 月 8 日 10:52（太平洋夏令时间 10 月 7 日 19:52），v2-lite 已完成 step51，最近验证和保存为 step50。step50 普通验证 OCR 为 91.25%、depth 为 54.76%、宏平均为 68.73%，是当前最佳宏平均。OCR 保持较高水平、depth 回升，但机制轨迹存在明显的重调绕过，同题 GPU 评测尚未执行，不能据此认定反事实方法有效。64 卡主实验尚未启动。

## 五项任务进度

| 任务 | 状态 | 验收结果 / 待完成事项 |
|---|---|---|
| step29–39 反事实 OCR 机制检查 | CPU 分析完成 | 960 条轨迹、60 组，保存 6 个典型例子及 HTML/PDF/PNG；“上面两项检查”的原始定义仍待补充，当前口径见下文 |
| 第三台机器同题评测 | 代码与启动入口准备完成，未运行 | 缺第三台机器执行入口；请求的 step30 权重已被滚动保留策略清理，等待备份位置或改评 step40 的确认 |
| v2-lite 继续至约 step50 | 已超过目标，最新完成 step51 | step50 验证及保存完成；40、45、50 均已独立归档；没有配置 step50 自动停止，也未执行停训 |
| v3 数据和 factual-only 消融 | 候选数据已生成并逐行核验 | 约 10% 替换；全部 data_source 和 original_source 行数与 vanilla 一致；比例尚待确认 |
| 从 base 启动 64 卡主实验 | 启动脚本准备完成，未提交 | 仅在机制与同题评测结果正面后启动；目前这个条件未满足 |

## 1. CPU 机制检查

本项只读训练数据与轨迹，分析脚本及产物位于仓库外。未因机制分析修改训练代码或占用 GPU。

训练 run：`qwen3base_multitool_vlocr_reliance_v2lite_n16_2node_20261006T141420415217_5464e0b8`。
轨迹范围为 step29–39 的 `visual-agent-ocr#counterfactual`、前缀工具 `ocr_read`。
暂按两项口径检查：① 同参数错误重放与换参数绕过；② 错误照抄与最终纠错。

| 子集 | 轨迹 | 16 条一组的组数 | 答对 | 不调用新工具 | 全对组 | 全错组 | 答案混合组 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 全部 | 960 | 60 | 708（73.75%） | 612 | 38 | 15 | 7（11.67%） |
| HME | 800 | 50 | 553（69.13%） | 475 | 31 | 15 | 4（8.00%） |
| 其他 OCR | 160 | 10 | 155（96.88%） | 137 | 7 | 0 | 3（30.00%） |

关键发现：

- 完全同参数重调仅 4 条，全部正确重放原错误值，没有发现这类重放失效。
- 336 条修改参数后取得正常 OCR 响应；其中 246 条（全部轨迹的 25.63%）仅省略默认 `mode=text` 或增加全图 bbox 等等价改写，192 条最终答对，占全部 708 条正确轨迹的 27.12%。另有 6 条先全图 crop、再用新图像编号 OCR。这些是绕过候选，不能直接记为有用的区域核查。
- 442 条没有调用任何新工具而最终答对，兼容直接看图纠错，但没有同题 base 对照，不能归因于此次训练。
- 9 条最终答案只匹配错误观察、不匹配原观察，是严格词面照抄代理。它会漏掉改写、格式变化和间接受误导，不能作为完整盲从率。
- 32 条注入文本仍含规范化真值；原 factual OCR 只有 640/960 条含规范化真值。历史正常工具响应不保证正确，注入也不保证真正改变了与答案有关的信息。
- HME 中 400/800 条的扰动修改了 LaTeX 命令名，可能提供明显异常线索。HME 仅 8% 的组为答案混合组，不能据总准确率判断 GRPO 信号充足。

上述包含关系仅去空白、折叠连续反斜杠、去数学定界符并忽略大小写，不证明数学或语义等价；正确率采用轨迹原始 `acc`。960 条轨迹来自 60 个采样组，不能视作 960 个独立问题。

当前机制结论是：存在纠错行为，但其中部分伴随重调绕过；HME 的答案组内差异较少，且扰动常有格式异常线索。不能以 73.75% 的总准确率或工具调用增加，认定模型已学会选择性核查。8% 是答案混合组比例，未据此断言其余组的完整 reward 或梯度必然为零。

典型案例（行号为对应 JSONL 的 1 起始行号）：

| 行为 | 位置 | 真值 → 注入 → 最终答案 | 观察 |
|---|---|---|---|
| 不重调仍纠正 | step32:137 | `450ml` → `469mU` → `450mL` | 没有生成工具调用，acc=1 |
| 全图 crop 后绕过 | step33:788 | `x=3` → `u=0` → `x=3` | 全图 crop 后改用 image1 取得正常 OCR |
| 等价参数绕过 | step33:1008 | `x=5` → `l=3` → `x=5` | 从 `{target_image:0,mode:text}` 改成 `{target_image:0}` |
| 同参正确重放后仍答对 | step29:900 | `medio campo` → 含 `medmo catpg` 的文本 → `medio campo` | 相同 bbox 和 mode 返回同一个错误观察 |
| 照抄错误 | step37:77 | `\vert c-4\vert=0` → `I-H=O` → `I-H=O` | acc=0 |
| 重读仍失败 | step38:254 | `z=5` → `1=2` → `2=5` | 正常 OCR 本身读成 `2=5`，acc=0 |

完整图册、未简写的工具参数与输出、原图位置、可复现 CPU 脚本：

```text
/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx/analysis/v2lite_ocr_step29_39_20261008_p2pkrnvx/
    audit_cpu.py
    report.md
    summary.json
    trajectories.jsonl
    examples.json
    qualitative_cases.html
    qualitative_cases.pdf
    case_contact_sheet.png
```

图册是用于选材的定性初稿，含成功、绕过和失败案例，尚不能作为独立测试集结论。

## 2. 同题评测设计与实现

入口：`scripts/run_visual_agent_eval_ocr_reliance_modelarts.sh`。使用现有 8 卡评测环境和四个 ModelArts 软链接，单节点 7 卡模型、1 卡工具，native tools，temperature=0，max_turns=8，max_tokens=512。三个 checkpoint 各运行三种条件，共 9 轮完整 OCRBench（每轮 1,000 题，包括 100 HME），固定 seed=20261008。

| 条件 | 行为 |
|---|---|
| `clean` | 不注入错误 |
| `training` | 首次符合条件的 OCR 响应使用原训练扰动函数；训练类型仍为同字符类别内的 1–3 位替换 |
| `ocr_confusable` | 首次符合条件的 OCR 响应替换一处 `0↔O`、`1↔l` 或 `rn↔m`；不改 LaTeX 命令名 |

原训练扰动保持字母/数字类别及长度，因此上述跨类别或改变长度的替换规则不属于原训练注入规则。这指人工注入规则未见，并不意味着模型预训练或正常 OCR 输出从未出现过这类自然错误。

评测端记录第一次注入的调用；同参重调直接返回已保存的错误字符串，不再调用后端。OCR 参数比较补齐默认 `mode=text` 与全图 bbox，堵住本次发现的等价参数绕过。缓存逐题重置，重放保留当前调用 ID。真正不同的区域仍可重读；全图 crop 产生新图像编号的别名绕过尚未消除，应结合轨迹单独统计。

`scripts/summarize_ocr_reliance_eval.py` 校验每臂 1,000 个样本以及问题、类别、答案一致，分别报告正确数、错误暴露数、注入后工具调用数、词面照抄代理、重放次数和 API 失败数。

- “真实 OCR 出错”子集固定从 `base/clean` 中提取：限 Recognition 类问题，对原图完整区域成功调用 OCR、但文本未匹配真值的题。其他模型在相同题号上比较。该子集仍有 base 调用选择偏差，且采用 OCRBench 词面规则，需要人工核查；不把 VQA 文本缺少答案自动视作 OCR 错误。
- 各模型可能选择不同的工具参数，所以相同题目和 seed 不保证相同错误暴露。报告额外列出所有模型的有效参数、原文本、错误文本均一致且确实破坏答案的 Recognition 交集，不能只用全体准确率声称抗错误能力提升。
- 尚未做训练与 OCRBench 的图像级去重审计，不能将这些结果直接写成严格的未见图像泛化结论。

默认 checkpoint 是 `base vanilla53 v2_step30`。目前 base、vanilla step53、归档 v2-lite step40 均通过权重完整性预检，但 step30 目录已不存在；`best_huggingface` 曾更新到 step40，目前已对应 step50。脚本会对缺失 step30 报错，绝不把其他 step 当作 step30。step40、45、50 的独立归档均已核对索引引用的 8 个非空权重分片。

取得 step30 备份后通过 `V2_STEP30_MODEL_PATH` 指定；若确认改评 step40，则设置 `EVAL_CHECKPOINTS='base vanilla53 v2_step40'`。CPU 配置检查示例：

```bash
CHECKPOINT_EVAL_CONFIG_ONLY=1 EVAL_CHECKPOINTS='base vanilla53 v2_step40' \
    bash scripts/run_visual_agent_eval_ocr_reliance_modelarts.sh
```

step45 已完成归档，可单独 `EVAL_CHECKPOINTS=v2_step45` 运行，再将首次与追加的两个结果目录一起传给汇总脚本。step50 的归档也已完成，但现有评测入口尚未增加 `v2_step50` 标签，不能把它传成其他 step：

```bash
python3 scripts/summarize_ocr_reliance_eval.py FIRST_RESULT_DIR STEP45_RESULT_DIR --output comparison.json
```

第三台机器访问方式 / ModelArts 作业提交入口仍待提供，GPU 评测尚未执行，当前没有三臂结果。
可提交的外部入口已准备为 `/home/ma-user/work/algorithm/codebkp/run_visual_agent/eval/run_visual_agent_eval_ocr_reliance_modelarts.sh`，先建立四个共享路径软链接，再调用仓库脚本。
脚本准备与 38 项聚焦测试基于集群本地提交 `636260e`；本次发布范围为实验记录，未将本地其他待推送改动一并发布。配置检查通过不等同于真实 GPU 评测成功。

## 3. v2-lite 当前状态与保存

最新完整训练指标为 step51，单步 reward=0.608、用时约 44.7 分钟，累计训练计时约 35 小时 40 分钟。不同 step 使用不同训练题，单步 reward 变化不直接表示能力变化。以下验证结果重新从各 step 的 1,312 条 JSONL 逐来源统计，百分比按四舍五入保留两位。

| 验证来源 | 样本数 | step0 | step30 | step40 | step45 | step50 |
|---|---:|---:|---:|---:|---:|---:|
| HRBench4K | 800 | 552 / 69.00% | 606 / 75.75% | 600 / 75.00% | 599 / 74.88% | 602 / 75.25% |
| depth | 84 | 44 / 52.38% | 43 / 51.19% | 41 / 48.81% | 44 / 52.38% | 46 / 54.76% |
| TallyQA | 108 | 69 / 63.89% | 90 / 83.33% | 92 / 85.19% | 91 / 84.26% | 91 / 84.26% |
| ChartQA | 160 | 43 / 26.88% | 60 / 37.50% | 64 / 40.00% | 64 / 40.00% | 61 / 38.13% |
| OCR | 160 | 122 / 76.25% | 146 / 91.25% | 145 / 90.63% | 144 / 90.00% | 146 / 91.25% |
| 宏平均 | — | 57.68% | 67.80% | 67.92% | 68.30% | **68.73%** |

step50 是当前最佳宏平均（0.6872923280423281），比 step30 高约 0.92 个百分点，比初始模型高约 11.05 个百分点。具体判断：

- OCR 在 step30–50 的这些验证点保持 90.00%–91.25%，step50 与 step30 都为 146/160 答对，支持普通 OCR 能力保持，但没有证明对注入错误的鲁棒性提高。
- depth 从 step40 的 41/84 回升至 step50 的 46/84，比初始多答对 2 题。回落暂时缓解，但验证集只有 84 题，不能据此认定获得稳定提升或消除了退化风险。
- ChartQA 从 step45 的 64/160 降到 step50 的 61/160；宏平均上升不代表所有来源都改善。
- 当前缺同题错误注入评测和匹配训练预算的 vanilla 对照，不能把相对 base 的提升归因于反事实训练方法。

step40、45、50 已分别拷贝到该 run 的 `evaluation_snapshots/step40_huggingface/`、`step45_huggingface/`、`step50_huggingface/`。三个索引各引用 8 个非空权重分片；step50 归档于北京时间 10:52:29 完成。保存清单为各目录旁的 `step*_manifest.json`，分析目录的 `checkpoint_preservation_status.json`、`step50_preservation_status.json` 均记录归档完成。归档进程已结束，不负责停止训练。

`latest_checkpointed_iteration.txt` 指向 50，`global_step_50/data.pt` 存在，`best_checkpoint.json` 指向 step50；未进行 GPU 恢复测试。当前作业计划仍为 269 步、无时间上限，尚未配置“到 50 自动停”，也未执行停训。日志已经超过用户希望的“约 step50”并继续产生训练输出，需要通过作业控制停止；不能把完成归档当作训练已停止。

本次新增证据为上述 run 的 `validation/{40,45,50}.jsonl`、checkpoint 元数据与 node0 日志；分析目录的 `validation_through_step50.json` 保存 step0/30/40/45/50 的逐来源样本数、正确数和宏平均。机制审计仍限定 step29–39，没有将其统计冒充为 step40–50 的反事实轨迹分析。

## 4. v3 候选数据与消融

数据脚本增加 `--same-source`，按 `(data_source, original_source)` 替换，保留 vanilla 的各来源和子来源行数，包括 HME。Profile 为 `configs/tool_reliance_v3.json`：只有 `visual-agent-ocr/ocr_read` 与 `visual-agent-tallyqa/object_count` 可以生成对子。ChartQA 不生成对子，原始 ChartQA 训练行继续保留。

原始数据共 33,921 行，其中 OCR 3,258 行、TallyQA 2,418 行。17% 替换需 5,766 行，超过这两类总共 5,676 行；严格保留 HME 等子来源后，现有前缀候选的容量上界进一步降为 1,707 对（约 10.06%）。因此先准备可复核的 **10% 候选**，待比例确认后使用。

| 数据 | 普通行 | factual 行 | counterfactual 行 | 总训练行 | 验证行 |
|---|---:|---:|---:|---:|---:|
| paired | 30,529 | 1,696 | 1,696 | 33,921 | 1,312 |
| factual-only | 30,529 | 3,392 | 0 | 33,921 | 1,312 |

实际替换率为 3,392/33,921=9.9997%，约普通 90%、factual 5%、反事实 5%。1,696 对包括 HME 916、其他 OCR 386、TallyQA 394。HME 中有 1,832/1,833 行带前缀，全局 10% 不表示各来源仅替换 10%；重复问题也会改变来源内部的独立题目覆盖。

factual-only 与 paired 使用完全相同的题目、替换位置、顺序和最终 shuffle，仅将每个反事实槽换成同题 factual 前缀。因此一个对子在对照中是两份 factual，同题重复程度、工具前缀和来源比例被控制住。

| 来源 | vanilla / paired / factual-only 行数 |
|---|---:|
| ZWZ | 14,665 |
| OCR（含 HME） | 3,258 |
| DeepEyesV2 | 2,981 |
| TallyQA | 2,418 |
| ChartQA | 3,257 |
| depth | 4,342 |
| FSC147 | 3,000 |

两个最终候选目录（相对于仓库根目录）：

```text
data/vlocr_reliance_pairs_v3_strict10_20261008/
data/vlocr_reliance_factual_v3_strict10_20261008/
```

已逐行验证：所有 data_source/original_source 计数一致；两臂替换位置相同；普通行除采样替换位置外原样保留；其他来源（包括 depth 和 ChartQA）完整保留；JSONL 与 Parquet 内容一致；基础 schema 保留；验证集文件逐字节相同。

| 文件 | SHA256 |
|---|---|
| paired train.parquet | `35c348008b4546d46458f0e94cd595ba51f48a618b8d854f5a30c87b0a6afbfe` |
| factual-only train.parquet | `4cbf0e57a69034985156f821e7f164e5930401ebf5020fa4d709fa393743d002` |
| 两臂 val.parquet | `5a23cb23492426272a58e778afaf561d6c226fc342cd68575ca2e6300c7b29ee` |

分析目录中的 `v3_strict_data_validation.json` 和 `validate_v3_strict_data.py` 保存核验结果与复现脚本。早期 `*_v3_candidate10_20261008` 数据只保持大来源数量、改变了 HME 占比，已添加 `SUPERSEDED.md` 标识，不能用于主实验，未删除。

生成命令使用 RL 环境中的 Python 3.11，在仓库根目录运行；输出目录必须是新目录：

```bash
PYTHON_BIN=/home/ma-user/work/dataset/Common_wl/miniconda3/envs/visual-agent-qwen3vl-rl/bin/python3
ROLLOUT_DIR=/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx/rollouts/visual-agent-zwz-rl/qwen3base_multitool_vlocr_hme_chartqa_tallyhalf_fsc3000_n16_8node_20261003T202135115738_f9c41af7
"$PYTHON_BIN" scripts/prepare_reliance_pairs.py \
    --base-data-dir data/zwz_multitool_relation20_hme_chartqa_tallyhalf_fsc3000_20261004 \
    --rollout-dir "$ROLLOUT_DIR" --steps 1-53 --fraction 0.10 --seed 20261005 \
    --profile configs/tool_reliance_v3.json --same-source --output-dir NEW_PAIRED_DIR
```

消融使用同一命令加 `--factual-only`，并指定另一新目录。

## 5. 64 卡准备与启动条件

`scripts/run_visual_agent_multitool_vlocr_reliance_v3_8node_64gpu_modelarts.sh` 复用 vanilla 64 卡环境、软链接、工具服务和训练设置，从 Qwen3-VL-8B-Instruct base 启动新实验：8 节点、56 卡训练 + 8 卡工具，batch=336、mini_batch=112、n=16、并发=448。默认 paired；`V3_VARIANT=factual` 选择配套消融。验证与保存间隔均为 5 步，无 10 小时时限。脚本检查来源及子来源数量匹配。
外部 ModelArts 入口为 `/home/ma-user/work/algorithm/codebkp/run_visual_agent/run_visual_agent_multitool_vlocr_reliance_v3_8node_64gpu_modelarts.sh`，各节点运行同一文件。当前仅准备入口，未提交作业。

仅配置检查、不启动训练：

```bash
MULTITOOL_CONFIG_ONLY=1 bash scripts/run_visual_agent_multitool_vlocr_reliance_v3_8node_64gpu_modelarts.sh
MULTITOOL_CONFIG_ONLY=1 V3_VARIANT=factual bash scripts/run_visual_agent_multitool_vlocr_reliance_v3_8node_64gpu_modelarts.sh
```

训练来源比例相同支持与现有 vanilla 比较，但两者训练进度、采样问题和预算仍需对齐；仅与 step53 比较不能自动消除训练预算差异。

启动前应看到：堵住等价重调后，对训练类型及形近错误的同题纠错有提升，且不靠明显增加所有条件的工具调用；固定自然 OCR 错误子集有支持性结果；同时记录 depth 回落。目前只完成机制审计，GPU 评测缺失，故没有启动 64 卡主实验。

接下来优先取得第三台机器执行入口，并恢复 step30 备份或确定使用有明确 step 标签的替代 checkpoint；完成三种条件的同题评测后，再决定是否投入 64 卡。step50 普通验证向好，本身不满足“第 1、2 项结果正面”的启动条件。

验证：38 项聚焦单测通过，覆盖错误类型、重放与逐题缓存、同来源及消融匹配、评测汇总；9 臂调度用 CPU 假执行器验证；paired/factual 两套真实数据配置预检通过；Bash 语法与 `git diff --check` 通过。尚未进行真实 GPU 推理或新训练。
