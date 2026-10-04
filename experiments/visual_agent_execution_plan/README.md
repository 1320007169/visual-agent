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

## step160 工具使用诊断（2026-09-17）

同一 16 卡混合数据 RL 权重 `zwz_deepeyesv2_3k_nocount_v1_n8_2node/global_step_160`，
三组最终退出码均为 0。准确率单位为 %；尚未逐条核验 API 失败数与空预测。

| 模式 | 最大 assistant 回合 | 每回合 tokens | VStar | HR4K | HR8K | 状态 |
|---|---:|---:|---:|---:|---:|---|
| direct | 1 | 6144 | 84.82 | 79.63 | **75.88** | 超时判断修复后补测完成 |
| auto | 6 | 6144 | 86.91 | 78.88 | 75.13 | 完成 |
| tool_first | 6 | 6144 | **87.43** | **80.00** | 75.50 | 完成 |

auto 的 system prompt 与 step160 保存的全部 896 条训练 rollout 文本一致。
direct 关闭工具并使用原简洁回答提示词；tool_first 仅在 auto 提示词末尾追加首轮用工具
的指令，实际遵从率尚未统计。本次不含 legacy（旧提示词、12 回合、512 tokens）组。
不应将提示词实验的分数差直接解释为工具的因果收益。

详细协议、补测原因及结果目录见
[三项 Benchmark 评测汇总](../../docs/three_benchmark_evaluation_summary.md#16-卡混合数据-rl-step160工具使用诊断2026-09-17)。

## 多工具 RL 数据版本记录（2026-10-01）

八节点 VL-OCR 启动配置新增 OCR、Chart 原图 QA，各 1,600 条，其中各 1,440 条训练、160 条验证。
旧版数据目录为 `data/zwz_deepeyesv2_depth_tallyqa5k_multitool_20260924`，
新版为 `data/zwz_deepeyesv2_depth_tallyqa5k_ocr_chart_multitool_20261001`。

| 数据版本 | 训练集 | 验证集 | 总计 |
|---|---:|---:|---:|
| 旧版混合数据 | 30,808 | 992 | 31,800 |
| 新版追加 OCR、Chart | 33,688 | 1,312 | 35,000 |

运行名改为 `qwen3base_multitool_vlocr_ocr_chart1600_n16_8node`，追加任务已注册到数据加载器和奖励入口。
旧来源及其划分保留；OCR 使用多答案文本匹配，Chart 使用文本匹配或 5% 数值容差。
新版验证宏平均从 3 个来源增为 5 个来源，不能直接与旧版宏平均比较。
本版已有 64 卡训练日志及 step10 检查点；尚未登记新增 Benchmark 成绩。
脚本配置和实际运行进度见本文末尾的运行记录。

完整来源分布、OCR 配额、评分差异、启动配置及核验范围见
[多工具 RL 数据版本对照](../../docs/rl_multitool_training.md#ocr-and-chart-data-version-record-2026-10-01)。

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


## VL-OCR 脚本配置与运行记录（2026-10-01）

状态核对时间：北京时间 2026-10-01 12:28。以下是本批四个 ModelArts 入口的默认配置，
实际任务以启动日志中的最终参数为准；历史运行使用当时的脚本版本，不能用当前默认值反推。
目标 step 是完整数据计划的终点，不代表已经完成；10 小时退出可能提前结束本次任务。

### 脚本配置对照

| 入口脚本 | GPU 分配（训练 + 工具） | 数据 | 恢复起点 → 目标 global step | train batch / PPO mini-batch / rollout n | 保存周期 / 定时退出 |
|---|---|---|---|---|---|
| [24 卡原数据](../../scripts/run_visual_agent_multitool_vlocr_3node_24gpu_modelarts.sh) | 21 + 3 | 原版 | 60 → 244 | 126 / 42 / 16 | 20 step / 关闭 |
| [64 卡新数据](../../scripts/run_visual_agent_multitool_vlocr_8node_64gpu_modelarts.sh) | 56 + 8 | OCR、Chart 全量新版 | 从基座 step0 → 100 | 336 / 112 / 16 | 10 step / 10 小时 |
| [16 卡原数据续训](../../scripts/run_visual_agent_multitool_vlocr_2node_16gpu_modelarts.sh) | 14 + 2 | 原版 | 60 → 244 | 126 / 42 / 16 | 10 step / 10 小时 |
| [16 卡补充 OCR、Chart 续训](../../scripts/run_visual_agent_multitool_vlocr_ocr_chart_resume_2node_16gpu_modelarts.sh) | 14 + 2 | step60 后混合续训版 | 60 → 267 | 126 / 42 / 16 | 10 step / 10 小时 |

数据目录均相对仓库根目录：

- 原版：`data/zwz_deepeyesv2_depth_tallyqa5k_multitool_20260924`，训练 30,808、验证 992。
- 全量新版：`data/zwz_deepeyesv2_depth_tallyqa5k_ocr_chart_multitool_20261001`，训练 33,688、验证 1,312。
- 混合续训版：`data/vlocr_ocr_chart_continuation_step60_20261001`。文件保留已消费的前 7,560 条，恢复游标后跳过；后续实际训练 23,202 条旧数据和全部 2,880 条新增训练 QA，共 207 step。离线以 seed 20261001 混洗，启动时 `TRAIN_SHUFFLE=False` 防止再次打乱游标；新增数据并非集中放在最后。验证使用新版的 1,312 条。

四个入口均使用 Qwen3-VL-8B-Instruct 基座及原 KL reference、
`visual_tool_multitool_vlocr_config.yaml` 工具配置和
`prompts/visual_agent_rl_system_multitool_vlocr.txt` 提示词，验证周期均为 40 step。
16 卡两个分支恢复模型、Adam 状态、LR scheduler 和 global step；混合数据分支单独重建数据游标。
卡数变化会改变随机数分配和归约顺序，不保证与继续使用 24 卡逐位一致。
64 卡的 batch 更大且从基座开始，不能当作 24→16 卡续训的直接速度或效果对照。

### 实际运行记录

下表只记录已核对的日志和磁盘状态；存在启动日志不等于完成训练更新。
“最新完整 ckpt”与“最后完成 step”分别列出，恢复应使用前者。

| 运行 | 对应入口及起点 | 最后确认完成 step | 最新完整 ckpt | 状态 / 说明 |
|---|---|---:|---:|---|
| R1 | 24 卡，基座启动 | 25 | 20 | 旧 10 小时机制退出，退出码 0；21–25 未保存 |
| R2 | 24 卡，从 R1 step20 恢复 | 44 | 44 | 10 小时到期补存当前 step 后退出，退出码 0 |
| R3 | 24 卡，从 R2 step44 恢复 | 64 | 60 | 用户已取消；后续恢复源为 step60，不能从未保存的 step64 恢复 |
| R4 | 64 卡，新数据从基座启动 | 10 | 10 | 已有训练更新；56 份 model、optim、extra_state 分片及 data.pt 均存在且非空，尚无正常退出证据 |
| R5 | 16 卡原数据，从 R3 step60 恢复 | 尚未见恢复后的完成 step | 尚无新 ckpt | 已记录 21→14 ranks 加载及 rollout 日志；尚不能据此认定完成 GPU 更新验证 |
| 待运行 | 16 卡补充 OCR、Chart，从 R3 step60 恢复 | — | — | 数据、恢复游标及 CPU 验证已准备；未发现该分支训练日志 |

运行 ID（脚本会为每次任务增加后缀）：

- R1：`qwen3base_multitool_vlocr_n16_3node_20260929T194607938677_bf81dbe1`
- R2：`qwen3base_multitool_vlocr_n16_3node_resume_step20_20260930T075142966836_dcaf3c62`
- R3：`qwen3base_multitool_vlocr_n16_3node_resume_step44_20260930T190545597224_5a18e952`
- R4：`qwen3base_multitool_vlocr_ocr_chart1600_n16_8node_20260930T195425656668_e9c99c3c`
- R5：`qwen3base_multitool_vlocr_n16_2node_resume_step60_20261001T040151262391_e7978b7e`
- 混合续训版 RUN_ID 前缀：`qwen3base_multitool_vlocr_ocr_chart_n16_2node_from_step60`

核对路径：检查点为 `saves/visual_agent_zwz_rl/qwen3/<run_id>/`，
节点 0 日志为 `../logs/visual-agent-zwz-rl/<run_id>/<run_id>-node0.log`，
rollout 为 `../rollouts/visual-agent-zwz-rl/<run_id>/<step>.jsonl`。
当前 24 卡入口已改为从 R3 step60 恢复，尚未发现以该新前缀启动的运行。

R4 step10 的日志快照：`rollout/truncated_rate=0.000`、`actor/entropy=0.072`、
`rollout_consistency/turn_exact_match_rate=0.967`。数值使用日志打印精度，
单步快照不代表整体质量；token 一致率尚未达到 1，不能标注为一致性问题已解决。
旧版三来源验证宏平均与新版五来源宏平均不可直接比较，应分别记录共同来源和新增来源指标。

### 检查点保留及验证范围

新任务使用独立输出目录，读取 R3 的 step60 不会改写或清理它。
混合续训版的 `resume/global_step_60/actor` 是指向原 actor 目录的软链接，
因此原 step60 目录必须保留。每个新任务默认 `MAX_CHECKPOINTS_TO_KEEP=1`，
只轮转保留本任务最新完整训练检查点，最佳 HF 导出另行保存；这不表示保留所有中间 step。

10 小时计时从模型加载和初始验证之后开始，在 step 边界补存当前检查点再退出，
实际墙钟时间可能超过 10 小时。手动取消不保证触发补存。

已完成的 CPU 检查包括真实 FSDP 的 3→2 rank 恢复、下一次参数和 Adam 更新对拍、
同卡数恢复，以及混合数据游标、已消费前缀排除、新增 QA 全覆盖和源文件不变检查。
这不替代 Qwen3-VL 16 卡完整训练更新验证；R5 尚未记录完成 step，混合续训分支尚未上 GPU。
实现细节和可复现的数据准备命令见
[多工具 RL 技术记录](../../docs/rl_multitool_training.md#continuing-the-original-24-gpu-run-on-16-gpus)。

### 16 卡混合数据断点续训更新（2026-10-03）

混合数据任务 `qwen3base_multitool_vlocr_ocr_chart_n16_2node_from_step60_20261001T042830523880_74d4f11e`
在北京时间 2026-10-01 23:39:17 因 10 小时时限正常退出，退出码 0。
最后完成并保存 step79，14 个 rank 的 model、optim、extra_state 共 42 个分片及 `data.pt` 齐全。

同一个 `run_visual_agent_multitool_vlocr_ocr_chart_resume_2node_16gpu_modelarts.sh`
入口及外部副本默认改为从该任务的 `global_step_79` 完整恢复，
`TRAINER_STOP_AFTER_SECONDS` 从 36000 改为 0，关闭按时间自动停止。
模型、Adam、scheduler、随机状态及数据游标随 checkpoint 恢复；从 step80 继续到 step267，
剩余 188 个 batch。数据、batch126、mini-batch42、rollout16、`TRAIN_SHUFFLE=False`
和每 10 步保存的配置保持一致。新任务使用独立输出及同步目录，原 step79 必须保留。
本次只修改重新提交时使用的默认配置，尚未启动新的 16 卡任务。

### VL-OCR 最新 checkpoint 双权重评测入口（2026-10-03）

新增单节点 8 卡入口
`run_visual_agent/eval/run_visual_agent_eval_multitool_vlocr_64gpu_16gpu_latest_8gpu_modelarts.sh`，
依次评测以下两个运行的最新完整 HF checkpoint：

- 64 卡：`qwen3base_multitool_vlocr_ocr_chart1600_n16_8node_20261001T072342925318_021ea85c`。
- 16 卡：`qwen3base_multitool_vlocr_ocr_chart_n16_2node_from_step60_20261002T170651989009_040bfc67`。

添加入口时磁盘最新保存步数分别为 step70、step100；正式启动时重新读取各自的保存标记。
两份权重均为 Qwen3-VL，HF 导出各有 8 个非空分片，大小约 32.66 GiB。
与其他评测入口一致，直接读取原始 HF 模型目录；步数及模型源路径记录在 `checkpoints.tsv`，评测期间需保留选中的 checkpoint。
沿用现有 VL-OCR 单节点入口的五工具、native/hermes、8 轮、每轮 512 token、PaddleOCR-VL-1.6
和默认 10 项 benchmark，包含 OCRBench、ChartQA_TEST；ChartQA 使用规则评分。
两边结果独立保存，退出状态汇总到 `status.tsv`。当前只准备入口，未启动 GPU 评测，尚无新 benchmark 成绩。
环境与 9 月 29 日的 ModelArts 评测入口及 9 月 30 日的 VL-OCR 入口一致：
模型/VLMEval 使用 `qwenvl3_xmx_vLLM`，CUDA/GCC/G++ 使用 `spacetools-rl`；
补齐新入口的 `CUDA_LIBRARY_DIR`、`CC`、`CXX`、`CUDAHOSTCXX`，直接调用仓库入口也使用相同设置。
工具环境沿用公共 VL-OCR 脚本：depth 为 `starVLA_flash_dzw1`，count 为 `.env` 中的 `VTS_COUNT_ENV`，
PaddleOCR-VL 为 `qwen38-vllm-clean`，GroundingDINO 为 `visual-tools`，工具 CUDA 为 `cuda118`。
针对两个 checkpoint 的环境传递及评测入口执行 17 项测试，全部通过；Bash 语法检查及真实 checkpoint 配置检查通过。
配置预检、结果目录和单独指定两项 benchmark 的方式见
[脚本说明](../../scripts/README.md#vl-ocr-两个最新-checkpoint-评测)。

## 最新双权重评测与工具轨迹分析（2026-10-04）

核对时间：北京时间 2026-10-04 13:26。本节更新前文的历史快照。
最新评测目录为
`outputs/vlmeval/multitool_vlocr_64gpu_16gpu_latest_8gpu/multitool_vlocr_64gpu_16gpu_latest_20261003T194114860354878_343`。
`checkpoints.tsv` 确认评测的是 64 卡 step80 和 16 卡 step110；两行 `status.tsv`
均为退出码 0，耗时分别 6,086 秒、20,368 秒，十项 benchmark 的成绩文件均已生成。
此前“16 卡 FSC147 暂无结果、双权重尚未全部完成”的记录是早期快照，已被本节更新。

### 完整成绩及评测口径

以下是 Agent/tool-on 结果，使用五工具、native/hermes、最多 8 个 assistant 回合、
每回合 512 token。除特别注明的 FSC147 误差外，单位均为百分比。
HRBench 使用 `Average / all`；CV-Bench-2D 的 Overall 按导出 CSV 的源域聚合口径记录，
不等于把 Count 和 Relation 的逐题正确数直接合并。成绩包含残留 API 失败，未排除失败题。

| Benchmark / 指标 | 64 卡 step80 | 16 卡 step110 | 样本数 |
| --- | ---: | ---: | ---: |
| VStarBench Overall | 87.96 | 86.39 | 191 |
| HRBench4K Average / all | 83.63 | 79.75 | 800 |
| HRBench8K Average / all | 79.13 | 78.00 | 800 |
| OCRBench Final Score Norm | 85.40 | 85.00 | 1,000 |
| MME-RealWorld-Lite Overall | 55.50 | 55.08 | 1,919 |
| MME-RealWorld-CN Overall | 68.19 | 67.21 | 5,917 |
| CV-Bench-2D Overall | 81.32 | 82.30 | 1,438 |
| CV-Bench-3D Overall | 91.08 | 90.33 | 1,200 |
| ChartQA_TEST Overall | 76.96 | 77.36 | 2,500 |
| FSC147_TEST 精确正确率 | 4.03 | 4.12 | 1,190 |
| FSC147_TEST MAE_valid，越低越好 | 17.04 | 16.53 | 1,188 / 1,189 个有效答案 |
| FSC147_TEST RMSE_valid，越低越好 | 120.12 | 116.42 | 1,188 / 1,189 个有效答案 |

FSC147 分别有 2 / 1 个无效答案，API failure 均为 0；全量 MAE/RMSE 在 CSV 中为空。
上述误差只覆盖有效答案，不能标为全量 MAE/RMSE；精确正确率以全部 1,190 题为分母。
CV-Bench-2D 的 Count 分别为 562/788（71.32%）、576/788（73.10%），
Relation 分别为 615/650（94.62%）、614/650（94.46%）。

残留 API 失败从最终预测列核对，而不是用进程退出码代替：

| Benchmark | 64 卡 step80 API 失败 | 16 卡 step110 API 失败 |
| --- | ---: | ---: |
| HRBench4K | 10 | 0 |
| HRBench8K | 22 | 6 |
| MME-RealWorld-Lite | 7 | 5 |
| MME-RealWorld-CN | 16 | 8 |
| 其余六项 | 0 | 0 |

因此退出码 0 表示评测流程完成，不代表所有推理请求成功。需补测时应保持原权重与协议，
不能通过删除失败题抬高准确率。

### OCR、Chart 与 Qwen3 基座的参考比较

Qwen3-VL-8B-Instruct 直答参考来自
`outputs/vlmeval/qwen3_base_direct_other4/qwen3_base_direct_other4_20260922_195211`，
OCRBench 为 873/1,000（87.30%），ChartQA 为 81.28%。该基座不使用工具，
与本次 Agent 的提示词、工具和回合预算不同，属于历史参考，不是严格的 RL 控制变量对照。

| 项目 | Qwen3 基座直答 | 64 卡 step80 | 16 卡 step110 |
| --- | ---: | ---: | ---: |
| OCRBench 总分，/1,000 | 873 | 854 | 850 |
| Text Recognition，/300 | 276 | 283 | 281 |
| Scene Text-centric VQA，/200 | 176 | 178 | 179 |
| Doc-oriented VQA，/200 | 164 | 177 | 170 |
| Key Information Extraction，/200 | 177 | 171 | 176 |
| 手写数学表达式，/100 | 80 | 45 | 44 |
| ChartQA human，% | 70.24 | 74.56 | 74.32 |
| ChartQA augmented，% | 92.32 | 79.36 | 80.40 |
| ChartQA Overall，% | 81.28 | 76.96 | 77.36 |

OCR 的差距主要集中在手写公式，其他分项并非全面退化；Chart 的差距主要在 augmented，
human 反而高于此基座参考。ChartQA 评测使用本地标准数值 5% 容差或文本匹配，
未调用训练用 judge API；不能与训练 reward 的数值规则混用。

原始成绩、分项、预测失败数、结果路径及 SHA256 见
[最新评测审计](latest_eval_results_audit_20261004.json)。

### 模型是否按场景选择工具

新数据任务为
`qwen3base_multitool_vlocr_hme_chartqa_tallyhalf_fsc3000_n16_8node_20261003T202135115738_f9c41af7`。
本次固定分析其 step1–11 的 59,136 条训练轨迹，覆盖 3,696 道题，每题 16 次采样。
调用来自实际执行的 `rollout_trace.tool_calls`，按 `source_metadata` 识别题目、来源和标准答案。
下表比较 step1–3 与 step9–11，各覆盖 1,008 道题、16,128 条轨迹。
“调用率”指至少调用一次该工具的轨迹比例，不是该工具占全部调用的比例。

| 场景 / 工具 | step1–3 调用率 | step9–11 调用率 |
| --- | ---: | ---: |
| HME100K 手写公式 → ocr_read | 100.00% | 100.00% |
| 官方 ChartQA → ocr_read | 91.03% | 91.85% |
| 官方 ChartQA → ocr_read(mode=chart) | 72.64% | 70.79% |
| 旧 codevision Chart → ocr_read(mode=chart) | 38.41% | 42.76% |
| 原始深度题 → depth_measure | 76.85% | 84.66% |
| TallyQA → object_count | 70.07% | 39.53% |
| TallyQA → grounding_detect | 34.63% | 63.07% |
| FSC147 → object_count | 99.93% | 98.78% |
| 二维关系标签 → depth_measure | 66.08% | 44.43% |
| 前后关系标签 → depth_measure | 70.77% | 31.56% |

结论：模型有明显的场景区分，并非所有题都走同一个工具流程；但 HME/FSC 等路由
在基座的 step1 采样时已存在，不能全部归功于本次 RL。前后两段题目不同，且同题的
16 条采样并不独立；比例变化是描述性证据，不是固定验证集上的学习增益。
前后关系的深度调用下降尤其值得注意，不能据此认定路由已全面改善。

小数量 TallyQA（标准答案 1–5）调用 object_count 的比例从 70.56% 降至 37.08%，
grounding_detect 从 34.26% 升至 65.54%；FSC147 则继续主要走专用计数工具。
119 道 TallyQA 同题同时出现有/无 object_count 的采样，前者平均 acc 低 20.80 个百分点，
说明少量物体改用检测框计数有一定合理性，但这是采样路径的观察关联，不是工具开关的因果收益。

已完成评测也能看到这种区分。16 卡 step110 的完整可读轨迹中：CV2D Count 的
grounding_detect 调用率为 99.49%、object_count 为 5.46%；CV3D Depth 的
depth_measure 为 97.67%；OCRBench 的 ocr_read 为 89.40%，其中手写公式为 100%；
ChartQA 的 chart 模式为 81.44%；FSC147 的 object_count 为 72.34%。
FSC 的工具比例只覆盖 1,186 条可读轨迹，另有 4 条截断 JSON 不可恢复。
64 卡 OCRBench 全部缺少保存的工具轨迹，CV2D/CV3D 也仅有 257/227 条可读轨迹，
不能把缺失轨迹当成“没有调用工具”，也不能将这些补测子集的比例当成全量路由率。

### 主要失败原因与具体轨迹

1. **计数工具结果被直接采纳，纠错很少。** 新任务 FSC147 的 5,147 条可比较轨迹中，
   5,106 条最终答案等于首次工具 count（99.20%）；首次错误的 2,887 条中，
   2,852 条直接沿用错误 count，只有 7 条修正为标准答案。已完成 benchmark 也类似：
   64 卡为 933/975（95.69%）沿用首次 count，16 卡为 819/858（95.45%）。
   因而“会选 object_count”和“能可靠计数”是两个不同的验证目标。

2. **手写公式选对 OCR，仍在最终表达时丢分。** 16 卡 step110 的 100 道公式题全部使用
   ocr_read；首次工具文本按相同 OCRBench 规则可匹配 69 题，最终答案却只匹配 44 题。
   其中 26 题工具已匹配而最终不匹配，工具未匹配但最终修正的仅 1 题。
   例如 index906 的工具正确返回 `\frac{6.8}{x}=\frac{1.7}{4}`，模型在 `<answer>` 中
   输出 `\\frac{6.8}{x}=\\frac{1.7}{4}`，多了一层反斜杠转义。
   原始模型 response 已有双反斜杠，问题不是 Excel 显示转义。
   仅对最终文本做诊断性的双反斜杠归一化，匹配数变为 66/100，修复了 22 个原失败题。
   **正式成绩仍为 44/100**；该处理没有写入评分器，也不是通用 LaTeX 归一化方案。

3. **部分 Chart 路径仍把图表当自然场景处理。** step9–11 的旧 codevision 图表题中，
   depth_measure 调用率 29.83%，chart 模式 42.76%，工具错误轨迹率 15.06%；
   官方 ChartQA 对应为 3.67%、70.79%、2.04%。实际失败例子：step9 第14行问
   “IDA only 哪一年清洁燃料覆盖率最高”，模型调用 grounding_detect 检测折线，再用
   depth_measure 得到米制深度，回答 2002，标准答案是 2014；这条路线没有读出图表数值。
   当前新任务两类题的训练 acc 为 29.83% / 64.54%，不能把混合 Chart 来源视为同一分布。

4. **多工具衔接有坐标风险。** 在部分 crop_zoom 后的 OCR 中，模型对新生成的
   `target_image=1` 继续使用原图检测框；例如 step9 第43行先裁剪手机显示区域，再把
   原图框原样用于裁剪图上的 ocr_read。全体 Chart 训练轨迹有 593/5,792 条存在这种
   框复用迹象。这里只识别坐标衔接风险，不把每次复用都判为错误或断言它导致了失败。

5. **当前 reward 没有直接评价工具路由。** 实现为
   `max(0, 0.9 * acc + 0.1 * format - query_penalty)`，`tool_used` 只是记录指标，
   不直接增加分数；也没有“Chart 应使用 chart 模式”“二维关系不需要米制深度”或
   计数工具结果纠错的专门奖励。除定位 query 的额外惩罚外，工具路径主要通过最终答案
   间接学习，因此答对题时冗余调用也可能得到相同奖励。这能解释为什么工具调用率高，
   却仍存在路线不合适或工具输出利用不足的情况；它不是本次采样单独证明的因果结论。

详细分组、调用顺序、成功/错误状态、题目、标准答案、模型输出、文件行号和工具返回值见
[工具路由与轨迹审计](tool_routing_trajectory_audit_20261004.json)。
训练 acc 使用当前 reward 的规则和 judge 结果，不等于独立 benchmark 的评分。

下一步最有信息量的验证是：同一固定题集比较基座和新 checkpoint 的路由，
并对计数/HME 做同题工具开关或原始工具文本对照；分别核对工具读取错误、模型纠错失败、
LaTeX 输出格式、Chart 模式选择和裁剪坐标，避免只用整体工具调用率判断模型是否学会使用工具。

### 新任务评测状态

截至上述核对时间，新任务已保存 step12；当前只有 step0 的训练前验证，
宏平均 57.75%，尚无训练后 benchmark 或 step40 验证结果。
本节的 step80/step110 benchmark 成绩属于旧数据任务，不能登记为 HME/FSC 新混合数据的成绩。
新任务训练总量 33,921，TallyQA 2,418、FSC147 3,000，默认 1 epoch、100 step，
每 step 保存，时间限制关闭。前11步平均 40.40 分钟；原版前11步 43.95 分钟，
原版前80步 43.62 分钟，最近 skip83 续训 step81–82 为 54.52 分钟。
速度统计不含初始化和训练前验证，仅表示这些完成 step 的日志耗时。
