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

### 多工具、OCR/Chart 数据和计数评测核对（2026-10-04）

本节更新前文的磁盘快照；只记录本次实际核对和完成的工作。

#### 合成管线与 RL 工具

合成目录为 `../../groundingdino_offline_pipeline`（相对仓库根目录），RL 为本仓库。
合成管线原生产配置有 12 个工具，定位走 LocateAnything，使用像素坐标及
`image_id/bbox/bboxes`；当前 VL-OCR RL 使用
`grounding_detect/crop_zoom/depth_measure/object_count/ocr_read` 五工具，
使用 `target_image/bbox_2d/bboxes_2d` 和 0–1000 相对坐标。
RL 的 depth、CountGD++ 接入合成侧服务，OCR 使用 RL 的 PaddleOCR-VL 服务。
因此两侧工具库此前并不一致，不能直接互换配置或调用参数。

按“现有 RL 工具不动，只补齐其他工具”的要求，已新增
`image_resize/image_enhance/image_rotate/image_flip/image_draw/sam_segment/bbox_geometry`
七个 RL 接口。原五工具 schema 和执行逻辑保持原样，新工具沿用 RL 相对坐标协议；
新图像追加到 image list，`sam_segment` 经 `VTS_SEGMENT_ENDPOINT` 接入已有 SAM3 服务。
新增完整 12 工具配置和提示词，启用时需显式选择，现有入口仍使用各自原配置。
本次没有改动合成管线。接口、服务启动和启用方法见
[完整 RL 工具说明](../../docs/rl_full_visual_tools.md)。

#### 新下载的数据

| 数据 | 本地目录（相对 gx） | 官方 train | 官方 val | 官方 test | 内容 |
| --- | --- | ---: | ---: | ---: | --- |
| HME100K | `datasets/hme100k` | 74,502 | — | 24,607 | 手写数学公式图像及 LaTeX 转写；图像嵌在 Parquet 中 |
| ChartQA | `datasets/chartqa/ChartQA Dataset` | 28,299 QA / 18,316 图 | 1,920 QA / 1,056 图 | 2,500 QA / 1,509 图 | PNG、human/augmented 问答、表格和标注 |

HME100K 的 train 为两个各 37,251 行的 Parquet shard，下载 manifest 记录校验通过。
ChartQA train 的 human/augmented 分别为 7,398 / 20,901 QA。
此前 RL 的 OCR 来自 TextVQA、DocVQA、SROIE、InfographicsVQA，Chart 来自
CodeVision RL；下述两个旧版本尚未包含新下载的 HME100K 和官方 ChartQA train。

#### 加入 OCR 后的最新两个续训数据版本

质量清理基准为 `data/zwz_deepeyesv2_depth_tallyqa5k_ocr_chart_multitool_quality_20261003`，
从 10 月 1 日的 33,688 条训练 QA 中移除 12 个指定 UID，得到 33,676 条；验证仍为 1,312 条。

| 数据目录（相对仓库） | train | val | step80 已消费前缀 | 后续实际训练行 | 剩余更新 | 最终 step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `data/vlocr_quality_continuation_step80_20261003` | 33,676 | 1,312 | 26,880 | 6,720 | 20 | 100 |
| `data/vlocr_quality_skip_step83_continuation_step80_20261004` | 33,340 | 1,312 | 26,880 | 6,384 | 19 | 99 |

两者 batch336、`TRAIN_SHUFFLE=False`，从同一个 64 卡运行的 step80 完整 checkpoint 恢复。
最新版本精确移除旧版原计划 step83 的 336 条，即零基切片 `[27552:27888]`：
关系 187、计数 55、depth 44、DeepEyes 19、OCR 15、Chart 16。
核对 JSONL 逐行内容及 manifest 的 schedule indices，最新版本恰等于旧版去掉该切片，
前 26,880 条顺序保持一致，两份验证文件逐字节相同，各自 train/val Parquet 的 SHA256
与 manifest 一致。新 step83 对应旧 step84，新 step99 对应旧 step100。
原始 step83 rollout 在 manifest 对应路径未找到，step83 的来源依据为已有 manifest 和调度索引，
不能声称本次重新核对了原始 step83 rollout。逐行与哈希核对结果保存在
[两个续训版本审计](ocr_chart_dataset_audit_20261004.json)。

| 来源 | quality step80 续训版 | skip-step83 版 |
| --- | ---: | ---: |
| 位置关系 | 18,518 | 18,331 |
| depth | 4,386 | 4,342 |
| TallyQA 计数 | 4,892 | 4,837 |
| DeepEyesV2 | 3,000 | 2,981 |
| OCR | 1,440 | 1,425 |
| Chart | 1,440 | 1,424 |

两版验证构成为 HRBench800、depth84、计数108、OCR160、Chart160。
核对时 64 卡 VL-OCR ModelArts 入口的默认数据已为 skip-step83 版，仍选择原五工具。

#### 新实验：替换 20% 位置关系题，增加 HME100K 和 ChartQA

用户确认采用新实验数据，不作为 step80 的延续调度。
以 skip-step83 版的全部 33,340 行为基底，对 18,331 条位置关系题随机替换约 20%，
向下取偶数共 3,666 条，增加 1,833 条 HME100K 和 1,833 条官方 ChartQA train。
seed 为 20261004，混合后重新打乱整个训练集，原验证文件逐字节复制。

新数据目录：`data/zwz_multitool_relation20_hme_chartqa_20261004`。

| 来源 | 原 skip-step83 数据 | 新实验数据 |
| --- | ---: | ---: |
| 位置关系 | 18,331 | 14,665 |
| depth | 4,342 | 4,342 |
| TallyQA 计数 | 4,837 | 4,837 |
| DeepEyesV2 | 2,981 | 2,981 |
| OCR | 1,425 | 3,258 |
| Chart | 1,424 | 3,257 |
| **train 合计** | **33,340** | **33,340** |
| **val 合计** | **1,312** | **1,312** |

新增题只取官方 train，排除与原验证图像、OCRBench、ChartQA_TEST 以及官方 ChartQA
val/test 图像重合的候选，并在新增来源内部按图像去重；Chart 摘要覆盖旋转及镜像。
此检查按解码 RGB 像素进行，不覆盖缩放、裁剪等变体。已有基底训练题只移除抽中的关系题。
HME100K 答案中含字面 `<` / `>` 的候选被排除，避免与现有 answer 标签解析冲突。
HME 图像导出至新数据目录，Chart 图像引用 `gx/datasets/chartqa` 下的绝对路径，
迁移到其他环境时须保留对应共享路径或重建数据。

HME100K 使用手写公式转写问题；新增 `original_source=hme100k` 的规则匹配保留大小写和
LaTeX 命令边界，忽略 token 间空白和外围数学定界符。
当前工作区其他奖励改动包含 Chart 数值精确匹配，以及非关系任务规则失败后使用语义 judge；
HME100K 同样沿用这个 fallback，judge 提示要求转写保留文本和数学符号。
不能把整个奖励过程描述为只做 token 匹配。旧数据文件和旧恢复游标未改写。

可复现准备命令（仓库根目录）：

```bash
PYTHONDONTWRITEBYTECODE=1 ../conda_envs/visual-agent-eval/bin/python3 \
    scripts/prepare_rl_ocr_chart_replacement.py \
    --base-data-dir data/vlocr_quality_skip_step83_continuation_step80_20261004 \
    --hme-root ../datasets/hme100k \
    --chart-root ../datasets/chartqa \
    --eval-dir ../DeepEyesV2/evaluation/VLMEvalKit/evaluation/VLMEvalKit/LMUData \
    --output-dir data/zwz_multitool_relation20_hme_chartqa_20261004
```

新目录提供 `train.parquet/train.jsonl/val.parquet/val.jsonl/manifest.json`，
并单独保存移出的 `removed_relations.jsonl` 和新增的 `added_ocr_chart.jsonl`，便于追踪。
实际抽取的 ChartQA 为 human495、augmented1,338。
候选筛选记录：HME100K 有 3,600 条答案协议不兼容，候选图像重合 1 条；
ChartQA 抽样过程中排除图像重合或重复候选 62 条。这些数字不表示全部来源的重复率。
验证集按 645 张独立图像计算摘要，避免同一图像的多道题重复进行旋转、镜像计算。
生成文件和核对结果见 [新数据 manifest](../../data/zwz_multitool_relation20_hme_chartqa_20261004/manifest.json)
及 [实际数据完整性审计](../../data/zwz_multitool_relation20_hme_chartqa_20261004/validation_audit.json)。
该目录不带旧 checkpoint 的 `data.pt` 游标；新实验需从新的数据调度开始。
如初始化时复用某份权重，也不能原样恢复 step80 的完整训练状态和数据游标。
原验证集用于对照，未新增 HME100K 专用验证；公式识别能力需要另用官方 test 评测。
本次只准备数据，未启动 GPU 训练。

未来提交新实验时，可直接使用基础 VL-OCR 入口，并设置：

```bash
export MULTITOOL_DATA_DIR="$PWD/data/zwz_multitool_relation20_hme_chartqa_20261004"
export TRAIN_FILES="$MULTITOOL_DATA_DIR/train.parquet"
export VAL_FILES="$MULTITOOL_DATA_DIR/val.parquet"
export RESUME_MODE=disable
export TRAIN_SHUFFLE=True
export RUN_ID=qwen3base_multitool_vlocr_relation20_hme_chartqa_n16_8node
# Submit on all eight nodes when ready:
# bash scripts/run_visual_agent_multitool_vlocr_8node_64gpu.sh
```

这里使用基础 `.sh` 入口，不能直接替换旧 skip-step83 ModelArts 入口的 DATA_DIR，
因为后者硬编码了 step80 断点续训。上述参数默认仍为五工具；完整 12 工具需按工具说明另行显式启用。

#### 计数效果核对

最新评测目录为
`outputs/vlmeval/multitool_vlocr_64gpu_16gpu_latest_8gpu/multitool_vlocr_64gpu_16gpu_latest_20261003T194114860354878_343`。
`checkpoints.tsv` 实际选择 64 卡 step80 和 16 卡 step110。
64 卡在 `status.tsv` 有退出码 0；16 卡已有 CV-Bench 成绩，但此快照没有完成状态行，
也没有 FSC147 成绩，不能标注双权重评测全部完成。

| 模型 | CV-Bench-2D 计数（788题） | CV-Bench-2D 关系（650题） | FSC147 精确正确率 | FSC147 MAE |
| --- | ---: | ---: | ---: | ---: |
| 64 卡 step80 | 562/788 = 71.32% | 615/650 = 94.62% | 48/1,190 = 4.03% | 有效答案 MAE 17.04 |
| 16 卡 step110 | 576/788 = 73.10% | 614/650 = 94.46% | 暂无结果 | 暂无结果 |

FSC147 的 1,190 题中有效 1,188、无效 2、API failure 0，`RMSE_valid=120.12`；
CSV 的总体 MAE/RMSE 因无效答案为空，不能把 valid 指标称为全量指标。
有效答案绝对误差中位数为 3，90 分位为 25，密集场景有大误差：
marker 实际 3,704、回答 100；yellow lego stud 实际 2,563、回答 1,000。
因此计数确实较弱，尤其密集精确计数；CV 的 Count 和 Relation 难度不同，差值不是严格的任务能力归因。

进一步检查 XLSX 的工具轨迹：1,186 条可读取完整 tool_calls（其中 1 条从截断 JSON 前缀恢复），
975 条具有 `object_count` 返回值，933 条最终答案等于首次工具 count（95.69%）。
这些题首次计数均作用于原图，只有 45/975 的工具 count 精确匹配标签。
这提示计数工具误差及模型直接采纳工具结果都值得排查；它不是独立的纯工具评测，
查询词由模型选择，无法仅凭这些轨迹认定后端本身有 bug。
另有 4 条轨迹不可恢复，统计不覆盖它们。

9 月 29 日旧评测的 FSC147：64 卡 step20 MAE15.82、精确4.37%，24 卡 step40
MAE15.46、精确4.12%，两者均 1,190 有效答案。最新 step80 的精确计数没有显示明显改善；
旧评测与 VL-OCR 的工具协议及配置不同，不能直接将差异归因于 OCR/Chart 数据或 RL 更新。
本次替换没有增加计数题，不能预期它会直接解决计数弱项。
指标、样本和轨迹统计保存于 [计数审计结果](counting_eval_audit_20261004.json)。

本次执行数据替换、OCR/Chart 注册、当前奖励规则和新增工具相关测试，
47 项测试、172 个子用例全部通过；验证图像摘要改为并行去重后，另行重跑数据替换的
2 项测试、4 个子用例，全部通过。
实际数据核对通过：训练行集合精确等于基底减去 3,666 条关系题再加上 3,666 条新题，
其余题的完整行内容保留，JSONL 与 Parquet 一致，Parquet schema 相同；
新增 UID 唯一且不与基底冲突，3,666 条新增图片路径全部存在，原验证文件逐字节不变，
基底及新数据 Parquet SHA256 校验通过。
新 train SHA256：`876b616a8e7fab6b643597a25a0c389a5fae6aa9e8bb1455038b0ae91a6033d7`；
val SHA256：`5a23cb23492426272a58e778afaf561d6c226fc342cd68575ca2e6300c7b29ee`。

### 从 step80 续训的实时状态核对（2026-10-04 约 03:00，北京时间）

发现两次独立的 64 卡续训运行：

| 运行 | 数据版本 | 最后确认完成更新 | 当前可恢复的新 checkpoint | 状态 |
| --- | --- | ---: | --- | --- |
| `qwen3base_multitool_vlocr_quality_n16_8node_from_step80_20261003T112455929638_cc497bb8` | quality step80 续训版 | 82 | 未发现新完整 checkpoint | step83 阶段 NCCL collective timeout、Ray actor died，10月3日22:36:44退出，exit1 |
| `qwen3base_multitool_vlocr_quality_skip83_n16_8node_from_step80_20261003T164815747596_68f808a0` | skip-step83 版 | 81 | `global_step_81` | 仍有实时 GPU 监控；step82 rollout及奖励计算已完成，尚未见完成更新的 step82 日志 |

新运行的 `latest_checkpointed_iteration.txt=81`，step81 有 56 份 model、56 份 optim、
56 份 extra_state，均非空，`data.pt` 存在，HF 导出有 8 个分片。
该 checkpoint 在 10月4日约02:10–02:13保存，step81 日志的训练平均 reward0.695，
`rollout/truncated_rate=0.000`，单步耗时约3,261秒（54.35分钟）。
截至本次核对，八个节点的 GPU CSV 仍在持续更新，训练 GPU 有高利用率；
没有新任务正常结束或失败退出的记录。这些是当前快照，不能据此保证任务之后不会失败。

两次运行目前都只有恢复时 step80 的验证结果。
最新 skip-step83 运行的宏平均为0.7136177248677249（71.36%），best记录仍指向原step80；
旧运行对应为0.711765873015873（71.18%）。这些都是同一起点权重的重新验证结果，
不能将差异当作续训提升；尚无step81及之后的验证成绩。

最新日志出现 Ray 累计 spilled 1,049,575 MiB（约1 TiB）的记录，提示内存和I/O压力，
需要继续关注训练速度。该值是日志中的累计溢写量，不表示单时刻内存占用，也不能据此认定挂死。
主judge的训练请求返回403，备用judge成功完成评分：两个训练batch分别fallback281、278次，
`final_unresolved=0`，奖励计算分别约9.805秒、7.813秒。403本身没有导致这两批评分中断。
本次仅检查运行和保存状态，未重启、停止或更改训练任务。

#### 续训速度对照与原因核对

按原64卡运行与新skip-step83续训的同一个step81对照：

| 阶段 | 原64卡 step81（秒） | skip-step83 step81（秒） | 新减原（秒） |
| --- | ---: | ---: | ---: |
| rollout，`timing_s/gen` | 625.055 | 598.378 | -26.677 |
| 奖励计算 | 14.901 | 15.217 | +0.316 |
| old log-prob | 565.315 | 646.513 | +81.198 |
| reference log-prob | 556.821 | 586.311 | +29.490 |
| actor更新 | 1,239.452 | 1,205.672 | -33.780 |
| 检查点保存 | 本步未保存 | 127.638 | +127.638 |
| **整个step** | **3,078.600 / 51.31分钟** | **3,260.986 / 54.35分钟** | **+182.386 / +5.92%** |

新续训每步保存，原任务每10步保存；新增保存耗时解释了step81差额的约70%。
扣除本步保存，新续训耗时3,133.348秒（52.22分钟），比原step81只慢1.78%。
剩余净增加主要体现在log-prob计算，rollout与actor更新在该步反而较快；
原、新token总量分别19,889,395与19,851,804，接近一致，不能解释成新样本或输出总量大幅增加。
这里只完成了一个新续训step，尚不足以判断稳定吞吐率下降或其具体系统原因。

若与上一轮未skip83的续训比较，它的step81为56.06分钟、step82为58.83分钟，
当前skip版本的step81并没有比这次重启更慢。
若与原任务step70–79比较，扣除验证及保存的平均耗时为43.54分钟；
但原任务后来的step81–82本身已达平均52.79分钟，不能把此前所有差额都归因于恢复或skip83。

原任务也有大量Ray溢写记录，因此新任务出现约1TiB累计spill，不能单独作为
“这次恢复变慢的原因”的证据。主judge403已由备用judge完成评分，奖励阶段只多0.316秒，
也不是本步变慢的主要来源。计时原始指标及差额见
[续训计时审计](step80_resume_timing_audit_20261004.json)。

#### 与完整前80步平均耗时比较

补齐9月30日运行 `qwen3base_multitool_vlocr_ocr_chart1600_n16_8node_20260930T195425656668_e9c99c3c`
的step1–14，与10月1日从step14恢复运行的step15–80合并，80步计时全部齐全。
当前skip-step83续训已有step81、82两步完成更新，比较结果如下：

| 口径 | 原完整step1–80平均 | 当前step81–82平均 | 当前增幅 |
| --- | ---: | ---: | ---: |
| `timing_s/step`，含该步验证和保存 | 43.62分钟 | 54.52分钟 | +25.01% |
| 扣除验证和保存的训练时间 | 42.89分钟 | 52.42分钟 | +22.23% |

因此，相对完整前80步均值，当前每步多10.91分钟；扣除验证、保存仍多9.54分钟。
前20步、21–40、41–60、61–80的总耗时平均分别43.77、43.11、41.78、45.80分钟。
不能把“与原step81单步比较慢5.92%”替代为“与完整前80步比较慢5.92%”，两者基准不同。

与前80步均值相比，当前old log-prob增加171.42秒、reference log-prob增加145.28秒，
actor更新增加166.78秒，三项合计增加约8.06分钟；检查点保存平均增加111.96秒。
这说明与长期均值的差距主要体现在模型前向和参数更新阶段，不能仅用每步保存解释。
目前只有两步新续训样本，尚不足以判断稳定速度，或区分批次形状、系统资源、缓存和通信的具体影响。
本统计使用每步日志计时，不含进程启动、加载checkpoint和训练循环前的初始验证。
全部80步及新续训两步的原始计时见
[前80步平均耗时审计](step80_average_timing_audit_20261004.json)。

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

### 64 卡 step80 API 失败补测准备（2026-10-04）

ModelArts 提交入口开头已写入 `export VLOCR_RETRY_FAILED_ONLY=1`，直接提交同一个脚本即可补测。
补测固定读取原评测 `checkpoints.tsv` 中的 64 卡 step80 权重，复用成功预测，结果写入新目录。
现有 VLMEvalKit 缓存筛选逻辑核对：HRBench4K 重试10条、HRBench8K 22条、
MME-RealWorld-Lite 7条、MME-RealWorld-CN 16条，共55条。
9项补测测试、9项既有入口用例、Bash语法检查及真实 step80 八分片权重预检通过。
当前仅完成补测准备与配置预检，未启动 GPU 推理，尚无补测后成绩。
恢复双权重完整评测时，将 ModelArts 外部入口开头该变量改为 `0`。

## 16 卡反事实 RL 首轮训练进展（2026-10-06）

核对时间：北京时间 2026-10-06 15:53。以下统计固定覆盖已完成的 step1–10，
以及 step0、step5、step10 的验证；记录时 step11 正在运行，任务尚未结束。
step10 已保存，`latest_checkpointed_iteration.txt` 和 `best_checkpoint.json` 均指向 step10。
普通验证宏平均从 57.53% 提高至 63.92%，但尚不能将提升归因于反事实训练。

### 实验目的与实际配置

本轮验证远端 main 的工具观察反事实训练方法，从原始 `Qwen3-VL-8B-Instruct`
开始训练，使用新的优化器和数据游标。初始权重不是已有 16 卡 step210。
factual 分支提供历史真实工具观察，counterfactual 分支提供同题被扰动的工具观察，
模型在该前缀之后继续生成。两种分支作为独立训练行分别进行 GRPO 分组，
固定前缀不参与生成 loss；本轮没有增加成对差分奖励。

| 项目 | 本轮实际设置 |
|---|---|
| Run ID | `qwen3base_multitool_vlocr_reliance_n16_2node_20261006T005557607690_90df3b94` |
| 启动前代码版本 | `858dda1`；反事实功能来源为 main 的 `6801dfa` |
| 仓库入口 | [run_visual_agent_multitool_vlocr_reliance_2node_16gpu_modelarts.sh](../../scripts/run_visual_agent_multitool_vlocr_reliance_2node_16gpu_modelarts.sh) |
| ModelArts 提交入口 | `/home/ma-user/work/algorithm/codebkp/run_visual_agent/run_visual_agent_multitool_vlocr_reliance_2node_16gpu_modelarts.sh` |
| 资源 | 2 节点 × 8 卡；14 卡训练、2 卡工具，每节点 GPU7 为工具卡 |
| 初始模型 | `$BASE/DeepEyesV2/models/Qwen3-VL-8B-Instruct` |
| train batch / PPO mini-batch / rollout n | `126 / 42 / 16`，每 step 2,016 条轨迹 |
| val batch / 最大并发 | `126 / 112` |
| 恢复与 shuffle | `RESUME_MODE=disable`，`TRAIN_SHUFFLE=True` |
| 总训练计划 | 1 epoch，269 step；本次提交另受 10 小时时限约束 |
| 验证 / 保存周期 | 外部提交入口设置 `TEST_FREQ=5`、`SAVE_FREQ=5` |
| 退出设置 | 外部提交入口设置 `TRAINER_STOP_AFTER_SECONDS=36000`，到时完成当前 step 后保存退出 |

`BASE=/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx`。
仓库入口本身默认保存周期为 10、时间限制为 0，本次实际的 5 步保存和 10 小时限制
来自 ModelArts 外部入口，复现实验时需要保留这三个覆盖参数。
外部入口先建立与原 16 卡任务相同的四处数据/存储软链接，再访问仓库；
沿用原有 CUDA、编译器、Python 环境回退和 OpenCV overlay 配置。
实际训练 Python 为
`/home/ma-user/work/dataset/Common_wl/miniconda3/envs/visual-agent-qwen3vl-rl/bin/python3`。

### 数据总量与配比

数据目录：`data/vlocr_reliance_pairs_tallyhalf_fsc3000_20261006`。
使用 [prepare_reliance_pairs.py](../../scripts/prepare_reliance_pairs.py)，
从 `data/zwz_multitool_relation20_hme_chartqa_tallyhalf_fsc3000_20261004` 构造，
`fraction=0.2`、`seed=20261005`；替换原训练行，不扩大训练总行数。

| 划分 / 分支 | 行数 | 占训练集比例 |
|---|---:|---:|
| 普通训练行 | 27,137 | 80.00% |
| factual 训练行 | 3,392 | 10.00% |
| counterfactual 训练行 | 3,392 | 10.00% |
| 训练总计 | 33,921 | 100.00% |
| 验证集 | 1,312 | — |
| 训练 + 验证 | 35,233 | — |

3,392 对 factual/counterfactual 同题数据各占两行，因此上述行数不等于独立题目数。
替换后的训练来源为：ZWZ 14,605、depth 4,266、OCR 3,297、ChartQA 3,294、
FSC147 3,110、DeepEyesV2 2,904、TallyQA 2,445。
验证集沿用原文件，没有注入反事实观察。

前缀来源为 64 卡运行
`qwen3base_multitool_vlocr_hme_chartqa_tallyhalf_fsc3000_n16_8node_20261003T202135115738_f9c41af7`
的 step1–53 rollout。当前构造器只选择首轮第一个成功且可扰动的工具调用，
3,392 对中 `grounding_detect` 为 2,379 对、`ocr_read` 609 对、
`object_count` 403 对、`depth_measure` 1 对。
定位工具约占 70.14%，工具覆盖明显不均衡，尤其不能据此评估深度工具纠错能力。
本轮仍使用 80/10/10，60/20/20 只是候选后续对照，尚未实施。

### 训练 reward

下表 reward 为训练轨迹 `score` 的均值，与日志 `critic/rewards/mean` 对齐。
逐条核对 step1–10 共 20,160 条轨迹，本轮均满足
`score = 0.9 × acc + 0.1 × format`，`query_penalty` 均为 0，
`reward_valid` 均为 1。reward 包含格式分，不能直接当作答案准确率。

| Step | 平均 reward | 答案准确率 |
|---|---:|---:|
| 1 | 0.4240 | 37.25% |
| 2 | 0.5198 | 47.07% |
| 3 | 0.4838 | 42.76% |
| 4 | 0.5447 | 49.50% |
| 5 | 0.5183 | 46.68% |
| 6 | 0.4771 | 42.06% |
| 7 | 0.5252 | 47.37% |
| 8 | 0.5379 | 48.81% |
| 9 | 0.5237 | 47.17% |
| 10 | 0.5543 | 50.50% |

按轨迹数加权比较前后两个窗口：

| 分支 | step1–5 轨迹数 | 平均 reward | 答案准确率 | step6–10 轨迹数 | 平均 reward | 答案准确率 |
|---|---:|---:|---:|---:|---:|---:|
| 全部 | 10,080 | 0.4981 | 44.65% | 10,080 | 0.5237 | 47.18% |
| 普通 | 7,984 | 0.5032 | 45.22% | 7,968 | 0.5431 | 49.35% |
| factual | 1,168 | 0.5174 | 47.09% | 1,008 | 0.4971 | 44.25% |
| counterfactual | 928 | 0.4303 | 36.75% | 1,104 | 0.4078 | 34.24% |

整体 reward 有波动，后五步均值比前五步高 0.0255，提升主要体现在普通分支。
反事实分支的窗口均值尚未上升；两个窗口题目不同，也不是同题配对评测，
不能直接据此认定反事实能力下降或方法有效。
前十步实际包含 2,032 条反事实轨迹，来自 127 个训练行，每行采样 16 条；
这些轨迹不是 2,032 道独立问题。

### 普通验证集结果

准确率单位为 %。五个来源等权计算宏平均，非按 1,312 条样本汇总的微平均。
每次验证各来源样本数相同，step0、step5、step10 的 `reward_valid=0` 数量均为 0。

| 验证来源 | 样本数 | step0 | step5 | step10 |
|---|---:|---:|---:|---:|
| HRBench4K | 800 | 69.50 | 68.88 | 70.50 |
| depth | 84 | 52.38 | 55.95 | 55.95 |
| TallyQA | 108 | 63.89 | 64.81 | 73.15 |
| ChartQA | 160 | 25.63 | 26.88 | 30.63 |
| OCR | 160 | 76.25 | 90.63 | 89.38 |
| **宏平均** | — | **57.53** | **61.43** | **63.92** |

step10 宏平均较初始提高 6.39 个百分点，较 step5 提高 2.49 个百分点。
计数和图表继续改善，OCR 较 step5 回落 1.25 个百分点。
当前最佳指标为 `val-core/visual-agent/acc/macro_mean=0.6392010582010581`。
这些是训练内验证结果，本轮尚无新的完整三榜成绩，也没有独立反事实验证成绩。

### 运行情况、限制与后续判断

截至 step10 验证完成，主 judge 累计 7,328 次请求均失败，日志显示 HTTP 403；
备用 judge 接管 7,328 次，失败数为 0，`final_unresolved=0`。
主配置为 `deepseek-v4.1-flash`，实际外部 judge 评分来自备用 `deepseek-v4-flash`；
确定性规则可判定的题目不请求外部 judge。后续对照需要统一实际评分口径。

普通 step 日志耗时约 29–33 分钟；step5 和 step10 含验证、保存，约 65 分钟。
step10 结束时训练计时约 6 小时 15 分，10 小时时限从初始验证后计时。
按该快照预计北京时间 19:30 后完成当前 step 并保存退出，具体时间依后续耗时变化；
此处不是已完成或已正常退出的记录。

当前可以确认训练已完成十次更新并保存权重，普通验证出现提升。
要验证反事实方法的贡献，还需同初始权重、数据来源、训练预算和 judge 的普通 RL 对照，
以及同题 factual/counterfactual 的独立错误注入评测，观察错误观察下的最终答对率和纠错行为。
仅增加反事实比例或观察总 reward，不能代替上述对照。

原始证据路径（`RUN` 为上表完整 Run ID，以下文件未纳入 Git）：

- 数据构造信息：`data/vlocr_reliance_pairs_tallyhalf_fsc3000_20261006/manifest.json`。
- 训练日志：`$BASE/logs/visual-agent-zwz-rl/$RUN/$RUN-node0.log`。
- 训练轨迹：`$BASE/rollouts/visual-agent-zwz-rl/$RUN/{1..10}.jsonl`。
- 验证轨迹：`saves/visual_agent_zwz_rl/qwen3/$RUN/validation/{0,5,10}.jsonl`。
- 最近保存：`saves/visual_agent_zwz_rl/qwen3/$RUN/global_step_10/`。
- 最佳权重：`saves/visual_agent_zwz_rl/qwen3/$RUN/best_huggingface/`。
- 最佳指标：`saves/visual_agent_zwz_rl/qwen3/$RUN/best_checkpoint.json`。

## 16 卡 v2-lite 反事实 RL 训练进展（2026-10-07）

最新快照（2026-10-08 01:47）：已完成 step39，最新保存为 step35（验证 67.08%），
最佳为 step30（67.80%）。本节保留 10 月 7 日历史快照，最新结果见末尾“10 月 8 日增量更新”。

核对时间：北京时间 2026-10-07 12:37。本节固定覆盖已完成的 step1–20，
以及 step0、step5、step10、step15、step20 的验证，不代表整轮训练已经结束。
最近保存和最佳 checkpoint 均为 step20，普通验证宏平均由 57.68% 提高至 67.52%。
这说明当前训练配置能带来普通验证提升，尚不能证明提升来自反事实机制。

### 实验目的、数据与启动配置

v2-lite 检验错误工具观察前缀与现有 GRPO 能否在有组内答题差异的场景中学到纠错，
以及遇到错误观察时是否会更有针对性地核查。数据选择依据见
[v1 组构成、重调审计与 v2-lite 准备记录](../../crt16_v2_lite_audit.md)；
该文件的“尚未启动”描述属于准备阶段，本节补充实际运行结果。

本轮继续从原始 `Qwen3-VL-8B-Instruct` 开始，未继承 v1 权重或优化器。
factual 与 counterfactual 训练行各自组成 16 条采样的 GRPO 组；固定工具前缀不参与生成 loss，
本轮没有增加成对差分奖励或强制核查分支。

| 项目 | 本轮设置 |
|---|---|
| Run ID | `qwen3base_multitool_vlocr_reliance_v2lite_n16_2node_20261006T141420415217_5464e0b8` |
| 启动时间 | 北京时间 2026-10-06 22:14 左右 |
| 仓库入口 | [run_visual_agent_multitool_vlocr_reliance_v2_lite_2node_16gpu_modelarts.sh](../../scripts/run_visual_agent_multitool_vlocr_reliance_v2_lite_2node_16gpu_modelarts.sh) |
| ModelArts 提交入口 | `/home/ma-user/work/algorithm/codebkp/run_visual_agent/run_visual_agent_multitool_vlocr_reliance_v2_lite_2node_16gpu_modelarts.sh` |
| 资源 | 2 节点 × 8 卡；14 卡训练、2 卡工具，每节点 GPU7 为工具卡 |
| 初始模型 | `$BASE/DeepEyesV2/models/Qwen3-VL-8B-Instruct` |
| train batch / PPO mini-batch / rollout n | `126 / 42 / 16`，每 step 2,016 条轨迹 |
| val batch / 最大并发 | `126 / 112` |
| 实际恢复配置 | `trainer.resume_mode=disable`，本次为首次训练 |
| shuffle / 总计划 | `TRAIN_SHUFFLE=True`；1 epoch，269 step |
| 验证 / 保存周期 | `TEST_FREQ=5`、`SAVE_FREQ=5` |
| 时间限制 | `TRAINER_STOP_AFTER_SECONDS=0`，已取消原 10 小时限制 |

`BASE=/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx`。
外部入口沿用原 16 卡任务的四处软链接、CUDA/toolchain 和 Python 环境初始化。
脚本已支持显式设置 `RESUME_FROM_PATH`，恢复同一 v2-lite 数据与训练拓扑下的完整
`global_step_N`，包括优化器和数据游标；本次从原始模型开始，尚未实测中断后的 GPU 恢复。

数据目录为 `data/vlocr_reliance_pairs_v2_lite_20261006`，使用
[严格来源与工具 profile](../../configs/tool_reliance_v2_lite.json)，
`fraction=0.17`、`seed=20261005`。前缀仍来自原 64 卡运行的 step1–53。

| 划分 / 分支 | 行数 | 占训练集比例 |
|---|---:|---:|
| 普通训练行 | 28,155 | 约 83% |
| factual 训练行 | 2,883 | 约 8.5% |
| counterfactual 训练行 | 2,883 | 约 8.5% |
| 训练总计 | 33,921 | 100% |
| 验证集 | 1,312 | — |
| 训练 + 验证 | 35,233 | — |

2,883 对中，ChartQA/ocr_read 为 1,367 对，OCR/ocr_read 为 1,250 对，
TallyQA/object_count 为 266 对。严格筛选后当前算法与种子最多可提供 2,936 对，
不足原 20% 替换所需的 3,392 对，因此本轮使用已确认的 17% 替换比例。
普通行仍覆盖原有七类来源，验证文件与基础数据保持一致，未注入反事实观察。

### 训练 reward 与普通验证

逐条核对 step1–20 共 40,320 条轨迹，均满足
`score = 0.9 × acc + 0.1 × format`，`reward_valid` 均为 1。
reward 包含格式分，不能直接当作答案准确率；下表按每五步的全部轨迹汇总。

| 训练窗口 | 轨迹数 | 平均 reward | 答案准确率 |
|---|---:|---:|---:|
| step1–5 | 10,080 | 0.5029 | 45.31% |
| step6–10 | 10,080 | 0.5467 | 49.79% |
| step11–15 | 10,080 | 0.5795 | 53.35% |
| step16–20 | 10,080 | 0.6150 | 57.26% |

窗口均值上升，但单步仍有波动；step18、step19、step20 的日志 reward 分别为
0.513、0.662、0.669。各窗口题目不同，不能把窗口差值当成同题能力提升。

验证准确率单位为 %。每次验证共 1,312 条，均无无效 reward；
宏平均为下列五个来源等权平均，不是全部样本的微平均。

| 验证来源 | 样本数 | step0 | step5 | step10 | step15 | step20 |
|---|---:|---:|---:|---:|---:|---:|
| HRBench4K | 800 | 69.00 | 70.88 | 71.13 | 73.38 | 74.25 |
| depth | 84 | 52.38 | 55.95 | 54.76 | 58.33 | 55.95 |
| TallyQA | 108 | 63.89 | 74.07 | 77.78 | 81.48 | 82.41 |
| ChartQA | 160 | 26.88 | 29.38 | 33.75 | 33.75 | 35.00 |
| OCR | 160 | 76.25 | 88.75 | 90.00 | 89.38 | 90.00 |
| **宏平均** | — | **57.68** | **63.81** | **65.48** | **67.26** | **67.52** |

step20 较本轮初始提高 9.84 个百分点，较 step15 提高 0.26 个百分点，
其中 depth 较 step15 回落。最佳指标为
`val-core/visual-agent/acc/macro_mean=0.6752195767195767`。
本轮初始成绩是 57.68%，不能误用 v1 的 57.53% 作为基线。

### 反事实分支表现与组内学习信号

以下固定汇总 step1–20。直接作答指首个生成轮次没有工具调用，与
`analyze_tool_reliance.py` 的 `first_action_effect.answer.share` 口径一致，
不计固定前缀中的工具调用。每组都已核验为 16 条轨迹；全对、全错和混合按答案 acc 分类。

| 来源 / 前缀工具 | 分支 | 轨迹数 | 答案准确率 | 直接作答率 | 组数 | 全对 | 全错 | 混合 | reward 有差异 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ChartQA / ocr_read | factual | 1,536 | 50.00% | 83.98% | 96 | 40 | 38 | 18 | 28 |
| ChartQA / ocr_read | counterfactual | 1,712 | 55.55% | 90.19% | 107 | 43 | 33 | 31 | 35 |
| OCR / ocr_read | factual | 1,296 | 68.75% | 97.92% | 81 | 49 | 20 | 12 | 13 |
| OCR / ocr_read | counterfactual | 1,440 | 54.10% | 95.83% | 90 | 38 | 32 | 20 | 22 |
| TallyQA / object_count | factual | 336 | 79.46% | 100.00% | 21 | 16 | 4 | 1 | 1 |
| TallyQA / object_count | counterfactual | 272 | 52.57% | 78.31% | 17 | 6 | 5 | 6 | 6 |

反事实组的答案混合比例分别为 ChartQA 28.97%、OCR 22.22%、TallyQA 35.29%，
说明存在组内答题奖励差异，并非全部整组同分。单纯格式分差异也会产生 reward 差异，
不能把最后一列全部解释为纠错学习信号。

以首轮调用工具作为核查的代理指标，ChartQA 的 factual/counterfactual 为
16.02%/9.81%，OCR 为 2.08%/4.17%，TallyQA 为 0%/21.69%。
不同来源未表现出一致的选择性核查，TallyQA 反事实仅有 17 个训练组，样本尤其有限。
这些分支来自不同训练题和不同更新阶段，不是同题、同 checkpoint 的配对对照。
直接作答也可能是直接看图纠正错误观察，不能仅据此认定盲从或方法失败。

### 运行健康、保存与后续验证

截至 step20，训练加五次验证共记录 46,880 条评分样本。
主 judge 的 14,909 次外部请求均失败；备用 judge 接管全部 14,909 次，
备用失败数为 0，`final_unresolved=0`。主配置为 `deepseek-v4.1-flash`，
实际外部 judge 评分由备用 `deepseek-v4-flash` 完成；后续对照需统一实际 judge。

step1–20 共记录 83,840 次工具调用，其中 1,120 次为 error（1.34%），
包含 OCR 空响应与工具参数校验等错误；2 条训练轨迹被截断。
日志中未发现 CUDA OOM、RayTaskError 或任务退出错误。
最近普通 step 约 29–35 分钟，step20 含验证和保存约 70 分钟；
step20 结束时训练计时约 13 小时 25 分，总计划尚余 249 步。

已核验 `global_step_20/data.pt` 非空，actor 下 14 个训练 rank 的模型、优化器和
extra_state 共 42 个 `.pt` 分片均非空，Hugging Face 索引引用的 8 个权重分片均存在且非空。
`latest_checkpointed_iteration.txt` 与 `best_checkpoint.json` 均指向 step20。
文件完整性核验不等同于已完成实际恢复训练测试。

当前结论是训练 reward 和普通验证改善，反事实组存在部分学习信号，
但尚未证明模型学会了识别错误工具观察或选择性核查。
下一步应使用 step0 与 step20 在同一批独立题目上做 factual/counterfactual 配对评测，
统计错误观察下答对率、错误内容照抄率、纠错率及两分支核查率差；
若要归因到反事实方法，还需同初始模型、来源、预算与 judge 的普通 RL 对照。
目前没有独立反事实评测或同预算普通 RL 对照结果。

原始证据路径（`RUN` 为本节完整 Run ID，以下文件未纳入 Git）：

- 数据构造信息：`data/vlocr_reliance_pairs_v2_lite_20261006/manifest.json`。
- 训练日志：`$BASE/logs/visual-agent-zwz-rl/$RUN/$RUN-node0.log`。
- 训练轨迹：`$BASE/rollouts/visual-agent-zwz-rl/$RUN/{1..20}.jsonl`。
- 验证轨迹：`saves/visual_agent_zwz_rl/qwen3/$RUN/validation/{0,5,10,15,20}.jsonl`。
- 最近保存：`saves/visual_agent_zwz_rl/qwen3/$RUN/global_step_20/`。
- 最佳权重：`saves/visual_agent_zwz_rl/qwen3/$RUN/best_huggingface/`。
- 最佳指标：`saves/visual_agent_zwz_rl/qwen3/$RUN/best_checkpoint.json`。

### 17:53 增量更新（截至 step28）

北京时间 2026-10-07 17:53，以上同一运行已确认完成 step28。
step29 的 rollout 已落盘，但快照时尚无该步更新完成的日志，因此本次统计只纳入 step1–28。
最新验证为 step25，最新保存和最佳 checkpoint 均为 step25；尚无 step30 验证结果。

**普通验证。** step25 宏平均为 67.65%，较 step20 提高 0.13 个百分点，
较本轮 step0 提高 9.97 个百分点。准确率单位为 %，宏平均仍为五个来源等权平均。

| 验证来源 | 样本数 | step20 | step25 |
|---|---:|---:|---:|
| HRBench4K | 800 | 74.25 | 74.50 |
| depth | 84 | 55.95 | 54.76 |
| TallyQA | 108 | 82.41 | 81.48 |
| ChartQA | 160 | 35.00 | 36.88 |
| OCR | 160 | 90.00 | 90.63 |
| **宏平均** | — | **67.52** | **67.65** |

step25 的 1,312 条验证样本均无无效 reward。HRBench4K、ChartQA 和 OCR 上升，
depth 与 TallyQA 回落，整体增益已比前期小；当前不能据此认定收敛或退化。
最佳指标为 `val-core/visual-agent/acc/macro_mean=0.6764867724867725`。

**训练 reward。** 本次新增 step21–28 共 16,128 条轨迹，累计 56,448 条。
累计轨迹的 `reward_valid` 均为 1，均满足 `score = 0.9 × acc + 0.1 × format`。

| 训练窗口 | 轨迹数 | 平均 reward | 答案准确率 |
|---|---:|---:|---:|
| step16–20 | 10,080 | 0.6150 | 57.26% |
| step21–25 | 10,080 | 0.6172 | 57.53% |
| step26–28 | 6,048 | 0.5955 | 55.11% |

step21–25 与前一窗口接近，最近三步有所回落；step28 的日志 reward 为 0.578。
最后一个窗口只有三步，且各窗口题目不同，不能解释为同题准确率下降。

**反事实分支与组构成。** 下表重新汇总 step1–28，每组均为 16 条轨迹，
沿用上文的 acc 分类与首轮直接作答口径。表中轨迹数不等于独立问题数。

| 来源 / 前缀工具 | 分支 | 轨迹数 | 答案准确率 | 直接作答率 | 组数 | 全对 | 全错 | 混合 | reward 有差异 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ChartQA / ocr_read | factual | 2,160 | 56.48% | 84.54% | 135 | 62 | 44 | 29 | 40 |
| ChartQA / ocr_read | counterfactual | 2,480 | 54.64% | 83.31% | 155 | 64 | 50 | 41 | 50 |
| OCR / ocr_read | factual | 1,856 | 71.93% | 98.44% | 116 | 75 | 27 | 14 | 16 |
| OCR / ocr_read | counterfactual | 2,064 | 56.98% | 95.30% | 129 | 60 | 41 | 28 | 31 |
| TallyQA / object_count | factual | 512 | 83.40% | 96.88% | 32 | 26 | 5 | 1 | 1 |
| TallyQA / object_count | counterfactual | 336 | 61.61% | 82.44% | 21 | 10 | 5 | 6 | 6 |

ChartQA、OCR、TallyQA 的反事实答案混合组比例分别为 26.45%、21.71%、28.57%。
三类仍有组内答题奖励差异；reward 有差异组中的格式分差异不能单独视为纠错信号。
以首轮工具调用为核查代理，factual/counterfactual 分别为 ChartQA 15.46%/16.69%、
OCR 1.56%/4.70%、TallyQA 3.13%/17.56%。累计统计中反事实分支均更常首轮调用工具，
但 ChartQA 差距很小，TallyQA 反事实仅 21 组，且两分支不是同题、同 checkpoint 对照；
这些差异只能作为后续验证线索，尚不足以证明已经学会选择性核查。

**评分、工具与权重。** 截至 step28，训练加六次验证共 64,320 条评分样本，
主 judge 累计 20,003 次外部请求均失败，备用 judge 完成全部 20,003 次，
备用失败数和 `final_unresolved` 均为 0，实际外部评分仍使用 `deepseek-v4-flash`。
累计 118,890 次工具调用中有 1,650 次 error（1.39%），其中 OCR 调用错误 1,334 次；
累计截断轨迹仍为 2 条。已完成步骤的日志未发现 CUDA OOM、RayTaskError 或任务退出错误。

step28 结束时训练计时约 18 小时 26 分，总计划尚余 241 步。
step25 含验证和保存约 71 分钟；step26–28 分别约 28、33、38 分钟。
`global_step_25/data.pt`、actor 下 42 个 `.pt` 分片和索引引用的 8 个 Hugging Face
权重分片均已核验存在且非空。`latest_checkpointed_iteration.txt` 与
`best_checkpoint.json` 均指向 step25，实际 GPU 恢复仍未测试。

本次新增证据为 `$BASE/rollouts/visual-agent-zwz-rl/$RUN/{21..28}.jsonl`、
`saves/visual_agent_zwz_rl/qwen3/$RUN/validation/25.jsonl` 和
`saves/visual_agent_zwz_rl/qwen3/$RUN/global_step_25/`，运行日志路径沿用上文。

后续应优先用原始模型与当前最佳 step25 做固定同题的独立错误注入评测，
核对错误照抄、纠错和两分支核查率差；普通验证的小幅提升与当前累计训练统计，
仍不能替代反事实专项评测和同预算普通 RL 对照。

### 10 月 8 日增量更新（截至 step39）

北京时间 2026-10-08 01:47，同一 v2-lite 运行已完成 step39，正在执行 step40。
step40 的 rollout 已落盘，但快照时尚无该步完成日志和验证结果，本次统计固定截至 step39。
最新完整保存为 step35，最佳 checkpoint 为 step30。训练仍在运行，时间限制仍为 0。

**普通验证。** 新增 step30、step35 验证，各 1,312 条，均无无效 reward。
准确率单位为 %，宏平均沿用五来源等权口径。

| 验证来源 | 样本数 | step25 | step30 | step35 |
|---|---:|---:|---:|---:|
| HRBench4K | 800 | 74.50 | 75.75 | 74.75 |
| depth | 84 | 54.76 | 51.19 | 47.62 |
| TallyQA | 108 | 81.48 | 83.33 | 82.41 |
| ChartQA | 160 | 36.88 | 37.50 | 40.63 |
| OCR | 160 | 90.63 | 91.25 | 90.00 |
| **宏平均** | — | **67.65** | **67.80** | **67.08** |

step30 最佳宏平均为 `0.678047619047619`，较原始模型的 step0 提高 10.13 个百分点。
step35 宏平均为 `0.67080291005291`，较最佳回落 0.72 个百分点，仍比 step0 高 9.40 个百分点。
ChartQA 继续上升，其他四项较 step30 回落；depth 连续从 step25 的 54.76% 降至 47.62%，
已低于本轮初始 52.38%，后续需要关注。不能将“最新权重”和“最佳权重”混为一谈。

**训练 reward。** step1–39 共 78,624 条轨迹，比上次 step28 快照新增 22,176 条。
所有轨迹的 `reward_valid=1`，均满足 `score = 0.9 × acc + 0.1 × format`。

| 训练窗口 | 轨迹数 | 平均 reward | 答案准确率 |
|---|---:|---:|---:|
| step21–25 | 10,080 | 0.6172 | 57.53% |
| step26–30 | 10,080 | 0.6052 | 56.20% |
| step31–35 | 10,080 | 0.6111 | 56.84% |
| step36–39 | 8,064 | 0.6573 | 61.94% |

step39 单步日志 reward 为 0.757，最近四步窗口均值上升；各窗口题目不同，最后一个窗口仅四步，
不能据此推断同题能力提升，也不能用训练 reward 的上升掩盖 step35 普通验证的回落。

**反事实分支与组构成。** 下表累计覆盖 step1–39，每组均为 16 条采样，
直接作答与全对/全错/混合的定义沿用前文。

| 来源 / 前缀工具 | 分支 | 轨迹数 | 答案准确率 | 直接作答率 | 组数 | 全对 | 全错 | 混合 | reward 有差异 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ChartQA / ocr_read | factual | 2,896 | 58.63% | 83.25% | 181 | 89 | 59 | 33 | 46 |
| ChartQA / ocr_read | counterfactual | 3,440 | 58.69% | 80.90% | 215 | 98 | 62 | 55 | 66 |
| OCR / ocr_read | factual | 2,592 | 72.99% | 98.88% | 162 | 106 | 37 | 19 | 21 |
| OCR / ocr_read | counterfactual | 3,024 | 62.30% | 85.28% | 189 | 98 | 56 | 35 | 38 |
| TallyQA / object_count | factual | 704 | 87.93% | 93.18% | 44 | 38 | 5 | 1 | 1 |
| TallyQA / object_count | counterfactual | 576 | 64.58% | 76.74% | 36 | 20 | 9 | 7 | 7 |

反事实答案混合组比例为 ChartQA 25.58%、OCR 18.52%、TallyQA 19.44%，仍存在部分组内答题信号。
首轮工具调用率 factual/counterfactual 分别为 ChartQA 16.75%/19.10%、
OCR 1.12%/14.72%、TallyQA 6.82%/23.26%。OCR 与 TallyQA 出现较大的描述性差距，
但这是不同训练题和不同模型阶段的累计统计，不能直接认定已学会选择性核查。
直接看图纠正也可能表现为直接作答；后续仍需同题、同 checkpoint 的错误注入评测，
结合错误照抄率和最终答对率判断方法效果。

**运行与保存。** 截至 step39，训练加八次验证共 89,120 条评分样本。
主 judge 累计 27,381 次外部请求均失败，由备用 judge 完成全部请求；
备用失败数和 `final_unresolved` 均为 0，实际外部评分仍来自 `deepseek-v4-flash`。
累计 166,304 次工具调用中有 2,290 次 error（1.38%），其中 OCR 调用错误 1,934 次；
累计截断轨迹 18 条。已完成步骤的日志未发现 CUDA OOM、RayTaskError 或任务退出错误。

step39 结束时训练计时约 26 小时 34 分，总计划尚余 230 步。
最近保存的 `global_step_35/data.pt`、actor 下 42 个 `.pt` 分片和索引引用的
8 个 Hugging Face 权重分片均存在且非空；`latest_checkpointed_iteration.txt` 指向 35，
`best_checkpoint.json` 指向 30。实际 GPU 恢复仍未测试。

新增证据为 `$BASE/rollouts/visual-agent-zwz-rl/$RUN/{29..39}.jsonl`、
`saves/visual_agent_zwz_rl/qwen3/$RUN/validation/{30,35}.jsonl` 和
`saves/visual_agent_zwz_rl/qwen3/$RUN/global_step_35/`。最佳权重目录仍为
`saves/visual_agent_zwz_rl/qwen3/$RUN/best_huggingface/`，对应 step30。

后续专项评测优先比较原始模型与最佳 step30；若比较最新 step35，应单独标注其身份，
同时核查 depth 的回落。当前仍没有独立反事实评测或同预算普通 RL 对照结果。

### 10 月 8 日 OCR 机制审计与 v3 准备（截至 step41）

最新记录见 [OCR 机制检查与 v3 准备](ocr_mechanism_v3_20261008.md)。step40 宏平均更新为
67.92%、OCR 90.63%、depth 48.81%；`best_huggingface` 已更新为 step40，不能再把它当作 step30。
step29–39 的 960 条反事实 OCR 轨迹中，246 条存在等价参数重调绕过候选，HME 答案混合组仅 8%。
同题评测代码及严格保持来源/子来源比例的 v3 候选数据已准备，GPU 评测与 64 卡主实验尚未启动。
