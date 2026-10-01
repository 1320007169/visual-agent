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
