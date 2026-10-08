# v3 k=2 P/F/V 启动与续训记录

数据生成版本为 `49a7a1ae246212b898989c032f5b2b0773c75426`，已推送至 main。
三份数据的 manifest 已补记生成提交和代码 SHA256。版本清单见
[`configs/tool_reliance_v3_k2_data_20261008.json`](../../configs/tool_reliance_v3_k2_data_20261008.json)。
启动入口核验生成代码、训练数据及验证数据的哈希；生成脚本和注错代码必须与该版本相同。

## 数据构成

每臂 33,921 行，927 组，每组 3 行；相同题目、槽位、顺序、来源比例。
P：31,140 普通、927 factual、1,854 counterfactual；F：31,140 普通、2,781 factual；
V：对应 2,781 个槽位放无前缀原题，其余 31,140 行与 P/F 相同。

| 来源 | 组数 | 全部组占比 | original_correct=false | 原始错误占该来源 |
|---|---:|---:|---:|---:|
| HME | 305 | 32.90% | 124 | 40.66% |
| 其他 OCR | 236 | 25.46% | 25 | 10.59% |
| TallyQA | 386 | 41.64% | 169 | 43.78% |
| 合计 | 927 | 100% | 318 | 34.30% |

OCR 的 original_correct 是规范化后答案/全部别名的包含匹配，属于文本代理指标；TallyQA 是精确计数匹配。
其他 OCR 分别为 TextVQA 94 组（16 错）、DocVQA 95 组（2 错）、SROIE 23 组（1 错）、
InfographicsVQA 24 组（6 错）。TallyQA imported_genome 364 组（157 错）、amt 22 组（12 错）。
没有来源占过半。比较 P−F 时同时报告来源与 original_correct 分层，以检查原始错误样本是否稀释差异。

## 启动入口

使用 `scripts/run_visual_agent_multitool_vlocr_reliance_v3_8node_64gpu_modelarts.sh`。
文件名保留，但支持 `NNODES=2` 或 `8`；P/F 必须选择相同节点数，固定各臂的训练拓扑。
2 节点 batch/mini-batch 为 126/42；8 节点为 336/112。每节点 7 张训练卡、1 张工具卡，rollout n=16。
各臂从 Qwen3-VL 原始权重启动，每 5 step 保存与验证，无默认时间限制。

以下是每臂 2 台机器的示例。分配尚需确认，当前 Notebook 没有可用训练节点提交入口；本文记录的是已准备的配置，尚未挂起 P/F。
先在训练端更新到包含本文件和新入口的远程 main，再在每个节点使用相同环境设置运行同一入口。

```bash
# P: continuous run when the daytime allocation permits.
NNODES=2 V3_VARIANT=paired bash scripts/run_visual_agent_multitool_vlocr_reliance_v3_8node_64gpu_modelarts.sh

# F: run in a separate job and allocation during the night window.
NNODES=2 V3_VARIANT=factual bash scripts/run_visual_agent_multitool_vlocr_reliance_v3_8node_64gpu_modelarts.sh

# V: start later with the same topology and training budget.
NNODES=2 V3_VARIANT=unprefixed bash scripts/run_visual_agent_multitool_vlocr_reliance_v3_8node_64gpu_modelarts.sh
```

默认 RUN_ID 分别包含 `v3_paired_k2`、`v3_factual_k2`、`v3_unprefixed_k2`，再附带节点数和作业 token。
模型、日志、rollout 和同步目录分别隔离。每次续训分配新目录，保留源 checkpoint。
P 连续跑、F 夜间跑时，按双方都有的 step5/10/15/20 比较；不能按运行时长比较。
日夜窗口不同只改变作业停止时间，不改变训练总步数、学习率计划、batch 或数据。

## P step5 的真实保存—恢复测试（待执行）

1. 正常启动 P，不要把 TOTAL_TRAINING_STEPS 改成 5，以免改变学习率调度的总预算。
2. 等输出目录 `latest_checkpointed_iteration.txt` 写为 5，并确认 `global_step_5/data.pt` 与
   actor 的 model/optim/extra_state 分片完整：2 节点 world_size=14，8 节点 world_size=56。
3. 停止该 P 作业。记录源目录、日志、checkpoint 文件清单与哈希。
4. 使用相同节点数、相同 V3_VARIANT、相同训练设置、新作业 token，传入真实路径恢复：

```bash
NNODES=2 V3_VARIANT=paired \
RESUME_FROM_PATH=/absolute/path/to/P_run/global_step_5 \
bash scripts/run_visual_agent_multitool_vlocr_reliance_v3_8node_64gpu_modelarts.sh
```

5. 验收日志中的 `Setting global step to 5`、`Loaded model`、`Loaded optimizer`、
   `Loaded rng`、`Loaded lr_scheduler`；确认没有 dataloader 缺失警告，训练从 step6 继续，
   后续 step10 能保存。保留恢复日志与恢复前后数据游标检查结果。

当前 CPU 测试覆盖续训参数传递、分片完整性、错误臂/拓扑拒绝、源 checkpoint 保留，
不等价于上述 GPU 保存—恢复验收。实际验收尚未执行。

## 前 20 步审计

```bash
python3 scripts/audit_reliance_training.py \
    --data-dir data/vlocr_reliance_pairs_v3_fixed_20261008 \
    --rollout-dir /absolute/path/to/P_rollouts \
    --validation-dir /absolute/path/to/P_validation \
    --steps 1-20 --output p_steps1_20.json
```

F/V 替换数据及输出路径。续训后的目录可重复传入 `--rollout-dir`、`--validation-dir`。
若旧进程停止前已写出重复 step，比较时只选用最终保留的训练链；脚本遇到重复 step 会拒绝合并。
只读取已经写完的 step 文件；同组不足 16 条会报错，避免把写到一半的文件当成完成结果。

新轨迹记录 `dataset_index`（训练槽位）。旧 `source_index` 是原始题号，会重复，不能区分 k=2 变体；
新审计入口要求 dataset_index，不能直接拿旧 v2-lite 轨迹冒充新格式。

| 指标 | 报告字段与检查 | 异常处理 |
|---|---|---|
| 等价重调获得真值 | counterfactual 的 bypass_rate 应接近 0，同时看 replay_violation_rate 和 equivalent_recalls | 非零时暂停并定位轨迹；OCR 真值仍是文本匹配代理 |
| 每来源的混合 GRPO 组 | by_source_branch 的 mixed_group_fraction，并看 original_correct 分层；完整组分别计数 | 对照约 20% 的 v2-lite 参考及 HME 原约 8%，同时报告样本组数；过低先查错误质量/组内零优势，再考虑比例 |
| reward 与验证宏平均 | mean_reward、validation.macro_mean，对齐共同 step；验证宏平均按来源等权 | 明显掉点先核对配置和数据版本；P 的训练题更难，reward 不要求数值完全相等 |
| 每题调用数 | tools_per_trajectory：生成的工具调用，不含固定前缀 | P 明显高于 F 时检查是否学成额外核查，而非看图纠错 |

还会报告 tool_errors 和重放次数。没有等价重调样本时，零绕过率不构成修复覆盖充分的证据。
日间仅够一臂时优先让 P 连续训练，F 下一个夜间用完整 checkpoint 恢复；V 延后。
