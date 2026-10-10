# Online CRT 实验执行记录：64 卡转 16 卡

核对时间：2026-10-11 00:35（北京时间），结果范围为 16 卡续训已完成的 step5–9。后续进度以原始日志和 checkpoint 为准。
实际训练代码位于同级目录 `visual-agent-online-crt`，分支为 `feat/online-crt`；启动代码提交为 `a530319799d56de095e3b97ef3f5ca065a52ba60`。
本次更新提交到实验分支；主仓库 `visual-agent/main` 中的同名文件仍为 2026-10-10 17:05 的历史快照。
可复核的原始指标及保存事件见 [step5–9 日志摘录](online_crt_16gpu_step5_step9_20261010.log.txt)。摘录保留原始数值，不包含启动环境、请求凭据或样本内容。

## 当前结论

- 64 卡已完成 step1–4，保存了完整 step4 checkpoint；2026-10-10 07:24:50 以退出码 1 结束，最终异常为 `KeyError: 'acc'`。
- 16 卡已从上述 step4 恢复，日志确认模型、优化器、RNG、LR scheduler 加载及 `Setting global step to 4`。
- 16 卡已完成 step5–9 的 actor 更新和保存，`latest_checkpointed_iteration.txt=9`，step9 保存完成时间为 2026-10-10 23:35:34。
- 每步均有 5,376 条轨迹、336 个训练槽位，每槽位恰有 16 条采样；step5、step7 各有 1 条 `reward_valid=0`，其余三步全部有效。
- 五个 checkpoint 均包含 14 个非空 model、14 个 optim、14 个 extra_state 分片，以及数据游标、CRT 状态和完整 HF 权重；控制器分别推进到对应的 step。
- step5–9 的 qualified groups 分别为 30、19、25、33、32；这五步累计 139 组，其中 56 mixed、34 全对、49 全错。
- 核对时任务正在执行 step10，尚无 `10.jsonl` 或 step10 checkpoint；主日志最后的奖励计算在 23:58:35 完成，两节点 GPU 监控持续更新。step10 当前阶段不能仅凭 GPU 利用率精确判断。
- 尚无训练后的验证或独立 benchmark 成绩，不能据训练 reward 宣称能力提升。

## 运行标识与实际配置

| 项目 | 64 卡首段 | 16 卡续训 |
|---|---|---|
| Run ID | `qwen3base_multitool_vlocr_online_crt_n16_8node_20261009T194559026966_2bf73edd` | `qwen3base_multitool_vlocr_online_crt_n16_2node_resume_step4_20261010T073736040085_12803986` |
| 节点 / RL GPU / 工具 GPU | 8 / 56 / 8 | 2 / 14 / 2 |
| Train batch / PPO mini-batch / rollout n | 336 / 112 / 16 | 336 / 112 / 16 |
| 初始化 | Qwen3-VL-8B-Instruct 基座，`resume_mode=disable` | 首段 `global_step_4`，`resume_mode=resume_path` |
| 数据 | 33,921 条训练、1,312 条验证 | 同一数据目录，恢复数据游标 |
| 学习率 / actor KL | 1e-6 / 0.001 | 1e-6 / 0.001 |
| Prompt / response 预算 | 8192 / 16384 | 8192 / 16384 |
| 工具服务 | 每节点 GPU7，9000/9001 两个进程 | 同左 |
| 实际保存 / 验证间隔 | 日志为 save1 / test40 | 日志及续训入口为 save1 / test40 |

这里使用实际日志配置；通用 online CRT 入口默认的 2 节点 batch126/mini42、save5/test5 不代表本次作业设置。
16 卡专用入口保持源 batch 和优化器调度边界，启用 `ALLOW_FSDP_WORLD_SIZE_CHANGE=True`，允许 56→14 ranks 重分片，`DATALOADER_NUM_WORKERS=2`、`VAL_BEFORE_TRAIN=False`。
数据来自主仓库 `data/zwz_multitool_relation20_hme_chartqa_tallyhalf_fsc3000_20261004/`，本次没有使用可选的 MME 数据补充。
首段启动日志虽显示一个 `global_step_40` 路径，但恢复模式为 `disable`，不能据路径字符串认定从 step40 续训。

## 训练结果与首段停止原因

以下 acc、score 为逐条 JSONL 的算术平均；耗时取 node0 的 `timing_s/step`，包含保存等步骤。每步均核验为 336×16 条轨迹。

| Step | 平均 acc | 平均 score | Step 耗时（秒） | 状态 |
|---|---:|---:|---:|---|
| 1 | 0.413318 | 0.458929 | 2398.737 | 已更新并保存 |
| 2 | 0.472470 | 0.519531 | 2315.890 | 已更新并保存 |
| 3 | 0.450707 | 0.504018 | 2285.542 | 已更新并保存 |
| 4 | 0.452195 | 0.505692 | 2577.532 | 已更新并保存 |
| 5，16 卡 | 0.486049 | 0.535826 | 6331.079 | 已更新并保存，17:40:14 |
| 6，16 卡 | 0.519159 | 0.565551 | 5236.119 | 已更新并保存，19:07:33 |
| 7，16 卡 | 0.495350 | 0.545145 | 5106.855 | 已更新并保存，20:32:53 |
| 8，16 卡 | 0.477121 | 0.527641 | 5590.628 | 已更新并保存，22:06:06 |
| 9，16 卡 | 0.486607 | 0.537240 | 5354.774 | 已更新并保存，23:35:34 |

保存时间均为 2026-10-10 北京时间。首段前四步约 38–43 分钟/step；16 卡首个恢复步为 105.5 分钟，后四步为 85.1–93.2 分钟。
全局 batch 保持 336×16，训练 GPU 从 56 减为 14；这些耗时来自不同训练批次，不能作为同批次拓扑吞吐对照。
step5、step7 的均值各包含一条无效奖励轨迹；有效轨迹 acc 分别为 0.486140、0.495442。不同 step 的题目和注错分配不同，均值差异不是配对收益。

各阶段耗时取完成步指标，单位为秒。`生成`使用 `timing_s/gen`，包含工具调用；`奖励`包含 judge；总耗时还包含优势计算、轨迹写出等。

| Step | 生成 | 奖励 | 策略 log probability | 参考 log probability | Actor 更新 | 保存 | 总耗时 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5 | 1043.361 | 178.027 | 1002.535 | 907.165 | 2968.246 | 181.170 | 6331.079 |
| 6 | 1048.598 | 160.289 | 731.832 | 727.519 | 2359.716 | 166.215 | 5236.119 |
| 7 | 948.294 | 164.132 | 734.502 | 726.519 | 2326.257 | 164.718 | 5106.855 |
| 8 | 1079.356 | 186.047 | 777.947 | 761.665 | 2579.354 | 165.858 | 5590.628 |
| 9 | 925.678 | 183.653 | 779.218 | 764.919 | 2483.733 | 172.677 | 5354.774 |

step9 的 actor 更新约 41.4 分钟，策略及参考 log probability 合计约 25.7 分钟，奖励约 3.1 分钟。当前主要耗时来自模型计算。

64 卡最终堆栈位于 `OnlineFaultController.update` 读取 `rows['acc']`。停止前出现 primary judge 限流及超时。
实验分支已有修复 `04ebe2a`：奖励异常直接暴露；无法得到 judge 结果时保留奖励元数据，标记无效，排除该轨迹的 GRPO advantage 和对应注错组的难度统计。
首段停止原因与 16 卡当前仍在执行的状态分别记录，不将首段异常当成续训已经失败。

## Checkpoint 与恢复证据

首段 `latest_checkpointed_iteration.txt=4`。step1–4 均存在 56 个非空 model、56 个 optim、56 个 extra_state 分片；step4 还存在非空 `data.pt`、`online_faults.json`、HF 权重索引及索引引用的权重文件。
step4 控制器状态为 `step=4`，六个 bucket 的故障等级均为 0。
16 卡 node0 日志第 896–923 行及第 1835–1837 行包含加载状态和 global step 恢复证据。
恢复实现读取 `data.pt` 和 `online_faults.json`；当前日志没有缺失 dataloader 的警告，step5–9 的保存已确认控制器恢复后继续推进。
五个 16 卡 checkpoint 的 `data.pt` 均为非空；`online_faults.json` 中的 step 分别为 5–9。每个 HF 索引引用 8 个权重文件，均存在且非空。
step9 checkpoint：`visual-agent-online-crt/saves/visual_agent_zwz_rl/qwen3/qwen3base_multitool_vlocr_online_crt_n16_2node_resume_step4_20261010T073736040085_12803986/global_step_9/`。
检查点保存在共享存储，未通过 Git 上传权重。

## 注错学习信号

仅统计真正收到 `injected_fault` 的轨迹；每组至少 4 条收到错误观察、且这些轨迹奖励有效，才算合格组。
下表由首段 step1–4 轨迹复核，并与 step4 的控制器累计 pending 统计一致；mixed 表示同一组收到错误观察的采样中同时存在答对与答错。

| Bucket | 合格组 | mixed | 全对 | 全错 | step4 下次分配比例 |
|---|---:|---:|---:|---:|---:|
| ZWZ / grounding_detect | 55 | 31 | 4 | 20 | 16.20% |
| DeepEyesV2 / grounding_detect | 12 | 7 | 2 | 3 | 14.15% |
| FSC147 / object_count | 12 | 0 | 0 | 12 | 3.96% |
| TallyQA / object_count | 5 | 2 | 1 | 2 | 10.86% |
| HME / ocr_read | 3 | 1 | 0 | 2 | 8.84% |
| 其他 OCR / ocr_read | 3 | 0 | 3 | 0 | 5.99% |

FSC147 在前四步的 12 个合格注错组全部答错，暂未出现组内区分信号。OCR/HME 组数较少，不支持效果结论。
16 卡 step5–9 的合格组统计如下，已与完成步日志中的 `qualified_groups` 和各 outcome rate 核对。

| Step | 合格组 | mixed | 全对 | 全错 |
|---|---:|---:|---:|---:|
| 5 | 30 | 12 | 6 | 12 |
| 6 | 19 | 7 | 9 | 3 |
| 7 | 25 | 8 | 4 | 13 |
| 8 | 33 | 18 | 8 | 7 |
| 9 | 32 | 11 | 7 | 14 |
| 合计 | 139 | 56 | 34 | 49 |

| Bucket，16 卡 step5–9 | 合格组 | mixed | 全对 | 全错 |
|---|---:|---:|---:|---:|
| ZWZ / grounding_detect | 101 | 41 | 26 | 34 |
| DeepEyesV2 / grounding_detect | 9 | 3 | 3 | 3 |
| FSC147 / object_count | 5 | 0 | 0 | 5 |
| TallyQA / object_count | 12 | 5 | 2 | 5 |
| HME / ocr_read | 11 | 7 | 2 | 2 |
| 其他 OCR / ocr_read | 1 | 0 | 1 | 0 |

step5–9 的六个 bucket 故障等级均为 0，分配比例和累计证据窗口已保存到各步 `online_faults.json`。上述为注错训练批次的学习信号，尚无同权重 clean/fault 对照评测。

## 奖励 judge 日志

下表来自这次 16 卡任务的五条 `[judge batch]` 记录，依次对应 step5–9。请求数为尝试次数，未记录实际 token usage，不能直接换算账单金额。

| Step | 进入 judge 的样本 | 主接口请求 | 主接口 429 | 主接口超时 | 备用接口请求 | 最终无有效判分 |
|---|---:|---:|---:|---:|---:|---:|
| 5 | 1561 | 1561 | 960 | 62 | 1025 | 1 |
| 6 | 1585 | 1585 | 1026 | 59 | 1091 | 0 |
| 7 | 1459 | 1459 | 944 | 67 | 1013 | 1 |
| 8 | 1577 | 1577 | 908 | 72 | 983 | 0 |
| 9 | 1594 | 1594 | 931 | 73 | 1009 | 0 |
| 合计 | 7776 | 7776 | 4769 | 333 | 5121 | 2 |

每批 5,376 条样本中约 27%–30% 进入 judge，其他样本由本地规则判分或因缺失答案直接记分。备用请求包含少量格式无效后的再次尝试。
两个最终无有效判分样本分别落在 step5、step7，均通过 `reward_valid` 排除；两步的 actor 更新和 checkpoint 保存正常完成。

## 验证与下一次验收

首段只有 `validation/0.jsonl`，共 1,312 条：HRBench4K 71.25%（800），depth 53.57%（84），TallyQA 45.37%（108），ChartQA 31.25%（160），OCR 74.38%（160）。
五来源等权宏平均为 55.1634%；`best_checkpoint.json` 仍指向基座 step0。该分数是本次奖励与工具配置下的初始验证，不能直接代替外部 benchmark。

16 卡续训配置为 `VAL_BEFORE_TRAIN=False`、`TEST_FREQ=40`，step5–9 未生成验证文件；这些新 checkpoint 尚无独立 benchmark 成绩。
后续验收：检查 step10 的完成指标和保存状态，再记录下一次固定验证点及同权重 clean/fault 对照。
本次记录更新没有修改训练配置、重启或停止作业。

## 证据位置与检查

以 `gx/` 为根目录：

- 代码与模型：`visual-agent-online-crt/`；checkpoint 为 `saves/visual_agent_zwz_rl/qwen3/<Run ID>/global_step_<N>/`。
- 日志：`logs/visual-agent-zwz-rl/<Run ID>/<Run ID>-node0.log`；GPU 监控为同目录 `*-node{0,1}-gpu.csv`。
- 轨迹：`rollouts/visual-agent-zwz-rl/<Run ID>/<N>.jsonl`。
- 验证：`visual-agent-online-crt/saves/visual_agent_zwz_rl/qwen3/<64 卡 Run ID>/validation/0.jsonl`。
- 64 卡入口：实验分支 `scripts/run_visual_agent_multitool_vlocr_online_crt_modelarts.sh`，通过环境指定 8 节点及上述 batch/save/test 设置。
- 16 卡入口：实验分支 `scripts/run_visual_agent_multitool_vlocr_online_crt_resume_step4_2node_16gpu_modelarts.sh`。
- 控制器配置 SHA256：`435ab2544006e32c50f579b1c30042c1605b69162a5c1f216a6ac8213aaaaf33`。
- 数据 manifest SHA256：`2dd18e7c9ee95d684d611c8a547031871afbf02e239b7481972d8410399eb608`。

2026-10-10 17:05 的代码检查记录：六个现有测试文件（online_crt、online_tool_faults、reward_failure_recovery、rl_eval_judge_alignment、visual_agent_thyme_reward、visual_agent_run_paths）通过标准库 unittest 执行：**79 tests，OK**。
使用已有 `visual-agent-qwen3vl-rl/bin/python3`，`PYTHONPATH=reinforcement_learning`；详细输出保存在本机 `/tmp/online-crt-focused-tests-20261010.log`。
当时额外的 `test_trainer_resume_checkpoint.py` 依赖 pytest，环境未安装，未能执行；online CRT 保存/恢复及 16 卡参数已有上述测试覆盖。此次已进一步核对真实 GPU 运行的 step5–9 完成指标、轨迹及保存文件。
本次只更新结果记录与日志摘录，检查每步轨迹数量、奖励有效标志、合格组统计、14 ranks 保存文件、HF 索引引用及文档数值。完整运行日志、样本数据、模型和请求凭据仍保留在原目录。
