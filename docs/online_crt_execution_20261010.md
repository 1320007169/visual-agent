# Online CRT 实验执行记录：64 卡转 16 卡

核对时间：2026-10-10 17:03–17:05（北京时间）。这是执行快照，后续进度以原始日志和 checkpoint 为准。
实际训练代码位于同级目录 `visual-agent-online-crt`，分支为 `feat/online-crt`；核对时 HEAD 为 `a530319799d56de095e3b97ef3f5ca065a52ba60`。
当前 `visual-agent/main` 保存同名记录，便于从主仓库追踪；实验代码及运行状态记录提交到实验分支。

## 当前结论

- 64 卡已完成 step1–4，保存了完整 step4 checkpoint；2026-10-10 07:24:50 以退出码 1 结束，最终异常为 `KeyError: 'acc'`。
- 16 卡已从上述 step4 恢复，日志确认模型、优化器、RNG、LR scheduler 加载及 `Setting global step to 4`。
- 16 卡已写出完整 `5.jsonl`：5,376 条轨迹、336 个训练槽位，每槽位恰有 16 条采样；1 条轨迹的 `reward_valid=0`。
- 核对时 16 卡尚无 step5 完成指标、`global_step_5` 或 `latest_checkpointed_iteration.txt`。轨迹写出不能作为 actor 更新完成的证据。
- 17:03:09 两节点 GPU 监控仍在更新，每节点 0–6 卡利用率均为 100%，7 卡工具 GPU 为 0%。监控显示硬件为 A800-SXM4-80GB；当前没有正常退出或新 checkpoint 证据。
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

## 64 卡训练与停止原因

以下 acc、score 为逐条 JSONL 的算术平均；耗时取 node0 的 `timing_s/step`，包含保存等步骤。每步均核验为 336×16 条轨迹。

| Step | 平均 acc | 平均 score | Step 耗时（秒） | 状态 |
|---|---:|---:|---:|---|
| 1 | 0.413318 | 0.458929 | 2398.737 | 已更新并保存 |
| 2 | 0.472470 | 0.519531 | 2315.890 | 已更新并保存 |
| 3 | 0.450707 | 0.504018 | 2285.542 | 已更新并保存 |
| 4 | 0.452195 | 0.505692 | 2577.532 | 已更新并保存 |
| 5，16 卡 | 0.486049 | 0.535826 | 尚无完成指标 | 仅确认轨迹与奖励落盘 |

首段前四步约 38–43 分钟/step；目前没有完整 16 卡 step 耗时，不能比较两种拓扑吞吐。
step5 均值包含那条无效奖励轨迹，不是排除无效项后的均值；不同 step 的题目和注错分配不同，均值差异不是配对收益。

64 卡最终堆栈位于 `OnlineFaultController.update` 读取 `rows['acc']`。停止前出现 primary judge 限流及超时。
实验分支已有修复 `04ebe2a`：奖励异常直接暴露；无法得到 judge 结果时保留奖励元数据，标记无效，排除该轨迹的 GRPO advantage 和对应注错组的难度统计。
首段停止原因与 16 卡当前仍在执行的状态分别记录，不将首段异常当成续训已经失败。

## Checkpoint 与恢复证据

首段 `latest_checkpointed_iteration.txt=4`。step1–4 均存在 56 个非空 model、56 个 optim、56 个 extra_state 分片；step4 还存在非空 `data.pt`、`online_faults.json`、HF 权重索引及索引引用的权重文件。
step4 控制器状态为 `step=4`，六个 bucket 的故障等级均为 0。
16 卡 node0 日志第 896–923 行及第 1835–1837 行包含加载状态和 global step 恢复证据。
恢复实现读取 `data.pt` 和 `online_faults.json`；当前日志没有缺失 dataloader 的警告，但控制器恢复后是否正常推进到 step5 仍需新 checkpoint 验收。

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
16 卡 step5 轨迹中：ZWZ 有 26 个合格组（11 mixed、5 全对、10 全错），其他 OCR 1 组全对，TallyQA 2 组（1 mixed、1 全错），FSC147 1 组全错；DeepEyesV2 的 2 个分配组均未实际注错。上述仅是轨迹统计，不代表控制器更新已经完成。

## 验证与下一次验收

首段只有 `validation/0.jsonl`，共 1,312 条：HRBench4K 71.25%（800），depth 53.57%（84），TallyQA 45.37%（108），ChartQA 31.25%（160），OCR 74.38%（160）。
五来源等权宏平均为 55.1634%；`best_checkpoint.json` 仍指向基座 step0。该分数是本次奖励与工具配置下的初始验证，不能直接代替外部 benchmark。

后续验收：确认 16 卡 `step:5` 完成行、`latest_checkpointed_iteration.txt=5`，以及 `global_step_5` 中 14 ranks 的模型/优化器/额外状态、数据游标、控制器 `step=5` 均完整；再记录下一次固定验证点及同权重 clean/fault 对照。
本次没有修改训练配置、重启或停止作业。

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

六个现有测试文件（online_crt、online_tool_faults、reward_failure_recovery、rl_eval_judge_alignment、visual_agent_thyme_reward、visual_agent_run_paths）通过标准库 unittest 执行：**79 tests，OK**。
使用已有 `visual-agent-qwen3vl-rl/bin/python3`，`PYTHONPATH=reinforcement_learning`；详细输出保存在本机 `/tmp/online-crt-focused-tests-20261010.log`。
额外的 `test_trainer_resume_checkpoint.py` 依赖 pytest，当前环境未安装，未能执行；online CRT 保存/恢复及 16 卡参数已有上述测试覆盖。单元测试不等于完整 GPU 恢复后更新验收。
日志、数据、模型、token 不纳入 Git，本记录保留可复核的路径和统计快照。
