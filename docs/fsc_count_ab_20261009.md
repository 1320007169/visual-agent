# FSC 计数工具 Agent 对照

执行顺序：统一纠正 FSC 评分基线 → 自动示例框 Agent 全量对照 → 验证分块计数 → 根据结果决定 observation 格式及 RL 训练。前两步完成前，不调整计数数据配比或采信奖励。

评分基线已完成：9 组已有预测统一按官方 `annotation_FSC147_384.json` 的 `points` 数重评分。历史预测及评分文件保留，实验总表已经更新；无效输出处理不变。详见 [评分基线报告](../outputs/vlmeval/fsc_baseline_corrected_20261009/report.md)。

## ModelArts 启动

- 资源：单节点 8×A100，0–6 卡运行模型副本，7 卡运行工具。
- 启动文件：`run_visual_agent/eval/run_visual_agent_eval_fsc_count_ab_8gpu_modelarts.sh`。
- 本地部署位置：`/home/ma-user/work/algorithm/codebkp/run_visual_agent/eval/run_visual_agent_eval_fsc_count_ab_8gpu_modelarts.sh`。
- 仓库入口：[run_visual_agent_eval_fsc_count_ab_8gpu_modelarts.sh](../scripts/run_visual_agent_eval_fsc_count_ab_8gpu_modelarts.sh)。
- 使用之前评测任务相同的镜像及共享存储挂载；不设评测超时。

固定新数据 64 卡 step80：`qwen3base_multitool_vlocr_hme_chartqa_tallyhalf_fsc3000_n16_8node_20261003T202135115738_f9c41af7/global_step_80/actor/huggingface`。

两组均重新推理全部 1190 题，依次运行 `text_only`、`auto_exemplar`。只切换 CountGD++ 后端；权重、提示词、原生工具协议、8 个回合、每回合 512 token、temperature=0、API 并发 7、只返回 count 的格式均相同。本阶段不启用自适应分块。参数和输入文件哈希记录于 `protocol.json`。

## 结果与验收

结果目录：`outputs/vlmeval/fsc_count_ab_8gpu/<run_id>/`。

- `text_only/`、`auto_exemplar/`：两组预测、评分与工具服务日志。
- `status.tsv`：退出状态及端到端耗时，包含数据校验、服务启动、推理、评分和进程退出。
- `comparison.json`、`report.md`：两组完成后自动汇总 MAE、RMSE、MAE_valid、RMSE_valid、无效输出数和耗时。
- `paired_examples.csv`：逐题预测、计数和绝对误差变化；另统计改善、退化、误差不变、有效转无效、无效转有效及两组均无效。

存在无效输出时，不将 MAE_valid 当成全量 MAE，也不只凭平均分判断收益。后续分块实验使用同样指标，并与这次结果分开记录。

## 启动状态

2026-10-09：入口及汇总程序已通过配置预检和评分测试，尚未提交 ModelArts。当前会话未配置可用的 ModelArts 提交客户端及认证。

此前本地 GPU 1 尝试因其他进程占用显存，在 vLLM 初始化时退出，没有产生新预测。失败记录保存在 `outputs/vlmeval/fsc_agent_count_ab_20261009/`；ModelArts 入口使用全新的结果目录。
