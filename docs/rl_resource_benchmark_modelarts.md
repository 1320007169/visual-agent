# ModelArts 16 卡 RL 资源配比测试

这是真实 RL 短训练对照，不是工具 HTTP 微基准。仅在专用 ModelArts 任务执行，
每个任务 2 节点、每节点 8 GPU；两节点启动相同命令。每个任务只跑一组，
A/B/C 共提交三个任务，可以顺序复用同一批 16 卡，不需要同时占用 48 卡。
不要在已有训练任务内执行：公共启动器会清理节点上的 Ray。

## 三组设置

| 参数 | A：当前基线 | B：精简工具 | C：CPU 工具 |
|---|---|---|---|
| 训练 GPU 总数 | 14 | 14 | 16 |
| 每节点工具 GPU 数 | 1 | 1 | 0 |
| 每节点工具服务进程数 | 2 | 1 | 2 |
| 每进程 GroundingDINO 副本数 | 3 | 1 | 1 |
| CPU 工具线程 | 无显式调整 | 无显式调整 | 每进程 16，可修改 |

三组固定 batch=112、PPO minibatch=28、rollout n=8、temperature=1、max turns=6、
rollout 并发=112。112 能整除 14 和 16；本仓库 PPO 会先乘 rollout n 再按卡数分摊，
因此 28×8=224 也满足两种拓扑。沿用混合数据、GroundingDINO prompt、KL 和 judge 配置。
其余环境变量和代码版本应保持一致，不要只给一组改温度、超时、数据或 judge。

默认总计 7 step，前 2 step 预热，只汇总 step 3–7。不运行验证、不保存 checkpoint，
避免引入评测与大文件写入开销。仍保留训练日志、GPU 采样和 rollout 轨迹用于审核。

所有组从同一份**固定 HF 权重目录**开始，优化器/调度器和数据游标重置。
不能把 14 卡完整 FSDP checkpoint 原样用于 16 卡恢复。传入导出的 HF 目录，
不要传 `global_step_N` 根目录，也不要用仍被训练更新的 `best_huggingface`。
这个协议比较资源效率，不是原实验的断点续训，也不用于报告模型成绩。

## ModelArts 启动命令

入口：`scripts/run_visual_agent_rl_resource_benchmark_modelarts.sh`。
在平台配置 2 节点 × 8 卡，让每个节点执行相同命令：

```bash
bash /path/to/visual-agent/scripts/run_visual_agent_rl_resource_benchmark_modelarts.sh \
  --scenario A \
  --model-path /path/to/frozen_huggingface \
  --run-id resource_compare_01 \
  --run
```

第二个任务仅将 `--scenario A` 改为 `B`，第三个改为 `C`。三个任务使用相同
`--run-id` 和 `--model-path`。入口复用现有 ModelArts host/rank 自动识别、共享模型盘、
Qwen3-VL 环境、OpenCV overlay 和 judge 密钥加载；需具备当前 RL 任务的相同环境和挂载。
脚本不存储密钥，保持现有秘密文件或平台注入方式。

不加 `--run` 只打印计划，不启动服务、不写结果、不要求本地安装训练依赖。
CPU 预览根据当前进程允许的核心数生成，最终以任务节点的启动记录为准。
输出默认位于 `visual-agent/outputs/rl-resource-benchmark/resource_compare_01/{A,B,C}/`。
可用 `--output-root /shared/path` 指定两个节点都可见的目录；**不能使用节点本地目录**。
重试某组须换新的 run-id，不能向旧日志继续追加。

## CPU 组

```bash
bash /path/to/visual-agent/scripts/run_visual_agent_rl_resource_benchmark_modelarts.sh \
  --scenario C --model-path /path/to/frozen_huggingface \
  --run-id resource_compare_01 --cpu-servers 2 --cpu-threads 16 --run
```

默认从每个节点允许的逻辑 CPU ID 中取最后 32 个，分成互不重叠的两组；
工具进程限制 PyTorch intra-op=16、inter-op=1，训练进程及 Ray 使用其余核心。
至少给训练留下 16 个逻辑 CPU，否则拒绝启动。逻辑 CPU 不等于物理核，默认划分
不保证 NUMA/超线程最优；实际分配写入每节点 JSON，可在首轮后按节点拓扑进一步调整。
这是防止 CPU 工具挤占训练的基本隔离，不是容器级 CPU 配额。

CPU 服务使用现有 GroundingDINO FP32 路径，不引入量化、不自动回退 GPU。
如环境算子不支持 CPU，会直接失败并保留工具日志。保持现有请求/副本排队超时，
不要单独放宽 CPU 组超时掩盖性能退化；工具慢到报错是此次测试要观察的结果。

## 汇总

三个任务退出后执行以下命令，可在任何能读共享日志的机器上运行：

```bash
python /path/to/visual-agent/scripts/summarize_rl_resource_benchmark.py \
  /path/to/visual-agent/outputs/rl-resource-benchmark/resource_compare_01
```

生成 `summary.csv` 和 `summary.json`，包含：

- 去除预热后的完整 step、gen、reward、old_log_prob、ref、update_actor 耗时中位数。
- JSON 额外包含各数值指标的均值、最小值、最大值，便于核对生成长度、工具调用量。
- 工具调用数、轨迹记录的错误数、调用 P95（包含失败调用），以及轨迹覆盖是否完整。
- 每张卡的整任务显存峰值，包含启动和预热，不是仅测量窗口的峰值。
- 完整性状态：需要 5 个测量 step 和两节点成功退出记录。`complete` 仅表示任务完整，
  不代表工具零失败或该方案可用；必须同时查看工具错误、judge 日志和轨迹覆盖。

先看完整 step 是否更快，再看工具错误和生成长度是否相近。不能把工具失败导致
轨迹提前结束当作加速；跨 GPU 拓扑的采样结果也不保证逐位相同。
保留真实 judge，网络波动会进入 reward 耗时，需一起报告，不可私自剔除后只报总加速。
5 个 step 是首轮筛选，候选方案应重复独立任务再确认稳定收益。

此前的 [工具微基准](visual_tool_resource_benchmark.md) 仍可用于单独排查 CPU 推理，
但不替代本页的完整 RL 对照。
