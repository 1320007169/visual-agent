# 工具 CPU / GPU 资源配比测试

入口：`scripts/benchmark_visual_tool_resources.py`。只启动独立 GroundingDINO 工具服务，
不启动 RL、不改训练参数、不连接现有工具端点。默认仅预览，添加 `--run` 才执行。

## 测试方案

| 方案 | GPU 数 | 服务进程数 | 每进程模型副本 | CPU 设置 |
|---|---:|---:|---:|---|
| `gpu1_w1_r1` | 1 | 1 | 1 | 每进程最多 4 线程 |
| `gpu1_w2_r3` | 1 | 2 | 3 | 每进程最多 4 线程；对应当前单节点工具配置 |
| `gpu2_w4_r3` | 2 | 4 | 3 | 每卡 2 进程；模拟当前两节点工具总副本数 |
| `cpu_w1_t4/t8/t16` | 0 | 1 | 1 | 4 / 8 / 16 线程 |
| `cpu_w2_t4/t8/t16` | 0 | 2 | 1 | 每进程 4 / 8 / 16 线程，核心集合不重叠 |

CPU 方案只生成满足核心预算的组合。可通过 `--cpu-threads`、`--cpu-workers` 扩展，
通过 `--only` 指定方案。GPU 方案仅在显式提供 `--gpu-ids` 时生成。
CPU 使用当前服务的 FP32 推理，不引入量化或替代后端；CPU/GPU 不保证逐位一致。

## 准备样本

准备 JSONL，每行一个实际图片路径和 query。相对路径按 JSONL 所在目录解析：

```json
{"image":"/absolute/path/image1.jpg","query":"person"}
{"image":"/absolute/path/image2.jpg","name":"grounding_detect","arguments":{"query":"red car","target_image":0}}
{"image":"/absolute/path/image3.jpg","name":"sam3_segment_multi","arguments":{"queries":[{"role":"target","query":"person"},{"role":"reference","query":"chair"}],"target_image":0}}
```

默认复用训练工具的 `sam3_segment_multi` 接口，由 GroundingDINO 后端执行。
建议选择真实 rollout 的 20–50 个不同图片与工具参数，覆盖多 query、大图、无检测结果；
不要只重复一张小图。所有方案使用同一批次、同一请求顺序。

## 执行

在空闲测试节点上使用训练工具环境的 Python。以下 CPU / GPU 编号均为示例，
运行前确认没有训练或其他任务占用；本脚本不会自动抢占或停止任何已有服务。
CPU 核心编号是逻辑 CPU ID，需结合节点拓扑选择，不能直接把逻辑线程数当作物理核心数。

```bash
python scripts/benchmark_visual_tool_resources.py \
  --tool-python /path/to/visual-tools/bin/python \
  --model-path /path/to/grounding-dino-base-transformers \
  --cases /path/to/tool_cases.jsonl \
  --cpu-cores 32-63 --gpu-ids 6,7 \
  --concurrency 1,8,32,112 --requests 112 --repeats 3 \
  --output outputs/tool-resource-benchmark/run01
```

检查预览后，在同一命令末尾添加 `--run`。CPU-only 测试省略 `--gpu-ids`。
先做低并发筛选，例如 `--only cpu_w1_t8,gpu1_w2_r3 --concurrency 1,8`，
避免一开始运行所有 CPU 高并发组合而等待很久。完整矩阵可能需要数小时。

模型必须已在本地，禁止隐式下载。需要 CUDA 动态库路径等设置时，沿用工具环境的
`CUDA_HOME`、`LD_LIBRARY_PATH`；脚本不猜测服务器安装路径。CPU 跑不起来时保留错误日志，
不会静默切换 GPU。默认端口从 19100 起，占用时更换 `--port`，不杀占用进程。

## 结果与判读

- `results.json`：配置、服务启动耗时、每次测试结果、错误详情。
- `summary.csv`：每方案、每并发、每次重复的总耗时、成功吞吐、P50/P95/P99 和失败数。
- `*-server*.log`：独立服务日志。输出目录必须不存在，防止覆盖旧记录。

启动和预热不计入负载耗时；HTTP 客户端耗时包括服务排队，整批耗时还包括客户端调度。
失败请求不计入成功延迟分位数，因此必须同时看失败数。发生失败后停止当前方案，
清理后台推理再继续下一方案；整体返回非零退出码。Ctrl-C / SIGTERM 会清理本脚本启动的服务。
默认 112 次请求在并发 112 时只有一轮突发，可增大请求数至 448 观察持续负载。

优先比较 3 次重复的中位数和波动，要求零失败。CPU 核心多不保证推理快，
需要同时比较单请求延迟、高并发吞吐和尾延迟。建议用现有 GPU 方案作为基线，
先筛选工具 P95 和整批耗时可接受的候选，再做相同训练参数的短 RL 对照。

这是本机 HTTP 工具微基准，不包含跨节点网络、LLM rollout、judge、模型更新，
也不测训练/工具竞争 CPU 和内存时的性能。双 GPU 本机结果不等价于两节点结果。
工具加速不能直接换算成完整 step 加速；正式改配比前仍需验证完整 step 耗时、
显存峰值、工具错误率以及返回结果是否一致。不要依据本测试直接把训练改为 15 卡。
