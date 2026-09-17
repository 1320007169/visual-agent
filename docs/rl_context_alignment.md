# RL rollout 与 actor 输入一致性

2026-09-17 修复了当前异步多轮 RL 中的两处输入差异：

- rollout 的 `tools` 字段会让聊天模板追加工具定义，但旧数据编码没有传入相同定义。
- actor 使用数据加载器预处理的图片，旧 rollout 则发送原图，由服务端按默认像素预算处理。

Git 历史可将差异追溯到本地适配：

- `c2672f0` 中的旧 scheduler 会重新编码包含 `tools` 的 prompt，再交给 actor。
  `8f7d245`（2026-07-20）为保留多模态 tokens 和位置编码，改为复用数据集的
  `input_ids`，但数据集仍未传入 `tools`，因此遗漏了模板追加的工具定义。
- `73d16f6`（2026-07-14）把 `origin_multi_modal_data` 原图接入模型 HTTP 请求。
  `8f7d245` 增加本地 `VISUAL_AGENT_IMAGE_MAX_PIXELS` 处理，`1446d13`
  （2026-07-21）将 Qwen3 训练默认上限设为 2359296；这个本地环境变量没有传入
  vLLM 的图像处理配置。`af93fdd` 的原图缓存优化沿用了这条路径。

这些提交说明了当前代码差异的来源，不能单凭提交时间断言某个历史训练进程的实际配置，
也不能据此认定它们是工具调用退化的唯一原因。

数据配置 `data.tool_config_path` 现在引用 rollout 的工具配置。数据加载器只读取和规范化
schema，不初始化工具客户端；训练、验证和超长样本过滤都使用同一份工具定义。
Native 双流分支显式禁用 schema，并继续使用自己的回答 prompt。

模型请求使用 `multi_modal_data.image` 中已经预处理的图片，按源样本缓存 PNG 编码。
视觉工具使用独立的 `origin_multi_modal_data.image` 原图列表，继续从原分辨率裁剪。
返回的裁剪图同时进入后续请求和 actor 多模态输入；观察内容的 loss mask 保持为零。
每条 rollout 深拷贝消息，防止图片替换影响重复采样或 Native 分支。

修复前的 CPU 复现使用官方 Qwen3-VL-8B-Instruct 处理器：工具定义产生 398 tokens
的 prompt 差值；4096×2048 测试图在 rollout 中为 8192 个视觉 tokens，actor 中为 2211。
修复后的测试逐元素比较初始和多轮完整序列的 token IDs、image_grid_thw 和 pixel_values，
并检查连续两次裁剪、原图分辨率、两种图片传输模式、重复采样和 Native 分支。

在 RL 环境中运行以下测试。`QWEN3VL_PROCESSOR_PATH` 可以指向基座或 checkpoint 的
本地分词器／处理器目录，只需相关 JSON 文件，不加载模型权重，也不连接模型或工具服务：

```bash
PYTHONPATH=reinforcement_learning \
QWEN3VL_PROCESSOR_PATH=/path/to/Qwen3-VL-8B-Instruct \
python -m pytest -q \
  reinforcement_learning/tests/workers/rollout/test_visual_agent_context_alignment_on_cpu.py \
  reinforcement_learning/tests/workers/rollout/test_visual_tool_image_response_on_cpu.py \
  reinforcement_learning/tests/workers/rollout/test_multiturn_training_contracts_cpu.py
```

未设置处理器路径时，真实处理器集成用例会跳过；必须设置路径才能完成此次回归验证。
CPU 输入对齐不等于 GPU 上已经验证了更新或生成概率，正式长训练前仍应完成短程 GPU 验证。
已经运行的训练进程不会自动加载修复。修复后应记录新的代码版本，并将新旧运行分别标记，
避免将输入协议变化混入原实验的因果比较。
