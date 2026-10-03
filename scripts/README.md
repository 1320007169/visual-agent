# Scripts 入口索引

以下路径均相对仓库根目录。表中的资源和参数为脚本默认值，作业环境中的覆盖值以启动日志为准。

## 当前多工具 RL

| 用途 | 入口 | 默认资源 | 模型可调用工具 |
|---|---|---|---|
| 多工具 + 文本检测/识别 | [run_visual_agent_multitool_text_det_rec_modelarts_2node_16gpu.sh](run_visual_agent_multitool_text_det_rec_modelarts_2node_16gpu.sh) | 2 节点 x 8 卡；14 张 RL 卡 + 2 张工具卡 | 检测、裁剪、深度、计数、文本检测、文本识别 |
| 多工具 depth/count | [run_visual_agent_multitool_depth_count_modelarts_2node_16gpu.sh](run_visual_agent_multitool_depth_count_modelarts_2node_16gpu.sh) | 2 节点 x 8 卡；14 张 RL 卡 + 2 张工具卡 | 检测、裁剪、深度、计数 |
| 三节点 depth/count | [run_visual_agent_multitool_depth_count_3node_24gpu.sh](run_visual_agent_multitool_depth_count_3node_24gpu.sh) | 3 节点 x 8 卡；21 张 RL 卡 + 3 张工具卡 | 检测、裁剪、深度、计数 |

每节点默认 GPU 0-6 用于 RL，GPU 7 承载全部工具。depth/count 各启动两个服务进程；两个 Visual-Agent 服务进程各加载一个 GroundingDINO 副本。文本检测/识别入口另外启动一个 OCR 服务进程，分别加载 `PP-OCRv5_server_det` 和 `PP-OCRv5_server_rec`。

### 配置与 Prompt

| 配置项 | 两节点默认值 | 三节点默认值 |
|---|---|---|
| 初始权重 | Qwen3-VL-8B-Instruct | Qwen3-VL-8B-Instruct |
| 数据目录 | `data/zwz_deepeyesv2_depth_tallyqa5k_multitool_20260924` | 同左 |
| Prompt batch | 112 | 126 |
| `ROLLOUT_N` | 8 | 16 |
| 同时运行的轨迹上限 | 112，来自 ModelArts 入口 | 168 |
| 最大 assistant 回合数 | 8，含最终答案 | 8，含最终答案 |

- 六工具 RL：[tool schema](../reinforcement_learning/examples/sglang_multiturn/config/tool_config/visual_tool_multitool_depth_count_ocr_config.yaml)、[system prompt](../prompts/visual_agent_rl_system_multitool_depth_count_ocr.txt)。
- 四工具 RL：[tool schema](../reinforcement_learning/examples/sglang_multiturn/config/tool_config/visual_tool_multitool_depth_count_config.yaml)、[system prompt](../prompts/visual_agent_rl_system_multitool_depth_count.txt)。
- `depth_measure` 接收同一图片上的一个或多个 `bboxes_2d`，返回 `regions`，每项直接包含 `bbox_2d` 和 `depth_m`。
- `text_detect(target_image)` 返回 `regions`，每项只有 `bbox_2d`；`text_recognize(target_image, bboxes_2d)` 返回 `regions`，每项包含 `bbox_2d` 和 `text`。
- 模型看到精简后的工具返回；`grounding_detect` 保留 `boxes`、`labels` 和 `confidence`。RL 训练及验证 JSONL 的 `rollout_trace.tool_calls` 同时保存 `raw_result`（原始工具结果）和 `model_observation`（模型实际收到的文本）。返回图片仍按原协议传入模型，这两个字段不额外归档图片内容。
- 两节点 ModelArts 入口默认先做验证，每 40 步验证一次，每 20 步保存；最佳权重按各数据源验证准确率的宏平均选取。恢复方式见 [RL 训练说明](../docs/rl_multitool_training.md)。

### 运行命令

在两个训练节点分别执行相同命令，新的训练使用新的 `TRAIN_RUN_TOKEN`：

```bash
TRAIN_RUN_TOKEN=text_det_rec_01 \
    bash scripts/run_visual_agent_multitool_text_det_rec_modelarts_2node_16gpu.sh
```

OCR 服务使用 `PIPELINE_ROOT/.env` 中的 `VTS_OCR_ENV`，模型目录由 `PADDLEX_MODEL_ROOT` 指定，默认位于 `$BASE/visual-tools/paddlex-cache/official_models`。该目录需要包含 `PP-OCRv5_server_det` 和 `PP-OCRv5_server_rec`。新 OCR 入口的目标机器启动与推理效果尚未验证。

## 评测入口

| 用途 | 入口 | 默认行为 |
|---|---|---|
| Qwen3 base 五工具 prompt 评测 | [run_qwen3_vl_8b_four_tool_prompt_eval_modelarts.sh](run_qwen3_vl_8b_four_tool_prompt_eval_modelarts.sh) | 原始 Qwen3-VL-8B-Instruct；VStar、HR4K、HR8K、OCRBench、MME-RealWorld-Lite/CN |
| CountGD++ pseudo-exemplar 对照 | [run_qwen3_vl_8b_count_pseudo_v2_eval_modelarts.sh](run_qwen3_vl_8b_count_pseudo_v2_eval_modelarts.sh) | 替换计数后端和 prompt；默认 MME-RealWorld-Lite |
| 通用 checkpoint 评测 | [run_visual_agent_eval_qwen3.sh](run_visual_agent_eval_qwen3.sh) | 通过 `EVAL_MODELS` 等变量选择权重；默认配置仍指向历史 step80/90 |
| 连续评测多份 checkpoint | [run_visual_agent_eval_mixed_checkpoints_modelarts.sh](run_visual_agent_eval_mixed_checkpoints_modelarts.sh) | 多权重评测任务 |
| 64 卡、16 卡 VL-OCR 最新 checkpoint 对照 | [run_visual_agent_eval_multitool_vlocr_64gpu_16gpu_latest_8gpu_modelarts.sh](run_visual_agent_eval_multitool_vlocr_64gpu_16gpu_latest_8gpu_modelarts.sh) | 单节点 8 卡，依次评测两个运行的最新完整 checkpoint；使用训练时的五工具和 PaddleOCR-VL-1.6 |
| 工具行为诊断 | [run_visual_agent_eval_tool_diagnostic_modelarts.sh](run_visual_agent_eval_tool_diagnostic_modelarts.sh) | direct、auto、tool-first 等诊断模式 |
| 两卡评测入口 | [run_visual_agent_eval_qwen3_local_2gpu.sh](run_visual_agent_eval_qwen3_local_2gpu.sh) | 通用评测的两卡配置 |

历史文件名 `four_tool` 对应当前五工具评测。该评测使用 `ocr_read` 和 `ground_depth`；新 RL 使用拆分后的 `text_detect`/`text_recognize` 和批量 `depth_measure`，两套工具协议不同。

五工具评测的默认卡位：GPU 0 为 GroundingDINO，GPU 1 为 PP-OCRv5 与 depth，GPU 2 为 count，GPU 3-7 为五个模型推理副本。实验记录见 [五工具 prompt 评测](../docs/qwen3_vl_8b_four_tool_prompt_experiment.md)，成绩见 [评测汇总](../docs/three_benchmark_evaluation_summary.md)。

## SFT 入口

| 用途 | 入口 |
|---|---|
| Qwen3-VL SFT，默认单节点八卡 | [run_visual_agent_parquet_sft_qwen3.sh](run_visual_agent_parquet_sft_qwen3.sh)、[run_visual_tool_sft_qwen3.sh](run_visual_tool_sft_qwen3.sh) |
| 通用全量 SFT 启动器 | [train_visual_tool_sft_full.sh](train_visual_tool_sft_full.sh) |
| SFT smoke | [train_visual_tool_sft_smoke.sh](train_visual_tool_sft_smoke.sh) |
| Slurm 多节点提交 | [slurm/train_visual_tool_sft.sbatch](slurm/train_visual_tool_sft.sbatch) |
| 运行时 SFT 配置生成 | [prepare_visual_tool_sft_config.py](prepare_visual_tool_sft_config.py) |

两个 Qwen3 SFT 入口当前均默认使用 `visual_agent_parquet_sft` 数据集、单节点八卡和全局 batch 32。具体数据与配置以脚本及作业覆盖值为准。更多说明见 [SFT 指南](../docs/visual_tool_sft.md)。

## 数据准备与轨迹处理

| 用途 | 脚本 |
|---|---|
| 混合 RL 数据加入 depth 与 TallyQA | [prepare_visual_agent_multitool_rl.py](prepare_visual_agent_multitool_rl.py) |
| 原始关系数据、DeepEyesV2 混合数据 | [prepare_zwz_original_relation_rl.py](prepare_zwz_original_relation_rl.py)、[prepare_zwz_deepeyesv2_mixed_rl.py](prepare_zwz_deepeyesv2_mixed_rl.py) |
| HRBench4K 训练内验证集 | [prepare_hrbench4k_rl_validation.py](prepare_hrbench4k_rl_validation.py) |
| Camera-depth pilot 数据 | [prepare_raw_depth_camera_rl.py](prepare_raw_depth_camera_rl.py) |
| RL rollout 转 SFT、候选检查、审核导出 | [convert_rl_rollouts_to_sft.py](convert_rl_rollouts_to_sft.py)、[validate_rl_sft_candidates.py](validate_rl_sft_candidates.py)、[export_reviewed_rl_sft.py](export_reviewed_rl_sft.py) |
| 工具 SFT 转换与检查 | [convert_visual_tool_sft.py](convert_visual_tool_sft.py)、[validate_visual_tool_sft.py](validate_visual_tool_sft.py) |
| P2R SFT 准备与修复，本地未提交 | `prepare_p2r_sft.py`、`repair_p2r_natural_v2_bundle.py` |
| Visual-Agent Parquet 构建与素材生成，本地未提交 | `build_visual_agent_parquet.py`、`materialize_visual_agent_parquet.py` |
| Visual-Agent Parquet 同步 | [update_visual_agent_parquet.py](update_visual_agent_parquet.py) |
| Synthetic attribute Parquet，本地未提交 | `build_synthetic_attribute_parquet.py`、`materialize_synthetic_attribute_parquet.py` |
| 图片分辨率分析，本地未提交 | `analyze_image_resolutions.py` |

本地 `repair_data/` 保存 P2R 修复用的标注数据，尚未提交。

## 公共内部脚本

### VL-OCR 两个最新 checkpoint 评测

ModelArts 入口：`run_visual_agent/eval/run_visual_agent_eval_multitool_vlocr_64gpu_16gpu_latest_8gpu_modelarts.sh`。
提交一个单节点 8 卡任务，先评测 64 卡训练的最新完整 checkpoint，再评测 16 卡续训的最新完整 checkpoint。
启动时分别读取当前两个运行目录的 `latest_checkpointed_iteration.txt`，并验证 HF 权重分片；
这里的“最新”指已完整保存的 checkpoint，不是最后完成但尚未保存的训练 step。
沿用 [VL-OCR 评测](run_visual_agent_eval_multitool_vlocr_8gpu.sh)：7 个模型副本、1 张工具卡、
并发 14、五工具 native/hermes 协议、训练 prompt、最多 8 轮、每轮 512 token，OCR 使用 PaddleOCR-VL-1.6。
默认评测现有 10 个 benchmark，包括 OCRBench 和 ChartQA_TEST；可用 `EVAL_DATASETS="OCRBench ChartQA_TEST"` 只跑这两项。
环境沿用最近的评测入口：模型及 VLMEval 使用 `qwenvl3_xmx_vLLM`，CUDA 和 GCC/G++ 使用
`$BASE/conda_envs/spacetools-rl`，入口设置相同的 `CUDA_LIBRARY_DIR`、`CC`、`CXX` 和 `CUDAHOSTCXX`。
工具 CUDA 使用 `cuda118`；depth 使用 `starVLA_flash_dzw1`，count 使用 `.env` 中的 `VTS_COUNT_ENV`，
PaddleOCR-VL 使用 `$BASE/conda_envs/qwen38-vllm-clean`，GroundingDINO 使用 `$BASE/conda_envs/visual-tools`。

结果分别保存在 `outputs/vlmeval/multitool_vlocr_64gpu_16gpu_latest_8gpu/<run_id>/64gpu_step<N>/dino_latest/`
及 `16gpu_step<N>/dino_latest/`，`checkpoints.tsv` 记录步数及模型源路径，`status.tsv` 记录两边退出码及耗时。
与其他评测入口一致，直接读取原始 HF 模型目录；评测期间需保留选中的 checkpoint。
两个模型依次调用现有评测入口，各自启动并清理工具服务；第一份失败仍尝试第二份，整体返回非零状态。
已有结果目录拒绝复用；`CHECKPOINT_EVAL_CONFIG_ONLY=1` 只检查选中的 checkpoint，不启动服务或创建结果目录。

```bash
CHECKPOINT_EVAL_CONFIG_ONLY=1 bash scripts/run_visual_agent_eval_multitool_vlocr_64gpu_16gpu_latest_8gpu_modelarts.sh
```

### 六工具双权重评测

入口：[run_visual_agent_eval_multitool_ocr_step20_8gpu.sh](run_visual_agent_eval_multitool_ocr_step20_8gpu.sh)。
沿用此前 step80 的单节点 8 卡评测协议，依次评测 64 卡 OCR 训练的 step20 和 24 卡训练当前最新完整的 step40（路径固定，便于复现）。
使用六工具 RL prompt，最多 8 轮、每轮 512 token；评测 VStarBench、HRBench8K、OCRBench、MME-RealWorld-Lite、HRBench4K、MME-RealWorld-CN、CV-Bench-2D、CV-Bench-3D、ChartQA_TEST、FSC147_TEST。
ChartQA 使用 `VLMEVAL_CHARTQA_RULE_ONLY=1`，沿用 2026-09-22 Qwen3 base 直答基准的规则评分模式。
FSC147_TEST 使用已有 CountGDPlusPlus 的 FSCD 点标注（1,190 张测试图片），按每张图片的类别生成文字计数问题，不提供 exemplar 框。
这是文字指定目标的六工具 Agent 计数评测，与原始 FSC 的 few-shot exemplar 协议不同。
FSC 图片预先下载到共享目录 `$BASE/visual-tools/datasets/fsc147/images_384_VarV2`，下载包约 204 MB，校验 SHA256 和图片完整性。
提交后，脚本在 GPU 服务启动前调用 `prepare_fsc147_eval.py`，校验本地图片并生成 TSV。默认 `FSC147_DOWNLOAD_IMAGES=0`，无需在评测作业里下载。
可通过 `FSC147_IMAGE_ROOT` 指定其他已有图片目录；确需在线补下载时设置 `FSC147_DOWNLOAD_IMAGES=1`。
评分输出 `MAE`、`RMSE`、有效回答率和失败数，无需 judge API；若有无效回答，完整集 MAE/RMSE 留空，同时单独报告 `MAE_valid`/`RMSE_valid`，避免跳过失败样本而高估效果。
7 卡用于模型副本，1 卡用于 GroundingDINO、深度、计数和独立 OCR 检测/识别服务。
保留并发 14、GroundingDINO 副本 1、原有 vLLM 和 CUDA 环境。
每次新建 `outputs/vlmeval/multitool_ocr_step20_8gpu/<run_id>/`，保留服务日志及评测轨迹；已有目录会拒绝复用。
预检使用独立临时结果目录，成功或失败后自动清理，日志仍保存在 `LOG_DIR`；可以固定 `RUN_ID`/`WORK_ROOT` 先预检再正式启动。
两个权重分别写入 `64gpu_step20/`、`24gpu_step40/`，共用 OCR、深度、计数服务，模型副本依次加载。
`status.tsv` 保存完成状态；第一个权重评测失败时停止。`HF_HOME` 指向已有 CN 数据集缓存 `$BASE/hf_cache`。

```bash
# 沿用已有预检模式，不启动 GPU 服务。
EVAL_PREFLIGHT_ONLY=1 bash scripts/run_visual_agent_eval_multitool_ocr_step20_8gpu.sh

# 使用现有转换器生成 ModelArts 提交入口，保留平台软链接初始化。
python scripts/convert_training_launcher_to_modelarts.py \
  scripts/run_visual_agent_eval_multitool_ocr_step20_8gpu.sh --profile basic \
  --output /home/ma-user/work/algorithm/codebkp/run_visual_agent/eval/run_visual_agent_eval_multitool_ocr_step20_8gpu_modelarts.sh
```

可用 `STEP20_MODEL_PATH`、`STEP24_MODEL_PATH`、`EVAL_DATASETS` 覆盖默认选择。
ModelArts 申请单节点 8 卡；提交生成的 `.sh` 入口，使用 Bash 执行。

这些文件负责启动链中的某一步。使用上面的具体实验入口可以得到对应的数据、环境、工具与训练参数。

| 职责 | 文件 |
|---|---|
| 多工具进程启动、健康检查与退出清理 | [run_visual_agent_multitool_depth_count_2node_16gpu.sh](run_visual_agent_multitool_depth_count_2node_16gpu.sh) |
| ZWZ 参数基础层 | [run_visual_agent_zwz_rl_2node_16gpu.sh](run_visual_agent_zwz_rl_2node_16gpu.sh) |
| VERL/Ray 集群、训练参数与工具进程启动 | [run_visual_tool_rl_2node_16gpu.sh](run_visual_tool_rl_2node_16gpu.sh) |
| 输出目录和跨节点共享 run token | [prepare_visual_agent_run_paths.sh](prepare_visual_agent_run_paths.sh)、[resolve_visual_agent_run_token.py](resolve_visual_agent_run_token.py) |
| 混合 RL 恢复路径选择 | [resolve_mixed_rl_resume.py](resolve_mixed_rl_resume.py) |
| Visual-Agent HTTP 工具服务 | [visual_tool_server.py](visual_tool_server.py) |
| 独立 PP-OCRv5 检测/识别服务 | [paddleocr_split_server.py](paddleocr_split_server.py) |
| VTS 文件与 HTTP 协议桥接 | [vts_tool_bridge.py](vts_tool_bridge.py) |
| 模型推理服务与客户端 | [serve_visual_agent_model.sh](serve_visual_agent_model.sh)、[serve_visual_agent_transformers.py](serve_visual_agent_transformers.py)、[visual_agent_inference.py](visual_agent_inference.py) |

## 其他训练与对照实验

这些入口保留各自实验设置；选择时结合数据、起始权重、恢复方式和对应文档。

| 实验 | 入口或记录 |
|---|---|
| GroundingDINO + KL 混合 RL | [run_visual_agent_zwz_deepeyesv2_n8_2node_16gpu.sh](run_visual_agent_zwz_deepeyesv2_n8_2node_16gpu.sh)；默认支持自动恢复 |
| 原始关系 GroundingDINO 对照 | [run_visual_agent_zwz_rl_groundingdino_2node_16gpu.sh](run_visual_agent_zwz_rl_groundingdino_2node_16gpu.sh) |
| Step130 双流续训 | [run_visual_agent_step130_stream_ablation_32gpu.sh](run_visual_agent_step130_stream_ablation_32gpu.sh)；[实验记录](../docs/three_benchmark_evaluation_summary.md#step130-双流续训) |
| Thyme RL 与 smoke | [run_visual_agent_thyme_rl_2node_16gpu.sh](run_visual_agent_thyme_rl_2node_16gpu.sh)、[run_visual_agent_thyme_rl_8gpu_smoke.sh](run_visual_agent_thyme_rl_8gpu_smoke.sh) |
| SLIME 后端实验 | [run_visual_agent_slime_poc.sh](run_visual_agent_slime_poc.sh)；[说明](../docs/slime_visual_agent_poc.md) |
| Perceive2Reason | [run_visual_agent_perceive2reason_2node_16gpu.sh](run_visual_agent_perceive2reason_2node_16gpu.sh)；[说明](../PERCEIVE2REASON_README.md) |

## 资源测量与环境工具

| 用途 | 入口或说明 |
|---|---|
| 实际 RL 的 GPU/CPU 工具配比对照 | [run_visual_agent_rl_resource_benchmark_modelarts.sh](run_visual_agent_rl_resource_benchmark_modelarts.sh)；[协议](../docs/rl_resource_benchmark_modelarts.md) |
| 独立工具服务压测 | [benchmark_visual_tool_resources.py](benchmark_visual_tool_resources.py)、[benchmark_visual_tool_pool.py](benchmark_visual_tool_pool.py)；[说明](../docs/visual_tool_resource_benchmark.md) |
| Count 后端与 depth pilot 对照 | [compare_count_backends.py](compare_count_backends.py)、[evaluate_depth_camera_pilot.py](evaluate_depth_camera_pilot.py) |
| RL 与五工具评测结果汇总 | [summarize_rl_resource_benchmark.py](summarize_rl_resource_benchmark.py)、[summarize_four_tool_prompt_eval.py](summarize_four_tool_prompt_eval.py) |
| GPU 与 W&B 监控 | [monitor_gpu_load.sh](monitor_gpu_load.sh)、[wandb_gpu_monitor.py](wandb_gpu_monitor.py) |
| ModelArts 入口生成 | [convert_training_launcher_to_modelarts.py](convert_training_launcher_to_modelarts.py) |
| 环境与模型准备 | [install_qwen3vl_sft_rl_shared.sh](install_qwen3vl_sft_rl_shared.sh)、[download_visual_tool_models.sh](download_visual_tool_models.sh) |
