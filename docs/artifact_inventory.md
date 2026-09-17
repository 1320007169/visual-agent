# 产物与保留清单

日期：2026-09-15；范围仅为 `visual-agent/`。初始盘点未删除文件；后续经用户授权清理了
下方记录的 7 个中间 checkpoint。其余表格为清理前快照。
占用使用 `du -x -h --max-depth=1`，不跟随目录软链接，数字为二进制单位的近似磁盘占用。
共享存储的压缩、快照、硬链接和配额会影响实际释放量；不能将目录大小直接当作可释放空间。

## 权重与模型

### 已执行清理（2026-09-15）

以下路径均相对于 `saves/visual_agent_execution_plan/`。删除前已确认最新指针指向保留项，
并检查保留项的 HF 索引对应分片存在且非空；未执行权重内容校验。

| 实验目录 | 已删除 | 保留 |
|---|---|---|
| `step130_agent4_vision_opd_54step_64gpu` | `global_step_40` | `global_step_54` |
| `step130_dual_agent4_native4_vision_opd_54step_64gpu` | `global_step_40` | `global_step_54` |
| `step130_agent4_100step_64gpu` | `global_step_70` | `global_step_75` |
| `step130_agent4_100step_32gpu_from_64gpu_step75` | `global_step_95` | `global_step_100` |
| `step130_dual_agent4_native4_100step_64gpu` | `global_step_80` | `global_step_100` |
| `step130_dual_agent4_native4_nativefmtfix_100step_64gpu` | `global_step_60` | `global_step_80` |
| `step130_dual_agent4_native4_nativefmtfix_100step_16gpu_from_64gpu_step80` | `global_step_90` | `global_step_100` |

删除后该目录实测占用约 915G（清理前约 1.8T）。本次为直接删除，未创建备份或放入回收站；
这些中间步不能从本次操作恢复。最终 checkpoint、续训来源、预测和配置未删除。

### 清理前占用快照

| 路径 | 实测占用 | 用途 / 建议 |
|---|---:|---|
| `saves/` | 4.8T | 训练产物总计；先建立 checkpoint 登记表，不整体清理 |
| `saves/visual_agent_execution_plan/` | 1.8T | A/D、Vision-OPD 与续训实验；保留各实验最终权重、必要恢复点及评测来源 |
| `saves/visual_agent_zwz_rl/` | 1.4T | 原始关系 RL；保留 step130 对照、已登记成绩对应权重与恢复点 |
| `saves/visual_agent_thyme_slack015_four_experiments/` | 344G | SFT 消融及 RL 起点；核对下游依赖后决定保留 |
| `saves/visual_agent_groundingdino_mixed_56874/` | 344G | Mixed SFT；核对被引用的训练起点 |
| `saves/visual_agent_groundingdino_56874/` | 344G | 工具 SFT；核对对照实验与恢复需求 |
| `saves/visual_agent_combined_4679/` | 243G | 历史 SFT；列入逐 checkpoint 审查 |
| `saves/visual_agent_parquet_sft/` | 243G | 历史 SFT；列入逐 checkpoint 审查 |
| `saves/visual_agent_combined_8679/` | 126G | 历史 SFT 对照；保留成绩来源 |
| `saves/visual_agent_combined_64659/` | 12K | 几乎无权重空间收益，先核对配置文件 |
| `models/` | 236G | 下载模型总计，不能当作训练缓存删除 |
| `models/Qwen3-Next-80B-A3B-Instruct/` | 152G | 模型资源；需检查 judge/服务引用 |
| `models/Qwen3.8-27B/` | 52G | 模型资源；需检查合成/推理引用 |
| `models/Qwen3-VL-8B-Thinking/` | 17G | 模型资源；暂保留 |
| `models/ZwZ-8B/` | 17G | 模型资源；暂保留 |

子目录和父目录是包含关系，不重复相加。checkpoint 保留清单应记录：run ID、step、
HF 导出、优化器恢复状态、父 checkpoint、结果路径、是否仍被脚本引用；缺少这些信息前，
不按“旧日期”或 `best_huggingface` 名称删除其他恢复文件。

## 数据与评测产物

| 路径 | 已确认内容 | 建议 |
|---|---|---|
| `model/` | 约 10G，实际是嵌套在 `xiaoyi_tmpstorage/.../datasets/DeepEyesV2_RL/` 的 Parquet 数据 | 名称误导，但尚未验证是否重复；保留，后续比较数据版本和引用 |
| `data/` | 181G，多套 SFT/RL 数据、审核数据、VLMEval 数据 | 保留数据版本与 split；不能仅因文件名前缀相似就去重 |
| `Visual_Agent/` | 3.1G，原始轨迹及图片资源 | 保留，`images` 软链接依赖此目录 |
| `images` | 指向 `Visual_Agent/training_trajectories_natural/images` | 不删除目标，不跟随链接重复清理 |
| `outputs/` | 4.3G，总输出目录 | 下列子目录不要与总计重复相加 |
| `outputs/vlmeval/` | 3.7G，三榜预测、补测与评分结果 | 保留预测及协议；API failed 目录仍包含可复用成功行 |
| `outputs/rl_rollout_recovery_20260912*` | 原版 159M、v2 324M，轨迹恢复版本 | 保留审核来源；先核对新版是否完整覆盖旧版 |
| `outputs/rl_tool_screening_*` | `_20` 为 11M、`_v1` 为 42M | 保留人工审核结果和来源标识 |
| `outputs/analysis/` | 4K | 按是否可重建逐项核实 |
| `reinforcement_learning/outputs/` | 2.3M，按日期组织的运行目录 | 检查配置与运行元数据，不能统称临时文件 |

`data/` 主要子目录：`zwz_rl_vqa/` 90G、`thyme_rl_snapshot/` 45G、
`thyme_rl_realworld_perception/` 30G、`vlmeval/` 11G、`visual_agent_parquet_snapshot/` 3.1G、
`rl_distill/` 283M、`perceive2reason/` 171M。命名相似不证明内容相同。

数据目录的递归统计会触发网络盘元数据读取，不建议在每次启动训练时自动运行。
当前清单是一级目录审查，不是重复文件鉴定或每个 checkpoint 的删除清单。

## 缓存与自动备份

| 路径 | 实测占用 | 建议 |
|---|---:|---|
| 根 `__pycache__/` | 52K | 可重建缓存候选 |
| `scripts/__pycache__/` | 492K | 可重建缓存候选 |
| `tests/__pycache__/` | 288K | 可重建缓存候选 |
| 根 `.ipynb_checkpoints/` | 8K | 先比较是否有独有内容 |
| `scripts/.ipynb_checkpoints/` | 32K | 含启动脚本备份，先与当前版本比较 |
| `docs/.ipynb_checkpoints/` | 8K | 含旧实验统计备份，先比较 |
| `data/.ipynb_checkpoints/` | 11M | 数据自动备份，先检查独有内容 |
| `Visual_Agent/.cache/` | 9.3M | 先确认下载缓存是否参与断点恢复 |

以上仅为已统计的路径，不代表全仓库缓存总量。前三项 Python 字节码缓存不足 1MiB，真正的大头
是 checkpoint，但需先确认恢复和复现需求。`reinforcement_learning/verl.egg-info/` 可能是
当前可编辑安装的元数据，不建议当作普通缓存直接删除。
