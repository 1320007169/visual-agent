# Qwen3 GroundingDINO SFT 简要统计

> 历史阶段记录：以下分数和“当前”状态保留原统计语境，不代表最新评测。
> 最新成绩、补测情况和统计口径见[三项 Benchmark 汇总](three_benchmark_evaluation_summary.md)。

统计范围：Qwen3-VL-8B base、旧版 56874 SFT、mixed 56874 SFT，以及基于 mixed SFT 的 RL step60。评测统一采用 VStarBench，共 191 题。

## 数据量

旧版 SFT：56,874 条，全部为工具轨迹；GroundingDINO 调用 61,599 次，crop_zoom 调用 14,286 次，assistant 文本约 198 万词。

mixed SFT：仍为同一批 56,874 个图文问题，其中直接回答 34,644 条（60.9%），工具轨迹 22,230 条（39.1%）；GroundingDINO 调用 24,102 次，crop_zoom 调用 4,594 次，assistant 文本约 115 万词。

两版数据都包含 17,736 张不同图像和 54,555 个不同问题。51,418 条（90.4%）来自 Flickr30k，数据来源比较集中；所有样本都没有 system message。

## SFT loss

两次训练配置相同：full SFT，冻结视觉塔和 projector，学习率 1e-5，训练 3 epochs，共 5,334 steps，没有验证集 loss。

旧版 SFT：初始 loss 4.954；第 1/2/3 个 epoch 的日志平均 loss 分别为 0.291、0.112、0.036；最终日志 loss 0.030，整体 train_loss 0.147。

mixed SFT：初始 loss 3.825；第 1/2/3 个 epoch 的日志平均 loss 分别为 0.472、0.248、0.090；最终日志 loss 0.077，整体 train_loss 0.270。

两条曲线都持续下降，第三个 epoch 已降得很低。训练过程本身稳定，但没有 eval_loss，无法证明更低的训练 loss 带来了更好的泛化。旧版 loss 更低，主要说明全工具轨迹更容易拟合，不代表效果更好。

## VStarBench 结果

Qwen3 base：总分 76.44%，属性 77.39%，相对位置 75.00%。

旧版 SFT：总分 62.30%，属性 64.35%，相对位置 59.21%。

mixed SFT：总分 62.83%，属性 71.30%，相对位置 50.00%。

mixed SFT + RL step60：总分 76.44%，属性 78.26%，相对位置 73.68%。

mixed SFT 相比旧版只提升 0.52 个百分点：属性提高 6.96 个百分点，但相对位置下降 9.21 个百分点。相比 base，mixed SFT 总分下降 13.61 个百分点，其中相对位置下降 25.00 个百分点，是主要退化项。

RL step60 将总分恢复到 base 水平，但还没有形成总体净增益；当前继续训练的 step90 及以后 checkpoint 尚未评测。

## 简要判断

当前 SFT 的主要问题不是 loss 不收敛，而是数据过窄、训练与评测格式不一致，以及工具轨迹的证据质量不足。mixed 数据减少了无必要工具调用，但直接回答样本没有覆盖 VStar 的四选一和空间关系分布，因此没有解决遗忘问题。后续更值得尝试的是减少 SFT epoch，并加入通用视觉、空间关系和长推理回放数据，而不是继续追求更低的训练 loss。
