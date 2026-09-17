# 旧文档审查与归档候选

审查日期：2026-09-15。本轮只建立索引和历史标记，原文件未搬动或删除。
归档候选不等于内容无用，也不表示已确认其中每项技术结论正确。

| 文档 | 审查结论 | 后续建议 |
|---|---|---|
| [P2R STATUS](../../PERCEIVE2REASON_STATUS.md) | 2026-08-08 阶段状态，同时列出已完成与待集成工作 | 作为历史快照归档，不作当前操作指南 |
| [P2R GAPS](../../PERCEIVE2REASON_GAPS.md) | 当时的数据集与集成缺口清单 | 保留缺口背景，逐项核实后再合并 |
| [P2R DELIVERY](../../PERCEIVE2REASON_DELIVERY.md) | 交付报告仍含未完成工作及预期收益 | 历史归档，预期值不纳入成绩表 |
| [P2R COMPLETE](../../PERCEIVE2REASON_COMPLETE.md) | 声称完成，但仍列测试和风险；与 STATUS/GAPS 所处阶段不同 | 保留原文，不以标题判定可用性 |
| [P2R 使用说明](../../PERCEIVE2REASON_README.md) | quickstart 脚本直接引用 | 暂留原路径，后续核实命令与状态 |
| [P2R 实现说明](../../PERCEIVE2REASON_IMPLEMENTATION.md) | quickstart 脚本直接引用，含历史完成情况 | 暂留原路径，后续更新为与代码对应的说明 |
| [P2R 设计](../../perceive2reason_rl_design.md) | 设计动机与历史错误分析，不是效果验证 | 暂留，后续移至设计文档区时同步引用 |
| [GroundingDINO SFT 统计](../qwen3_groundingdino_experiment_summary.md) | 阶段性 VStar 与 SFT loss 分析，与当前汇总口径不同 | 保留并标注历史，不覆盖原统计 |

前四份 P2R 报告当前被 `.gitignore` 排除，不能假设 Git 已备份。正式迁移时需核对
归档副本与原文内容一致、将归档纳入版本控制，再移除原路径；本轮未执行迁移。

后续整理顺序：先核实 P2R 当前执行链，再合并使用说明和实现说明，最后修复 quickstart
中的文档路径及旧链接。不要仅凭“COMPLETE/STATUS”文件名自动删除。
