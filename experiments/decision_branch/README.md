# 反事实决策分支与分层 advantage

状态：代码已实现，CPU 单元测试已编写；**尚未在集群上运行任何测试或训练**。

## 假设

标准 GRPO 给整条轨迹一个 advantage，把"选对工具"和"用好工具"混在一起。工具执行不稳时，
工具决策被系统性低估，策略塌向直接回答；概率很低的决策在组内采不到，成为吸收态。

## 方法与代码位置

决策 `d` 指第一轮 assistant 的第一个动作：某个工具，或 `direct`（直接回答）。后续轮次都算执行。

| 部分 | 实现 | 位置 |
|---|---|---|
| 强制分支 | 每题前 `forced_per_decision × |D|` 条 rollout 用 assistant prefill 强制决策前缀，其余 on-policy；总条数仍为 `rollout.n` | `verl/workers/rollout/decision_branch.py`、`chat_scheduler.py` |
| 决策 token | 前缀从模型 chat template 渲染：工具为 `<tool_call>\n{"name": "X", "arguments":`，直接回答为 `<answer`；均截在分词预切分边界，强制与采样的同一调用渲染出相同 token | 同上 |
| `π(d|x)` | 强制分支决策 token 的 `old_log_probs` 之和，在 `D` 内 softmax 归一 | `verl/trainer/ppo/decision_branch.py` |
| 效用与基线 | `U(x,d)` 为该题决策为 d 的全部轨迹平均回报；`V(x)=Σ π(d|x) U(x,d)`；`π` 不完整时退回组均值 | 同上 |
| 分层 advantage | 决策 token：`(U−V)/σ`；其余可训练 token：`(R−U)/σ`；σ 为题内回报标准差（同 GRPO） | 同上 |
| off-policy 更新 | 强制决策 `U−V>0` 乘 `forced_positive_weight`（默认 1），`<0` 乘 `forced_negative_weight`（默认 0）；采样决策不加权 | 同上 |

奖励、工具、prompt 均不改；actor 的 PPO loss 不改。

## 运行

```bash
bash scripts/run_visual_agent_multitool_vlocr_decision_branch_3node_24gpu.sh
```

其余配置与 `run_visual_agent_multitool_vlocr_3node_24gpu.sh` 相同。开关（均为环境变量）：

| 变量 | 默认 | 含义 |
|---|---|---|
| `DECISION_BRANCH_ADV` | `decision_branch` | `grpo`：保留强制分支但用普通 GRPO（消融） |
| `DECISION_BRANCH_FORCED_PER_DECISION` | 1 | 每个决策强制条数；`rollout.n` 必须大于 `|D|×该值` |
| `DECISION_BRANCH_FORCED_POSITIVE_WEIGHT` | 1.0 | 0 即关闭 off-policy 更新（消融） |
| `DECISION_BRANCH_FORCED_NEGATIVE_WEIGHT` | 0.0 | 强制负决策的权重 |
| `DECISION_BRANCH_DECISION_WEIGHT` | 1.0 | 决策 token advantage 倍数（`token-mean` 下决策 token 占比很小） |

未设置 `DECISION_BRANCH_ENABLE=True` 时，共用启动器生成的命令与原来完全相同。

## 消融顺序

1. vanilla GRPO（原入口）
2. 强制分支 + GRPO（`DECISION_BRANCH_ADV=grpo`）：区分"多了探索"与"advantage 分解"
3. 分层 advantage（`DECISION_BRANCH_FORCED_POSITIVE_WEIGHT=0`）
4. 完整方法（默认）

工具数扩展复用现有 tool config：GroundingDINO（2 个）、depth_count（4 个）、vlocr（5 个）。

## 诊断指标（wandb `decision_branch/*`，仅开启时记录）

- `entropy_mean`：`E_x H(D|x)`（nats），上限 `max_entropy = log|D|`
- `mutual_information`：`I(X;D) = H(E_x π) − E_x H(D|x)`
- `pi/<d>`、`utility/<d>`、`sampled_rate/<d>`：决策概率、反事实效用、实际采样频率
- `oracle_value_mean`、`value_mean`、`oracle_gain_mean`：`max_d U`、`V` 及差值，即 oracle 路由收益
- `pi_coverage_rate`：能得到完整 `π` 的题目比例；`span_missing_rate`：决策 token 未定位的比例（应接近 0）
- `singleton_decision_rate`：决策组只有 1 条样本、执行 advantage 为 0 的比例

离线分析已有训练 rollout 的决策分布（不需要 GPU）：

```bash
python3 experiments/decision_branch/analysis/rollout_decisions.py "$ROLLOUT_DATA_DIR"
```

## 实现取舍与已知限制

- 只处理首个决策；多步决策树未做。
- 强制负决策默认不更新（权重 0），这是对"advantage 为正的决策按 off-policy 方式更新"的解读；可调。
- `π(d|x)` 只计规范前缀写法的概率并在 `D` 内归一；模型极少输出其他写法，因为采样轨迹也经 hermes 解析后按模板重新渲染。
- `forced_per_decision=1` 时，强制分支若组内没有同决策的采样样本，执行 advantage 为 0；看 `singleton_decision_rate`。
- PPO clip 限制低概率决策每步的上升幅度；若跳出太慢，再考虑给决策 token 单独设 clip。
- 要求第一轮只输出 `<tool_call>` 或 `<answer>`（vlocr prompt 满足）；其他 prompt 需先确认。
- 不能与 A/D 双流同时开启。

## 待验证（集群）

```bash
python3 -m pytest -q tests/test_decision_branch_advantage.py tests/test_decision_branch_rollout.py experiments/decision_branch/tests tests/test_rl_response_budget.py tests/test_rollout_consistency.py
```

`test_decision_branch_rollout.py` 中真实分词器部分需要 `QWEN3_VL_PROCESSOR_PATH`；它检查前缀分词边界，
以及强制与采样的同一工具调用渲染出相同 token。随后先跑 20 步短训练，确认 `span_missing_rate≈0`、无 NaN。
