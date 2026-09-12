# Perceive2Reason代码实现总结

## 已完成的工作 ✅

### 1. 核心Rollout Worker
**文件**: `reinforcement_learning/verl/workers/rollout/sglang_rollout/perceive2reason_rollout.py`

**类**: `Perceive2ReasonRollout(SGLangRollout)`

**实现的功能**:
- ✅ 继承标准SGLangRollout，保持兼容性
- ✅ 通过环境变量`PERCEIVE2REASON_MODE`控制开关
- ✅ 从`non_tensor_batch`中提取`perception_prefix`字段
- ✅ 将固定的工具调用历史注入到消息中
- ✅ 通过logit_bias禁止`<tool_call>` token生成
- ✅ 对无prefix的样本自动fallback到标准rollout

**关键方法**:
```python
_preprocess_prompt_to_async_rollout_requests()  # 提取perception_prefix
_async_rollout_a_request()                       # 注入固定prefix
_handle_engine_call()                            # 添加logit_bias
_inject_perception_prefix()                      # 构建消息历史
```

### 2. 数据准备脚本
**文件**: `reinforcement_learning/data_preprocessing/prepare_perceive2reason_data_from_train.py`

**功能**:
- ✅ 从训练集rollout结果中提取P2R样本
- ✅ 筛选条件：工具成功 + 检测有效 + 答案错误
- ✅ 提取perception_prefix（工具调用历史）
- ✅ 生成统计报告和样本计数

**使用方法**:
```bash
python prepare_perceive2reason_data_from_train.py \
  --rollout-dir outputs/train_rollout \
  --output data/perceive2reason/p2r_train.jsonl
```

### 3. 训练启动脚本
**文件**: `scripts/run_visual_agent_perceive2reason_2node_16gpu.sh`

**配置**:
- ✅ 基于成功的groundingdino_kl_safe配置
- ✅ 添加P2R专用环境变量
- ✅ 保持相同的超参数（ACTOR_LR=1e-6, KL_COEF=0.001）
- ✅ 数据路径和混合比例配置
- ✅ 验证P2R数据文件存在性

**关键环境变量**:
```bash
PERCEIVE2REASON_MODE=true
PERCEIVE2REASON_DATA_PATH=/path/to/p2r_train.jsonl
PERCEIVE2REASON_MIX_RATIO=0.3
ROLLOUT_WORKER_TYPE=perceive2reason
```

### 4. 配置文件
**文件**: `reinforcement_learning/examples/sglang_multiturn/config/tool_config/perceive2reason_config.yaml`

**内容**:
- ✅ 工具定义（grounding_detect, crop_zoom）
- ✅ P2R特定配置（enabled, data_path, mix_ratio）
- ✅ 工具禁止设置（logit_bias模式）

### 5. 文档
**文件**:
- ✅ `PERCEIVE2REASON_README.md` - 完整使用说明
- ✅ `perceive2reason_rl_design.md` - 设计文档（已更新）

## 架构设计

### 数据流

```
训练集 → 用step165跑inference → 提取rollout traces
                                    ↓
                          筛选：检测成功+答案错
                                    ↓
                          p2r_train.jsonl (perception_prefix)
                                    ↓
                          DataLoader混合30% P2R + 70% 标准
                                    ↓
                          Perceive2ReasonRollout处理
                                    ↓
                          - 检测perception_prefix字段
                          - 注入固定工具调用历史
                          - 禁止<tool_call> token
                          - 只生成最终答案
                                    ↓
                          GRPO更新只影响推理部分
```

### Hook点设计

```python
# 1. 数据预处理阶段
_preprocess_prompt_to_async_rollout_requests():
    从non_tensor_batch提取perception_prefix
    附加到req.meta_info["perception_prefix"]

# 2. Rollout启动阶段
_async_rollout_a_request():
    检测perception_prefix存在
    → 调用_inject_perception_prefix()
    → 设置req.meta_info["prohibit_tool_calls"] = True
    → 设置req.state = RUNNING (跳过PENDING阶段)

# 3. 生成调用阶段
_handle_engine_call():
    检测prohibit_tool_calls标志
    → 构建logit_bias字典
    → 添加到kwargs传给sglang引擎
```

## 待完成的工作 ⚠️

### 关键缺失部分

1. **训练数据inference脚本**
   - 需要实现`run_inference_on_train.py`
   - 用step165对训练集跑推理获取rollout traces
   - 保存每个样本的tool_calls和prediction

2. **数据加载和混合逻辑**
   - 修改DataLoader读取p2r_train.jsonl
   - 按`PERCEIVE2REASON_MIX_RATIO`混合P2R和标准样本
   - 将perception_prefix放入non_tensor_batch

3. **Rollout Worker集成**
   - 在训练配置中指定使用Perceive2ReasonRollout
   - 可能需要修改trainer代码来实例化正确的worker类
   - 或者通过factory pattern动态选择

4. **Token序列和Loss Mask**
   - Perceive2ReasonRollout中`_merge_prefix_and_suffix()`未实现
   - 需要正确构建：
     - response_ids = prefix_ids + reasoning_ids
     - loss_mask = [0, 0, ..., 0, 1, 1, ..., 1]
                    ↑ prefix部分   ↑ reasoning部分

5. **测试和验证**
   - 单元测试Perceive2ReasonRollout
   - 小规模实验验证P2R逻辑正确
   - 确认工具调用真的被禁止了

## 使用流程

### 完整训练流程（理想状态）

```bash
# Step 1: 准备P2R数据（从训练集）
python run_inference_on_train.py \
  --checkpoint saves/.../global_step_165 \
  --data /path/to/zwz_train.jsonl \
  --output outputs/train_rollout

python prepare_perceive2reason_data_from_train.py \
  --rollout-dir outputs/train_rollout \
  --output data/perceive2reason/p2r_train.jsonl

# Step 2: 启动P2R训练
export PERCEIVE2REASON_MODE=true
export PERCEIVE2REASON_DATA_PATH=data/perceive2reason/p2r_train.jsonl
bash scripts/run_visual_agent_perceive2reason_2node_16gpu.sh

# Step 3: 评测
python run_visual_agent_eval.py \
  --checkpoint saves/visual_agent_perceive2reason/.../global_step_XX \
  --benchmarks VStarBench HRBench4K HRBench8K
```

### 当前可以做的（部分实现）

```bash
# 1. 测试数据准备脚本（但需要先有train_rollout数据）
python prepare_perceive2reason_data_from_train.py --help

# 2. 检查Perceive2ReasonRollout代码
cat reinforcement_learning/verl/workers/rollout/sglang_rollout/perceive2reason_rollout.py

# 3. 阅读文档理解设计
cat PERCEIVE2REASON_README.md
cat perceive2reason_rl_design.md
```

## 技术细节

### Logit Bias实现

```python
# 在_handle_engine_call中
if prohibit_tool_calls:
    # 获取所有可能触发工具调用的token ID
    tool_call_token_ids = [123, 456, 789, ...]  # <tool_call>, <tool, tool_call等

    # 构建logit bias字典
    logit_bias = {tid: -100.0 for tid in tool_call_token_ids}

    # 传递给sglang引擎
    kwargs["logit_bias"] = logit_bias
    output = await self._engine.async_generate(..., sampling_params=...)
```

这会让模型在生成时，所有工具调用相关的token概率变为接近0，强制模型直接生成答案。

### Perception Prefix格式

```json
{
  "perception_prefix": [
    {
      "role": "assistant",
      "content": "<tool_call>{\"name\": \"grounding_detect\", \"arguments\": {\"query\": \"red car\"}}</tool_call>"
    },
    {
      "role": "tool",
      "content": "<tool_response>{\"boxes\": [[100,200,300,400]], \"confidence\": [0.95]}</tool_response>"
    }
  ]
}
```

注入后的消息历史：
```
[system] You are a visual agent...
[user] <image> What color is the car?
[assistant] <tool_call>{"name": "grounding_detect", ...}</tool_call>  ← 固定
[tool] <tool_response>{"boxes": ...}</tool_response>                 ← 固定
[assistant] ___待生成___                                             ← rollout优化这部分
```

## 预期效果

基于step165错误分析（~90%检测成功但推理错误），P2R训练应该能：

1. **减少跨类型关系错误**: 30% → 20%
2. **改进深度关系判断**: 系统性错误 → 部分正确
3. **提升计数准确性**: 多实例聚合错误减少
4. **整体准确率提升**:
   - HRBench4K: 81.8% → 85%+
   - HRBench8K: 76.1% → 80%+

同时保持：
- ✅ 工具调用能力不退化（70%标准样本保持训练）
- ✅ 首次检测成功率保持90%+
- ✅ KL散度保持<0.1（不崩溃）

## 文件清单

```
visual-agent/
├── PERCEIVE2REASON_README.md                          # 使用说明
├── perceive2reason_rl_design.md                       # 设计文档
├── scripts/
│   └── run_visual_agent_perceive2reason_2node_16gpu.sh  # 训练脚本
├── reinforcement_learning/
│   ├── data_preprocessing/
│   │   └── prepare_perceive2reason_data_from_train.py  # 数据准备
│   ├── verl/workers/rollout/sglang_rollout/
│   │   └── perceive2reason_rollout.py                 # P2R Rollout Worker
│   └── examples/sglang_multiturn/config/tool_config/
│       └── perceive2reason_config.yaml                # 配置文件
└── data/perceive2reason/                              # P2R数据目录
    ├── p2r_train.jsonl                                # 训练数据（待生成）
    └── p2r_train.summary.json                         # 统计摘要（待生成）
```

## 下一步建议

1. **最优先**: 实现`run_inference_on_train.py`以生成rollout traces
2. **其次**: 实现数据加载混合逻辑
3. **然后**: 小规模测试验证P2R逻辑正确性
4. **最后**: 完整训练并评测效果

或者，如果你想快速验证概念：
1. 手工构造几个P2R样本
2. 单步测试Perceive2ReasonRollout能否正确处理
3. 确认logit_bias确实禁止了工具调用
4. 再投入完整实现

代码框架已经完成，核心逻辑清晰，剩余工作主要是数据流和集成。
