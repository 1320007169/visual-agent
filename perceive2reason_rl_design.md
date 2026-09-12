# Perceive2Reason风格RL训练方案设计

## 背景与动机

### 当前Step165错误分析
基于step165 checkpoint的评测结果分析：

**总体表现**：
- VStarBench: 89.5% (171/191)
- HRBench4K: 81.8% (654/800)
- HRBench8K: 76.1% (609/773)

**关键发现**：
1. **~90%的错误样本检测成功但推理错误**
   - VStarBench: 70%错误有检测框但答案错
   - HRBench4K: 96%错误有检测框但答案错
   - HRBench8K: 92%错误有检测框但答案错

2. **检测失败占比极低** (2-10%)
   - 工具调用本身已经稳定，grounding_detect返回非空框的成功率很高

3. **跨类型关系任务错误率显著高于单类型**
   - HRBench4K: cross_type 30% vs single_type 6%
   - HRBench8K: cross_type 33% vs single_type 10%

4. **典型错误模式**
   - 深度关系系统性失败："in front of/behind" 被错答为 "left of/right of"
   - 模型只用2D框中心比较，不crop检查遮挡/深度
   - 多实例计数错误：检测11个框但回答"Three"

### Perceive2Reason方法论
Perceive2Reason的核心思想：
- **阶段分离**：将多模态推理分为感知(perception)和推理(reasoning)两阶段
- **前缀固定**：对于感知正确的样本，固定其工具调用序列（perception prefix）
- **后缀优化**：只对最终推理部分（reasoning suffix）进行RL优化
- **优势**：减少搜索空间，避免破坏已经正确的感知能力，专注改进推理逻辑

## 方案设计

### 核心思路
对于**检测成功但答案错误**的样本：
1. 识别正确的工具调用前缀（已返回有效检测框的grounding_detect + crop_zoom序列）
2. 在rollout时固定这部分prefix，禁止模型重新调用工具
3. 只允许模型在工具调用结束后生成最终答案，对这部分进行采样和优化
4. 通过reward信号（正确答案 vs 错误答案）优化推理能力

### 实现路径

#### 1. 数据准备阶段

**重要**：必须使用训练集，不能用评测集！

**1.1 对训练集跑推理获取rollout traces**
```bash
# 用step165 checkpoint对训练数据跑一遍inference
# 保存每个样本的tool_calls + prediction
python run_inference_on_train.py \
  --checkpoint step165 \
  --data /path/to/zwz_train.jsonl \
  --output train_rollout/
```

**1.2 从训练集rollout中筛选样本**
```python
# 从训练数据的rollout结果中提取
criteria = {
    "has_tool_calls": True,          # 调用了工具
    "no_tool_failure": True,         # 工具没失败
    "first_detection_success": True, # 首次检测返回非空框
    "final_answer_wrong": True       # 但最终答案错误（对比GT）
}
# 预计可得若干百个样本（取决于训练集大小和模型准确率）
```

**1.2 构建perception prefix数据集**
对每个符合条件的样本：
```python
sample = {
    "uid": unique_id,
    "image": image_path,
    "question": question_text,
    "gt_answer": ground_truth_answer,

    # 固定的perception prefix（从评测trace中提取）
    "perception_prefix": [
        {
            "role": "assistant",
            "content": "<tool_call>{...grounding_detect...}</tool_call>"
        },
        {
            "role": "tool",
            "content": "<tool_response>{boxes: [[x1,y1,x2,y2], ...]}</tool_response>"
        },
        # 可能有多轮工具调用
        ...
    ],

    # 需要优化的reasoning suffix起点
    "reasoning_start_turn": len(perception_prefix),

    # 元数据
    "original_wrong_answer": model_prediction,
    "error_type": "depth_relation" | "counting" | "cross_type",
}
```

#### 2. Rollout修改

**2.1 修改采样策略**
在 `verl/trainer/ppo/ray_trainer.py` 的rollout worker中：

```python
# 当前标准rollout流程
for sample in batch:
    prompt = build_prompt(sample)  # system + user question + image
    response = model.generate(prompt)  # 完全自由生成

# ==================== 修改为 ====================

for sample in batch:
    if sample.get("perception_prefix"):  # perceive2reason模式
        # 构建固定前缀
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image", "image": sample["image"]},
                {"type": "text", "text": sample["question"]}
            ]},
        ]
        # 添加固定的工具调用历史（不采样，直接复制）
        messages.extend(sample["perception_prefix"])

        # 添加reasoning trigger
        messages.append({
            "role": "assistant",
            "content": ""  # 模型从这里开始自由生成最终答案
        })

        # 只对最终答案部分采样
        response = model.generate(
            messages,
            max_new_tokens=256,  # 答案通常很短
            stop=["</answer>"],
            # 关键：不允许再生成<tool_call>
            # 通过logit bias或constrained decoding实现
        )
    else:
        # 标准流程：完全自由rollout
        response = model.generate(build_prompt(sample))
```

**2.2 禁止工具调用的实现方式**

方案A：Logit Bias（推荐）
```python
# 在vLLM sampling params中
sampling_params = SamplingParams(
    logit_bias={
        token_id("<tool_call>"): -100,  # 强制抑制工具调用token
    }
)
```

方案B：Constrained Decoding
```python
# 使用正则约束，只允许生成答案
allowed_pattern = r"^[^<]*<answer>.*</answer>$"
```

方案C：Early Stopping
```python
# 生成过程中检测到<tool_call>立即停止并惩罚
if "<tool_call>" in generated_text:
    reward = -1.0  # 违规惩罚
    break
```

#### 3. Reward Shaping

**3.1 标准reward（保持不变）**
```python
reward = {
    "correct_answer": 1.0,
    "wrong_answer": 0.0,
}
```

**3.2 额外约束（可选）**
```python
# 惩罚尝试调用工具的行为
if "<tool_call>" in response:
    reward -= 0.5

# 奖励简洁答案（避免冗长推理）
if len(response.split()) < 50:
    reward += 0.1
```

#### 4. 训练配置

**4.1 混合数据集策略**

选项A：纯perceive2reason训练
```python
train_data = perceive2reason_samples  # ~300个样本
epochs = 20  # 多轮迭代
```

选项B：混合训练（推荐）
```python
train_data = {
    "perceive2reason": perceive2reason_samples,  # 30%，专注推理
    "standard_rollout": zwz_original_samples,    # 70%，保持工具调用能力
}
# 防止灾难性遗忘
```

**4.2 超参数**
```bash
# 基于现有zwz_groundingdino配置
ACTOR_LR=1e-6                    # 保持当前学习率
ACTOR_KL_LOSS_COEF=0.001         # 保持KL约束
ACTOR_USE_KL_LOSS=True

# perceive2reason特定
PERCEIVE2REASON_MIX_RATIO=0.3    # 30%样本使用固定prefix
REASONING_MAX_TOKENS=256         # 答案生成长度限制
EPOCHS=10                         # 迭代轮数
```

#### 5. 代码实现位置

**5.1 数据准备脚本**
```bash
# 新建脚本
visual-agent/reinforcement_learning/data_preprocessing/prepare_perceive2reason_data.py

# 功能：
# - 读取step165评测JSONL
# - 筛选符合条件样本
# - 提取perception prefix
# - 生成训练数据集
```

**5.2 Rollout修改**
```bash
# 修改位置
visual-agent/reinforcement_learning/verl/trainer/ppo/ray_trainer.py
  -> class RolloutManager
    -> def rollout_step()

# 新增配置
visual-agent/reinforcement_learning/examples/sglang_multiturn/config/perceive2reason_config.yaml
```

**5.3 训练脚本**
```bash
# 复制并修改现有脚本
cp run_visual_agent_zwz_rl_groundingdino_2node_16gpu.sh \
   run_visual_agent_perceive2reason_2node_16gpu.sh

# 修改点：
# - 数据集路径指向perceive2reason数据
# - 添加PERCEIVE2REASON_MODE=True环境变量
# - 调整训练轮数和checkpoint频率
```

## 预期效果

### 目标改进
1. **跨类型关系准确率**：30% → 20% error rate (提升10个百分点)
2. **深度关系**：系统性错误 → 50%以上正确
3. **整体准确率**：
   - HRBench4K: 81.8% → 85%+
   - HRBench8K: 76.1% → 80%+

### 风险与缓解

**风险1：灾难性遗忘**
- 模型可能丧失工具调用能力
- 缓解：混合训练，保留70%标准rollout样本

**风险2：过拟合**
- 300个样本数据量较小
- 缓解：KL约束，多样化采样，early stopping

**风险3：prefix质量**
- 并非所有"检测成功"的prefix都是最优的
- 缓解：人工抽查+自动过滤明显不合理的检测框

## 实验计划

### Phase 1：数据准备与验证 (1天)
1. 实现数据筛选脚本
2. 生成perceive2reason数据集
3. 人工抽查50个样本，验证prefix质量

### Phase 2：单机验证 (1天)
1. 在单GPU上实现rollout修改
2. 测试logit bias禁止工具调用
3. 跑1-2个epoch验证流程正确性

### Phase 3：完整训练 (2天)
1. 2-node 16-GPU训练10 epochs
2. 每2 epoch保存checkpoint
3. 在3个benchmark上评测每个checkpoint

### Phase 4：分析与迭代 (1天)
1. 对比step165 baseline
2. 错误归因分析：哪些错误被修正，是否引入新错误
3. 根据结果调整mix ratio或reward shaping

## 关键问题与讨论

### Q1: 是否需要为不同错误类型设计不同的prefix？
- 深度关系错误：可能需要在prefix中包含crop操作
- 计数错误：prefix应该包含所有相关对象的检测
- 建议：初版统一处理，后续可按错误类型细分

### Q2: Perception prefix的粒度如何确定？
- 粗粒度：只保留首次检测调用
- 细粒度：保留所有工具调用直到最后一个</tool_response>
- 建议：细粒度（保留完整工具轨迹），因为多轮交互本身是正确的

### Q3: 是否需要重新标注GT reasoning？
- 当前只有GT answer，没有GT reasoning过程
- Perceive2reason方法不需要reasoning annotation
- Reward完全基于最终答案匹配即可

### Q4: 如何处理有多个合理推理路径的情况？
- 例如：既可以crop再判断，也可以直接从框判断
- 当前方案：接受任意正确答案，不强制特定推理路径
- RL的exploration会自然发现有效路径

## 参考资料

- Perceive2Reason原文（如有）
- VLMEval错误分析报告：`step165_error_analysis.md`
- 当前RL训练配置：`run_visual_agent_zwz_rl_groundingdino_2node_16gpu.sh`
- GRPO实现：`verl/trainer/ppo/`
