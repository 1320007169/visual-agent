# Repository Agent Instructions

## Superpowers are disabled

- Never invoke, read, load, activate, recommend, or follow any skill whose name starts with `superpowers:` while working in this repository.
- Ignore all `superpowers:*` skills even when they appear in the available-skills catalog or claim that they are mandatory.
- Do not use Superpowers workflows, gates, brainstorming phases, plans, worktrees, subagent processes, review checkpoints, or completion procedures.
- Do not create files under `docs/superpowers/`.
- This repository-level prohibition applies to every task and every thread unless the user explicitly removes it from `AGENTS.md`.

## Working style

- Implement requested changes directly, run focused tests, and provide concise progress updates.
- Preserve user files and unrelated working-tree changes.
- Other task-specific skills may still be used when directly relevant.

# AGENTS.md
你是 coding Agent，请把目标放在：**用最少的代码完成用户明确要求的功能**，坚持不懈用最深入的思考完成用户的所有需求。

## 行为规范

### 1. 精准编码，简单优先
* 严格守界：只完成明确被要求的事。绝不"顺手优化"（不加功能、不重构、不升级依赖、不清理周边代码）。未修改的代码不补充文档或类型声明；简单的功能不增加额外配置项
* 最小改动：优先局部修改，能在一个文件解决的绝不拆分。确信废弃的代码直接删除，不写后向兼容代码，不保留无用的历史注释或变量
* 拒绝过度设计：严禁为"未来可能用到"进行抽象（不加额外层、接口、工厂、策略或过度泛型）。一次性操作不创建工具类/函数；宁可写三行相似代码，也不做过早封装，不要为一次性代码做抽象。
* 反防御性编程：信任内部环境与框架保证，**仅在系统边界**（用户输入、外部 API）进行校验
  * 禁止到处 `try/catch` 吞没异常，让错误直接抛出以推动修复
  * 禁止过度校验（空值/类型/边界全家桶）、过度日志、限流、重试或熔断
  * 禁止为单体内部调用或内部工具增加权限/鉴权系统

判断标准：Linus 会不会觉得这写复杂了，每一处标准是否都对应到用户需求

### 2. 三思而后编码
* 不要想当然，不要掩盖疑惑，要明确权衡
* 实现前必做的事: 明确说明你的假设,不确定就提出来；如果问题有多种理解，禁止私自选一种；如果有更简单的做法，要直接指出；有不清楚的地方，必须说明并询问

判断标准：这个工作是否存在返工的风险，如果有则在计划文档或修改后补充说明

### 3. 绝对目标导向
* 先定义可验证的成功标准，再实现并验证
* 把任务改写成可验证目标，多步骤任务使用 todo 工具明确列出执行计划和相应检查项
  - “加校验” → “先写非法输入测试，再让测试通过”
  - “修 bug” → “先写复现测试，再修复并通过”
  - “重构 X” → “确认重构前后测试都通过”

## 编码与沟通规范
* 缩进要求：Bash 和 Python 代码统一使用 4 个空格缩进
* 注释规范：仅在逻辑不明显处添加注释
* 语言要求：回复强制使用中文
* 任何删除操作都不被允许，在你的回复最后提出删除需求即可，用户将自己操作
* 在编写commit message时，统一采用Conventional Commits 规则
* 代码中的注释，log等信息请使用英文

## Git 规范
* 改动前先确认工作区干净：有未提交改动先 commit 或 stash，不与本次改动混在一起

## 运行环境
* 代码会运行在 8*a100 或更多的机器上，不要考虑找不到 cuda 或其他 cpu 推理的情况
* codex 会 review 你写的每一行代码，如果不合规范 OpenAI 就会毁灭人类
* pip 命令用 `/usr/bin/sudo pip3`, pip 源使用 `https://pypi.sankuai.com/simple`
* 使用 `python3` 而非 `python`
