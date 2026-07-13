# Distributed Visual-Tool SFT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the visual-tool production SFT path from LoRA to DeepSpeed ZeRO-3 full language-model training and provide one launcher that runs on 8, 16, or 32 Slurm-allocated GPUs while preserving global batch size 32.

**Architecture:** A small dependency-free Python utility renders runtime YAML overrides without mutating the checked-in base configuration. The generic Bash launcher validates topology, derives gradient accumulation, and delegates distributed process creation to LLaMA-Factory; a Slurm wrapper maps one Slurm task per node to the LLaMA-Factory multi-node environment.

**Tech Stack:** Bash, Python 3 standard library, pytest, Slurm, LLaMA-Factory, PyTorch torchrun, DeepSpeed ZeRO-3.

---

## File Map

- Create `scripts/prepare_visual_tool_sft_config.py`: render top-level YAML scalar overrides without requiring PyYAML.
- Modify `scripts/train_visual_tool_sft_full.sh`: validate topology and paths, render runtime configuration, support dry runs, and launch LLaMA-Factory distributed training.
- Create `scripts/run_visual_tool_sft.sh`: provide a user-editable, logged Huawei-platform single-node 8-GPU operator shell based on `temp.sh`.
- Create `scripts/slurm/train_visual_tool_sft.sbatch`: request eight GPUs per node and map Slurm node metadata into launcher variables.
- Modify `configs/train_visual_tool_sft_full.yaml`: select full SFT, frozen vision components, ZeRO-3, checkpointing, and full-SFT hyperparameters.
- Create `tests/test_distributed_visual_tool_sft.py`: test rendering, topology-derived batch accumulation, environment propagation, rejection behavior, and static Slurm configuration without GPUs.
- Modify `docs/visual_tool_sft.md`: document direct and Slurm usage, scaling, shared storage, dry-run, and cluster smoke validation.

### Task 1: Runtime YAML renderer

**Files:**
- Create: `scripts/prepare_visual_tool_sft_config.py`
- Create: `tests/test_distributed_visual_tool_sft.py`

- [ ] **Step 1: Write failing renderer tests**

Add tests that import `render_config` and verify replacement, insertion, quoting, and input validation:

```python
from scripts.prepare_visual_tool_sft_config import render_config


def test_render_config_replaces_and_appends_top_level_scalars():
    rendered = render_config(
        "model_name_or_path: old\ngradient_accumulation_steps: 4\n",
        {
            "model_name_or_path": "/models/Qwen VL",
            "gradient_accumulation_steps": 2,
            "max_steps": 5,
        },
    )
    assert 'model_name_or_path: "/models/Qwen VL"' in rendered
    assert "gradient_accumulation_steps: 2" in rendered
    assert "max_steps: 5" in rendered


@pytest.mark.parametrize("key", ["bad-key", "nested.key", " spaced"])
def test_render_config_rejects_invalid_top_level_key(key):
    with pytest.raises(ValueError, match="invalid override key"):
        render_config("stage: sft\n", {key: "value"})
```

- [ ] **Step 2: Run renderer tests and verify failure**

Run:

```bash
pytest tests/test_distributed_visual_tool_sft.py -v
```

Expected: collection fails because `scripts.prepare_visual_tool_sft_config` does not exist.

- [ ] **Step 3: Implement the renderer and CLI**

Create a renderer with these public interfaces:

```python
KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def format_yaml_scalar(value: str | int | float | bool | None) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def render_config(source: str, overrides: Mapping[str, Scalar]) -> str:
    for key in overrides:
        if not KEY_RE.fullmatch(key):
            raise ValueError(f"invalid override key: {key}")

    remaining = dict(overrides)
    rendered: list[str] = []
    for line in source.splitlines():
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):(?:\\s.*)?$", line)
        if match and match.group(1) in remaining:
            key = match.group(1)
            rendered.append(f"{key}: {format_yaml_scalar(remaining.pop(key))}")
        else:
            rendered.append(line)

    if remaining:
        rendered.append("")
        rendered.append("### runtime overrides")
        for key, value in remaining.items():
            rendered.append(f"{key}: {format_yaml_scalar(value)}")
    return "\n".join(rendered) + "\n"
```

The CLI accepts `--source`, `--destination`, repeated `--set KEY VALUE`, repeated `--set-int KEY VALUE`, and `--set-null KEY`. It reads the source with UTF-8, calls `render_config`, and writes the destination with UTF-8.

- [ ] **Step 4: Run renderer tests and verify success**

Run:

```bash
pytest tests/test_distributed_visual_tool_sft.py -v
```

Expected: renderer tests pass.

- [ ] **Step 5: Commit the renderer**

```bash
git add scripts/prepare_visual_tool_sft_config.py tests/test_distributed_visual_tool_sft.py
git commit -m "feat: add runtime sft config renderer"
```

### Task 2: Generic distributed launcher

**Files:**
- Modify: `scripts/train_visual_tool_sft_full.sh`
- Modify: `tests/test_distributed_visual_tool_sft.py`

- [ ] **Step 1: Write failing launcher tests**

Use a temporary executable as `LLAMAFACTORY_CLI`. The fake executable copies its second argument to `CAPTURE_CONFIG` and writes selected environment variables to `CAPTURE_ENV`:

```bash
#!/usr/bin/env bash
set -euo pipefail
cp "$2" "$CAPTURE_CONFIG"
env | sort > "$CAPTURE_ENV"
```

Parameterize the successful test with `(nnodes, expected_accumulation)` values `(1, 4)`, `(2, 2)`, and `(4, 1)`. Invoke the launcher with `NPROC_PER_NODE=8`, a temporary local model directory, a temporary output path, and multi-node rendezvous values. Assert that:

```python
assert result.returncode == 0
assert f"gradient_accumulation_steps: {expected_accumulation}" in captured_config
assert 'finetuning_type: "full"' in captured_config or "finetuning_type: full" in captured_config
assert "FORCE_TORCHRUN=1" in captured_env
assert f"NNODES={nnodes}" in captured_env
assert "NPROC_PER_NODE=8" in captured_env
```

Add a rejection test with `NNODES=3`, `NPROC_PER_NODE=8`, and `TARGET_GLOBAL_BATCH_SIZE=32`; assert nonzero exit and `does not divide TARGET_GLOBAL_BATCH_SIZE` in stderr.

Add a dry-run test with `DRY_RUN=1` and a fake CLI that exits 99; assert exit zero, the output contains `world_size=8` and `gradient_accumulation_steps: 4`, and no capture file was created.

- [ ] **Step 2: Run launcher tests and verify failure**

Run:

```bash
pytest tests/test_distributed_visual_tool_sft.py -k launcher -v
```

Expected: failures because the current launcher does not derive topology or render runtime overrides.

- [ ] **Step 3: Implement topology and path validation**

In `scripts/train_visual_tool_sft_full.sh`, define defaults and validate positive integers:

```bash
NNODES="${NNODES:-${SLURM_NNODES:-1}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NODE_RANK="${NODE_RANK:-${SLURM_PROCID:-0}}"
MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT="${MASTER_PORT:-29500}"
TARGET_GLOBAL_BATCH_SIZE="${TARGET_GLOBAL_BATCH_SIZE:-32}"
DRY_RUN="${DRY_RUN:-0}"

require_uint() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "error: $name must be a non-negative integer, got: $value" >&2
    exit 2
  fi
}
```

Require nodes, processes per node, master port, and global batch to be positive; allow node rank zero. Set `MASTER_ADDR=127.0.0.1` for one node and require it for multiple nodes. Require `NODE_RANK < NNODES`.

Calculate and validate:

```bash
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
if (( TARGET_GLOBAL_BATCH_SIZE % WORLD_SIZE != 0 )); then
  echo "error: world size $WORLD_SIZE does not divide TARGET_GLOBAL_BATCH_SIZE=$TARGET_GLOBAL_BATCH_SIZE" >&2
  exit 2
fi
GRADIENT_ACCUMULATION_STEPS=$((TARGET_GLOBAL_BATCH_SIZE / WORLD_SIZE))
```

Resolve `LLAMAFACTORY_CLI` using the existing PATH and `.venv-llamafactory` behavior. Validate `CONFIG_PATH`, `DEEPSPEED_CONFIG`, the renderer, the CLI, and an explicitly supplied local `MODEL_NAME_OR_PATH`.

- [ ] **Step 4: Render overrides and launch LLaMA-Factory**

Create a temporary YAML with `mktemp`, register a cleanup trap, and call the renderer with these always-on overrides:

```bash
--set-int gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
--set deepspeed "$DEEPSPEED_CONFIG"
```

Conditionally add `model_name_or_path`, `output_dir`, `resume_from_checkpoint`, and integer `max_steps` overrides. Export the exact distributed variables expected by LLaMA-Factory:

```bash
export FORCE_TORCHRUN=1
export NNODES NPROC_PER_NODE NODE_RANK MASTER_ADDR MASTER_PORT
```

For `DRY_RUN=1`, print the topology, shell-escaped command, and complete rendered YAML, then exit zero. Otherwise execute:

```bash
"$LLAMAFACTORY_CLI" train "$RUNTIME_CONFIG"
```

- [ ] **Step 5: Run launcher tests and shell syntax checks**

Run:

```bash
pytest tests/test_distributed_visual_tool_sft.py -k launcher -v
bash -n scripts/train_visual_tool_sft_full.sh
```

Expected: all selected tests pass and `bash -n` exits zero.

- [ ] **Step 6: Commit the launcher**

```bash
git add scripts/train_visual_tool_sft_full.sh tests/test_distributed_visual_tool_sft.py
git commit -m "feat: launch visual tool sft across gpu nodes"
```

### Task 3: Full-SFT production configuration

**Files:**
- Modify: `configs/train_visual_tool_sft_full.yaml`
- Modify: `tests/test_distributed_visual_tool_sft.py`

- [ ] **Step 1: Write a failing static configuration test**

Read the YAML as text and extract top-level scalar values with a small test helper. Assert:

```python
assert values["finetuning_type"] == "full"
assert "lora_target" not in values
assert values["freeze_vision_tower"] == "true"
assert values["freeze_multi_modal_projector"] == "true"
assert values["freeze_language_model"] == "false"
assert values["deepspeed"] == "cold_start/examples/deepspeed/ds_z3_config.json"
assert values["disable_gradient_checkpointing"] == "false"
assert values["per_device_train_batch_size"] == "1"
assert values["gradient_accumulation_steps"] == "4"
assert values["learning_rate"] == "1.0e-5"
assert values["output_dir"] == "saves/visual_tool_sft_v0/full/sft"
```

- [ ] **Step 2: Run the static test and verify failure**

Run:

```bash
pytest tests/test_distributed_visual_tool_sft.py -k full_sft_config -v
```

Expected: failure because the checked-in config still selects LoRA.

- [ ] **Step 3: Update the production YAML**

Use these method and training values:

```yaml
### method
stage: sft
do_train: true
finetuning_type: full
freeze_vision_tower: true
freeze_multi_modal_projector: true
freeze_language_model: false
deepspeed: cold_start/examples/deepspeed/ds_z3_config.json
disable_gradient_checkpointing: false

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
learning_rate: 1.0e-5
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null
```

Change `output_dir` to `saves/visual_tool_sft_v0/full/sft`. Retain the current model, image limits, dataset, cutoff length, workers, logging, and checkpoint retention values.

- [ ] **Step 4: Run static and launcher tests**

Run:

```bash
pytest tests/test_distributed_visual_tool_sft.py -v
```

Expected: all distributed SFT tests pass.

- [ ] **Step 5: Commit the full-SFT configuration**

```bash
git add configs/train_visual_tool_sft_full.yaml tests/test_distributed_visual_tool_sft.py
git commit -m "feat: switch visual tool training to full sft"
```

### Task 4: Single-node operator shell

**Files:**
- Create: `scripts/run_visual_tool_sft.sh`
- Modify: `tests/test_distributed_visual_tool_sft.py`

- [ ] **Step 1: Write failing operator-shell tests**

Read the script as text and assert it contains the operational controls needed from the supplied reference without copying unrelated or destructive behavior:

```python
assert 'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"' in script
assert 'NPROC_PER_NODE="${NPROC_PER_NODE:-8}"' in script
assert 'TARGET_GLOBAL_BATCH_SIZE="${TARGET_GLOBAL_BATCH_SIZE:-32}"' in script
assert 'CONDA_ENV_PATH="${CONDA_ENV_PATH:-}"' in script
assert 'PLATFORM_WORK_ROOT="${PLATFORM_WORK_ROOT:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx}"' in script
assert 'PLATFORM_SETUP="${PLATFORM_SETUP:-1}"' in script
assert "/opt/huawei/explorer-env/dataset/trellis_ckpt/cuda/cuda118" in script
assert "/opt/huawei/explorer-env/dataset/Common_wl/miniconda3" in script
assert "/opt/huawei/quoteModel/xiaoyi_tmpstorage" in script
assert 'OFFLINE_MODE="${OFFLINE_MODE:-0}"' in script
assert "NCCL_DEBUG" in script
assert "nvidia-smi" in script
assert "train_visual_tool_sft_full.sh" in script
assert "rm -rf" not in script
assert "synaflow_wl" not in script
```

- [ ] **Step 2: Run the operator-shell test and verify failure**

Run:

```bash
pytest tests/test_distributed_visual_tool_sft.py -k operator_shell -v
```

Expected: failure because `scripts/run_visual_tool_sft.sh` does not exist.

- [ ] **Step 3: Implement the safe operator shell**

Create an executable Bash script that resolves the repository root and defines:

```bash
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TARGET_GLOBAL_BATCH_SIZE="${TARGET_GLOBAL_BATCH_SIZE:-32}"
MASTER_PORT="${MASTER_PORT:-29500}"
PLATFORM_SETUP="${PLATFORM_SETUP:-1}"
PLATFORM_WORK_ROOT="${PLATFORM_WORK_ROOT:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx}"
PLATFORM_DATASET_ROOT="${PLATFORM_DATASET_ROOT:-/opt/huawei/dataset}"
PLATFORM_MODEL_ROOT="${PLATFORM_MODEL_ROOT:-/opt/huawei/quoteModel/xiaoyi_tmpstorage}"
CUDA_HOME="${CUDA_HOME:-/opt/huawei/explorer-env/dataset/trellis_ckpt/cuda/cuda118}"
MINICONDA_ROOT="${MINICONDA_ROOT:-/opt/huawei/explorer-env/dataset/Common_wl/miniconda3}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-}"
CONDA_SH="${CONDA_SH:-}"
OFFLINE_MODE="${OFFLINE_MODE:-0}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/visual-tool-sft}"
PRINT_HARDWARE_INFO="${PRINT_HARDWARE_INFO:-1}"
```

When `PLATFORM_SETUP=1`, create the three platform parent directories when permitted and create only missing symlinks for the dataset and persistent model storage. Never replace an existing file, directory, or symlink. Require `CUDA_HOME`, prepend its `bin` and `lib64` directories, and print `nvcc --version` when available.

When `CONDA_ENV_PATH` is non-empty, require a usable `CONDA_SH`, source it, and run `conda activate "$CONDA_ENV_PATH"`. If `CONDA_SH` is empty, check `$MINICONDA_ROOT/etc/profile.d/conda.sh`, `$HOME/miniconda3/etc/profile.d/conda.sh`, and `$HOME/anaconda3/etc/profile.d/conda.sh` in order.

When `OFFLINE_MODE=1`, export:

```bash
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1
```

Export `TOKENIZERS_PARALLELISM=false`, `NCCL_DEBUG=${NCCL_DEBUG:-WARN}`, and `NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}`. Create a timestamped per-rank log, redirect stdout and stderr through `tee`, print resolved paths/topology/batch settings, optionally run `nvidia-smi`, and execute the generic launcher. Do not delete or overwrite output directories in this wrapper.

- [ ] **Step 4: Run operator tests and shell syntax validation**

Run:

```bash
pytest tests/test_distributed_visual_tool_sft.py -k operator_shell -v
bash -n scripts/run_visual_tool_sft.sh
```

Expected: the selected tests pass and Bash reports no syntax errors.

- [ ] **Step 5: Commit the operator shell**

```bash
git add scripts/run_visual_tool_sft.sh tests/test_distributed_visual_tool_sft.py
git commit -m "feat: add eight gpu sft operator script"
```

### Task 5: Slurm wrapper

**Files:**
- Create: `scripts/slurm/train_visual_tool_sft.sbatch`
- Modify: `tests/test_distributed_visual_tool_sft.py`

- [ ] **Step 1: Write failing Slurm wrapper tests**

Read the wrapper as text and assert it contains:

```python
assert "#SBATCH --nodes=1" in script
assert "#SBATCH --ntasks-per-node=1" in script
assert "#SBATCH --gres=gpu:8" in script
assert 'MASTER_ADDR="$(scontrol show hostnames' in script
assert 'NODE_RANK="${SLURM_PROCID}"' in script
assert "--ntasks-per-node=1" in script
assert "train_visual_tool_sft_full.sh" in script
```

- [ ] **Step 2: Run the Slurm test and verify failure**

Run:

```bash
pytest tests/test_distributed_visual_tool_sft.py -k slurm -v
```

Expected: failure because the wrapper does not exist.

- [ ] **Step 3: Implement the Slurm wrapper**

Create an executable Bash script with these allocation defaults:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=visual-tool-sft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=logs/visual-tool-sft-%j.out
#SBATCH --error=logs/visual-tool-sft-%j.err
```

Require `SLURM_JOB_ID`, `SLURM_NNODES`, and `SLURM_JOB_NODELIST`. Resolve and export:

```bash
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-29500}"
NNODES="$SLURM_NNODES"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
```

Create the log directory before `srun`. Export the variables and launch one task per node:

```bash
srun --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 --kill-on-bad-exit=1 \
  bash -c 'export NODE_RANK="${SLURM_PROCID}"; exec bash "${REPO_ROOT}/scripts/train_visual_tool_sft_full.sh"'
```

- [ ] **Step 4: Run Slurm tests and syntax validation**

Run:

```bash
pytest tests/test_distributed_visual_tool_sft.py -k slurm -v
bash -n scripts/slurm/train_visual_tool_sft.sbatch
```

Expected: the selected tests pass and Bash reports no syntax errors.

- [ ] **Step 5: Commit the Slurm wrapper**

```bash
git add scripts/slurm/train_visual_tool_sft.sbatch tests/test_distributed_visual_tool_sft.py
git commit -m "feat: add slurm launcher for visual tool sft"
```

### Task 6: Operator documentation and final verification

**Files:**
- Modify: `docs/visual_tool_sft.md`

- [ ] **Step 1: Document the production strategy**

Add a distributed full-SFT section stating that the language model is fully trained while the vision tower and multimodal projector remain frozen. Record BF16, ZeRO-3, per-device batch size 1, and global batch size 32.

- [ ] **Step 2: Document direct and Slurm commands**

Include these exact examples:

```bash
# Inside an existing one-node, eight-GPU allocation
bash scripts/run_visual_tool_sft.sh

# Inspect without starting workers
DRY_RUN=1 NNODES=1 NPROC_PER_NODE=8 \
  bash scripts/train_visual_tool_sft_full.sh

# Default 1 x 8 GPUs
sbatch scripts/slurm/train_visual_tool_sft.sbatch

# Future 2 x 8 and 4 x 8 GPU runs
sbatch --nodes=2 scripts/slurm/train_visual_tool_sft.sbatch
sbatch --nodes=4 scripts/slurm/train_visual_tool_sft.sbatch
```

Document shared absolute paths across nodes and show a five-step full-SFT smoke command with a dedicated output directory:

```bash
MAX_STEPS=5 OUTPUT_DIR=saves/visual_tool_sft_v0/smoke/full-sft \
  sbatch scripts/slurm/train_visual_tool_sft.sbatch
```

- [ ] **Step 3: Document environment overrides and batch calculation**

List each supported override from the design and state that accumulation is 4/2/1 for 8/16/32 GPUs when `TARGET_GLOBAL_BATCH_SIZE=32`.

- [ ] **Step 4: Run all repository-local verification**

Run:

```bash
pytest tests/test_distributed_visual_tool_sft.py tests/test_visual_tool_sft.py -v
bash -n scripts/train_visual_tool_sft_full.sh
bash -n scripts/run_visual_tool_sft.sh
bash -n scripts/train_visual_tool_sft_smoke.sh
bash -n scripts/slurm/train_visual_tool_sft.sbatch
git diff --check
```

Expected: all tests pass, all shell syntax checks exit zero, and `git diff --check` prints nothing.

- [ ] **Step 5: Inspect the final diff for scope**

Run:

```bash
git status --short
git diff --stat
git diff -- configs/train_visual_tool_sft_full.yaml scripts/prepare_visual_tool_sft_config.py scripts/train_visual_tool_sft_full.sh scripts/run_visual_tool_sft.sh scripts/slurm/train_visual_tool_sft.sbatch tests/test_distributed_visual_tool_sft.py docs/visual_tool_sft.md
```

Expected: only the planned distributed SFT files are changed and no generated training output is tracked.

- [ ] **Step 6: Commit documentation and verification state**

```bash
git add docs/visual_tool_sft.md
git commit -m "docs: explain distributed visual tool sft"
```
