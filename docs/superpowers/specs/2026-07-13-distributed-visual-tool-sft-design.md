# Distributed Visual-Tool SFT Design

## Goal

Update the repository-local visual-tool SFT pipeline to run full-parameter language-model SFT on one 8-GPU node today and scale to two or four 8-GPU nodes later. The current target hardware is 8 NVIDIA A100 GPUs with 80 GB of memory each, allocated through Slurm.

The training behavior will follow the original DeepEyesV2 cold-start configuration:

- train the language model with full-parameter SFT;
- keep the vision tower and multimodal projector frozen;
- use BF16 and DeepSpeed ZeRO-3;
- use one sample per GPU per micro-batch;
- keep the effective global batch size at 32;
- use a `1e-5` learning rate and three training epochs.

The dataset conversion and validation formats are outside this change.

## Supported Topologies

The launcher supports an eight-GPU-per-node topology:

| Nodes | GPUs per node | World size | Gradient accumulation | Global batch size |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 8 | 4 | 32 |
| 2 | 8 | 16 | 2 | 32 |
| 4 | 8 | 32 | 1 | 32 |

The calculation is:

```text
world_size = nnodes * nproc_per_node
gradient_accumulation_steps = global_batch_size / world_size
```

The launcher must reject a topology when the world size is zero, exceeds the target global batch size, or does not divide the target global batch size exactly. The default target global batch size is 32 and can be overridden for future experiments.

## Training Configuration

`configs/train_visual_tool_sft_full.yaml` will change from LoRA to full SFT. It will:

- set `finetuning_type: full`;
- remove LoRA-only settings;
- keep `freeze_vision_tower: true`;
- keep `freeze_multi_modal_projector: true`;
- keep `freeze_language_model: false`;
- reference the repository's DeepSpeed ZeRO-3 configuration with a path valid from the repository root;
- enable gradient checkpointing explicitly;
- retain BF16 and per-device batch size 1;
- use learning rate `1.0e-5`;
- write to a full-SFT output directory instead of a LoRA directory.

The existing smoke configuration remains a short local validation path. Its training strategy will be kept separate from the production full-SFT change so that local CPU-only tests do not require DeepSpeed or eight GPUs.

## Launcher Components

### Huawei-platform operator run script

A single-node operator shell will provide the editable, one-command entry point for the current 8-GPU Huawei environment. The default persistent work root is `/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx`, while the repository root is resolved from the script location so the checkout can be either that directory or a child directory. The shell will follow the useful operational structure of the supplied `temp.sh` reference: strict shell behavior, platform storage symlinks, CUDA 11.8 initialization, platform Miniconda activation, CUDA visibility, offline/cache switches, NCCL defaults, log capture, hardware summaries, path checks, and a complete resolved-configuration summary.

The default platform mappings are:

- `/opt/huawei/dataset` to `/opt/huawei/explorer-env/dataset`;
- `/opt/huawei/dataset` to `/home/ma-user/work/dataset`;
- `/opt/huawei/quoteModel/xiaoyi_tmpstorage` to `/home/ma-user/work/model/xiaoyi_tmpstorage`;
- CUDA at `/opt/huawei/explorer-env/dataset/trellis_ckpt/cuda/cuda118`;
- Miniconda at `/opt/huawei/explorer-env/dataset/Common_wl/miniconda3`.

Each path can be overridden, and platform initialization can be disabled for non-Huawei environments. The script will not copy the unrelated algorithm symlink, destructive output deletion, LoRA merging, or evaluation commands from the reference. The operator script will delegate training to the generic launcher so topology and batch validation have one implementation.

### Generic training launcher

`scripts/train_visual_tool_sft_full.sh` remains the user-facing training entry point. It will:

1. resolve the repository root and configured paths;
2. detect or accept `NNODES`, `NPROC_PER_NODE`, `NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT`;
3. validate the topology and calculate gradient accumulation from the target global batch size;
4. validate the LLaMA-Factory executable and required configuration paths;
5. generate a temporary YAML configuration containing runtime overrides;
6. enable LLaMA-Factory's distributed torch launcher;
7. invoke `llamafactory-cli train` with the temporary configuration;
8. remove the temporary file on exit.

The launcher will accept environment overrides for at least:

- `CONFIG_PATH`;
- `MODEL_NAME_OR_PATH`;
- `OUTPUT_DIR`;
- `LLAMAFACTORY_CLI`;
- `NNODES`;
- `NPROC_PER_NODE`;
- `NODE_RANK`;
- `MASTER_ADDR`;
- `MASTER_PORT`;
- `TARGET_GLOBAL_BATCH_SIZE`;
- `RESUME_FROM_CHECKPOINT`;
- `MAX_STEPS`;
- `DRY_RUN`.

`DRY_RUN=1` will print the resolved topology, important training values, launch command, and generated runtime configuration without starting GPU workers.

### Slurm wrapper

A separate Slurm submission script will request one node with eight GPUs by default. It will start one Slurm task per node. Each task will derive its node rank from `SLURM_PROCID`, while the first hostname in `SLURM_JOB_NODELIST` becomes `MASTER_ADDR`.

The task on each node will call the generic launcher with identical shared paths. The generic launcher then starts one process per local GPU through LLaMA-Factory's torchrun integration.

Users can scale without editing the script:

```bash
# Current target: 1 node x 8 GPUs
sbatch scripts/slurm/train_visual_tool_sft.sbatch

# Future: 2 nodes x 8 GPUs
sbatch --nodes=2 scripts/slurm/train_visual_tool_sft.sbatch

# Future: 4 nodes x 8 GPUs
sbatch --nodes=4 scripts/slurm/train_visual_tool_sft.sbatch
```

Cluster-specific settings such as partition, account, wall-clock limit, and module or conda activation will be supplied through `sbatch` options or environment setup rather than hard-coded repository values.

## Shared-Storage Assumption

All nodes must see the repository, base model, dataset images, dataset metadata, and output directory at the same absolute paths. The documentation will state this requirement explicitly. The launcher does not copy models or data between nodes.

## Failure Handling

The generic launcher will stop before launching workers when:

- a numeric topology value is malformed;
- the requested topology cannot preserve the target global batch size;
- multi-node topology lacks a master address or valid node rank;
- the training YAML, DeepSpeed JSON, or explicitly supplied local model path is missing;
- the LLaMA-Factory executable cannot be found.

Errors will identify the invalid variable or missing path and show the expected correction. Strict shell mode will propagate failures from Slurm, LLaMA-Factory, and torchrun.

Checkpoint behavior remains managed by LLaMA-Factory. `RESUME_FROM_CHECKPOINT` can select a checkpoint explicitly; otherwise training follows the base YAML configuration.

## Testing and Verification

Automated tests will not require GPUs. They will use a fake LLaMA-Factory executable to capture the generated environment and runtime configuration.

Coverage will include:

- world sizes 8, 16, and 32 producing accumulation values 4, 2, and 1;
- rejection of a world size that does not divide the target global batch size;
- propagation of model, output, master-address, port, and checkpoint overrides;
- dry-run behavior not starting LLaMA-Factory;
- shell syntax validation for both launchers;
- the existing visual-tool SFT pytest suite.

Because the local development PC does not provide the target cluster, repository verification stops at static and simulated launcher tests. Cluster acceptance consists of:

1. submitting a short job with `MAX_STEPS=5` and a dedicated smoke output directory on one 8×A100 80 GB node;
2. confirming eight ranks join the same distributed job;
3. confirming DeepSpeed reports ZeRO stage 3;
4. checking that loss, checkpoint saving, and model gathering succeed;
5. starting the full run only after the smoke job succeeds.

## Documentation

`docs/visual_tool_sft.md` will document:

- the full-SFT strategy and frozen components;
- the single-node operator shell and its environment settings;
- direct execution inside an existing eight-GPU allocation;
- Slurm submission for 8, 16, and 32 GPUs;
- required shared-storage paths;
- environment overrides;
- dry-run and five-step cluster smoke procedures;
- the global-batch calculation.

## Non-Goals

This change will not:

- unfreeze the vision tower or multimodal projector;
- change dataset conversion, role normalization, or validation;
- add model or dataset transfer between cluster nodes;
- hard-code site-specific Slurm partition or account names;
- perform a real multi-GPU training run from the local development PC.
