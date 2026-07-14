import os
import subprocess
from pathlib import Path

import pytest

from scripts.prepare_visual_tool_sft_config import render_config


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "train_visual_tool_sft_full.sh"


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
def test_render_config_rejects_invalid_top_level_key(key: str):
    with pytest.raises(ValueError, match="invalid override key"):
        render_config("stage: sft\n", {key: "value"})


def _write_fake_cli(path: Path) -> None:
    path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'cp "$2" "$CAPTURE_CONFIG"\n'
        'env | sort > "$CAPTURE_ENV"\n',
        encoding="utf-8",
    )
    path.chmod(0o755)


def _launcher_env(tmp_path: Path, nnodes: int) -> tuple[dict[str, str], Path, Path]:
    fake_cli = tmp_path / "llamafactory-cli"
    model_dir = tmp_path / "Qwen VL"
    checkpoint_dir = tmp_path / "checkpoint-10"
    captured_config = tmp_path / "captured.yaml"
    captured_env = tmp_path / "captured.env"
    model_dir.mkdir()
    checkpoint_dir.mkdir()
    _write_fake_cli(fake_cli)

    env = os.environ.copy()
    env.update(
        {
            "LLAMAFACTORY_CLI": str(fake_cli),
            "MODEL_NAME_OR_PATH": str(model_dir),
            "OUTPUT_DIR": str(tmp_path / "output dir"),
            "RESUME_FROM_CHECKPOINT": str(checkpoint_dir),
            "MAX_STEPS": "5",
            "NNODES": str(nnodes),
            "NPROC_PER_NODE": "8",
            "NODE_RANK": "0",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29611",
            "TARGET_GLOBAL_BATCH_SIZE": "32",
            "CAPTURE_CONFIG": str(captured_config),
            "CAPTURE_ENV": str(captured_env),
        }
    )
    return env, captured_config, captured_env


@pytest.mark.parametrize(("nnodes", "expected_accumulation"), [(1, 4), (2, 2), (4, 1)])
def test_launcher_derives_accumulation_and_propagates_distributed_env(
    tmp_path: Path, nnodes: int, expected_accumulation: int
):
    env, captured_config, captured_env = _launcher_env(tmp_path, nnodes)

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    config = captured_config.read_text(encoding="utf-8")
    launch_env = captured_env.read_text(encoding="utf-8")
    assert f"gradient_accumulation_steps: {expected_accumulation}" in config
    assert f'model_name_or_path: "{tmp_path}/Qwen VL"' in config
    assert f'output_dir: "{tmp_path}/output dir"' in config
    assert "max_steps: 5" in config
    assert f'resume_from_checkpoint: "{tmp_path}/checkpoint-10"' in config
    assert "FORCE_TORCHRUN=1" in launch_env
    assert f"NNODES={nnodes}" in launch_env
    assert "NPROC_PER_NODE=8" in launch_env
    assert "MASTER_PORT=29611" in launch_env


def test_launcher_rejects_world_size_that_cannot_preserve_global_batch(tmp_path: Path):
    env, _, _ = _launcher_env(tmp_path, nnodes=3)

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "does not divide TARGET_GLOBAL_BATCH_SIZE" in result.stderr


def test_launcher_dry_run_prints_config_without_starting_cli(tmp_path: Path):
    env, captured_config, _ = _launcher_env(tmp_path, nnodes=1)
    env["DRY_RUN"] = "1"

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "world_size=8" in result.stdout
    assert "gradient_accumulation_steps: 4" in result.stdout
    assert not captured_config.exists()


def _top_level_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line[0].isspace() or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key] = value.strip().split("  #", 1)[0]
    return values


def test_full_sft_config_matches_original_project_strategy():
    values = _top_level_values(ROOT / "configs" / "train_visual_tool_sft_full.yaml")

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


def test_operator_shell_contains_huawei_platform_setup_without_destructive_cleanup():
    script = (ROOT / "scripts" / "run_visual_tool_sft.sh").read_text(encoding="utf-8")

    assert 'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"' in script
    assert 'NPROC_PER_NODE="${NPROC_PER_NODE:-8}"' in script
    assert 'TARGET_GLOBAL_BATCH_SIZE="${TARGET_GLOBAL_BATCH_SIZE:-32}"' in script
    assert 'BASE="${BASE:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx}"' in script
    assert 'ROOT_DIR="${ROOT_DIR:-$BASE/visual-agent}"' in script
    assert 'ENV_DIR="${ENV_DIR-$BASE/conda_envs/deepeyes-sft-conda}"' in script
    assert 'MODEL_DIR="${MODEL_DIR-$BASE/DeepEyesV2/models/Qwen2.5-VL-7B-Instruct}"' in script
    qwen3_script = (ROOT / "scripts" / "run_visual_tool_sft_qwen3.sh").read_text(encoding="utf-8")
    assert 'MODEL_DIR="${MODEL_DIR-$BASE/DeepEyesV2/models/Qwen3-VL-8B-Instruct}"' in qwen3_script
    assert "Visual-agent Qwen3-VL SFT launcher" in qwen3_script
    assert 'PLATFORM_SETUP="${PLATFORM_SETUP:-1}"' in script
    assert "/opt/huawei/explorer-env/dataset/trellis_ckpt/cuda/cuda118" in script
    assert "/opt/huawei/explorer-env/dataset/Common_wl/miniconda3" in script
    assert "/opt/huawei/schedule-train/algorithm/algorithmrefs/synaflow_wl" in script
    assert "/opt/huawei/quoteModel/xiaoyi_tmpstorage" in script
    assert "/home/ma-user/work/algorithm/synaflow_wl" in script
    assert 'MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH-$MODEL_DIR}"' in script
    assert 'HF_HOME="${HF_HOME:-$PLATFORM_WORK_ROOT/cache/huggingface}"' in script
    assert 'OFFLINE_MODE="${OFFLINE_MODE:-0}"' in script
    assert "NCCL_DEBUG" in script
    assert "nvidia-smi" in script
    assert "train_visual_tool_sft_full.sh" in script
    assert "rm -rf" not in script


def test_operator_shell_can_dry_run_outside_huawei_platform(tmp_path: Path):
    env, _, _ = _launcher_env(tmp_path, nnodes=1)
    env.update(
        {
            "ROOT_DIR": str(ROOT),
            "PLATFORM_SETUP": "0",
            "CONDA_ENV_PATH": "",
            "LOG_DIR": str(tmp_path / "logs"),
            "HF_HOME": str(tmp_path / "cache" / "huggingface"),
            "XDG_CACHE_HOME": str(tmp_path / "cache" / "xdg"),
            "TORCH_HOME": str(tmp_path / "cache" / "torch"),
            "PRINT_HARDWARE_INFO": "0",
            "DRY_RUN": "1",
        }
    )

    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "run_visual_tool_sft.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Visual-agent Qwen2.5-VL SFT launcher" in result.stdout
    assert "world_size=8" in result.stdout
    assert list((tmp_path / "logs").glob("train-node0-*.log"))


def test_operator_shell_can_run_when_copied_outside_repository(tmp_path: Path):
    env, _, _ = _launcher_env(tmp_path, nnodes=1)
    copied_launcher = tmp_path / "run_visual_tool_sft.sh"
    copied_launcher.write_bytes(
        (ROOT / "scripts" / "run_visual_tool_sft.sh").read_bytes()
    )
    copied_launcher.chmod(0o755)
    env.update(
        {
            "BASE": str(ROOT.parent),
            "PLATFORM_SETUP": "0",
            "CONDA_ENV_PATH": "",
            "LOG_DIR": str(tmp_path / "logs"),
            "HF_HOME": str(tmp_path / "cache" / "huggingface"),
            "XDG_CACHE_HOME": str(tmp_path / "cache" / "xdg"),
            "TORCH_HOME": str(tmp_path / "cache" / "torch"),
            "PRINT_HARDWARE_INFO": "0",
            "DRY_RUN": "1",
        }
    )

    result = subprocess.run(
        ["bash", str(copied_launcher)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert f"ROOT_DIR: {ROOT}" in result.stdout
    assert "world_size=8" in result.stdout


def test_slurm_wrapper_requests_one_task_and_eight_gpus_per_node():
    script = (ROOT / "scripts" / "slurm" / "train_visual_tool_sft.sbatch").read_text(
        encoding="utf-8"
    )

    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --ntasks-per-node=1" in script
    assert "#SBATCH --gres=gpu:8" in script
    assert 'REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"' in script
    assert 'MASTER_ADDR="$(scontrol show hostnames' in script
    assert 'NODE_RANK="${SLURM_PROCID}"' in script
    assert "--ntasks-per-node=1" in script
    assert "train_visual_tool_sft_full.sh" in script
