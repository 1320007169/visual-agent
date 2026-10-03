import ast
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


TRAINER = Path(__file__).resolve().parents[1] / "reinforcement_learning/verl/trainer/ppo/ray_trainer.py"


@pytest.mark.parametrize("resume_override", [None, "/previous/global_step_20"])
def test_eight_node_entrypoint_resumes_step80_skipping_step83(resume_override):
    root = TRAINER.parents[4]
    source_run = "qwen3base_multitool_vlocr_ocr_chart1600_n16_8node_20260930T195425656668_e9c99c3c"
    data_dir = root / "data/vlocr_quality_skip_step83_continuation_step80_20261004"
    env = {"PATH": os.environ["PATH"], "REPO_ROOT": str(root), "BASE": str(root.parent),
           "MULTITOOL_CONFIG_ONLY": "1", "TRAIN_RUN_TOKEN": "test_step80_skip83"}
    if resume_override:
        env["RESUME_FROM_PATH"] = resume_override
    result = subprocess.run(["bash", str(root / "scripts/run_visual_agent_multitool_vlocr_8node_64gpu_modelarts.sh")],
                            env=env, capture_output=True, text=True, check=True)
    config = dict(line.split("=", 1) for line in result.stdout.splitlines())
    assert config["RESUME_MODE"] == "resume_path"
    assert config["RESUME_FROM_PATH"] == (resume_override or str(data_dir / "resume/global_step_80"))
    assert config["TRAIN_FILES"] == str(data_dir / "train.parquet")
    assert config["VAL_FILES"] == str(data_dir / "val.parquet")
    assert config["TRAIN_SHUFFLE"] == "False"
    assert config["TRAINER_STOP_AFTER_SECONDS"] == "0"
    assert config["NNODES"] == "8" and config["TRAIN_BATCH_SIZE"] == "336"
    assert config["TOTAL_TRAINING_STEPS"] == "null" and config["SAVE_FREQ"] == "1"
    assert config["RUN_ID"].startswith("qwen3base_multitool_vlocr_quality_skip83_n16_8node_from_step80_")
    assert config["RUN_ID"].endswith("_test_step80_skip83")
    for key in ("RL_OUTPUT_DIR", "RL_LOG_DIR", "ROLLOUT_DATA_DIR", "SYNC_DIR"):
        assert config[key].endswith("_test_step80_skip83")
        assert source_run not in config[key]


@pytest.mark.parametrize("resume_path", [None, "/previous/global_step_20"])
def test_initial_validation_records_the_weights_actually_loaded(resume_path):
    tree = ast.parse(TRAINER.read_text())
    call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == "record_best_checkpoint"
                and any(keyword.arg == "checkpoint_path" for keyword in node.keywords))
    runner = SimpleNamespace(config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(
        model=SimpleNamespace(path="/base-model"))))
    if resume_path:
        runner._loaded_checkpoint_path = resume_path
    recorded = []
    namespace = {
        "self": runner, "os": os, "best_val_step": 20 if resume_path else 0,
        "best_val_metric": 0.6, "record_best_checkpoint": lambda **kwargs: recorded.append(kwargs),
    }
    module = ast.fix_missing_locations(ast.Module(body=[ast.Expr(value=call)], type_ignores=[]))
    exec(compile(module, str(TRAINER), "exec"), namespace)
    expected = f"{resume_path}/actor/huggingface" if resume_path else "/base-model"
    assert recorded[0]["checkpoint_path"] == expected


@pytest.mark.parametrize("resume_override", [None, "/previous/global_step_90"])
def test_two_node_ocr_chart_entrypoint_resumes_step79_without_deadline(resume_override):
    root = TRAINER.parents[4]
    source_run = "qwen3base_multitool_vlocr_ocr_chart_n16_2node_from_step60_20261001T042830523880_74d4f11e"
    env = {"PATH": os.environ["PATH"], "REPO_ROOT": str(root), "BASE": str(root.parent),
           "MULTITOOL_CONFIG_ONLY": "1", "TRAIN_RUN_TOKEN": "test_step79_resume"}
    if resume_override:
        env["RESUME_FROM_PATH"] = resume_override
    result = subprocess.run(["bash", str(root / "scripts/run_visual_agent_multitool_vlocr_ocr_chart_resume_2node_16gpu_modelarts.sh")],
                            env=env, capture_output=True, text=True, check=True)
    config = dict(line.split("=", 1) for line in result.stdout.splitlines())
    assert config["RESUME_MODE"] == "resume_path"
    assert config["RESUME_FROM_PATH"] == (resume_override or str(
        root / "saves/visual_agent_zwz_rl/qwen3" / source_run / "global_step_79"))
    assert config["TRAINER_STOP_AFTER_SECONDS"] == "0"
    assert config["NNODES"] == "2" and config["TRAIN_BATCH_SIZE"] == "126"
    assert config["TRAIN_SHUFFLE"] == "False" and config["DATALOADER_NUM_WORKERS"] == "2"
    assert config["TOTAL_TRAINING_STEPS"] == "267" and config["SAVE_FREQ"] == "10"
    for key in ("RL_OUTPUT_DIR", "RL_LOG_DIR", "ROLLOUT_DATA_DIR", "SYNC_DIR"):
        assert config[key].endswith("_test_step79_resume")
        assert source_run not in config[key]


def test_full_resume_loads_actor_and_dataloader_and_preserves_step():
    tree = ast.parse(TRAINER.read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_load_checkpoint")
    calls = []
    checkpoint = "/previous/global_step_20"
    runner = SimpleNamespace(
        config=SimpleNamespace(trainer=SimpleNamespace(
            resume_mode="resume_path", default_hdfs_dir=None, default_local_dir="/new-output",
            resume_from_path=checkpoint, del_local_ckpt_after_load=False)),
        actor_rollout_wg=SimpleNamespace(load_checkpoint=lambda path, **kwargs: calls.append(("actor", path, kwargs))),
        train_dataloader=SimpleNamespace(load_state_dict=lambda state: calls.append(("data", state))),
        use_critic=False,
    )
    namespace = {
        "os": SimpleNamespace(path=SimpleNamespace(isabs=os.path.isabs, join=os.path.join, exists=lambda _: True)),
        "find_latest_ckpt_path": lambda _: None,
        "torch": SimpleNamespace(load=lambda path, **kwargs: {"source": path}),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(TRAINER), "exec"), namespace)
    namespace["_load_checkpoint"](runner)
    assert runner.global_steps == 20
    assert runner._loaded_checkpoint_path == checkpoint
    assert calls == [
        ("actor", checkpoint + "/actor", {"del_local_after_load": False}),
        ("data", {"source": checkpoint + "/data.pt"}),
    ]
