import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/run_visual_agent_eval_multitool_vlocr_64gpu_16gpu_latest_8gpu_modelarts.sh"
RUNS = (
    "qwen3base_multitool_vlocr_ocr_chart1600_n16_8node_20261001T072342925318_021ea85c",
    "qwen3base_multitool_vlocr_ocr_chart_n16_2node_from_step60_20261002T170651989009_040bfc67",
)


@pytest.fixture
def evaluation(tmp_path):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    sources = []
    for run, step in zip(RUNS, (70, 100)):
        output = repo / "saves/visual_agent_zwz_rl/qwen3" / run
        model = output / f"global_step_{step}/actor/huggingface"
        model.mkdir(parents=True)
        (output / "latest_checkpointed_iteration.txt").write_text(str(step))
        (model / "config.json").write_text('{"model_type": "qwen3_vl"}')
        (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "model.safetensors"}}))
        (model / "model.safetensors").write_text(f"weights_step_{step}\n")
        (model / "tokenizer.json").write_text("{}")
        sources.append(model)
    (scripts / "run_visual_agent_eval_multitool_vlocr_8gpu.sh").write_text(
        '#!/usr/bin/env bash\n'
        'set -eu\n'
        'printf "EVAL|%s|%s|%s|%s\\n" "$VLOCR_STEP" "$VLOCR_MODEL_PATH" "$WORK_ROOT" "${EVAL_DATASETS:-default}"\n'
        'cat "$VLOCR_MODEL_PATH/model.safetensors"\n'
        'if [[ "$VLOCR_STEP" == 70 ]]; then exit "${TEST_FIRST_EXIT:-0}"; fi\n'
    )
    env = {"PATH": os.environ["PATH"], "BASE": str(tmp_path), "REPO_ROOT": str(repo),
           "RUN_ID": "latest_eval_test", "WORK_ROOT": str(tmp_path / "results")}
    return env, sources


def run_launcher(env, **changes):
    return subprocess.run(["bash", str(LAUNCHER)], env={**env, **changes},
                          capture_output=True, text=True, timeout=30)


def test_latest_checkpoints_are_read_directly_and_evaluated_in_separate_outputs(evaluation):
    env, sources = evaluation
    result = run_launcher(env, EVAL_DATASETS="OCRBench ChartQA_TEST")
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [line.split("|")[1:] for line in result.stdout.splitlines() if line.startswith("EVAL|")]
    assert [row[0] for row in rows] == ["70", "100"]
    output = Path(env["WORK_ROOT"])
    for row, label, source in zip(rows, ("64gpu_step70", "16gpu_step100"), sources):
        assert row[1] == str(source)
        assert row[2] == str(output / label)
        assert row[3] == "OCRBench ChartQA_TEST"
        assert (Path(row[1]) / "tokenizer.json").is_file()
    assert not (output / "checkpoints").exists()
    for source, step in zip(sources, (70, 100)):
        assert (source / "model.safetensors").read_text() == f"weights_step_{step}\n"
    status = (output / "status.tsv").read_text().splitlines()
    assert [line.split("\t")[:2] for line in status[1:]] == [["64gpu_step70", "0"], ["16gpu_step100", "0"]]
    assert str(sources[1]) in (output / "checkpoints.tsv").read_text()


def test_missing_second_weight_shard_stops_before_output_or_evaluation(evaluation):
    env, sources = evaluation
    (sources[1] / "model.safetensors").write_bytes(b"")
    result = run_launcher(env)
    assert result.returncode != 0
    assert "Missing or empty HF weight shards" in result.stdout + result.stderr
    assert "EVAL|" not in result.stdout
    assert not Path(env["WORK_ROOT"]).exists()


def test_existing_results_are_preserved(evaluation):
    env, _ = evaluation
    output = Path(env["WORK_ROOT"])
    output.mkdir()
    marker = output / "old_result.txt"
    marker.write_text("previous results")
    result = run_launcher(env)
    assert result.returncode != 0
    assert "EVAL|" not in result.stdout
    assert marker.read_text() == "previous results"
    assert list(output.iterdir()) == [marker]


def test_failed_first_evaluation_records_status_and_attempts_second(evaluation):
    env, _ = evaluation
    result = run_launcher(env, TEST_FIRST_EXIT="17")
    assert result.returncode == 1
    assert len([line for line in result.stdout.splitlines() if line.startswith("EVAL|")]) == 2
    status = (Path(env["WORK_ROOT"]) / "status.tsv").read_text().splitlines()
    assert [line.split("\t")[:2] for line in status[1:]] == [["64gpu_step70", "17"], ["16gpu_step100", "0"]]


@pytest.mark.parametrize("reuse_same_checkpoint", [True, False])
@pytest.mark.parametrize("tool_gpu", ["7", "1"])
def test_chart_service_starts_without_conda_on_path(evaluation, reuse_same_checkpoint, tool_gpu):
    env, sources = evaluation
    base = Path(env["BASE"])
    previous = base / "previous_eval"
    previous_predictions = previous / "64gpu_step70/dino_latest/VisualAgent-vllm/T20261003_G"
    previous_predictions.mkdir(parents=True)
    reuse_model = sources[0] if reuse_same_checkpoint else sources[1]
    (previous / "checkpoints.tsv").write_text(
        f"checkpoint\tstep\tmodel_path\n64gpu\t70\t{reuse_model}\n")
    prediction = previous_predictions / "VisualAgent-vllm_VStarBench.xlsx"
    prediction.write_bytes(b"previous successful predictions")
    partial = previous_predictions / "VisualAgent-vllm_VStarBench_supp.pkl"
    partial.write_bytes(b"partial predictions")
    (previous_predictions / "VisualAgent-vllm_VStarBench_acc.csv").write_text("old scores")
    pipeline = base / "pipeline"
    pipeline.mkdir()
    service_env = base / "service_env"
    (service_env / "bin").mkdir(parents=True)
    (service_env / "bin/python3").write_text(
        '#!/bin/bash\nprintf "SERVICE_GPU|%s\\n" "$CUDA_VISIBLE_DEVICES"\nexec /bin/sleep 300\n')
    (service_env / "bin/python3").chmod(0o755)
    chart_env = base / "chart_env"
    (chart_env / "bin").mkdir(parents=True)
    (chart_env / "lib/python3.10/site-packages/torch/lib").mkdir(parents=True)
    (chart_env / "bin/python3").write_text(
        '#!/bin/bash\n'
        'printf "CHART|%s|%s|%s|%s|%s|%s\\n" "$0" "$CUDA_VISIBLE_DEVICES" '
        '"$HF_HUB_OFFLINE" "$TRANSFORMERS_OFFLINE" "$LD_LIBRARY_PATH" "$*"\n'
        'touch "$TEST_CHART_STARTED"\n'
        'exec /bin/sleep 300\n')
    (chart_env / "bin/python3").chmod(0o755)
    (pipeline / ".env").write_text(
        f'VTS_OUTPUT_ROOT="{base}/bridge"\nVTS_COUNT_ENV="{service_env}"\n')
    fake_bin = base / "bin"
    fake_bin.mkdir()
    (fake_bin / "curl").write_text('#!/bin/bash\nsleep 0.1\ntest -f "$TEST_CHART_STARTED"\n')
    (fake_bin / "curl").chmod(0o755)
    repo = Path(env["REPO_ROOT"])
    (repo / "scripts/run_visual_agent_eval_qwen3.sh").write_text(
        'echo "EVALUATION_STARTED|$MODEL_CUDA_VISIBLE_DEVICES|$ENV_DIR"\n')
    overrides = {"TOOL_CUDA_VISIBLE_DEVICES": tool_gpu}
    if tool_gpu == "1":
        overrides.update(MODEL_CUDA_VISIBLE_DEVICES="0", ENV_DIR=str(service_env))
    assert shutil.which("conda", path=f"{fake_bin}:/usr/bin:/bin") is None
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/run_visual_agent_eval_multitool_vlocr_8gpu.sh")],
        env={**env, **overrides, "PATH": f"{fake_bin}:/usr/bin:/bin", "PIPELINE_ROOT": str(pipeline),
             "VLOCR_DEPTH_ENV": str(service_env), "VTS_CHART_ENV": str(chart_env),
             "VLOCR_STEP": "70", "VLOCR_MODEL_PATH": str(sources[0]),
             "VLOCR_REUSE_GROUP_ROOT": str(previous),
             "EVAL_DATASETS": "VStarBench", "LOG_DIR": str(base / "logs"),
             "LD_LIBRARY_PATH": "/platform/driver/lib", "TEST_CHART_STARTED": str(base / "chart.started")},
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "EVALUATION_STARTED" in result.stdout
    if tool_gpu == "1":
        assert f"EVALUATION_STARTED|0|{service_env}" in result.stdout
    for service in ("depth", "count"):
        assert f"SERVICE_GPU|{tool_gpu}" in (Path(env["WORK_ROOT"]) / f"services/{service}.log").read_text()
    cached = Path(env["WORK_ROOT"]) / "dino_latest/VisualAgent-vllm" / env["RUN_ID"]
    assert (cached / prediction.name).exists() == reuse_same_checkpoint
    assert (cached / partial.name).exists() == reuse_same_checkpoint
    assert not (cached / "VisualAgent-vllm_VStarBench_acc.csv").exists()
    assert prediction.read_bytes() == b"previous successful predictions"
    if reuse_same_checkpoint:
        assert (cached / prediction.name).read_bytes() == prediction.read_bytes()
        assert (cached / partial.name).read_bytes() == partial.read_bytes()
    chart_log = (Path(env["WORK_ROOT"]) / "services/chart.log").read_text()
    expected_libraries = f"{chart_env}/lib:{chart_env}/lib/python3.10/site-packages/torch/lib:/platform/driver/lib"
    assert f"CHART|{chart_env}/bin/python3|{tool_gpu}|1|1|{expected_libraries}|" in chart_log
    assert "paddleocr_vl_chart_server.py" in chart_log


@pytest.mark.parametrize("mode", ["CHECKPOINT_EVAL_CONFIG_ONLY", "EVAL_PREFLIGHT_ONLY"])
def test_configuration_and_preflight_do_not_create_result_or_snapshot_directories(evaluation, mode):
    env, _ = evaluation
    result = run_launcher(env, **{mode: "1"})
    assert result.returncode == 0, result.stdout + result.stderr
    assert not Path(env["WORK_ROOT"]).exists()
    if mode == "CHECKPOINT_EVAL_CONFIG_ONLY":
        assert "EVAL|" not in result.stdout


def test_both_checkpoints_reuse_recent_vlocr_evaluation_environment(evaluation):
    env, _ = evaluation
    base = Path(env["BASE"])
    scripts = Path(env["REPO_ROOT"]) / "scripts"
    (scripts / "run_visual_agent_eval_multitool_vlocr_8gpu.sh").write_text(
        (ROOT / "scripts/run_visual_agent_eval_multitool_vlocr_8gpu.sh").read_text())
    pipeline = base / "pipeline"
    pipeline.mkdir()
    (pipeline / ".env").write_text(
        f'VTS_OUTPUT_ROOT="{base}/bridge"\n'
        f'VTS_COUNT_ENV="{base}/conda_envs/vts-count"\n')
    cuda = str(base / "conda_envs/spacetools-rl")
    expected = {
        "ENV_DIR": "/opt/huawei/explorer-env/dataset/Common_wl/miniconda3/envs/qwenvl3_xmx_vLLM",
        "CUDA_HOME": cuda,
        "CUDA_LIBRARY_DIR": cuda + "/targets/x86_64-linux/lib",
        "CC": cuda + "/bin/x86_64-conda-linux-gnu-gcc",
        "CXX": cuda + "/bin/x86_64-conda-linux-gnu-g++",
        "CUDAHOSTCXX": cuda + "/bin/x86_64-conda-linux-gnu-g++",
        "MODEL_CUDA_HOME": cuda,
        "MODEL_CC": cuda + "/bin/x86_64-conda-linux-gnu-gcc",
        "MODEL_CXX": cuda + "/bin/x86_64-conda-linux-gnu-g++",
        "TOOL_CUDA_HOME": "/opt/huawei/explorer-env/dataset/trellis_ckpt/cuda/cuda118",
        "VTS_DEPTH_ENV": "/opt/huawei/explorer-env/dataset/Common_wl/miniconda3/envs/starVLA_flash_dzw1",
        "VTS_COUNT_ENV": str(base / "conda_envs/vts-count"),
        "VTS_CHART_ENV": str(base / "conda_envs/qwen38-vllm-clean"),
        "PADDLEOCR_VL_MODEL_ROOT": str(base / "visual-tools/paddlex-cache/official_models/PaddleOCR-VL-1.6"),
        "MODEL_CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6",
        "TOOL_CUDA_VISIBLE_DEVICES": "7",
        "GPU_MEMORY_UTILIZATION": "0.80",
        "VLMEVAL_API_NPROC": "7",
        "MODEL_SERVER_BACKEND": "vllm",
        "SKIP_CONDA_ACTIVATION": "1",
        "VLMEVAL_IMPORT_PREFLIGHT": "1",
    }
    (scripts / "run_visual_agent_eval_qwen3.sh").write_text(
        "#!/usr/bin/env bash\npython3 - <<'PY'\nimport json\nimport os\n"
        f"keys = {list(expected)!r}\n"
        'print("ENV|" + json.dumps({key: os.environ.get(key) for key in keys}))\n'
        "PY\n")
    result = run_launcher(env, EVAL_PREFLIGHT_ONLY="1", PIPELINE_ROOT=str(pipeline),
                          LOG_DIR=str(base / "logs"), CUDA_HOME="/wrong/cuda",
                          CC="/wrong/gcc", CXX="/wrong/g++", CUDAHOSTCXX="/wrong/g++")
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [json.loads(line.removeprefix("ENV|")) for line in result.stdout.splitlines()
            if line.startswith("ENV|")]
    assert rows == [expected, expected]
    assert not Path(env["WORK_ROOT"]).exists()
