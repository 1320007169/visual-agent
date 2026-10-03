import json
import os
from pathlib import Path
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
        'if [[ "$VLOCR_STEP" == 70 && -n "${TEST_ROTATE_MODEL:-}" ]]; then\n'
        '    mv "$TEST_ROTATE_MODEL" "$TEST_ROTATE_MODEL.retired"\n'
        'fi\n'
        'if [[ "$VLOCR_STEP" == 70 ]]; then exit "${TEST_FIRST_EXIT:-0}"; fi\n'
    )
    env = {"PATH": os.environ["PATH"], "BASE": str(tmp_path), "REPO_ROOT": str(repo),
           "RUN_ID": "latest_eval_test", "WORK_ROOT": str(tmp_path / "results")}
    return env, sources


def run_launcher(env, **changes):
    return subprocess.run(["bash", str(LAUNCHER)], env={**env, **changes},
                          capture_output=True, text=True, timeout=30)


def test_latest_checkpoints_are_pinned_and_evaluated_in_separate_outputs(evaluation):
    env, sources = evaluation
    result = run_launcher(env, TEST_ROTATE_MODEL=str(sources[1]), EVAL_DATASETS="OCRBench ChartQA_TEST")
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [line.split("|")[1:] for line in result.stdout.splitlines() if line.startswith("EVAL|")]
    assert [row[0] for row in rows] == ["70", "100"]
    output = Path(env["WORK_ROOT"])
    for row, label in zip(rows, ("64gpu_step70", "16gpu_step100")):
        assert row[1] == str(output / "checkpoints" / label)
        assert row[2] == str(output / label)
        assert row[3] == "OCRBench ChartQA_TEST"
        assert (Path(row[1]) / "tokenizer.json").is_file()
    assert not sources[1].exists()
    assert (output / "checkpoints/16gpu_step100/model.safetensors").stat().st_ino == (
        Path(str(sources[1]) + ".retired") / "model.safetensors").stat().st_ino
    assert (output / "checkpoints/64gpu_step70/model.safetensors").stat().st_ino == (
        sources[0] / "model.safetensors").stat().st_ino
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
        "VLMEVAL_API_NPROC": "14",
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
