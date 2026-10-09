import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("failed_stage", ["", "auto_exemplar", "mme_baseline"])
def test_fsc_then_mme_preserves_order_and_isolates_protocols(tmp_path, failed_stage):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    prompts = repo / "prompts"
    prompts.mkdir()
    for name in ("visual_agent_rl_system_multitool_vlocr.txt", "visual_agent_eval_mme_evidence.txt"):
        shutil.copy2(ROOT / "prompts" / name, prompts / name)
    shutil.copy2(ROOT / "scripts/run_visual_agent_eval_mme_prompt_8gpu.sh", scripts)
    model = repo / (
        "saves/visual_agent_zwz_rl/qwen3/"
        "qwen3base_multitool_vlocr_hme_chartqa_tallyhalf_fsc3000_n16_8node_20261003T202135115738_f9c41af7/"
        "global_step_80/actor/huggingface")
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "weights"}}))
    (model / "weights").write_text("fixture")
    data = tmp_path / "datasets/fsc147"
    data.mkdir(parents=True)
    points = {str(i): {"points": [[0, 0]]} for i in range(1190)}
    (data / "annotation_FSC147_384.json").write_text(json.dumps(points))
    (data / "Train_Test_Val_FSC_147.json").write_text(json.dumps({"test": list(points)}))
    config = repo / "reinforcement_learning/examples/sglang_multiturn/config/tool_config"
    config.mkdir(parents=True)
    (config / "visual_tool_multitool_vlocr_config.yaml").write_text("{}")
    pipeline = tmp_path / "pipeline"
    (pipeline / "configs/services").mkdir(parents=True)
    for name in ("countgd_plusplus.yaml", "countgd_plusplus_pseudo_eval.yaml"):
        (pipeline / "configs/services" / name).write_text("{}")
    binary = tmp_path / "env/bin"
    binary.mkdir(parents=True)
    (binary / "python3").symlink_to(sys.executable)
    (scripts / "run_visual_agent_eval_multitool_vlocr_8gpu.sh").write_text(
        "python3 - <<'PY'\nimport json, os\nfrom pathlib import Path\n"
        'stage = Path(os.environ["WORK_ROOT"]).name\n'
        'keys = ["EVAL_DATASETS", "COUNT_SERVICE_CONFIG", "VISUAL_AGENT_SYSTEM_PROMPT_FILE", '
        '"VLOCR_MODEL_PATH", "WORK_ROOT", "VLMEVAL_EVAL_ID"]\n'
        'with open(os.environ["TEST_EVENTS"], "a") as stream:\n'
        '    stream.write(json.dumps({"stage": stage, **{k: os.environ[k] for k in keys}}) + "\\n")\n'
        'raise SystemExit(17 if stage == os.environ["TEST_FAILED_STAGE"] else 0)\nPY\n')
    (scripts / "summarize_fsc_count_ab.py").write_text(
        'import os\nwith open(os.environ["TEST_EVENTS"], "a") as stream:\n'
        '    stream.write(\'{"stage": "summary"}\\n\')\n')
    events = tmp_path / "events.jsonl"
    output = tmp_path / "results"
    env = {"PATH": os.environ["PATH"], "BASE": str(tmp_path), "REPO_ROOT": str(repo),
           "PIPELINE_ROOT": str(pipeline), "ENV_DIR": str(binary.parent), "SKIP_MODELARTS_BOOTSTRAP": "1",
           "RUN_ID": "test", "WORK_ROOT": str(output), "TEST_EVENTS": str(events),
           "TEST_FAILED_STAGE": failed_stage, "VISUAL_AGENT_SYSTEM_PROMPT_FILE": "/inherited/experiment.txt"}
    result = subprocess.run(["bash", str(ROOT / "scripts/run_visual_agent_eval_fsc_count_ab_8gpu_modelarts.sh")],
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == (17 if failed_stage else 0), result.stdout + result.stderr
    rows = [json.loads(line) for line in events.read_text().splitlines()]
    stages = ["text_only", "auto_exemplar", "summary", "mme_baseline", "mme_evidence"]
    if failed_stage:
        stages = stages[:stages.index(failed_stage) + 1]
    assert [row["stage"] for row in rows] == stages
    for row in rows:
        stage = row["stage"]
        if stage == "summary":
            continue
        assert row["EVAL_DATASETS"] == ("MME-RealWorld-Lite" if stage.startswith("mme_") else "FSC147_TEST")
        backend = "countgd_plusplus_pseudo_eval.yaml" if stage == "auto_exemplar" else "countgd_plusplus.yaml"
        assert row["COUNT_SERVICE_CONFIG"] == str(pipeline / "configs/services" / backend)
        prompt = "visual_agent_eval_mme_evidence.txt" if stage == "mme_evidence" else "visual_agent_rl_system_multitool_vlocr.txt"
        assert row["VISUAL_AGENT_SYSTEM_PROMPT_FILE"] == str(prompts / prompt)
        assert row["VLOCR_MODEL_PATH"] == str(model)
        assert row["WORK_ROOT"] == str(output / stage)
        assert row["VLMEVAL_EVAL_ID"] == f"test_{stage}"
    if failed_stage == "mme_baseline":
        assert (output / "mme_status.tsv").read_text().splitlines()[-1].startswith("baseline\t17\t")
    elif not failed_stage:
        assert len((output / "status.tsv").read_text().splitlines()) == 3
        assert len((output / "mme_status.tsv").read_text().splitlines()) == 3
