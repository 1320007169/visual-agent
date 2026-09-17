"""Exercise the diagnostic wrapper with local stub launchers, never GPU services."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "scripts/run_visual_agent_eval_tool_diagnostic_modelarts.sh"
STUB = '''#!/usr/bin/env bash
"$CONFIG_PYTHON" - <<'PY'
import json, os
from pathlib import Path
import sys
out = Path(os.environ['WORK_ROOT'])
out.mkdir(parents=True, exist_ok=True)
keys = ['EVAL_MODELS', 'QWEN3_MODEL_PATH', 'RL_DINO_LATEST_MODEL_PATH',
        'VISUAL_AGENT_SYSTEM_PROMPT_FILE', 'VISUAL_AGENT_MAX_TURNS',
        'VISUAL_AGENT_MAX_TOKENS', 'MODEL_CUDA_VISIBLE_DEVICES',
        'EVAL_DATASETS', 'VLMEVAL_EVAL_ID', 'VLLM_PYTHON']
(out / 'receipt.json').write_text(json.dumps({k: os.environ.get(k) for k in keys}))
sys.exit(7 if out.name == os.environ.get('FAIL_MODE') else 0)
PY
'''


@pytest.fixture
def setup(tmp_path):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    for name in ("run_visual_agent_eval_qwen3.sh", "run_visual_agent_eval_qwen3_compare.sh"):
        (scripts / name).write_text(STUB)
    model = repo / "checkpoint"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "model.safetensors"}}))
    (model / "model.safetensors").write_bytes(b"test-only-not-model-weights")
    env = dict(os.environ, REPO_ROOT=str(repo), CONFIG_PYTHON=sys.executable,
               DIAGNOSTIC_MODEL_PATH=str(model), DIAGNOSTIC_STEP="120",
               DIAGNOSTIC_RUN_ID="test_run", DIAGNOSTIC_OUTPUT_ROOT=str(tmp_path / "out"),
               DIAGNOSTIC_PROMPT_FILE=str(ROOT / "prompts/visual_agent_rl_system_groundingdino.txt"),
               DIAGNOSTIC_CONFIG_ONLY="0", DIAGNOSTIC_MODES="direct auto tool_first",
               VISUAL_AGENT_MAX_TURNS="6", VISUAL_AGENT_MAX_TOKENS="6144",
               EVAL_DATASETS="VStarBench HRBench4K HRBench8K", NNODES="1")
    return env, tmp_path / "out/test_run", model


def run(env):
    return subprocess.run(["bash", str(ENTRY)], env=env, text=True, capture_output=True)


def test_modes_reuse_existing_launchers_and_the_same_checkpoint(setup):
    env, output, model = setup
    result = run(env)
    assert result.returncode == 0, result.stderr
    receipts = {mode: json.loads((output / mode / "receipt.json").read_text())
                for mode in ("direct", "auto", "tool_first")}
    assert receipts["direct"]["EVAL_MODELS"] == "qwen3_base"
    assert receipts["direct"]["QWEN3_MODEL_PATH"] == str(model)
    assert receipts["direct"]["VISUAL_AGENT_SYSTEM_PROMPT_FILE"] is None
    for mode in ("auto", "tool_first"):
        assert receipts[mode]["EVAL_MODELS"] == "dino_latest"
        assert receipts[mode]["RL_DINO_LATEST_MODEL_PATH"] == str(model)
        assert Path(receipts[mode]["VISUAL_AGENT_SYSTEM_PROMPT_FILE"]).is_file()
    for key in ("VISUAL_AGENT_MAX_TOKENS", "MODEL_CUDA_VISIBLE_DEVICES", "EVAL_DATASETS", "VLLM_PYTHON"):
        assert len({receipt[key] for receipt in receipts.values()}) == 1
    assert len({receipt["VLMEVAL_EVAL_ID"] for receipt in receipts.values()}) == 3
    auto = (output / "auto_prompt.txt").read_text()
    required = (output / "tool_first_prompt.txt").read_text()
    assert auto.strip() == Path(env["DIAGNOSTIC_PROMPT_FILE"]).read_text().strip()
    assert required.startswith(auto.strip())
    assert "Your first assistant response must call" in required
    assert len((output / "status.tsv").read_text().splitlines()) == 4
    assert json.loads((output / "protocol.json").read_text())["checkpoint"] == str(model)


def test_config_only_does_not_start_or_write(setup):
    env, output, _ = setup
    result = run(dict(env, DIAGNOSTIC_CONFIG_ONLY="1"))
    assert result.returncode == 0, result.stderr
    assert not output.exists()


def test_failure_is_recorded_and_other_modes_still_run(setup):
    env, output, _ = setup
    result = run(dict(env, FAIL_MODE="auto"))
    assert result.returncode == 1
    rows = (output / "status.tsv").read_text().splitlines()
    assert rows[2].startswith("auto\t7\t")
    assert (output / "tool_first/receipt.json").is_file()


def test_existing_results_are_never_reused(setup):
    env, output, _ = setup
    output.mkdir(parents=True)
    marker = output / "user_result.txt"
    marker.write_text("keep")
    assert run(env).returncode != 0
    assert marker.read_text() == "keep"
    assert not (output / "direct").exists()


@pytest.mark.parametrize("modes", ["", "auto auto", "unknown"])
def test_invalid_modes_fail_before_starting(setup, modes):
    env, output, _ = setup
    # Empty uses the wrapper's documented default; whitespace is truly empty.
    assert run(dict(env, DIAGNOSTIC_MODES=modes or " ")).returncode != 0
    assert not output.exists()


def test_custom_turn_budget_updates_both_agent_prompts(setup):
    env, output, _ = setup
    result = run(dict(env, VISUAL_AGENT_MAX_TURNS="12", VISUAL_AGENT_MAX_TOKENS="512"))
    assert result.returncode == 0, result.stderr
    for mode in ("auto", "tool_first"):
        prompt = (output / f"{mode}_prompt.txt").read_text()
        assert "at most 12 assistant turns" in prompt
        assert "at most 11 tool calls" in prompt


def test_legacy_budget_does_not_leak_into_other_modes(setup):
    env, output, _ = setup
    legacy = output.parent / "legacy.txt"
    legacy.write_text("Historical tool prompt\n")
    result = run(dict(env, DIAGNOSTIC_MODES="legacy direct auto tool_first",
                      DIAGNOSTIC_LEGACY_PROMPT_FILE=str(legacy)))
    assert result.returncode == 0, result.stderr
    for mode, turns, tokens in [("legacy", "12", "512"), ("direct", "1", "6144"),
                                ("auto", "6", "6144"), ("tool_first", "6", "6144")]:
        receipt = json.loads((output / mode / "receipt.json").read_text())
        assert receipt["VISUAL_AGENT_MAX_TURNS"] == turns
        assert receipt["VISUAL_AGENT_MAX_TOKENS"] == tokens
    assert (output / "legacy_prompt.txt").read_text() == legacy.read_text()
    assert len((output / "status.tsv").read_text().splitlines()) == 5
