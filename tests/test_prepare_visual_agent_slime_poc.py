import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/prepare_visual_agent_slime_poc.py"
SPEC = importlib.util.spec_from_file_location("prepare_visual_agent_slime_poc", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_convert_row_builds_structured_chat_with_image_pixel_limit():
    converted = MODULE.convert_row(
        {
            "question": "Where is the cup?",
            "solution": "RIGHT OF",
            "images": ["/data/a.jpg", "/data/b.jpg"],
            "bbox": [1, 2, 3, 4],
            "source_index": 17,
        },
        "Use tools.",
        0,
        image_max_pixels=2359296,
    )
    assert converted["messages"] == [
        {"role": "system", "content": "Use tools."},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "/data/a.jpg", "max_pixels": 2359296},
                {"type": "image", "image": "/data/b.jpg", "max_pixels": 2359296},
                {"type": "text", "text": "Where is the cup?"},
            ],
        },
    ]
    assert converted["images"] == ["/data/a.jpg", "/data/b.jpg"]
    assert converted["solution"] == "right of"
    assert converted["metadata"]["source_index"] == 17


def test_convert_row_rejects_missing_multimodal_input():
    with pytest.raises(ValueError, match="no images"):
        MODULE.convert_row({"question": "q", "solution": "above", "images": []}, "prompt", 3)


def test_convert_row_preserves_mixed_source_and_free_form_answer():
    converted = MODULE.convert_row(
        {
            "question": "What color is the cushion?",
            "solution": "The cushion is Brown.",
            "images": ["/data/cushion.jpg"],
            "data_source": "visual-agent-deepeyesv2",
            "ability": "attribute",
            "split": "train",
            "source_image": "perception.parquet:17",
        },
        "Use tools.",
        4,
    )

    assert converted["solution"] == "The cushion is Brown."
    assert converted["metadata"]["data_source"] == "visual-agent-deepeyesv2"
    assert converted["metadata"]["source_name"] == "visual-agent-deepeyesv2"
    assert converted["metadata"]["ability"] == "attribute"
    assert converted["metadata"]["source_image"] == "perception.parquet:17"


def test_slime_launcher_expands_multinode_kl_training_args(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    slime_root = tmp_path / "slime"
    slime_root.mkdir()
    (slime_root / "train.py").write_text("# test checkout\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(slime_root)], check=True)
    subprocess.run(["git", "-C", str(slime_root), "add", "train.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(slime_root),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "test",
        ],
        check=True,
    )

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}), encoding="utf-8")
    source = tmp_path / "train.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [{"question": "Where?", "solution": "above", "images": ["/not/read/in/dry-run.jpg"]}]
        ),
        source,
    )
    prompt = tmp_path / "system.txt"
    prompt.write_text("Use tools.", encoding="utf-8")
    rollout_config = tmp_path / "rollout.yaml"
    rollout_config.write_text("max_turns: 6\nmax_tokens_per_turn: 512\n", encoding="utf-8")

    env = {
        **os.environ,
        "SLIME_ROOT": str(slime_root),
        "ALLOW_UNPINNED_SLIME": "1",
        "MODEL_PATH": str(model),
        "REF_MODEL_PATH": str(model),
        "SOURCE_DATA": str(source),
        "SYSTEM_PROMPT_FILE": str(prompt),
        "CUSTOM_CONFIG_PATH": str(rollout_config),
        "OUTPUT_DIR": str(tmp_path / "output"),
        "VISUAL_TOOL_API_BASES": "http://node0:9000,http://node1:9000",
        "ACTOR_NUM_NODES": "1",
        "ACTOR_GPUS_PER_NODE": "2",
        "NUM_GPUS_PER_NODE": "2",
        "ROLLOUT_NUM_GPUS": "2",
        "ROLLOUT_BATCH_SIZE": "4",
        "N_SAMPLES_PER_PROMPT": "2",
        "NUM_STEPS_PER_ROLLOUT": "2",
        "GLOBAL_BATCH_SIZE": "4",
        "KL_LOSS_ENABLED": "1",
        "KL_LOSS_COEF": "0.001",
        "SGLANG_SERVER_CONCURRENCY": "16",
        "SGLANG_MAX_RUNNING_REQUESTS": "16",
        "SGLANG_CHUNKED_PREFILL_SIZE": "32768",
        "PREPARE_LIMIT": "0",
        "REQUIRE_IMAGES": "0",
        "CHECK_WEIGHT_UPDATE_EQUAL": "0",
        "DUMP_DETAILS": "",
        "DRY_RUN": "1",
    }
    completed = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts/run_visual_agent_slime_poc.sh")],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"rows": 1' in completed.stdout
    assert "optimizer steps / global sample batch: 2 / 4" in completed.stdout
    assert "--num-gpus-per-node 2" in completed.stdout
    assert "--use-kl-loss" in completed.stdout
    assert "--kl-loss-coef 0.001" in completed.stdout
    assert "--ref-load" in completed.stdout
    assert "--sglang-server-concurrency 16" in completed.stdout
    assert "--sglang-max-running-requests 16" in completed.stdout
    assert "--sglang-chunked-prefill-size 32768" in completed.stdout
    assert "--dump-details" not in completed.stdout
