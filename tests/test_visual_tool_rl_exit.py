import os
from pathlib import Path
import subprocess

import pytest


LAUNCHER = Path(__file__).resolve().parents[1] / "scripts/run_visual_tool_rl_2node_16gpu.sh"


@pytest.mark.parametrize("node_rank", ["0", "1"])
@pytest.mark.parametrize("action,expected", [
    ("exit 0", 0),
    ("exit 7", 7),
    ("false", 1),
    ('kill -INT "$$"', 130),
    ('kill -TERM "$$"', 143),
])
def test_exit_status_and_cleanup_once(tmp_path, node_rank, action, expected):
    source = LAUNCHER.read_text()
    start = source.index("cleanup() {\n")
    end = source.index('\necho "====', start)
    cleanup = source[start:end]
    done_file = tmp_path / "done"
    cleanup_log = tmp_path / "cleanup.log"
    script = "\n".join([
        "GPU_MONITOR_PID=''", "WANDB_MONITOR_PID=''", "TOOL_PIDS=()",
        'fake_ray() { printf "%s\\n" "$*" >> "$EXIT_TEST_CLEANUP_LOG"; }',
        "RAY_BIN=fake_ray", cleanup, action,
    ])
    result = subprocess.run(
        ["bash", "-euc", script], capture_output=True, text=True, timeout=5,
        env={**os.environ, "NODE_RANK": node_rank, "DONE_FILE": str(done_file),
             "EXIT_TEST_CLEANUP_LOG": str(cleanup_log)},
    )
    assert result.returncode == expected, result.stderr
    assert cleanup_log.read_text().splitlines() == ["stop --force"]
    if node_rank == "0":
        assert done_file.read_text().strip() == str(expected)
    else:
        assert not done_file.exists()
