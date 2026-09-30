import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/run_visual_agent_multitool_depth_count_2node_16gpu.sh"


def test_cuda_failure_reports_environment_and_stops(tmp_path):
    (tmp_path / "torch.py").write_text(
        'from types import SimpleNamespace\n'
        '__version__ = "test+cu128"\n'
        'version = SimpleNamespace(cuda="12.8")\n'
        'def ones(*args, **kwargs):\n'
        '    raise RuntimeError("The NVIDIA driver is too old (found version 11070)")\n'
    )
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_visual_agent_cuda.py"), "PaddleOCR-VL"],
        env=dict(os.environ, PYTHONPATH=str(tmp_path), CUDA_VISIBLE_DEVICES=""),
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 2
    assert '"torch_cuda": "12.8"' in result.stdout
    assert "11070" in result.stderr
    assert "loaded_libcuda" in result.stderr
    assert "nvidia_smi" in result.stderr
    assert "Changing CUDA_HOME alone does not upgrade the driver" in result.stderr


def test_later_service_exit_interrupts_wait_for_first_service():
    source = LAUNCHER.read_text()
    wait_loop = source[source.index('for index in "${!vts_endpoints[@]}"; do'):]
    wait_loop = wait_loop.split('\nbash "$REPO_ROOT/scripts/run_visual_agent_zwz_rl_2node_16gpu.sh"')[0]
    result = subprocess.run(
        ["bash", "-c", 'set -Eeuo pipefail\n'
         'vts_endpoints=(http://depth/health http://ocr/health)\n'
         'VTS_SERVICE_PIDS=("$$" 999999999)\n'
         'START_VTS_SERVICES=1\nVTS_SERVICE_DIR=/test-logs\nVTS_SERVICE_STARTUP_TIMEOUT=3600\n'
         'curl() { return 1; }\n'
         'sleep() { echo UNEXPECTED_WAIT >&2; exit 9; }\n' + wait_loop],
        capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == 2
    assert "http://ocr/health" in result.stderr
    assert "UNEXPECTED_WAIT" not in result.stderr


def test_rl_cuda_failure_stops_before_ocr_or_services(tmp_path):
    python = tmp_path / "bin/python3"
    python.parent.mkdir()
    python.write_text('#!/bin/bash\necho "RL_DRIVER_FAILURE" >&2\nexit 2\n')
    python.chmod(0o755)
    source = LAUNCHER.read_text()
    preflight = source[source.index('        if [[ -n "${RL_ENV_DIR:-}" ]]; then'):]
    preflight = preflight.split('\n    fi\n    "$VTS_DEPTH_ENV/bin/python3"')[0]
    result = subprocess.run(
        ["bash", "-c", 'set -Eeuo pipefail\n'
         'conda() { echo UNEXPECTED_OCR >&2; exit 9; }\n' + preflight + '\necho UNEXPECTED_SERVICES\n'],
        env=dict(os.environ, RL_ENV_DIR=str(tmp_path), RL_CUDA_VISIBLE_DEVICES="0,1", REPO_ROOT=str(ROOT)),
        capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == 2
    assert "RL_DRIVER_FAILURE" in result.stderr
    assert "UNEXPECTED" not in result.stdout + result.stderr


@pytest.mark.parametrize("command_index", [0, 1], ids=["preflight", "ocr_server"])
def test_ocr_commands_restore_platform_libraries_after_conda_activation(tmp_path, command_index):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    torch_lib = tmp_path / "lib/python3.10/site-packages/torch/lib"
    torch_lib.mkdir(parents=True)
    conda = bin_dir / "conda"
    conda.write_text(
        '#!/bin/bash\nshift 4\n'
        'export LD_LIBRARY_PATH=/conda/override\nexec "$@"\n'
    )
    conda.chmod(0o755)
    python = bin_dir / "python3"
    python.write_text('#!/bin/bash\nprintf "%s\\n" "$LD_LIBRARY_PATH"\n')
    python.chmod(0o755)
    source = LAUNCHER.read_text()
    setup = source[source.index('        chart_library_path='):source.index('        if [[ -n "${RL_ENV_DIR:-}" ]]; then')]
    lines = source.splitlines()
    starts = [i for i, line in enumerate(lines)
              if 'conda run --no-capture-output -p "$VTS_CHART_ENV"' in line]
    start = starts[command_index]
    end = next(i for i in range(start + 1, len(lines)) if '>"$VTS_SERVICE_DIR/' in lines[i])
    command = '\n'.join(lines[start:end]).strip().removesuffix('\\')
    result = subprocess.run(
        ["bash", "-c", 'set -Eeuo pipefail\n' + setup + command],
        env=dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}",
                 LD_LIBRARY_PATH="/platform/compat/lib.real:/platform/compat/lib",
                 VTS_CHART_ENV=str(tmp_path), TOOL_GPU="7", REPO_ROOT=str(ROOT),
                 PADDLEOCR_VL_MODEL_ROOT="/model", VTS_TOOL_BRIDGE_ROOT="/bridge"),
        capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"{tmp_path}/lib:{torch_lib}:/platform/compat/lib.real:/platform/compat/lib"
