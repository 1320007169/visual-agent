import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts/prepare_visual_agent_run_paths.sh"
RESOLVER = ROOT / "scripts/resolve_visual_agent_run_token.py"


class RunTokenTest(unittest.TestCase):
    def launch_cluster(self, directory, via_shell=False):
        processes = []
        for rank in (2, 1, 0):
            env = {
                "PATH": os.environ["PATH"], "BASE": directory, "REPO_ROOT": str(ROOT),
                "NNODES": "3", "VC_TASK_INDEX": str(rank), "VC_WORKER_HOSTS": '["node0", "node1", "node2"]',
                "RUN_ID": "token_test", "RESUME_MODE": "disable", "RUN_TOKEN_TIMEOUT_SECONDS": "5",
            }
            command = [sys.executable, str(RESOLVER)]
            if via_shell:
                command = ["bash", "-euc", 'source "$1"; printf "%s\\n" "$RUN_ID" "$OUTPUT_DIR" "$LOG_DIR"', "bash", str(HELPER)]
            processes.append(subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
            time.sleep(0.05)
        results = [process.communicate(timeout=10) for process in processes]
        for process, (stdout, stderr) in zip(processes, results):
            self.assertEqual(process.returncode, 0, stderr)
        outputs = [stdout.strip() for stdout, _ in results]
        self.assertEqual(len(set(outputs)), 1)
        return outputs[0]

    def test_workers_start_first_and_relaunch_on_same_hosts_uses_new_token(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self.launch_cluster(directory)
            second = self.launch_cluster(directory)
            self.assertNotEqual(first, second)

    def test_shell_initialization_shares_paths_without_job_id(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self.launch_cluster(directory, via_shell=True)
            second = self.launch_cluster(directory, via_shell=True)
            self.assertNotEqual(first, second)

    def test_missing_worker_times_out(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(RESOLVER)], capture_output=True, text=True, timeout=5,
                env={"PATH": os.environ["PATH"], "BASE": directory, "NNODES": "2",
                     "VC_TASK_INDEX": "0", "VC_WORKER_HOSTS": "node0,node1", "RUN_TOKEN_TIMEOUT_SECONDS": "0.2"},
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("timed out", result.stderr)


if __name__ == "__main__":
    unittest.main()
