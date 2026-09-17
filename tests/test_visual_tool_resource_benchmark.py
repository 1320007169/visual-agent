import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch, Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchmark_visual_tool_resources as resources
from benchmark_visual_tool_pool import run_level


class BenchmarkTests(unittest.TestCase):
    def test_cpu_budget_and_gpu_mapping(self):
        plans = resources.scenarios([6, 7], list(range(16)), [4, 8, 16], [1, 2])
        self.assertNotIn("cpu_w2_t16", [p["name"] for p in plans])
        self.assertEqual(plans[-1]["gpus"], [6, 7])
        self.assertEqual(plans[-1]["workers"] * plans[-1]["replicas"], 12)

    def test_ids(self):
        self.assertEqual(resources.integers("4-6,8"), [4, 5, 6, 8])
        with self.assertRaises(ValueError):
            resources.integers("4-6,5")

    def test_cases_and_strict_json(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "image.jpg").write_bytes(b"image")
            cases = root / "cases.jsonl"
            cases.write_text(json.dumps({"image": "image.jpg", "query": "red car"}) + "\n")
            payload = json.loads(resources.load_payloads(cases)[0])
            self.assertEqual(payload["arguments"]["queries"][0]["query"], "red car")
            self.assertTrue(payload["images"][0].startswith("data:image/jpeg;base64,"))
            self.assertEqual(resources.finite_json({"x": [float("nan")]}), {"x": [None]})

    def test_http_load_and_tool_errors(self):
        seen = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                seen.append((payload["id"], self.headers.get("Authorization")))
                result = {"status": "success", "result": {"status": payload["status"]},
                          "metrics": {"latency_ms": 0.1}}
                body = json.dumps(result).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            with patch.dict(os.environ, {"VISUAL_TOOL_API_KEY": "test-token", "no_proxy": "127.0.0.1"}):
                result = run_level([f"http://127.0.0.1:{server.server_port}"], [
                    b'{"id": 1, "status": "ok"}', b'{"id": 2, "status": "error"}'
                ], concurrency=2, requests=6, timeout=2)
            self.assertEqual(result["successful"], 3)
            self.assertEqual(result["failed"], 3)
            self.assertEqual(sorted(x[0] for x in seen), [1, 1, 1, 2, 2, 2])
            self.assertTrue(all(x[1] == "Bearer test-token" for x in seen))
            self.assertGreater(result["elapsed_s"], 0)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_startup_exit(self):
        process = Mock()
        process.poll.return_value = 1
        with self.assertRaisesRegex(RuntimeError, "exited"):
            resources.wait_ready([process], ["http://127.0.0.1:1"], "token", 1)

    def test_cleanup_only_owned_processes(self):
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
        resources.stop_processes([process])
        self.assertIsNotNone(process.poll())

    def test_cpu_launch_and_failure_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            args = argparse.Namespace(port=19100, core_ids=list(range(8)), model_path=Path(temp),
                                      output=Path(temp), tool_python=sys.executable,
                                      startup_timeout=1, warmup=1, timeout=1, repeats=1,
                                      levels=[1, 2], requests=2)
            plan = resources.scenarios([], args.core_ids, [4], [2])[0]
            with patch.dict(os.environ, {"VISUAL_TOOL_API_KEY": "token"}), \
                 patch.object(resources.socket, "socket"), \
                 patch.object(resources.subprocess, "Popen") as popen, \
                 patch.object(resources, "wait_ready"), \
                 patch.object(resources, "stop_processes") as stop, \
                 patch.object(resources, "execute_request", return_value={"ok": True}), \
                 patch.object(resources, "run_level", return_value={
                     "failed": 1, "elapsed_s": 1, "throughput_rps": 1, "client_p95_ms": 1}) as level:
                result = resources.run_scenario(plan, args, [b"payload"])
            self.assertEqual(popen.call_count, 2)
            self.assertEqual(popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
            self.assertIn("[4, 5, 6, 7]", popen.call_args.args[0][2])
            self.assertEqual(level.call_count, 1)
            self.assertIn("error", result)
            stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
