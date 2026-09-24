import json
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import threading
from unittest.mock import patch

from PIL import Image


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from vts_tool_bridge import VtsRemoteBridge  # noqa: E402


def test_bridge_translates_images_to_shared_paths_and_back(tmp_path):
    received = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def do_POST(self):
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length))
            received.update(payload)
            source = Path(payload["context"]["images"][0]["path"])
            assert source.is_file()
            artifact = Path(payload["context"]["artifact_dir"]) / "annotated.png"
            with Image.open(source) as opened:
                opened.convert("RGB").save(artifact)
            body = json.dumps(
                {
                    "status": "success",
                    "success": True,
                    "text": "ok",
                    "images": [
                        {
                            "image_id": 1,
                            "path": str(artifact),
                            "parent_image_id": 0,
                            "source": "test",
                        }
                    ],
                    "structured": {"text": "ok"},
                    "metrics": {},
                    "provenance": {},
                    "error_code": None,
                    "error_message": None,
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        bridge = VtsRemoteBridge(
            endpoint=f"http://127.0.0.1:{server.server_port}",
            tool_name="ocr_read",
            shared_root=tmp_path,
        )
        result = bridge.execute(
            images=[Image.new("RGB", (32, 24), "white")],
            arguments={"image_id": 0},
            instance_id="rollout/1",
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert received["tool"] == "ocr_read"
    assert received["args"] == {"image_id": 0}
    assert received["context"]["images"][0]["image_id"] == 0
    assert result.result["structured"] == {"text": "ok"}
    assert len(result.images) == 1
    assert result.images[0].startswith("data:image/png;base64,")


def test_bridge_balances_across_replicas_and_fails_over(tmp_path):
    calls = [0, 0]
    servers = []
    threads = []

    def handler_for(index):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                return

            def do_POST(self):
                calls[index] += 1
                self.rfile.read(int(self.headers["Content-Length"]))
                body = json.dumps({"status": "success", "structured": {"replica": index}}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return Handler

    try:
        for index in range(2):
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(index))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            servers.append(server)
            threads.append(thread)

        endpoints = ",".join(f"http://127.0.0.1:{server.server_port}" for server in servers)
        with patch.dict("os.environ", {"VTS_DEPTH_ENDPOINT": endpoints, "VTS_TOOL_BRIDGE_ROOT": str(tmp_path)}):
            bridge = VtsRemoteBridge.from_env(endpoint_env="VTS_DEPTH_ENDPOINT", tool_name="depth_estimate")
        for index in range(20):
            bridge.execute(images=[Image.new("RGB", (4, 4))], arguments={"image_id": 0}, instance_id=f"rollout-{index}")
        assert all(count > 0 for count in calls)

        servers[0].shutdown()
        servers[0].server_close()
        threads[0].join(timeout=5)
        candidate = next(
            f"failover-{index}" for index in range(20)
            if int.from_bytes(hashlib.sha256(f"failover-{index}".encode()).digest()[:8], "big") % 2 == 0
        )
        result = bridge.execute(images=[Image.new("RGB", (4, 4))], arguments={"image_id": 0}, instance_id=candidate)
        assert result.result["structured"]["replica"] == 1
    finally:
        for server, thread in zip(servers, threads):
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
