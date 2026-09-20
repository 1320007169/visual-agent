import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import threading

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
