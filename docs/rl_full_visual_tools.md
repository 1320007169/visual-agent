# Full RL visual tools

The full profile adds seven tools to the existing VL-OCR five-tool profile.
The original five schemas, executors and result projections are unchanged.
The format validator also accepts the seven new names so their valid calls
receive the existing format reward.
Existing launchers continue to select their original tool configurations.

| New tool | Arguments in addition to target_image | Result |
| --- | --- | --- |
| image_resize | width + height, or max_side | New image index and dimensions |
| image_enhance | Optional contrast, sharpness, brightness, gamma, denoise | New image index and dimensions |
| image_rotate | angle; optional expand | New image index and dimensions |
| image_flip | direction: horizontal, vertical, both | New image index and dimensions |
| image_draw | boxes, points, lines; optional color, width | New annotated image index and dimensions |
| sam_segment | bbox_2d | Mask + overlay image indices, score and area |
| bbox_geometry | bbox_a, bbox_b | Image-plane relations, IoU, overlap and distance |

All boxes, points and line endpoints use the selected image's relative 0–1000
coordinate system. Resize dimensions and drawing width are in pixels; rotation
angles are counterclockwise degrees. Images are appended to the rollout's
existing image list. For segmentation, the mask is appended before the overlay.
Geometry returns no image; center_distance_px accounts for the image's width
and height. Geometry does not measure depth.

The five image operations use Pillow inside the existing Visual-Agent server.
Segmentation uses the existing VTS SAM3 box-prompted service through
VTS_SEGMENT_ENDPOINT. There is no change to the synthesis repository.

## Enable for a new experiment

Start the SAM3 service in its existing environment from the pipeline checkout,
alongside the existing depth, count and VL-OCR services. Select a GPU with enough
capacity for SAM3 and export the pipeline source path:

```bash
cd /home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/groundingdino_offline_pipeline
set -a
source .env
set +a
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES=7 "$VTS_SAM3_ENV/bin/python3" -m vts.tool_server \
    --config configs/services/sam3.yaml
```

On each training node, configure the bridge and select the new profile before
running an existing VL-OCR launcher:

```bash
cd /home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx/visual-agent
export VTS_SEGMENT_ENDPOINT=http://127.0.0.1:9001
export TOOL_CONFIG_PATH="$PWD/reinforcement_learning/examples/sglang_multiturn/config/tool_config/visual_tool_multitool_full_config.yaml"
export VISUAL_AGENT_RL_SYSTEM_PROMPT_FILE="$PWD/prompts/visual_agent_rl_system_multitool_full.txt"
bash scripts/run_visual_agent_multitool_vlocr_3node_24gpu.sh
```

The SAM3 service and Visual-Agent server must use the same
VTS_TOOL_SERVICE_TOKEN. VTS_TOOL_BRIDGE_ROOT must be inside the SAM3 service's
server.allowed_roots, normally VTS_OUTPUT_ROOT. The existing launcher manages
depth, count and VL-OCR; start and manage SAM3 separately. If using remote
segmentation, both processes must see the same shared filesystem paths.

The new profile has 12 tools. To enable a subset, copy the full YAML and keep
the desired entries. For evaluation, use the same full prompt and set
VISUAL_AGENT_TOOL_CONFIG_PATH to the full YAML when using native tools.
Adding schemas changes the policy's tool context; choose the full profile
explicitly for a new experiment. Dataset and reward definitions are unchanged.

```bash
python3 -m pytest -q tests/test_visual_tool_extensions.py tests/test_visual_tool_server.py
```
