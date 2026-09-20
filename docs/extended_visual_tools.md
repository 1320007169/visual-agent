# Extended Visual Tools

The extended tool profile adds OCR, metric-depth estimation, and open-world
counting without changing the existing Visual-Agent call or image-return
protocol. Existing GroundingDINO/SAM configurations remain unchanged.

## Model-visible tools

- `ocr_read(target_image)`
- `depth_measure(target_image, bbox_2d)`
- `ground_depth(target_image, query)`
- `object_count(query, target_image)`

All model-visible coordinates use the independent `relative_0_1000` coordinate
space of `target_image`. The server translates them to the pixel/path-based VTS
service contract and converts VTS output images back to data URLs.

OCR always applies the server threshold, returns at most 50 regions, and includes
its annotated image. Counting always includes its annotated evidence image.
These execution policies are intentionally not model parameters.

`depth_measure` consumes one existing relative box. `ground_depth` localizes
one concrete query with the existing GroundingDINO backend. Both always return
the region median metric-depth estimate and a depth visualization. Values may
only be compared when measured on the same `target_image`.

## Service setup

Run the isolated services from the `groundingdino_offline_pipeline` checkout:

```bash
vts-tool-server --config configs/services/paddleocr_vl.yaml
vts-tool-server --config configs/services/depth_anything_3.yaml
vts-tool-server --config configs/services/countgd_plusplus.yaml
```

The services and Visual-Agent server must share a writable directory included
in each service's `server.allowed_roots` configuration. Configure the bridge:

```bash
export VTS_OCR_ENDPOINT=http://127.0.0.1:9002
export VTS_DEPTH_ENDPOINT=http://127.0.0.1:9003
export VTS_COUNT_ENDPOINT=http://127.0.0.1:9004
export VTS_TOOL_SERVICE_TOKEN=...
export VTS_TOOL_BRIDGE_ROOT=/shared/vts-output/visual_agent_bridge
```

Start the existing Visual-Agent server with GroundingDINO available for
query-based depth targets. Use these files for a new experiment:

```text
reinforcement_learning/examples/sglang_multiturn/config/tool_config/visual_tool_extended_config.yaml
prompts/visual_agent_rl_system_extended.txt
```

Do not replace `visual_tool_groundingdino_config.yaml` in historical runs. Tool
schemas are part of the tokenized prompt, so adding tools changes the policy
context even when none of the new tools are called.

## Result projection

Full VTS `ToolResult`, provenance, backend statistics, and metric-depth arrays
remain inside the VTS service and its audit artifacts. The model receives only:

- OCR text, regions, and an optional annotated image.
- Count, relative evidence points/boxes, and an optional annotated image.
- One region-level `depth_m` estimate, its relative bbox, and a depth
  visualization.

This keeps the model protocol stable while allowing a future local Python
executor to replace the remote services behind the same schemas.
