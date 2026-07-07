# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

import json
from typing import Any, Optional, Tuple
from uuid import uuid4

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema


class OfflineVisualTool(BaseTool):
    """Schema-only visual tool for offline SFT and parser compatibility.

    This class intentionally does not connect to SAM3 or GroundingDINO. Offline
    SFT samples already contain tool observations, so training only needs the
    tool names and schemas to be valid.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"calls": []}
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        if instance_id in self._instance_dict:
            self._instance_dict[instance_id]["calls"].append(parameters)
        response = {
            "error": "offline_visual_tool_has_no_executor",
            "tool": self.name,
            "message": "Visual tool execution is disabled for offline SFT.",
        }
        return json.dumps(response), 0.0, {"offline": True, "tool": self.name}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)
