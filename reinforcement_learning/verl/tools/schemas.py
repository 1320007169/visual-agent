# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from typing import Any, Literal

from pydantic import BaseModel


class OpenAIFunctionPropertySchema(BaseModel):
    """The schema of a parameter in OpenAI format."""

    type: str
    description: str | None = None
    enum: list[str] | None = None


class OpenAIFunctionParametersSchema(BaseModel):
    """The schema of parameters in OpenAI format."""

    type: str
    properties: dict[str, OpenAIFunctionPropertySchema]
    required: list[str]


class OpenAIFunctionSchema(BaseModel):
    """The schema of a function in OpenAI format."""

    name: str
    description: str
    parameters: OpenAIFunctionParametersSchema
    strict: bool = False


class OpenAIFunctionToolSchema(BaseModel):
    """The schema of a tool in OpenAI format."""

    type: str
    function: OpenAIFunctionSchema


def load_tool_schemas_from_config(path: str | None) -> list[dict]:
    """Read the same schemas as the rollout without constructing tool clients."""
    if not path:
        return []
    from omegaconf import OmegaConf

    config = OmegaConf.load(path)
    schemas = []
    for tool in config.tools:
        if tool.get("tool_schema") is None:
            raise ValueError(f"Dataset prompt encoding requires an explicit tool_schema: {tool.class_name}")
        schema = OpenAIFunctionToolSchema.model_validate(
            OmegaConf.to_container(tool.tool_schema, resolve=True)
        )
        schemas.append(schema.model_dump(exclude_unset=True, exclude_none=True))
    return schemas


class OpenAIFunctionParsedSchema(BaseModel):
    """The parsed schema of a tool in OpenAI format."""

    name: str
    arguments: str  # JSON string


class OpenAIFunctionCallSchema(BaseModel):
    """The parsed schema of a tool in OpenAI format."""

    name: str
    arguments: dict[str, Any]

    @staticmethod
    def from_openai_function_parsed_schema(parsed_schema: OpenAIFunctionParsedSchema) -> tuple["OpenAIFunctionCallSchema", bool]:
        has_decode_error = False
        try:
            arguments = json.loads(parsed_schema.arguments)
        except json.JSONDecodeError:
            arguments = {}
            has_decode_error = True
        # If the arguments is not a dict, it means the arguments is not a valid JSON string
        if not isinstance(arguments, dict):
            arguments = {}
            has_decode_error = True

        return OpenAIFunctionCallSchema(name=parsed_schema.name, arguments=arguments), has_decode_error


class OpenAIFunctionToolCall(BaseModel):
    """The tool call in OpenAI format."""

    id: str
    type: Literal["function"] = "function"
    function: OpenAIFunctionCallSchema
