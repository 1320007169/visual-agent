"""Whole-message response budgets with reusable image preprocessing."""

import copy

import torch


class CachedImageProcessor:
    def __init__(self, processor):
        self.processor = processor
        self.cache = {}

    def __getattr__(self, name):
        return getattr(self.processor, name)

    def __call__(self, images, **kwargs):
        # Qwen VL image features concatenate along the image/patch dimension.
        from transformers.feature_extraction_utils import BatchFeature

        features = []
        for image in images:
            key = (id(image), repr(kwargs))
            if key not in self.cache:
                self.cache[key] = (image, self.processor(images=[image], **kwargs))
            features.append(self.cache[key][1])
        return BatchFeature({
            key: torch.cat([feature[key] for feature in features], dim=0)
            if torch.is_tensor(features[0][key]) else
            [item for feature in features for item in feature[key]]
            for key in features[0]
        })


class ResponseBudget:
    def __init__(self, tokenizer, processor, prompt, tool_schemas, limit, encode, collect_images):
        if limit <= 0:
            raise ValueError("response_length must be positive")
        self.tokenizer = tokenizer
        self.processor = copy.copy(processor) if processor is not None else None
        if self.processor is not None:
            self.processor.image_processor = CachedImageProcessor(processor.image_processor)
        self.prompt_length = len(prompt)
        self.prompt_text = tokenizer.apply_chat_template(
            prompt, tools=tool_schemas, add_generation_prompt=True, tokenize=False
        )
        self.tools = tool_schemas
        self.limit = limit
        self.encode_response = encode
        self.collect_images = collect_images
        self.image_cache = {}
        self.truncated = False
        self.reason = None
        self.last_encoding = None
        self.last_text = None
        self.last_images = None
        self.removed_groups = 0

    def release_cache(self, keep_encoding=False):
        """Keep only final output tensors while waiting for batch postprocessing."""
        self.image_cache.clear()
        if self.processor is not None:
            self.processor.image_processor.cache.clear()
        if not keep_encoding:
            self.last_encoding = self.last_text = self.last_images = None

    def _retain_image_cache(self, messages):
        images = self.collect_images(messages[self.prompt_length:])
        keys = {image if isinstance(image, str) else id(image) for image in images}
        for key in list(self.image_cache):
            if key not in keys:
                del self.image_cache[key]
        if self.processor is not None:
            retained_ids = {id(image) for image in self.image_cache.values()}
            cache = self.processor.image_processor.cache
            for key in list(cache):
                if key[0] not in retained_ids:
                    del cache[key]
        self.last_encoding = self.last_text = self.last_images = None

    def encode(self, messages):
        if len(messages) == self.prompt_length:
            self.release_cache()
            empty = torch.empty(0, dtype=torch.long)
            return empty, empty.clone(), {}
        text = self.tokenizer.apply_chat_template(
            messages, tools=self.tools, add_generation_prompt=False, tokenize=False
        )[len(self.prompt_text):]
        images = self.collect_images(messages[self.prompt_length:])
        image_keys = tuple(image if isinstance(image, str) else id(image) for image in images)
        if text != self.last_text or image_keys != self.last_images:
            self.last_encoding = self.encode_response(
                self.tokenizer, self.processor, text,
                images, self.image_cache,
            )
            self.last_text = text
            self.last_images = image_keys
        return self.last_encoding

    def drop_last_group(self, messages):
        for index in range(len(messages) - 1, self.prompt_length - 1, -1):
            if messages[index].get("role") == "assistant":
                del messages[index:]
                self.removed_groups += 1
                self._retain_image_cache(messages)
                return
        raise ValueError("Response observations have no preceding assistant message")

    def enforce(self, messages):
        encoded = self.encode(messages)
        while encoded[0].numel() > self.limit:
            self.truncated = True
            self.reason = "response_length"
            self.drop_last_group(messages)
            encoded = self.encode(messages)
        return encoded

    def generation_allowance(self, messages):
        used = self.encode(messages)[0].numel()
        # Include the next complete assistant template, including its terminator.
        empty_assistant = {"role": "assistant", "content": ""}
        overhead = max(1, self.encode([*messages, empty_assistant])[0].numel() - used)
        return max(0, self.limit - used - overhead)


def retain_trace(trace, turns, truncated, reason):
    if trace is None:
        return
    discarded = []
    for key in ("model_calls", "tool_calls"):
        kept = []
        for index, call in enumerate(trace[key], 1):
            turn = index if key == "model_calls" else call["model_turn"]
            if turn <= turns:
                kept.append(call)
            else:
                discarded.append({**call, "kind": key, "discarded": True})
        trace[key] = kept
    trace["discarded_calls"] = discarded
    trace["truncated"] = truncated
    trace["truncation_reason"] = reason
    trace["retained_turns"] = turns
    # Wall-clock execution includes discarded calls. Concurrent call durations
    # are cumulative costs and cannot be subtracted from this wall-clock time.
    trace["execution_active_latency_ms"] = trace.get("active_latency_ms", 0)
    for prefix in ("model", "tool"):
        calls = trace[prefix + "_calls"]
        trace[prefix + "_call_count"] = len(calls)
        trace["retained_" + prefix + "_cumulative_latency_ms"] = sum(call["latency_ms"] for call in calls)
    trace["retained_call_cumulative_latency_ms"] = (
        trace["retained_model_cumulative_latency_ms"] + trace["retained_tool_cumulative_latency_ms"]
    )
