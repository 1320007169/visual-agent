"""Adaptive allocation of online counterfactual tool faults."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path


MAX_LEVEL = {"ocr_read": 2, "object_count": 3, "grounding_detect": 2}


class OnlineFaultController:
    def __init__(self, config_path: str):
        config = json.loads(Path(config_path).read_text())
        self.eligible = defaultdict(list)
        for row in config["eligible"]:
            self.eligible[row["data_source"]].append(row["tool"])
        if any(tool not in MAX_LEVEL for tools in self.eligible.values() for tool in tools):
            raise ValueError("Online fault config contains an unsupported tool")
        self.initial_fraction = float(config["initial_fraction"])
        self.minimum_fraction = float(config["minimum_fraction"])
        self.target_mixed_fraction = float(config["target_mixed_fraction"])
        self.ema_alpha = float(config["ema_alpha"])
        self.level_cooldown_steps = int(config["level_cooldown_steps"])
        if not (0 < self.minimum_fraction <= self.initial_fraction <= 1
                and 0 < self.target_mixed_fraction <= 1 and 0 <= self.ema_alpha < 1
                and self.level_cooldown_steps > 0):
            raise ValueError("Invalid online fault controller parameters")
        self.step = 0
        self.buckets = {}

    def _bucket(self, source: str, extra_info: dict, tool: str | None = None) -> tuple[str, str] | None:
        tools = self.eligible.get(source)
        if not tools:
            return None
        tool = tool or tools[0]
        variant = "hme" if tool == "ocr_read" and extra_info.get("original_source") == "hme100k" else "standard"
        key = f"{source}|{tool}" + ("|hme" if variant == "hme" else "")
        if key not in self.buckets:
            self.buckets[key] = {"tool": tool, "fraction": self.initial_fraction,
                                 "level": 0, "mixed_ema": self.target_mixed_fraction,
                                 "last_level_step": 0}
        return key, variant

    def assign(self, batch) -> list[str]:
        specs = []
        rows = batch.non_tensor_batch
        for index in range(len(batch)):
            extra_info = rows["extra_info"][index]
            source = str(rows["data_source"][index])
            identity = extra_info.get("uid", extra_info.get("index", index))
            tools = self.eligible.get(source)
            tool = random.Random(f"{self.step}:{identity}:{index}:{source}").choice(tools) if tools else None
            bucket = self._bucket(source, extra_info, tool)
            if bucket is None:
                specs.append("")
                continue
            key, variant = bucket
            rng = random.Random(f"{self.step}:{identity}:{index}:{key}")
            if rng.random() >= self.buckets[key]["fraction"]:
                specs.append("")
                continue
            answer = rows["reward_model"][index]["ground_truth"]
            aliases = extra_info.get("answer_aliases")
            aliases = [aliases] if isinstance(aliases, str) else list(aliases) if aliases is not None else []
            specs.append(json.dumps({"tool": self.buckets[key]["tool"],
                                     "level": self.buckets[key]["level"],
                                     "seed": rng.getrandbits(64), "answer": str(answer),
                                     "aliases": aliases, "count_truth": str(answer),
                                     "variant": variant, "bucket_key": key}))
        return specs

    def update(self, batch) -> dict[str, float]:
        rows = batch.non_tensor_batch
        groups = defaultdict(list)
        for index, uid in enumerate(rows["uid"]):
            groups[str(uid)].append(index)
        counts = defaultdict(lambda: {"assigned": 0, "injected": 0, "mixed": 0,
                                     "all_correct": 0, "all_wrong": 0})
        for indices in groups.values():
            if len(indices) != 16:
                raise ValueError("Online CRT requires 16 rollouts per prompt")
            spec_text = rows["online_fault"][indices[0]]
            if not spec_text:
                continue
            key = json.loads(spec_text)["bucket_key"]
            counts[key]["assigned"] += 1
            if not any(any("injected_fault" in call for call in rows["rollout_trace"][index]["tool_calls"])
                       for index in indices):
                continue
            counts[key]["injected"] += 1
            correct = [float(rows["acc"][index]) > 0 for index in indices]
            outcome = "all_correct" if all(correct) else "all_wrong" if not any(correct) else "mixed"
            counts[key][outcome] += 1

        self.step += 1
        for key, values in counts.items():
            if not values["injected"]:
                continue
            state = self.buckets[key]
            mixed_rate = values["mixed"] / values["injected"]
            state["mixed_ema"] = self.ema_alpha * state["mixed_ema"] + (1 - self.ema_alpha) * mixed_rate
            if self.step - state["last_level_step"] >= self.level_cooldown_steps:
                if values["all_wrong"] > values["injected"] / 2:
                    state["level"] = min(MAX_LEVEL[state["tool"]], state["level"] + 1)
                    state["last_level_step"] = self.step
                elif values["all_correct"] > values["injected"] / 2:
                    state["level"] = max(0, state["level"] - 1)
                    state["last_level_step"] = self.step

        if self.buckets and any(values["injected"] for values in counts.values()):
            weights = {key: max(0.1, state["mixed_ema"] / self.target_mixed_fraction)
                       for key, state in self.buckets.items()}
            remaining = len(weights) * (self.initial_fraction - self.minimum_fraction)
            total_weight = sum(weights.values())
            for key, state in self.buckets.items():
                state["fraction"] = self.minimum_fraction + remaining * weights[key] / total_weight

        metrics = {}
        for key, state in self.buckets.items():
            prefix = f"online_faults/{key.replace('|', '/')}"
            values = counts[key]
            injected = values["injected"]
            metrics.update({
                f"{prefix}/assigned_groups": float(values["assigned"]),
                f"{prefix}/injected_groups": float(injected),
                f"{prefix}/mixed_rate": values["mixed"] / injected if injected else 0.0,
                f"{prefix}/all_correct_rate": values["all_correct"] / injected if injected else 0.0,
                f"{prefix}/all_wrong_rate": values["all_wrong"] / injected if injected else 0.0,
                f"{prefix}/mixed_ema": state["mixed_ema"],
                f"{prefix}/fraction": state["fraction"],
                f"{prefix}/level": float(state["level"]),
            })
        return metrics

    def state_dict(self) -> dict:
        return {"step": self.step, "buckets": self.buckets}

    def load_state_dict(self, state: dict) -> None:
        self.step = int(state["step"])
        self.buckets = state["buckets"]
