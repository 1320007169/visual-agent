"""Adaptive allocation of online counterfactual tool faults."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path


MAX_LEVEL = {"ocr_read": 2, "object_count": 3, "grounding_detect": 2}
# A group says how hard a fault is only through rollouts that actually saw it.
MIN_FAULTED_ROLLOUTS = 4


def _count_truth(value) -> int | None:
    """Ground truth as a non-negative integer count, or None when it is not one."""
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() and number >= 0 else None


def _empty_pending() -> dict:
    return {"groups": 0, "all_correct": 0, "all_wrong": 0}


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
        self.min_level_groups = int(config["min_level_groups"])
        self.level_cooldown_steps = int(config["level_cooldown_steps"])
        if not (0 < self.minimum_fraction <= self.initial_fraction <= 1
                and 0 < self.target_mixed_fraction <= 1 and 0 <= self.ema_alpha < 1
                and self.min_level_groups > 0 and self.level_cooldown_steps > 0):
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
                                 "last_level_step": 0, "pending": _empty_pending()}
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
            answer = rows["reward_model"][index]["ground_truth"]
            truth = _count_truth(answer)
            if self.buckets[key]["tool"] == "object_count" and truth is None:
                specs.append("")
                continue
            rng = random.Random(f"{self.step}:{identity}:{index}:{key}")
            if rng.random() >= self.buckets[key]["fraction"]:
                specs.append("")
                continue
            aliases = extra_info.get("answer_aliases")
            aliases = [aliases] if isinstance(aliases, str) else list(aliases) if aliases is not None else []
            specs.append(json.dumps({"tool": self.buckets[key]["tool"],
                                     "level": self.buckets[key]["level"],
                                     "seed": rng.getrandbits(64), "answer": str(answer),
                                     "aliases": aliases,
                                     "count_truth": str(truth) if truth is not None else str(answer),
                                     "variant": variant, "bucket_key": key}))
        return specs

    def update(self, batch, rollout_n: int) -> dict[str, float]:
        rows = batch.non_tensor_batch
        reward_valid = rows.get("reward_valid", [True] * len(batch))
        groups = defaultdict(list)
        for index, uid in enumerate(rows["uid"]):
            groups[str(uid)].append(index)
        counts = defaultdict(lambda: {"assigned": 0, "injected": 0, "qualified": 0, "mixed": 0,
                                      "all_correct": 0, "all_wrong": 0, "coverage": [], "levels": []})
        for indices in groups.values():
            if len(indices) != rollout_n:
                raise ValueError(f"Online CRT expects rollout.n={rollout_n} rollouts per prompt, got {len(indices)}")
            spec_text = rows["online_fault"][indices[0]]
            if not spec_text:
                continue
            values = counts[json.loads(spec_text)["bucket_key"]]
            values["assigned"] += 1
            injected = [index for index in indices if rows["online_fault_injected"][index]]
            coverage = len(injected) / rollout_n
            values["coverage"].append(coverage)
            values["levels"].extend(int(rows["online_fault_level"][index]) for index in injected)
            values["injected"] += bool(injected)
            if any(not reward_valid[index] for index in injected):
                continue
            if len(injected) < MIN_FAULTED_ROLLOUTS:
                continue
            values["qualified"] += 1
            correct = [float(rows["acc"][index]) > 0 for index in injected]
            outcome = "all_correct" if all(correct) else "all_wrong" if not any(correct) else "mixed"
            values[outcome] += 1

        self.step += 1
        for key, values in counts.items():
            print(f"[online_faults] step={self.step} bucket={key} "
                  f"group_coverage={[round(value, 3) for value in values['coverage']]}", flush=True)
            if not values["qualified"]:
                continue
            state = self.buckets[key]
            mixed_rate = values["mixed"] / values["qualified"]
            state["mixed_ema"] = self.ema_alpha * state["mixed_ema"] + (1 - self.ema_alpha) * mixed_rate
            # Evaluate fresh evidence windows so old results at a level bound cannot block adaptation.
            pending = state["pending"]
            pending["groups"] += values["qualified"]
            pending["all_correct"] += values["all_correct"]
            pending["all_wrong"] += values["all_wrong"]
            # A level is evaluated only with enough groups and after a cooldown since the
            # last change, so one bucket cannot jump levels on consecutive steps.
            if (pending["groups"] < self.min_level_groups
                    or self.step - state["last_level_step"] < self.level_cooldown_steps):
                continue
            level = state["level"]
            if pending["all_wrong"] > pending["groups"] / 2:
                level = min(MAX_LEVEL[state["tool"]], level + 1)
            elif pending["all_correct"] > pending["groups"] / 2:
                level = max(0, level - 1)
            if level != state["level"]:
                state["level"] = level
                state["last_level_step"] = self.step
            state["pending"] = _empty_pending()

        if self.buckets and any(values["qualified"] for values in counts.values()):
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
            qualified = values["qualified"]
            coverage = values["coverage"]
            metrics.update({
                f"{prefix}/assigned_groups": float(values["assigned"]),
                f"{prefix}/injected_groups": float(values["injected"]),
                f"{prefix}/qualified_groups": float(qualified),
                f"{prefix}/coverage_mean": sum(coverage) / len(coverage) if coverage else 0.0,
                f"{prefix}/coverage_min": min(coverage) if coverage else 0.0,
                f"{prefix}/injected_level_mean": (sum(values["levels"]) / len(values["levels"])
                                                  if values["levels"] else -1.0),
                f"{prefix}/mixed_rate": values["mixed"] / qualified if qualified else 0.0,
                f"{prefix}/all_correct_rate": values["all_correct"] / qualified if qualified else 0.0,
                f"{prefix}/all_wrong_rate": values["all_wrong"] / qualified if qualified else 0.0,
                f"{prefix}/pending_groups": float(state["pending"]["groups"]),
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
