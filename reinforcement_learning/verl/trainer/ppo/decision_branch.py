"""Hierarchical advantages for counterfactual first-decision branches.

For each prompt x and first decision d (a tool, or "direct"):
  U(x, d) = mean reward of the prompt's trajectories whose first decision is d
  V(x)    = sum_d pi(d|x) U(x, d), with pi from the forced branches' decision log-probs
Decision tokens receive (U(x, d) - V(x)); all other trainable tokens receive
(R - U(x, d)), i.e. execution quality given the decision. Both are divided by the
prompt's reward std, as in GRPO.
"""

from collections import defaultdict
import math

import numpy as np
import torch


def compute_decision_branch_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    decision_mask: torch.Tensor,
    old_log_probs: torch.Tensor,
    index: np.ndarray,
    decision_ids: np.ndarray,
    decision_forced: np.ndarray,
    decisions: list[str],
    reward_valid_mask=None,
    norm_adv_by_std: bool = True,
    forced_positive_weight: float = 1.0,
    forced_negative_weight: float = 0.0,
    decision_weight: float = 1.0,
    epsilon: float = 1e-6,
):
    """Return per-token advantages and diagnostic metrics.

    Forced decisions are not sampled from the policy. A positive forced decision is
    reinforced with ``forced_positive_weight`` (an off-policy update that lets a
    collapsed decision regain probability); a negative one uses
    ``forced_negative_weight``. Sampled decisions use their advantage unchanged.
    """
    if not decisions:
        raise ValueError("decision_branch advantages require forced branches in the training batch")
    # Per-sample scalars use float64 so U - V stays zero when all rewards in a group are
    # equal; in float32 sum(pi) != 1 leaves a residue that the epsilon scale amplifies.
    # Token-level tensors stay in their original dtype.
    scores = token_level_rewards.sum(dim=-1).double()
    valid = response_mask.any(dim=-1).cpu().numpy()
    if reward_valid_mask is not None:
        valid &= np.asarray(reward_valid_mask, dtype=bool)
    decision_logp = (old_log_probs * decision_mask).sum(dim=-1).double()
    has_span = decision_mask.any(dim=-1).cpu().numpy()

    decision_adv = torch.zeros_like(scores)
    execution_adv = torch.zeros_like(scores)
    groups = defaultdict(list)
    for row, uid in enumerate(index):
        groups[uid].append(row)

    entropies, policies, utilities, oracle_gains, values, oracle_values = [], [], defaultdict(list), [], [], []
    singleton_rows = forced_positive = forced_total = 0
    for rows in groups.values():
        rows = [row for row in rows if valid[row]]
        if not rows:
            continue
        group_scores = scores[rows]
        scale = group_scores.std() + epsilon if norm_adv_by_std and len(rows) > 1 else 1.0
        by_decision = defaultdict(list)
        for row in rows:
            by_decision[decision_ids[row]].append(row)
        utility = {decision: scores[members].mean() for decision, members in by_decision.items()}

        # Every decision is forced at least once, so pi(d|x) is read from the
        # policy's log-probability of each canonical decision prefix.
        log_policy = {}
        for row in rows:
            if decision_forced[row] and has_span[row]:
                log_policy.setdefault(decision_ids[row], decision_logp[row].item())
        if set(log_policy) == set(decisions) and all(decision in utility for decision in decisions):
            logits = torch.tensor([log_policy[decision] for decision in decisions], dtype=torch.float64)
            policy = torch.softmax(logits, dim=0)
            value = sum(policy[i] * utility[decision] for i, decision in enumerate(decisions))
            entropies.append(-(policy * policy.clamp_min(1e-12).log()).sum().item())
            policies.append(policy)
            best = max(utility[decision] for decision in decisions)
            oracle_gains.append((best - value).item())
            values.append(float(value))
            oracle_values.append(float(best))
        else:
            value = group_scores.mean()
        for decision in decisions:
            if decision in utility:
                utilities[decision].append(float(utility[decision]))

        for row in rows:
            decision = decision_ids[row]
            singleton_rows += len(by_decision[decision]) == 1
            execution_adv[row] = (scores[row] - utility[decision]) / scale
            advantage = (utility[decision] - value) / scale
            if decision_forced[row]:
                positive = bool(advantage > 0)
                forced_total += 1
                forced_positive += positive
                advantage = advantage * (forced_positive_weight if positive else forced_negative_weight)
            decision_adv[row] = advantage * decision_weight

    # Decision tokens lie inside response_mask, so they get decision_adv and every other
    # trainable token gets execution_adv.
    dtype = token_level_rewards.dtype
    execution_adv, decision_adv = execution_adv.to(dtype).unsqueeze(-1), decision_adv.to(dtype).unsqueeze(-1)
    advantages = execution_adv * response_mask + (decision_adv - execution_adv) * decision_mask

    valid_rows = int(valid.sum())
    eligible = [row for row in range(len(index)) if valid[row] and decision_ids[row] in decisions]
    metrics = {
        "decision_branch/pi_coverage_rate": len(policies) / max(len(groups), 1),
        "decision_branch/span_missing_rate": float(np.mean([not has_span[row] for row in eligible])) if eligible else 0.0,
        "decision_branch/singleton_decision_rate": singleton_rows / max(valid_rows, 1),
        "decision_branch/forced_positive_rate": forced_positive / max(forced_total, 1),
    }
    if policies:
        mean_policy = torch.stack(policies).mean(dim=0)
        marginal_entropy = -(mean_policy * mean_policy.clamp_min(1e-12).log()).sum().item()
        metrics["decision_branch/entropy_mean"] = float(np.mean(entropies))
        # I(X;D) = H(E_x pi) - E_x H(pi(.|x)), in nats.
        metrics["decision_branch/mutual_information"] = marginal_entropy - float(np.mean(entropies))
        metrics["decision_branch/max_entropy"] = math.log(len(decisions))
        metrics["decision_branch/value_mean"] = float(np.mean(values))
        metrics["decision_branch/oracle_value_mean"] = float(np.mean(oracle_values))
        metrics["decision_branch/oracle_gain_mean"] = float(np.mean(oracle_gains))
        for i, decision in enumerate(decisions):
            metrics[f"decision_branch/pi/{decision}"] = mean_policy[i].item()
    sampled = [row for row in range(len(index)) if valid[row] and not decision_forced[row]]
    for decision in decisions:
        if utilities[decision]:
            metrics[f"decision_branch/utility/{decision}"] = float(np.mean(utilities[decision]))
        if sampled:
            metrics[f"decision_branch/sampled_rate/{decision}"] = float(
                np.mean([decision_ids[row] == decision for row in sampled])
            )
    return advantages, advantages, metrics
