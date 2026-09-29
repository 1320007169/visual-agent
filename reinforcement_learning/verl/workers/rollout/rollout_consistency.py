"""Compare vLLM-sampled assistant tokens with the re-encoded training tokens."""


def parse_sampled_logprobs(logprobs):
    """Return sampled token ids and log probabilities, or None when unavailable."""
    if logprobs is None or getattr(logprobs, "content", None) is None:
        return None
    ids, values = [], []
    for item in logprobs.content:
        if not item.token.startswith("token_id:"):
            return None
        ids.append(int(item.token[len("token_id:"):]))
        values.append(float(item.logprob))
    return ids, values


def template_turn_roles(messages):
    """Roles of chat-template turns; adjacent tool messages share one template turn."""
    roles = []
    for message in messages:
        role = message.get("role")
        if role == "tool" and roles and roles[-1] == "tool":
            continue
        roles.append(role)
    return roles


def assistant_spans(response_ids, roles, eos_id, header_ids):
    """Return [start, end) of each assistant turn's generated part, including its terminator."""
    eos = [index for index, token in enumerate(response_ids) if token == eos_id][: len(roles)]
    if len(eos) != len(roles):
        return None
    spans = []
    for turn, role in enumerate(roles):
        if role != "assistant":
            continue
        start = eos[turn - 1] + 1 if turn else 0
        # Later assistant turns begin with the template header, which vLLM did not sample.
        window = response_ids[start : start + len(header_ids) + 2]
        for offset in range(len(window) - len(header_ids) + 1):
            if window[offset : offset + len(header_ids)] == header_ids:
                start += offset + len(header_ids)
                break
        spans.append((start, eos[turn] + 1))
    return spans


def align_sampled_turns(response_ids, roles, eos_id, header_ids, sampled_turns):
    """Map sampled log probabilities onto response positions for exactly matching turns."""
    stats = {"turns": 0, "matched_turns": 0, "sampled_tokens": 0, "matched_tokens": 0, "unaligned": 0}
    spans = assistant_spans(response_ids, roles, eos_id, header_ids)
    if spans is None:
        stats["unaligned"] = 1
        return [], [], stats
    positions, values = [], []
    for (start, end), sampled in zip(spans, sampled_turns):
        stats["turns"] += 1
        if sampled is None:
            continue
        ids, logprobs = sampled
        trained = response_ids[start:end]
        stats["sampled_tokens"] += len(ids)
        # The template appends the terminator when vLLM stopped without sampling it.
        if ids == trained or (ids == trained[:-1] and trained[-1] == eos_id):
            stats["matched_turns"] += 1
            stats["matched_tokens"] += len(ids)
            positions.extend(range(start, start + len(ids)))
            values.extend(logprobs)
    return positions, values, stats
