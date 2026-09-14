from typing import Any


def error_response_status_code(error: Any, default: int = 400) -> int:
    """Extract an HTTP status from old and new vLLM ErrorResponse layouts."""
    payload = error.model_dump() if hasattr(error, "model_dump") else {}
    nested_error = getattr(error, "error", None)
    candidates = [
        getattr(error, "code", None),
        getattr(nested_error, "code", None),
        payload.get("code") if isinstance(payload, dict) else None,
    ]
    if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
        candidates.append(payload["error"].get("code"))

    for candidate in candidates:
        try:
            status_code = int(candidate)
        except (TypeError, ValueError):
            continue
        if 400 <= status_code <= 599:
            return status_code
    return default
