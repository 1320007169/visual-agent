from types import SimpleNamespace

from verl.workers.rollout.http_utils import error_response_status_code


class FakeErrorResponse:
    def __init__(self, payload, **attributes):
        self.payload = payload
        for name, value in attributes.items():
            setattr(self, name, value)

    def model_dump(self):
        return self.payload


def test_error_response_status_code_supports_legacy_layout():
    error = FakeErrorResponse({"code": 422}, code=422)
    assert error_response_status_code(error) == 422


def test_error_response_status_code_supports_nested_layout():
    error = FakeErrorResponse({"error": {"code": 429}}, error=SimpleNamespace(code=429))
    assert error_response_status_code(error) == 429


def test_error_response_status_code_defaults_for_non_http_code():
    error = FakeErrorResponse({"error": {"code": "BadRequestError"}})
    assert error_response_status_code(error) == 400
