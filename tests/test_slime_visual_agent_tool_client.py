from slime_visual_agent import tool_client


def test_least_busy_endpoint_spreads_calls_and_avoids_failed_endpoint():
    urls = ("http://node0:9000", "http://node0:9001", "http://node1:9000")
    original = dict(tool_client._PENDING_BY_ENDPOINT)
    try:
        tool_client._PENDING_BY_ENDPOINT.clear()
        tool_client._PENDING_BY_ENDPOINT.update(
            {
                "http://node0:9000": 4,
                "http://node0:9001": 1,
                "http://node1:9000": 2,
            }
        )
        assert tool_client._least_busy_endpoint(urls) == "http://node0:9001"
        assert tool_client._least_busy_endpoint(urls, "http://node0:9001") == "http://node1:9000"
    finally:
        tool_client._PENDING_BY_ENDPOINT.clear()
        tool_client._PENDING_BY_ENDPOINT.update(original)
