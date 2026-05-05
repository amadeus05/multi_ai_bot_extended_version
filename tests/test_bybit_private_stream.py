import hashlib
import hmac
import json

from infrastructure.exchanges.bybit.bybit_private_stream import (
    BYBIT_PRIVATE_WS_TESTNET_URL,
    BybitPrivateStream,
)


class FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    def send(self, message: str) -> None:
        self.sent.append(json.loads(message))

    def close(self) -> None:
        return None


def test_auth_payload_uses_private_ws_signature(monkeypatch) -> None:
    monkeypatch.setattr("time.time", lambda: 1000.0)
    stream = BybitPrivateStream(
        "key",
        "secret",
        testnet=True,
        websocket_app_factory=lambda *args, **kwargs: None,
    )

    payload = stream.auth_payload()

    expires = 1001000
    expected = hmac.new(
        b"secret",
        f"GET/realtime{expires}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    assert stream.url == BYBIT_PRIVATE_WS_TESTNET_URL
    assert payload == {"op": "auth", "args": ["key", expires, expected]}


def test_private_stream_auth_success_subscribes_default_topics() -> None:
    ws = FakeWebSocket()
    stream = BybitPrivateStream(
        "key",
        "secret",
        websocket_app_factory=lambda *args, **kwargs: None,
        ping_interval_sec=999,
    )
    stream._ws = ws

    stream._on_message(ws, json.dumps({"success": True, "op": "auth"}))

    assert ws.sent == [{"op": "subscribe", "args": ["execution", "order", "position"]}]
    assert stream.wait_until_authenticated(timeout=0.01) is True


def test_private_stream_queues_topic_payloads() -> None:
    ws = FakeWebSocket()
    stream = BybitPrivateStream(
        "key",
        "secret",
        websocket_app_factory=lambda *args, **kwargs: None,
    )

    payload = {"topic": "execution", "data": [{"execId": "exec-1"}]}
    stream._on_message(ws, json.dumps(payload))

    message = stream.get_message(timeout=0.01)
    assert message.topic == "execution"
    assert message.payload == payload
