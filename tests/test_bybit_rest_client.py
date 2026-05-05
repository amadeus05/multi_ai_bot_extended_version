import hashlib
import hmac

import pytest

from infrastructure.exchanges.bybit.bybit_rest_client import BybitRestClient, BybitRestError


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class FakeSession:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def get(self, url, *, params=None, headers=None, timeout=None):
        self.calls.append({"method": "GET", "url": url, "params": params, "headers": headers, "timeout": timeout})
        return FakeResponse(self.responses.pop(0))

    def post(self, url, *, data=None, headers=None, timeout=None):
        self.calls.append({"method": "POST", "url": url, "data": data, "headers": headers, "timeout": timeout})
        return FakeResponse(self.responses.pop(0))


def expected_hmac(secret: str, payload: str) -> str:
    return hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def test_signed_get_uses_bybit_headers_and_query_signature(monkeypatch) -> None:
    monkeypatch.setattr(BybitRestClient, "_local_timestamp_ms", staticmethod(lambda: 1658384314791))
    session = FakeSession([{"retCode": 0, "retMsg": "OK", "result": {}}])
    client = BybitRestClient(
        "key",
        "secret",
        testnet=True,
        recv_window_ms=5000,
        session=session,
    )

    client.get("/v5/order/realtime", {"category": "linear", "symbol": "BTCUSDT"}, request_name="status")

    call = session.calls[0]
    headers = call["headers"]
    query = "category=linear&symbol=BTCUSDT"
    assert call["url"] == "https://api-testnet.bybit.com/v5/order/realtime"
    assert headers["X-BAPI-API-KEY"] == "key"
    assert headers["X-BAPI-TIMESTAMP"] == "1658384314791"
    assert headers["X-BAPI-RECV-WINDOW"] == "5000"
    assert headers["X-BAPI-SIGN"] == expected_hmac("secret", "1658384314791key5000" + query)


def test_signed_post_sends_same_json_body_that_is_signed(monkeypatch) -> None:
    monkeypatch.setattr(BybitRestClient, "_local_timestamp_ms", staticmethod(lambda: 1658385579423))
    session = FakeSession([{"retCode": 0, "retMsg": "OK", "result": {"orderId": "1"}}])
    client = BybitRestClient("key", "secret", session=session)

    client.post("/v5/order/create", {"category": "linear", "qty": "1"}, request_name="order_create")

    call = session.calls[0]
    headers = call["headers"]
    assert call["data"] == '{"category":"linear","qty":"1"}'
    assert headers["Content-Type"] == "application/json"
    assert headers["X-BAPI-SIGN"] == expected_hmac(
        "secret",
        '1658385579423key5000{"category":"linear","qty":"1"}',
    )


def test_retries_retryable_ret_code_then_succeeds(monkeypatch) -> None:
    monkeypatch.setattr(BybitRestClient, "_local_timestamp_ms", staticmethod(lambda: 1))
    monkeypatch.setattr("time.sleep", lambda _: None)
    session = FakeSession(
        [
            {"retCode": 10006, "retMsg": "rate limit", "result": {}},
            {"retCode": 0, "retMsg": "OK", "result": {"ok": True}},
        ]
    )
    client = BybitRestClient("key", "secret", session=session, retry_count=2)

    payload = client.get("/v5/order/realtime", {"category": "linear"})

    assert payload["result"] == {"ok": True}
    assert len(session.calls) == 2


def test_non_retryable_ret_code_raises() -> None:
    session = FakeSession([{"retCode": 110001, "retMsg": "order not exists", "result": {}}])
    client = BybitRestClient("key", "secret", session=session)

    with pytest.raises(BybitRestError) as exc:
        client.get("/v5/order/realtime", {"category": "linear"})

    assert exc.value.ret_code == 110001
    assert "order not exists" in str(exc.value)
