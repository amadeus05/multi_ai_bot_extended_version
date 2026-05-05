from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import requests
from requests.adapters import HTTPAdapter


logger = logging.getLogger(__name__)


BYBIT_TESTNET_BASE_URL = "https://api-testnet.bybit.com"
BYBIT_MAINNET_BASE_URL = "https://api.bybit.com"


@dataclass(frozen=True)
class BybitRestError(RuntimeError):
    request_name: str
    message: str
    ret_code: int | None = None
    ret_msg: str | None = None
    payload: dict[str, Any] | None = None

    def __str__(self) -> str:
        if self.ret_code is None:
            return f"{self.request_name}: {self.message}"
        return f"{self.request_name}: retCode={self.ret_code} retMsg={self.ret_msg or self.message}"


class BybitRestClient:
    """Small signed REST client for Bybit V5.

    The client owns HTTP details only: base URL, HMAC signing, timestamp
    offset, retry policy, and retCode handling. Higher-level order semantics
    live in BybitExecutionExchange and mapper classes.
    """

    _RETRYABLE_RET_CODES = {10000, 10006, 10016}

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        *,
        testnet: bool = False,
        base_url: str | None = None,
        recv_window_ms: int = 5000,
        timeout: float = 10.0,
        retry_count: int = 3,
        retry_sleep: float = 0.25,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = str(api_key or "")
        self.api_secret = str(api_secret or "")
        self.testnet = bool(testnet)
        self.base_url = (base_url or (BYBIT_TESTNET_BASE_URL if self.testnet else BYBIT_MAINNET_BASE_URL)).rstrip("/")
        self.recv_window_ms = int(recv_window_ms)
        self.timeout = float(timeout)
        self.retry_count = max(1, int(retry_count))
        self.retry_sleep = float(retry_sleep)
        self._session = session or self._build_session()
        self._server_time_offset_ms = 0

    @staticmethod
    def _build_session() -> requests.Session:
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=8, pool_maxsize=8)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update({"User-Agent": "mlV2-bybit-execution/1.0"})
        return session

    def sync_server_time(self) -> int:
        payload = self.get("/v5/market/time", signed=False, request_name="server_time")
        server_ms = self._extract_server_time_ms(payload)
        local_ms = self._local_timestamp_ms()
        self._server_time_offset_ms = server_ms - local_ms
        logger.info("Bybit server time synced | offset_ms=%s", self._server_time_offset_ms)
        return self._server_time_offset_ms

    def get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        signed: bool = True,
        request_name: str | None = None,
    ) -> dict[str, Any]:
        return self._request("GET", path, params=params or {}, body=None, signed=signed, request_name=request_name)

    def post(
        self,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        signed: bool = True,
        request_name: str | None = None,
    ) -> dict[str, Any]:
        return self._request("POST", path, params={}, body=body or {}, signed=signed, request_name=request_name)

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any],
        body: dict[str, Any] | None,
        signed: bool,
        request_name: str | None,
    ) -> dict[str, Any]:
        method = method.upper()
        name = request_name or f"{method} {path}"
        url = self._url(path)
        query_string = self._query_string(params)
        body_string = self._body_string(body) if body is not None else ""
        headers = self._headers(method, query_string, body_string) if signed else {}
        if body is not None:
            headers["Content-Type"] = "application/json"

        last_error: Exception | None = None
        for attempt in range(self.retry_count):
            try:
                response = self._send(method, url, params, body_string, headers)
                response.raise_for_status()
                payload = response.json()
                self._ensure_success(payload, name)
                logger.info(
                    "Bybit REST result | request=%s | retCode=%s | attempt=%s",
                    name,
                    payload.get("retCode"),
                    attempt + 1,
                )
                return payload
            except BybitRestError as exc:
                last_error = exc
                if exc.ret_code not in self._RETRYABLE_RET_CODES:
                    logger.warning("Bybit REST failed | request=%s | error=%s", name, exc)
                    raise
            except (requests.RequestException, ValueError) as exc:
                last_error = exc

            if attempt + 1 < self.retry_count:
                sleep_s = self.retry_sleep * (2**attempt)
                logger.warning(
                    "Bybit REST retry | request=%s | attempt=%s/%s | error=%s",
                    name,
                    attempt + 1,
                    self.retry_count,
                    last_error,
                )
                time.sleep(sleep_s)

        raise BybitRestError(name, str(last_error or "request failed")) from last_error

    def _send(
        self,
        method: str,
        url: str,
        params: dict[str, Any],
        body_string: str,
        headers: dict[str, str],
    ) -> requests.Response:
        if method == "GET":
            return self._session.get(url, params=params, headers=headers, timeout=self.timeout)
        if method == "POST":
            return self._session.post(url, data=body_string, headers=headers, timeout=self.timeout)
        raise ValueError(f"Unsupported Bybit REST method: {method}")

    def _headers(self, method: str, query_string: str, body_string: str) -> dict[str, str]:
        if not self.api_key or not self.api_secret:
            raise BybitRestError(method, "missing Bybit API key/secret")
        timestamp = str(self.timestamp_ms())
        recv_window = str(self.recv_window_ms)
        sign_payload = timestamp + self.api_key + recv_window + (query_string if method == "GET" else body_string)
        return {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
            "X-BAPI-SIGN": self.sign(sign_payload),
        }

    def sign(self, payload: str) -> str:
        return hmac.new(
            self.api_secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def timestamp_ms(self) -> int:
        return self._local_timestamp_ms() + self._server_time_offset_ms

    @staticmethod
    def _local_timestamp_ms() -> int:
        return int(time.time() * 1000)

    @staticmethod
    def _query_string(params: dict[str, Any]) -> str:
        clean = {key: value for key, value in params.items() if value is not None}
        return urlencode(clean, doseq=True)

    @staticmethod
    def _body_string(body: dict[str, Any]) -> str:
        return json.dumps(body, separators=(",", ":"), ensure_ascii=False)

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return self.base_url + path

    @staticmethod
    def _ensure_success(payload: dict[str, Any], request_name: str) -> None:
        ret_code = payload.get("retCode")
        if ret_code == 0:
            return
        raise BybitRestError(
            request_name=request_name,
            message=str(payload.get("retMsg") or "Bybit request failed"),
            ret_code=int(ret_code) if ret_code is not None else None,
            ret_msg=str(payload.get("retMsg") or ""),
            payload=payload,
        )

    @staticmethod
    def _extract_server_time_ms(payload: dict[str, Any]) -> int:
        result = payload.get("result") or {}
        for key in ("timeNano", "timeSecond"):
            value = result.get(key)
            if value in (None, ""):
                continue
            if key == "timeNano":
                return int(int(value) / 1_000_000)
            return int(float(value) * 1000)
        if payload.get("time") is not None:
            return int(payload["time"])
        raise BybitRestError("server_time", "server time is missing from response", payload=payload)
