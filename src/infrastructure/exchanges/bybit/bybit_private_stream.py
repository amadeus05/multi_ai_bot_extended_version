from __future__ import annotations

import hashlib
import hmac
import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

try:
    import websocket
except ImportError:  # pragma: no cover - handled by caller environment
    websocket = None


logger = logging.getLogger(__name__)


BYBIT_PRIVATE_WS_MAINNET_URL = "wss://stream.bybit.com/v5/private"
BYBIT_PRIVATE_WS_TESTNET_URL = "wss://stream-testnet.bybit.com/v5/private"


@dataclass(frozen=True)
class BybitPrivateStreamMessage:
    topic: str
    payload: dict[str, Any]


class BybitPrivateStream:
    """Private Bybit websocket transport for order/execution/position topics."""

    DEFAULT_TOPICS = frozenset({"order", "execution", "position"})

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        *,
        testnet: bool = False,
        url: str | None = None,
        topics: set[str] | frozenset[str] | None = None,
        ping_interval_sec: float = 20.0,
        reconnect_sleep_sec: float = 3.0,
        auth_expires_sec: float = 1.0,
        websocket_app_factory: Callable[..., Any] | None = None,
    ) -> None:
        if websocket is None and websocket_app_factory is None:
            raise ImportError("Install websocket-client to use BybitPrivateStream")

        self.api_key = str(api_key or "")
        self.api_secret = str(api_secret or "")
        self.url = url or (BYBIT_PRIVATE_WS_TESTNET_URL if testnet else BYBIT_PRIVATE_WS_MAINNET_URL)
        self.ping_interval_sec = float(ping_interval_sec)
        self.reconnect_sleep_sec = float(reconnect_sleep_sec)
        self.auth_expires_sec = float(auth_expires_sec)
        self._websocket_app_factory = websocket_app_factory or websocket.WebSocketApp

        self._queue: queue.Queue[BybitPrivateStreamMessage] = queue.Queue()
        self._ws = None
        self._thread: threading.Thread | None = None
        self._ping_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._connected_event = threading.Event()
        self._authenticated_event = threading.Event()
        self._lock = threading.Lock()
        self._desired_topics: set[str] = set(topics or self.DEFAULT_TOPICS)
        self._subscribed_topics: set[str] = set()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_forever, daemon=True, name="bybit-private-stream")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._connected_event.clear()
        self._authenticated_event.clear()
        ws = self._ws
        if ws is not None:
            try:
                ws.close()
            except Exception:
                logger.exception("Failed to close Bybit private websocket cleanly")

    def get_message(self, timeout: float | None = None) -> BybitPrivateStreamMessage:
        return self._queue.get(timeout=timeout)

    def wait_until_connected(self, timeout: float = 15.0) -> bool:
        return self._connected_event.wait(timeout=timeout)

    def wait_until_authenticated(self, timeout: float = 15.0) -> bool:
        return self._authenticated_event.wait(timeout=timeout)

    def sync_topics(self, topics: set[str]) -> None:
        with self._lock:
            self._desired_topics = set(topics)
            to_subscribe = self._desired_topics - self._subscribed_topics
            to_unsubscribe = self._subscribed_topics - self._desired_topics

        if not self._authenticated_event.is_set():
            logger.info("Queued private WS topic sync | desired=%s", len(self._desired_topics))
            return
        if to_subscribe:
            self._send_command("subscribe", sorted(to_subscribe))
        if to_unsubscribe:
            self._send_command("unsubscribe", sorted(to_unsubscribe))

    def auth_payload(self) -> dict[str, Any]:
        expires = int((time.time() + self.auth_expires_sec) * 1000)
        signature_payload = f"GET/realtime{expires}"
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            signature_payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return {"op": "auth", "args": [self.api_key, expires, signature]}

    def _run_forever(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._run_once()
            except Exception:
                logger.exception("Bybit private websocket loop failed")
            if not self._stop_event.is_set():
                time.sleep(self.reconnect_sleep_sec)

    def _run_once(self) -> None:
        app = self._websocket_app_factory(
            self.url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self._ws = app
        app.run_forever()

    def _on_open(self, ws) -> None:
        logger.info("Bybit private WS connected: %s", self.url)
        self._connected_event.set()
        self._authenticated_event.clear()
        with self._lock:
            self._subscribed_topics.clear()
        self._send_raw(self.auth_payload())
        self._start_ping_loop()

    def _on_message(self, ws, message: str) -> None:
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            logger.warning("Invalid private WS JSON payload: %s", message[:200])
            return

        op = payload.get("op")
        if op == "auth":
            if payload.get("success") is True or payload.get("retCode") == 0:
                self._authenticated_event.set()
                with self._lock:
                    desired = sorted(self._desired_topics)
                if desired:
                    self._send_command("subscribe", desired)
            else:
                logger.error("Bybit private WS auth failed: %s", payload)
            return

        if op in {"ping", "pong"}:
            return

        if op in {"subscribe", "unsubscribe"}:
            success = payload.get("success", True)
            if not success:
                logger.error("Bybit private WS command failed: %s", payload)
            return

        topic = str(payload.get("topic") or "")
        if not topic:
            return
        self._queue.put(BybitPrivateStreamMessage(topic=topic, payload=payload))

    def _on_error(self, ws, error) -> None:
        if self._stop_event.is_set():
            return
        logger.warning("Bybit private WS error: %s", error)

    def _on_close(self, ws, status_code, close_msg) -> None:
        self._connected_event.clear()
        self._authenticated_event.clear()
        with self._lock:
            self._subscribed_topics.clear()
        if not self._stop_event.is_set():
            logger.warning("Bybit private WS closed | code=%s | msg=%s", status_code, close_msg)

    def _start_ping_loop(self) -> None:
        def ping_loop() -> None:
            while self._connected_event.is_set() and not self._stop_event.is_set():
                time.sleep(self.ping_interval_sec)
                if not self._connected_event.is_set() or self._stop_event.is_set():
                    break
                try:
                    self._send_raw({"op": "ping"})
                except Exception:
                    logger.exception("Failed to send Bybit private websocket ping")
                    break

        self._ping_thread = threading.Thread(target=ping_loop, daemon=True, name="bybit-private-ping")
        self._ping_thread.start()

    def _send_command(self, op: str, topics: list[str]) -> None:
        if not topics:
            return
        self._send_raw({"op": op, "args": topics})
        with self._lock:
            if op == "subscribe":
                self._subscribed_topics.update(topics)
            elif op == "unsubscribe":
                for topic in topics:
                    self._subscribed_topics.discard(topic)

    def _send_raw(self, payload: dict[str, Any]) -> None:
        ws = self._ws
        if ws is None:
            return
        ws.send(json.dumps(payload))
