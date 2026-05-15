from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Iterable, TextIO

import requests

from core.interfaces.notifier import Notifier
from core.types.domain_types import Order
from core.types.notifications import SignalNotification, SystemNotification, TradeExitNotification
from infrastructure.notifications.formatters import (
    format_signal_html,
    format_signal_text,
    format_system_html,
    format_system_text,
    format_trade_exit_html,
    format_trade_exit_text,
)

logger = logging.getLogger(__name__)


class SilentNotifier(Notifier):
    async def notify_signal(self, notification: SignalNotification) -> None:
        return None

    async def notify_trade_exit(self, notification: TradeExitNotification) -> None:
        return None

    async def notify_system(self, notification: SystemNotification) -> None:
        return None

    async def notify_error(self, where: str, error: Exception, order: Order | None = None) -> None:
        return None


class ConsoleNotifier(Notifier):
    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream

    async def notify_signal(self, notification: SignalNotification) -> None:
        self._write(format_signal_text(notification))

    async def notify_trade_exit(self, notification: TradeExitNotification) -> None:
        self._write(format_trade_exit_text(notification))

    async def notify_system(self, notification: SystemNotification) -> None:
        self._write(format_system_text(notification))

    async def notify_error(self, where: str, error: Exception, order: Order | None = None) -> None:
        order_ref = f" | order={order.client_order_id or order.id or order.symbol}" if order is not None else ""
        self._write(
            format_system_text(
                SystemNotification(
                    level="error",
                    title="Runtime error",
                    message=order_ref.lstrip(" | ") or "Unhandled runtime error",
                    where=where,
                    error=f"{type(error).__name__}: {error}",
                )
            )
        )

    def _write(self, message: str) -> None:
        print(message, file=self._stream)
        print("", file=self._stream)


class TelegramNotifier(Notifier):
    def __init__(
        self,
        *,
        bot_token: str,
        chat_id: str,
        timeout: float = 10.0,
        fail_silently: bool = True,
    ) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._timeout = float(timeout)
        self._fail_silently = bool(fail_silently)

    async def notify_signal(self, notification: SignalNotification) -> None:
        text = format_signal_html(notification)
        if notification.chart_path is not None:
            await self._send_photo(text, notification.chart_path)
            return
        await self._send(text)

    async def notify_trade_exit(self, notification: TradeExitNotification) -> None:
        await self._send(format_trade_exit_html(notification))

    async def notify_system(self, notification: SystemNotification) -> None:
        await self._send(format_system_html(notification))

    async def notify_error(self, where: str, error: Exception, order: Order | None = None) -> None:
        order_msg = ""
        if order is not None:
            order_msg = f"Order: {order.client_order_id or order.id or order.symbol}"
        await self.notify_system(
            SystemNotification(
                level="error",
                title="Runtime error",
                message=order_msg or "Unhandled runtime error",
                where=where,
                error=f"{type(error).__name__}: {error}",
            )
        )

    async def _send(self, text: str) -> None:
        try:
            await asyncio.to_thread(self._post, text)
        except Exception:
            if not self._fail_silently:
                raise
            logger.exception("Telegram notification failed")

    async def _send_photo(self, caption: str, photo_path: Path) -> None:
        try:
            await asyncio.to_thread(self._post_photo, caption, photo_path)
        except Exception:
            if not self._fail_silently:
                raise
            logger.exception("Telegram photo notification failed")

    def _post(self, text: str) -> None:
        response = requests.post(
            f"https://api.telegram.org/bot{self._bot_token}/sendMessage",
            json={
                "chat_id": self._chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=self._timeout,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError:
            logger.error(
                "Telegram API error | status=%s | response=%s",
                response.status_code,
                response.text,
            )
            raise

    def _post_photo(self, caption: str, photo_path: Path) -> None:
        with Path(photo_path).open("rb") as photo:
            response = requests.post(
                f"https://api.telegram.org/bot{self._bot_token}/sendPhoto",
                data={
                    "chat_id": self._chat_id,
                    "caption": caption,
                    "parse_mode": "HTML",
                },
                files={"photo": photo},
                timeout=self._timeout,
            )
        try:
            response.raise_for_status()
        except requests.HTTPError:
            logger.error(
                "Telegram API error | status=%s | response=%s",
                response.status_code,
                response.text,
            )
            raise


class CompositeNotifier(Notifier):
    def __init__(self, notifiers: Iterable[Notifier]) -> None:
        self._notifiers = tuple(notifiers)

    async def notify_signal(self, notification: SignalNotification) -> None:
        await self._dispatch("notify_signal", notification)

    async def notify_trade_exit(self, notification: TradeExitNotification) -> None:
        await self._dispatch("notify_trade_exit", notification)

    async def notify_system(self, notification: SystemNotification) -> None:
        await self._dispatch("notify_system", notification)

    async def notify_error(self, where: str, error: Exception, order: Order | None = None) -> None:
        await asyncio.gather(
            *(self._call(notifier.notify_error(where, error, order)) for notifier in self._notifiers)
        )

    async def _dispatch(self, method_name: str, notification: object) -> None:
        await asyncio.gather(
            *(
                self._call(getattr(notifier, method_name)(notification))
                for notifier in self._notifiers
            )
        )

    async def _call(self, task) -> None:
        try:
            await task
        except Exception:
            logger.exception("Notification channel failed")
