from __future__ import annotations

import logging

from core.config.settings import NotificationSettings
from core.interfaces.notifier import Notifier
from infrastructure.notifications.notifiers import CompositeNotifier, ConsoleNotifier, SilentNotifier, TelegramNotifier

logger = logging.getLogger(__name__)


def build_notifier(settings: NotificationSettings | None) -> Notifier:
    if settings is None:
        return SilentNotifier()

    channels = tuple(channel.strip().lower() for channel in settings.channels if channel.strip())
    if not channels or "silent" in channels:
        return SilentNotifier()

    notifiers: list[Notifier] = []
    for channel in channels:
        if channel == "console":
            notifiers.append(ConsoleNotifier())
        elif channel == "telegram":
            if settings.telegram_bot_token and settings.telegram_chat_id:
                notifiers.append(
                    TelegramNotifier(
                        bot_token=settings.telegram_bot_token,
                        chat_id=settings.telegram_chat_id,
                    )
                )
            else:
                logger.warning("Telegram notifier skipped: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is empty")
        else:
            logger.warning("Unknown notifier channel skipped: %s", channel)

    if not notifiers:
        return SilentNotifier()
    if len(notifiers) == 1:
        return notifiers[0]
    return CompositeNotifier(notifiers)
