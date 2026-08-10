import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter
from tools import send_message_tool


def test_cli_send_is_blocked_before_telegram_network_when_outbound_is_disabled(
    monkeypatch,
):
    pconfig = PlatformConfig(
        enabled=True,
        token="synthetic-token",
        extra={"outbound_enabled": False},
    )
    config = SimpleNamespace(
        platforms={Platform.TELEGRAM: pconfig},
        get_home_channel=lambda _platform: SimpleNamespace(chat_id="synthetic-chat"),
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
    network = AsyncMock()
    monkeypatch.setattr(send_message_tool, "_send_to_platform", network)

    result = json.loads(
        send_message_tool._handle_send(
            {"target": "telegram", "message": "synthetic outbound guard probe"}
        )
    )

    assert result == {
        "error": "Telegram outbound delivery is disabled for this profile."
    }
    network.assert_not_called()


@pytest.mark.asyncio
async def test_gateway_reply_is_blocked_when_telegram_outbound_is_disabled():
    adapter = TelegramAdapter(
        PlatformConfig(
            enabled=True,
            token="synthetic-token",
            extra={"outbound_enabled": False},
        )
    )
    adapter._bot = AsyncMock()

    result = await adapter.send("synthetic-chat", "synthetic reply")

    assert result.success is False
    assert result.error == "telegram_outbound_disabled"
    assert result.retryable is False
    adapter._bot.send_message.assert_not_called()
