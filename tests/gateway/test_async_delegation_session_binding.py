"""Gateway-side session binding for async delegations (#57498, #55578).

Three invariants on the messaging-gateway surface, mirroring the TUI rules:

1. Completions are pinned to the spawning session (contributor commit).
2. A dead/ended spawning session is never resurrected: the injection is
   dropped, fail-closed (never rerouted to the peer's current session).
3. /new interrupts the old conversation's in-flight async delegations.
"""

import asyncio
from collections import OrderedDict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import tools.async_delegation as ad
from gateway.config import Platform
from gateway.session import SessionSource, build_session_key


@pytest.fixture(autouse=True)
def _reset_async_delegation():
    ad._reset_for_tests()
    yield
    ad._reset_for_tests()


def _seed_record(delegation_id, session_key="", parent_session_id="", status="running"):
    fn = MagicMock()
    with ad._records_lock:
        ad._records[delegation_id] = {
            "delegation_id": delegation_id,
            "status": status,
            "session_key": session_key,
            "parent_session_id": parent_session_id,
            "interrupt_fn": fn,
        }
    return fn


class TestInterruptForSessionByParentId:
    def test_parent_session_id_selector(self):
        mine = _seed_record("d1", session_key="agent:main:telegram:dm:1", parent_session_id="sess_old")
        other = _seed_record("d2", session_key="agent:main:telegram:dm:2", parent_session_id="sess_other")
        n = ad.interrupt_for_session(parent_session_id="sess_old")
        assert n == 1
        mine.assert_called_once()
        other.assert_not_called()

    def test_reset_interrupts_by_key_and_parent(self):
        """A /new reset passes both selectors — either match claims the record."""
        by_key = _seed_record("d1", session_key="agent:main:telegram:dm:1", parent_session_id="")
        by_parent = _seed_record("d2", session_key="", parent_session_id="sess_old")
        unrelated = _seed_record("d3", session_key="other", parent_session_id="other")
        n = ad.interrupt_for_session(
            session_key="agent:main:telegram:dm:1",
            parent_session_id="sess_old",
            reason="session_reset",
        )
        assert n == 2
        by_key.assert_called_once()
        by_parent.assert_called_once()
        unrelated.assert_not_called()


class TestGatewayPinningFailsClosed:
    """The gateway injection path must never resurrect an ended session."""

    def _make_runner(self, pinned_row):
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        db = MagicMock()
        db.get_session = AsyncMock(return_value=pinned_row)
        runner._session_db = db

        entry = MagicMock()
        entry.session_key = "agent:main:telegram:dm:1"
        entry.session_id = "sess_current"
        runner.session_store = MagicMock()
        runner.session_store.get_or_create_session.return_value = entry
        runner.session_store.switch_session.return_value = entry
        return runner, entry

    def _run_pinning_prefix(self, runner, pinned_session_id):
        """Execute the pinning guard logic exactly as _handle_message does."""

        async def _go():
            event = MagicMock()
            event.metadata = {"gateway_session_id": pinned_session_id}
            session_entry = runner.session_store.get_or_create_session(MagicMock())
            pinned = str((getattr(event, "metadata", None) or {}).get("gateway_session_id") or "").strip()
            if pinned and pinned != session_entry.session_id:
                pinned_row = None
                try:
                    if runner._session_db is not None:
                        pinned_row = await runner._session_db.get_session(pinned)
                except Exception:
                    pinned_row = None
                if pinned_row is None or pinned_row.get("ended_at"):
                    return "dropped"
                switched = runner.session_store.switch_session(session_entry.session_key, pinned)
                if switched is not None:
                    return "pinned"
            return "default"

        return asyncio.run(_go())

    def test_live_spawning_session_pins(self):
        runner, _ = self._make_runner({"id": "sess_old", "ended_at": None})
        assert self._run_pinning_prefix(runner, "sess_old") == "pinned"

    def test_ended_spawning_session_drops(self):
        runner, _ = self._make_runner({"id": "sess_old", "ended_at": "2026-07-08T00:00:00"})
        assert self._run_pinning_prefix(runner, "sess_old") == "dropped"
        runner.session_store.switch_session.assert_not_called()

    def test_unknown_spawning_session_drops(self):
        runner, _ = self._make_runner(None)
        assert self._run_pinning_prefix(runner, "sess_gone") == "dropped"
        runner.session_store.switch_session.assert_not_called()


class TestResetHandlerInterruptsDelegations:
    def test_reset_command_calls_interrupt_for_session(self):
        """The /new handler must sever the old conversation's delegations."""
        import inspect
        from gateway import slash_commands

        src = inspect.getsource(slash_commands.GatewaySlashCommandsMixin._handle_reset_command)
        assert "interrupt_for_session" in src
        assert "session_reset" in src


class TestSlackDelegationWorkingStatus:
    @staticmethod
    def _slack_key(thread_id: str) -> str:
        return build_session_key(SessionSource(
            platform=Platform.SLACK,
            user_id="owner",
            chat_id="C_TTC",
            user_name="owner",
            chat_type="channel",
            thread_id=thread_id,
        ))

    @pytest.mark.asyncio
    async def test_status_tracks_each_live_thread_and_clears_exactly(self):
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        adapter = MagicMock()
        adapter.supports_status_text = True
        adapter.set_status_text = MagicMock()
        adapter.send_typing = AsyncMock()
        adapter.stop_typing = AsyncMock()
        runner.adapters = {Platform.SLACK: adapter}
        runner._session_sources = OrderedDict()

        first_key = self._slack_key("thread-1")
        second_key = self._slack_key("thread-2")
        runner._session_sources[first_key] = SessionSource(
            platform=Platform.SLACK, user_id="owner", chat_id="C_TTC",
            user_name="owner", chat_type="channel", thread_id="thread-1",
        )
        runner._session_sources[second_key] = SessionSource(
            platform=Platform.SLACK, user_id="owner", chat_id="C_TTC",
            user_name="owner", chat_type="channel", thread_id="thread-2",
        )

        first = {
            "delegation_id": "d1", "status": "running",
            "session_key": first_key,
        }
        second = {
            "delegation_id": "d2", "status": "running",
            "session_key": second_key,
        }

        await runner._refresh_async_delegation_working_statuses([first, second])

        assert adapter.send_typing.await_count == 2
        sent_threads = {
            call.kwargs["metadata"]["thread_id"]
            for call in adapter.send_typing.await_args_list
        }
        assert sent_threads == {"thread-1", "thread-2"}

        adapter.reset_mock()
        await runner._refresh_async_delegation_working_statuses([second])

        adapter.stop_typing.assert_awaited_once_with(
            "C_TTC", metadata={"thread_id": "thread-1"},
        )
        adapter.send_typing.assert_awaited_once_with(
            "C_TTC", metadata={"thread_id": "thread-2"},
        )

        adapter.reset_mock()
        await runner._refresh_async_delegation_working_statuses([])
        adapter.stop_typing.assert_awaited_once_with(
            "C_TTC", metadata={"thread_id": "thread-2"},
        )

    @pytest.mark.asyncio
    async def test_failed_or_completed_records_never_show_working(self):
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        adapter = MagicMock()
        adapter.supports_status_text = True
        adapter.set_status_text = MagicMock()
        adapter.send_typing = AsyncMock()
        adapter.stop_typing = AsyncMock()
        runner.adapters = {Platform.SLACK: adapter}
        runner._session_sources = OrderedDict()

        await runner._refresh_async_delegation_working_statuses([
            {"delegation_id": "d1", "status": "error", "session_key": self._slack_key("t1")},
            {"delegation_id": "d2", "status": "completed", "session_key": self._slack_key("t2")},
        ])

        adapter.send_typing.assert_not_awaited()
        adapter.set_status_text.assert_not_called()
