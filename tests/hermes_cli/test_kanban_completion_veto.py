"""Behavioral coverage for task-scoped, pre-commit completion vetoes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb
from hermes_cli.middleware import KANBAN_COMPLETION_VETO_MIDDLEWARE
from hermes_cli.plugins import get_plugin_manager


POLICY = "example-receipt-v1"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


@pytest.fixture(autouse=True)
def isolated_completion_middleware():
    manager = get_plugin_manager()
    original = list(manager._middleware.get(KANBAN_COMPLETION_VETO_MIDDLEWARE, []))
    manager._middleware[KANBAN_COMPLETION_VETO_MIDDLEWARE] = []
    try:
        yield manager
    finally:
        if original:
            manager._middleware[KANBAN_COMPLETION_VETO_MIDDLEWARE] = original
        else:
            manager._middleware.pop(KANBAN_COMPLETION_VETO_MIDDLEWARE, None)


def _guarded_task(conn, *, title="guarded"):
    return kb.create_task(
        conn,
        title=title,
        assignee="worker",
        completion_vetoes=[POLICY],
    )


def _register(manager, callback):
    manager._middleware[KANBAN_COMPLETION_VETO_MIDDLEWARE].append(callback)


def _allow(**kwargs):
    assert kwargs["inside_write_transaction"] is True
    assert kwargs["connection"].in_transaction is True
    assert POLICY in kwargs["required_policies"]
    return {"policy": POLICY, "decision": "allow"}


def test_unguarded_tasks_keep_legacy_completion_behavior(kanban_home):
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="legacy")
        assert kb.complete_task(conn, task_id, summary="done") is True
        assert kb.get_task(conn, task_id).status == "done"


def test_idempotent_guard_contract_mismatch_is_rejected(kanban_home):
    with kb.connect_closing() as conn:
        existing_id = kb.create_task(
            conn,
            title="unguarded webhook task",
            idempotency_key="same-request",
        )
        with pytest.raises(ValueError, match="idempotency.*completion_vetoes"):
            kb.create_task(
                conn,
                title="guarded retry",
                idempotency_key="same-request",
                completion_vetoes=[POLICY],
            )
        assert kb.get_task(conn, existing_id).completion_vetoes is None


@pytest.mark.parametrize("raw_contract", ["", "[]"])
def test_idempotent_empty_persisted_contract_is_invalid(
    kanban_home, raw_contract
):
    with kb.connect_closing() as conn:
        existing_id = kb.create_task(
            conn,
            title="corrupted idempotent task",
            idempotency_key="corrupted-request",
        )
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET completion_vetoes = ? WHERE id = ?",
                (raw_contract, existing_id),
            )
        with pytest.raises(
            ValueError, match="idempotency completion_vetoes contract is invalid"
        ):
            kb.create_task(
                conn,
                title="unguarded retry",
                idempotency_key="corrupted-request",
            )


def test_guarded_task_fails_closed_without_provider_and_audits_attempt(kanban_home):
    with kb.connect_closing() as conn:
        task_id = _guarded_task(conn)
        assert kb.complete_task(conn, task_id, summary="attempt") is False
        assert kb.get_task(conn, task_id).status == "ready"
        event = kb.list_events(conn, task_id)[-1]
        assert event.kind == "completion_vetoed"
        assert event.payload == {
            "policies": [POLICY],
            "codes": ["provider-missing-or-no-allow"],
        }


def test_every_policy_must_allow_and_any_block_wins(
    kanban_home, isolated_completion_middleware
):
    _register(isolated_completion_middleware, _allow)
    _register(
        isolated_completion_middleware,
        lambda **_: {
            "policy": POLICY,
            "decision": "block",
            "code": "receipt-invalid",
        },
    )
    with kb.connect_closing() as conn:
        task_id = _guarded_task(conn)
        assert kb.complete_task(conn, task_id) is False
        assert kb.get_task(conn, task_id).status == "ready"


def test_provider_exception_is_fail_closed(kanban_home, isolated_completion_middleware):
    def broken(**_):
        raise RuntimeError("provider unavailable")

    _register(isolated_completion_middleware, broken)
    with kb.connect_closing() as conn:
        task_id = _guarded_task(conn)
        assert kb.complete_task(conn, task_id) is False
        event = kb.list_events(conn, task_id)[-1]
        assert set(event.payload["codes"]) == {
            "provider-error",
            "provider-missing-or-no-allow",
        }


@pytest.mark.parametrize(
    "malformed",
    [
        "allow",
        {"policy": "another-policy", "decision": "allow"},
        {"policy": POLICY, "decision": "maybe"},
    ],
)
def test_any_malformed_provider_output_blocks_even_with_an_allow(
    kanban_home, isolated_completion_middleware, malformed
):
    _register(isolated_completion_middleware, _allow)
    _register(isolated_completion_middleware, lambda **_: malformed)
    with kb.connect_closing() as conn:
        task_id = _guarded_task(conn)
        assert kb.complete_task(conn, task_id) is False
        event = kb.list_events(conn, task_id)[-1]
        assert "provider-response-invalid" in event.payload["codes"]


def test_malformed_persisted_policy_is_fail_closed(kanban_home):
    with kb.connect_closing() as conn:
        task_id = _guarded_task(conn)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET completion_vetoes = ? WHERE id = ?",
                ("not-json", task_id),
            )
        assert kb.complete_task(conn, task_id) is False
        assert kb.get_task(conn, task_id).status == "ready"
        event = kb.list_events(conn, task_id)[-1]
        assert event.payload == {
            "policies": ["invalid-completion-veto-config"],
            "codes": ["policy-config-invalid"],
        }
        assert kb.archive_task(conn, task_id) is False
        assert kb.delete_task(conn, task_id) is False


@pytest.mark.parametrize("raw_contract", ["", "[]"])
def test_empty_persisted_policy_contract_is_fail_closed(
    kanban_home, raw_contract
):
    with kb.connect_closing() as conn:
        task_id = _guarded_task(conn)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET completion_vetoes = ? WHERE id = ?",
                (raw_contract, task_id),
            )
        assert kb.complete_task(conn, task_id) is False
        event = kb.list_events(conn, task_id)[-1]
        assert event.payload == {
            "policies": ["invalid-completion-veto-config"],
            "codes": ["policy-config-invalid"],
        }
        assert kb.archive_task(conn, task_id) is False
        assert kb.delete_task(conn, task_id) is False


def test_failed_plugin_registration_rolls_back_veto_callback(
    kanban_home, monkeypatch
):
    from hermes_cli import plugins as plugins_module

    plugin_dir = kanban_home / "plugins" / "broken_veto_provider"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        "name: broken_veto_provider\nversion: 0.1.0\n",
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        "def _allow(**kwargs):\n"
        f"    return {{'policy': '{POLICY}', 'decision': 'allow'}}\n\n"
        "def register(ctx):\n"
        "    ctx.register_middleware('kanban_completion_veto', _allow)\n"
        "    raise RuntimeError('registration failed')\n",
        encoding="utf-8",
    )
    (kanban_home / "config.yaml").write_text(
        "plugins:\n  enabled:\n    - broken_veto_provider\n",
        encoding="utf-8",
    )
    fresh_manager = plugins_module.PluginManager()
    monkeypatch.setattr(plugins_module, "_plugin_manager", fresh_manager)
    middleware_staged = threading.Event()
    registration_can_fail = threading.Event()
    original_register_middleware = plugins_module.PluginContext.register_middleware

    def pause_after_registration(context, kind, callback):
        original_register_middleware(context, kind, callback)
        if context.manifest.name == "broken_veto_provider":
            middleware_staged.set()
            assert registration_can_fail.wait(timeout=10)

    monkeypatch.setattr(
        plugins_module.PluginContext,
        "register_middleware",
        pause_after_registration,
    )
    discovery = threading.Thread(target=fresh_manager.discover_and_load)
    discovery.start()
    assert middleware_staged.wait(timeout=10)

    with kb.connect_closing() as conn:
        task_id = _guarded_task(conn)
        assert kb.complete_task(conn, task_id) is False
        event = kb.list_events(conn, task_id)[-1]
        assert "provider-missing-or-no-allow" in event.payload["codes"]

    registration_can_fail.set()
    discovery.join(timeout=10)
    assert not discovery.is_alive()
    assert fresh_manager._plugins["broken_veto_provider"].enabled is False
    assert fresh_manager._plugins["broken_veto_provider"].error
    assert fresh_manager._middleware.get(KANBAN_COMPLETION_VETO_MIDDLEWARE, []) == []


def test_completion_waits_for_the_entire_plugin_discovery_sweep(
    kanban_home, monkeypatch
):
    from hermes_cli import plugins as plugins_module

    for name, decision in (
        ("aaa_allow_veto_provider", "allow"),
        ("zzz_block_veto_provider", "block"),
    ):
        plugin_dir = kanban_home / "plugins" / name
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.yaml").write_text(
            f"name: {name}\nversion: 0.1.0\n",
            encoding="utf-8",
        )
        (plugin_dir / "__init__.py").write_text(
            "def _decision(**kwargs):\n"
            f"    return {{'policy': '{POLICY}', 'decision': '{decision}'}}\n\n"
            "def register(ctx):\n"
            "    ctx.register_middleware('kanban_completion_veto', _decision)\n",
            encoding="utf-8",
        )
    (kanban_home / "config.yaml").write_text(
        "plugins:\n"
        "  enabled:\n"
        "    - aaa_allow_veto_provider\n"
        "    - zzz_block_veto_provider\n",
        encoding="utf-8",
    )
    fresh_manager = plugins_module.PluginManager()
    monkeypatch.setattr(plugins_module, "_plugin_manager", fresh_manager)
    blocker_staged = threading.Event()
    discovery_can_finish = threading.Event()
    completion_started = threading.Event()
    completion_finished = threading.Event()
    result = {}
    original_register_middleware = plugins_module.PluginContext.register_middleware

    def pause_second_provider(context, kind, callback):
        original_register_middleware(context, kind, callback)
        if context.manifest.name == "zzz_block_veto_provider":
            blocker_staged.set()
            assert discovery_can_finish.wait(timeout=10)

    monkeypatch.setattr(
        plugins_module.PluginContext,
        "register_middleware",
        pause_second_provider,
    )
    with kb.connect_closing() as conn:
        task_id = _guarded_task(conn)

    discovery = threading.Thread(target=fresh_manager.discover_and_load)
    discovery.start()
    assert blocker_staged.wait(timeout=10)

    def complete_in_parallel():
        completion_started.set()
        with kb.connect_closing() as conn:
            result["allowed"] = kb.complete_task(conn, task_id)
        completion_finished.set()

    completion = threading.Thread(target=complete_in_parallel)
    completion.start()
    assert completion_started.wait(timeout=10)
    completion_finished.wait(timeout=2)
    discovery_can_finish.set()
    discovery.join(timeout=10)
    completion.join(timeout=10)

    assert not discovery.is_alive()
    assert not completion.is_alive()
    assert result["allowed"] is False
    assert len(
        fresh_manager._middleware[KANBAN_COMPLETION_VETO_MIDDLEWARE]
    ) == 2


def test_completion_uses_snapshot_across_concurrent_forced_rediscovery(
    kanban_home, monkeypatch
):
    from hermes_cli import plugins as plugins_module

    for name, decision in (
        ("aaa_allow_veto_provider", "allow"),
        ("zzz_block_veto_provider", "block"),
    ):
        plugin_dir = kanban_home / "plugins" / name
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.yaml").write_text(
            f"name: {name}\nversion: 0.1.0\n",
            encoding="utf-8",
        )
        (plugin_dir / "__init__.py").write_text(
            "def _decision(**kwargs):\n"
            f"    return {{'policy': '{POLICY}', 'decision': '{decision}'}}\n\n"
            "def register(ctx):\n"
            "    ctx.register_middleware('kanban_completion_veto', _decision)\n",
            encoding="utf-8",
        )
    (kanban_home / "config.yaml").write_text(
        "plugins:\n"
        "  enabled:\n"
        "    - aaa_allow_veto_provider\n"
        "    - zzz_block_veto_provider\n",
        encoding="utf-8",
    )
    fresh_manager = plugins_module.PluginManager()
    monkeypatch.setattr(plugins_module, "_plugin_manager", fresh_manager)
    fresh_manager.discover_and_load()
    with kb.connect_closing() as conn:
        task_id = _guarded_task(conn)

    evaluation_waiting = threading.Event()
    evaluation_can_continue = threading.Event()
    blocker_staged = threading.Event()
    rediscovery_can_finish = threading.Event()
    result = {}
    original_evaluate = kb._evaluate_completion_vetoes
    original_register_middleware = plugins_module.PluginContext.register_middleware

    def pause_before_evaluation(*args, **kwargs):
        evaluation_waiting.set()
        assert evaluation_can_continue.wait(timeout=10)
        return original_evaluate(*args, **kwargs)

    def pause_forced_block_provider(context, kind, callback):
        original_register_middleware(context, kind, callback)
        if (
            context.manifest.name == "zzz_block_veto_provider"
            and threading.current_thread().name == "forced-rediscovery"
        ):
            blocker_staged.set()
            assert rediscovery_can_finish.wait(timeout=10)

    monkeypatch.setattr(kb, "_evaluate_completion_vetoes", pause_before_evaluation)
    monkeypatch.setattr(
        plugins_module.PluginContext,
        "register_middleware",
        pause_forced_block_provider,
    )

    def complete_in_parallel():
        with kb.connect_closing() as conn:
            result["allowed"] = kb.complete_task(conn, task_id)

    completion = threading.Thread(target=complete_in_parallel)
    completion.start()
    assert evaluation_waiting.wait(timeout=10)
    rediscovery = threading.Thread(
        target=lambda: fresh_manager.discover_and_load(force=True),
        name="forced-rediscovery",
    )
    rediscovery.start()
    assert blocker_staged.wait(timeout=10)
    evaluation_can_continue.set()
    completion.join(timeout=10)
    rediscovery_can_finish.set()
    rediscovery.join(timeout=10)

    assert not completion.is_alive()
    assert not rediscovery.is_alive()
    assert result["allowed"] is False


def test_reentrant_completion_during_discovery_is_fail_closed(
    kanban_home, monkeypatch
):
    from hermes_cli import plugins as plugins_module

    for name, decision in (
        ("aaa_allow_veto_provider", "allow"),
        ("zzz_block_veto_provider", "block"),
    ):
        plugin_dir = kanban_home / "plugins" / name
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.yaml").write_text(
            f"name: {name}\nversion: 0.1.0\n",
            encoding="utf-8",
        )
        (plugin_dir / "__init__.py").write_text(
            "def _decision(**kwargs):\n"
            f"    return {{'policy': '{POLICY}', 'decision': '{decision}'}}\n\n"
            "def register(ctx):\n"
            "    ctx.register_middleware('kanban_completion_veto', _decision)\n",
            encoding="utf-8",
        )
    (kanban_home / "config.yaml").write_text(
        "plugins:\n"
        "  enabled:\n"
        "    - aaa_allow_veto_provider\n"
        "    - zzz_block_veto_provider\n",
        encoding="utf-8",
    )
    with kb.connect_closing() as conn:
        task_id = _guarded_task(conn)
    fresh_manager = plugins_module.PluginManager()
    monkeypatch.setattr(plugins_module, "_plugin_manager", fresh_manager)
    result = {}
    original_register_middleware = plugins_module.PluginContext.register_middleware

    def complete_before_late_block_registration(context, kind, callback):
        if context.manifest.name == "zzz_block_veto_provider":
            with kb.connect_closing() as conn:
                result["allowed"] = kb.complete_task(conn, task_id)
        original_register_middleware(context, kind, callback)

    monkeypatch.setattr(
        plugins_module.PluginContext,
        "register_middleware",
        complete_before_late_block_registration,
    )
    fresh_manager.discover_and_load()

    assert result["allowed"] is False
    assert len(
        fresh_manager._middleware[KANBAN_COMPLETION_VETO_MIDDLEWARE]
    ) == 2


def test_allow_decision_and_status_cas_share_one_transaction(
    kanban_home, isolated_completion_middleware
):
    _register(isolated_completion_middleware, _allow)
    with kb.connect_closing() as conn:
        task_id = _guarded_task(conn)
        assert kb.complete_task(conn, task_id, summary="receipt verified") is True
        task = kb.get_task(conn, task_id)
        assert task.status == "done"
        assert task.completion_vetoes == [POLICY]


def test_veto_provider_receives_the_board_bound_to_the_connection(
    kanban_home, isolated_completion_middleware
):
    seen_boards = []

    def allow_for_secondary(*, board, **kwargs):
        seen_boards.append(board)
        return _allow(board=board, **kwargs)

    _register(isolated_completion_middleware, allow_for_secondary)
    kb.init_db(board="secondary")
    with kb.connect_closing(board="secondary") as conn:
        task_id = _guarded_task(conn)
        assert kb.complete_task(conn, task_id) is True
    assert seen_boards == ["secondary"]


def test_complete_discovers_an_enabled_veto_provider_before_write_transaction(
    kanban_home, monkeypatch
):
    from hermes_cli import plugins as plugins_module

    plugin_dir = kanban_home / "plugins" / "completion_veto_provider"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        "name: completion_veto_provider\nversion: 0.1.0\n",
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        "def _allow(**kwargs):\n"
        "    if not kwargs['connection'].in_transaction:\n"
        "        raise RuntimeError('outside transaction')\n"
        f"    return {{'policy': '{POLICY}', 'decision': 'allow'}}\n\n"
        "def register(ctx):\n"
        "    ctx.register_middleware('kanban_completion_veto', _allow)\n",
        encoding="utf-8",
    )
    (kanban_home / "config.yaml").write_text(
        "plugins:\n  enabled:\n    - completion_veto_provider\n",
        encoding="utf-8",
    )
    fresh_manager = plugins_module.PluginManager()
    monkeypatch.setattr(plugins_module, "_plugin_manager", fresh_manager)

    with kb.connect_closing() as conn:
        task_id = _guarded_task(conn)
        assert kb.complete_task(conn, task_id) is True
        assert fresh_manager._discovered is True


def test_standalone_cli_process_discovers_an_enabled_veto_provider(kanban_home):
    plugin_dir = kanban_home / "plugins" / "cli_completion_veto_provider"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        "name: cli_completion_veto_provider\nversion: 0.1.0\n",
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        "def _allow(**kwargs):\n"
        f"    return {{'policy': '{POLICY}', 'decision': 'allow'}}\n\n"
        "def register(ctx):\n"
        "    ctx.register_middleware('kanban_completion_veto', _allow)\n",
        encoding="utf-8",
    )
    (kanban_home / "config.yaml").write_text(
        "plugins:\n  enabled:\n    - cli_completion_veto_provider\n",
        encoding="utf-8",
    )
    with kb.connect_closing() as conn:
        task_id = _guarded_task(conn)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "hermes_cli.main",
            "kanban",
            "complete",
            task_id,
            "--summary",
            "verified by enabled plugin",
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    with kb.connect_closing() as conn:
        assert kb.get_task(conn, task_id).status == "done"


def test_cli_complete_cannot_bypass_veto(kanban_home, capsys):
    from hermes_cli.kanban import _cmd_complete

    with kb.connect_closing() as conn:
        task_id = _guarded_task(conn)
    args = argparse.Namespace(
        task_ids=[task_id],
        summary=None,
        metadata=None,
        result="manual",
    )
    assert _cmd_complete(args) == 1
    assert "cannot complete" in capsys.readouterr().err
    with kb.connect_closing() as conn:
        assert kb.get_task(conn, task_id).status == "ready"


def test_cli_create_preserves_completion_veto_contract(
    kanban_home, capsys, monkeypatch
):
    from hermes_cli import kanban

    monkeypatch.setattr(kanban, "_check_dispatcher_presence", lambda: (True, None))
    args = argparse.Namespace(
        title="cli guarded",
        body=None,
        assignee="worker",
        created_by="user",
        workspace="scratch",
        branch=None,
        project=None,
        tenant=None,
        priority=0,
        parent=[],
        triage=False,
        idempotency_key=None,
        max_runtime=None,
        skills=[],
        max_retries=None,
        model_override=None,
        provider_override=None,
        goal_mode=False,
        goal_max_turns=None,
        initial_status="running",
        completion_vetoes=[POLICY],
        json=True,
    )
    assert kanban._cmd_create(args) == 0
    task_id = json.loads(capsys.readouterr().out)["id"]
    with kb.connect_closing() as conn:
        assert kb.get_task(conn, task_id).completion_vetoes == [POLICY]


def test_worker_tool_cannot_bypass_veto(kanban_home, monkeypatch):
    from tools import kanban_tools

    with kb.connect_closing() as conn:
        task_id = _guarded_task(conn)
        assert kb.claim_task(conn, task_id) is not None
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    output = json.loads(
        kanban_tools._handle_complete(
            {"task_id": task_id, "summary": "worker attempt", "metadata": {}}
        )
    )
    assert output["error"] is not None
    with kb.connect_closing() as conn:
        assert kb.get_task(conn, task_id).status == "running"


def test_worker_create_preserves_completion_veto_contract(kanban_home):
    from tools import kanban_tools

    output = json.loads(
        kanban_tools._handle_create(
            {
                "title": "worker-created guarded task",
                "assignee": "worker",
                "completion_vetoes": [POLICY],
            }
        )
    )
    assert output["ok"] is True, output
    with kb.connect_closing() as conn:
        assert kb.get_task(conn, output["task_id"]).completion_vetoes == [POLICY]


def _dashboard_client():
    plugin_file = (
        Path(__file__).resolve().parents[2]
        / "plugins"
        / "kanban"
        / "dashboard"
        / "plugin_api.py"
    )
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_completion_veto_test", plugin_file
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    app = FastAPI()
    app.include_router(module.router, prefix="/api/plugins/kanban")
    return TestClient(app)


def test_dashboard_complete_cannot_bypass_veto(kanban_home):
    with kb.connect_closing() as conn:
        task_id = _guarded_task(conn)
    response = _dashboard_client().patch(
        f"/api/plugins/kanban/tasks/{task_id}",
        json={"status": "done", "summary": "dashboard attempt"},
    )
    assert response.status_code == 409
    with kb.connect_closing() as conn:
        assert kb.get_task(conn, task_id).status == "ready"


def test_dashboard_create_preserves_completion_veto_contract(kanban_home):
    response = _dashboard_client().post(
        "/api/plugins/kanban/tasks",
        json={
            "title": "dashboard guarded",
            "assignee": "worker",
            "completion_vetoes": [POLICY],
        },
    )
    assert response.status_code == 200, response.text
    task_id = response.json()["task"]["id"]
    with kb.connect_closing() as conn:
        assert kb.get_task(conn, task_id).completion_vetoes == [POLICY]


def test_dashboard_complete_succeeds_only_after_policy_allow(
    kanban_home, isolated_completion_middleware
):
    _register(isolated_completion_middleware, _allow)
    with kb.connect_closing() as conn:
        task_id = _guarded_task(conn)
    response = _dashboard_client().patch(
        f"/api/plugins/kanban/tasks/{task_id}",
        json={"status": "done", "summary": "dashboard verified"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["task"]["status"] == "done"


def test_archive_and_hard_delete_cannot_bypass_guard(
    kanban_home, isolated_completion_middleware
):
    with kb.connect_closing() as conn:
        task_id = _guarded_task(conn)
        assert kb.archive_task(conn, task_id) is False
        assert kb.delete_task(conn, task_id) is False
        assert kb.get_task(conn, task_id).status == "ready"

        _register(isolated_completion_middleware, _allow)
        assert kb.complete_task(conn, task_id) is True
        assert kb.archive_task(conn, task_id) is True
        assert kb.delete_archived_task(conn, task_id) is True


def test_guarded_swarm_activation_denial_rolls_back_the_whole_topology(kanban_home):
    from hermes_cli import kanban_swarm as swarm

    with kb.connect_closing() as conn:
        with pytest.raises(RuntimeError, match="activate"):
            swarm.create_swarm(
                conn,
                goal="guarded swarm",
                workers=[
                    swarm.SwarmWorkerSpec(
                        profile="worker", title="work", body=None
                    )
                ],
                verifier_assignee="reviewer",
                synthesizer_assignee="writer",
                root_completion_vetoes=[POLICY],
            )
        count = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        assert count == 0


def test_guarded_swarm_activation_allow_commits_the_whole_topology(
    kanban_home, isolated_completion_middleware
):
    from hermes_cli import kanban_swarm as swarm

    _register(isolated_completion_middleware, _allow)
    with kb.connect_closing() as conn:
        created = swarm.create_swarm(
            conn,
            goal="guarded swarm",
            workers=[
                swarm.SwarmWorkerSpec(profile="worker", title="work", body=None)
            ],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
            root_completion_vetoes=[POLICY],
        )
        assert kb.get_task(conn, created.root_id).status == "done"
        assert kb.get_task(conn, created.worker_ids[0]).status == "ready"
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 4
