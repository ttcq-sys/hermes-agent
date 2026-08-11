"""Regression tests for repository-pinned Hermes update branches."""

from types import SimpleNamespace
from unittest.mock import patch

from hermes_cli.main import _resolve_update_branch


def test_explicit_update_branch_wins_over_repository_pin():
    with patch("hermes_cli.main._read_repository_update_branch", return_value="safe/channel"):
        assert _resolve_update_branch(SimpleNamespace(branch="release/candidate")) == "release/candidate"


def test_repository_pin_is_used_when_cli_branch_is_absent():
    with patch("hermes_cli.main._read_repository_update_branch", return_value="safe/channel"):
        assert _resolve_update_branch(SimpleNamespace(branch=None)) == "safe/channel"


def test_blank_repository_pin_preserves_main_default():
    with patch("hermes_cli.main._read_repository_update_branch", return_value=None):
        assert _resolve_update_branch(SimpleNamespace(branch=None)) == "main"
