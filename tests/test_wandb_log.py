from __future__ import annotations

import sys
from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner

from utils.wandb_log import (
    _flatten,
    log_metrics,
    log_summary,
    wandb_options,
    wandb_run,
)


def test_flatten_dot_keys() -> None:
    out = _flatten({"a": {"b": 1, "c": {"d": 2.5}}, "e": 3})
    assert out == {"a/b": 1, "a/c/d": 2.5, "e": 3}


def test_flatten_drops_non_numeric() -> None:
    out = _flatten({"a": "string", "b": True, "c": None, "d": 4.0, "e": [1, 2, 3]})
    assert out == {"d": 4.0}


def test_flatten_handles_empty() -> None:
    assert _flatten({}) == {}


def test_wandb_run_disabled_when_project_none() -> None:
    # When project is None, wandb is not imported and not initialized.
    with wandb_run(None) as run:
        assert run is None


def test_wandb_run_disabled_for_empty_string() -> None:
    with wandb_run("") as run:
        assert run is None


def test_log_metrics_no_op_on_none() -> None:
    log_metrics(None, {"recall_at_k_genre": {"5": 0.7}})


def test_log_summary_no_op_on_none() -> None:
    log_summary(None, {"foo": 1.0})


def test_wandb_run_warns_when_package_missing(capsys: pytest.CaptureFixture) -> None:
    # Force the import of wandb to fail and confirm we yield None + warn once.
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name, *args, **kwargs):
        if name == "wandb":
            raise ImportError("no wandb in this env")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        # Drop any cached import.
        sys.modules.pop("wandb", None)
        with wandb_run("some-project") as run:
            assert run is None
    captured = capsys.readouterr()
    assert "wandb" in captured.err.lower()


def test_wandb_run_uses_disabled_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    # WANDB_MODE=disabled gives us a real wandb.Run object that no-ops on
    # network and disk — perfect for integration testing.
    monkeypatch.setenv("WANDB_MODE", "disabled")

    with wandb_run("test-project", config={"x": 1}, tags=("unit-test",)) as run:
        assert run is not None
        # log_metrics + log_summary should accept the real run without error.
        log_metrics(run, {"a": 1.0, "nested": {"b": 2.0}})
        log_summary(run, {"final": 0.5})


def test_wandb_options_adds_four_flags() -> None:
    @click.command()
    @wandb_options
    def fake_cmd(
        wandb_project: str | None,
        wandb_entity: str | None,
        wandb_run_name: str | None,
        wandb_tags: tuple[str, ...],
    ) -> None:
        click.echo(repr((wandb_project, wandb_entity, wandb_run_name, list(wandb_tags))))

    runner = CliRunner()
    result = runner.invoke(
        fake_cmd,
        [
            "--wandb-project",
            "p",
            "--wandb-entity",
            "e",
            "--wandb-run-name",
            "r",
            "--wandb-tag",
            "t1",
            "--wandb-tag",
            "t2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "('p', 'e', 'r', ['t1', 't2'])" in result.output


def test_wandb_options_defaults_are_none() -> None:
    @click.command()
    @wandb_options
    def fake_cmd(
        wandb_project: str | None,
        wandb_entity: str | None,
        wandb_run_name: str | None,
        wandb_tags: tuple[str, ...],
    ) -> None:
        click.echo(repr((wandb_project, wandb_entity, wandb_run_name, list(wandb_tags))))

    result = CliRunner().invoke(fake_cmd, [])
    assert result.exit_code == 0
    assert "(None, None, None, [])" in result.output
