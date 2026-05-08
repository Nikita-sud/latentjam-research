"""Optional Weights & Biases logging for eval / verify / bench runs.

The eval and conversion CLIs all share the same shape: produce a JSON report
of scalar metrics plus a configuration dict. This module turns that into a
W&B run when (and only when) the caller passes ``--wandb-project``. Without
that flag, every helper here is a no-op — wandb is never imported, so the
dependency stays optional in terms of runtime cost.

Auth follows the standard wandb conventions: run ``wandb login`` once, or set
``WANDB_API_KEY`` in the environment, or set ``WANDB_MODE=offline`` to record
runs locally without uploading. We do not configure auth in code.
"""

from __future__ import annotations

import os
import platform
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import click


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts to ``parent/child`` keys, dropping non-numeric leaves."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        elif isinstance(v, bool):
            # Skip — wandb treats bools oddly in scalar plots.
            continue
        elif isinstance(v, (int, float)):
            out[key] = v
    return out


def _baseline_config() -> dict[str, Any]:
    """Environment fields worth attaching to every run for reproducibility."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": os.getcwd(),
    }


@contextmanager
def wandb_run(
    project: str | None,
    *,
    run_name: str | None = None,
    tags: tuple[str, ...] | list[str] | None = None,
    config: dict[str, Any] | None = None,
    job_type: str | None = None,
    entity: str | None = None,
) -> Iterator[Any | None]:
    """Context manager that yields a wandb Run (or ``None`` when disabled).

    ``project=None`` disables logging entirely — no wandb import, no warnings.
    If wandb is requested but the package isn't installed, we print a single
    stderr warning and yield ``None`` so the caller's main path still runs.

    Run lifecycle (init / finish) is owned by the context. ``run.log(...)`` is
    safe to call inside; on context exit, ``wandb.finish()`` runs even on
    exceptions.
    """
    if not project:
        yield None
        return

    try:
        import wandb  # type: ignore[import-not-found]
    except ImportError:
        print(
            "wandb requested via --wandb-project but the package is not installed. "
            "Install with `pip install -e .[dev]` (or `pip install wandb`).",
            file=sys.stderr,
        )
        yield None
        return

    merged_config = {**_baseline_config(), **(config or {})}
    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        tags=list(tags) if tags else None,
        config=merged_config,
        job_type=job_type,
        reinit=True,
    )
    try:
        yield run
    finally:
        wandb.finish()


def log_metrics(run: Any | None, metrics: dict[str, Any]) -> None:
    """Log a (possibly nested) metrics dict to ``run`` if non-None."""
    if run is None:
        return
    flat = _flatten(metrics)
    if flat:
        run.log(flat)


def log_summary(run: Any | None, summary: dict[str, Any]) -> None:
    """Set summary fields (top-line scalars shown in the W&B run header)."""
    if run is None:
        return
    flat = _flatten(summary)
    for k, v in flat.items():
        run.summary[k] = v


def wandb_options(f: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator adding the standard ``--wandb-*`` flags to a Click command.

    Adds: ``--wandb-project``, ``--wandb-entity``, ``--wandb-run-name``,
    ``--wandb-tag`` (repeatable, surfaced as ``wandb_tags`` tuple). When
    ``--wandb-project`` is unset all wandb logic is bypassed.
    """
    f = click.option(
        "--wandb-tag",
        "wandb_tags",
        multiple=True,
        help="Repeatable tag attached to the W&B run.",
    )(f)
    f = click.option(
        "--wandb-run-name",
        default=None,
        help="W&B run name (default: auto-generated).",
    )(f)
    f = click.option(
        "--wandb-entity",
        default=None,
        help="W&B entity (team or username); uses default from `wandb login` if unset.",
    )(f)
    f = click.option(
        "--wandb-project",
        default=None,
        help="Enable W&B logging in this project. Unset = disabled (no wandb import).",
    )(f)
    return f


def log_artifact(
    run: Any | None,
    path: str,
    *,
    name: str,
    artifact_type: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Upload a file as a wandb artifact (parquet store, ONNX model, JSON report).

    No-op when ``run`` is None. Use sparingly — large parquet stores or model
    weights upload to W&B and count against quota.
    """
    if run is None:
        return
    import wandb  # type: ignore[import-not-found]

    artifact = wandb.Artifact(name=name, type=artifact_type, metadata=metadata or {})
    artifact.add_file(path)
    run.log_artifact(artifact)
