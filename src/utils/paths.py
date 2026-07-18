# CHANGED(baseline): NEW FILE — not present in the Akama et al. upstream. Path/worker resolution.
"""Path and DataLoader-worker resolution, env-overridable for cloud runs.

Each path falls back to whatever the caller passes (the YAML default or an
explicit ``--flag``); an ``EEG_*`` environment variable overrides it — the
Colab / shared-storage escape hatch. Workers auto-size on the sentinel ``-1`` /
``"auto"``, reserving headroom on shared machines. Stdlib-only and side-effect
free, so it imports safely very early in ``main.py`` / ``checkpoint_test.py``.
"""
import os
import socket
from pathlib import Path

# Repo root: src/utils/paths.py -> parents[2] is the repository root.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_dataset_dir(fallback: str) -> str:
    """Dataset root: ``EEG_DATASET_DIR`` if set, else the caller's value."""
    return os.environ.get("EEG_DATASET_DIR") or fallback


def resolve_log_dir(fallback: str) -> str:
    """Log/checkpoint base dir: ``EEG_LOG_DIR`` if set, else the caller's value."""
    return os.environ.get("EEG_LOG_DIR") or fallback


def _effective_cpu_count() -> int:
    """CPUs actually available to this process. ``sched_getaffinity`` reflects the
    real SLURM/cgroup/taskset allocation on Linux (``os.cpu_count()`` reports the
    whole host); falls back to ``cpu_count`` on macOS/Windows."""
    try:
        return len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def resolve_workers(requested) -> int:
    """num_workers for DataLoaders.

    ``EEG_NUM_WORKERS`` wins; an explicit ``>= 0`` value is honored; the sentinel
    ``-1`` / ``"auto"`` (or any unparseable string) auto-sizes: nearly all CPUs
    inside an isolated SLURM/cgroup allocation (we own them), but ``(cpu-4)//2`` on
    a shared box so we don't starve other users. Capped at 16, floored at 1 — both
    empirical knobs for this pipeline.
    """
    env = os.environ.get("EEG_NUM_WORKERS")
    if env:
        try:
            return max(0, int(env))
        except ValueError:
            pass

    if isinstance(requested, str):
        try:
            requested = int(requested)
        except ValueError:
            requested = -1  # "auto" or any non-integer string -> auto-size
    if isinstance(requested, int) and requested >= 0:
        return requested

    effective = _effective_cpu_count()
    isolated = effective < (os.cpu_count() or 1)  # SLURM/cgroup/container vs bare metal
    suggested = effective - 1 if isolated else (effective - 4) // 2
    return max(1, min(16, suggested))


def describe() -> dict:
    """Diagnostic snapshot useful for printing at startup."""
    return {
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "cpu_effective": _effective_cpu_count(),
        "repo_root": str(_REPO_ROOT),
    }
