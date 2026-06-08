# CHANGED(baseline): NEW FILE — not present in the Akama et al. upstream. Host-aware path/worker resolution.
"""
Dynamic path resolution and resource sizing.

The goal is to keep the *repository layout* identical across machines (so the
default values in ``configs/baseline.yaml`` reflect a self-contained checkout
that other users / other machines can clone and run as-is), while still letting
specific hosts redirect heavy I/O paths to shared storage and pick CPU-friendly
defaults.

Resolution order, evaluated independently for each path:
    1. Explicit environment variable (highest priority — escape hatch).
    2. Host-specific profile (matched by hostname prefix).
    3. The value passed in (typically the default from the YAML config or
       an explicit ``--flag`` from the CLI).

Workers are auto-sized when the caller asks for the sentinel ``-1`` or
``"auto"``; otherwise the explicit value is honored. The auto-sizing reserves
a few cores for the operating system / other users on shared servers.

This module is intentionally side-effect free and depends only on the standard
library, so it can be imported very early in ``main.py`` /
``checkpoint_test.py`` without risking circular imports.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path

# Repo root: src/utils/paths.py -> parents[2] is the repository root.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Per-host overrides.
#
# Keys are matched as *prefixes* of socket.gethostname() so that e.g.
# "meg-server-3.cimec.unitn.it" matches "meg-server-3".
#
# To add a new host, append an entry here. Any key omitted from a profile
# falls back to the value the caller passed in (typically the YAML default).
_HOST_PROFILES: dict[str, dict[str, str]] = {
    "meg-server-3": {
        "dataset_dir": "/mnt/storage/tier2/danbal/projects/BRACHR002N3B/Andrea/data/dataset",
        "log_dir":     "/mnt/storage/tier2/danbal/projects/BRACHR002N3B/Andrea/outputs",
    },
}


def _hostname() -> str:
    return socket.gethostname()


def _host_profile() -> dict[str, str]:
    host = _hostname()
    for prefix, profile in _HOST_PROFILES.items():
        if host.startswith(prefix):
            return profile
    return {}


def _resolve(env_var: str, profile_key: str, fallback: str) -> str:
    """Pick env-var > host-profile > fallback (in that order)."""
    v = os.environ.get(env_var)
    if v:
        return v
    p = _host_profile().get(profile_key)
    if p:
        return p
    return fallback


def resolve_dataset_dir(fallback: str) -> str:
    """Return the dataset root for the current host.

    ``fallback`` is whatever the caller already had (typically the YAML
    default ``"../dataset"`` or an explicit ``--dataset_dir`` value).
    """
    return _resolve("EEG_DATASET_DIR", "dataset_dir", fallback)


def resolve_log_dir(fallback: str) -> str:
    """Return the base directory for TensorBoard logs / Lightning outputs."""
    return _resolve("EEG_LOG_DIR", "log_dir", fallback)


def _effective_cpu_count() -> int:
    """Return the number of CPUs actually available to *this process*.

    Inside SLURM jobs, Docker containers, k8s pods, or any cgroup-restricted
    environment, ``os.cpu_count()`` lies — it reports the host-wide CPU count
    even if only a fraction is allocated. ``os.sched_getaffinity(0)`` reflects
    the real allocation on Linux. On macOS / Windows this attribute doesn't
    exist, so we fall back to ``os.cpu_count()``.
    """
    try:
        # Linux: rispetta cgroup, SLURM cpus-per-task, taskset, ecc.
        return len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def resolve_workers(requested) -> int:
    """Pick a sensible num_workers for DataLoaders.

    Behavior:
        * ``EEG_NUM_WORKERS`` env var, if set, always wins.
        * If ``requested`` is the sentinel ``-1`` or the string ``"auto"``,
          auto-size.
        * Otherwise the explicit integer is honored.

    Auto-sizing strategy:
        Two regimes, distinguished by whether the process is running inside a
        cgroup/SLURM allocation (i.e. ``sched_getaffinity`` reports fewer CPUs
        than the host actually has):

        * **Isolated allocation** (SLURM job, container): we own those CPUs
          exclusively, so use most of them — ``effective_cpu - 1`` (one left
          for the main process), capped at 16 to avoid runaway worker counts
          on huge allocations.
        * **Bare-metal / shared machine** (no isolation): be conservative,
          ``(cpu - 4) // 2``, capped at 16, so we don't monopolize a server
          shared with other users.

    The cap of 16 is empirical: above that, DataLoader workers compete more
    than they help on this kind of pipeline. Always returns at least 1.
    """
    env = os.environ.get("EEG_NUM_WORKERS")
    if env is not None and env != "":
        try:
            return max(0, int(env))
        except ValueError:
            pass

    if isinstance(requested, str):
        if requested.strip().lower() == "auto":
            requested = -1
        else:
            try:
                requested = int(requested)
            except ValueError:
                requested = -1

    if isinstance(requested, int) and requested >= 0:
        return requested

    host_cpu = os.cpu_count() or 1
    effective_cpu = _effective_cpu_count()
    in_isolated_alloc = effective_cpu < host_cpu

    if in_isolated_alloc:
        # SLURM / cgroup / container: those CPUs are ours, use them.
        suggested = max(1, min(16, effective_cpu - 1))
    else:
        # Shared box, leave headroom for OS and other users.
        suggested = max(1, min(16, (effective_cpu - 4) // 2))
    return suggested


def describe() -> dict:
    """Diagnostic snapshot useful for printing at startup."""
    return {
        "hostname": _hostname(),
        "host_profile_keys": sorted(_host_profile().keys()),
        "cpu_count": os.cpu_count(),
        "cpu_effective": _effective_cpu_count(),
        "repo_root": str(_REPO_ROOT),
    }
