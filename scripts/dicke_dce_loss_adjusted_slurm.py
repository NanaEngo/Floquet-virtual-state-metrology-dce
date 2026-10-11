#!/usr/bin/env python3
"""Launch one resumable Dicke truncation-extension SLURM array task.

This wrapper executes exactly one configured ``(N_cav, kappa/wc)`` row and
never submits jobs or overwrites a completed checkpoint.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    from config import DEFAULT_DICKE_LOSS_EXTENSION
except ImportError as exc:  # pragma: no cover
    raise ImportError("Run this launcher from the Track B scripts directory.") from exc


CONFIG = DEFAULT_DICKE_LOSS_EXTENSION


def parse_args() -> argparse.Namespace:
    """Parse one bounded array task.

    Returns:
        argparse.Namespace: Truncation, loss index, and checkpoint options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-cav", type=int, required=True, choices=CONFIG.cavity_truncations)
    parser.add_argument(
        "--kappa-index",
        type=int,
        required=True,
        choices=range(len(CONFIG.kappa_values)),
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run one extension producer task without automatic submission."""
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir or Path(__file__).resolve().parents[1] / "data" / "_cache" / "dicke_dce_loss_truncation_extension"
    kappa = CONFIG.kappa_values[args.kappa_index]
    producer = Path(__file__).with_name("dicke_dce_loss_truncation_extension.py")
    command = [
        sys.executable,
        str(producer),
        "--n-cav",
        str(args.n_cav),
        "--kappa",
        str(kappa),
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]
    if args.n_steps is not None:
        command.extend(["--n-steps", str(args.n_steps)])
    if args.no_resume:
        command.append("--no-resume")
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
