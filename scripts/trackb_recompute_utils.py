"""Shared deterministic utilities for Track B recomputation producers.

The helpers in this module are deliberately small and dependency-light. They
support resumable HPC jobs without silently repairing numerical results.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


def stable_row_id(parameters: dict[str, Any], prefix: str = "row") -> str:
    """Return a deterministic filesystem-safe identifier for one parameter row.

    Args:
        parameters: JSON-compatible scalar parameter mapping.
        prefix: Identifier prefix.

    Returns:
        str: Prefix plus the first 16 hexadecimal SHA-256 characters.
    """
    raw = json.dumps(parameters, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write one JSON payload and create parent directories.

    Args:
        path: Destination JSON path.
        payload: JSON-serializable mapping.

    Raises:
        TypeError: If the payload is not JSON serializable.
        OSError: If the destination cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def require_schema(payload: dict[str, Any], schema: str) -> None:
    """Require a matching artifact schema.

    Args:
        payload: Artifact mapping.
        schema: Required schema identifier.

    Raises:
        ValueError: If the schema is absent or mismatched.
    """
    if payload.get("schema") != schema:
        raise ValueError(f"Expected schema {schema!r}, found {payload.get('schema')!r}")


def _walk_values(value: Any, path: str = "$") -> Iterable[tuple[str, Any]]:
    """Yield scalar values from a nested JSON-compatible object."""
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk_values(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_values(child, f"{path}[{index}]")
    else:
        yield path, value


def assert_finite_json(payload: dict[str, Any], ignored_keys: set[str] | None = None) -> None:
    """Reject non-finite floating-point values in an artifact.

    Args:
        payload: Nested JSON-compatible artifact.
        ignored_keys: Optional terminal-key names exempted from this check.

    Raises:
        ValueError: If a NaN or infinity is found.
    """
    ignored = ignored_keys or set()
    for path, value in _walk_values(payload):
        if path.rsplit(".", 1)[-1] in ignored:
            continue
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Non-finite value at {path}: {value!r}")


def numerical_gate(
    trace_error: float,
    hermiticity_error: float,
    minimum_eigenvalue: float,
    eigenvalue_floor: float,
    trace_tolerance: float = 1.0e-8,
    hermiticity_tolerance: float = 1.0e-8,
) -> dict[str, Any]:
    """Evaluate a non-repairing numerical state-acceptance gate.

    Args:
        trace_error: Absolute trace error.
        hermiticity_error: Norm of rho-rho.dag().
        minimum_eigenvalue: Smallest recorded real eigenvalue.
        eigenvalue_floor: Accepted numerical floor; not exact positivity.
        trace_tolerance: Maximum accepted trace error.
        hermiticity_tolerance: Maximum accepted Hermiticity error.

    Returns:
        dict[str, Any]: Diagnostics and boolean acceptance.
    """
    accepted = (
        math.isfinite(trace_error)
        and math.isfinite(hermiticity_error)
        and math.isfinite(minimum_eigenvalue)
        and trace_error <= trace_tolerance
        and hermiticity_error <= hermiticity_tolerance
        and minimum_eigenvalue >= eigenvalue_floor
    )
    return {
        "trace_error": float(trace_error),
        "hermiticity_error": float(hermiticity_error),
        "minimum_eigenvalue": float(minimum_eigenvalue),
        "eigenvalue_floor": float(eigenvalue_floor),
        "trace_tolerance": float(trace_tolerance),
        "hermiticity_tolerance": float(hermiticity_tolerance),
        "numerical_acceptance": bool(accepted),
        "note": "Numerical acceptance is not a proof of exact positivity.",
    }


def merge_checkpoints(
    checkpoint_dir: Path,
    expected_ids: set[str],
    prefix: str = "row",
) -> tuple[list[dict[str, Any]], list[str]]:
    """Load valid row checkpoints and return rows plus missing identifiers.

    Args:
        checkpoint_dir: Directory containing one JSON file per row.
        expected_ids: Expected deterministic row IDs.
        prefix: Filename prefix used for checkpoint discovery.

    Returns:
        tuple[list[dict[str, Any]], list[str]]: Sorted rows and missing IDs.

    Raises:
        ValueError: If a checkpoint has an invalid JSON object or mismatched ID.
    """
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(checkpoint_dir.glob(f"{prefix}_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not payload.get("row_id"):
            raise ValueError(f"Invalid checkpoint object: {path}")
        row_id = str(payload["row_id"])
        if row_id in seen:
            raise ValueError(f"Duplicate checkpoint row ID: {row_id}")
        seen.add(row_id)
        rows.append(payload)
    missing = sorted(expected_ids - seen)
    unexpected = seen - expected_ids
    if unexpected:
        raise ValueError(f"Unexpected checkpoint row IDs: {sorted(unexpected)}")
    rows.sort(key=lambda row: row["row_id"])
    return rows, missing


def failure_row(row_id: str, parameters: dict[str, Any], error: Exception) -> dict[str, Any]:
    """Create a deterministic failed-row record without hiding the exception.

    Args:
        row_id: Stable row identifier.
        parameters: Row parameter mapping.
        error: Captured exception.

    Returns:
        dict[str, Any]: Failed row payload.
    """
    return {
        "row_id": row_id,
        "parameters": parameters,
        "status": "failed",
        "error_type": type(error).__name__,
        "error_message": str(error),
    }
