#!/usr/bin/env python3
"""Validate the active Dicke+DCE truncation-extension checkpoints.

This validator is fail-closed: missing, malformed, duplicate, unexpected, or
numerically rejected rows are reported and are never promoted to a complete
artifact.  The eigenvalue criterion is a numerical acceptance floor, not a
proof of exact positivity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from config import DEFAULT_DICKE_LOSS_EXTENSION
    from trackb_recompute_utils import require_schema, stable_row_id
except ImportError as exc:  # pragma: no cover
    raise ImportError("Run this validator from the Track B scripts directory.") from exc


CONFIG = DEFAULT_DICKE_LOSS_EXTENSION
SCHEMA = "trackB_dicke_dce_loss_truncation_extension_v1"


def expected_parameters() -> list[dict[str, Any]]:
    """Return the configured ordered matrix of expected parameter rows."""
    return [
        {"n_cav": int(n_cav), "kappa_rel": float(kappa)}
        for n_cav in CONFIG.cavity_truncations
        for kappa in CONFIG.kappa_values
    ]


def _load_row(path: Path, parameters: dict[str, Any]) -> dict[str, Any]:
    """Load and validate one row checkpoint."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint is not an object: {path}")
    expected_id = stable_row_id(parameters)
    if payload.get("row_id") != expected_id:
        raise ValueError(
            f"row ID mismatch in {path}: expected {expected_id}, found {payload.get('row_id')}"
        )
    if payload.get("parameters") != parameters:
        raise ValueError(f"parameter mismatch in {path}")
    status = payload.get("status")
    if status != "complete":
        raise ValueError(f"row is not numerically complete ({status!r}): {path}")
    physicality = payload.get("physicality")
    if not isinstance(physicality, dict):
        raise ValueError(f"missing physicality diagnostics: {path}")
    if not physicality.get("numerical_acceptance", False):
        raise ValueError(f"numerical physicality gate failed: {path}")
    for key in ("max_trace_error", "max_hermiticity_error", "minimum_eigenvalue"):
        value = physicality.get(key)
        if not isinstance(value, (int, float)):
            raise ValueError(f"missing physicality value {key!r}: {path}")
        if not __import__("math").isfinite(float(value)):
            raise ValueError(f"non-finite physicality value {key!r}: {path}")
    declared_floor = physicality.get("eigenvalue_floor")
    if declared_floor is None or not __import__("math").isfinite(float(declared_floor)):
        raise ValueError(f"missing numerical eigenvalue floor: {path}")
    if float(physicality["minimum_eigenvalue"]) < float(declared_floor):
        raise ValueError(
            f"minimum_eigenvalue {physicality['minimum_eigenvalue']:.3e} below declared "
            f"floor {declared_floor:.3e}: {path}"
        )
    required = (
        "qfi_peak",
        "qfi_late_mean",
        "photon_peak",
        "photon_mean",
        "qfi_trace",
        "photon_trace",
        "time_drive_periods",
        "resource_profile",
        "solver",
        "qfi_retention_vs_same_truncation_zero_loss",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"missing row fields in {path}: {missing}")
    return payload


def validate(checkpoint_dir: Path) -> dict[str, Any]:
    """Validate all expected rows and the 16/20 truncation comparison."""
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    failures: list[str] = []
    expected = expected_parameters()
    expected_ids = {stable_row_id(parameters) for parameters in expected}
    seen_ids: set[str] = set()

    for parameters in expected:
        path = checkpoint_dir / f"{stable_row_id(parameters)}.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        try:
            row = _load_row(path, parameters)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            failures.append(str(exc))
            continue
        row_id = row["row_id"]
        if row_id in seen_ids:
            failures.append(f"duplicate row ID: {row_id}")
        seen_ids.add(row_id)
        rows.append(row)

    for path in sorted(checkpoint_dir.glob("row_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            failures.append(f"invalid JSON: {path}: {exc}")
            continue
        row_id = payload.get("row_id") if isinstance(payload, dict) else None
        if row_id and row_id not in expected_ids:
            failures.append(f"unexpected row checkpoint: {path} ({row_id})")

    by_key = {
        (int(row["parameters"]["n_cav"]), float(row["parameters"]["kappa_rel"])): row
        for row in rows
    }
    truncation_comparison: list[dict[str, Any]] = []
    for kappa in CONFIG.kappa_values:
        low = by_key.get((min(CONFIG.cavity_truncations), float(kappa)))
        high = by_key.get((max(CONFIG.cavity_truncations), float(kappa)))
        if low is None or high is None:
            continue
        low_peak = float(low["qfi_peak"])
        high_peak = float(high["qfi_peak"])
        relative_difference = abs(high_peak - low_peak) / max(abs(high_peak), 1.0)
        within_tolerance = relative_difference <= CONFIG.truncation_convergence_tolerance
        truncation_comparison.append({
            "kappa_rel": float(kappa),
            "relative_peak_qfi_difference": relative_difference,
            "within_tolerance": within_tolerance,
        })
        if not within_tolerance:
            failures.append(
                f"truncation difference exceeds tolerance at kappa={kappa}: "
                f"{relative_difference:.6g} > {CONFIG.truncation_convergence_tolerance:.6g}"
            )

    return {
        "schema": SCHEMA,
        "status": "PASS" if not missing and not failures else "FAIL",
        "expected_rows": len(expected),
        "valid_rows": len(rows),
        "missing_checkpoints": missing,
        "failures": failures,
        "truncation_convergence_tolerance": CONFIG.truncation_convergence_tolerance,
        "truncation_comparison": truncation_comparison,
    }


def validate_merged_artifact(artifact_path: Path) -> dict[str, Any]:
    """Validate the merged truncation-extension artifact against its declared schema.

    This is a complementary, fail-closed check for the canonical merged JSON
    artifact. It reproduces the row-level completeness and truncation-comparison
    logic from ``validate`` directly from the merged ``rows`` block, so the
    merged artifact can be audited independently of the raw per-row checkpoint
    directory.
    """
    if not artifact_path.is_file():
        raise FileNotFoundError(f"merged artifact not found: {artifact_path}")

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("merged artifact is not an object")
    if payload.get("schema") != SCHEMA:
        raise ValueError(
            f"schema mismatch: expected {SCHEMA!r}, found {payload.get('schema')!r}"
        )

    validation = payload.get("validation")
    if not isinstance(validation, dict):
        raise ValueError("merged artifact is missing a validation block")

    rows_block = payload.get("rows")
    if not isinstance(rows_block, list):
        raise ValueError("merged artifact rows block is not a list")

    expected = expected_parameters()
    expected_ids = [stable_row_id(parameters) for parameters in expected]
    expected_set = set(expected_ids)

    row_ids: list[str] = []
    loaded_rows = 0
    complete_rows = 0
    missing_ids: list[str] = []
    inconclusive_or_failed_ids: list[str] = []

    for row in rows_block:
        if not isinstance(row, dict):
            raise ValueError("merged artifact contains a non-object row")
        row_id = row.get("row_id")
        if not row_id:
            raise ValueError("merged artifact row is missing row_id")
        row_ids.append(row_id)
        if row_id not in expected_set:
            raise ValueError(
                f"unexpected row in merged artifact: {row_id}"
            )
        loaded_rows += 1
        if row.get("status") == "complete":
            complete_rows += 1
        else:
            inconclusive_or_failed_ids.append(row_id)

    if set(row_ids) != expected_set:
        missing_ids = sorted(expected_set - set(row_ids))

    by_key = {
        (int(row["parameters"]["n_cav"]), float(row["parameters"]["kappa_rel"])): row
        for row in rows_block
    }
    truncation_comparison: list[dict[str, Any]] = []
    for kappa in CONFIG.kappa_values:
        low = by_key.get((min(CONFIG.cavity_truncations), float(kappa)))
        high = by_key.get((max(CONFIG.cavity_truncations), float(kappa)))
        if low is None or high is None:
            continue
        low_peak = float(low["qfi_peak"])
        high_peak = float(high["qfi_peak"])
        relative_difference = abs(high_peak - low_peak) / max(abs(high_peak), 1.0)
        within_tolerance = relative_difference <= CONFIG.truncation_convergence_tolerance
        truncation_comparison.append({
            "kappa_rel": float(kappa),
            "relative_peak_qfi_difference": relative_difference,
            "within_tolerance": within_tolerance,
        })

    all_rows_complete = loaded_rows == len(expected) and complete_rows == len(expected)
    all_available_comparisons_within_tolerance = all(
        item["within_tolerance"] for item in truncation_comparison
    )

    return {
        "artifact_path": str(artifact_path),
        "schema": SCHEMA,
        "status": "PASS"
        if all_rows_complete and not missing_ids and not inconclusive_or_failed_ids
        else "FAIL",
        "expected_rows": len(expected),
        "loaded_rows": loaded_rows,
        "complete_rows": complete_rows,
        "missing_row_ids": missing_ids,
        "inconclusive_or_failed_row_ids": sorted(inconclusive_or_failed_ids),
        "all_rows_complete": all_rows_complete,
        "all_available_truncation_comparisons_within_tolerance": all_available_comparisons_within_tolerance,
        "truncation_convergence_tolerance": CONFIG.truncation_convergence_tolerance,
        "truncation_comparison": truncation_comparison,
    }


def main() -> int:
    """Run the fail-closed checkpoint validation gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data"
        / "_cache"
        / "dicke_dce_loss_truncation_extension",
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--merged-artifact",
        type=Path,
        default=None,
        help="validate the canonical merged JSON artifact instead of raw checkpoints",
    )
    args = parser.parse_args()
    if args.merged_artifact:
        report = validate_merged_artifact(args.merged_artifact)
        output = args.report or args.merged_artifact.parent / "merged_validation_report.json"
    else:
        report = validate(args.checkpoint_dir)
        output = args.report or args.checkpoint_dir / "validation_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
