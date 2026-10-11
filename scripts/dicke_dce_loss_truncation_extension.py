#!/usr/bin/env python3
"""Run the HPC-aware larger-truncation collective Dicke+DCE loss study.

Each task is one deterministic ``(N_cav, kappa/wc)`` row.  The producer can
run one row for a SLURM array task or merge already-computed row checkpoints.
It uses atomic writes and never overwrites an existing checkpoint unless
``--no-resume`` is supplied.  Numerical acceptance is a finite numerical gate,
not a proof of exact positivity.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from config import DEFAULT_DICKE_LOSS_EXTENSION
    from trackb_recompute_utils import assert_finite_json, atomic_write_json, failure_row, stable_row_id
except ImportError as exc:  # pragma: no cover
    raise ImportError("Run this producer from the Track B scripts directory.") from exc

import dicke_dce_loss_adjusted as base


CONFIG = DEFAULT_DICKE_LOSS_EXTENSION
SCHEMA = "trackB_dicke_dce_loss_truncation_extension_v1"


def _configure_base(n_cav: int) -> None:
    """Set the validated base-producer globals for one cavity truncation.

    Args:
        n_cav: Cavity Fock-space dimension, dimensionless.
    """
    base.N_CAV = int(n_cav)
    base.N_MAX = int(n_cav)
    base.N_QUBITS = CONFIG.n_qubits
    base.J = CONFIG.n_qubits / 2.0
    base.ETA = CONFIG.eta
    base.GD_OVER_G0 = CONFIG.gd_over_g0
    base.WQ_WC = CONFIG.wq_wc
    base.WD_WC = CONFIG.wd_wc
    base.WC = 1.0
    base.WQ = CONFIG.wq_wc
    base.WD = CONFIG.wd_wc
    base.T_D = 2.0 * np.pi / base.WD
    base.T_MAX = CONFIG.drive_periods * base.T_D
    base.DELTA = CONFIG.delta_eta
    # The base producer interprets this value as an allowed negative
    # eigenvalue magnitude.  Preserve the configured signed floor in the
    # extension-level acceptance record below.
    base.PHYSICALITY_EIGENVALUE_FLOOR = CONFIG.numerical_eigenvalue_floor
    base.SOLVER_ATOL = CONFIG.solver_atol
    base.SOLVER_RTOL = CONFIG.solver_rtol
    base.H0_STATIC = None
    base.H_DRIVE = None


def parameter_rows() -> list[dict[str, Any]]:
    """Return the deterministic extension grid.

    Returns:
        list[dict[str, Any]]: Ordered task parameter mappings.
    """
    return [
        {"n_cav": int(n_cav), "kappa_rel": float(kappa)}
        for n_cav in CONFIG.cavity_truncations
        for kappa in CONFIG.kappa_values
    ]


def _physicality_summary(physicality: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the base producer's nested physicality diagnostics.

    Args:
        physicality: Per-time dictionaries containing ``rho0`` and ``rhod``
            diagnostic mappings.

    Returns:
        dict[str, Any]: Flattened numerical acceptance summary.
    """
    diagnostics = [
        diagnostic
        for sample in physicality
        for diagnostic in (sample.get("rho0", {}), sample.get("rhod", {}))
    ]
    if not diagnostics:
        raise ValueError("No physicality diagnostics were returned")
    max_trace_error = max(float(item["trace_error"]) for item in diagnostics)
    max_hermiticity_error = max(float(item["hermiticity_error"]) for item in diagnostics)
    minimum_eigenvalue = min(float(item["min_eigenvalue"]) for item in diagnostics)
    accepted = (
        np.isfinite(max_trace_error)
        and np.isfinite(max_hermiticity_error)
        and np.isfinite(minimum_eigenvalue)
        and max_trace_error <= 1.0e-8
        and max_hermiticity_error <= 1.0e-8
        and minimum_eigenvalue >= CONFIG.numerical_eigenvalue_floor
    )
    return {
        "max_trace_error": max_trace_error,
        "max_hermiticity_error": max_hermiticity_error,
        "minimum_eigenvalue": minimum_eigenvalue,
        "eigenvalue_floor": CONFIG.numerical_eigenvalue_floor,
        "numerical_acceptance": bool(accepted),
        "samples_checked": len(diagnostics),
        "note": "Numerical acceptance is not a proof of exact positivity.",
    }


def _run_row(parameters: dict[str, Any], n_steps: int | None = None) -> dict[str, Any]:
    """Run and validate one collective-loss row.

    Args:
        parameters: Row parameters containing ``n_cav`` and ``kappa_rel``.
        n_steps: Optional bounded number of output times, dimensionless.

    Returns:
        dict[str, Any]: JSON-ready completed or inconclusive row.
    """
    n_cav = int(parameters["n_cav"])
    kappa_rel = float(parameters["kappa_rel"])
    _configure_base(n_cav)
    output_points = int(n_steps or CONFIG.time_points)
    if output_points < 2:
        raise ValueError("n_steps must be at least 2")
    tlist, qfi, photons, physicality = base.compute_dicke_dce_qfi(
        kappa_rel, n_steps=output_points
    )
    qfi = np.asarray(qfi, dtype=float)
    photons = np.asarray(photons, dtype=float)
    physicality_summary = _physicality_summary(physicality)
    finite_traces = np.all(np.isfinite(qfi)) and np.all(np.isfinite(photons))
    nonnegative_qfi = np.all(qfi >= -1.0e-8)
    accepted = bool(physicality_summary["numerical_acceptance"] and finite_traces and nonnegative_qfi)
    peak_index = int(np.argmax(qfi))
    late_count = max(1, int(round(len(qfi) * CONFIG.late_window_fraction)))
    return {
        "schema": SCHEMA,
        "row_id": stable_row_id(parameters),
        "parameters": parameters,
        "status": "complete" if accepted else "inconclusive",
        "qfi_peak": float(np.max(qfi)),
        "qfi_peak_time_periods": float(tlist[peak_index] / base.T_D),
        "qfi_late_mean": float(np.mean(qfi[-late_count:])),
        "photon_peak": float(np.max(photons)),
        "photon_mean": float(np.mean(photons)),
        "qfi_trace": qfi.tolist(),
        "photon_trace": photons.tolist(),
        "time_drive_periods": (tlist / base.T_D).tolist(),
        "physicality": physicality_summary,
        "resource_profile": asdict(CONFIG.resources),
        "zero_loss_reference_peak": float(np.max(qfi)) if kappa_rel == 0.0 else None,
        "qfi_retention_vs_same_truncation_zero_loss": 1.0 if kappa_rel == 0.0 else None,
        "solver": {
            "name": "qutip.mesolve",
            "atol": CONFIG.solver_atol,
            "rtol": CONFIG.solver_rtol,
            "output_points": output_points,
        },
    }


def parse_args() -> argparse.Namespace:
    """Parse task, merge, and bounded-smoke options.

    Returns:
        argparse.Namespace: Validated command-line options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--n-cav", type=int, choices=CONFIG.cavity_truncations)
    parser.add_argument("--kappa", type=float)
    parser.add_argument("--n-steps", type=int, default=None, help="Override output points for a bounded smoke run.")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def _checkpoint_dir(args: argparse.Namespace) -> Path:
    """Resolve the extension checkpoint directory."""
    return args.checkpoint_dir or Path(__file__).resolve().parents[1] / "data" / "_cache" / "dicke_dce_loss_truncation_extension"


def _merge(args: argparse.Namespace, checkpoint_dir: Path, output: Path) -> int:
    """Merge valid checkpoints without hiding missing or failed rows."""
    loaded: list[dict[str, Any]] = []
    expected = parameter_rows()
    expected_ids = {stable_row_id(row) for row in expected}
    for parameters in expected:
        path = checkpoint_dir / f"{stable_row_id(parameters)}.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema") != SCHEMA:
                raise ValueError(f"Unexpected checkpoint schema: {path}")
            expected_id = stable_row_id(parameters)
            if payload.get("row_id") != expected_id or payload.get("parameters") != parameters:
                raise ValueError(f"Checkpoint identity mismatch: {path}")
            loaded.append(payload)
    loaded.sort(key=lambda item: item["row_id"])
    # Derive retention only against the zero-loss row at the same cavity
    # truncation.  Missing or inconclusive references remain explicit None.
    zero_loss_by_n_cav = {
        int(item["parameters"]["n_cav"]): item
        for item in loaded
        if item.get("status") == "complete"
        and float(item["parameters"]["kappa_rel"]) == 0.0
    }
    for item in loaded:
        n_cav = int(item["parameters"]["n_cav"])
        kappa = float(item["parameters"]["kappa_rel"])
        reference = zero_loss_by_n_cav.get(n_cav)
        if reference is None or reference.get("qfi_peak", 0.0) == 0.0:
            item["qfi_retention_vs_same_truncation_zero_loss"] = None
        else:
            item["qfi_retention_vs_same_truncation_zero_loss"] = float(
                item.get("qfi_peak", 0.0) / reference["qfi_peak"]
            ) if item.get("status") == "complete" else None
        item["retention_reference"] = {
            "n_cav": n_cav,
            "kappa_rel": 0.0,
            "available": reference is not None,
            "same_truncation": True,
            "note": "Retention is normalized to the complete same-truncation zero-loss row.",
        }
    loaded_ids = [item["row_id"] for item in loaded]
    if len(loaded_ids) != len(set(loaded_ids)):
        raise ValueError("Duplicate row IDs in merged checkpoints")
    missing = sorted(expected_ids - set(loaded_ids))
    failed = sorted(item["row_id"] for item in loaded if item.get("status") != "complete")
    truncation_comparison = []
    for kappa in CONFIG.kappa_values:
        low = next(
            (item for item in loaded if item.get("status") == "complete"
             and int(item["parameters"]["n_cav"]) == min(CONFIG.cavity_truncations)
             and float(item["parameters"]["kappa_rel"]) == float(kappa)),
            None,
        )
        high = next(
            (item for item in loaded if item.get("status") == "complete"
             and int(item["parameters"]["n_cav"]) == max(CONFIG.cavity_truncations)
             and float(item["parameters"]["kappa_rel"]) == float(kappa)),
            None,
        )
        if low is None or high is None:
            truncation_comparison.append({
                "kappa_rel": float(kappa),
                "available": False,
                "relative_peak_qfi_difference": None,
                "within_tolerance": False,
            })
            continue
        difference = abs(float(high["qfi_peak"]) - float(low["qfi_peak"])) / max(abs(float(high["qfi_peak"])), 1.0)
        truncation_comparison.append({
            "kappa_rel": float(kappa),
            "available": True,
            "n_cav_low": min(CONFIG.cavity_truncations),
            "n_cav_high": max(CONFIG.cavity_truncations),
            "relative_peak_qfi_difference": float(difference),
            "within_tolerance": bool(difference <= CONFIG.truncation_convergence_tolerance),
        })
    convergence_pass = bool(
        not missing
        and not failed
        and all(item["within_tolerance"] for item in truncation_comparison if item["available"])
        and any(item["available"] for item in truncation_comparison)
    )
    payload = {
        "schema": SCHEMA,
        "status": "complete" if not missing and not failed and convergence_pass else "partial_or_inconclusive",
        "purpose": "HPC-aware collective Dicke+DCE cavity-loss truncation extension.",
        "configuration": asdict(CONFIG),
        "units": {
            "n_cav": "cavity Fock-space dimension, dimensionless",
            "kappa_rel": "kappa/wc, dimensionless",
            "qfi": "inverse squared normalized coupling",
            "time": "drive periods",
            "photon_number": "photons",
        },
        "validation": {
            "expected_rows": len(expected),
            "loaded_rows": len(loaded),
            "complete_rows": sum(item.get("status") == "complete" for item in loaded),
            "missing_row_ids": missing,
            "inconclusive_or_failed_row_ids": failed,
            "all_rows_complete": not missing and not failed,
            "truncation_convergence_tolerance": CONFIG.truncation_convergence_tolerance,
            "truncation_comparison": truncation_comparison,
            "all_available_truncation_comparisons_within_tolerance": convergence_pass,
            "numerical_physicality_note": "Numerical acceptance is not a proof of exact positivity.",
        },
        "caveats": [
            "This extension does not extrapolate beyond the largest computed cavity truncation.",
            "Retention must be normalized to the same-truncation zero-loss reference.",
            "The reduced collective model is not a device-level validation.",
        ],
        "rows": loaded,
        "determinism": "No wall-clock values are embedded in the merged artifact.",
    }
    assert_finite_json(payload)
    atomic_write_json(output, payload)
    print(f"Merged {len(loaded)}/{len(expected)} rows into {output}")
    return 0 if payload["status"] == "complete" else 2


def main() -> int:
    """Run one resumable row or merge the configured extension matrix."""
    args = parse_args()
    checkpoint_dir = _checkpoint_dir(args)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output = args.output or Path(__file__).resolve().parents[1] / "data" / "dicke_dce_loss_truncation_extension.json"
    if args.merge:
        return _merge(args, checkpoint_dir, output)
    if args.n_cav is None or args.kappa is None:
        raise SystemExit("Task mode requires --n-cav and --kappa; use --merge to merge checkpoints.")
    parameters = {"n_cav": int(args.n_cav), "kappa_rel": float(args.kappa)}
    if parameters not in parameter_rows():
        raise ValueError(f"Requested row is outside configured grid: {parameters}")
    row_id = stable_row_id(parameters)
    checkpoint = checkpoint_dir / f"{row_id}.json"
    if checkpoint.exists() and not args.no_resume:
        existing = json.loads(checkpoint.read_text(encoding="utf-8"))
        if existing.get("schema") == SCHEMA and existing.get("row_id") == row_id:
            print(f"Resumed existing checkpoint: {checkpoint}")
            return 0
        raise ValueError(f"Existing checkpoint is incompatible: {checkpoint}")
    try:
        row = _run_row(parameters, n_steps=args.n_steps)
    except Exception as error:  # preserve task failure for array continuation
        row = failure_row(row_id, parameters, error)
        row["schema"] = SCHEMA
    assert_finite_json(row)
    atomic_write_json(checkpoint, row)
    print(f"Wrote {row['status']} row {row_id} to {checkpoint}")
    return 0 if row.get("status") == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
