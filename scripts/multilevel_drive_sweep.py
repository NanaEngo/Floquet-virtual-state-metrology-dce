#!/usr/bin/env python3
"""Run a staged multilevel drive-amplitude and Pareto screen.

This producer evaluates only the declared effective-spectrum model, not a full
transmon/flux-qubit/fluxonium circuit.  It keeps leakage, QFI, photon number,
and drive-amplitude proxy as separate objectives; no usefulness threshold or
single unsupported advantage factor is applied.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

try:
    from config import DEFAULT_MULTILEVEL, DEFAULT_MULTILEVEL_DRIVE_SWEEP
    from trackb_recompute_utils import assert_finite_json, atomic_write_json, failure_row, stable_row_id
except ImportError as exc:  # pragma: no cover
    raise ImportError("Run this producer from the Track B scripts directory.") from exc

import multilevel_emitters as engine


CONFIG = DEFAULT_MULTILEVEL_DRIVE_SWEEP
SCHEMA = "trackB_multilevel_drive_sweep_v1"


def parameter_rows() -> list[dict[str, Any]]:
    """Return the deterministic amplitude-screening grid."""
    return [
        {
            "model": model,
            "gd": float(gd),
            "kappa_rel": float(kappa),
            "d_e": int(d_e),
            "eta": float(CONFIG.eta),
            "drive_offset": float(CONFIG.drive_offset),
        }
        for model in CONFIG.models
        for d_e in CONFIG.emitter_levels
        for gd in CONFIG.drive_coefficients
        for kappa in CONFIG.kappa_values
    ]


def _engine_config() -> Any:
    """Create the existing multilevel configuration for the staged screen."""
    return replace(
        DEFAULT_MULTILEVEL,
        emitter_levels=CONFIG.emitter_levels,
        eta=CONFIG.eta,
        n_cav=CONFIG.n_cav,
        n_drive_periods=CONFIG.drive_periods,
        n_time_points=CONFIG.time_points,
        delta_eta=CONFIG.delta_eta,
        leakage_tolerance=CONFIG.leakage_tolerance,
        gamma1_rel=CONFIG.gamma1_rel,
        gamma_phi_rel=CONFIG.gamma_phi_rel,
        solver_atol=CONFIG.solver_atol,
        solver_rtol=CONFIG.solver_rtol,
        numerical_eigenvalue_floor=CONFIG.numerical_eigenvalue_floor,
    )


def _run_row(parameters: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one staged amplitude row through the existing engine."""
    config = _engine_config()
    transitions = config.transition_ratios_by_model[parameters["model"]]
    metrics = engine._driven_row(parameters, transitions, config)
    physicality = {
        "reference": metrics.get("physicality", {}),
        "perturbed": metrics.get("perturbed_physicality", {}),
    }
    accepted = all(
        bool(item.get("numerical_acceptance", False))
        for item in physicality.values()
    ) and np.isfinite(float(metrics["qfi_peak"])) and np.isfinite(float(metrics["leakage_max"]))
    pump_proxy = float(parameters["gd"] ** 2)
    photon_proxy = max(float(metrics["photon_number_peak"]), 1.0e-30)
    return {
        "schema": SCHEMA,
        "row_id": stable_row_id(parameters),
        "parameters": parameters,
        "status": "complete" if accepted else "inconclusive",
        "qfi_peak": float(metrics["qfi_peak"]),
        "qfi_final": float(metrics["qfi_final"]),
        "leakage_max": float(metrics["leakage_max"]),
        "leakage_at_qfi_peak": float(metrics["leakage_at_qfi_peak"]),
        "photon_number_peak": float(metrics["photon_number_peak"]),
        "pump_proxy_gd_squared": pump_proxy,
        "qfi_per_pump_proxy": float(metrics["qfi_peak"] / max(pump_proxy, 1.0e-30)),
        "qfi_per_peak_photon": float(metrics["qfi_peak"] / photon_proxy),
        "two_level_adequate": metrics["two_level_adequate"],
        "physicality": physicality,
        "solver": metrics["solver"],
        "leakage_tolerance": float(CONFIG.leakage_tolerance),
    }


def _pareto_rows(rows: list[dict[str, Any]]) -> list[str]:
    """Return row IDs not dominated on QFI, leakage, and pump proxy.

    A row is dominated when another row has at least as much QFI and no more
    leakage or pump proxy, with at least one strict improvement.
    """
    valid = [row for row in rows if row.get("status") == "complete"]
    frontier: list[str] = []
    for candidate in valid:
        c = (candidate["qfi_peak"], candidate["leakage_max"], candidate["pump_proxy_gd_squared"])
        dominated = False
        for other in valid:
            if other is candidate:
                continue
            o = (other["qfi_peak"], other["leakage_max"], other["pump_proxy_gd_squared"])
            no_worse = o[0] >= c[0] and o[1] <= c[1] and o[2] <= c[2]
            strict = o[0] > c[0] or o[1] < c[1] or o[2] < c[2]
            if no_worse and strict:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate["row_id"])
    return sorted(frontier)


def parse_args() -> argparse.Namespace:
    """Parse staged producer options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "multilevel_drive_sweep.json")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--row-index", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run selected rows and write a deterministic merged screen."""
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir or args.output.parent / "_cache" / "multilevel_drive_sweep"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    all_parameters = parameter_rows()
    if args.row_index is not None:
        if not 0 <= args.row_index < len(all_parameters):
            raise ValueError(f"row index outside configured grid: {args.row_index}")
        selected = [all_parameters[args.row_index]]
    elif args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        selected = all_parameters[:args.limit]
    else:
        selected = all_parameters
    results: list[dict[str, Any]] = []
    for parameters in selected:
        row_id = stable_row_id(parameters)
        path = checkpoint_dir / f"{row_id}.json"
        if path.is_file() and not args.no_resume:
            row = json.loads(path.read_text(encoding="utf-8"))
        else:
            try:
                row = _run_row(parameters)
            except Exception as error:
                row = failure_row(row_id, parameters, error)
                row["schema"] = SCHEMA
            assert_finite_json(row)
            atomic_write_json(path, row)
        results.append(row)
    results.sort(key=lambda row: row["row_id"])
    payload = {
        "schema": SCHEMA,
        "status": "staged_screen_partial" if len(results) < len(all_parameters) else "staged_screen_complete",
        "purpose": "Staged multilevel drive-amplitude screening with explicit leakage/QFI/resource proxies.",
        "configuration": asdict(CONFIG),
        "validation": {
            "rows_requested": len(selected),
            "rows_in_configured_grid": len(all_parameters),
            "rows_completed": sum(row.get("status") == "complete" for row in results),
            "inconclusive_or_failed_rows": [row["row_id"] for row in results if row.get("status") != "complete"],
            "pareto_frontier_row_ids": _pareto_rows(results),
            "pareto_objectives": ["maximize qfi_peak", "minimize leakage_max", "minimize pump_proxy_gd_squared"],
            "usefulness_threshold_applied": False,
            "physicality_note": "Numerical acceptance is not a proof of exact positivity.",
        },
        "model_note": "Effective spectra and shared ladder matrix elements are model assumptions, not full device simulations.",
        "caveats": [
            "The Pareto frontier is conditional on the declared finite grid and proxy objectives.",
            "No post hoc usefulness threshold or universal platform ranking is inferred.",
            "Failed and inconclusive rows remain visible and are excluded from the frontier.",
        ],
        "rows": results,
        "determinism": "Payload contains no wall-clock values or random state.",
    }
    assert_finite_json(payload)
    atomic_write_json(args.output, payload)
    print(f"Saved {args.output} ({len(results)}/{len(all_parameters)} selected-grid rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
