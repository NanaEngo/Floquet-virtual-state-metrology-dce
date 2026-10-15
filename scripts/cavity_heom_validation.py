#!/usr/bin/env python3
"""Run a scoped cavity-mode structured-bath HEOM/Lindblad benchmark.

The current implementation is an explicit adapter around the repository's
local HEOM engine.  It uses a two-level system coupled through sigma-x to a
Drude-Lorentz bath, which is a cavity-mode surrogate rather than a full cavity
HEOM calculation.  The distinction is recorded in every merged artifact.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

try:
    from config import DEFAULT_CAVITY_HEOM, DEFAULT_HEOM_EXTENDED
    from trackb_recompute_utils import assert_finite_json, atomic_write_json, failure_row, stable_row_id
except ImportError as exc:  # pragma: no cover
    raise ImportError("Run this producer from the Track B scripts directory.") from exc

import heom_extended_validation as engine


CONFIG = DEFAULT_CAVITY_HEOM
SCHEMA = "trackB_cavity_heom_validation_v1"


def parameter_rows() -> list[dict[str, Any]]:
    """Return the deterministic structured-bath benchmark grid."""
    return [
        {
            "depth": int(depth),
            "lam": float(lam),
            "gamma": float(gamma),
            "temperature": float(temperature),
            "t_final": float(t_final),
        }
        for depth in CONFIG.hierarchy_depths
        for lam in CONFIG.bath_couplings
        for gamma in CONFIG.bath_cutoffs
        for temperature in CONFIG.temperatures
        for t_final in CONFIG.final_times
    ]


def _engine_config() -> Any:
    """Map cavity benchmark settings to the existing normalized HEOM engine."""
    return replace(
        DEFAULT_HEOM_EXTENDED,
        wq=CONFIG.cavity_frequency,
        wd=2.0 * CONFIG.cavity_frequency,
        hierarchy_depths=CONFIG.hierarchy_depths,
        couplings_lam=CONFIG.bath_couplings,
        cutoffs_gamma=CONFIG.bath_cutoffs,
        temperatures=CONFIG.temperatures,
        final_times=CONFIG.final_times,
        n_time_points=CONFIG.time_points,
        matsurbara_terms=CONFIG.matsubara_terms,
        solver_atol=CONFIG.solver_atol,
        solver_rtol=CONFIG.solver_rtol,
        numerical_eigenvalue_floor=CONFIG.numerical_eigenvalue_floor,
    )


def _run_row(parameters: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one structured-bath row with the existing HEOM engine."""
    metrics = engine._simulate_row(parameters, _engine_config())
    status = "complete" if metrics.get("valid_for_agreement") else "inconclusive"
    return {
        "schema": SCHEMA,
        "row_id": stable_row_id(parameters),
        "parameters": parameters,
        "status": status,
        "metrics": metrics,
        "scope": "two-level sigma-x Drude-Lorentz structured-bath surrogate; not full cavity-mode HEOM",
    }


def parse_args() -> argparse.Namespace:
    """Parse bounded row/checkpoint options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "cavity_heom_validation.json")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--row-index", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run selected rows and merge a scoped structured-bath artifact."""
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir or args.output.parent / "_cache" / "cavity_heom_validation"
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
    rows: list[dict[str, Any]] = []
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
                row["scope"] = "two-level sigma-x Drude-Lorentz structured-bath surrogate; not full cavity-mode HEOM"
            assert_finite_json(row)
            atomic_write_json(path, row)
        rows.append(row)
    rows.sort(key=lambda row: row["row_id"])
    payload = {
        "schema": SCHEMA,
        "status": "partial_scoped_surrogate_screen" if len(rows) < len(all_parameters) else "scoped_surrogate_screen_complete",
        "purpose": "Benchmark the working Lindblad model against a declared structured-bath HEOM surrogate.",
        "configuration": asdict(CONFIG),
        "scope": {
            "system": "two-level system",
            "bath": "Drude-Lorentz structured bath coupled through sigma-x",
            "cavity_mode_included": False,
            "full_cavity_heom_claim": False,
            "relationship_to_track_b": "local surrogate benchmark only; does not validate the collective Dicke protocol",
        },
        "validation": {
            "rows_requested": len(selected),
            "rows_in_configured_grid": len(all_parameters),
            "rows_complete": sum(row.get("status") == "complete" for row in rows),
            "inconclusive_or_failed_rows": [row["row_id"] for row in rows if row.get("status") != "complete"],
            "numerical_physicality_note": "Numerical acceptance is not a proof of exact positivity.",
        },
        "caveats": [
            "This is not a cavity-mode HEOM implementation; the requested cavity benchmark remains explicitly bounded to this surrogate.",
            "Agreement is conditional on the tested bath, hierarchy, temperature, and finite time window.",
            "Failed or inconclusive rows are retained and excluded from agreement summaries.",
        ],
        "rows": rows,
        "determinism": "Payload contains no wall-clock values or random state.",
    }
    assert_finite_json(payload)
    atomic_write_json(args.output, payload)
    print(f"Saved {args.output} ({len(rows)}/{len(all_parameters)} selected-grid rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
