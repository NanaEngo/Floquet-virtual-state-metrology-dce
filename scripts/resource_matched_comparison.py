#!/usr/bin/env python3
"""Compute scenario-based resource proxies for Track B comparisons.

The producer intentionally does not collapse resources into one universal
advantage factor.  It reports separate time, pump, emitted-photon, and
readout/detection proxies for the declared scenarios.  These are bookkeeping
models, not calibrated experimental costs or a claim of superiority.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

try:
    from config import DEFAULT_RESOURCE_MATCHED
    from trackb_recompute_utils import assert_finite_json, atomic_write_json, stable_row_id
except ImportError as exc:  # pragma: no cover
    raise ImportError("Run this producer from the Track B scripts directory.") from exc


CONFIG = DEFAULT_RESOURCE_MATCHED
SCHEMA = "trackB_resource_matched_comparison_v1"


def parameter_rows() -> list[dict[str, Any]]:
    """Return the deterministic resource-comparison scenario grid."""
    return [
        {
            "scenario": scenario,
            "kappa_rel": float(kappa),
            "observation_periods": int(periods),
        }
        for scenario in CONFIG.scenarios
        for kappa in CONFIG.loss_values
        for periods in CONFIG.observation_periods
    ]


def _row(parameters: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one declared scenario using separate resource proxies.

    Args:
        parameters: Scenario, normalized loss, and observation window.

    Returns:
        dict[str, Any]: JSON-ready proxy accounting row.
    """
    kappa = float(parameters["kappa_rel"])
    periods = float(parameters["observation_periods"])
    scenario = str(parameters["scenario"])
    # These quantities are intentionally dimensionless proxies.  They are not
    # inferred from a device calibration and are never combined into a claim.
    time_proxy = periods + float(CONFIG.readout_overhead)
    pump_proxy = periods if scenario == "pump_proxy_matched" else 1.0
    emitted_photon_proxy = kappa * periods
    detection_proxy = 1.0 / max(float(CONFIG.detector_efficiency), 1.0e-30)
    if scenario == "photon_matched":
        matched_resource = emitted_photon_proxy
    elif scenario == "pump_proxy_matched":
        matched_resource = pump_proxy
    elif scenario == "time_loss_matched":
        matched_resource = time_proxy * (1.0 + kappa)
    else:  # pragma: no cover - guarded by config but retained fail-closed
        raise ValueError(f"Unsupported scenario: {scenario}")
    return {
        "schema": SCHEMA,
        "row_id": stable_row_id(parameters),
        "parameters": parameters,
        "status": "complete",
        "proxies": {
            "time_proxy_periods": float(time_proxy),
            "pump_proxy": float(pump_proxy),
            "emitted_photon_proxy": float(emitted_photon_proxy),
            "detection_readout_proxy": float(detection_proxy),
            "scenario_matched_resource_proxy": float(matched_resource),
        },
        "interpretation": {
            "separate_objectives": True,
            "universal_advantage_factor": False,
            "calibrated_experimental_cost": False,
        },
    }


def parse_args() -> argparse.Namespace:
    """Parse output and optional bounded-row options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "resource_matched_comparison.json")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--row-index", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    """Generate the deterministic scenario-based resource artifact."""
    args = parse_args()
    parameters = parameter_rows()
    if args.row_index is not None:
        if not 0 <= args.row_index < len(parameters):
            raise ValueError(f"row index outside configured grid: {args.row_index}")
        selected = [parameters[args.row_index]]
    elif args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        selected = parameters[:args.limit]
    else:
        selected = parameters
    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in selected:
        row_id = stable_row_id(item)
        checkpoint = checkpoint_dir / f"{row_id}.json" if checkpoint_dir is not None else None
        if checkpoint is not None and checkpoint.is_file():
            row = json.loads(checkpoint.read_text(encoding="utf-8"))
        else:
            row = _row(item)
            if checkpoint is not None:
                atomic_write_json(checkpoint, row)
        rows.append(row)
    payload = {
        "schema": SCHEMA,
        "status": "scenario_proxy_screen",
        "purpose": "Keep resource assumptions separate while comparing declared Track B scenarios.",
        "configuration": asdict(CONFIG),
        "units": {
            "loss": "kappa/wc, dimensionless",
            "observation": "drive periods, dimensionless",
            "time_proxy": "drive-period-equivalent bookkeeping units",
            "pump_proxy": "dimensionless declared proxy",
            "emitted_photon_proxy": "kappa_rel * observation_periods, dimensionless proxy",
            "detection_readout_proxy": "inverse detector efficiency, dimensionless proxy",
        },
        "validation": {
            "rows_requested": len(selected),
            "rows_in_configured_grid": len(parameters),
            "all_rows_complete": len(rows) == len(selected),
            "separate_resource_axes": ["time", "pump", "emitted_photon", "detection_readout"],
            "universal_advantage_factor_claim": False,
            "calibrated_cost_claim": False,
        },
        "caveats": [
            "The proxies are scenario assumptions, not measured pump power, output flux, detector burden, or laboratory time.",
            "Rows do not contain QFI and must not be read as a performance ranking without a separately validated signal model.",
            "No universal resource-normalized advantage is claimed.",
        ],
        "rows": rows,
        "determinism": "Payload contains no wall-clock values or random state.",
    }
    assert_finite_json(payload)
    atomic_write_json(args.output, payload)
    print(f"Saved {args.output} ({len(rows)}/{len(parameters)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
