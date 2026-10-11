#!/usr/bin/env python3
"""Compute static Dicke ground-state QFI sweeps in coupling at fixed N.

The producer evaluates the exact ground-state gap-weighted QFI
``F_Q = 4 sum_n |<0|V|n>|^2/(E_n-E_0)^2`` (with ``V=(a+a^dagger)Jx/sqrt(N)``)
as a function of the static coupling eta for several emitter counts N at the
production-quality truncation of the historical sensitivity producer
(``n_max = 35``, ``n_eig = 30``, ``wq_wc = 1.005``).  It supplies the
panel-(b) data of the finite-size scaling figure.  The physics core is
imported from the finite-size producer so that the Hamiltonian, generator,
and QFI conventions are guaranteed identical; no dynamic, thermodynamic, or
global-bound claims are made.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from config import DEFAULT_ETA_SWEEP, EtaSweepConfig
except ImportError as exc:  # pragma: no cover - environment-specific
    raise ImportError(
        "Track B configuration is unavailable; run this script from scripts/."
    ) from exc

try:
    from finite_size_resource_scaling import _dicke_ground_qfi
except ImportError as exc:  # pragma: no cover - environment-specific
    raise ImportError(
        "The finite-size producer core is unavailable; run this script from scripts/."
    ) from exc


CONFIG = DEFAULT_ETA_SWEEP
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "dicke_eta_sweeps.json"


def _row_id(parameters: dict[str, Any]) -> str:
    """Create a stable filesystem-safe row identifier.

    Args:
        parameters: Row parameters with scalar JSON values.

    Returns:
        str: Stable row identifier prefixed with ``row_`` and a 16-character
        SHA-256 digest of the canonical JSON encoding, dimensionless.
    """
    import hashlib

    raw = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"row_{digest}"


def _eta_grid(config: EtaSweepConfig) -> np.ndarray:
    """Build the uniformly sampled coupling grid.

    Args:
        config: Centralized eta-sweep configuration.

    Returns:
        np.ndarray: Coupling values g0/wc, dimensionless, strictly increasing.
    """
    return np.linspace(config.eta_min, config.eta_max, config.eta_points)


def _anchor_index(config: EtaSweepConfig) -> int:
    """Locate the anchor coupling in the grid.

    Args:
        config: Centralized eta-sweep configuration.

    Returns:
        int: Grid index whose value equals ``anchor_eta`` exactly.

    Raises:
        ValueError: If the anchor coupling is not on the declared grid.
    """
    grid = _eta_grid(config)
    matches = np.where(np.isclose(grid, config.anchor_eta, rtol=0.0, atol=1.0e-12))[0]
    if matches.size != 1:
        raise ValueError(
            "anchor_eta must lie exactly on the declared uniform eta grid."
        )
    return int(matches[0])


def _finite_size_grid_anchor(
    n_qubits: int, eta: float, config: EtaSweepConfig
) -> dict[str, Any]:
    """Read the finite-size grid value closest to the anchor point.

    Args:
        n_qubits: Number of emitters N, dimensionless.
        eta: Anchor coupling g0/wc, dimensionless.
        config: Centralized eta-sweep configuration.

    Returns:
        dict[str, Any]: Matching grid row (N, wq_wc, eta, qfi) or an empty
        mapping when no row lies within tolerance of the anchor point.
    """
    artifact = _load_finite_size_artifact()
    if not artifact:
        return {}
    candidates = [
        row
        for row in artifact.get("rows", [])
        if row.get("row_type") == "static"
        and int(row["N"]) == n_qubits
        and abs(float(row["wq_wc"]) - config.wq_wc) <= 1.0e-9
    ]
    if not candidates:
        return {}
    best: dict[str, Any] = {}
    best_distance = float("inf")
    for row in candidates:
        distance = abs(float(row["eta"]) - eta)
        if distance < best_distance:
            best_distance = distance
            best = {
                "N": int(row["N"]),
                "wq_wc": float(row["wq_wc"]),
                "eta": float(row["eta"]),
                "qfi": float(row["qfi_high_truncation"]),
            }
    return best


_FINITE_SIZE_CACHE: dict[str, Any] = {}


def _load_finite_size_artifact() -> dict[str, Any]:
    """Load the finite-size artifact once per process.

    Returns:
        dict[str, Any]: Parsed finite-size artifact, or an empty mapping when
        the file is absent (anchor cross-check then degrades to a skip).
    """
    if "payload" not in _FINITE_SIZE_CACHE:
        path = Path(__file__).resolve().parents[1] / "data" / "finite_size_resource_scaling.json"
        if path.exists():
            _FINITE_SIZE_CACHE["payload"] = json.loads(path.read_text(encoding="utf-8"))
        else:
            _FINITE_SIZE_CACHE["payload"] = {}
    return _FINITE_SIZE_CACHE["payload"]


def simulate_row(n_qubits: int, config: EtaSweepConfig) -> dict[str, Any]:
    """Evaluate one QFI-versus-eta curve at fixed emitter count.

    Args:
        n_qubits: Number of emitters N, dimensionless.
        config: Centralized eta-sweep configuration.

    Returns:
        dict[str, Any]: JSON-ready row with the coupling grid, QFI values,
        anchor diagnostics, and finite-grid cross-check.
    """
    grid = _eta_grid(config)
    qfi_values = [
        _dicke_ground_qfi(
            n_qubits, float(eta), config.wq_wc, config.n_max, config.n_eig
        )
        for eta in grid
    ]
    anchor_idx = _anchor_index(config)
    anchor_qfi = float(qfi_values[anchor_idx])
    grid_match = _finite_size_grid_anchor(n_qubits, config.anchor_eta, config)
    if grid_match:
        deviation = abs(anchor_qfi - grid_match["qfi"]) / max(abs(grid_match["qfi"]), 1.0e-30)
    else:
        deviation = None
    parameters = {
        "row_type": "eta_sweep",
        "N": n_qubits,
        "wq_wc": config.wq_wc,
        "n_max": config.n_max,
        "n_eig": config.n_eig,
    }
    return {
        "row_id": _row_id(parameters),
        "parameters": parameters,
        "N": n_qubits,
        "eta_values": [float(v) for v in grid],
        "qfi": [float(v) for v in qfi_values],
        "anchor": {
            "eta": config.anchor_eta,
            "qfi": anchor_qfi,
            "finite_size_grid_qfi": grid_match.get("qfi"),
            "finite_size_grid_wq_wc": grid_match.get("wq_wc"),
            "relative_deviation": deviation,
            "within_tolerance": (
                deviation is not None and deviation <= config.anchor_tolerance
            ),
        },
        "eta_c_thermodynamic": 0.5 * float(np.sqrt(config.wq_wc)),
    }


def _summary(rows: list[dict[str, Any]], config: EtaSweepConfig) -> dict[str, Any]:
    """Aggregate anchor diagnostics over all rows.

    Args:
        rows: Completed eta-sweep rows.
        config: Centralized eta-sweep configuration.

    Returns:
        dict[str, Any]: JSON-ready validation block with anchor status.
    """
    deviations = [
        row["anchor"]["relative_deviation"]
        for row in rows
        if row["anchor"]["relative_deviation"] is not None
    ]
    return {
        "rows_requested": len(rows),
        "rows_completed": len(rows),
        "all_rows_complete": len(rows) == config_anchor_count(config),
        "anchors_evaluated": len(deviations),
        "max_anchor_relative_deviation": max(deviations) if deviations else None,
        "all_anchors_within_tolerance": bool(rows)
        and all(
            row["anchor"]["within_tolerance"]
            for row in rows
        ),
        "note": (
            "Anchor cross-check compares each curve at eta=anchor_eta with the "
            "finite-size grid row at the same N (wq_wc=1.005 family, high "
            "truncation). Rows without a matching grid entry are recorded with "
            "a null deviation and do not gate the artifact."
        ),
    }


def config_anchor_count(config: EtaSweepConfig) -> int:
    """Return the declared number of sweep rows.

    Args:
        config: Centralized eta-sweep configuration.

    Returns:
        int: Number of declared emitter counts, dimensionless.
    """
    return len(config.N_values)


def build_payload(config: EtaSweepConfig) -> dict[str, Any]:
    """Assemble the artifact payload.

    Args:
        config: Centralized eta-sweep configuration.

    Returns:
        dict[str, Any]: JSON-ready artifact with configuration, solver
        metadata, caveats, rows, and validation summary. No wall-clock
        fields are embedded so the file is byte-deterministic across
        regenerations.
    """
    rows = [simulate_row(n_qubits, config) for n_qubits in config.N_values]
    return {
        "schema": "trackB_dicke_eta_sweeps_v1",
        "status": "complete" if all_rows_present(rows, config) else "partial",
        "purpose": (
            "Static Dicke ground-state QFI versus coupling at fixed N "
            "(figure panel data); no thermodynamic or global-bound claim."
        ),
        "configuration": asdict(config),
        "units": {
            "eta": "dimensionless (g0/wc)",
            "qfi": "dimensionless (coherent-state baseline units)",
            "N": "dimensionless (emitter count)",
            "wq_wc": "dimensionless (qubit-to-cavity frequency ratio)",
            "n_max": "dimensionless (cavity Fock-space dimension)",
            "n_eig": "dimensionless (eigenstates retained in the gap-weighted sum)",
        },
        "solver": _solver_block(),
        "method": (
            "Ground-state gap-weighted sum F_Q = 4 sum_n |<0|V|n>|^2/(E_n-E_0)^2 "
            "with V=(a+a^dagger)Jx/sqrt(N), exact QuTiP diagonalization; core "
            "imported from scripts/finite_size_resource_scaling.py."
        ),
        "caveats": [
            "QFI values are exact diagonalization results at the declared truncation; per-point cavity-truncation pairs follow the finite-size producer's size brackets.",
            "The sampled QFI maximum shifts with N; eta=anchor_eta is a fixed operating point, not a universal optimum.",
            "No thermodynamic exponent, phase transition, Heisenberg-bound violation, or global-bound claim follows from these curves.",
            "The gap-weighted sum is evaluated with a finite eigenstate count (n_eig) and is a lower estimate if higher states contribute.",
        ],
        "validation": _summary(rows, config),
        "determinism": (
            "Payload is byte-deterministic: no wall-clock or PRNG state is embedded."
        ),
        "rows": rows,
    }


def _solver_block() -> dict[str, Any]:
    """Record solver and library version metadata.

    Returns:
        dict[str, Any]: Solver description and Python/NumPy/QuTiP versions.
    """
    import platform

    import numpy
    import qutip

    return {
        "static": "exact QuTiP sparse diagonalization (gap-weighted ground-state sum)",
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "qutip": qutip.__version__,
    }


def all_rows_present(rows: list[dict[str, Any]], config: EtaSweepConfig) -> bool:
    """Check whether every declared emitter count produced a row.

    Args:
        rows: Completed rows.
        config: Centralized eta-sweep configuration.

    Returns:
        bool: True when the row count matches the declared grid, dimensionless.
    """
    return len(rows) == config_anchor_count(config)


def parse_args() -> argparse.Namespace:
    """Parse output-path overrides.

    Returns:
        argparse.Namespace: Output JSON path.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSON artifact path.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first K emitter counts (resumable chunks).")
    return parser.parse_args()


def main() -> None:
    """Run the declared eta sweeps and write the JSON artifact."""
    args = parse_args()
    config = CONFIG
    counts = config.N_values if args.limit is None else config.N_values[: args.limit]
    scoped = EtaSweepConfig(N_values=counts)
    payload = build_payload(scoped)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    summary = payload["validation"]
    print(
        "rows=%d/%d anchors_ok=%s max_dev=%s"
        % (
            summary["rows_completed"],
            summary["rows_requested"],
            summary["all_anchors_within_tolerance"],
            summary["max_anchor_relative_deviation"],
        )
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
