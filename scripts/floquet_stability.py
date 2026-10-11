#!/usr/bin/env python3
"""Run a resumable Tier 1 Floquet-stability screen for Track B.

The producer compares finite-window photon-number and coupling-estimand QFI
trajectories for the driven single-qubit Rabi model across cavity truncations,
drive parameters, and cavity-loss rates.  It deliberately reports only
finite-window classifications; it does not infer an asymptotic Floquet
stability theorem.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import qutip as qt

try:
    from config import DEFAULT_FLOQUET_STABILITY, FloquetStabilityConfig
except ImportError as exc:  # pragma: no cover - environment-specific
    raise ImportError(
        "Track B configuration is unavailable; run this script from scripts/."
    ) from exc


CONFIG = DEFAULT_FLOQUET_STABILITY


def _build_operators(n_cav: int) -> tuple[qt.Qobj, qt.Qobj, qt.Qobj, qt.Qobj]:
    """Construct tensor-ordered single-qubit Rabi operators.

    Args:
        n_cav: Cavity Fock-space dimension, dimensionless.

    Returns:
        tuple[qt.Qobj, qt.Qobj, qt.Qobj, qt.Qobj]: Annihilation operator,
        cavity number, qubit sigma-x, and qubit sigma-z in normalized units.
    """
    a = qt.tensor(qt.destroy(n_cav), qt.qeye(2))
    number = a.dag() * a
    sigma_x = qt.tensor(qt.qeye(n_cav), qt.sigmax())
    sigma_z = qt.tensor(qt.qeye(n_cav), qt.sigmaz())
    return a, number, sigma_x, sigma_z


def _hamiltonian(
    eta: float,
    gd_over_g0: float,
    drive_ratio: float,
    phase: float,
    n_cav: int,
    wq_wc: float,
) -> qt.QobjEvo:
    """Construct the normalized time-dependent Rabi/DCE Hamiltonian.

    Args:
        eta: Static coupling g0/wc, dimensionless.
        gd_over_g0: Modulation amplitude divided by g0, dimensionless.
        drive_ratio: Drive frequency divided by wc+wq, dimensionless.
        phase: Drive phase, radians.
        n_cav: Cavity Fock-space dimension, dimensionless.

    Returns:
        qt.QobjEvo: Hamiltonian in normalized angular-frequency units.
    """
    a, _, sigma_x, sigma_z = _build_operators(n_cav)
    drive_frequency = drive_ratio * (1.0 + wq_wc)
    interaction = sigma_x * (a + a.dag())
    static = 0.5 * wq_wc * sigma_z + a.dag() * a

    def coefficient(t: float, _args: dict[str, Any] | None = None) -> float:
        """Evaluate the dimensionless time-dependent coupling coefficient.

        Args:
            t: Time in inverse cavity angular-frequency units.
            _args: Optional QuTiP coefficient arguments, unused.

        Returns:
            float: Instantaneous coupling coefficient, dimensionless.
        """
        return eta * (1.0 + gd_over_g0 * math.cos(drive_frequency * t + phase))

    return qt.QobjEvo([static, [interaction, coefficient]])


def _squared_fidelity(state_a: qt.Qobj, state_b: qt.Qobj) -> float:
    """Compute squared fidelity for two QuTiP states.

    Args:
        state_a: First ket or density operator, dimensionless.
        state_b: Second ket or density operator, dimensionless.

    Returns:
        float: Squared Uhlmann fidelity, dimensionless and clipped to [0, 1].
    """
    rho_a = state_a.proj() if state_a.isket else state_a
    rho_b = state_b.proj() if state_b.isket else state_b
    if state_a.isket and state_b.isket:
        overlap = state_a.dag() * state_b
        return float(np.clip(abs(complex(overlap)) ** 2, 0.0, 1.0))
    root_a = rho_a.sqrtm()
    product = root_a * rho_b * root_a
    eigenvalues = np.real_if_close(product.eigenenergies()).real
    root_fidelity = float(np.sum(np.sqrt(np.clip(eigenvalues, 0.0, None))))
    return float(np.clip(root_fidelity**2, 0.0, 1.0))


def _qfi_trace(
    states_a: list[qt.Qobj],
    states_b: list[qt.Qobj],
    delta_eta: float,
) -> np.ndarray:
    """Calculate the coupling-estimand QFI from paired fidelity traces.

    Args:
        states_a: State trace at eta, dimensionless.
        states_b: State trace at eta+delta_eta, dimensionless.
        delta_eta: Coupling finite-difference step, dimensionless.

    Returns:
        np.ndarray: QFI trace for the normalized coupling estimand,
            dimensionless.
    """
    if delta_eta <= 0.0:
        raise ValueError("delta_eta must be positive")
    return np.asarray(
        [
            8.0 * (1.0 - math.sqrt(_squared_fidelity(a, b))) / delta_eta**2
            for a, b in zip(states_a, states_b)
        ],
        dtype=float,
    )


def _relative_tail_slope(values: np.ndarray, periods: np.ndarray, fraction: float) -> float:
    """Estimate a dimensionless relative growth slope in the final window.

    Args:
        values: Observable trace, dimensionless.
        periods: Time in drive periods, dimensionless.
        fraction: Fraction of the final window used for the fit, dimensionless.

    Returns:
        float: Fitted slope divided by the tail mean magnitude, per drive period.
    """
    count = max(3, int(math.ceil(len(values) * fraction)))
    tail_values = values[-count:]
    tail_periods = periods[-count:]
    slope = float(np.polyfit(tail_periods, tail_values, 1)[0])
    scale = max(float(np.mean(np.abs(tail_values))), 1.0e-12)
    return slope / scale


def _physicality_summary(
    states: list[qt.Qobj],
    numerical_eigenvalue_floor: float,
) -> dict[str, Any]:
    """Summarize trace, Hermiticity, and minimum eigenvalue diagnostics.

    Args:
        states: State trajectory, consisting of kets or density operators.
        numerical_eigenvalue_floor: Accepted minimum eigenvalue for numerical
            diagnostics, dimensionless; not a positivity proof.

    Returns:
        dict[str, Any]: Numerical acceptance diagnostics; not a proof of exact
            positivity.
    """
    traces = []
    hermiticity_errors = []
    minimum_eigenvalue = float("inf")
    for state in states:
        rho = state.proj() if state.isket else state
        traces.append(float(np.real(rho.tr())))
        hermiticity_errors.append(float((rho - rho.dag()).norm()))
        if rho.isoper:
            minimum_eigenvalue = min(
                minimum_eigenvalue,
                float(np.min(np.real_if_close(rho.eigenenergies()).real)),
            )
    return {
        "max_trace_error": max(abs(value - 1.0) for value in traces),
        "max_hermiticity_error": max(hermiticity_errors),
        "minimum_eigenvalue": minimum_eigenvalue,
        "numerical_acceptance": bool(
            max(abs(value - 1.0) for value in traces) < 1.0e-8
            and max(hermiticity_errors) < 1.0e-8
            and minimum_eigenvalue >= numerical_eigenvalue_floor
        ),
        "note": "Numerical acceptance is not a proof of exact positivity.",
    }


def _simulate_truncation(
    eta: float,
    gd_over_g0: float,
    drive_ratio: float,
    phase: float,
    kappa_rel: float,
    n_cav: int,
    config: FloquetStabilityConfig,
) -> dict[str, Any]:
    """Simulate reference and perturbed trajectories at one truncation.

    Args:
        eta: Static coupling g0/wc, dimensionless.
        gd_over_g0: Modulation amplitude divided by g0, dimensionless.
        drive_ratio: Drive frequency divided by wc+wq, dimensionless.
        phase: Drive phase, radians.
        kappa_rel: Cavity loss divided by wc, dimensionless.
        n_cav: Cavity Fock-space dimension, dimensionless.
        config: Centralized stability configuration.

    Returns:
        dict[str, Any]: JSON-ready observables, diagnostics, and classifications.
    """
    a, number, _, _ = _build_operators(n_cav)
    drive_frequency = drive_ratio * (1.0 + config.wq_wc)
    drive_period = 2.0 * math.pi / drive_frequency
    times = np.linspace(
        0.0,
        config.n_drive_periods * drive_period,
        config.n_time_points,
    )
    periods = times / drive_period
    initial_state = qt.tensor(qt.basis(n_cav, 0), qt.basis(2, 0))
    collapse_ops = [math.sqrt(kappa_rel) * a] if kappa_rel > 0.0 else []
    options = {
        "atol": config.solver_atol,
        "rtol": config.solver_rtol,
        "method": "adams",
        "max_step": config.max_step_fraction * drive_period,
        "store_states": True,
        "progress_bar": "",
    }
    hamiltonian = _hamiltonian(
        eta, gd_over_g0, drive_ratio, phase, n_cav, config.wq_wc
    )
    perturbed_hamiltonian = _hamiltonian(
        eta + config.delta_eta,
        gd_over_g0,
        drive_ratio,
        phase,
        n_cav,
        config.wq_wc,
    )
    if kappa_rel == 0.0:
        result = qt.sesolve(
            hamiltonian,
            initial_state,
            times,
            e_ops=[],
            options=options,
        )
        perturbed = qt.sesolve(
            perturbed_hamiltonian,
            initial_state,
            times,
            e_ops=[],
            options=options,
        )
        solver_name = "qutip.sesolve"
    else:
        result = qt.mesolve(
            hamiltonian,
            initial_state,
            times,
            c_ops=collapse_ops,
            e_ops=[],
            options=options,
        )
        perturbed = qt.mesolve(
            perturbed_hamiltonian,
            initial_state,
            times,
            c_ops=collapse_ops,
            e_ops=[],
            options=options,
        )
        solver_name = "qutip.mesolve"
    qfi = _qfi_trace(result.states, perturbed.states, config.delta_eta)
    photons = np.asarray(
        [float(np.real(qt.expect(number, state))) for state in result.states],
        dtype=float,
    )
    qfi_peak_index = int(np.argmax(qfi))
    photon_peak_index = int(np.argmax(photons))
    qfi_slope = _relative_tail_slope(qfi, periods, config.boundary_fraction)
    photon_slope = _relative_tail_slope(photons, periods, config.boundary_fraction)
    qfi_boundary = qfi_peak_index >= len(qfi) - max(2, int(0.05 * len(qfi)))
    photon_boundary = photon_peak_index >= len(photons) - max(2, int(0.05 * len(photons)))
    boundary_growth = (
        qfi_boundary
        or photon_boundary
        or qfi_slope > config.boundary_slope_tolerance
        or photon_slope > config.boundary_slope_tolerance
    )
    return {
        "n_cav": n_cav,
        "time_drive_periods": periods.tolist(),
        "qfi": qfi.tolist(),
        "photon_number": photons.tolist(),
        "qfi_peak": float(qfi[qfi_peak_index]),
        "qfi_peak_period": float(periods[qfi_peak_index]),
        "qfi_final": float(qfi[-1]),
        "photon_number_peak": float(photons[photon_peak_index]),
        "photon_number_peak_period": float(periods[photon_peak_index]),
        "photon_number_final": float(photons[-1]),
        "qfi_tail_relative_slope_per_period": qfi_slope,
        "photon_tail_relative_slope_per_period": photon_slope,
        "qfi_peak_at_boundary": bool(qfi_boundary),
        "photon_peak_at_boundary": bool(photon_boundary),
        "boundary_growth_flag": bool(boundary_growth),
        "solver": solver_name,
        "physicality": _physicality_summary(
            result.states, config.numerical_eigenvalue_floor
        ),
        "perturbed_physicality": _physicality_summary(
            perturbed.states, config.numerical_eigenvalue_floor
        ),
    }


def _row_id(
    eta: float,
    gd_over_g0: float,
    drive_ratio: float,
    phase: float,
    kappa_rel: float,
) -> str:
    """Create a stable filesystem-safe row identifier.

    Args:
        eta: Static coupling g0/wc, dimensionless.
        gd_over_g0: Modulation ratio, dimensionless.
        drive_ratio: Drive-frequency ratio, dimensionless.
        phase: Drive phase, radians.
        kappa_rel: Cavity loss divided by wc, dimensionless.

    Returns:
        str: Stable row identifier.
    """
    raw = f"eta={eta:.8g}|gd={gd_over_g0:.8g}|dr={drive_ratio:.8g}|phase={phase:.8g}|kappa={kappa_rel:.8g}"
    digest = hashlib.sha256(raw.encode("ascii")).hexdigest()[:12]
    return f"row_{digest}"


def _parameter_rows(config: FloquetStabilityConfig) -> list[dict[str, float]]:
    """Enumerate the deterministic Tier 1 parameter grid.

    Args:
        config: Centralized stability configuration.

    Returns:
        list[dict[str, float]]: Ordered parameter rows.
    """
    rows = []
    for eta in config.eta_values:
        for gd_over_g0 in config.gd_over_g0_values:
            for drive_ratio in config.drive_ratio_values:
                for phase in config.phase_values:
                    for kappa_rel in config.kappa_values:
                        rows.append(
                            {
                                "eta": eta,
                                "gd_over_g0": gd_over_g0,
                                "drive_ratio": drive_ratio,
                                "phase": phase,
                                "kappa_rel": kappa_rel,
                            }
                        )
    return rows


def _aggregate_row(parameters: dict[str, float], config: FloquetStabilityConfig) -> dict[str, Any]:
    """Run both truncations and classify one Tier 1 parameter row.

    Args:
        parameters: Scalar row parameters, with normalized quantities except
            phase in radians.
        config: Centralized stability configuration.

    Returns:
        dict[str, Any]: JSON-ready row result and finite-window classification.
    """
    truncations = [
        _simulate_truncation(n_cav=n_cav, config=config, **parameters)
        for n_cav in config.cavity_truncations
    ]
    low, high = truncations[0], truncations[-1]
    qfi_difference = abs(high["qfi_peak"] - low["qfi_peak"]) / max(abs(high["qfi_peak"]), 1.0e-12)
    photon_difference = abs(high["photon_number_peak"] - low["photon_number_peak"]) / max(
        abs(high["photon_number_peak"]), 1.0e-12
    )
    truncation_sensitive = max(qfi_difference, photon_difference) > config.truncation_difference_tolerance
    boundary_growth = any(item["boundary_growth_flag"] for item in truncations)
    if boundary_growth:
        classification = "growth_at_boundary"
    elif truncation_sensitive:
        classification = "inconclusive"
    else:
        classification = "bounded_in_screen"
    return {
        "row_id": _row_id(**parameters),
        "parameters": parameters,
        "classification": classification,
        "truncation_sensitive": bool(truncation_sensitive),
        "truncation_relative_difference": {
            "qfi_peak": float(qfi_difference),
            "photon_number_peak": float(photon_difference),
        },
        "truncations": truncations,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the stability producer.

    Returns:
        argparse.Namespace: Output, checkpoint, resume, and row-limit options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "floquet_stability.json",
        help="Merged JSON output path.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Per-row checkpoint directory; defaults to data/_cache/floquet_stability.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N deterministic rows; useful for Tier 1 smoke screens.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing row checkpoints and recompute selected rows.",
    )
    parser.add_argument(
        "--periods",
        type=int,
        default=None,
        help="Override the configured drive-window length for validation runs.",
    )
    parser.add_argument(
        "--time-points",
        type=int,
        default=None,
        help="Override the configured number of stored time points.",
    )
    parser.add_argument(
        "--truncations",
        type=str,
        default=None,
        help="Comma-separated cavity dimensions, e.g. 20,30, for validation runs.",
    )
    parser.add_argument(
        "--row-ids",
        type=str,
        default=None,
        help="Comma-separated stable row IDs to validate; defaults to the full configured grid.",
    )
    return parser.parse_args()


def main() -> int:
    """Run and merge the resumable Tier 1 stability screen.

    Returns:
        int: Zero after successful checkpoint merge and JSON serialization.
    """
    args = parse_args()
    started = time.perf_counter()
    config = CONFIG
    if args.periods is not None:
        if args.periods < 3:
            raise ValueError("--periods must be at least 3")
        config = replace(config, n_drive_periods=args.periods)
    if args.time_points is not None:
        if args.time_points < 5:
            raise ValueError("--time-points must be at least 5")
        config = replace(config, n_time_points=args.time_points)
    if args.truncations is not None:
        truncations = tuple(int(value.strip()) for value in args.truncations.split(",") if value.strip())
        if len(truncations) < 2 or any(value < 2 for value in truncations):
            raise ValueError("--truncations must contain at least two dimensions >= 2")
        config = replace(config, cavity_truncations=truncations)
    checkpoint_dir = args.checkpoint_dir or args.output.parent / "_cache" / "floquet_stability"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    parameters = _parameter_rows(config)
    if args.row_ids is not None:
        requested = {value.strip() for value in args.row_ids.split(",") if value.strip()}
        selected = [row for row in parameters if _row_id(**row) in requested]
        missing = requested.difference({_row_id(**row) for row in selected})
        if missing:
            raise ValueError(f"Unknown row IDs: {sorted(missing)}")
        parameters = selected
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        parameters = parameters[: args.limit]
    results = []
    for index, row_parameters in enumerate(parameters, start=1):
        row_id = _row_id(**row_parameters)
        checkpoint = checkpoint_dir / f"{row_id}.json"
        if checkpoint.is_file() and not args.no_resume:
            row_result = json.loads(checkpoint.read_text(encoding="utf-8"))
            print(f"[{index}/{len(parameters)}] resumed {row_id}")
        else:
            print(f"[{index}/{len(parameters)}] simulating {row_id}")
            row_result = _aggregate_row(row_parameters, config)
            checkpoint.write_text(json.dumps(row_result, indent=2) + "\n", encoding="utf-8")
        results.append(row_result)
    results.sort(key=lambda item: item["row_id"])
    payload = {
        "schema": "single_qubit_floquet_stability_tier1_v1",
        "status": "finite_window_stability_screen",
        "purpose": "Test whether two-level saturation limits the observed driven response in the declared finite-dimensional model.",
        "configuration": asdict(config),
        "units": {
            "eta": "g0/wc, dimensionless",
            "gd_over_g0": "dimensionless",
            "drive_ratio": "wd/(wc+wq), dimensionless",
            "phase": "radians",
            "kappa_rel": "kappa/wc, dimensionless",
            "time": "drive periods",
            "qfi": "inverse squared normalized coupling",
            "photon_number": "photons",
        },
        "solver": {
            "name": "qutip.sesolve for kappa_rel=0; qutip.mesolve otherwise",
            "options": {
                "atol": CONFIG.solver_atol,
                "rtol": CONFIG.solver_rtol,
                "method": "adams",
                "max_step_fraction_of_drive_period": CONFIG.max_step_fraction,
            },
            "python": platform.python_version(),
            "numpy": np.__version__,
            "qutip": qt.__version__,
        },
        "validation": {
            "rows_requested": len(parameters),
            "rows_completed": len(results),
            "all_rows_complete": len(parameters) == len(results),
            "classification_note": "Classifications are finite-window diagnostics, not asymptotic stability results.",
            "numerical_physicality_note": "Trace/Hermiticity/eigenvalue acceptance is not a proof of exact positivity.",                "numerical_eigenvalue_floor": config.numerical_eigenvalue_floor,

        },
        "caveats": [
            "Tier 1 does not establish an asymptotic Floquet-stability theorem.",
            "A boundary-growth flag means the observable is still rising or peaks at the finite-window boundary.",
            "Inconclusive rows require longer windows, tighter convergence, or larger truncation.",
            "The effective time-dependent USC control convention retains the manuscript gauge/control caveat.",
        ],
        "rows": results,
        "elapsed_s": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
