#!/usr/bin/env python3
"""Compute multiparameter QFI and a controlled representation check for Track B.

The producer evaluates pure-state QFI metric tensors for static ground states and
for the driven single-qubit DCE protocol.  It also evolves a finite-dimensional
unitary-similarity representation as an internal consistency check.  This is a
model-specific numerical diagnostic, not a proof of arbitrary gauge invariance
or of phase-noise protection in a laboratory device.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import qutip as qt

try:
    from config import DEFAULT_MULTIPARAMETER_GAUGE, MultiparameterGaugeConfig
except ImportError as exc:  # pragma: no cover - environment-specific import failure
    raise ImportError(
        "Track B configuration is unavailable; run this script from its scripts directory."
    ) from exc


PARAMETER_NAMES = ("eta", "wq_wc", "gd_over_g0", "phase")


def _operators(n_cav: int) -> dict[str, qt.Qobj]:
    """Build tensor-ordered cavity and qubit operators.

    Args:
        n_cav: Cavity Fock-space dimension, dimensionless.

    Returns:
        dict[str, qt.Qobj]: Tensor-ordered annihilation, number, Pauli, and
        identity operators in normalized cavity-frequency units.
    """
    cavity = qt.tensor(qt.destroy(n_cav), qt.qeye(2))
    return {
        "a": cavity,
        "n": cavity.dag() * cavity,
        "x": qt.tensor(qt.qeye(n_cav), qt.sigmax()),
        "z": qt.tensor(qt.qeye(n_cav), qt.sigmaz()),
        "identity": qt.tensor(qt.qeye(n_cav), qt.qeye(2)),
    }


def _static_hamiltonian(
    eta: float, wq_wc: float, operators: dict[str, qt.Qobj]
) -> qt.Qobj:
    """Construct the normalized static quantum Rabi Hamiltonian.

    Args:
        eta: Static coupling g0/wc, dimensionless.
        wq_wc: Qubit-to-cavity angular-frequency ratio, dimensionless.
        operators: Tensor-ordered operators from ``_operators``.

    Returns:
        qt.Qobj: Static Hamiltonian in units of the cavity angular frequency.
    """
    return 0.5 * wq_wc * operators["z"] + operators["n"] + eta * operators["x"] * (
        operators["a"] + operators["a"].dag()
    )


def _driven_components(
    eta: float,
    wq_wc: float,
    gd_over_g0: float,
    wd_wc: float,
    phase: float,
    operators: dict[str, qt.Qobj],
) -> tuple[qt.Qobj, qt.Qobj, Callable[[float, dict | None], float]]:
    """Construct the static and modulated components of the DCE Hamiltonian.

    The modulation follows the existing Track B convention: ``gd_over_g0`` is
    retained as the dimensionless additive drive coefficient used by the
    canonical DCE producers, despite its historical ``g_d/g_0`` label.

    Args:
        eta: Static coupling g0/wc, dimensionless.
        wq_wc: Qubit-to-cavity frequency ratio, dimensionless.
        gd_over_g0: Existing normalized modulation coefficient, dimensionless.
        wd_wc: Drive-to-cavity frequency ratio, dimensionless.
        phase: Drive phase, in radians.
        operators: Tensor-ordered operators.

    Returns:
        tuple[qt.Qobj, qt.Qobj, Callable]: Time-independent Hamiltonian,
        interaction operator, and scalar drive coefficient function.
    """
    static = 0.5 * wq_wc * operators["z"] + operators["n"]
    interaction = operators["x"] * (operators["a"] + operators["a"].dag())

    def coefficient(time_normalized: float, _args: dict | None = None) -> float:
        """Return the instantaneous normalized coupling coefficient.

        Args:
            time_normalized: Time in inverse cavity angular-frequency units.
            _args: Optional QuTiP callback arguments, unused.

        Returns:
            float: Dimensionless coefficient multiplying the interaction.
        """
        return eta + gd_over_g0 * math.cos(wd_wc * time_normalized + phase)

    return static, interaction, coefficient


def _qobj_evolution(
    eta: float,
    wq_wc: float,
    gd_over_g0: float,
    wd_wc: float,
    phase: float,
    operators: dict[str, qt.Qobj],
    initial_state: qt.Qobj,
    n_drive_periods: int,
    n_time_points: int,
    unitary: qt.Qobj | None = None,
) -> qt.Qobj:
    """Evolve one driven pure state to the end of a specified time window.

    Args:
        eta: Static coupling g0/wc, dimensionless.
        wq_wc: Qubit-to-cavity frequency ratio, dimensionless.
        gd_over_g0: Existing normalized modulation coefficient, dimensionless.
        wd_wc: Drive-to-cavity frequency ratio, dimensionless.
        phase: Drive phase, in radians.
        operators: Tensor-ordered operators.
        initial_state: Initial normalized ket, dimensionless state.
        n_drive_periods: Evolution duration, in drive periods.
        n_time_points: Number of solver output points, dimensionless.
        unitary: Optional fixed representation transformation, dimensionless.

    Returns:
        qt.Qobj: Final normalized ket, dimensionless state.
    """
    static, interaction, coefficient = _driven_components(
        eta, wq_wc, gd_over_g0, wd_wc, phase, operators
    )
    if unitary is not None:
        static = unitary * static * unitary.dag()
        interaction = unitary * interaction * unitary.dag()
        initial_state = unitary * initial_state
    hamiltonian = qt.QobjEvo([static, [interaction, coefficient]])
    period = 2.0 * math.pi / wd_wc
    times = np.linspace(0.0, n_drive_periods * period, n_time_points)
    result = qt.sesolve(
        hamiltonian,
        initial_state,
        times,
        e_ops=[],
        options={
            "atol": 1.0e-10,
            "rtol": 1.0e-8,
            "method": "adams",
            "store_states": True,
            "progress_bar": "",
        },
    )
    state = result.states[-1]
    norm = float(state.norm())
    if norm <= 0.0:
        raise ValueError("Driven evolution returned a zero-norm state.")
    return state / norm


def _ground_state(eta: float, wq_wc: float, operators: dict[str, qt.Qobj]) -> qt.Qobj:
    """Return the normalized ground-state ket of the static Rabi Hamiltonian.

    Args:
        eta: Static coupling g0/wc, dimensionless.
        wq_wc: Qubit-to-cavity frequency ratio, dimensionless.
        operators: Tensor-ordered operators.

    Returns:
        qt.Qobj: Normalized lowest-energy eigenket, dimensionless state.
    """
    _, state = _static_hamiltonian(eta, wq_wc, operators).groundstate()
    return state / state.norm()


def _phase_align(state: qt.Qobj, reference: qt.Qobj) -> qt.Qobj:
    """Remove an arbitrary ket phase relative to a reference ket.

    Args:
        state: Ket to align, dimensionless state.
        reference: Reference ket, dimensionless state.

    Returns:
        qt.Qobj: Phase-aligned ket with unchanged physical state.
    """
    overlap = complex(reference.overlap(state))
    if abs(overlap) == 0.0:
        return state
    return state * (overlap.conjugate() / abs(overlap))


def _pure_qfi(states_plus: list[qt.Qobj], states_minus: list[qt.Qobj], base: qt.Qobj, step: float) -> np.ndarray:
    """Compute a pure-state QFI matrix from central finite-difference kets.

    Args:
        states_plus: Positive perturbation kets, one per parameter.
        states_minus: Negative perturbation kets, one per parameter.
        base: Unperturbed ket, dimensionless state.
        step: Central finite-difference increment, dimensionless or radians.

    Returns:
        np.ndarray: Real symmetric QFI matrix in the corresponding inverse
        parameter-unit products.
    """
    derivatives = []
    for plus, minus in zip(states_plus, states_minus):
        plus_aligned = _phase_align(plus, base)
        minus_aligned = _phase_align(minus, base)
        derivatives.append((plus_aligned - minus_aligned) / (2.0 * step))
    qfi = np.empty((len(derivatives), len(derivatives)), dtype=float)
    for i, derivative_i in enumerate(derivatives):
        for j, derivative_j in enumerate(derivatives):
            projected_i = derivative_i - base * complex(base.overlap(derivative_i))
            projected_j = derivative_j - base * complex(base.overlap(derivative_j))
            qfi[i, j] = 4.0 * float(np.real(projected_i.overlap(projected_j)))
    return 0.5 * (qfi + qfi.T)


def _fidelity_diagonal(
    states_plus: list[qt.Qobj], states_minus: list[qt.Qobj], base: qt.Qobj, step: float
) -> np.ndarray:
    """Compute independent fidelity-based diagonal QFI estimates.

    Args:
        states_plus: Positive perturbation kets.
        states_minus: Negative perturbation kets.
        base: Unperturbed ket, dimensionless state.
        step: Central finite-difference increment, dimensionless or radians.

    Returns:
        np.ndarray: Diagonal QFI estimates in inverse squared parameter units.
    """
    values = []
    for plus, minus in zip(states_plus, states_minus):
        plus_fidelity = abs(complex(base.overlap(plus))) ** 2
        minus_fidelity = abs(complex(base.overlap(minus))) ** 2
        values.append(4.0 * (2.0 - plus_fidelity - minus_fidelity) / (2.0 * step**2))
    return np.asarray(values, dtype=float)


def _metric_summary(qfi: np.ndarray) -> dict:
    """Summarize a QFI matrix, nuisance penalty, and conditioning.

    Args:
        qfi: Real symmetric QFI matrix, inverse parameter-unit products.

    Returns:
        dict: JSON-serializable matrix diagnostics and normalized correlations.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(qfi)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    psd_residual = float(min(0.0, float(np.min(eigenvalues))))
    positive = np.abs(eigenvalues[np.abs(eigenvalues) > scale * 1.0e-12])
    condition = float(np.max(positive) / np.min(positive)) if positive.size else float("inf")
    rank = int(np.count_nonzero(np.abs(eigenvalues) > scale * 1.0e-8))
    diagonal = np.sqrt(np.clip(np.diag(qfi), 0.0, None))
    denominator = np.outer(diagonal, diagonal)
    correlations = np.divide(qfi, denominator, out=np.zeros_like(qfi), where=denominator > 0.0)
    nuisance_block = qfi[1:, 1:]
    nuisance_penalty = float(qfi[0, 0] - qfi[0, 1:] @ np.linalg.pinv(nuisance_block) @ qfi[1:, 0])
    return {
        "matrix": qfi.tolist(),
        "eigenvalues": eigenvalues.tolist(),
        "eigenvectors": eigenvectors.tolist(),
        "normalized_correlations": correlations.tolist(),
        "condition_number": condition,
        "numerical_rank": rank,
        "psd_residual": psd_residual,
        "eta_effective_information_with_nuisance": max(0.0, nuisance_penalty),
    }


def _static_row(config: MultiparameterGaugeConfig, eta: float, wq_wc: float, n_cav: int) -> dict:
    """Evaluate one static ground-state multiparameter metric row.

    Args:
        config: Centralized Work Package B configuration.
        eta: Static coupling g0/wc, dimensionless.
        wq_wc: Qubit-to-cavity ratio, dimensionless.
        n_cav: Cavity truncation, dimensionless.

    Returns:
        dict: Static QFI metrics and finite-difference convergence diagnostics.
    """
    operators = _operators(n_cav)
    base = _ground_state(eta, wq_wc, operators)
    metrics = []
    for step in config.finite_difference_steps:
        plus = [_ground_state(eta + step, wq_wc, operators), _ground_state(eta, wq_wc + step, operators)]
        minus = [_ground_state(eta - step, wq_wc, operators), _ground_state(eta, wq_wc - step, operators)]
        qfi = _pure_qfi(plus, minus, base, step)
        fidelity_diag = _fidelity_diagonal(plus, minus, base, step)
        metrics.append({"step": step, "metric": _metric_summary(qfi), "fidelity_diagonal": fidelity_diag.tolist()})
    changes = []
    for previous, current in zip(metrics, metrics[1:]):
        denom = max(1.0, float(np.linalg.norm(np.asarray(current["metric"]["matrix"]))))
        changes.append(float(np.linalg.norm(np.asarray(previous["metric"]["matrix"]) - np.asarray(current["metric"]["matrix"])) / denom))
    final = metrics[-1]
    return {
        "kind": "static",
        "parameters": {"eta": eta, "wq_wc": wq_wc},
        "n_cav": n_cav,
        "parameter_names": ["eta", "wq_wc"],
        "finite_difference": metrics,
        "relative_step_changes": changes,
        "convergence_status": "converged" if changes and changes[-1] <= config.qfi_convergence_tolerance else "unresolved",
        "final_metric": final["metric"],
        "state_norm": float(base.norm()),
    }


def _driven_parameters(
    eta: float, wq_wc: float, gd_over_g0: float, phase: float, offset: float, config: MultiparameterGaugeConfig
) -> tuple[float, float, float, float, float]:
    """Return the driven parameter tuple for a detuning offset.

    Args:
        eta: Static coupling g0/wc, dimensionless.
        wq_wc: Qubit-to-cavity ratio, dimensionless.
        gd_over_g0: Modulation coefficient, dimensionless.
        phase: Drive phase, radians.
        offset: Relative drive detuning offset, dimensionless.
        config: Centralized configuration.

    Returns:
        tuple[float, float, float, float, float]: Eta, qubit ratio, modulation
        coefficient, drive ratio, and phase.
    """
    wd_wc = config.wd_wc * (1.0 + offset)
    return eta, wq_wc, gd_over_g0, wd_wc, phase


def _driven_row(config: MultiparameterGaugeConfig, offset: float, n_periods: int, n_cav: int) -> dict:
    """Evaluate one driven final-state QFI metric row.

    Args:
        config: Centralized Work Package B configuration.
        offset: Relative drive-frequency offset, dimensionless.
        n_periods: Evolution duration in drive periods, dimensionless.
        n_cav: Cavity truncation, dimensionless.

    Returns:
        dict: Driven QFI tensor, convergence metrics, and state diagnostics.
    """
    operators = _operators(n_cav)
    eta, wq_wc, gd, wd_wc, phase = _driven_parameters(config.eta, config.wq_wc, config.gd_over_g0, config.phase, offset, config)
    initial = _ground_state(eta, wq_wc, operators)
    metrics = []
    for step in config.finite_difference_steps:
        base = _qobj_evolution(eta, wq_wc, gd, wd_wc, phase, operators, initial, n_periods, config.time_points)
        plus_params = [
            (eta + step, wq_wc, gd, phase),
            (eta, wq_wc + step, gd, phase),
            (eta, wq_wc, gd + step, phase),
            (eta, wq_wc, gd, phase + step),
        ]
        minus_params = [
            (eta - step, wq_wc, gd, phase),
            (eta, wq_wc - step, gd, phase),
            (eta, wq_wc, gd - step, phase),
            (eta, wq_wc, gd, phase - step),
        ]
        plus = [_qobj_evolution(values[0], values[1], values[2], wd_wc, values[3], operators, initial, n_periods, config.time_points) for values in plus_params]
        minus = [_qobj_evolution(values[0], values[1], values[2], wd_wc, values[3], operators, initial, n_periods, config.time_points) for values in minus_params]
        qfi = _pure_qfi(plus, minus, base, step)
        fidelity_diag = _fidelity_diagonal(plus, minus, base, step)
        metrics.append({"step": step, "metric": _metric_summary(qfi), "fidelity_diagonal": fidelity_diag.tolist(), "base_norm": float(base.norm())})
    changes = []
    for previous, current in zip(metrics, metrics[1:]):
        denom = max(1.0, float(np.linalg.norm(np.asarray(current["metric"]["matrix"]))))
        changes.append(float(np.linalg.norm(np.asarray(previous["metric"]["matrix"]) - np.asarray(current["metric"]["matrix"])) / denom))
    final = metrics[-1]
    row = {
        "kind": "driven",
        "parameters": {"eta": eta, "wq_wc": wq_wc, "gd_over_g0": gd, "wd_wc": wd_wc, "phase": phase, "drive_detuning_offset": offset},
        "n_cav": n_cav,
        "n_drive_periods": n_periods,
        "parameter_names": list(PARAMETER_NAMES),
        "finite_difference": metrics,
        "relative_step_changes": changes,
        "convergence_status": "converged" if changes and changes[-1] <= config.qfi_convergence_tolerance else "unresolved",
        "final_metric": final["metric"],
        "state_norm": final["base_norm"],
    }
    return row


def _gauge_check(config: MultiparameterGaugeConfig) -> dict:
    """Compare independently evolved baseline and transformed representations.

    Args:
        config: Centralized Work Package B configuration.

    Returns:
        dict: Unitary check, state fidelity, observable differences, and QFI
        tensor difference for the canonical selected row.
    """
    n_cav = config.selected_gauge_truncation
    operators = _operators(n_cav)
    eta, wq_wc, gd, wd_wc, phase = _driven_parameters(config.eta, config.wq_wc, config.gd_over_g0, config.phase, 0.0, config)
    initial = _ground_state(eta, wq_wc, operators)
    number = operators["n"]
    unitary = (-1j * config.transform_angle * number).expm()
    identity_error = float((unitary.dag() * unitary - operators["identity"]).norm())
    base = _qobj_evolution(eta, wq_wc, gd, wd_wc, phase, operators, initial, config.drive_windows[-1], config.time_points)
    transformed = _qobj_evolution(eta, wq_wc, gd, wd_wc, phase, operators, initial, config.drive_windows[-1], config.time_points, unitary=unitary)
    expected = unitary * base
    state_fidelity = float(abs(complex(expected.overlap(transformed))) ** 2)
    observable_names = ("n", "z")
    observable_differences = {}
    for name in observable_names:
        observable = operators[name]
        transformed_observable = unitary * observable * unitary.dag()
        left = complex(base.dag() * observable * base)
        right = complex(transformed.dag() * transformed_observable * transformed)
        observable_differences[name] = {"baseline": float(np.real(left)), "transformed": float(np.real(right)), "absolute_difference": abs(float(np.real(left - right)))}
    step = config.finite_difference_steps[-1]
    base_plus = [_qobj_evolution(values[0], values[1], values[2], wd_wc, values[3], operators, initial, config.drive_windows[-1], config.time_points) for values in ((eta + step, wq_wc, gd, phase), (eta, wq_wc + step, gd, phase), (eta, wq_wc, gd + step, phase), (eta, wq_wc, gd, phase + step))]
    base_minus = [_qobj_evolution(values[0], values[1], values[2], wd_wc, values[3], operators, initial, config.drive_windows[-1], config.time_points) for values in ((eta - step, wq_wc, gd, phase), (eta, wq_wc - step, gd, phase), (eta, wq_wc, gd - step, phase), (eta, wq_wc, gd, phase - step))]
    transformed_plus = [_qobj_evolution(values[0], values[1], values[2], wd_wc, values[3], operators, initial, config.drive_windows[-1], config.time_points, unitary=unitary) for values in ((eta + step, wq_wc, gd, phase), (eta, wq_wc + step, gd, phase), (eta, wq_wc, gd + step, phase), (eta, wq_wc, gd, phase + step))]
    transformed_minus = [_qobj_evolution(values[0], values[1], values[2], wd_wc, values[3], operators, initial, config.drive_windows[-1], config.time_points, unitary=unitary) for values in ((eta - step, wq_wc, gd, phase), (eta, wq_wc - step, gd, phase), (eta, wq_wc, gd - step, phase), (eta, wq_wc, gd, phase - step))]
    base_metric = _pure_qfi(base_plus, base_minus, base, step)
    transformed_metric = _pure_qfi(transformed_plus, transformed_minus, transformed, step)
    metric_difference = float(np.linalg.norm(base_metric - transformed_metric) / max(1.0, np.linalg.norm(base_metric)))
    return {
        "representation": "fixed_cavity_phase_similarity_transform",
        "transform_angle_rad": config.transform_angle,
        "transformation": "U=exp(-i*angle*a^dagger*a); H'=UHU^dagger; |psi'>=U|psi>; O'=UOU^dagger",
        "unitarity_error_frobenius": identity_error,
        "state_fidelity_to_transformed_baseline": state_fidelity,
        "observable_comparisons": observable_differences,
        "qfi_metric_relative_difference": metric_difference,
        "status": "consistent" if identity_error <= config.gauge_tolerance and 1.0 - state_fidelity <= config.gauge_tolerance and metric_difference <= config.gauge_tolerance else "disagreement_reported",
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the producer.

    Returns:
        argparse.Namespace: Output path and optional selected-row mode.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "multiparameter_gauge_qfi.json")
    return parser.parse_args()


def main() -> None:
    """Run all configured static, driven, and representation-check calculations."""
    args = parse_args()
    config = DEFAULT_MULTIPARAMETER_GAUGE
    rows = []
    for n_cav in config.cavity_truncations:
        for eta in config.static_eta_values:
            for wq_wc in config.static_wq_values:
                rows.append(_static_row(config, eta, wq_wc, n_cav))
    for n_cav in config.cavity_truncations:
        for offset in config.drive_detuning_offsets:
            for n_periods in config.drive_windows:
                rows.append(_driven_row(config, offset, n_periods, n_cav))
    gauge = _gauge_check(config)
    output = {
        "schema": "trackB_multiparameter_gauge_qfi_v1",
        "status": "model_specific_metric_and_representation_check",
        "purpose": "Directional pure-state QFI and controlled unitary-representation consistency for the single-qubit Rabi/DCE model.",
        "configuration": asdict(config),
        "units": {
            "eta": "g0/wc, dimensionless",
            "wq_wc": "dimensionless",
            "gd_over_g0": "existing normalized DCE coefficient, dimensionless",
            "wd_wc": "dimensionless",
            "phase": "radians",
            "qfi": "inverse squared parameter units",
            "time": "inverse cavity angular-frequency units; periods are dimensionless",
        },
        "solver": {"method": "QuTiP sesolve", "atol": config.solver_atol, "rtol": config.solver_rtol, "collapse_operators": []},
        "parameter_names": list(PARAMETER_NAMES),
        "validation": {
            "all_rows_have_finite_metrics": bool(all(np.all(np.isfinite(np.asarray(row["final_metric"]["matrix"]))) for row in rows)),
            "minimum_psd_residual": float(min(row["final_metric"]["psd_residual"] for row in rows)),
            "converged_row_count": int(sum(row["convergence_status"] == "converged" for row in rows)),
            "row_count": len(rows),
            "gauge_check": gauge,
        },
        "software": {"python": platform.python_version(), "numpy": np.__version__, "qutip": qt.__version__},
        "caveats": [
            "The four-parameter tensor uses normalized parameters; raw entries with different units must not be compared without normalization.",
            "The representation check is an explicitly unitary finite-dimensional similarity transform, not a proof of arbitrary USC gauge invariance under a laboratory control transformation.",
            "Pure-state QFI excludes detector inefficiency, output-field measurement, thermal baths, and resource-normalized pump cost.",
            "The existing Track B drive convention treats gd_over_g0 as an additive normalized coefficient; this is recorded rather than silently reinterpreted.",
        ],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} ({len(rows)} rows)")
    print(f"Gauge check: {gauge['status']}; converged rows: {output['validation']['converged_row_count']}/{len(rows)}")


if __name__ == "__main__":
    main()
