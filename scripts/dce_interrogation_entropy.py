#!/usr/bin/env python3
"""Compute the Track B DCE interrogation-window trade-off.

This producer evaluates mixed-state fidelity QFI, von Neumann entropy, a
finite-difference entropy-rate proxy, and the information-rate objective
``F_Q / t`` for the canonical single-qubit DCE protocol.  The calculation is
model-specific: it does not infer a universal thermodynamic optimum.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import qutip as qt

try:
    from config import DEFAULT_DCE_INTERROGATION, DCEInterrogationConfig
except ImportError as exc:  # pragma: no cover - import failure is environment-specific
    raise ImportError(
        "Track B configuration is unavailable; run this script from its scripts directory."
    ) from exc


CONFIG = DEFAULT_DCE_INTERROGATION


def _build_operators(config: DCEInterrogationConfig) -> tuple[qt.Qobj, qt.Qobj, qt.Qobj, qt.Qobj]:
    """Build tensor-ordered cavity and qubit operators.

    Args:
        config: DCE configuration with the cavity Fock truncation, dimensionless.

    Returns:
        tuple[qt.Qobj, qt.Qobj, qt.Qobj, qt.Qobj]: Cavity annihilation,
        cavity number, qubit sigma-x, and static Hamiltonian operators.  The
        Hamiltonian is in normalized angular-frequency units.
    """
    cavity_annihilation = qt.tensor(qt.destroy(config.n_cav), qt.qeye(2))
    cavity_number = cavity_annihilation.dag() * cavity_annihilation
    qubit_x = qt.tensor(qt.qeye(config.n_cav), qt.sigmax())
    qubit_z = qt.tensor(qt.qeye(config.n_cav), qt.sigmaz())
    static_hamiltonian = (
        0.5 * config.wq_wc * qubit_z + cavity_number
    )
    return cavity_annihilation, cavity_number, qubit_x, static_hamiltonian


def _hamiltonian(
    eta: float,
    config: DCEInterrogationConfig,
    static_hamiltonian: qt.Qobj,
    interaction: qt.Qobj,
) -> qt.QobjEvo:
    """Construct the time-dependent normalized DCE Hamiltonian.

    Args:
        eta: Static normalized coupling, dimensionless.
        config: DCE configuration containing drive parameters, dimensionless.
        static_hamiltonian: Time-independent Hamiltonian in units of the
            cavity angular frequency.
        interaction: Counter-rotating-term-containing interaction operator.

    Returns:
        qt.QobjEvo: Time-dependent Hamiltonian in normalized angular-frequency
        units.
    """
    def drive_coefficient(time_normalized: float, _args: dict | None = None) -> float:
        """Return the normalized coupling coefficient at time ``time_normalized``.

        Args:
            time_normalized: Time in inverse cavity angular-frequency units.
            _args: QuTiP coefficient arguments, unused.

        Returns:
            float: Dimensionless instantaneous coupling coefficient.
        """
        return eta + config.gd * math.cos(config.wd_wc * time_normalized)

    return qt.QobjEvo([static_hamiltonian, [interaction, drive_coefficient]])


def _density_matrix(state: qt.Qobj) -> qt.Qobj:
    """Convert a stored ket or density operator to a normalized density matrix.

    Args:
        state: QuTiP ket or density operator, dimensionless state object.

    Returns:
        qt.Qobj: Unit-trace density operator with the original subsystem dims.
    """
    rho = state.proj() if state.isket else state
    trace = float(np.real(rho.tr()))
    if trace <= 0.0:
        raise ValueError("Encountered a non-positive state trace during analysis.")
    return rho / trace


def _fidelity(rho_a: qt.Qobj, rho_b: qt.Qobj) -> float:
    """Compute squared Uhlmann fidelity for two density operators.

    Args:
        rho_a: First unit-trace density operator, dimensionless.
        rho_b: Second unit-trace density operator, dimensionless.

    Returns:
        float: Squared Uhlmann fidelity in the interval [0, 1].
    """
    root_a = rho_a.sqrtm()
    product = root_a * rho_b * root_a
    eigenvalues = np.real_if_close(product.eigenenergies()).real
    root_fidelity = float(np.sum(np.sqrt(np.clip(eigenvalues, 0.0, None))))
    return float(np.clip(root_fidelity * root_fidelity, 0.0, 1.0))


def _von_neumann_entropy(rho: qt.Qobj, log_base: float) -> float:
    """Compute the von Neumann entropy of a density operator.

    Args:
        rho: Unit-trace density operator, dimensionless.
        log_base: Logarithm base, dimensionless; base 2 reports bits.

    Returns:
        float: Von Neumann entropy in units determined by ``log_base``.
    """
    eigenvalues = np.real_if_close(rho.eigenenergies()).real
    probabilities = np.clip(eigenvalues, 0.0, 1.0)
    probabilities = probabilities[probabilities > 0.0]
    return float(-np.sum(probabilities * np.log(probabilities)) / np.log(log_base))


def _finite_difference(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Estimate a time derivative with second-order finite differences.

    Args:
        values: Scalar trace sampled at ``times``, dimensionless.
        times: Strictly increasing normalized times in inverse angular-frequency
            units.

    Returns:
        np.ndarray: Derivative with reciprocal normalized-time units.
    """
    if len(values) < 3:
        raise ValueError("At least three time points are required for a derivative.")
    return np.gradient(values, times, edge_order=2)


def _qfi_trace(
    states_a: Iterable[qt.Qobj],
    states_b: Iterable[qt.Qobj],
    delta: float,
) -> np.ndarray:
    """Compute fidelity-based QFI for paired parameterized state traces.

    Args:
        states_a: Reference state trace at coupling ``eta``, dimensionless.
        states_b: Perturbed state trace at coupling ``eta + delta``,
            dimensionless.
        delta: Dimensionless finite-difference coupling increment.

    Returns:
        np.ndarray: QFI trace in inverse squared normalized-coupling units.
    """
    if delta <= 0.0:
        raise ValueError("The finite-difference increment must be positive.")
    values = []
    for state_a, state_b in zip(states_a, states_b):
        fidelity = _fidelity(_density_matrix(state_a), _density_matrix(state_b))
        values.append(8.0 * (1.0 - math.sqrt(fidelity)) / delta**2)
    return np.asarray(values, dtype=float)


def _simulate_loss_row(
    kappa_rel: float,
    config: DCEInterrogationConfig,
    operators: tuple[qt.Qobj, qt.Qobj, qt.Qobj, qt.Qobj],
) -> dict:
    """Simulate one cavity-loss row and derive interrogation metrics.

    Args:
        kappa_rel: Cavity loss divided by the cavity angular frequency,
            dimensionless.
        config: DCE configuration, dimensionless except for normalized times.
        operators: Tensor-ordered operators returned by ``_build_operators``.

    Returns:
        dict: JSON-serializable time-series metrics and optimum summaries.
    """
    cavity_annihilation, cavity_number, qubit_x, static_hamiltonian = operators
    interaction = qubit_x * (cavity_annihilation + cavity_annihilation.dag())
    hamiltonian_a = _hamiltonian(config.eta, config, static_hamiltonian, interaction)
    hamiltonian_b = _hamiltonian(
        config.eta + config.delta, config, static_hamiltonian, interaction
    )
    initial_state = qt.tensor(qt.basis(config.n_cav, 0), qt.basis(2, 0))
    collapse_ops = [math.sqrt(kappa_rel) * cavity_annihilation] if kappa_rel > 0.0 else []
    drive_period = 2.0 * math.pi / config.wd_wc
    times = np.linspace(
        0.0,
        config.n_drive_periods * drive_period,
        config.n_time_points,
    )
    options = {
        "atol": 1.0e-10,
        "rtol": 1.0e-8,
        "method": "adams",
        "store_states": True,
        "progress_bar": "",
    }
    result_a = qt.mesolve(
        hamiltonian_a,
        initial_state,
        times,
        c_ops=collapse_ops,
        e_ops=[],
        options=options,
    )
    result_b = qt.mesolve(
        hamiltonian_b,
        initial_state,
        times,
        c_ops=collapse_ops,
        e_ops=[],
        options=options,
    )
    qfi = _qfi_trace(result_a.states, result_b.states, config.delta)
    entropy = np.asarray(
        [_von_neumann_entropy(_density_matrix(state), config.entropy_log_base) for state in result_a.states],
        dtype=float,
    )
    entropy_rate = _finite_difference(entropy, times)
    photon_number = np.asarray(
        [float(np.real(qt.expect(cavity_number, state))) for state in result_a.states],
        dtype=float,
    )
    valid = times >= config.min_objective_period * drive_period
    rate = np.full_like(qfi, np.nan)
    rate[valid] = qfi[valid] / times[valid]
    rate_per_period = np.full_like(qfi, np.nan)
    rate_per_period[valid] = qfi[valid] / (times[valid] / drive_period)
    objective_index = int(np.nanargmax(rate))
    peak_index = int(np.argmax(qfi))
    return {
        "kappa_rel": float(kappa_rel),
        "kappa_2pi_MHz_at_6GHz": float(kappa_rel * 6.0e3),
        "time_normalized": times.tolist(),
        "time_drive_periods": (times / drive_period).tolist(),
        "qfi": qfi.tolist(),
        "von_neumann_entropy_bits": entropy.tolist(),
        "entropy_rate_bits_per_normalized_time": entropy_rate.tolist(),
        "photon_number": photon_number.tolist(),
        "information_rate_qfi_per_normalized_time": rate.tolist(),
        "information_rate_qfi_per_drive_period": rate_per_period.tolist(),
        "qfi_peak": float(qfi[peak_index]),
        "qfi_peak_period": float(times[peak_index] / drive_period),
        "entropy_at_qfi_peak_bits": float(entropy[peak_index]),
        "entropy_rate_at_qfi_peak_bits_per_normalized_time": float(entropy_rate[peak_index]),
        "qfi_per_normalized_time_max": float(rate[objective_index]),
        "qfi_per_normalized_time_max_period": float(times[objective_index] / drive_period),
        "qfi_at_information_rate_max": float(qfi[objective_index]),
        "entropy_at_information_rate_max_bits": float(entropy[objective_index]),
        "photon_number_at_information_rate_max": float(photon_number[objective_index]),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Optional output path and solver timing controls.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "dce_interrogation_entropy.json",
        help="JSON output path.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the deterministic interrogation-window study.

    Returns:
        int: Zero after successful simulation and JSON serialization.
    """
    args = parse_args()
    started = time.perf_counter()
    operators = _build_operators(CONFIG)
    rows = []
    for kappa_rel in CONFIG.kappa_values:
        print(f"Simulating kappa/wc={kappa_rel:.4g}")
        rows.append(_simulate_loss_row(kappa_rel, CONFIG, operators))
    payload = {
        "schema": "single_qubit_dce_interrogation_entropy_v1",
        "status": "model_specific_information_entropy_tradeoff",
        "units": {
            "time": "inverse cavity angular frequency",
            "qfi": "inverse squared normalized coupling",
            "entropy": "bits",
            "loss": "cavity loss divided by cavity angular frequency",
        },
        "configuration": {
            "eta": CONFIG.eta,
            "gd": CONFIG.gd,
            "wq_wc": CONFIG.wq_wc,
            "wd_wc": CONFIG.wd_wc,
            "n_cav": CONFIG.n_cav,
            "n_drive_periods": CONFIG.n_drive_periods,
            "n_time_points": CONFIG.n_time_points,
            "delta": CONFIG.delta,
            "entropy_log_base": CONFIG.entropy_log_base,
            "min_objective_period": CONFIG.min_objective_period,
            "kappa_values": list(CONFIG.kappa_values),
        },
        "objective": "maximize F_Q(t)/t after one drive period; t is normalized inverse cavity angular-frequency time",
        "caveats": [
            "The entropy derivative is a finite-difference rate proxy, not a thermodynamic entropy-production rate.",
            "The objective is model-specific and does not define a universal thermodynamic interrogation time.",
            "The output field, pump cost, bandwidth, and detector efficiency are not modeled.",
            "The time-dependent effective USC Hamiltonian retains the manuscript gauge/control caveat.",
        ],
        "solver": {
            "name": "qutip.mesolve",
            "options": {"atol": 1.0e-10, "rtol": 1.0e-8, "method": "adams"},
            "python": platform.python_version(),
            "numpy": np.__version__,
            "qutip": qt.__version__,
        },
        "rows": rows,
        "elapsed_s": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
