#!/usr/bin/env python3
"""Assess multilevel-emitter leakage for the Track B DCE/Rabi models.

The producer implements a common oscillator-based emitter interface.  Each
platform enters only through a declared effective spectrum (transition
frequencies in units of the cavity angular frequency) with an explicitly
shared dipole-like ladder coupling.  Effective spectra are labeled as such;
they are not full circuit simulations.  It reports leakage out of the
computational subspace, fidelity-based QFI, photon number, purity/entropy,
and a declared adequacy criterion for the two-level approximation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import qutip as qt

try:
    from config import DEFAULT_MULTILEVEL, MultilevelConfig
except ImportError as exc:  # pragma: no cover - environment-specific
    raise ImportError(
        "Track B configuration is unavailable; run this script from scripts/."
    ) from exc


CONFIG = DEFAULT_MULTILEVEL


def emitter_energies(transitions: tuple[float, ...], d_e: int) -> list[float]:
    """Return emitter level energies from declared transition frequencies.

    Args:
        transitions: Level spacings Delta_{e,e+1}/wc, dimensionless.
        d_e: Number of retained emitter levels, dimensionless.

    Returns:
        list[float]: Level energies E_e in normalized angular-frequency units,
        with E_0 = 0.
    """
    if d_e < 2:
        raise ValueError("d_e must be at least 2")
    if len(transitions) < d_e - 1:
        raise ValueError("declared transitions are shorter than the requested level count")
    energies = [0.0]
    for index in range(d_e - 1):
        energies.append(energies[-1] + float(transitions[index]))
    return energies


def _operators(n_cav: int, d_e: int) -> dict[str, qt.Qobj]:
    """Build tensor-ordered cavity and emitter operators.

    Args:
        n_cav: Cavity Fock-space dimension, dimensionless.
        d_e: Emitter Hilbert-space dimension, dimensionless.

    Returns:
        dict[str, qt.Qobj]: Cavity annihilation, number, emitter ladder
        lowering/raising operators, and the computational-subspace projector.
    """
    a = qt.tensor(qt.destroy(n_cav), qt.qeye(d_e))
    ladder = np.zeros((d_e, d_e), dtype=complex)
    for index in range(d_e - 1):
        ladder[index, index + 1] = math.sqrt(index + 1.0)
    lowering = qt.Qobj(ladder, dims=[[d_e], [d_e]])
    high = np.zeros((d_e, d_e), dtype=complex)
    for index in range(2, d_e):
        high[index, index] = 1.0
    projector_high = qt.Qobj(high, dims=[[d_e], [d_e]])
    return {
        "a": a,
        "n": a.dag() * a,
        "lowering": qt.tensor(qt.qeye(n_cav), lowering),
        "raising": qt.tensor(qt.qeye(n_cav), lowering.dag()),
        "projector_high": qt.tensor(qt.qeye(n_cav), projector_high),
        "projector_01": qt.tensor(
            qt.qeye(n_cav), qt.qeye(d_e) - projector_high
        ),
        "identity": qt.tensor(qt.qeye(n_cav), qt.qeye(d_e)),
    }


def _static_hamiltonian(
    eta: float,
    energies: list[float],
    operators: dict[str, qt.Qobj],
    n_cav: int,
) -> qt.Qobj:
    """Construct the static multilevel Rabi Hamiltonian.

    Args:
        eta: Static coupling g0/wc, dimensionless.
        energies: Emitter level energies in normalized angular-frequency units.
        operators: Tensor-ordered operators from ``_operators``.
        n_cav: Cavity Fock-space dimension, dimensionless.

    Returns:
        qt.Qobj: Static Hamiltonian in normalized angular-frequency units.
    """
    del n_cav
    emitter_part = qt.Qobj(
        np.diag(energies),
        dims=[[len(energies)], [len(energies)]],
    )
    emitter_hamiltonian = qt.tensor(
        qt.qeye(operators["a"].dims[0][0]), emitter_part
    )
    interaction = (operators["a"] + operators["a"].dag()) * (
        operators["lowering"] + operators["raising"]
    )
    return emitter_hamiltonian + operators["n"] + eta * interaction


def _driven_hamiltonian(
    eta: float,
    gd: float,
    energies: list[float],
    operators: dict[str, qt.Qobj],
    config: MultilevelConfig,
) -> qt.QobjEvo:
    """Construct the driven multilevel Hamiltonian.

    Args:
        eta: Static coupling g0/wc, dimensionless.
        gd: Additive normalized DCE coefficient, dimensionless.
        energies: Emitter level energies, dimensionless.
        operators: Tensor-ordered operators from ``_operators``.
        config: Centralized multilevel configuration.

    Returns:
        qt.QobjEvo: Time-dependent Hamiltonian in normalized angular-frequency
        units.
    """
    emitter_part = qt.Qobj(
        np.diag(energies),
        dims=[[len(energies)], [len(energies)]],
    )
    emitter_hamiltonian = qt.tensor(
        qt.qeye(operators["a"].dims[0][0]), emitter_part
    )
    interaction = (operators["a"] + operators["a"].dag()) * (
        operators["lowering"] + operators["raising"]
    )
    static = emitter_hamiltonian + operators["n"]

    def coefficient(time_normalized: float, _args: dict | None = None) -> float:
        """Return the instantaneous driven coupling coefficient.

        Args:
            time_normalized: Time in inverse cavity angular-frequency units.
            _args: Optional QuTiP coefficient arguments, unused.

        Returns:
            float: Dimensionless time-dependent coupling coefficient.
        """
        return eta + gd * math.cos(config.wd_wc * time_normalized)

    return qt.QobjEvo([static, [interaction, coefficient]])


def _collapse_operators(
    kappa_rel: float,
    operators: dict[str, qt.Qobj],
    d_e: int,
    config: MultilevelConfig,
) -> list[qt.Qobj]:
    """Construct cavity-loss, relaxation, and dephasing collapse operators.

    Args:
        kappa_rel: Cavity-loss rate divided by wc, dimensionless.
        operators: Tensor-ordered operators.
        d_e: Emitter dimension, dimensionless.
        config: Centralized multilevel configuration.

    Returns:
        list[qt.Qobj]: Collapse operators for the declared loss rows.
    """
    collapse: list[qt.Qobj] = []
    if kappa_rel > 0.0:
        collapse.append(math.sqrt(kappa_rel) * operators["a"])
    # Declared zero-temperature emitter relaxation and dephasing rates
    # (normalized by wc), imported from centralized configuration.
    gamma1_rel = config.gamma1_rel
    gamma_phi_rel = config.gamma_phi_rel
    if d_e >= 2:
        collapse.append(math.sqrt(gamma1_rel) * operators["lowering"])
    number_like = qt.Qobj(
        np.diag(np.arange(d_e, dtype=float)),
        dims=[[d_e], [d_e]],
    )
    dephasing = qt.tensor(qt.qeye(operators["a"].dims[0][0]), number_like)
    collapse.append(math.sqrt(gamma_phi_rel) * dephasing)
    return collapse


def _density_matrix(state: qt.Qobj) -> qt.Qobj:
    """Return a unit-trace density operator for a ket or density operator.

    Args:
        state: QuTiP state, dimensionless.

    Returns:
        qt.Qobj: Unit-trace density operator, dimensionless.
    """
    rho = state.proj() if state.isket else state
    trace = float(np.real(rho.tr()))
    if trace <= 0.0:
        raise ValueError("State trace is non-positive.")
    return rho / trace


def _squared_fidelity(state_a: qt.Qobj, state_b: qt.Qobj) -> float:
    """Compute squared Uhlmann fidelity for two states.

    Args:
        state_a: First ket or density operator, dimensionless.
        state_b: Second ket or density operator, dimensionless.

    Returns:
        float: Squared Uhlmann fidelity in [0, 1], dimensionless.
    """
    if state_a.isket and state_b.isket:
        overlap = abs(complex(state_a.dag() * state_b)) ** 2
        return float(np.clip(overlap, 0.0, 1.0))
    rho_a = _density_matrix(state_a)
    rho_b = _density_matrix(state_b)
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
    values = []
    for state_a, state_b in zip(states_a, states_b):
        fidelity = _squared_fidelity(state_a, state_b)
        values.append(8.0 * (1.0 - math.sqrt(float(np.clip(fidelity, 0.0, 1.0)))) / delta_eta**2)
    return np.asarray(values, dtype=float)


def _entropy_bits(rho: qt.Qobj) -> float:
    """Return the von Neumann entropy in bits of a density operator.

    Args:
        rho: Unit-trace density operator, dimensionless.

    Returns:
        float: Von Neumann entropy in bits.
    """
    eigenvalues = np.real_if_close(rho.eigenenergies()).real
    probabilities = np.clip(eigenvalues, 0.0, 1.0)
    probabilities = probabilities[probabilities > 0.0]
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _purity(rho: qt.Qobj) -> float:
    """Return the purity of a density operator.

    Args:
        rho: Unit-trace density operator, dimensionless.

    Returns:
        float: Purity Tr(rho^2), dimensionless, clipped to [0, 1].
    """
    return float(np.clip(np.real((rho * rho).tr()), 0.0, 1.0))


def _physicality_summary(states: list[qt.Qobj], numerical_eigenvalue_floor: float) -> dict[str, Any]:
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
        rho = _density_matrix(state)
        traces.append(float(np.real(rho.tr())))
        hermiticity_errors.append(float((rho - rho.dag()).norm()))
        if rho.isoper:
            minimum_eigenvalue = min(
                minimum_eigenvalue,
                float(np.min(np.real_if_close(rho.eigenenergies()).real)),
            )
    max_trace_error = max(abs(value - 1.0) for value in traces)
    max_hermiticity_error = max(hermiticity_errors)
    return {
        "max_trace_error": float(max_trace_error),
        "max_hermiticity_error": float(max_hermiticity_error),
        "minimum_eigenvalue": float(minimum_eigenvalue),
        "numerical_acceptance": bool(
            max_trace_error < 1.0e-8
            and max_hermiticity_error < 1.0e-8
            and minimum_eigenvalue >= numerical_eigenvalue_floor
        ),
        "note": "Numerical acceptance is not a proof of exact positivity.",
    }


def _row_id(parameters: dict[str, Any]) -> str:
    """Create a stable filesystem-safe row identifier.

    Args:
        parameters: Row parameters with scalar JSON values.

    Returns:
        str: Stable row identifier.
    """
    raw = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"row_{digest}"


def _static_row(
    parameters: dict[str, Any],
    transitions: tuple[float, ...],
    config: MultilevelConfig,
) -> dict[str, Any]:
    """Evaluate one static ground-state row.

    Args:
        parameters: Row parameters.
        transitions: Declared emitter transition ratios, dimensionless.
        config: Centralized multilevel configuration.

    Returns:
        dict[str, Any]: JSON-ready static diagnostics.
    """
    d_e = int(parameters["d_e"])
    eta = float(parameters["eta"])
    energies = emitter_energies(transitions, d_e)
    operators = _operators(config.n_cav, d_e)
    hamiltonian = _static_hamiltonian(eta, energies, operators, config.n_cav)
    perturbed = _static_hamiltonian(
        eta + config.delta_eta, energies, operators, config.n_cav
    )
    _, ground = hamiltonian.groundstate()
    _, ground_perturbed = perturbed.groundstate()
    fidelity = _squared_fidelity(ground, ground_perturbed)
    qfi = 8.0 * (1.0 - math.sqrt(fidelity)) / config.delta_eta**2
    ground_leakage = float(
        np.real(qt.expect(operators["projector_high"], ground))
    )
    photon_number = float(
        np.real(qt.expect(operators["n"], ground))
    )
    return {
        "row_type": "static",
        "qfi_ground_state": float(qfi),
        "ground_state_leakage_levels_ge_2": float(np.clip(ground_leakage, 0.0, 1.0)),
        "ground_state_photon_number": float(photon_number),
        "ground_state_energy": float(np.real(hamiltonian.matrix_element(ground, ground))),
        "two_level_adequate": None,
    }


def _driven_row(
    parameters: dict[str, Any],
    transitions: tuple[float, ...],
    config: MultilevelConfig,
) -> dict[str, Any]:
    """Simulate one driven leakage/QFI row.

    Args:
        parameters: Row parameters.
        transitions: Declared emitter transition ratios, dimensionless.
        config: Centralized multilevel configuration.

    Returns:
        dict[str, Any]: JSON-ready time-series metrics and diagnostics.
    """
    d_e = int(parameters["d_e"])
    eta = float(parameters["eta"])
    gd = float(parameters["gd"])
    kappa_rel = float(parameters["kappa_rel"])
    energies = emitter_energies(transitions, d_e)
    operators = _operators(config.n_cav, d_e)
    drive_period = 2.0 * math.pi / config.wd_wc
    final_time = config.n_drive_periods * drive_period
    times = np.linspace(0.0, final_time, config.n_time_points)
    hamiltonian = _hamiltonian_for_row(parameters, energies, operators, config, eta)
    perturbed_hamiltonian = _hamiltonian_for_row(
        parameters, energies, operators, config, eta + config.delta_eta
    )
    initial = qt.tensor(qt.basis(config.n_cav, 0), qt.basis(d_e, 0))
    collapse_ops = _collapse_operators(kappa_rel, operators, d_e, config)
    options = {
        "atol": config.solver_atol,
        "rtol": config.solver_rtol,
        "method": "adams",
        "max_step": config.max_step_fraction * drive_period,
        "store_states": True,
        "progress_bar": "",
    }
    if kappa_rel == 0.0:
        reference = qt.sesolve(hamiltonian, initial, times, e_ops=[], options=options)
        perturbed = qt.sesolve(
            perturbed_hamiltonian, initial, times, e_ops=[], options=options
        )
        solver_name = "qutip.sesolve"
    else:
        reference = qt.mesolve(
            hamiltonian,
            initial,
            times,
            c_ops=collapse_ops,
            e_ops=[],
            options=options,
        )
        perturbed = qt.mesolve(
            perturbed_hamiltonian,
            initial,
            times,
            c_ops=collapse_ops,
            e_ops=[],
            options=options,
        )
        solver_name = "qutip.mesolve"
    qfi = _qfi_trace(reference.states, perturbed.states, config.delta_eta)
    periods = times / drive_period
    photon_number = np.asarray(
        [float(np.real(qt.expect(operators["n"], state))) for state in reference.states],
        dtype=float,
    )
    leakage = np.asarray(
        [
            float(np.real(qt.expect(operators["projector_high"], state)))
            for state in reference.states
        ],
        dtype=float,
    )
    purity = np.asarray(
        [_purity(_density_matrix(state)) for state in reference.states],
        dtype=float,
    )
    entropy = np.asarray(
        [_entropy_bits(_density_matrix(state)) for state in reference.states],
        dtype=float,
    )
    leakage = np.clip(leakage, 0.0, 1.0)
    qfi_peak_index = int(np.argmax(qfi))
    leakage_max_index = int(np.argmax(leakage))
    leakage_tolerance = config.leakage_tolerance
    return {
        "row_type": "driven",
        "qfi": qfi.tolist(),
        "photon_number": photon_number.tolist(),
        "leakage_levels_ge_2": leakage.tolist(),
        "purity": purity.tolist(),
        "von_neumann_entropy_bits": entropy.tolist(),
        "time_drive_periods": periods.tolist(),
        "qfi_peak": float(qfi[qfi_peak_index]),
        "qfi_peak_period": float(periods[qfi_peak_index]),
        "qfi_final": float(qfi[-1]),
        "leakage_max": float(leakage[leakage_max_index]),
        "leakage_max_period": float(periods[leakage_max_index]),
        "leakage_final": float(leakage[-1]),
        "leakage_at_qfi_peak": float(leakage[qfi_peak_index]),
        "photon_number_peak": float(np.max(photon_number)),
        "purity_at_qfi_peak": float(purity[qfi_peak_index]),
        "entropy_at_qfi_peak_bits": float(entropy[qfi_peak_index]),
        "two_level_adequate": (
            bool(leakage_max_index < len(leakage))
            and float(leakage[leakage_max_index]) <= leakage_tolerance
        ),
        "leakage_tolerance": float(leakage_tolerance),
        "solver": solver_name,
        "physicality": _physicality_summary(
            reference.states, config.numerical_eigenvalue_floor
        ),
        "perturbed_physicality": _physicality_summary(
            perturbed.states, config.numerical_eigenvalue_floor
        ),
    }


def _hamiltonian_for_row(
    parameters: dict[str, Any],
    energies: list[float],
    operators: dict[str, qt.Qobj],
    config: MultilevelConfig,
    eta: float,
) -> qt.QobjEvo:
    """Build the driven Hamiltonian with an optional frequency offset.

    Args:
        parameters: Row parameters (``gd`` and optional ``drive_offset``).
        energies: Emitter level energies, dimensionless.
        operators: Tensor-ordered operators.
        config: Centralized multilevel configuration.
        eta: Static coupling for this propagation, dimensionless.

    Returns:
        qt.QobjEvo: Time-dependent Hamiltonian in normalized angular-frequency
        units.
    """
    gd = float(parameters["gd"])
    offset = float(parameters.get("drive_offset", 0.0))
    effective_wd = config.wd_wc * (1.0 + offset)
    emitter_part = qt.Qobj(
        np.diag(energies),
        dims=[[len(energies)], [len(energies)]],
    )
    emitter_hamiltonian = qt.tensor(
        qt.qeye(operators["a"].dims[0][0]), emitter_part
    )
    interaction = (operators["a"] + operators["a"].dag()) * (
        operators["lowering"] + operators["raising"]
    )
    static = emitter_hamiltonian + operators["n"]

    def coefficient(time_normalized: float, _args: dict | None = None) -> float:
        """Return the instantaneous driven coupling coefficient.

        Args:
            time_normalized: Time in inverse cavity angular-frequency units.
            _args: Optional QuTiP coefficient arguments, unused.

        Returns:
            float: Dimensionless time-dependent coupling coefficient.
        """
        return eta + gd * math.cos(effective_wd * time_normalized)

    return qt.QobjEvo([static, [interaction, coefficient]])


def _parameter_rows(config: MultilevelConfig) -> list[dict[str, Any]]:
    """Enumerate the deterministic multilevel parameter grid.

    Args:
        config: Centralized multilevel configuration.

    Returns:
        list[dict[str, Any]]: Ordered parameter rows.
    """
    rows: list[dict[str, Any]] = []
    for model, transitions in config.transition_ratios_by_model.items():
        for d_e in config.emitter_levels:
            if len(transitions) < d_e - 1:
                continue
            for eta in config.static_eta_values:
                rows.append(
                    {
                        "row_kind": "static",
                        "model": model,
                        "d_e": d_e,
                        "eta": eta,
                    }
                )
            # Driven rows at the canonical DCE frequency.  Mixed rows use the
            # physical cavity-loss point with declared emitter relaxation and
            # dephasing; pure rows screen leakage without dissipation.
            for gd in config.drive_coefficients:
                for kappa_rel in config.kappa_values:
                    rows.append(
                        {
                            "row_kind": "driven",
                            "model": model,
                            "d_e": d_e,
                            "eta": config.eta,
                            "gd": gd,
                            "kappa_rel": kappa_rel,
                        }
                    )
    # Detuned strong-drive screening rows for the weak-anharmonicity model.
    for offset in (-0.02, 0.02):
        for d_e in (2, 5):
            rows.append(
                {
                    "row_kind": "driven",
                    "model": "transmon_like",
                    "d_e": d_e,
                    "eta": config.eta,
                    "gd": 0.5,
                    "kappa_rel": 0.0,
                    "drive_offset": offset,
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the multilevel producer.

    Returns:
        argparse.Namespace: Output, checkpoint, limit, and window options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "multilevel_emitters.json",
        help="Merged JSON output path.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Per-row checkpoint directory; defaults to data/_cache/multilevel_emitters.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N deterministic rows.",
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
        "--truncation",
        type=int,
        default=None,
        help="Override the configured cavity dimension for validation runs.",
    )
    return parser.parse_args()


def main() -> int:
    """Run and merge the resumable multilevel-emitter screen.

    Returns:
        int: Zero after successful checkpoint merge and JSON serialization.
    """
    args = parse_args()
    config = CONFIG
    if args.truncation is not None:
        if args.truncation < 4:
            raise ValueError("--truncation must be at least 4")
        config = replace(config, n_cav=args.truncation)
    if args.periods is not None:
        if args.periods < 3:
            raise ValueError("--periods must be at least 3")
        config = replace(config, n_drive_periods=args.periods)
    checkpoint_dir = args.checkpoint_dir or args.output.parent / "_cache" / "multilevel_emitters"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    parameters_list = _parameter_rows(config)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        parameters_list = parameters_list[: args.limit]

    results: list[dict[str, Any]] = []
    for index, parameters in enumerate(parameters_list, start=1):
        row_id = _row_id(parameters)
        checkpoint = checkpoint_dir / f"{row_id}.json"
        if checkpoint.is_file() and not args.no_resume:
            row_result = json.loads(checkpoint.read_text(encoding="utf-8"))
            print(f"[{index}/{len(parameters_list)}] resumed {row_id}")
        else:
            print(f"[{index}/{len(parameters_list)}] simulating {row_id}")
            transitions = config.transition_ratios_by_model[parameters["model"]]
            if parameters["row_kind"] == "static":
                metrics = _static_row(parameters, transitions, config)
            else:
                metrics = _driven_row(parameters, transitions, config)
            row_result = {"row_id": row_id, "parameters": parameters, **metrics}
            checkpoint.write_text(json.dumps(row_result, indent=2) + "\n", encoding="utf-8")
        results.append(row_result)

    results.sort(key=lambda item: item["row_id"])
    # Two-level adequacy and baseline comparison summaries.
    driven = [row for row in results if row["row_type"] == "driven" and row["parameters"]["d_e"] > 2]
    baseline = {
        (
            row["parameters"]["model"],
            row["parameters"]["gd"],
            row["parameters"]["kappa_rel"],
            row["parameters"].get("drive_offset", 0.0),
        ): row
        for row in results
        if row["row_type"] == "driven" and row["parameters"]["d_e"] == 2
    }
    comparisons = []
    for row in driven:
        key = (
            row["parameters"]["model"],
            row["parameters"]["gd"],
            row["parameters"]["kappa_rel"],
            row["parameters"].get("drive_offset", 0.0),
        )
        two_level = baseline.get(key)
        if two_level is None:
            continue
        comparisons.append(
            {
                "row_id": row["row_id"],
                "two_level_qfi_peak": two_level["qfi_peak"],
                "multilevel_qfi_peak": row["qfi_peak"],
                "qfi_peak_relative_change_from_two_level": float(
                    (row["qfi_peak"] - two_level["qfi_peak"]) / max(two_level["qfi_peak"], 1.0e-30)
                ),
                "leakage_max": row["leakage_max"],
                "two_level_adequate_in_window": row["two_level_adequate"],
            }
        )
    payload = {
        "schema": "trackB_multilevel_emitters_v1",
        "status": "effective_spectrum_leakage_screen",
        "purpose": "Assess leakage out of the computational subspace and QFI change for declared effective multilevel emitter spectra.",
        "configuration": asdict(config),
        "units": {
            "eta": "g0/wc, dimensionless",
            "gd": "existing normalized additive DCE coefficient, dimensionless",
            "transition_ratios": "Delta_{e,e+1}/wc, dimensionless",
            "kappa_rel": "kappa/wc, dimensionless",
            "time": "drive periods",
            "qfi": "inverse squared normalized coupling",
            "leakage": "population of emitter levels >= 2, dimensionless",
            "photon_number": "photons",
            "entropy": "bits",
        },
        "model_note": (
            "Each platform is an effective spectrum with the shared dipole-like ladder "
            "operator sum_e sqrt(e+1)(|e><e+1|+|e+1><e|); the models differ only in the "
            "declared level spacings. Effective spectra are not full circuit simulations."
        ),
        "solver": {
            "name": "qutip.sesolve for kappa_rel=0; qutip.mesolve otherwise",
            "collapse_note": "Loss rows add sqrt(kappa) a, declared zero-temperature emitter relaxation sqrt(gamma1) b, and number-like dephasing; gamma1/wc=1e-6 and gamma_phi/wc=3e-5.",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "qutip": qt.__version__,
        },
        "validation": {
            "rows_requested": len(parameters_list),
            "rows_completed": len(results),
            "all_rows_complete": len(parameters_list) == len(results),
            "numerical_physicality_note": "Trace/Hermiticity/eigenvalue acceptance is not a proof of exact positivity.",
            "leakage_tolerance_note": (
                "The adequacy criterion (max leakage <= tolerance over the window) is a "
                "declared model choice, not a universal validity condition."
            ),
            "two_level_baseline_comparisons": comparisons,
        },
        "caveats": [
            "Effective spectra are model proxies with shared ladder matrix elements; they are not device simulations.",
            "Adequacy is reported for the explicitly parameterized model and finite drive window only.",
            "No universal two-level validity, definitive platform choice, universal anharmonicity requirement, or measured flux-noise limit is inferred.",
            "The time-dependent effective USC Hamiltonian retains the manuscript gauge/control caveat.",
        ],
        "rows": results,
        "determinism": "Payload is byte-deterministic: no wall-clock or PRNG state is embedded.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output} ({len(results)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
