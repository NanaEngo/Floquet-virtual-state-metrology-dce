#!/usr/bin/env python3
"""Compute the Track B finite-temperature thermal-competition screen.

The producer propagates the canonical single-qubit Rabi/DCE model under
explicitly declared finite-temperature cavity (and selected qubit) Lindblad
channels with separate emission and absorption operators, and reports the
fidelity-based coupling QFI, thermal occupations, entropy, purity, and
retention relative to a matched zero-temperature reference row.

Every result is model-specific and finite-window.  The producer does not
infer a universal thermal threshold, a thermodynamic critical temperature,
or a platform-independent operating limit.
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
    from config import (
        DEFAULT_THERMAL,
        PLANCK_OVER_KB_S_K,
        ThermalConfig,
    )
except ImportError as exc:  # pragma: no cover - environment-specific
    raise ImportError(
        "Track B configuration is unavailable; run this script from scripts/."
    ) from exc


CONFIG = DEFAULT_THERMAL


def bose_einstein_occupation(frequency_GHz: float, temperature_mK: float) -> float:
    """Evaluate the Bose-Einstein occupation at a declared frequency.

    Args:
        frequency_GHz: Physical frequency in gigahertz (a linear frequency).
        temperature_mK: Bath temperature in millikelvin.

    Returns:
        float: Dimensionless thermal occupation n_th = 1/(exp(h nu/k_B T)-1).
    """
    if frequency_GHz <= 0.0:
        raise ValueError("frequency_GHz must be positive")
    if temperature_mK <= 0.0:
        raise ValueError("temperature_mK must be positive for a thermal occupation")
    exponent = PLANCK_OVER_KB_S_K * 1.0e9 * frequency_GHz / (1.0e-3 * temperature_mK)
    if exponent > 700.0:
        return 0.0
    return float(1.0 / (math.exp(exponent) - 1.0))


def _operators(n_cav: int, d_qubit: int = 2) -> dict[str, qt.Qobj]:
    """Build tensor-ordered cavity and qubit operators.

    Args:
        n_cav: Cavity Fock-space dimension, dimensionless.
        d_qubit: Qubit Hilbert-space dimension, dimensionless (default 2).

    Returns:
        dict[str, qt.Qobj]: Annihilation, number, sigma-x, sigma-z, sigma-,
        and sigma+ operators in normalized angular-frequency units.
    """
    a = qt.tensor(qt.destroy(n_cav), qt.qeye(d_qubit))
    sigma_z = qt.tensor(qt.qeye(n_cav), qt.sigmaz())
    return {
        "a": a,
        "n": a.dag() * a,
        "x": qt.tensor(qt.qeye(n_cav), qt.sigmax()),
        "z": sigma_z,
        "minus": qt.tensor(qt.qeye(n_cav), qt.sigmam()),
        "plus": qt.tensor(qt.qeye(n_cav), qt.sigmap()),
        "identity": qt.tensor(qt.qeye(n_cav), qt.qeye(d_qubit)),
    }


def _hamiltonian(
    eta: float,
    protocol: str,
    config: ThermalConfig,
    operators: dict[str, qt.Qobj],
) -> qt.Qobj | qt.QobjEvo:
    """Construct a static or driven normalized Rabi Hamiltonian.

    Args:
        eta: Static normalized coupling g0/wc, dimensionless.
        protocol: ``static_rabi`` or ``driven_dce``.
        config: Centralized thermal configuration.
        operators: Tensor-ordered operators from ``_operators``.

    Returns:
        qt.Qobj | qt.QobjEvo: Hamiltonian in normalized angular-frequency
        units.

    Raises:
        ValueError: If ``protocol`` is not supported.
    """
    static = 0.5 * config.wq_wc * operators["z"] + operators["n"]
    interaction = operators["x"] * (operators["a"] + operators["a"].dag())
    if protocol == "static_rabi":
        return static + eta * interaction
    if protocol != "driven_dce":
        raise ValueError(f"Unsupported protocol: {protocol}")

    def coefficient(time_normalized: float, _args: dict | None = None) -> float:
        """Return the instantaneous driven interaction coefficient.

        Args:
            time_normalized: Time in inverse cavity angular-frequency units.
            _args: Optional QuTiP coefficient arguments, unused.

        Returns:
            float: Dimensionless time-dependent coupling coefficient.
        """
        return eta + config.gd * math.cos(config.wd_wc * time_normalized)

    return qt.QobjEvo([static, [interaction, coefficient]])


def _dressed_ground_state(
    static_hamiltonian: qt.Qobj, n_cav: int
) -> qt.Qobj:
    """Return the normalized ground ket of a static Hamiltonian.

    Args:
        static_hamiltonian: Static Hamiltonian in normalized angular-frequency
            units.
        n_cav: Cavity Fock-space dimension, dimensionless (kept for a clear
            subsystem convention).

    Returns:
        qt.Qobj: Normalized ground-state ket.
    """
    del n_cav
    energy, state = static_hamiltonian.groundstate()
    del energy
    return state / state.norm()


def _initial_state(
    parameters: dict[str, Any],
    operators: dict[str, qt.Qobj],
    config: ThermalConfig,
    eta: float,
    n_cav: int,
) -> qt.Qobj:
    """Construct the declared initial state for one reference or perturbed row.

    Args:
        parameters: Row parameters including ``init_kind`` and, for thermal
            rows, ``cavity_frequency_GHz`` and ``temperature_mK``.
        operators: Tensor-ordered operators, used for static protocols.
        config: Centralized thermal configuration.
        eta: Coupling used for dressed-state construction, dimensionless.
        n_cav: Cavity Fock-space dimension, dimensionless.

    Returns:
        qt.Qobj: Initial density operator or ket.

    Raises:
        ValueError: If ``init_kind`` is not supported.
    """
    init_kind = parameters["init_kind"]
    if init_kind == "vacuum":
        return qt.tensor(qt.basis(n_cav, 0), qt.basis(2, 0))
    if init_kind == "bare_thermal":
        temperature_mK = float(parameters["temperature_mK"])
        frequency_GHz = float(parameters["cavity_frequency_GHz"])
        occupation = (
            bose_einstein_occupation(frequency_GHz, temperature_mK)
            if temperature_mK > 0.0
            else 0.0
        )
        return qt.tensor(
            qt.thermal_dm(n_cav, occupation), qt.basis(2, 0).proj()
        )
    if init_kind == "dressed":
        static_hamiltonian = _hamiltonian(
            eta, "static_rabi", config, operators
        )
        return _dressed_ground_state(static_hamiltonian, n_cav)
    raise ValueError(f"Unsupported init_kind: {init_kind}")


def _collapse_operators(
    parameters: dict[str, Any],
    operators: dict[str, qt.Qobj],
    config: ThermalConfig,
) -> list[qt.Qobj]:
    """Construct the declared thermal Lindblad collapse operators.

    Args:
        parameters: Row parameters with temperature, cavity frequency, cavity
            loss, and optional qubit relaxation rate.
        operators: Tensor-ordered operators.
        config: Centralized thermal configuration.

    Returns:
        list[qt.Qobj]: Emission/absorption collapse operators for the cavity
        channel and, when declared, the qubit channel.  A zero-temperature row
        keeps only spontaneous emission terms.
    """
    kappa_rel = float(parameters["kappa_rel"])
    gamma1_rel = float(parameters.get("gamma1_rel", 0.0))
    temperature_mK = float(parameters["temperature_mK"])
    cavity_frequency_GHz = float(parameters["cavity_frequency_GHz"])
    if temperature_mK > 0.0:
        nth = bose_einstein_occupation(cavity_frequency_GHz, temperature_mK)
    else:
        nth = 0.0
    collapse = []
    if kappa_rel > 0.0:
        collapse.append(math.sqrt(kappa_rel * (nth + 1.0)) * operators["a"])
        if nth > 0.0:
            collapse.append(math.sqrt(kappa_rel * nth) * operators["a"].dag())
    if gamma1_rel > 0.0:
        qubit_frequency_GHz = float(parameters["qubit_frequency_GHz"])
        nq = (
            bose_einstein_occupation(qubit_frequency_GHz, temperature_mK)
            if temperature_mK > 0.0
            else 0.0
        )
        collapse.append(math.sqrt(gamma1_rel * (nq + 1.0)) * operators["minus"])
        if nq > 0.0:
            collapse.append(math.sqrt(gamma1_rel * nq) * operators["plus"])
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
        if not math.isfinite(fidelity) or fidelity < -1.0e-8 or fidelity > 1.0 + 1.0e-8:
            raise ValueError(f"Invalid fidelity value: {fidelity}")
        values.append(8.0 * (1.0 - math.sqrt(float(np.clip(fidelity, 0.0, 1.0)))) / delta_eta**2)
    return np.asarray(values, dtype=float)


def _von_neumann_entropy_bits(rho: qt.Qobj) -> float:
    """Compute the von Neumann entropy in bits of a density operator.

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
    """Compute the purity of a density operator.

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


def _propagate(
    parameters: dict[str, Any],
    eta: float,
    operators: dict[str, qt.Qobj],
    config: ThermalConfig,
) -> dict[str, Any]:
    """Propagate one reference or perturbed row.

    Args:
        parameters: Row parameters.
        eta: Coupling g0/wc for this propagation, dimensionless.
        operators: Tensor-ordered operators.
        config: Centralized thermal configuration.

    Returns:
        dict[str, Any]: State trajectory, time grid, and solver label.
    """
    n_cav = config.n_cav
    drive_period = 2.0 * math.pi / config.wd_wc
    final_time = config.n_drive_periods * drive_period
    times = np.linspace(0.0, final_time, config.n_time_points)
    protocol = parameters["protocol"]
    hamiltonian = _hamiltonian(eta, protocol, config, operators)
    initial = _initial_state(parameters, operators, config, eta, n_cav)
    collapse_ops = _collapse_operators(parameters, operators, config)
    options = {
        "atol": config.solver_atol,
        "rtol": config.solver_rtol,
        "method": "adams",
        "max_step": config.max_step_fraction * drive_period,
        "store_states": True,
        "progress_bar": "",
    }
    solver_name = "qutip.mesolve"
    result = qt.mesolve(
        hamiltonian,
        initial,
        times,
        c_ops=collapse_ops,
        e_ops=[],
        options=options,
    )
    return {
        "states": result.states,
        "times": times,
        "drive_period": drive_period,
        "solver": solver_name,
    }


def _family_key(parameters: dict[str, Any]) -> dict[str, Any]:
    """Return the zero-temperature-reference family key for a row.

    Args:
        parameters: Row parameters.

    Returns:
        dict[str, Any]: Parameters that define the matched reference family;
        temperature is forced to zero and frequency metadata is retained.
    """
    key = dict(parameters)
    key["temperature_mK"] = 0.0
    key.pop("row_id", None)
    key.pop("is_zero_temperature_reference", None)
    return key


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


def _simulate_row(parameters: dict[str, Any], config: ThermalConfig) -> dict[str, Any]:
    """Simulate one parameter row and return JSON-ready metrics.

    Args:
        parameters: Row parameters.
        config: Centralized thermal configuration.

    Returns:
        dict[str, Any]: JSON-ready time-series metrics and diagnostics.
    """
    operators = _operators(config.n_cav)
    eta = float(parameters["eta"])
    reference = _propagate(parameters, eta, operators, config)
    perturbed = _propagate(parameters, eta + config.delta_eta, operators, config)
    qfi = _qfi_trace(reference["states"], perturbed["states"], config.delta_eta)
    times = reference["times"]
    drive_period = reference["drive_period"]
    periods = times / drive_period
    number_operator = operators["n"]
    photons = np.asarray(
        [
            float(np.real(qt.expect(number_operator, state)))
            for state in reference["states"]
        ],
        dtype=float,
    )
    entropy = np.asarray(
        [_von_neumann_entropy_bits(_density_matrix(state)) for state in reference["states"]],
        dtype=float,
    )
    purities = np.asarray(
        [_purity(_density_matrix(state)) for state in reference["states"]],
        dtype=float,
    )
    qfi_peak_index = int(np.argmax(qfi))
    late_count = max(2, int(0.25 * len(qfi)))
    late_window_mean = float(np.mean(qfi[-late_count:]))
    temperature_mK = float(parameters["temperature_mK"])
    frequency_GHz = float(parameters["cavity_frequency_GHz"])
    kappa_rel = float(parameters["kappa_rel"])
    gamma1_rel = float(parameters.get("gamma1_rel", 0.0))
    return {
        "qfi": qfi.tolist(),
        "photon_number": photons.tolist(),
        "von_neumann_entropy_bits": entropy.tolist(),
        "purity": purities.tolist(),
        "time_drive_periods": periods.tolist(),
        "qfi_peak": float(qfi[qfi_peak_index]),
        "qfi_peak_period": float(periods[qfi_peak_index]),
        "qfi_late_window_mean": late_window_mean,
        "photon_number_peak": float(photons[np.argmax(photons)]),
        "photon_number_at_qfi_peak": float(photons[qfi_peak_index]),
        "photon_number_final": float(photons[-1]),
        "photon_number_mean": float(np.mean(photons)),
        "entropy_at_qfi_peak_bits": float(entropy[qfi_peak_index]),
        "purity_at_qfi_peak": float(purities[qfi_peak_index]),
        "thermal_occupation_cavity_nth": (
            bose_einstein_occupation(frequency_GHz, temperature_mK)
            if temperature_mK > 0.0
            else 0.0
        ),
        "thermal_occupation_qubit_nq": (
            bose_einstein_occupation(
                float(parameters["qubit_frequency_GHz"]), temperature_mK
            )
            if gamma1_rel > 0.0 and temperature_mK > 0.0
            else 0.0
        ),
        "physicality_reference": _physicality_summary(
            reference["states"], config.numerical_eigenvalue_floor
        ),
        "physicality_perturbed": _physicality_summary(
            perturbed["states"], config.numerical_eigenvalue_floor
        ),
        "solver": reference["solver"],
    }


def _main_rows(config: ThermalConfig) -> list[dict[str, Any]]:
    """Enumerate the declared finite-temperature parameter rows.

    Args:
        config: Centralized thermal configuration.

    Returns:
        list[dict[str, Any]]: Ordered parameter rows without row identifiers.
    """
    rows: list[dict[str, Any]] = []
    # Driven DCE rows from the zero-temperature vacuum initial state.
    for kappa_rel in config.kappa_values:
        for frequency_GHz in config.cavity_frequencies_GHz:
            for temperature_mK in config.temperatures_mK:
                rows.append(
                    {
                        "protocol": "driven_dce",
                        "eta": config.eta,
                        "init_kind": "vacuum",
                        "kappa_rel": kappa_rel,
                        "cavity_frequency_GHz": frequency_GHz,
                        "temperature_mK": temperature_mK,
                    }
                )
    # Driven rows with an explicitly declared qubit thermalization channel at
    # the physical cavity-loss point and 6 GHz reference frequency.
    for gamma1_rel in config.qubit_relaxation_rates:
        for qubit_frequency_GHz in config.qubit_thermal_frequencies_GHz:
            for temperature_mK in config.qubit_temperatures_mK:
                rows.append(
                    {
                        "protocol": "driven_dce",
                        "eta": config.eta,
                        "init_kind": "vacuum",
                        "kappa_rel": 4.1e-2,
                        "cavity_frequency_GHz": 6.0,
                        "temperature_mK": temperature_mK,
                        "gamma1_rel": gamma1_rel,
                        "qubit_frequency_GHz": qubit_frequency_GHz,
                    }
                )
    # Static Rabi rows from the dressed ground state at the physical loss.
    for eta in config.static_eta_values:
        for frequency_GHz in config.cavity_frequencies_GHz:
            for temperature_mK in config.temperatures_mK:
                rows.append(
                    {
                        "protocol": "static_rabi",
                        "eta": eta,
                        "init_kind": "dressed",
                        "kappa_rel": 4.1e-2,
                        "cavity_frequency_GHz": frequency_GHz,
                        "temperature_mK": temperature_mK,
                    }
                )
    # Bare-thermal-initial-state comparison rows at the 6 GHz reference.
    for protocol, eta in (("driven_dce", config.eta), ("static_rabi", config.eta)):
        for kappa_rel in (0.0, 4.1e-2):
            for temperature_mK in config.temperatures_mK:
                rows.append(
                    {
                        "protocol": protocol,
                        "eta": eta,
                        "init_kind": "bare_thermal",
                        "kappa_rel": kappa_rel,
                        "cavity_frequency_GHz": 6.0,
                        "temperature_mK": temperature_mK,
                    }
                )
    return rows


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the thermal producer.

    Returns:
        argparse.Namespace: Output, checkpoint, limit, and truncation options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "thermal_competition.json",
        help="Merged JSON output path.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Per-row checkpoint directory; defaults to data/_cache/thermal_competition.",
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
        "--truncation",
        type=int,
        default=None,
        help="Override the configured cavity dimension for validation runs.",
    )
    parser.add_argument(
        "--periods",
        type=int,
        default=None,
        help="Override the configured evolution window in drive periods.",
    )
    return parser.parse_args()


def _add_derived_retention(
    rows: list[dict[str, Any]],
    config: ThermalConfig,
) -> dict[str, Any]:
    """Attach zero-temperature-reference retention to finite-temperature rows.

    Args:
        rows: Simulated rows (references and finite-temperature rows).
        config: Centralized thermal configuration.

    Returns:
        dict[str, Any]: Per-family reference metrics and the sampled-crossover
        summary, plus the rows updated in place.
    """
    references: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["is_zero_temperature_reference"]:
            references[_row_id(_family_key(row["parameters"]))] = row
    families: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["is_zero_temperature_reference"]:
            continue
        family_id = _row_id(_family_key(row["parameters"]))
        reference = references.get(family_id)
        if reference is None:
            raise ValueError(f"Missing zero-temperature reference for family {family_id}")
        peak_retention = row["qfi_peak"] / reference["qfi_peak"]
        late_retention = row["qfi_late_window_mean"] / reference["qfi_late_window_mean"]
        retention_fraction = config.retention_fraction
        below_criterion = peak_retention < retention_fraction
        row["reference_row_id"] = reference["row_id"]
        row["reference_qfi_peak"] = reference["qfi_peak"]
        row["reference_qfi_late_window_mean"] = reference["qfi_late_window_mean"]
        row["retention_peak_qfi"] = float(peak_retention)
        row["retention_late_window_qfi"] = float(late_retention)
        row["peak_retention_below_criterion"] = bool(below_criterion)
        row["retention_criterion_fraction"] = float(retention_fraction)
        family = families.setdefault(
            family_id,
            {
                "family_id": family_id,
                "parameters": _family_key(row["parameters"]),
                "reference_row_id": reference["row_id"],
                "reference_qfi_peak": reference["qfi_peak"],
                "temperature_samples": [],
            },
        )
        family["temperature_samples"].append(
            {
                "temperature_mK": row["parameters"]["temperature_mK"],
                "retention_peak_qfi": peak_retention,
                "retention_late_window_qfi": late_retention,
                "peak_retention_below_criterion": below_criterion,
            }
        )
    summary = []
    for family_id, family in families.items():
        samples = sorted(family["temperature_samples"], key=lambda item: item["temperature_mK"])
        below = [sample for sample in samples if sample["peak_retention_below_criterion"]]
        first_below = below[0] if below else None
        family["first_sampled_temperature_below_criterion_mK"] = (
            first_below["temperature_mK"] if first_below else None
        )
        family["crossover_type"] = (
            "sampled_crossover"
            if first_below is not None
            else "no_sampled_crossover_in_range"
        )
        family.pop("temperature_samples", None)
        summary.append(family)
    return {"families": summary, "retention_criterion_fraction": config.retention_fraction}


def main() -> int:
    """Run and merge the resumable thermal-competition screen.

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
    checkpoint_dir = args.checkpoint_dir or args.output.parent / "_cache" / "thermal_competition"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    main_parameters = _main_rows(config)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        main_parameters = main_parameters[: args.limit]
    distinct_families: dict[str, dict[str, Any]] = {}
    for parameters in main_parameters:
        distinct_families.setdefault(_row_id(_family_key(parameters)), _family_key(parameters))
    reference_parameters = sorted(distinct_families.values(), key=lambda item: _row_id(item))
    for family in reference_parameters:
        family["temperature_mK"] = 0.0
    parameters_list = [*main_parameters, *reference_parameters]

    rows: list[dict[str, Any]] = []
    for index, parameters in enumerate(parameters_list, start=1):
        is_reference = parameters["temperature_mK"] == 0.0
        row_parameters = dict(parameters)
        row_id = _row_id(row_parameters)
        checkpoint = checkpoint_dir / f"{row_id}.json"
        if checkpoint.is_file() and not args.no_resume:
            row_result = json.loads(checkpoint.read_text(encoding="utf-8"))
            print(f"[{index}/{len(parameters_list)}] resumed {row_id}")
        else:
            print(f"[{index}/{len(parameters_list)}] simulating {row_id}")
            metrics = _simulate_row(parameters, config)
            row_result = {
                "row_id": row_id,
                "parameters": row_parameters,
                "is_zero_temperature_reference": bool(is_reference),
                **metrics,
            }
            checkpoint.write_text(json.dumps(row_result, indent=2) + "\n", encoding="utf-8")
        rows.append(row_result)

    rows.sort(key=lambda item: item["row_id"])
    retention_summary = _add_derived_retention(rows, config)
    payload = {
        "schema": "single_qubit_thermal_competition_v1",
        "status": "finite_temperature_lindblad_screen",
        "purpose": "Quantify how declared finite-temperature cavity/qubit thermal channels alter static and driven DCE QFI.",
        "configuration": asdict(config),
        "units": {
            "eta": "g0/wc, dimensionless",
            "gd": "existing normalized additive DCE coefficient, dimensionless",
            "temperature": "millikelvin",
            "cavity_frequency": "GHz (linear frequency used only for thermal occupations)",
            "kappa_rel": "kappa/wc, dimensionless",
            "gamma1_rel": "qubit relaxation rate / wc, dimensionless",
            "time": "drive periods of the canonical drive frequency",
            "qfi": "inverse squared normalized coupling",
            "photon_number": "photons",
            "entropy": "bits",
            "thermal_occupation": "dimensionless",
        },
        "thermal_model": {
            "cavity_channels": "L_- = sqrt(kappa (n_th+1)) a and L_+ = sqrt(kappa n_th) a^dagger",
            "qubit_channels": "L_- = sqrt(gamma1 (n_q+1)) sigma_minus and L_+ = sqrt(gamma1 n_q) sigma_plus, included only for declared rows",
            "occupation_formula": "n = 1/(exp(h nu / k_B T) - 1) evaluated at the declared physical frequency",
            "initial_states": {
                "vacuum": "bare cavity vacuum tensor qubit ground",
                "dressed": "ground state of the static Hamiltonian at the row coupling",
                "bare_thermal": "thermal cavity state at n_th tensor qubit ground",
            },
            "zero_temperature_reference": "matched family row at T=0 keeping spontaneous-emission collapse operators",
        },
        "solver": {
            "name": "qutip.mesolve",
            "options": {
                "atol": config.solver_atol,
                "rtol": config.solver_rtol,
                "method": "adams",
                "max_step_fraction_of_drive_period": config.max_step_fraction,
            },
            "python": platform.python_version(),
            "numpy": np.__version__,
            "qutip": qt.__version__,
        },
        "validation": {
            "rows_requested": len(parameters_list),
            "rows_completed": len(rows),
            "all_rows_complete": len(parameters_list) == len(rows),
            "zero_temperature_reference_count": int(
                sum(row["is_zero_temperature_reference"] for row in rows)
            ),
            "numerical_physicality_note": "Trace/Hermiticity/eigenvalue acceptance is not a proof of exact positivity.",
            "numerical_eigenvalue_floor": config.numerical_eigenvalue_floor,
            "retention_criterion_note": (
                "Sampled crossover flags use the declared retention fraction on peak QFI; "
                "this is a sampled criterion, not a critical temperature."
            ),
            "occupation_hypotheses": {
                "n_th_1e-4_at_20mK_6GHz": {
                    "status": "not_reproduced",
                    "basis": "Explicit Bose-Einstein occupation at the declared cavity frequency; the stored 6 GHz value at 20 mK is ~5.6e-7, three orders below 1e-4.",
                },
                "fourfold_reduction_at_100mK": {
                    "status": "not_reproduced",
                    "basis": "The stored 6 GHz occupations (5.6e-7 at 20 mK to 5.95e-2 at 100 mK) do not follow a fourfold-reduction law; QFI retention crosses the sampled 80% criterion near 100 mK for the canonical driven row instead.",
                },
            },
            **retention_summary,
        },
        "caveats": [
            "Results are model-specific finite-window Lindblad calculations; no universal thermal threshold or platform-independent operating limit follows.",
            "Thermal occupations are recorded from the Bose-Einstein formula at the declared frequency; they are not inferred without recording that frequency.",
            "The finite-temperature channel is not an exact USC-bath treatment; hierarchy/memory corrections are outside this package.",
            "Retention is relative to the matched zero-temperature reference row with the same protocol, coupling, loss, initial-state family, window, and truncation.",
            "The historical n_th approximately 1e-4 at 20 mK and fourfold-reduction hypotheses are treated as hypotheses to reproduce, not assumptions.",
            "Initial-state choices (vacuum, dressed, bare thermal) are declared models; their spread is reported rather than averaged away.",
            "The time-dependent effective USC Hamiltonian retains the manuscript gauge/control caveat.",
        ],
        "rows": rows,
        "determinism": "Payload is byte-deterministic: no wall-clock or PRNG state is embedded.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
