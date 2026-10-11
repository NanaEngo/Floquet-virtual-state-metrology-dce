#!/usr/bin/env python3
"""Run the extended HEOM/Lindblad validation screen for Track B.

The screen uses local spin-boson rows (two-level system coupled through
sigma-x to a declared Drude-Lorentz bath) with the canonical Track B
frequency ratios, plus driven rows at the DCE drive ratio.  The Lindblad
comparator is the Markovian weak-coupling limit of the same bath with
thermally weighted Pauli rates Gamma = 2*pi*J(omega_q).  Results are
explicitly scoped to the tested bath, parameter range, hierarchy depth,
and time window; no generalization to arbitrary USC baths is claimed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import qutip as qt
from qutip.core.environment import DrudeLorentzEnvironment
from qutip.solver.heom import DrudeLorentzBath, HEOMSolver

try:
    from config import (
        DEFAULT_HEOM_EXTENDED,
        HBAR_OVER_KB_S_K,
        HeomExtendedConfig,
    )
except ImportError as exc:  # pragma: no cover - environment-specific
    raise ImportError(
        "Track B configuration is unavailable; run this script from scripts/."
    ) from exc


CONFIG = DEFAULT_HEOM_EXTENDED


def spectral_density_j(lam: float, gamma: float, omega: float) -> float:
    """Evaluate the Drude-Lorentz spectral density used by the HEOM bath.

    Args:
        lam: Drude reorganization coupling, in wq units.
        gamma: Bath cutoff frequency, in wq units.
        omega: Positive frequency argument, in wq units.

    Returns:
        float: J(omega) = 2*lam*gamma*omega/(omega**2 + gamma**2).
    """
    return float(2.0 * lam * gamma * omega / (omega * omega + gamma * gamma))


def bose_occupation(omega: float, temperature: float) -> float:
    """Return the normalized Bose-Einstein occupation.

    Args:
        omega: Angular frequency in wq units, with hbar=k_B=1.
        temperature: Normalized temperature k_B T/(hbar wq), dimensionless.

    Returns:
        float: n = 1/(exp(omega/T) - 1), zero for very low temperature.
    """
    exponent = omega / temperature
    if exponent > 700.0:
        return 0.0
    return float(1.0 / (math.exp(exponent) - 1.0))


def _lindblad_comparator(
    config: HeomExtendedConfig,
    lam: float,
    gamma: float,
    temperature: float,
    driven: bool,
    eta_d: float,
) -> tuple[qt.Qobj | qt.QobjEvo, list[qt.Qobj]]:
    """Build the system Hamiltonian and Markovian-Lindblad collapse list.

    Args:
        config: Centralized HEOM configuration.
        lam: Drude reorganization coupling, in wq units.
        gamma: Bath cutoff frequency, in wq units.
        temperature: Normalized bath temperature, dimensionless.
        driven: Whether the system is amplitude-driven.
        eta_d: Effective driving amplitude for driven rows, dimensionless.

    Returns:
        tuple[qt.Qobj | qt.QobjEvo, list[qt.Qobj]]: System Hamiltonian and
        thermal Pauli collapse operators with rates from the same bath.
    """
    omega = spectral_density_j(lam, gamma, config.wq)
    occupation = bose_occupation(config.wq, temperature)
    rate_down = 2.0 * math.pi * omega * (1.0 + occupation)
    rate_up = 2.0 * math.pi * omega * occupation
    sigma_minus = qt.sigmam()
    sigma_plus = qt.sigmap()
    collapse = [
        math.sqrt(rate_down) * sigma_minus,
        math.sqrt(rate_up) * sigma_plus,
    ]
    if not driven:
        return 0.5 * config.wq * qt.sigmaz(), collapse
    gd = eta_d * config.driven_ratio

    def coefficient(time_normalized: float, _args: dict | None = None) -> float:
        """Return the instantaneous driving coefficient.

        Args:
            time_normalized: Time in inverse wq-frequency units.
            _args: Optional QuTiP coefficient arguments, unused.

        Returns:
            float: Dimensionless driving amplitude.
        """
        return eta_d + gd * math.cos(config.wd * time_normalized)

    static = 0.5 * config.wq * qt.sigmaz()
    return qt.QobjEvo([static, [qt.sigmax(), coefficient]]), collapse


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
    rho_a = _density_matrix(state_a)
    rho_b = _density_matrix(state_b)
    root_a = rho_a.sqrtm()
    product = root_a * rho_b * root_a
    eigenvalues = np.real_if_close(product.eigenenergies()).real
    root_fidelity = float(np.sum(np.sqrt(np.clip(eigenvalues, 0.0, None))))
    return float(np.clip(root_fidelity**2, 0.0, 1.0))


def _fidelity_qfi(states_a: list[qt.Qobj], states_b: list[qt.Qobj], delta: float) -> float:
    """Estimate coupling QFI from paired final states.

    Args:
        states_a: Final states at the nominal amplitude, dimensionless.
        states_b: Final states at the perturbed amplitude, dimensionless.
        delta: Finite-difference amplitude step, dimensionless.

    Returns:
        float: Fidelity-based QFI, inverse squared amplitude units.
    """
    if delta <= 0.0:
        raise ValueError("delta must be positive")
    return float(8.0 * (1.0 - math.sqrt(_squared_fidelity(states_a[-1], states_b[-1]))) / delta**2)


def _physicality_summary(states: list[qt.Qobj], floor: float) -> dict[str, Any]:
    """Summarize trace, Hermiticity, and minimum eigenvalue diagnostics.

    Args:
        states: State trajectory of density operators.
        floor: Numerical eigenvalue acceptance floor, dimensionless.

    Returns:
        dict[str, Any]: Numerical acceptance diagnostics; not exact positivity.
    """
    traces = []
    hermiticity_errors = []
    minimum_eigenvalue = float("inf")
    for state in states:
        rho = _density_matrix(state)
        traces.append(float(np.real(rho.tr())))
        hermiticity_errors.append(float((rho - rho.dag()).norm()))
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
            and minimum_eigenvalue >= floor
        ),
        "note": "Numerical acceptance is not a proof of exact positivity.",
    }


def _matsubara_terms(config: HeomExtendedConfig, temperature: float) -> int:
    """Choose the declared Matsubara term count for a temperature.

    Args:
        config: Centralized HEOM configuration.
        temperature: Normalized bath temperature, dimensionless.

    Returns:
        int: Declared Matsubara term count, dimensionless.
    """
    del config
    if temperature < 0.01:
        return 2
    if temperature < 0.3:
        return 4
    return 6


def _simulate_row(
    parameters: dict[str, Any],
    config: HeomExtendedConfig,
) -> dict[str, Any]:
    """Simulate one HEOM and Lindblad row and compare their states.

    Args:
        parameters: Row parameters (family scalars).
        config: Centralized HEOM configuration.

    Returns:
        dict[str, Any]: JSON-ready fidelity/QFI agreement metrics.
    """
    lam = float(parameters["lam"])
    gamma = float(parameters["gamma"])
    temperature = float(parameters["temperature"])
    depth = int(parameters["depth"])
    final_time = float(parameters["t_final"])
    driven = bool(parameters.get("driven", False))
    eta_d = float(parameters.get("eta_d", 0.0))
    tlist = np.linspace(0.0, final_time, config.n_time_points)
    nk = _matsubara_terms(config, temperature)
    initial = qt.basis(2, 1).proj()
    if not driven:
        heom_hamiltonian = 0.5 * config.wq * qt.sigmaz()
    else:
        gd = eta_d * config.driven_ratio

        def coefficient(time_normalized: float, _args: dict | None = None) -> float:
            """Return the instantaneous driving coefficient.

            Args:
                time_normalized: Time in inverse wq units.
                _args: Optional QuTiP coefficient arguments, unused.

            Returns:
                float: Dimensionless driving amplitude.
            """
            return eta_d + gd * math.cos(config.wd * time_normalized)

        heom_hamiltonian = qt.QobjEvo(
            [0.5 * config.wq * qt.sigmaz(), [qt.sigmax(), coefficient]]
        )
    bath = DrudeLorentzBath(qt.sigmax(), lam=lam, gamma=gamma, T=temperature, Nk=nk)
    heom_solver = HEOMSolver(
        heom_hamiltonian,
        bath,
        max_depth=depth,
        options={
            "atol": config.solver_atol,
            "rtol": config.solver_rtol,
            "progress_bar": False,
        },
    )
    heom_result = heom_solver.run(initial, tlist)
    lindblad_hamiltonian, collapse = _lindblad_comparator(
        config, lam, gamma, temperature, driven, eta_d
    )
    lindblad_result = qt.mesolve(
        lindblad_hamiltonian,
        initial,
        tlist,
        c_ops=collapse,
        e_ops=[],
        options={
            "atol": config.solver_atol,
            "rtol": config.solver_rtol,
            "method": "adams",
            "store_states": True,
            "progress_bar": "",
        },
    )
    errors = []
    for heom_state, lindblad_state in zip(heom_result.states, lindblad_result.states):
        errors.append(1.0 - _squared_fidelity(heom_state, lindblad_state))
    errors = np.asarray(errors[1:], dtype=float)  # omit the identical initial state
    heom_physicality = _physicality_summary(
        heom_result.states, config.numerical_eigenvalue_floor
    )
    lindblad_physicality = _physicality_summary(
        lindblad_result.states, config.numerical_eigenvalue_floor
    )
    valid_for_agreement = bool(
        heom_physicality["numerical_acceptance"]
        and lindblad_physicality["numerical_acceptance"]
        and np.all(np.isfinite(errors))
    )
    row_result: dict[str, Any] = {
        "mean_fidelity_error": float(np.mean(errors)) if len(errors) else None,
        "max_fidelity_error": float(np.max(errors)) if len(errors) else None,
        "final_fidelity": (
            float(np.clip(1.0 - float(errors[-1]), 0.0, 1.0)) if len(errors) else None
        ),
        "hierarchy_depth": depth,
        "matsubara_terms": nk,
        "valid_for_agreement": valid_for_agreement,
        "physicality_heom": heom_physicality,
        "physicality_lindblad": lindblad_physicality,
    }
    if not valid_for_agreement:
        row_result["status"] = "inconclusive_solver_positivity_gate"
        row_result["status_note"] = (
            "The stored HEOM trajectory violates the numerical positivity gate; "
            "this row is preserved but excluded from agreement claims."
        )
    else:
        row_result["status"] = "physical"
    if driven:
        # QFI agreement for the driven rows uses paired HEOM and paired
        # Lindblad propagations at eta_d and eta_d + delta_eta.
        delta = config.delta_eta
        gd_perturbed = (eta_d + delta) * config.driven_ratio

        def coefficient_perturbed(time_normalized: float, _args: dict | None = None) -> float:
            """Return the perturbed driving coefficient.

            Args:
                time_normalized: Time in inverse wq units.
                _args: Optional QuTiP coefficient arguments, unused.

            Returns:
                float: Dimensionless driving amplitude.
            """
            return (eta_d + delta) + gd_perturbed * math.cos(config.wd * time_normalized)

        perturbed_hamiltonian = qt.QobjEvo(
            [0.5 * config.wq * qt.sigmaz(), [qt.sigmax(), coefficient_perturbed]]
        )
        heom_perturbed_solver = HEOMSolver(
            perturbed_hamiltonian,
            bath,
            max_depth=depth,
            options={
                "atol": config.solver_atol,
                "rtol": config.solver_rtol,
                "progress_bar": False,
            },
        )
        heom_perturbed = heom_perturbed_solver.run(initial, tlist)
        lindblad_hamiltonian_perturbed, collapse_perturbed = _lindblad_comparator(
            config, lam, gamma, temperature, driven, eta_d + delta
        )
        lindblad_perturbed = qt.mesolve(
            lindblad_hamiltonian_perturbed,
            initial,
            tlist,
            c_ops=collapse_perturbed,
            e_ops=[],
            options={
                "atol": config.solver_atol,
                "rtol": config.solver_rtol,
                "method": "adams",
                "store_states": True,
                "progress_bar": "",
            },
        )
        heom_qfi = _fidelity_qfi(heom_result.states, heom_perturbed.states, delta)
        lindblad_qfi = _fidelity_qfi(
            lindblad_result.states, lindblad_perturbed.states, delta
        )
        row_result["heom_qfi"] = float(heom_qfi)
        row_result["lindblad_qfi"] = float(lindblad_qfi)
        row_result["qfi_relative_difference"] = float(
            abs(heom_qfi - lindblad_qfi) / max(abs(lindblad_qfi), 1.0e-30)
        )
        if not valid_for_agreement:
            row_result["heom_qfi"] = None
            row_result["lindblad_qfi"] = None
            row_result["qfi_relative_difference"] = None
    return row_result


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


def _parameter_rows(config: HeomExtendedConfig) -> list[dict[str, Any]]:
    """Enumerate the deterministic HEOM validation rows.

    Args:
        config: Centralized HEOM configuration.

    Returns:
        list[dict[str, Any]]: Ordered parameter rows.
    """
    rows: list[dict[str, Any]] = []
    for depth in config.hierarchy_depths:
        rows.append(
            {
                "family": "hierarchy_depth",
                "depth": depth,
                "lam": config.baseline_lambda,
                "gamma": config.baseline_gamma,
                "temperature": config.baseline_temperature,
                "t_final": 1.0,
            }
        )
    for temperature in config.temperatures:
        rows.append(
            {
                "family": "temperature",
                "depth": config.baseline_depth,
                "lam": config.baseline_lambda,
                "gamma": config.baseline_gamma,
                "temperature": temperature,
                "t_final": 2.0,
            }
        )
    for lam in config.couplings_lam:
        rows.append(
            {
                "family": "coupling",
                "depth": config.baseline_depth,
                "lam": lam,
                "gamma": config.baseline_gamma,
                "temperature": config.baseline_temperature,
                "t_final": 2.0,
            }
        )
    for gamma in config.cutoffs_gamma:
        rows.append(
            {
                "family": "cutoff",
                "depth": config.baseline_depth,
                "lam": config.baseline_lambda,
                "gamma": gamma,
                "temperature": config.baseline_temperature,
                "t_final": 2.0,
            }
        )
    for final_time in config.final_times:
        rows.append(
            {
                "family": "time_window",
                "depth": config.baseline_depth,
                "lam": config.baseline_lambda,
                "gamma": config.baseline_gamma,
                "temperature": config.baseline_temperature,
                "t_final": final_time,
            }
        )
    for eta_d in config.driven_eta_values:
        rows.append(
            {
                "family": "driven_dce_ratio",
                "depth": config.baseline_depth,
                "lam": config.baseline_lambda,
                "gamma": config.baseline_gamma,
                "temperature": config.baseline_temperature,
                "t_final": 2.0,
                "driven": True,
                "eta_d": eta_d,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the HEOM producer.

    Returns:
        argparse.Namespace: Output and checkpoint options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "heom_extended_validation.json",
        help="Merged JSON output path.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Per-row checkpoint directory; defaults to data/_cache/heom_extended_validation.",
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
    return parser.parse_args()


def _hierarchy_convergence(
    rows: list[dict[str, Any]], config: HeomExtendedConfig
) -> dict[str, Any]:
    """Summarize hierarchy-depth convergence from the depth rows.

    Args:
        rows: All simulated rows.
        config: Centralized HEOM configuration.

    Returns:
        dict[str, Any]: Per-depth metrics and convergence classification.
    """
    depth_rows = [
        row
        for row in rows
        if row["parameters"]["family"] == "hierarchy_depth"
        and row.get("valid_for_agreement")
    ]
    depth_rows.sort(key=lambda item: item["parameters"]["depth"])
    entries = []
    for row in depth_rows:
        depth = int(row["parameters"]["depth"])
        entries.append(
            {
                "depth": depth,
                "mean_fidelity_error": row["mean_fidelity_error"],
                "max_fidelity_error": row["max_fidelity_error"],
            }
        )
    converged_depth = None
    for index in range(len(entries) - 1):
        current = entries[index]
        following = entries[index + 1]
        change = abs(following["mean_fidelity_error"] - current["mean_fidelity_error"]) / max(
            current["mean_fidelity_error"], 1.0e-12
        )
        if change < config.hierarchy_convergence_tolerance:
            converged_depth = current["depth"]
            break
    return {
        "entries": entries,
        "convergence_tolerance": config.hierarchy_convergence_tolerance,
        "converged_depth_or_none": converged_depth,
        "note": (
            "Depth convergence uses the declared relative change of the mean fidelity "
            "error between neighboring depths over physical rows only; it is specific "
            "to the tested rows."
        ),
    }


def main() -> int:
    """Run and merge the resumable HEOM validation screen.

    Returns:
        int: Zero after successful checkpoint merge and JSON serialization.
    """
    args = parse_args()
    config = CONFIG
    checkpoint_dir = args.checkpoint_dir or args.output.parent / "_cache" / "heom_extended_validation"
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
            print(f"[{index}/{len(parameters_list)}] simulating {row_id} {parameters['family']}")
            metrics = _simulate_row(parameters, config)
            row_result = {"row_id": row_id, "parameters": parameters, **metrics}
            checkpoint.write_text(json.dumps(row_result, indent=2) + "\n", encoding="utf-8")
        results.append(row_result)
    results.sort(key=lambda item: item["row_id"])
    hierarchy = _hierarchy_convergence(results, config)
    # Physical-temperature equivalence at the canonical 6 GHz cavity with the
    # qubit transition at 1.027*wc, using the centralized hbar/k_B constant.
    cavity_reference_GHz = 6.0
    physical_equivalences = [
        {
            "normalized_temperature": value,
            "cavity_frequency_reference_GHz": cavity_reference_GHz,
            "qubit_frequency_reference_GHz": config.wq * cavity_reference_GHz,
            "equivalent_temperature_mK": float(
                value
                * config.wq
                * 2.0
                * math.pi
                * cavity_reference_GHz
                * 1.0e9
                * HBAR_OVER_KB_S_K
                * 1.0e3
            ),
        }
        for value in config.temperatures
    ]
    valid_count = sum(1 for row in results if row.get("valid_for_agreement"))
    inconclusive_count = len(results) - valid_count
    payload = {
        "schema": "trackB_heom_extended_validation_v1",
        "status": "local_spin_boson_hierarchy_screen_partial_inconclusive",
        "purpose": (
            "Compare the Markovian Lindblad working model with quasi-exact HEOM for "
            "declared Drude-Lorentz spin-boson rows and report hierarchy convergence "
            "where the stored hierarchy passes the numerical positivity gate."
        ),
        "configuration": asdict(config),
        "units": {
            "wq": "two-level transition angular frequency (1.027) in bath units",
            "wd": "drive angular frequency (2.077) in the same units",
            "lam": "Drude reorganization coupling in wq units",
            "gamma": "bath cutoff frequency in wq units",
            "temperature": "k_B T / (hbar wq), dimensionless",
            "t_final": "inverse wq angular-frequency units",
            "fidelity_error": "dimensionless (1 - squared Uhlmann fidelity)",
            "qfi": "inverse squared driving-amplitude units",
        },
        "lindblad_comparator": {
            "construction": (
                "Markovian weak-coupling limit of the same Drude-Lorentz bath with "
                "Gamma_down = 2*pi*J(wq)(1+n) and Gamma_up = 2*pi*J(wq)*n, "
                "J(w) = 2*lam*gamma*w/(w^2+gamma^2), n = 1/(exp(w/T)-1)."
            ),
            "convention_note": (
                "The 2*pi*J(omega_q) prefactor matches the stored local baseline "
                "artifact trackB_heom_validation.json (Gamma = 0.012535 at lam=0.02, "
                "gamma=20, wq=1)."
            ),
        },
        "physical_temperature_equivalences": physical_equivalences,
        "solver": {
            "heom": "qutip.solver.heom.HEOMSolver with DrudeLorentzBath (Matsubara)",
            "lindblad": "qutip.mesolve",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "qutip": qt.__version__,
        },
        "validation": {
            "rows_requested": len(parameters_list),
            "rows_completed": len(results),
            "all_rows_complete": len(parameters_list) == len(results),
            "valid_for_agreement_count": valid_count,
            "inconclusive_positivity_gate_count": inconclusive_count,
            "physical_family_counts": {
                family: sum(
                    1
                    for row in results
                    if row["parameters"]["family"] == family
                    and row.get("valid_for_agreement")
                )
                for family in sorted({row["parameters"]["family"] for row in results})
            },
            "hierarchy_depth_summary": hierarchy,
            "numerical_physicality_note": "Trace/Hermiticity/eigenvalue acceptance is not a proof of exact positivity.",
            "positivity_gate_note": (
                "Rows whose stored HEOM trajectory violates the numerical eigenvalue gate "
                "are preserved as inconclusive and excluded from all agreement claims."
            ),
            "scope_note": (
                "Rows are local spin-boson models with the canonical frequency ratios, "
                "including driven rows at the DCE ratio; they are not full USC cavity-bath "
                "simulations and do not validate the collective Dicke protocol."
            ),
        },
        "caveats": [
            "A short-time fidelity agreement does not validate a different driven protocol or a different bath.",
            "Hierarchy convergence is reported for the tested rows, parameters, and Matsubara counts only.",
            "The QuTiP 5.3.1 HEOM hierarchy for sigma-x Drude-Lorentz energy relaxation became nonphysical (numerical eigenvalue gate) outside a narrow temperature/coupling band in this environment; those rows are inconclusive and preserved rather than hidden.",
            "No driven row passed the positivity gate in this Tier-1 screen, so no HEOM QFI agreement row is claimed.",
            "The local model omits the cavity mode; driven rows use amplitude driving at the DCE frequency ratio as the tractable local analog.",
            "No generalization to arbitrary USC baths or the full DCE parameter space is claimed.",
            "The working Lindblad model is supported only for rows that pass the declared agreement gates.",
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
