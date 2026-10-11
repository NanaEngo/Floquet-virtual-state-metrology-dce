#!/usr/bin/env python3
"""Run a true cavity-mode structured-bath HEOM/Lindblad benchmark.

Unlike the surrogate cavity_heom_validation.py (which uses a two-level
sigma-x coupling), this script implements a genuine cavity Fock-space
system coupled through the position quadrature (a + a†) to a Drude-Lorentz
bath, with the Markovian Lindblad comparator using sqrt(kappa)*a collapse
operators.

This validates the Markovian cavity-decay assumption used throughout
Track B: is the standard input-output-theory treatment of cavity loss
accurate at the relevant kappa/omega_c values?
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
from qutip.solver.heom import DrudeLorentzBath, HEOMSolver

try:
    from config import HBAR_OVER_KB_S_K
    from trackb_recompute_utils import assert_finite_json, atomic_write_json, failure_row, stable_row_id
except ImportError as exc:  # pragma: no cover
    raise ImportError("Run this producer from the Track B scripts directory.") from exc


SCHEMA = "trackB_cavity_heom_validation_v2"

# --- Physical parameters (all in cavity-frequency units, omega_c = 1) ---

CAVITY_FREQUENCY = 1.0  # omega_c, normalized
PHYSICAL_KAPPA_RATIO = 4.1e-2  # kappa / omega_c at the CPW operating point
REFERENCE_KAPPA_RATIO = 1.0e-4  # low-loss reference

# Drude-Lorentz: J(omega) = 2*lam*gamma*omega / (omega^2 + gamma^2)
# Markovian limit: kappa = 2*pi*J(omega_c)
# Given target kappa and chosen gamma, solve for lam:
#   lam = kappa * (omega_c^2 + gamma^2) / (4*pi*gamma*omega_c)
DRUDE_CUTOFFS = (5.0, 10.0, 20.0, 40.0)  # gamma / omega_c
HIERARCHY_DEPTHS = (2, 3, 4)
TEMPERATURES = (0.01, 0.05)  # k_B T / (hbar omega_c)
KAPPA_TARGETS = (REFERENCE_KAPPA_RATIO, PHYSICAL_KAPPA_RATIO)
CAVITY_TRUNCATION = 12  # Fock space dimension
FINAL_TIMES = (1.0, 2.0)  # in 1/omega_c units
TIME_POINTS = 81
MATSUBARA_TERMS = 2
EIGENVALUE_FLOOR = -1.0e-6
SOLVER_ATOL = 1.0e-10
SOLVER_RTOL = 1.0e-8


def _drude_lam_for_kappa(kappa: float, gamma: float) -> float:
    """Compute Drude reorganization coupling to yield target kappa in Markovian limit.

    Args:
        kappa: Target kappa/omega_c, dimensionless.
        gamma: Drude cutoff gamma/omega_c, dimensionless.

    Returns:
        float: lambda in cavity-frequency units.
    """
    omega_c = CAVITY_FREQUENCY
    return float(kappa * (omega_c**2 + gamma**2) / (4.0 * math.pi * gamma * omega_c))


def _bose_occupation(omega: float, temperature: float) -> float:
    """Return the normalized Bose-Einstein occupation n = 1/(exp(omega/T) - 1)."""
    if temperature <= 0.0 or omega <= 0.0:
        return 0.0
    exponent = omega / temperature
    if exponent > 700.0:
        return 0.0
    return float(1.0 / (math.exp(exponent) - 1.0))


def _matsubara_count(temperature: float) -> int:
    """Choose Matsubara term count based on temperature."""
    if temperature < 0.01:
        return 2
    if temperature < 0.3:
        return 4
    return 6


def parameter_rows() -> list[dict[str, Any]]:
    """Enumerate the deterministic cavity-HEOM benchmark grid."""
    rows: list[dict[str, Any]] = []
    for kappa in KAPPA_TARGETS:
        for gamma in DRUDE_CUTOFFS:
            for depth in HIERARCHY_DEPTHS:
                for temperature in TEMPERATURES:
                    for t_final in FINAL_TIMES:
                        lam = _drude_lam_for_kappa(kappa, gamma)
                        rows.append({
                            "kappa_target": float(kappa),
                            "depth": int(depth),
                            "lam": float(lam),
                            "gamma": float(gamma),
                            "temperature": float(temperature),
                            "t_final": float(t_final),
                        })
    return rows


def _build_cavity_hamiltonian(driven: bool = False) -> qt.Qobj:
    """Build the bare cavity Hamiltonian H = omega_c * a†a."""
    a = qt.destroy(CAVITY_TRUNCATION)
    return CAVITY_FREQUENCY * a.dag() * a


def _simulate_row(parameters: dict[str, Any]) -> dict[str, Any]:
    """Simulate one cavity-HEOM row and compare with Markovian Lindblad.

    Args:
        parameters: Row parameters (kappa_target, depth, lam, gamma, temperature, t_final).

    Returns:
        dict[str, Any]: Fidelity agreement metrics and diagnostics.
    """
    kappa_target = float(parameters["kappa_target"])
    depth = int(parameters["depth"])
    lam = float(parameters["lam"])
    gamma = float(parameters["gamma"])
    temperature = float(parameters["temperature"])
    t_final = float(parameters["t_final"])

    a = qt.destroy(CAVITY_TRUNCATION)
    position = a + a.dag()  # position quadrature for bath coupling
    hamiltonian = _build_cavity_hamiltonian()
    tlist = np.linspace(0.0, t_final, TIME_POINTS)
    nk = _matsubara_count(temperature)

    # Initial state: coherent state |alpha=1> (displaced vacuum, nontrivial dynamics)
    alpha = 1.0
    initial = qt.coherent(CAVITY_TRUNCATION, alpha)
    rho0 = initial.proj()

    # --- HEOM propagation ---
    bath = DrudeLorentzBath(position, lam=lam, gamma=gamma, T=temperature, Nk=nk)
    heom_solver = HEOMSolver(
        hamiltonian,
        bath,
        max_depth=depth,
        options={
            "atol": SOLVER_ATOL,
            "rtol": SOLVER_RTOL,
            "progress_bar": False,
        },
    )
    heom_result = heom_solver.run(rho0, tlist)

    # --- Lindblad propagation (Markovian limit of the same bath) ---
    # kappa_markov = 2*pi*J(omega_c) where J is the Drude-Lorentz spectral density
    j_at_omega_c = 2.0 * lam * gamma * CAVITY_FREQUENCY / (
        CAVITY_FREQUENCY**2 + gamma**2
    )
    kappa_markov = 2.0 * math.pi * j_at_omega_c
    n_thermal = _bose_occupation(CAVITY_FREQUENCY, temperature)
    # Cavity decay: sqrt(kappa*(n+1)) a + sqrt(kappa*n) a†
    collapse = [
        math.sqrt(kappa_markov * (n_thermal + 1.0)) * a,
        math.sqrt(kappa_markov * n_thermal) * a.dag(),
    ]
    lindblad_result = qt.mesolve(
        hamiltonian,
        rho0,
        tlist,
        c_ops=collapse,
        e_ops=[],
        options={
            "atol": SOLVER_ATOL,
            "rtol": SOLVER_RTOL,
            "method": "adams",
            "store_states": True,
            "progress_bar": "",
        },
    )

    # --- Fidelity comparison ---
    def _squared_fidelity(rho_a: qt.Qobj, rho_b: qt.Qobj) -> float:
        root_a = rho_a.sqrtm()
        product = root_a * rho_b * root_a
        eigenvalues = np.real_if_close(product.eigenenergies()).real
        root_fidelity = float(np.sum(np.sqrt(np.clip(eigenvalues, 0.0, None))))
        return float(np.clip(root_fidelity**2, 0.0, 1.0))

    def _as_density_matrix(state: qt.Qobj) -> qt.Qobj:
        rho = state.proj() if state.isket else state
        tr = float(np.real(rho.tr()))
        return rho / tr if tr > 0.0 else rho

    errors = []
    for heom_state, lindblad_state in zip(heom_result.states, lindblad_result.states):
        rho_h = _as_density_matrix(heom_state)
        rho_l = _as_density_matrix(lindblad_state)
        errors.append(1.0 - _squared_fidelity(rho_h, rho_l))
    errors_arr = np.asarray(errors[1:], dtype=float)  # skip identical initial state

    # --- Physicality diagnostics ---
    def _physicality(states: list[qt.Qobj]) -> dict[str, Any]:
        min_eig = float("inf")
        max_trace_err = 0.0
        max_herm_err = 0.0
        for state in states:
            rho = _as_density_matrix(state)
            max_trace_err = max(max_trace_err, abs(float(np.real(rho.tr())) - 1.0))
            max_herm_err = max(max_herm_err, float((rho - rho.dag()).norm()))
            eigvals = np.real_if_close(rho.eigenenergies()).real
            min_eig = min(min_eig, float(np.min(eigvals)))
        accepted = (
            max_trace_err < 1.0e-8
            and max_herm_err < 1.0e-8
            and min_eig >= EIGENVALUE_FLOOR
        )
        return {
            "max_trace_error": float(max_trace_err),
            "max_hermiticity_error": float(max_herm_err),
            "minimum_eigenvalue": float(min_eig),
            "numerical_acceptance": bool(accepted),
            "note": "Numerical acceptance is not a proof of exact positivity.",
        }

    heom_phys = _physicality(heom_result.states)
    lindblad_phys = _physicality(lindblad_result.states)
    valid = bool(
        heom_phys["numerical_acceptance"]
        and lindblad_phys["numerical_acceptance"]
        and np.all(np.isfinite(errors_arr))
    )

    # Compare Markovian kappa to target kappa
    kappa_ratio_error = abs(kappa_markov - kappa_target) / max(kappa_target, 1.0e-30)

    return {
        "kappa_target": float(kappa_target),
        "kappa_markov": float(kappa_markov),
        "kappa_ratio_error": float(kappa_ratio_error),
        "mean_fidelity_error": float(np.mean(errors_arr)) if len(errors_arr) else None,
        "max_fidelity_error": float(np.max(errors_arr)) if len(errors_arr) else None,
        "final_fidelity": (
            float(np.clip(1.0 - float(errors_arr[-1]), 0.0, 1.0)) if len(errors_arr) else None
        ),
        "hierarchy_depth": depth,
        "matsubara_terms": nk,
        "valid_for_agreement": valid,
        "physicality_heom": heom_phys,
        "physicality_lindblad": lindblad_phys,
        "status": "physical" if valid else "inconclusive_solver_positivity_gate",
    }


def _kappa_convergence_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize hierarchy-depth convergence for each kappa target."""
    summary: dict[str, Any] = {}
    for kappa in KAPPA_TARGETS:
        kappa_rows = [
            r for r in rows
            if abs(r["kappa_target"] - kappa) < 1.0e-12 and r.get("valid_for_agreement")
        ]
        kappa_rows.sort(key=lambda r: r["hierarchy_depth"])
        entries = [
            {
                "depth": r["hierarchy_depth"],
                "mean_fidelity_error": r["mean_fidelity_error"],
                "max_fidelity_error": r["max_fidelity_error"],
            }
            for r in kappa_rows
        ]
        converged_depth = None
        for i in range(len(entries) - 1):
            curr = entries[i]["mean_fidelity_error"]
            nxt = entries[i + 1]["mean_fidelity_error"]
            if curr is not None and nxt is not None:
                change = abs(nxt - curr) / max(abs(curr), 1.0e-12)
                if change < 0.05:
                    converged_depth = entries[i]["depth"]
                    break
        summary[str(kappa)] = {
            "entries": entries,
            "converged_depth_or_none": converged_depth,
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "cavity_heom_validation_v2.json",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--row-index", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir or args.output.parent / "_cache" / "cavity_heom_validation_v2"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    all_params = parameter_rows()
    if args.row_index is not None:
        if not 0 <= args.row_index < len(all_params):
            raise ValueError(f"row index {args.row_index} outside grid [0, {len(all_params)})")
        selected = [all_params[args.row_index]]
    elif args.limit is not None:
        selected = all_params[:args.limit]
    else:
        selected = all_params

    results: list[dict[str, Any]] = []
    for idx, params in enumerate(selected, 1):
        row_id = stable_row_id(params)
        path = checkpoint_dir / f"{row_id}.json"
        if path.is_file() and not args.no_resume:
            row = json.loads(path.read_text(encoding="utf-8"))
            print(f"[{idx}/{len(selected)}] resumed {row_id}")
        else:
            print(f"[{idx}/{len(selected)}] simulating kappa={params['kappa_target']:.1e} depth={params['depth']}")
            try:
                metrics = _simulate_row(params)
                row = {
                    "schema": SCHEMA,
                    "row_id": row_id,
                    "parameters": params,
                    **metrics,
                }
            except Exception as exc:
                row = failure_row(row_id, params, exc)
                row["schema"] = SCHEMA
            assert_finite_json(row)
            atomic_write_json(path, row)
        results.append(row)

    results.sort(key=lambda r: r["row_id"])
    kappa_summary = _kappa_convergence_summary(results)
    valid_count = sum(1 for r in results if r.get("valid_for_agreement"))

    payload = {
        "schema": SCHEMA,
        "status": (
            "cavity_mode_heom_screen_complete"
            if len(results) == len(all_params)
            else "cavity_mode_heom_screen_partial"
        ),
        "purpose": (
            "Validate the Markovian cavity-decay assumption by comparing HEOM "
            "(cavity Fock space, position-quadrature Drude-Lorentz bath) against "
            "Lindblad (sqrt(kappa)*a collapse operators) at target kappa/omega_c values."
        ),
        "scope": {
            "system": f"Fock space, dim={CAVITY_TRUNCATION}",
            "bath": "Drude-Lorentz structured bath coupled through position quadrature (a + a†)",
            "cavity_mode_included": True,
            "full_cavity_heom_claim": True,
            "relationship_to_track_b": (
                "Validates the Markovian cavity-loss assumption used for all "
                "headline retention numbers in the manuscript"
            ),
        },
        "configuration": {
            "cavity_frequency": CAVITY_FREQUENCY,
            "cavity_truncation": CAVITY_TRUNCATION,
            "kappa_targets": list(KAPPA_TARGETS),
            "drude_cutoffs": list(DRUDE_CUTOFFS),
            "hierarchy_depths": list(HIERARCHY_DEPTHS),
            "temperatures": list(TEMPERATURES),
            "final_times": list(FINAL_TIMES),
            "time_points": TIME_POINTS,
            "matsubara_terms_base": MATSUBARA_TERMS,
            "eigenvalue_floor": EIGENVALUE_FLOOR,
        },
        "units": {
            "omega_cavity": "cavity angular frequency (1.0, normalized)",
            "kappa": "kappa/omega_c, dimensionless",
            "lam": "Drude reorganization coupling in omega_c units",
            "gamma": "Drude cutoff in omega_c units",
            "temperature": "k_B T / (hbar omega_c), dimensionless",
            "t_final": "inverse omega_c units",
            "fidelity_error": "1 - squared Uhlmann fidelity, dimensionless",
        },
        "markovian_comparator": {
            "construction": (
                "kappa_markov = 2*pi*J(omega_c) with J(w) = 2*lam*gamma*w/(w^2+gamma^2); "
                "collapse operators: sqrt(kappa*(n+1))*a, sqrt(kappa*n)*a†"
            ),
            "note": (
                "The Markovian kappa is derived from the same Drude-Lorentz parameters "
                "used for the HEOM bath; kappa_ratio_error reports the self-consistency."
            ),
        },
        "physical_temperature_equivalences": [
            {
                "normalized_temperature": T,
                "cavity_frequency_GHz": 6.0,
                "equivalent_temperature_mK": float(
                    T * 2.0 * math.pi * 6.0e9 * HBAR_OVER_KB_S_K * 1.0e3
                ),
            }
            for T in TEMPERATURES
        ],
        "solver": {
            "heom": "qutip.solver.heom.HEOMSolver with DrudeLorentzBath (Matsubara)",
            "lindblad": "qutip.mesolve (adams)",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "qutip": qt.__version__,
        },
        "validation": {
            "rows_requested": len(selected),
            "rows_in_grid": len(all_params),
            "rows_completed": len(results),
            "valid_for_agreement_count": valid_count,
            "inconclusive_count": len(results) - valid_count,
            "kappa_convergence_summary": kappa_summary,
            "numerical_physicality_note": "Trace/Hermiticity/eigenvalue acceptance is not a proof of exact positivity.",
            "positivity_gate_note": (
                "Rows whose HEOM trajectory violates the numerical eigenvalue gate "
                "are preserved as inconclusive and excluded from agreement claims."
            ),
        },
        "caveats": [
            "This validates the Markovian cavity-decay assumption, not the full Dicke+DCE protocol.",
            "The Drude-Lorentz spectral density is a model for the transmission-line continuum; real cavities may have different bath structures.",
            "Agreement is conditional on the tested truncation, hierarchy depth, temperature, and time window.",
            "Inconclusive rows are retained, not hidden.",
        ],
        "rows": results,
        "determinism": "Payload contains no wall-clock values or random state.",
    }
    assert_finite_json(payload)
    atomic_write_json(args.output, payload)
    print(f"Saved {args.output} ({len(results)}/{len(all_params)} rows, {valid_count} valid)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
