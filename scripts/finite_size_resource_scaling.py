#!/usr/bin/env python3
"""Compute larger-N finite-size scaling and resource accounting for Track B.

The producer evaluates the exact static Dicke ground-state QFI across a
declared emitter-count grid with two cavity truncations per size bracket,
and runs canonical Dicke+DCE driven rows (with and without cavity loss) for
resource accounting.  Scaling fits are reported as descriptive fits with
uncertainty intervals over declared ranges; no thermodynamic exponent, phase
transition, Heisenberg violation, or global bound is claimed.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import qutip as qt

from trackb_qutip_utils import density_matrix, squared_fidelity_pure

try:
    from scipy.optimize import curve_fit
except ImportError as exc:  # pragma: no cover - environment-specific
    raise ImportError("scipy is required for the descriptive scaling fits.") from exc

try:
    from config import DEFAULT_FINITE_SIZE, FiniteSizeConfig
except ImportError as exc:  # pragma: no cover - environment-specific
    raise ImportError(
        "Track B configuration is unavailable; run this script from scripts/."
    ) from exc


CONFIG = DEFAULT_FINITE_SIZE


def _spin_dimension(n_qubits: int) -> int:
    """Return the collective spin Hilbert-space dimension.

    Args:
        n_qubits: Number of emitters N, dimensionless.

    Returns:
        int: Spin dimension N+1, dimensionless.
    """
    return int(2 * (n_qubits / 2.0) + 1)


def _dicke_hamiltonian(
    n_qubits: int, eta: float, wq_wc: float, n_max: int
) -> qt.Qobj:
    """Construct the Dicke Hamiltonian in the cavity x collective-spin basis.

    Args:
        n_qubits: Number of emitters N, dimensionless.
        eta: Static coupling g0/wc, dimensionless.
        wq_wc: Qubit-to-cavity angular-frequency ratio, dimensionless.
        n_max: Cavity Fock-space dimension, dimensionless.

    Returns:
        qt.Qobj: Hamiltonian in normalized angular-frequency units.
    """
    spin_dim = _spin_dimension(n_qubits)
    a = qt.destroy(n_max)
    jx = qt.jmat(n_qubits / 2.0, "x")
    jz = qt.jmat(n_qubits / 2.0, "z")
    return (
        qt.tensor(a.dag() * a, qt.qeye(spin_dim))
        + wq_wc * qt.tensor(qt.qeye(n_max), jz)
        + (eta / math.sqrt(n_qubits)) * qt.tensor(a + a.dag(), jx)
    )


def _dicke_generator(n_qubits: int, n_max: int) -> qt.Qobj:
    """Return the normalized coupling generator V=(a+a^dagger)Jx/sqrt(N).

    Args:
        n_qubits: Number of emitters N, dimensionless.
        n_max: Cavity Fock-space dimension, dimensionless.

    Returns:
        qt.Qobj: Generator in normalized angular-frequency units.
    """
    spin_dim = _spin_dimension(n_qubits)
    a = qt.destroy(n_max)
    jx = qt.jmat(n_qubits / 2.0, "x")
    return qt.tensor(a + a.dag(), jx) / math.sqrt(n_qubits)


def _dicke_ground_qfi(
    n_qubits: int, eta: float, wq_wc: float, n_max: int, n_eig: int
) -> float:
    """Compute the ground-state gap-weighted static QFI for eta.

    Args:
        n_qubits: Number of emitters N, dimensionless.
        eta: Static coupling g0/wc, dimensionless.
        wq_wc: Qubit-to-cavity angular-frequency ratio, dimensionless.
        n_max: Cavity Fock-space dimension, dimensionless.
        n_eig: Number of eigenstates retained in the sum, dimensionless.

    Returns:
        float: Static ground-state QFI for eta, dimensionless.
    """
    hamiltonian = _dicke_hamiltonian(n_qubits, eta, wq_wc, n_max)
    evals, evecs = hamiltonian.eigenstates(eigvals=n_eig)
    generator = _dicke_generator(n_qubits, n_max)
    ground = evecs[0]
    ground_energy = evals[0]
    qfi = 0.0
    for index in range(1, n_eig):
        gap = evals[index] - ground_energy
        if abs(gap) > 1.0e-10:
            matrix_element = abs(complex(ground.dag() * generator * evecs[index])) ** 2
            qfi += 4.0 * matrix_element / gap**2
    return float(qfi)


def _truncation_pair_for_n(n_qubits: int, config: FiniteSizeConfig) -> tuple[int, int]:
    """Select the declared cavity truncation pair for a size bracket.

    Args:
        n_qubits: Number of emitters N, dimensionless.
        config: Centralized finite-size configuration.

    Returns:
        tuple[int, int]: (low, high) cavity truncations, dimensionless.
    """
    for (low, high), (n_low, n_high) in zip(
        config.static_truncation_pairs, config.static_truncation_brackets
    ):
        if n_low <= n_qubits <= n_high:
            return low, high
    return config.static_truncation_pairs[-1]


def _simulate_static_row(
    parameters: dict[str, Any], config: FiniteSizeConfig
) -> dict[str, Any]:
    """Evaluate the static ground-state QFI at two cavity truncations.

    Args:
        parameters: Row parameters (N, wq_wc).
        config: Centralized finite-size configuration.

    Returns:
        dict[str, Any]: JSON-ready static metrics and truncation flag.
    """
    n_qubits = int(parameters["N"])
    wq_wc = float(parameters["wq_wc"])
    low_truncation, high_truncation = _truncation_pair_for_n(n_qubits, config)
    low_qfi = _dicke_ground_qfi(
        n_qubits, config.static_eta, wq_wc, low_truncation, config.static_n_eig
    )
    high_qfi = _dicke_ground_qfi(
        n_qubits, config.static_eta, wq_wc, high_truncation, config.static_n_eig
    )
    relative_change = abs(high_qfi - low_qfi) / max(abs(high_qfi), 1.0e-30)
    return {
        "row_type": "static",
        "N": n_qubits,
        "wq_wc": wq_wc,
        "eta": config.static_eta,
        "qfi_low_truncation": float(low_qfi),
        "qfi_high_truncation": float(high_qfi),
        "cavity_truncations": [low_truncation, high_truncation],
        "n_eig": config.static_n_eig,
        "truncation_relative_change": float(relative_change),
        "truncation_converged": bool(relative_change < 0.05),
    }


def _dicke_dce_hamiltonian(
    n_qubits: int,
    eta: float,
    gd_additive: float,
    wq_wc: float,
    wd_wc: float,
    n_cav: int,
) -> qt.QobjEvo:
    """Construct the canonical driven Dicke+DCE Hamiltonian.

    Args:
        n_qubits: Number of emitters N, dimensionless.
        eta: Static coupling g0/wc, dimensionless.
        gd_additive: Additive drive amplitude in wc units, dimensionless.
        wq_wc: Qubit-to-cavity angular-frequency ratio, dimensionless.
        wd_wc: Drive-to-cavity angular-frequency ratio, dimensionless.
        n_cav: Cavity Fock-space dimension, dimensionless.

    Returns:
        qt.QobjEvo: Time-dependent Hamiltonian in normalized angular-frequency
        units.
    """
    spin_dim = _spin_dimension(n_qubits)
    a = qt.destroy(n_cav)
    jx = qt.jmat(n_qubits / 2.0, "x")
    jz = qt.jmat(n_qubits / 2.0, "z")
    static = qt.tensor(a.dag() * a, qt.qeye(spin_dim)) + wq_wc * qt.tensor(
        qt.qeye(n_cav), jz
    )
    drive_operator = qt.tensor(a + a.dag(), jx) / math.sqrt(n_qubits)
    g0 = eta

    def coefficient(time_normalized: float, _args: dict | None = None) -> float:
        """Return the instantaneous drive coefficient.

        Args:
            time_normalized: Time in inverse wc units.
            _args: Optional QuTiP coefficient arguments, unused.

        Returns:
            float: Instantaneous dimensionless drive coefficient.
        """
        return g0 + gd_additive * math.cos(wd_wc * time_normalized)

    return qt.QobjEvo([static, [drive_operator, coefficient]])




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
        fidelity = squared_fidelity_pure(state_a, state_b)
        values.append(8.0 * (1.0 - math.sqrt(float(np.clip(fidelity, 0.0, 1.0)))) / delta_eta**2)
    return np.asarray(values, dtype=float)


def _physicality_summary(states: list[qt.Qobj], floor: float = -1.0e-6) -> dict[str, Any]:
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
        rho = density_matrix(state)
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


def _simulate_driven_row(
    parameters: dict[str, Any], config: FiniteSizeConfig
) -> dict[str, Any]:
    """Simulate one canonical Dicke+DCE driven row.

    Args:
        parameters: Row parameters (N, kappa_rel).
        config: Centralized finite-size configuration.

    Returns:
        dict[str, Any]: JSON-ready driven metrics and resource diagnostics.
    """
    n_qubits = int(parameters["N"])
    kappa_rel = float(parameters["kappa_rel"])
    n_cav = config.driven_n_cav
    spin_dim = _spin_dimension(n_qubits)
    eta = config.driven_eta
    gd_additive = eta * config.driven_gd_over_g0
    drive_period = 2.0 * math.pi / config.driven_wd_wc
    final_time = config.driven_periods * drive_period
    times = np.linspace(0.0, final_time, config.driven_time_points)
    hamiltonian = _dicke_dce_hamiltonian(
        n_qubits, eta, gd_additive, config.driven_wq_wc, config.driven_wd_wc, n_cav
    )
    perturbed = _dicke_dce_hamiltonian(
        n_qubits,
        eta + config.delta_eta,
        gd_additive,
        config.driven_wq_wc,
        config.driven_wd_wc,
        n_cav,
    )
    initial = qt.tensor(qt.basis(n_cav, 0), qt.basis(spin_dim, 0))
    collapse_ops = [math.sqrt(kappa_rel) * qt.tensor(qt.destroy(n_cav), qt.qeye(spin_dim))] if kappa_rel > 0.0 else []
    options = {
        "atol": config.solver_atol,
        "rtol": config.solver_rtol,
        "method": "adams",
        "max_step": 0.05 * drive_period,
        "store_states": True,
        "progress_bar": "",
    }
    if kappa_rel == 0.0:
        reference = qt.sesolve(hamiltonian, initial, times, e_ops=[], options=options)
        perturbed_result = qt.sesolve(perturbed, initial, times, e_ops=[], options=options)
        solver_name = "qutip.sesolve"
    else:
        reference = qt.mesolve(
            hamiltonian, initial, times, c_ops=collapse_ops, e_ops=[], options=options
        )
        perturbed_result = qt.mesolve(
            perturbed, initial, times, c_ops=collapse_ops, e_ops=[], options=options
        )
        solver_name = "qutip.mesolve"
    qfi = _qfi_trace(reference.states, perturbed_result.states, config.delta_eta)
    periods = times / drive_period
    a = qt.tensor(qt.destroy(n_cav), qt.qeye(spin_dim))
    photons = np.asarray(
        [float(np.real(qt.expect(a.dag() * a, state))) for state in reference.states],
        dtype=float,
    )
    qfi_peak_index = int(np.argmax(qfi))
    emitted_proxy = float(kappa_rel * np.trapz(photons, times))
    return {
        "row_type": "driven",
        "N": n_qubits,
        "kappa_rel": kappa_rel,
        "eta": eta,
        "gd_additive_rel_wc": float(gd_additive),
        "gd_over_g0": config.driven_gd_over_g0,
        "wq_wc": config.driven_wq_wc,
        "wd_wc": config.driven_wd_wc,
        "n_cav": n_cav,
        "qfi": qfi.tolist(),
        "photon_number": photons.tolist(),
        "time_drive_periods": periods.tolist(),
        "qfi_peak": float(qfi[qfi_peak_index]),
        "qfi_peak_period": float(periods[qfi_peak_index]),
        "qfi_final": float(qfi[-1]),
        "photon_number_peak": float(np.max(photons)),
        "photon_number_at_qfi_peak": float(photons[qfi_peak_index]),
        "photon_number_mean": float(np.mean(photons)),
        "photon_number_final": float(photons[-1]),
        "emitted_photon_proxy_kappa_integral_n": emitted_proxy,
        "pump_proxy_gd_sq": float(gd_additive**2),
        "pump_proxy_formula": "gd_additive^2 in wc units (declared proxy, not calibrated pump power)",
        "observation_time_normalized": float(final_time),
        "solver": solver_name,
        "physicality": _physicality_summary(reference.states),
    }


def _row_id(parameters: dict[str, Any]) -> str:
    """Create a stable filesystem-safe row identifier.

    Args:
        parameters: Row parameters with scalar JSON values.

    Returns:
        str: Stable row identifier.
    """
    raw = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
    digest = __import__("hashlib").sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"row_{digest}"


def _parameter_rows(config: FiniteSizeConfig) -> list[dict[str, Any]]:
    """Enumerate the deterministic finite-size rows.

    Args:
        config: Centralized finite-size configuration.

    Returns:
        list[dict[str, Any]]: Ordered parameter rows.
    """
    rows: list[dict[str, Any]] = []
    for n_qubits in config.static_N_values:
        for wq_wc in config.static_wq_wc_values:
            rows.append({"row_type": "static", "N": n_qubits, "wq_wc": wq_wc})
    for n_qubits in config.driven_N_values:
        for kappa_rel in config.driven_kappa_values:
            rows.append({"row_type": "driven", "N": n_qubits, "kappa_rel": kappa_rel})
    for n_qubits in (16, 24):
        rows.append({"row_type": "driven", "N": n_qubits, "kappa_rel": 0.0})
    return rows


def _power_fit(xs: np.ndarray, ys: np.ndarray) -> dict[str, Any]:
    """Fit log10(y) = log10(a) + alpha log10(x) by linear regression.

    Args:
        xs: Emitter counts, dimensionless.
        ys: QFI values, dimensionless.

    Returns:
        dict[str, Any]: alpha, standard error, and prefactor on the
            log10-linear scale, plus the coefficient of determination
            ``r_squared``, the number of fitted points ``n_fit_points``,
            and the Gaussian (z-based) 95% confidence interval for alpha
            as ``confidence_interval_95 = [alpha - 1.96*SE, alpha + 1.96*SE]``.
            All values are computed on the log10 scale; ``alpha`` and its
            standard error are scale-invariant under a change of log base.
    """
    logx = np.log10(xs)
    logy = np.log10(ys)
    coefficients = np.polyfit(logx, logy, 1)
    residuals = logy - np.polyval(coefficients, logx)
    n_points = int(len(logx))
    variance = float(np.sum(residuals**2) / max(n_points - 2, 1))
    design = np.vstack([logx, np.ones_like(logx)]).T
    covariance = variance * np.linalg.inv(design.T @ design)
    alpha = float(coefficients[0])
    alpha_se = float(math.sqrt(max(covariance[0, 0], 0.0)))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((logy - np.mean(logy)) ** 2))
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
    return {
        "alpha": alpha,
        "alpha_standard_error": alpha_se,
        "prefactor_10^intercept": float(10.0 ** coefficients[1]),
        "r_squared": r_squared,
        "confidence_interval_95": [alpha - 1.96 * alpha_se, alpha + 1.96 * alpha_se],
        "n_fit_points": n_points,
    }


def _fit_saturated_forms(xs: np.ndarray, ys: np.ndarray) -> list[dict[str, Any]]:
    """Fit declared descriptive saturation forms by nonlinear least squares.

    Args:
        xs: Emitter counts, dimensionless.
        ys: QFI values, dimensionless.

    Returns:
        list[dict[str, Any]]: Fitted forms with parameters and SSE on log scale.
    """
    forms: list[dict[str, Any]] = []

    def exponential_saturation(n, f_inf, n0):
        return f_inf * (1.0 - np.exp(-n / n0))

    def rational_saturation(n, f_inf, n0):
        return f_inf * n / (n + n0)

    def crossover(n, a, alpha, n0, beta):
        return a * n**alpha / (1.0 + (n / n0) ** beta)

    for name, function, bounds, p0 in (
        (
            "saturated_exponential",
            exponential_saturation,
            ([0.0, 0.1], [np.inf, np.inf]),
            (np.max(ys) * 1.2, 10.0),
        ),
        (
            "saturated_rational",
            rational_saturation,
            ([0.0, 0.1], [np.inf, np.inf]),
            (np.max(ys) * 1.2, 10.0),
        ),
        (
            "crossover_power_saturated",
            crossover,
            ([0.0, 0.0, 0.1, 0.0], [np.inf, np.inf, np.inf, np.inf]),
            (np.max(ys) / 6.0**2, 2.0, 20.0, 2.0),
        ),
    ):
        try:
            fitted, _ = curve_fit(function, xs, ys, p0=p0, bounds=bounds, maxfev=20000)
            predicted = function(xs, *fitted)
            log_sse = float(np.sum((np.log10(predicted) - np.log10(ys)) ** 2))
            max_relative_error = float(np.max(np.abs(predicted - ys) / ys))
            forms.append(
                {
                    "form": name,
                    "parameters": {
                        key: float(value)
                        for key, value in zip(
                            ("f_inf", "n0") if len(fitted) == 2 else ("a", "alpha", "n0", "beta"),
                            fitted,
                        )
                    },
                    "log_sse": log_sse,
                    "max_relative_error": max_relative_error,
                }
            )
        except (RuntimeError, ValueError) as exc:
            forms.append({"form": name, "fit_error": str(exc)[:200]})
    return forms


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the finite-size producer.

    Returns:
        argparse.Namespace: Output, checkpoint, and limit options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "finite_size_resource_scaling.json",
        help="Merged JSON output path.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Per-row checkpoint directory; defaults to data/_cache/finite_size_resource_scaling.",
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


def _scaling_summary(
    static_rows: list[dict[str, Any]], config: FiniteSizeConfig
) -> dict[str, Any]:
    """Build descriptive scaling fits for each detuning family.

    Args:
        static_rows: All static rows with high-truncation QFI.
        config: Centralized finite-size configuration.

    Returns:
        dict[str, Any]: Fit summaries per wq_wc family.
    """
    families: dict[str, Any] = {}
    for row in static_rows:
        family = families.setdefault(
            str(row["wq_wc"]),
            {"wq_wc": row["wq_wc"], "N": [], "qfi": [], "truncation_converged": []},
        )
        family["N"].append(int(row["N"]))
        family["qfi"].append(row["qfi_high_truncation"])
        family["truncation_converged"].append(row["truncation_converged"])
    summaries = []
    for key, family in sorted(families.items()):
        order = np.argsort(family["N"])
        n_values = np.asarray(family["N"], dtype=float)[order]
        qfi_values = np.asarray(family["qfi"], dtype=float)[order]
        mask = (n_values >= config.fit_N_min) & (n_values <= config.fit_N_max)
        in_range = int(np.count_nonzero(mask))
        if in_range < 3:
            summaries.append(
                {
                    "wq_wc": family["wq_wc"],
                    "fit_N_range": [config.fit_N_min, config.fit_N_max],
                    "power_fit": None,
                    "power_fit_omitted_reason": (
                        f"fewer than 3 in-range static rows (have {in_range}); "
                        "fits are reported on the completed production grid."
                    ),
                    "boundary_sensitivity": None,
                    "saturated_form_fits": [],
                    "all_rows_truncation_converged": bool(
                        all(family["truncation_converged"])
                    ),
                }
            )
            continue
        power = _power_fit(n_values[mask], qfi_values[mask])
        boundary_sensitivity = None
        if config.boundary_exclusion_sensitivity and in_range >= 4:
            interior = mask.copy()
            interior[np.argmax(n_values[mask])] = False  # exclude largest in-range N
            power_interior = _power_fit(n_values[interior], qfi_values[interior])
            boundary_sensitivity = {
                "dropped_largest_in_fit_range": {
                    "N_max_dropped": int(n_values[mask][-1]),
                    "alpha_without_largest": power_interior["alpha"],
                }
            }
        saturated = _fit_saturated_forms(n_values, qfi_values)
        summaries.append(
            {
                "wq_wc": family["wq_wc"],
                "fit_N_range": [config.fit_N_min, config.fit_N_max],
                "power_fit": power,
                "boundary_sensitivity": boundary_sensitivity,
                "saturated_form_fits": saturated,
                "all_rows_truncation_converged": bool(
                    all(family["truncation_converged"])
                ),
                "note": (
                    "Descriptive fits over declared ranges; no thermodynamic exponent, "
                    "Heisenberg violation, phase transition, or global bound is claimed."
                ),
            }
        )
    return {"families": summaries, "power_fit_note": "Fit uses high-truncation QFI values."}


def main() -> int:
    """Run and merge the resumable finite-size screen.

    Returns:
        int: Zero after successful checkpoint merge and JSON serialization.
    """
    args = parse_args()
    config = CONFIG
    checkpoint_dir = args.checkpoint_dir or args.output.parent / "_cache" / "finite_size_resource_scaling"
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
            print(f"[{index}/{len(parameters_list)}] simulating {row_id} {parameters['row_type']} N={parameters['N']}")
            if parameters["row_type"] == "static":
                metrics = _simulate_static_row(parameters, config)
            else:
                metrics = _simulate_driven_row(parameters, config)
            row_result = {"row_id": row_id, "parameters": parameters, **metrics}
            checkpoint.write_text(json.dumps(row_result, indent=2) + "\n", encoding="utf-8")
        results.append(row_result)
    results.sort(key=lambda item: item["row_id"])
    static_rows = [row for row in results if row["row_type"] == "static"]
    driven_rows = [row for row in results if row["row_type"] == "driven"]
    scaling = _scaling_summary(static_rows, config)
    payload = {
        "schema": "trackB_finite_size_resource_scaling_v1",
        "status": "finite_size_scaling_and_resource_screen",
        "purpose": (
            "Quantify static Dicke ground-state QFI versus emitter count with truncation "
            "checks and record driven Dicke+DCE resource diagnostics."
        ),
        "configuration": asdict(config),
        "units": {
            "N": "emitter count, dimensionless",
            "eta": "g0/wc, dimensionless",
            "wq_wc": "dimensionless",
            "qfi": "inverse squared normalized coupling (static convention: 4 sum |<0|V|n>|^2 / gap^2)",
            "kappa_rel": "kappa/wc, dimensionless",
            "time": "drive periods",
            "photon_number": "photons",
            "emitted_photon_proxy": "kappa_rel * integral n(t) dt in normalized time",
            "pump_proxy": "gd_additive^2 in wc units",
        },
        "solver": {
            "static": "exact QuTiP sparse diagonalization (gap-weighted ground-state sum)",
            "driven": "qutip.sesolve for kappa_rel=0; qutip.mesolve otherwise",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "qutip": qt.__version__,
        },
        "validation": {
            "rows_requested": len(parameters_list),
            "rows_completed": len(results),
            "all_rows_complete": len(parameters_list) == len(results),
            "static_row_count": len(static_rows),
            "driven_row_count": len(driven_rows),
            "static_truncation_converged_fraction": (
                float(sum(1 for row in static_rows if row["truncation_converged"])) / len(static_rows)
                if static_rows
                else None
            ),
            "scaling_fits": scaling,
            "numerical_physicality_note": "Trace/Hermiticity/eigenvalue acceptance is not a proof of exact positivity.",
            "scope_note": (
                "Finite-size results are descriptive over the declared interval; no "
                "thermodynamic exponent, phase transition, Heisenberg violation, or "
                "global bound is claimed. Resource entries use the declared proxies."
            ),
        },
        "caveats": [
            "Static QFI values use the ground-state gap-weighted convention with the declared eigenstate count.",
            "Driven rows are reduced-truncation diagnostics (n_cav=6) as in the canonical loss-adjusted Dicke study.",
            "Resource accounting uses declared proxies (pump proxy gd_additive^2, emitted-photon proxy kappa integral n); they are not calibrated pump power or measured output.",
            "Rows with truncation change above the declared tolerance are flagged and excluded from the primary fit interpretation.",
            "No universal advantage claim is made without jointly matched resources.",
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
