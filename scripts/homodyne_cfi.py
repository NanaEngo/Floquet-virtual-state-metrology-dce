#!/usr/bin/env python3
"""Compute output-field homodyne CFI for the Track B DCE model.

The producer uses the declared Markovian input-output convention and a flat
finite temporal mode.  It evaluates first and second output-mode moments with
QuTiP's quantum-regression correlations, approximates the mode quadrature law
as Gaussian, and compares its classical Fisher information with the matching
joint-system fidelity QFI.  The result is deliberately model-specific and is
not an exact CFI for a generally non-Gaussian temporal-mode state.
"""

from __future__ import annotations

import argparse
import json
import math
import hashlib
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import qutip as qt

from trackb_qutip_utils import density_matrix, squared_fidelity

try:
    from config import DEFAULT_HOMODYNE, HomodyneConfig
except ImportError as exc:  # pragma: no cover - environment-specific
    raise ImportError(
        "Track B configuration is unavailable; run this script from scripts/."
    ) from exc


CONFIG = DEFAULT_HOMODYNE


def _operators(n_cav: int) -> dict[str, qt.Qobj]:
    """Build tensor-ordered cavity and qubit operators.

    Args:
        n_cav: Cavity Fock-space dimension, dimensionless.

    Returns:
        dict[str, qt.Qobj]: Annihilation, number, sigma-x, sigma-z, and
        identity operators in normalized cavity-frequency units.
    """
    a = qt.tensor(qt.destroy(n_cav), qt.qeye(2))
    return {
        "a": a,
        "n": a.dag() * a,
        "x": qt.tensor(qt.qeye(n_cav), qt.sigmax()),
        "z": qt.tensor(qt.qeye(n_cav), qt.sigmaz()),
        "identity": qt.tensor(qt.qeye(n_cav), qt.qeye(2)),
    }


def _hamiltonian(
    config: HomodyneConfig,
    operators: dict[str, qt.Qobj],
    eta: float,
    protocol: str,
) -> qt.Qobj | qt.QobjEvo:
    """Construct a static or driven normalized Rabi Hamiltonian.

    Args:
        config: Centralized homodyne configuration, dimensionless except for
            phase-like quantities recorded in radians.
        operators: Tensor-ordered operators from ``_operators``.
        eta: Coupling estimand g0/wc, dimensionless.
        protocol: Either ``static`` or ``driven``.

    Returns:
        qt.Qobj | qt.QobjEvo: Hamiltonian in normalized angular-frequency
        units.

    Raises:
        ValueError: If ``protocol`` is not supported.
    """
    static = 0.5 * config.wq_wc * operators["z"] + operators["n"]
    interaction = operators["x"] * (operators["a"] + operators["a"].dag())
    if protocol == "static":
        return static + eta * interaction
    if protocol != "driven":
        raise ValueError(f"Unsupported protocol: {protocol}")

    def coefficient(time_normalized: float, _args: dict | None = None) -> float:
        """Return the instantaneous driven interaction coefficient.

        Args:
            time_normalized: Time in inverse cavity angular-frequency units.
            _args: Optional QuTiP callback arguments, unused.

        Returns:
            float: Dimensionless time-dependent coupling coefficient.
        """
        return eta + config.gd * math.cos(config.wd_wc * time_normalized)

    return qt.QobjEvo([static, [interaction, coefficient]])


# State kernels (density_matrix, squared_fidelity) are shared with the other
# screen producers via trackb_qutip_utils (B-2); screen-specific acceptance
# criteria stay local.


def _fidelity_qfi(reference: qt.Qobj, plus: qt.Qobj, minus: qt.Qobj, step: float) -> float:
    """Estimate coupling QFI from symmetric one-sided fidelity estimates.

    Args:
        reference: State at the nominal coupling, dimensionless.
        plus: State at eta plus ``step``, dimensionless.
        minus: State at eta minus ``step``, dimensionless.
        step: Central coupling step, dimensionless.

    Returns:
        float: Fidelity-based QFI for eta, inverse squared normalized-coupling
        units.
    """
    if step <= 0.0:
        raise ValueError("The finite-difference step must be positive.")
    estimates = [
        8.0 * (1.0 - math.sqrt(squared_fidelity(reference, state))) / step**2
        for state in (plus, minus)
    ]
    return float(np.mean(estimates))


def _propagate(
    config: HomodyneConfig,
    eta: float,
    protocol: str,
    n_cav: int,
    kappa_rel: float,
    final_periods: int,
) -> dict[str, Any]:
    """Propagate one reference or perturbed state and output correlation data.

    Args:
        config: Centralized homodyne configuration.
        eta: Coupling estimand g0/wc, dimensionless.
        protocol: ``static`` or ``driven``.
        n_cav: Cavity Fock-space dimension, dimensionless.
        kappa_rel: Total loss divided by wc, dimensionless.
        final_periods: Output-mode window in drive periods, dimensionless.

    Returns:
        dict[str, Any]: State trajectory, time grid, mean cavity amplitude,
        and normal/anomalous two-time correlations.
    """
    operators = _operators(n_cav)
    a = operators["a"]
    drive_frequency = config.wd_wc if protocol == "driven" else config.wd_wc
    drive_period = 2.0 * math.pi / drive_frequency
    final_time = final_periods * drive_period
    times = np.linspace(0.0, final_time, config.n_solver_points)
    if protocol == "static":
        _, initial = _hamiltonian(config, operators, eta, protocol).groundstate()
        initial = initial / initial.norm()
    else:
        initial = qt.tensor(qt.basis(n_cav, 0), qt.basis(2, 0))
    collapse_ops = [math.sqrt(kappa_rel) * a] if kappa_rel > 0.0 else []
    options = {
        "atol": config.solver_atol,
        "rtol": config.solver_rtol,
        "method": "adams",
        "max_step": config.max_step_fraction * drive_period,
        "store_states": True,
        "progress_bar": "",
    }
    result = qt.mesolve(
        _hamiltonian(config, operators, eta, protocol),
        initial,
        times,
        c_ops=collapse_ops,
        e_ops=[a],
        options=options,
    )
    correlation_options = {
        "atol": config.solver_atol,
        "rtol": config.solver_rtol,
        "method": "adams",
        "max_step": config.max_step_fraction * drive_period,
        "progress_bar": "",
    }
    normal = qt.correlation_2op_2t(
        _hamiltonian(config, operators, eta, protocol),
        initial,
        times,
        times,
        collapse_ops,
        a.dag(),
        a,
        options=correlation_options,
        max_t_plus_tau=final_time,
    )
    anomalous_forward = qt.correlation_2op_2t(
        _hamiltonian(config, operators, eta, protocol),
        initial,
        times,
        times,
        collapse_ops,
        a,
        a,
        options=correlation_options,
        max_t_plus_tau=final_time,
    )
    anomalous_reverse = qt.correlation_2op_2t(
        _hamiltonian(config, operators, eta, protocol),
        initial,
        times,
        times,
        collapse_ops,
        a,
        a,
        reverse=True,
        options=correlation_options,
        max_t_plus_tau=final_time,
    )
    return {
        "state": result.states[-1],
        "states": result.states,
        "times": times,
        "drive_period": drive_period,
        "mean_a": np.asarray(result.expect[0], dtype=complex),
        "normal": np.asarray(normal, dtype=complex),
        "anomalous_forward": np.asarray(anomalous_forward, dtype=complex),
        "anomalous_reverse": np.asarray(anomalous_reverse, dtype=complex),
        "solver": "qutip.mesolve + qutip.correlation_2op_2t",
    }


def _ordered_correlation(
    normal: np.ndarray,
    times: np.ndarray,
    operator_order: str,
) -> np.ndarray:
    """Reconstruct a two-time correlation matrix from sum-grid correlations.

    Args:
        normal: Correlation matrix whose entries are ``<A(t+tau)B(t)>``.
        times: Uniform time grid in inverse cavity angular-frequency units.
        operator_order: Correlation label, currently ``normal`` or ``aa``.

    Returns:
        np.ndarray: Matrix indexed by the two absolute times, complex.

    Raises:
        ValueError: If the grid is not uniform or the label is unsupported.
    """
    if len(times) < 2 or not np.allclose(np.diff(times), np.diff(times)[0]):
        raise ValueError("Correlation reconstruction requires a uniform time grid.")
    if operator_order not in {"normal", "aa"}:
        raise ValueError(f"Unsupported correlation label: {operator_order}")
    output = np.empty((len(times), len(times)), dtype=complex)
    for first in range(len(times)):
        for second in range(len(times)):
            if first >= second:
                output[first, second] = normal[second, first - second]
            else:
                output[first, second] = np.conjugate(normal[first, second - first])
    return output


def _reconstruct_aa(forward: np.ndarray, reverse: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Reconstruct the ordered anomalous correlation matrix.

    Args:
        forward: ``<a(t+tau)a(t)>`` matrix on a sum-grid, complex.
        reverse: ``<a(t)a(t+tau)>`` matrix on a sum-grid, complex.
        times: Uniform time grid in inverse cavity angular-frequency units.

    Returns:
        np.ndarray: Matrix with entry ``<a(t_i)a(t_j)>``, complex.
    """
    if len(times) < 2 or not np.allclose(np.diff(times), np.diff(times)[0]):
        raise ValueError("Correlation reconstruction requires a uniform time grid.")
    output = np.empty((len(times), len(times)), dtype=complex)
    for first in range(len(times)):
        for second in range(len(times)):
            if first >= second:
                output[first, second] = forward[second, first - second]
            else:
                output[first, second] = reverse[first, second - first]
    return output


def _subsample_indices(
    times: np.ndarray, final_periods: int, mode_points: int, drive_period: float
) -> np.ndarray:
    """Select a uniform finite-mode grid from the propagation grid.

    Args:
        times: Full propagation grid in inverse cavity angular-frequency units.
        final_periods: Mode window in drive periods, dimensionless.
        mode_points: Number of mode quadrature points, dimensionless.
        drive_period: Drive period in inverse cavity angular-frequency units.

    Returns:
        np.ndarray: Integer indices including both mode-window endpoints.
    """
    end_time = final_periods * drive_period
    endpoint = int(np.argmin(np.abs(times - end_time)))
    if endpoint < mode_points - 1:
        raise ValueError("Propagation grid is too coarse for the requested mode.")
    return np.rint(np.linspace(0, endpoint, mode_points)).astype(int)


def _mode_moments(
    propagation: dict[str, Any],
    kappa_ext: float,
    final_periods: int,
    mode_points: int,
    config: HomodyneConfig,
) -> dict[str, float]:
    """Integrate output-mode first and second moments for a flat temporal mode.

    Args:
        propagation: Output of ``_propagate``.
        kappa_ext: Monitored loss divided by wc, dimensionless.
        final_periods: Mode window in drive periods, dimensionless.
        mode_points: Temporal-mode resolution, dimensionless.
        config: Centralized configuration, used for numerical validation.

    Returns:
        dict[str, float]: Complex mode amplitude and normal/anomalous moments,
        serialized as real/imaginary components plus normalization metadata.
    """
    times = propagation["times"]
    drive_period = float(propagation["drive_period"])
    indices = _subsample_indices(times, final_periods, mode_points, drive_period)
    mode_times = times[indices]
    tau = float(mode_times[-1] - mode_times[0])
    if tau <= 0.0:
        raise ValueError("Temporal mode must have positive duration.")
    mean_a = propagation["mean_a"][indices]
    normal_sum_grid = propagation["normal"]
    normal = _ordered_correlation(normal_sum_grid, times, "normal")[np.ix_(indices, indices)]
    aa_forward = propagation["anomalous_forward"]
    aa_reverse = propagation["anomalous_reverse"]
    anomalous = _reconstruct_aa(aa_forward, aa_reverse, times)[np.ix_(indices, indices)]
    integral_mean = np.trapz(mean_a, mode_times)
    integral_normal = np.trapz(np.trapz(normal, mode_times, axis=1), mode_times)
    integral_anomalous = np.trapz(np.trapz(anomalous, mode_times, axis=1), mode_times)
    amplitude = math.sqrt(kappa_ext / tau) * integral_mean
    normal_moment = (kappa_ext / tau) * integral_normal
    anomalous_moment = (kappa_ext / tau) * integral_anomalous
    if not np.all(np.isfinite([amplitude.real, amplitude.imag, normal_moment.real, normal_moment.imag, anomalous_moment.real, anomalous_moment.imag])):
        raise ValueError("Non-finite output-mode moment encountered.")
    return {
        "alpha_real": float(amplitude.real),
        "alpha_imag": float(amplitude.imag),
        "normal_moment": float(np.real(normal_moment)),
        "normal_moment_imag": float(np.imag(normal_moment)),
        "anomalous_moment_real": float(np.real(anomalous_moment)),
        "anomalous_moment_imag": float(np.imag(anomalous_moment)),
        "mode_duration_normalized": tau,
        "mode_points": int(mode_points),
        "flat_mode_normalization_integral": float(np.trapz(np.ones_like(mode_times) / math.sqrt(tau), mode_times)),
        "mode_time_start": float(mode_times[0]),
        "mode_time_end": float(mode_times[-1]),
        "shot_noise_quadrature_variance": 0.5,
    }


def _quadrature_statistics(
    moments: dict[str, float],
    local_oscillator_phase: float,
    detector_efficiency: float,
) -> tuple[float, float]:
    """Compute Gaussian homodyne mean and variance for one LO phase.

    Args:
        moments: Output-mode moments from ``_mode_moments``.
        local_oscillator_phase: LO quadrature phase, radians.
        detector_efficiency: Homodyne efficiency, dimensionless in [0, 1].

    Returns:
        tuple[float, float]: Detected quadrature mean and variance in the
        convention X=(A exp(-i theta)+A^dagger exp(i theta))/sqrt(2).
    """
    alpha = complex(moments["alpha_real"], moments["alpha_imag"])
    normal = moments["normal_moment"]
    anomalous = complex(moments["anomalous_moment_real"], moments["anomalous_moment_imag"])
    phase = local_oscillator_phase
    mean = math.sqrt(2.0) * float(np.real(np.exp(-1j * phase) * alpha))
    centered_normal = normal - abs(alpha) ** 2
    centered_anomalous = anomalous - alpha**2
    variance = 0.5 + centered_normal + float(np.real(np.exp(-2j * phase) * centered_anomalous))
    detected_mean = math.sqrt(detector_efficiency) * mean
    detected_variance = detector_efficiency * variance + (1.0 - detector_efficiency) * 0.5
    return float(detected_mean), float(detected_variance)


def _gaussian_mode_state(
    moments: dict[str, float],
    truncation: int,
) -> qt.Qobj:
    """Reconstruct a truncated single-mode Gaussian state from output moments.

    Args:
        moments: Output-mode moments containing amplitude, normal, and
            anomalous second moments, dimensionless.
        truncation: Fock-space dimension for the reconstructed mode state,
            dimensionless.

    Returns:
        qt.Qobj: Density operator approximating the Gaussian temporal-mode
        state, dimensionless.

    Raises:
        ValueError: If the centered moments are not physically Gaussian or the
            reconstruction is numerically unstable.
    """
    if truncation < 8:
        raise ValueError("Gaussian mode truncation must be at least 8.")
    alpha = complex(moments["alpha_real"], moments["alpha_imag"])
    normal = float(moments["normal_moment"])
    anomalous = complex(moments["anomalous_moment_real"], moments["anomalous_moment_imag"])
    centered_normal = normal - abs(alpha) ** 2
    centered_anomalous = anomalous - alpha**2
    symplectic = math.sqrt(max(0.0, (centered_normal + 0.5) ** 2 - abs(centered_anomalous) ** 2))
    if centered_normal < -1.0e-8 or symplectic < 0.5 - 1.0e-8:
        raise ValueError("Output moments violate the single-mode Gaussian uncertainty bound.")
    thermal_occupation = max(0.0, symplectic - 0.5)
    squeeze_argument = max(0.0, (centered_normal + 0.5) / max(symplectic, 1.0e-15))
    squeeze_radius = 0.5 * math.acosh(squeeze_argument)
    squeeze_phase = float(np.angle(-centered_anomalous)) if abs(centered_anomalous) > 1.0e-14 else 0.0
    squeeze_parameter = squeeze_radius * np.exp(1j * squeeze_phase)
    thermal = qt.thermal_dm(truncation, thermal_occupation)
    squeezed = qt.squeeze(truncation, squeeze_parameter) * thermal * qt.squeeze(truncation, squeeze_parameter).dag()
    displaced = qt.displace(truncation, alpha) * squeezed * qt.displace(truncation, alpha).dag()
    return displaced / displaced.tr()


def _gaussian_output_qfi(
    base_moments: dict[str, float],
    plus_moments: dict[str, float],
    minus_moments: dict[str, float],
    step: float,
    truncation: int,
) -> float:
    """Compute fidelity QFI of the reconstructed Gaussian output-mode state.

    Args:
        base_moments: Nominal output-mode moments.
        plus_moments: Moments at eta plus ``step``.
        minus_moments: Moments at eta minus ``step``.
        step: Central eta step, dimensionless.
        truncation: Reconstructed temporal-mode Fock dimension, dimensionless.

    Returns:
        float: Gaussian-output-state QFI in inverse squared normalized-coupling
        units.
    """
    if step <= 0.0:
        raise ValueError("The output-state finite-difference step must be positive.")
    base = _gaussian_mode_state(base_moments, truncation)
    qfi_terms = []
    for moments in (plus_moments, minus_moments):
        state = _gaussian_mode_state(moments, truncation)
        fidelity = squared_fidelity(base, state)
        qfi_terms.append(8.0 * (1.0 - math.sqrt(fidelity)) / step**2)
    return float(np.mean(qfi_terms))


def _row_result(
    base_moments: dict[str, float],
    plus_moments: dict[str, float],
    minus_moments: dict[str, float],
    qfi: float,
    joint_system_qfi: float,
    gaussian_state_truncation: int,
    phase: float,
    efficiency: float,
    finite_difference_step: float,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Calculate Gaussian homodyne CFI and its matching QFI ratio.

    Args:
        base_moments: Nominal output-mode moments.
        plus_moments: Moments at eta plus the finite-difference step.
        minus_moments: Moments at eta minus the finite-difference step.
    qfi: Matching output-mode Gaussian QFI, inverse squared eta units.
        joint_system_qfi: Residual cavity-qubit fidelity QFI diagnostic,
            inverse squared eta units.
        gaussian_state_truncation: Fock dimension of the reconstructed output
            Gaussian state, dimensionless.
        phase: LO phase, radians.
        efficiency: Detector efficiency, dimensionless.
        finite_difference_step: Eta finite-difference step, dimensionless.
        parameters: Row configuration and protocol metadata.

    Returns:
        dict[str, Any]: Gaussian quadrature statistics, CFI, QFI, ratio, and
        numerical validity status.
    """
    base_mean, base_variance = _quadrature_statistics(base_moments, phase, efficiency)
    plus_mean, plus_variance = _quadrature_statistics(plus_moments, phase, efficiency)
    minus_mean, minus_variance = _quadrature_statistics(minus_moments, phase, efficiency)
    mean_derivative = (plus_mean - minus_mean) / (2.0 * finite_difference_step)
    variance_derivative = (plus_variance - minus_variance) / (2.0 * finite_difference_step)
    cfi_mean = mean_derivative**2 / base_variance
    cfi_variance = 0.5 * variance_derivative**2 / base_variance**2
    cfi = float(cfi_mean + cfi_variance)
    ratio = float(cfi / qfi) if qfi > 0.0 else None
    valid = bool(
        np.isfinite(cfi)
        and np.isfinite(base_variance)
        and base_variance > 0.0
        and cfi >= -1.0e-10
        and qfi > 0.0
        and ratio is not None
        and np.isfinite(ratio)
        and ratio >= -1.0e-7
        and ratio <= 1.0 + 1.0e-7
    )
    if not valid and ratio is not None and np.isfinite(ratio):
        raise ValueError(
            "Same-state Gaussian homodyne CFI/QFI ratio violated [0, 1]: "
            f"{ratio:.6g} for {parameters}"
        )
    return {
        **parameters,
        "local_oscillator_phase_rad": float(phase),
        "detector_efficiency": float(efficiency),
        "finite_difference_step": float(finite_difference_step),
        "quadrature_mean": base_mean,
        "quadrature_variance": base_variance,
        "quadrature_mean_derivative": mean_derivative,
        "quadrature_variance_derivative": variance_derivative,
        "cfi_mean_term": float(cfi_mean),
        "cfi_variance_term": float(cfi_variance),
        "classical_fisher_information": cfi,
        "matching_output_gaussian_qfi": float(qfi),
        "qfi_comparison_scope": "same output temporal mode and same eta finite-difference step",
        "residual_joint_system_qfi": float(joint_system_qfi),
        "gaussian_state_truncation": int(gaussian_state_truncation),
        "cfi_qfi_ratio": ratio,
        "same_state_ratio_valid": valid,
        "status": "valid" if valid else "inconclusive_ratio_or_metric",
    }


def _physicality_summary(state: qt.Qobj) -> dict[str, float | bool]:
    """Record numerical state diagnostics for one final density operator.

    Args:
        state: Final QuTiP ket or density operator, dimensionless.

    Returns:
        dict[str, float | bool]: Trace, Hermiticity, and minimum eigenvalue
        diagnostics; numerical acceptance is not exact positivity.
    """
    rho = density_matrix(state)
    eigenvalues = np.real_if_close(rho.eigenenergies()).real
    trace_error = abs(float(np.real(rho.tr())) - 1.0)
    hermiticity_error = float((rho - rho.dag()).norm())
    minimum_eigenvalue = float(np.min(eigenvalues))
    return {
        "trace_error": trace_error,
        "hermiticity_error": hermiticity_error,
        "minimum_eigenvalue": minimum_eigenvalue,
        "numerical_acceptance": bool(
            trace_error <= 1.0e-8
            and hermiticity_error <= 1.0e-8
            and minimum_eigenvalue >= -1.0e-7
        ),
        "note": "Numerical acceptance is not a proof of exact positivity.",
    }


def _parameter_groups(config: HomodyneConfig) -> list[dict[str, Any]]:
    """Enumerate bounded simulation groups before phase/efficiency expansion.

    Args:
        config: Centralized homodyne configuration.

    Returns:
        list[dict[str, Any]]: Ordered protocol, loss, coupling, window,
        resolution, and truncation groups.
    """
    groups = []
    for protocol in ("static", "driven"):
        for kappa_rel in config.kappa_values:
            for external_fraction in config.external_fractions:
                for n_cav in config.cavity_truncations:
                    groups.append(
                        {
                            "protocol": protocol,
                            "kappa_rel": kappa_rel,
                            "external_fraction": external_fraction,
                            "n_cav": n_cav,
                        }
                    )
    return groups


def _group_id(group: dict[str, Any]) -> str:
    """Create a deterministic identifier for a homodyne simulation group.

    Args:
        group: Protocol, loss split, and cavity-truncation parameters.

    Returns:
        str: Stable filesystem-safe group identifier.
    """
    raw = json.dumps(group, sort_keys=True, separators=(",", ":"))
    return "group_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _simulate_group(config: HomodyneConfig, group: dict[str, Any]) -> list[dict[str, Any]]:
    """Simulate one group through the longest window and expand its metrics.

    Args:
        config: Centralized homodyne configuration.
        group: Protocol, loss split, and cavity-truncation parameters.

    Returns:
        list[dict[str, Any]]: JSON-ready rows for every window, mode
        resolution, finite-difference step, LO phase, and efficiency.
    """
    kappa_rel = float(group["kappa_rel"])
    kappa_ext = kappa_rel * float(group["external_fraction"])
    eta = config.eta
    max_periods = max(config.drive_windows)
    base = _propagate(config, eta, group["protocol"], group["n_cav"], kappa_rel, max_periods)
    physicality = _physicality_summary(base["state"])
    rows: list[dict[str, Any]] = []
    for step in config.finite_difference_steps:
        plus = _propagate(config, eta + step, group["protocol"], group["n_cav"], kappa_rel, max_periods)
        minus = _propagate(config, eta - step, group["protocol"], group["n_cav"], kappa_rel, max_periods)
        for final_periods in config.drive_windows:
            endpoint = int(np.argmin(np.abs(base["times"] - final_periods * base["drive_period"])))
            joint_system_qfi = _fidelity_qfi(
                base["states"][endpoint],
                plus["states"][endpoint],
                minus["states"][endpoint],
                step,
            )
            for mode_points in config.mode_points:
                base_moments = _mode_moments(base, kappa_ext, final_periods, mode_points, config)
                plus_moments = _mode_moments(plus, kappa_ext, final_periods, mode_points, config)
                minus_moments = _mode_moments(minus, kappa_ext, final_periods, mode_points, config)
                output_qfi = _gaussian_output_qfi(
                    base_moments,
                    plus_moments,
                    minus_moments,
                    step,
                    config.gaussian_state_truncation,
                )
                row_parameters = {
                    "protocol": group["protocol"],
                    "eta": eta,
                    "kappa_rel": kappa_rel,
                    "kappa_ext_rel": kappa_ext,
                    "kappa_int_rel": kappa_rel - kappa_ext,
                    "external_fraction": group["external_fraction"],
                    "drive_window_periods": final_periods,
                    "mode_points": mode_points,
                    "n_cav": group["n_cav"],
                    "mode_moments": base_moments,
                    "physicality": physicality,
                }
                for phase in config.local_oscillator_phases:
                    for efficiency in config.detector_efficiencies:
                        row = _row_result(
                            base_moments,
                            plus_moments,
                            minus_moments,
                            output_qfi,
                            joint_system_qfi,
                            config.gaussian_state_truncation,
                            phase,
                            efficiency,
                            step,
                            row_parameters,
                        )
                        row["perturbed_physicality"] = {
                            "plus": _physicality_summary(plus["states"][endpoint]),
                            "minus": _physicality_summary(minus["states"][endpoint]),
                        }
                        rows.append(row)
    return rows


def _hypothesis_status(rows: list[dict[str, Any]], protocol: str, target: float, tolerance: float) -> dict[str, Any]:
    """Assess one historical CFI/QFI percentage hypothesis.

    Args:
        rows: All produced rows, each with a CFI/QFI ratio.
        protocol: Protocol to filter, ``static`` or ``driven``.
        target: Historical ratio as a fraction, dimensionless.
        tolerance: Relative comparison tolerance, dimensionless.

    Returns:
        dict[str, Any]: Best observed row, target, tolerance, and verdict.
    """
    candidates = [row for row in rows if row["protocol"] == protocol and row["same_state_ratio_valid"]]
    if not candidates:
        return {"target_ratio": target, "relative_tolerance": tolerance, "status": "inconclusive", "reason": "no valid same-state rows"}
    best = min(candidates, key=lambda row: abs(row["cfi_qfi_ratio"] - target))
    relative_error = abs(best["cfi_qfi_ratio"] - target) / target
    return {
        "target_ratio": target,
        "target_percent": 100.0 * target,
        "relative_tolerance": tolerance,
        "best_observed_ratio": best["cfi_qfi_ratio"],
        "best_observed_percent": 100.0 * best["cfi_qfi_ratio"],
        "best_row": {
            "protocol": best["protocol"],
            "phase_rad": best["local_oscillator_phase_rad"],
            "efficiency": best["detector_efficiency"],
            "kappa_rel": best["kappa_rel"],
            "external_fraction": best["external_fraction"],
            "drive_window_periods": best["drive_window_periods"],
            "mode_points": best["mode_points"],
            "n_cav": best["n_cav"],
        },
        "relative_error": relative_error,
        "status": "reproduced" if relative_error <= tolerance else "not_reproduced",
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the homodyne producer.

    Returns:
        argparse.Namespace: Output path and optional bounded group limit.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "homodyne_cfi.json",
        help="JSON output path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N simulation groups for a resumable screening run.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Zero-based starting group index for bounded batch execution.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Per-group checkpoint directory; defaults to data/_cache/homodyne_cfi.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing group checkpoints and recompute selected groups.",
    )
    return parser.parse_args()


def _read_checkpoint(path: Path) -> dict[str, Any]:
    """Read one checkpoint, tolerating the known escaped-newline suffix.

    Args:
        path: Checkpoint JSON path.

    Returns:
        dict[str, Any]: Checkpoint group and completed rows.

    Raises:
        json.JSONDecodeError: If the checkpoint remains malformed after the
            narrowly scoped compatibility cleanup.
    """
    text = path.read_text(encoding="utf-8")
    if text.endswith("\\n"):
        text = text[:-2]
    return json.loads(text)


def main() -> int:
    """Run the deterministic Gaussian homodyne-CFI study.

    Returns:
        int: Zero when JSON serialization succeeds; numerical inconclusive rows
        remain recorded for evidence-bounded interpretation.
    """
    args = parse_args()
    started = time.perf_counter()
    groups = _parameter_groups(CONFIG)
    if args.offset < 0 or args.offset >= len(groups):
        raise ValueError("--offset must be within the configured group range")
    if args.offset:
        groups = groups[args.offset:]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        groups = groups[: args.limit]
    checkpoint_dir = args.checkpoint_dir or args.output.parent / "_cache" / "homodyne_cfi"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, group in enumerate(groups, start=1):
        group_id = _group_id(group)
        checkpoint = checkpoint_dir / f"{group_id}.json"
        print(f"[{index}/{len(groups)}] {group_id} {group['protocol']} kappa={group['kappa_rel']:.4g} ext={group['external_fraction']:.3g} Ncav={group['n_cav']}")
        if checkpoint.is_file() and not args.no_resume:
            checkpoint_data = _read_checkpoint(checkpoint)
            if checkpoint_data.get("group") != group:
                raise ValueError(f"Checkpoint parameters do not match group {group_id}.")
            group_rows = checkpoint_data["rows"]
            print(f"  resumed {len(group_rows)} rows")
        else:
            group_rows = _simulate_group(CONFIG, group)
            checkpoint.write_text(
                json.dumps({"group": group, "rows": group_rows}, indent=2) + "\n",
                encoding="utf-8",
            )
        rows.extend(group_rows)
    static_status = _hypothesis_status(rows, "static", 0.924, CONFIG.hypothesis_tolerance)
    driven_status = _hypothesis_status(rows, "driven", 0.881, CONFIG.hypothesis_tolerance)
    ratios = np.asarray(
        [row["cfi_qfi_ratio"] for row in rows if row["same_state_ratio_valid"]],
        dtype=float,
    )
    payload = {
        "schema": "trackB_homodyne_cfi_v1",
        "status": "gaussian_output_field_cfi_screen",
        "purpose": "Quantify accessible coupling information in a declared output-field homodyne measurement.",
        "configuration": asdict(CONFIG),
        "input_output": {
            "convention": "a_out(t) = a_in(t) + sqrt(kappa_ext) a(t)",
            "sign_convention": "plus sign; changing the input phase changes only the output phase convention",
            "input_state": "vacuum",
            "total_loss": "kappa = kappa_ext + kappa_int, all normalized by wc",
            "monitored_channel": "external cavity-loss channel only",
            "internal_channel": "unmonitored residual cavity loss",
            "temporal_mode": "flat f(t)=1/sqrt(tau) over [0,tau] in normalized time",
            "quadrature": "X_theta=(A_f exp(-i theta)+A_f^dagger exp(i theta))/sqrt(2)",
            "shot_noise_variance": 0.5,
        },
        "solver": {
            "propagation": "qutip.mesolve",
            "correlations": "qutip.correlation_2op_2t using the quantum regression theorem",
            "integration": "composite trapezoid on the stored temporal mode grid",
            "gaussian_distribution": "mean and variance of a single-mode Gaussian quadrature law",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "qutip": qt.__version__,
        },
        "validation": {
            "group_count_requested": len(groups),
            "group_count_completed": len({
                _group_id({
                    "protocol": row["protocol"],
                    "kappa_rel": row["kappa_rel"],
                    "external_fraction": row["external_fraction"],
                    "n_cav": row["n_cav"],
                })
                for row in rows
            }),
            "row_count": len(rows),
            "all_rows_finite": bool(all(np.all(np.isfinite([row["classical_fisher_information"], row["matching_output_gaussian_qfi"], row["residual_joint_system_qfi"]])) for row in rows)),
            "nonnegative_cfi": bool(all(row["classical_fisher_information"] >= -1.0e-10 for row in rows)),
            "valid_ratio_count": int(np.count_nonzero(np.isfinite(ratios))),
            "ratio_min_valid": float(np.min(ratios)) if ratios.size else None,
            "ratio_max_valid": float(np.max(ratios)) if ratios.size else None,
            "ratio_bound_tolerance": 1.0e-7,
            "same_state_ratio_bounds_pass": bool(
                all(
                    -1.0e-7 <= row["cfi_qfi_ratio"] <= 1.0 + 1.0e-7
                    for row in rows
                    if row["same_state_ratio_valid"]
                )
            ),
            "hypotheses": {"static_92_4_percent": static_status, "driven_88_1_percent": driven_status},
            "finite_difference_steps": list(CONFIG.finite_difference_steps),
            "convergence_note": "Rows include finite-difference, mode-resolution, window, phase, efficiency, external-coupling, and cavity-truncation scans; this is a bounded screen, not an asymptotic convergence proof.",
        },
        "caveats": [
            "CFI is computed for a Gaussian moment-based quadrature approximation, not the exact generally non-Gaussian temporal-mode distribution.",
            "The output mode includes the explicit vacuum shot-noise variance but does not reconstruct the full continuum output state or detector bandwidth chain.",
            "The primary QFI comparator is the fidelity QFI of the reconstructed Gaussian output-mode state; the residual cavity-qubit QFI is retained as a separate diagnostic.",
            "The input-output mapping treats only kappa_ext as monitored; internal loss is not measurable output.",
            "The flat temporal mode is a declared analysis choice, not an optimized matched filter.",
            "The calculation does not establish experimental QFI saturation, universal homodyne optimality, or resource-normalized advantage.",
        ],
        "rows": rows,
        "elapsed_s": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output} ({len(rows)} rows)")
    print(f"Historical hypotheses: static={static_status['status']}; driven={driven_status['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
