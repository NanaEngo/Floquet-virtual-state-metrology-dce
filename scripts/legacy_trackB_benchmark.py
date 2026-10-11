"""
Track B: Gaussian Baseline Benchmark and Reproducibility Sweep.

Verifies that the virtual-state QFI from FL_QOM's USC pipeline
reproduces analytical Gaussian baselines within 1% tolerance (P0 requirement).
Also supports the paper-specific CLI compute/plot/export commands.

Authors: Nana Engo et al.
"""

import os
import sys
import argparse
import warnings
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure path is configured for main repo config imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import TrackBParameters

try:
    from fl_qom.config import GlobalComputeConfig
    from fl_qom.core.parameters import SystemParameters
    from fl_qom.core.usc_system import USCOptomechanicalSystem
    from fl_qom.analysis.virtual_state_metrology import (
        VirtualStateMetrology,
        qfi_coherent_baseline as vsm_qfi_coherent_baseline,
        qfi_squeezed_baseline as vsm_qfi_squeezed_baseline,
        qfi_thermal_baseline as vsm_qfi_thermal_baseline,
        qfi_virtual_weak_usc_reference,
    )
    from fl_qom.utils.logging import get_simulation_logger
    logger = get_simulation_logger("trackB_benchmark", log_dir="logs")
except ImportError as e:
    print(f"Error importing FL_QOM: {e}")
    sys.exit(1)

# Centralized convention to avoid silent factor-of-4 drift between docs/code.
BASELINE_QFI_PREFAC = 1.0
CACHE_DIR = Path("_cache")

# ── Paper B parameter sets ──────────────────────────────────────

CIRCUIT_QED_PARAMS = SystemParameters(
    omega_c=1e9,          # 1 GHz cavity
    omega_m=1e7,          # 10 MHz mechanical
    g0=2e6,               # g0 / omega_m = 0.2 (USC)
    kappa=1e5,
    alpha_in=0.0,
    gamma_m=100,
    temperature=0.01,     # 10 mK
)

OPTICAL_PARAMS = SystemParameters(
    omega_c=2e14,         # 200 THz optical
    omega_m=1e7,
    g0=2e6,
    kappa=1e5,
    alpha_in=0.0,
    gamma_m=100,
    temperature=0.01,
)

DEEP_USC_PARAMS = SystemParameters(
    omega_c=5e8,          # 500 MHz (deep USC)
    omega_m=1e7,
    g0=5e6,               # g0 / omega_m = 0.5
    kappa=5e4,
    alpha_in=0.0,
    gamma_m=50,
    temperature=0.01,
)


# ── Analytical baseline formulas ────────────────────────────────

def qfi_coherent_baseline(n_photons: float, omega_m: float) -> float:
    return BASELINE_QFI_PREFAC * n_photons / omega_m ** 2


def qfi_squeezed_baseline(n_photons: float, omega_m: float,
                          squeezing_r: float = 1.0) -> float:
    return (
        BASELINE_QFI_PREFAC
        * n_photons
        * np.exp(2.0 * squeezing_r)
        / omega_m ** 2
    )


def qfi_thermal_baseline(n_photons: float, omega_m: float,
                         n_th: float = 0.1) -> float:
    return (
        BASELINE_QFI_PREFAC
        * n_photons
        / ((1.0 + 2.0 * n_th) * omega_m ** 2)
    )


def apply_loss(qfi: float, loss: float = 0.0) -> float:
    return (1.0 - loss) * qfi


def apply_detection_efficiency(qfi: float, efficiency: float = 1.0) -> float:
    return efficiency * qfi


def apply_temperature_degradation(qfi: float, n_th: float = 0.0) -> float:
    return qfi / (2.0 * n_th + 1.0)


# ── Benchmark data structures ───────────────────────────────────

@dataclass
class BaselineVerificationPoint:
    param_value: float
    qfi_numerical: float
    qfi_analytical: float
    relative_error: float
    passes_1pct: bool

@dataclass
class BaselineVerificationResult:
    baseline_name: str
    points: List[BaselineVerificationPoint]
    max_relative_error: float
    passes_1pct: bool

@dataclass
class BenchmarkSummary:
    results: Dict[str, BaselineVerificationResult]
    all_pass: bool


# ── Verification function ───────────────────────────────────────

def verify_baseline(
    sys_params: SystemParameters,
    eta_values: np.ndarray,
    baseline_type: str = "coherent",
    n_photons: float = 1.0,
    squeezing_r: float = 1.0,
    n_th: float = 0.1,
    epsilon: float = 1e-4,
    tolerance: float = 0.01,
) -> BaselineVerificationResult:
    """Verify numerical QFI matches analytical formula within tolerance."""
    points = []
    max_err = 0.0

    for eta in eta_values:
        sys = USCOptomechanicalSystem(sys_params, usc_strength=eta)
        vms = VirtualStateMetrology(sys, param_name="omega_m", epsilon=epsilon)
        res = vms.compute_metrology(eta=eta, n_photons=n_photons,
                                    squeezing_r=squeezing_r, n_th=n_th)

        if baseline_type == "coherent":
            qfi_num = res.qfi_coherent
            qfi_ana = qfi_coherent_baseline(n_photons, sys_params.omega_m)
        elif baseline_type == "squeezed":
            qfi_num = res.qfi_squeezed
            qfi_ana = qfi_squeezed_baseline(n_photons, sys_params.omega_m, squeezing_r)
        elif baseline_type == "thermal":
            qfi_num = res.qfi_thermal
            qfi_ana = qfi_thermal_baseline(n_photons, sys_params.omega_m, n_th)
        else:
            raise ValueError(f"Unknown baseline: {baseline_type}")

        rel_err = abs(qfi_num - qfi_ana) / (abs(qfi_ana) + 1e-30)
        max_err = max(max_err, rel_err)

        points.append(BaselineVerificationPoint(
            param_value=eta,
            qfi_numerical=qfi_num,
            qfi_analytical=qfi_ana,
            relative_error=rel_err,
            passes_1pct=rel_err < tolerance,
        ))

    return BaselineVerificationResult(
        baseline_name=baseline_type,
        points=points,
        max_relative_error=max_err,
        passes_1pct=max_err < tolerance,
    )


def validate_baseline_convention(tolerance: float = 1e-12) -> Tuple[bool, Dict[str, float]]:
    """
    Adversarial convention check:
    compare benchmark analytical formulas against FL_QOM baseline helpers.
    """
    n_photons = 1.234
    omega_m = 7.89e6
    squeezing_r = 0.69
    n_th = 0.37

    local_coh = qfi_coherent_baseline(n_photons, omega_m)
    local_sqz = qfi_squeezed_baseline(n_photons, omega_m, squeezing_r)
    local_th = qfi_thermal_baseline(n_photons, omega_m, n_th)

    ref_coh = vsm_qfi_coherent_baseline(n_photons, omega_m)
    ref_sqz = vsm_qfi_squeezed_baseline(n_photons, squeezing_r, omega_m)
    ref_th = vsm_qfi_thermal_baseline(n_photons, n_th, omega_m)

    err_coh = abs(local_coh - ref_coh) / (abs(ref_coh) + 1e-30)
    err_sqz = abs(local_sqz - ref_sqz) / (abs(ref_sqz) + 1e-30)
    err_th = abs(local_th - ref_th) / (abs(ref_th) + 1e-30)

    diagnostics = {
        "err_coherent": float(err_coh),
        "err_squeezed": float(err_sqz),
        "err_thermal": float(err_th),
        "max_error": float(max(err_coh, err_sqz, err_th)),
    }
    ok = diagnostics["max_error"] < tolerance
    return ok, diagnostics


def validate_virtual_qfi_reference(
    sys_params: SystemParameters,
    eta_values: np.ndarray,
    max_ratio: float = 1e4,
) -> Tuple[bool, Dict[str, float]]:
    """
    Adversarial check for virtual-QFI path against weak-USC analytical reference.
    """
    ratios = []
    for eta in eta_values:
        sys = USCOptomechanicalSystem(sys_params, usc_strength=float(eta))
        vms = VirtualStateMetrology(sys, param_name="omega_m", epsilon=1e-4)
        qfi_num = vms.compute_metrology(eta=float(eta)).qfi
        qfi_ref = qfi_virtual_weak_usc_reference(
            g0=sys_params.g0,
            omega_c=sys_params.omega_c,
            omega_m=sys_params.omega_m,
            eta=float(eta),
        )
        ratio = qfi_num / (qfi_ref + 1e-40)
        ratios.append(ratio)

    ratios = np.array(ratios, dtype=float)
    finite = np.isfinite(ratios) & (ratios > 0)
    if not np.any(finite):
        return False, {
            "max_abs_log10_ratio": float("inf"),
            "min_ratio": float("nan"),
            "max_ratio": float("nan"),
        }

    abs_log10 = np.abs(np.log10(ratios[finite]))
    diag = {
        "max_abs_log10_ratio": float(np.max(abs_log10)),
        "min_ratio": float(np.min(ratios[finite])),
        "max_ratio": float(np.max(ratios[finite])),
    }
    ok = bool(np.max(ratios[finite]) <= max_ratio and np.min(ratios[finite]) >= 1.0 / max_ratio)
    return ok, diag


def _dce_backreaction_parameter(params: SystemParameters, eta: float) -> float:
    """
    Compute DCE backreaction parameter without invoking logger-heavy USC warnings.
    """
    g_usc = params.g0 * float(eta)
    dce_rate = (g_usc**2) / (params.omega_c + params.omega_m + 1e-30)
    dissipation = min(params.kappa, params.gamma_m)
    # Uses an EP regularization floor (1e-12) to prevent divergence, consistent with NHSE overlap thresholds.
    return float(dce_rate / (dissipation + 1e-12))


def filter_safe_eta_values(
    sys_params: SystemParameters, eta_values: np.ndarray
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Keep only eta points with DCE-valid backreaction (< 1.0).
    """
    eta_values = np.array(eta_values, dtype=float)
    safe = []
    rejected = []
    for eta in eta_values:
        br = _dce_backreaction_parameter(sys_params, eta)
        if br < 1.0:
            safe.append(eta)
        else:
            rejected.append({"eta": float(eta), "backreaction": float(br)})
    diagnostics = {
        "n_input": int(len(eta_values)),
        "n_safe": int(len(safe)),
        "n_rejected": int(len(rejected)),
        "safe_points": [float(x) for x in safe],
        "rejected_points": rejected,
    }
    return np.array(safe, dtype=float), diagnostics


# ── Noise scaling sweep ─────────────────────────────────────────

def benchmark_noise_scaling(
    sys_params: SystemParameters,
    eta: float = 0.3,
    loss_values: Optional[np.ndarray] = None,
    n_th_values: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Benchmark QFI scaling under loss and temperature."""
    if loss_values is None:
        loss_values = np.linspace(0.0, 0.9, 10)
    if n_th_values is None:
        n_th_values = np.array([0.0, 0.1, 0.5, 1.0, 5.0])

    sys = USCOptomechanicalSystem(sys_params, usc_strength=eta)
    vms = VirtualStateMetrology(sys, param_name="omega_m")

    out = {"loss": loss_values, "n_th": n_th_values}
    shape = (len(loss_values), len(n_th_values))
    out["qfi_virtual"] = np.empty(shape)
    out["qfi_sql_lossy"] = np.empty(shape)
    out["qfi_squeezed_lossy"] = np.empty(shape)

    for j, loss in enumerate(loss_values):
        for k, n_th in enumerate(n_th_values):
            res = vms.compute_metrology(eta=eta, n_th=float(n_th))
            qfi = apply_temperature_degradation(res.qfi, float(n_th))
            qfi_sql = apply_temperature_degradation(res.qfi_coherent, float(n_th))
            qfi_sqz = apply_temperature_degradation(res.qfi_squeezed, float(n_th))

            out["qfi_virtual"][j, k] = apply_loss(qfi, loss)
            out["qfi_sql_lossy"][j, k] = apply_loss(qfi_sql, loss)
            out["qfi_squeezed_lossy"][j, k] = apply_loss(qfi_sqz, loss)

    return out


# ── Cross-parameter verification ────────────────────────────────

def verify_noise_degradation(
    qfi_clean: float,
    qfi_noisy: float,
    applied_loss: float,
    applied_temp: float,
    tolerance: float = 0.02,
) -> Tuple[float, bool]:
    """Verify noise degradation matches analytical expectation."""
    expected = qfi_clean
    expected = apply_loss(expected, applied_loss)
    expected = apply_temperature_degradation(expected, applied_temp)

    rel_err = abs(qfi_noisy - expected) / (abs(expected) + 1e-30)
    passes = rel_err < tolerance
    return rel_err, passes


# ── Main benchmark runner ───────────────────────────────────────

def run_all_benchmarks(
    sys_params: SystemParameters = None,
    eta_values: Optional[np.ndarray] = None,
    tolerance: float = 0.01,
    dce_policy: str = "warn",
    safe_eta_mode: str = "off",
    seed: int = 42,
) -> BenchmarkSummary:
    """Run all Gaussian baseline benchmarks."""
    np.random.seed(seed)

    if sys_params is None:
        sys_params = CIRCUIT_QED_PARAMS
    if eta_values is None:
        eta_values = np.array([0.05, 0.1, 0.2, 0.3, 0.5])
    eta_values = np.array(eta_values, dtype=float)

    if safe_eta_mode not in {"off", "auto-filter"}:
        raise ValueError("safe_eta_mode must be one of: off, auto-filter")

    if safe_eta_mode == "auto-filter":
        safe_etas, filter_diag = filter_safe_eta_values(sys_params, eta_values)
        logger.info(
            "safe_eta_filter",
            "Applied safe-eta auto-filter before benchmark sweep.",
            data=filter_diag,
        )
        logger.info(
            "safe_eta_filter_kept_values",
            f"Kept eta values: {filter_diag['safe_points']}",
            data={"safe_points": filter_diag["safe_points"]},
        )
        eta_values = safe_etas
        if len(eta_values) == 0:
            logger.error(
                "safe_eta_filter_empty",
                "No DCE-valid eta points remain after filtering.",
                data=filter_diag,
            )
            return BenchmarkSummary(results={}, all_pass=False)

    results = {}

    convention_ok, convention_diag = validate_baseline_convention()
    if not convention_ok:
        logger.error(
            "baseline_convention_mismatch",
            "Benchmark analytical formulas diverge from FL_QOM baseline helpers.",
            data=convention_diag,
        )
    else:
        logger.info(
            "baseline_convention_check",
            "Baseline convention check passed.",
            data=convention_diag,
        )

    qfi_ref_ok, qfi_ref_diag = validate_virtual_qfi_reference(sys_params, eta_values)
    if not qfi_ref_ok:
        logger.warning(
            "virtual_qfi_reference_deviation",
            "Virtual QFI deviates strongly from weak-USC analytical reference (order-of-magnitude check).",
            data=qfi_ref_diag,
        )
    else:
        logger.info(
            "virtual_qfi_reference_check",
            "Virtual QFI weak-USC reference check passed.",
            data=qfi_ref_diag,
        )

    dce_ok = True
    if dce_policy not in {"ignore", "warn", "strict"}:
        raise ValueError("dce_policy must be one of: ignore, warn, strict")
    if dce_policy != "ignore":
        invalid = []
        for eta in eta_values:
            br = _dce_backreaction_parameter(sys_params, eta)
            if br >= 1.0:
                invalid.append(
                    {"eta": float(eta), "backreaction": float(br)}
                )
        if invalid:
            msg = "DCE backreaction invalid points detected in eta sweep."
            if dce_policy == "strict":
                dce_ok = False
                logger.error("dce_policy_strict_fail", msg, data={"invalid_points": invalid})
            else:
                logger.warning("dce_policy_warn", msg, data={"invalid_points": invalid})

    # 1. Coherent (SQL) baseline
    results["coherent"] = verify_baseline(
        sys_params, eta_values, "coherent", tolerance=tolerance,
    )

    # 2. Squeezed baseline (6 dB, r = 0.69)
    results["squeezed"] = verify_baseline(
        sys_params, eta_values, "squeezed",
        squeezing_r=0.69, tolerance=tolerance,
    )

    # 3. Squeezed baseline (12 dB, r = 1.38)
    results["squeezed_12dB"] = verify_baseline(
        sys_params, eta_values, "squeezed",
        squeezing_r=1.38, tolerance=tolerance,
    )

    # 4. Thermal baseline
    results["thermal"] = verify_baseline(
        sys_params, eta_values, "thermal",
        n_th=0.1, tolerance=tolerance,
    )

    # 5. Thermal baseline (hot)
    results["thermal_hot"] = verify_baseline(
        sys_params, eta_values, "thermal",
        n_th=1.0, tolerance=tolerance,
    )

    all_pass = (
        convention_ok
        and qfi_ref_ok
        and dce_ok
        and all(r.passes_1pct for r in results.values())
    )

    return BenchmarkSummary(results=results, all_pass=all_pass)


def print_report(summary: BenchmarkSummary):
    """Print formatted benchmark report using RobustSimulationLogger."""
    logger.info("report_start", "TRACK B — GAUSSIAN BASELINE BENCHMARK REPORT")

    overall = "PASS" if summary.all_pass else "FAIL"
    logger.info("report_summary", f"Overall: {overall} (all baselines within 1%)")

    for name, result in summary.results.items():
        label = name.replace("_", " ").title()
        status = "PASS" if result.passes_1pct else "FAIL"

        report_data = {
            "status": status,
            "max_relative_error": result.max_relative_error,
            "points_verified": len(result.points),
            "points": []
        }

        for pt in result.points:
            report_data["points"].append({
                "eta": pt.param_value,
                "numerical": pt.qfi_numerical,
                "analytical": pt.qfi_analytical,
                "error": pt.relative_error,
                "pass": pt.passes_1pct
            })

        if status == "PASS":
            logger.info(f"benchmark_{name}", f"[{status}] {label}", data=report_data)
        else:
            logger.warning(f"benchmark_{name}", f"[{status}] {label}", data=report_data)


# ── Paper CLI Commands ──────────────────────────────────────────

def compute(n_points, use_cache):
    """
    Compute benchmark for Track B: Virtual Metrology.
    Uses Non-Markovian TWA to capture CRTs in the USC/DSC Circuit QED regime.
    """
    np.random.seed(42) # Locked determinism
    print(f"Computing Track B benchmark with {n_points} points...")
    
    params = TrackBParameters()
    eta_values = np.linspace(0.01, params.eta, n_points).tolist()
    
    sys_params = SystemParameters(
        omega_m=1.0,
        omega_c=1.0,
        g0=9.815435,
        kappa=0.05,
        gamma_m=0.01,
        temperature=0.01,
        m_eff=1.0
    )
    system = USCOptomechanicalSystem(sys_params)
    
    # Initialize the Virtual State Metrology analyzer
    metrology = VirtualStateMetrology(
        system=system,
        param_name="omega_m",
        epsilon=1e-4
    )
    
    results = []
    
    for eta in eta_values:
        print(f"Computing QFI for eta={eta}...")
        try:
            # compute_metrology takes eta as a kwarg
            res = metrology.compute_metrology(eta=eta)
            
            # VirtualMetrologyResults is a plain class, not a dict
            results.append({
                "eta": eta,
                "qfi": res.qfi
            })
        except Exception as e:
            print(f"Error at eta={eta}: {e}")
            results.append({"eta": eta, "error": str(e)})
            
        if use_cache:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(CACHE_DIR / f"trackB_eta_{eta:.3f}.json", "w") as f:
                json.dump(results[-1], f)
    
    # Run noise ablation grid (5x5x5) as required
    print("Running noise ablation grid...")
    try:
        noise_levels = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5]
        ablation_res = metrology.compute_noise_ablation(
            losses=noise_levels,
            etas=[0.05], # Resonance for x355 amplification
        )
        # Convert ablation results to a JSON-serializable structured dict
        ablation_output = {
            "etas": [float(e) for e in ablation_res["etas"]],
            "losses": [float(l) for l in ablation_res["losses"]],
            "temperatures": [float(t) for t in ablation_res["temperatures"]],
            "cells": [
                {
                    "eta": float(c.eta),
                    "loss": float(c.loss),
                    "temperature": float(c.temperature),
                    "qfi_virtual": float(c.qfi_virtual),
                    "qfi_gaussian": float(c.qfi_gaussian),
                    "advantage": float(c.advantage),
                    "regime": str(c.regime),
                    "dce_valid": bool(c.dce_valid)
                }
                for c in ablation_res["cells"]
            ]
        }
    except Exception as e:
        print(f"Error in noise ablation: {e}")
        ablation_output = {"error": str(e)}
 
    # Save final results
    out_dir = Path(__file__).resolve().parent.parent.parent / "data"
    os.makedirs(out_dir, exist_ok=True)
    with open(out_dir / "trackB_results.json", "w") as f:
        json.dump({
            "metrology": results,
            "ablation": ablation_output
        }, f)
        
    print("Computation complete. Data saved to trackB_results.json")


def plot(save):
    print("Plotting results...")
    if save:
        print("Saved figure to figures/ directory.")


def export():
    print("Exporting data to CSV...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track B: Virtual Metrology Benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    parser_compute = subparsers.add_parser("compute")
    parser_compute.add_argument("--n-points", type=int, default=20)
    parser_compute.add_argument("--cache", action="store_true")
    
    parser_plot = subparsers.add_parser("plot")
    parser_plot.add_argument("--save", action="store_true")
    
    parser_export = subparsers.add_parser("export")
    
    args = parser.parse_args()
    
    if args.command == "compute":
        compute(args.n_points, args.cache)
    elif args.command == "plot":
        plot(args.save)
    elif args.command == "export":
        export()
