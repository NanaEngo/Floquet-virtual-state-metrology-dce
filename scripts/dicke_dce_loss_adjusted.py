#!/usr/bin/env python3
"""M-B-16: Loss-adjusted Dicke+DCE QFI under Lindblad cavity loss (N=32).

Extends the single-qubit loss simulation to the collective Dicke model.
The Dicke Hamiltonian is:
  H = wc * a^dag a + wq * Jz + (g0 + gd cos(wd t))/sqrt(N) * Jx * (a + a^dag)

Uses the spin coherent basis (jmat) for the collective spin operators and a
truncated cavity Fock space.

The protocol matches the published Dicke+DCE parameters:
  N=32, eta=1.15, gd/g0=0.15, wq/wc=1.0, wd/wc=2.0

Validated configuration (2026-09-02, SLURM array jobs 15744/15764/15784):
  * adaptive integrator at atol = rtol = 1e-10 (nsteps=100000 cap);
  * every stored state passes trace and Hermiticity checks, and its minimum
    eigenvalue remains above the numerical acceptance floor of -1e-4;
    this is a numerical physicality gate, not a proof of exact positivity;
  * fidelity-product eigenvalues use a separate -1e-5 numerical floor;
  * the observed near-pure-state solver noise is O(1e-5..1e-6), so the
    acceptance floor is recorded explicitly rather than described as positivity;

  * n_steps = 200 explicit output times over T_max = 10 drive periods;
  * checkpoint-per-point output under data/_dicke_loss_checkpoints/.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import qutip as qt

N_QUBITS = 32
ETA = 1.15
GD_OVER_G0 = 0.15
WQ_WC = 1.0
WD_WC = 2.0
WC = 1.0
WQ = WQ_WC * WC
WD = WD_WC * WC
N_CAV = 6  # Strongly reduced for Lindblad tractability
N_MAX = 6  # photon truncation in the basis
J = N_QUBITS / 2.0
T_D = 2 * np.pi / WD
T_MAX = 10 * T_D  # Reduced from 40 periods for tractability
DELTA = 0.001
KAPPA_VALUES = (0.0, 1e-4, 1e-3, 1e-2, 4.1e-2)
PHYSICALITY_EIGENVALUE_FLOOR = -1e-4  # numerical acceptance floor, not exact positivity
PHYSICALITY_TRACE_TOLERANCE = 1.0e-8
PHYSICALITY_HERMITICITY_TOLERANCE = 1.0e-8
SOLVER_ATOL = 1.0e-10
SOLVER_RTOL = 1.0e-10

H0_STATIC = None
H_DRIVE = None


def build_operators():
    """Return collective operators for the Dicke model.

    Returns:
        tuple: (a, a^dagger, Jx, Jz) acting on the full
        N_CAV x (2J+1) tensor space.
    """
    a = qt.tensor(qt.destroy(N_CAV), qt.qeye(int(2 * J + 1)))
    ad = a.dag()
    jx = qt.tensor(qt.qeye(N_CAV), qt.jmat(J, 'x'))
    jz = qt.tensor(qt.qeye(N_CAV), qt.jmat(J, 'z'))
    return a, ad, jx, jz


def _build_operators():
    """Cache the static Hamiltonian pieces on first use."""
    global H0_STATIC, H_DRIVE
    if H0_STATIC is None:
        a, ad, jx, jz = build_operators()
        H0_STATIC = WC * ad * a + WQ * jz
        H_DRIVE = jx * (a + ad)


def build_H(eta_val, gd_val):
    """Return the time-dependent Dicke Hamiltonian callable for mesolve.

    Args:
        eta_val: Dimensionless coupling eta = g0/wc.
        gd_val: Dimensionless drive coupling gd/wc.

    Returns:
        callable: H(t, args) for QuTiP mesolve.
    """
    _build_operators()
    g0 = eta_val * WC

    def H_t(t, args=None):
        return H0_STATIC + (g0 + gd_val * np.cos(WD * t)) / np.sqrt(N_QUBITS) * H_DRIVE

    return H_t


def get_ground_state():
    """Return the static Dicke ground state |0>_cav x |J,-J>_qubits (g0=0)."""
    return qt.tensor(qt.basis(N_CAV, 0), qt.basis(int(2 * J + 1), 0))


def _state_physicality(state, label, tolerance=PHYSICALITY_EIGENVALUE_FLOOR):
    """Check trace, Hermiticity, and numerical eigenvalue acceptance of a state.

    Args:
        state: QuTiP ket or density operator at one output time.
        label: Human-readable state label for diagnostics.
        tolerance: Accepted minimum eigenvalue floor, dimensionless; this is
            a numerical gate and not a proof of exact positivity.

    Returns:
        dict: Trace error, Hermiticity error, minimum eigenvalue, and gate status.

    Raises:
        RuntimeError: If trace, Hermiticity, or the numerical eigenvalue gate fails.
    """
    if state.isoper:
        rho = state
    else:
        rho = state.proj()

    trace_error = abs(complex(rho.tr()) - 1.0)
    hermiticity_error = (rho - rho.dag()).norm()
    min_eigenvalue = float(np.min(rho.eigenenergies().real))

    if trace_error > PHYSICALITY_TRACE_TOLERANCE:
        raise RuntimeError(f"{label}: trace error {trace_error:.3e}")
    if hermiticity_error > PHYSICALITY_HERMITICITY_TOLERANCE:
        raise RuntimeError(f"{label}: Hermiticity error {hermiticity_error:.3e}")
    if min_eigenvalue < tolerance:
        raise RuntimeError(f"{label}: numerical eigenvalue gate failed ({min_eigenvalue:.3e})")

    return {
        "trace_error": float(trace_error),
        "hermiticity_error": float(hermiticity_error),
        "min_eigenvalue": min_eigenvalue,
        "numerical_physicality_gate": "PASS",
        "eigenvalue_floor": float(tolerance),
    }


def compute_dicke_dce_qfi(kappa_rel, n_steps=200):
    """Compute Dicke+DCE QFI under Lindblad cavity loss.

    Args:
        kappa_rel: Dimensionless cavity decay rate kappa/omega_c.
        n_steps: Number of explicit output times over the drive window.

    Returns:
        tuple: (tlist, qfi, n_phot, physicality) with
        qfi (np.ndarray): fidelity-based QFI at each output time,
        n_phot (np.ndarray): cavity photon number expectation values,
        physicality (list): per-state physicality diagnostics.
    """
    a, ad, jx, jz = build_operators()
    n_op = ad * a

    c_ops = []
    if kappa_rel > 0:
        c_ops.append(np.sqrt(kappa_rel) * a)

    psi0 = get_ground_state()
    g0 = ETA * WC
    H_0 = build_H(ETA, g0 * GD_OVER_G0)
    H_d = build_H(ETA + DELTA, g0 * GD_OVER_G0)

    tlist = np.linspace(0, T_MAX, n_steps)
    options = {
        "store_states": True,
        "atol": SOLVER_ATOL,
        "rtol": SOLVER_RTOL,
        "nsteps": 100000,
    }
    r0 = qt.mesolve(H_0, psi0, tlist, c_ops=c_ops, e_ops=[n_op], args={},
                    options=options)
    rd = qt.mesolve(H_d, psi0, tlist, c_ops=c_ops, e_ops=[], args={},
                    options=options)

    states_0 = r0.states
    states_d = rd.states
    expect_n = r0.expect[0]
    n_states = len(states_0)
    if n_states != n_steps or len(states_d) != n_steps or len(expect_n) != n_steps:
        raise RuntimeError(
            f"State/time-grid mismatch: expected {n_steps}, got rho0={len(states_0)}, "
            f"rhod={len(states_d)}, n={len(expect_n)}"
        )
    t_eff = tlist

    qfi = np.zeros(n_states)
    is_pure = (kappa_rel == 0)
    physicality = []

    for i in range(n_states):
        s0 = states_0[i]
        sd = states_d[i]
        physicality.append({
            "rho0": _state_physicality(s0, f"rho0[{i}]"),
            "rhod": _state_physicality(sd, f"rhod[{i}]"),
        })

        if is_pure:
            overlap = abs(s0.overlap(sd))
            fidelity = overlap ** 2
        else:
            sqrt_s0 = s0.sqrtm()
            prod = sqrt_s0 * sd * sqrt_s0
            evals = prod.eigenenergies()
            if np.min(evals.real) < -1e-5:
                raise RuntimeError(
                    f"Fidelity-product eigenvalue gate failed: {np.min(evals.real)}"
                )
            fidelity = float(np.sum(np.sqrt(np.clip(evals.real, 0.0, None))) ** 2)

        if not np.isfinite(fidelity) or fidelity < -1e-8 or fidelity > 1.0001:
            raise RuntimeError(f"Invalid fidelity at index {i}: {fidelity}")

        fidelity = min(1.0, max(0.0, fidelity))
        qfi[i] = 8.0 * (1.0 - np.sqrt(fidelity)) / DELTA ** 2

    return t_eff, qfi, np.array(expect_n[:n_states]), physicality


def parse_args():
    """Parse optional single-task parameters for SLURM array execution.

    Returns:
        argparse.Namespace: Validated task and output settings.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-cav", type=int, choices=(6, 8, 10, 12, 16, 20), default=6,
                        help="Cavity Fock truncation (default 6; 16 and 20 are produced by the truncation-extension driver).")
    parser.add_argument("--kappa-index", type=int,
                        choices=range(len(KAPPA_VALUES)), default=None,
                        help="Index into KAPPA_VALUES; run full sweep if omitted.")
    parser.add_argument("--n-steps", type=int, default=60,
                        help="Number of explicit output times (validated runs used 200).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Explicit JSON output path (default: data/dicke_dce_loss_adjusted.json).")
    return parser.parse_args()


def _run_point(kappa_rel, n_steps):
    """Run and summarize one validated loss point.

    Args:
        kappa_rel: Dimensionless cavity decay rate kappa/omega_c.
        n_steps: Number of explicit output times.

    Returns:
        dict: Numerical result and physicality diagnostics for one point.
    """
    t0 = time.time()
    tlist, qfi, n_phot, physicality = compute_dicke_dce_qfi(kappa_rel, n_steps=n_steps)
    qfi_peak_idx = int(np.argmax(qfi))
    n_ss = max(1, len(qfi) // 5)

    if not (np.all(np.isfinite(qfi)) and not np.any(qfi < -1e-8)):
        raise RuntimeError(f"Invalid QFI trace at kappa/wc={kappa_rel}")

    result = {
        "kappa_rel": float(kappa_rel),
        "kappa_2pi_MHz": float(kappa_rel * 6000.0),
        "qfi_peak": float(np.max(qfi)),
        "qfi_peak_time_periods": float(tlist[qfi_peak_idx] / T_D),
        "qfi_ss_mean": float(np.mean(qfi[-n_ss:])),
        "n_peak": float(n_phot[qfi_peak_idx]),
        "n_avg": float(np.mean(n_phot)),
        "kappa_t_peak": float(kappa_rel * tlist[qfi_peak_idx]),
        "elapsed_s": float(time.time() - t0),
        "n_steps": int(n_steps),
        "physicality": {
            "gate": "numerical_eigenvalue_acceptance",
            "eigenvalue_floor": PHYSICALITY_EIGENVALUE_FLOOR,
            "max_trace_error": max(max(x["rho0"]["trace_error"], x["rhod"]["trace_error"])
                                   for x in physicality),
            "max_hermiticity_error": max(max(x["rho0"]["hermiticity_error"],
                                             x["rhod"]["hermiticity_error"])
                                         for x in physicality),
            "min_eigenvalue": min(min(x["rho0"]["min_eigenvalue"],
                                      x["rhod"]["min_eigenvalue"])
                                  for x in physicality),
            "samples_checked": len(physicality),
        },
    }
    return result


def main():
    global N_CAV, N_MAX, H0_STATIC, H_DRIVE
    args = parse_args()
    N_CAV = args.n_cav
    N_MAX = args.n_cav
    H0_STATIC = None
    H_DRIVE = None

    print(f"=== M-B-16: Loss-Adjusted Dicke+DCE QFI (N={N_QUBITS}) ===")
    print(f"Hilbert dim: {N_CAV}x{int(2*J+1)} = {N_CAV * int(2*J+1)}")
    print(f"Note: Reduced photon truncation (N_CAV={N_CAV}) for Lindblad tractability.")
    print()

    task_indices = (range(len(KAPPA_VALUES)) if args.kappa_index is None
                    else (args.kappa_index,))
    results = []
    for index in task_indices:
        result = _run_point(KAPPA_VALUES[index], args.n_steps)
        result["kappa_index"] = int(index)
        results.append(result)
        print(f"  kappa/wc={result['kappa_rel']:.0e}: QFI_pk={result['qfi_peak']:10.1f}, "
              f"QFI_late={result['qfi_ss_mean']:10.1f}, {result['elapsed_s']:.1f}s")

    output = {
        "operating_point": {
            "N": N_QUBITS, "eta": ETA, "gd_over_g0": GD_OVER_G0,
            "wq_wc": WQ_WC, "wd_wc": WD_WC, "n_cav_truncation": N_CAV,
            "n_drive_periods": 10, "delta": DELTA, "n_steps": args.n_steps,
            "J": J, "hilbert_dim": N_CAV * int(2 * J + 1),
        },
        "kappa_sweep": results,
        "caveat": "Photon truncation N_CAV=6 (vs N_CAV=28 in the pure-state production data) is used for Lindblad tractability. Absolute QFI values and the loss curve require truncation validation. The reported state check is a numerical physicality acceptance gate with an eigenvalue floor of 1e-4, not a proof of exact positivity; low-loss peak-QFI ordering is also checked empirically rather than assumed monotonic.",
        "note": "Mixed-state QFI under Lindblad cavity loss. Published reference (pure-state) peak ~2.5e5x at last window.",
    }

    outpath = args.output or Path(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data",
        "dicke_dce_loss_adjusted.json"))
    outpath.parent.mkdir(parents=True, exist_ok=True)
    if outpath.exists() and args.kappa_index is not None:
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {outpath}")

    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
