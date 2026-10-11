#!/usr/bin/env python3
"""
M-B-01: DCE QFI time-step convergence test.

Runs the single-qubit DCE simulation at the published operating point
(eta=0.150, gd/g0=0.11923, wq_wc=1.027, wd_wc=2.077) with varying
numbers of time steps and finite-difference steps.

Produces: dce_time_step_convergence.json
"""

import json
import time
import numpy as np
import qutip as qt

# === Operating point ===
ETA = 0.150
GD = 0.11923  # modulation amplitude (same convention as scan script)
WQ_WC = 1.027
WD_WC = 2.077
WC = 1.0
WQ = WQ_WC * WC
WD = WD_WC * WC
N_CAV = 20
T_D = 2 * np.pi / WD
T_MAX = 40 * T_D
DELTA_DEFAULT = 0.001


def _ops():
    """Return tensor operators for the joint qubit-cavity space."""
    a = qt.tensor(qt.destroy(N_CAV), qt.qeye(2))
    ad = a.dag()
    sx = qt.tensor(qt.qeye(N_CAV), qt.sigmax())
    sz = qt.tensor(qt.qeye(N_CAV), qt.sigmaz())
    return a, ad, sx, sz


def _build_H0():
    """Build static part of Hamiltonian (independent of eta)."""
    a, ad, sx, sz = _ops()
    return 0.5 * WQ * sz + WC * ad * a


def _build_H_drive():
    """Build the drive coupling operator."""
    a, ad, sx, sz = _ops()
    return sx * (a + ad)


H0_STATIC = None
H_DRIVE = None


def build_H(eta_val):
    """Build time-dependent Hamiltonian matching original script format."""
    global H0_STATIC, H_DRIVE
    if H0_STATIC is None:
        H0_STATIC = _build_H0()
        H_DRIVE = _build_H_drive()

    def H_t(t, args=None):
        return H0_STATIC + (eta_val + GD * np.cos(WD * t)) * H_DRIVE

    return H_t


def get_ground_state():
    """Get bare ground state |0_cav> ⊗ |0_qubit> (as used in original DCE scripts)."""
    return qt.tensor(qt.basis(N_CAV, 0), qt.basis(2, 0))


def compute_qfi_trace(eta_val, delta, n_steps):
    """Compute the fidelity-based QFI trace."""
    psi0 = get_ground_state()
    H_0 = build_H(eta_val)
    H_d = build_H(eta_val + delta)

    tlist = np.linspace(0, T_MAX, n_steps)
    result_0 = qt.mesolve(H_0, psi0, tlist, c_ops=[], e_ops=[], args={})
    result_d = qt.mesolve(H_d, psi0, tlist, c_ops=[], e_ops=[], args={})

    qfi_trace = np.zeros(len(tlist))
    for i in range(len(tlist)):
        overlap = abs(result_0.states[i].overlap(result_d.states[i]))
        qfi_trace[i] = 8.0 * (1.0 - overlap) / delta**2

    return tlist, qfi_trace


def run_dt_convergence():
    """Test convergence with respect to time step (n_steps)."""
    step_counts = [100, 200, 500, 1000, 2000]
    results = []

    for n_steps in step_counts:
        t0 = time.time()
        tlist, qfi = compute_qfi_trace(ETA, DELTA_DEFAULT, n_steps)
        elapsed = time.time() - t0
        peak_qfi = np.max(qfi)
        peak_idx = np.argmax(qfi)
        peak_time = tlist[peak_idx]

        n_ss = max(1, len(qfi) // 5)
        ss_mean = np.mean(qfi[-n_ss:])

        results.append({
            "n_steps": n_steps,
            "dt_over_Td": (T_MAX / T_D) / n_steps,
            "peak_qfi": float(peak_qfi),
            "peak_time_periods": float(peak_time / T_D),
            "ss_mean_qfi": float(ss_mean),
            "elapsed_s": float(elapsed),
        })
        print(f"  n_steps={n_steps:5d}: peak={peak_qfi:.1f}, ss_mean={ss_mean:.1f}, dt/Td={results[-1]['dt_over_Td']:.4f}, {elapsed:.1f}s")

    return results


def run_delta_convergence():
    """Test convergence with respect to finite-difference step delta."""
    deltas = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    n_steps = 500
    results = []

    for delta in deltas:
        t0 = time.time()
        tlist, qfi = compute_qfi_trace(ETA, delta, n_steps)
        elapsed = time.time() - t0
        peak_qfi = np.max(qfi)

        results.append({
            "delta": delta,
            "peak_qfi": float(peak_qfi),
            "elapsed_s": float(elapsed),
        })
        print(f"  delta={delta:.0e}: peak={peak_qfi:.1f}, {elapsed:.1f}s")

    return results


def main():
    print("=== M-B-01: DCE QFI Time-Step Convergence ===")
    print(f"Operating point: eta={ETA}, gd={GD}, wq_wc={WQ_WC}, wd_wc={WD_WC}")
    print(f"Fock space: N_cav={N_CAV}, Hilbert dim={2*N_CAV}")
    print(f"Drive periods: 40, T_max={T_MAX:.4f}")
    print()

    print("--- dt convergence (delta=0.001) ---")
    dt_results = run_dt_convergence()
    print()

    print("--- delta convergence (n_steps=500) ---")
    delta_results = run_delta_convergence()
    print()

    output = {
        "operating_point": {
            "eta": ETA, "gd": GD, "wq_wc": WQ_WC,
            "wd_wc": WD_WC, "n_cav": N_CAV, "n_drive_periods": 40,
        },
        "dt_convergence": dt_results,
        "delta_convergence": delta_results,
        "reference_peak_qfi": 170903.33,
        "note": "Peak QFI from dce_optimum_final.json (2D scan maximum). This script tests convergence at the rounded operating point (gd=0.119 vs 0.11923).",
    }

    outpath = "Q1_Papers/manuscripts/track_B_metrology/data/dce_time_step_convergence.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {outpath}")


if __name__ == "__main__":
    main()
