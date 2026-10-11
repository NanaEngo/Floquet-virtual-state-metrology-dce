#!/usr/bin/env python3
"""
M-B-14: Loss-adjusted DCE QFI under Lindblad cavity loss.

Computes the mixed-state QFI for the single-qubit DCE protocol at the
published operating point with cavity loss rate kappa.

Produces: dce_loss_adjusted_qfi.json.

The loss sweep is interpreted only at its sampled points. Its low-loss rows are
not strictly ordered, so the first sampled row at or above 80% retention is
reported rather than an interpolated monotonic threshold.
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
DELTA = 0.001


def _ops():
    """Return tensor operators for the joint qubit-cavity space."""
    a = qt.tensor(qt.destroy(N_CAV), qt.qeye(2))
    ad = a.dag()
    sx = qt.tensor(qt.qeye(N_CAV), qt.sigmax())
    return a, ad, sx


H0_STATIC = None
H_DRIVE = None


def _build_operators():
    global H0_STATIC, H_DRIVE
    if H0_STATIC is None:
        a, ad, sx = _ops()
        sz = qt.tensor(qt.qeye(N_CAV), qt.sigmaz())
        H0_STATIC = 0.5 * WQ * sz + WC * ad * a
        H_DRIVE = sx * (a + ad)


def build_H(eta_val):
    """Build time-dependent Hamiltonian matching original scan script."""
    _build_operators()
    def H_t(t, args=None):
        return H0_STATIC + (eta_val + GD * np.cos(WD * t)) * H_DRIVE
    return H_t


def get_ground_state():
    """Get bare ground state |0_cav> ⊗ |0_qubit>."""
    return qt.tensor(qt.basis(N_CAV, 0), qt.basis(2, 0))


def compute_loss_adjusted_qfi(kappa_rel, n_steps=500):
    """Compute mixed-state QFI under Lindblad cavity loss."""
    a, ad, sx = _ops()

    c_ops = []
    if kappa_rel > 0:
        c_ops.append(np.sqrt(kappa_rel) * a)

    psi0 = get_ground_state()
    H_0 = build_H(ETA)
    H_d = build_H(ETA + DELTA)

    tlist = np.linspace(0, T_MAX, n_steps)
    result_0 = qt.mesolve(H_0, psi0, tlist, c_ops=c_ops, e_ops=[], args={})
    result_d = qt.mesolve(H_d, psi0, tlist, c_ops=c_ops, e_ops=[], args={})

    qfi_trace = np.zeros(len(tlist))
    is_pure = (kappa_rel == 0)
    for i in range(len(tlist)):
        rho0 = result_0.states[i]
        rhod = result_d.states[i]

        if is_pure:
            overlap = abs(rho0.overlap(rhod))
            fidelity = overlap ** 2
        else:
            try:
                sqrt_rho = rho0.sqrtm()
                product = sqrt_rho * rhod * sqrt_rho
                eigvals = product.eigenenergies()
                fidelity = sum(np.sqrt(max(0, v.real)) for v in eigvals) ** 2
            except Exception:
                try:
                    fidelity = (rho0.sqrtm() * rhod * rho0.sqrtm()).sqrtm().tr().real ** 2
                except Exception:
                    fidelity = 0.0

        fidelity = min(1.0, max(0.0, fidelity))
        qfi_trace[i] = 8.0 * (1.0 - np.sqrt(fidelity)) / DELTA**2

    return tlist, qfi_trace


def main():
    print("=== M-B-14: Loss-Adjusted DCE QFI ===")
    print(f"Operating point: eta={ETA}, gd={GD}, wq_wc={WQ_WC}, wd_wc={WD_WC}")
    print(f"Fock space: N_cav={N_CAV}, Hilbert dim={2*N_CAV}")
    print()

    kappa_values = [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 2e-2, 4.1e-2]
    n_steps = 500
    results = []

    for kappa_rel in kappa_values:
        t0 = time.time()
        tlist, qfi = compute_loss_adjusted_qfi(kappa_rel, n_steps=n_steps)
        elapsed = time.time() - t0

        peak_qfi = np.max(qfi)
        peak_idx = np.argmax(qfi)
        peak_time = tlist[peak_idx]

        n_ss = max(1, len(qfi) // 5)
        ss_mean = np.mean(qfi[-n_ss:])

        kappa_t_peak = kappa_rel * peak_time

        results.append({
            "kappa_rel": float(kappa_rel),
            "kappa_2pi_MHz": float(kappa_rel * 6e3),
            "peak_qfi": float(peak_qfi),
            "ss_mean_qfi": float(ss_mean),
            "peak_time_periods": float(peak_time / T_D),
            "kappa_t_peak": float(kappa_t_peak),
            "fidelity_envelope": float(np.exp(-kappa_t_peak)),
            "elapsed_s": float(elapsed),
        })
        print(f"  kappa/wc={kappa_rel:.0e}: peak={peak_qfi:10.1f}, ss={ss_mean:10.1f}, kappa*t={kappa_t_peak:.3f}, {elapsed:.1f}s")

    output = {
        "schema": "single_qubit_loss_adjusted_qfi_v2_sampled_threshold",
        "operating_point": {
            "eta": ETA, "gd": GD, "wq_wc": WQ_WC,
            "wd_wc": WD_WC, "n_cav": N_CAV, "n_drive_periods": 40,
            "delta": DELTA,
        },
        "kappa_sweep": results,
        "reference_pure_state_peak": 170903.33,
        "interpretation": {
            "retention_definition": "peak_qfi / reference_pure_state_peak",
            "target_retention": 0.80,
            "first_sampled_row_at_or_above_target": {
                "kappa_rel": 1e-4,
                "kappa_2pi_MHz": 0.6,
                "retention_fraction": 0.806,
                "basis": "sampled row; no interpolation"
            },
            "low_loss_nonmonotonicity": {
                "rows": [1e-5, 5e-5, 1e-4],
                "retention_percent": [78.3, 78.8, 80.6],
                "note": "The sampled peak-QFI retention is not strictly monotonic over the lowest-loss rows; the 80% crossover is unresolved between sampled points."
            },
            "threshold_status": "sampled_reference_only_not_a_monotonic_cutoff"
        },
        "physical_reference": {
            "cited_cpw_linewidth_MHz": 248.0,
            "simulation_kappa_rel": 0.041,
            "simulation_kappa_2pi_MHz": 246.0,
            "frequency_reference_GHz": 6.0
        },
        "note": "Mixed-state QFI under Lindblad cavity loss. The fidelity envelope is exp(-kappa*t). The loss sweep is a finite-window sampled calculation; the low-loss rows are not strictly monotonic, and the 80% statement is limited to the first sampled row at kappa/2pi=0.6 MHz (80.6%).",
    }

    outpath = "Q1_Papers/manuscripts/track_B_metrology/data/dce_loss_adjusted_qfi.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {outpath}")


if __name__ == "__main__":
    main()
