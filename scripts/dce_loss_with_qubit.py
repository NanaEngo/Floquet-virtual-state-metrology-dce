#!/usr/bin/env python3
"""
M-B-18: Loss-adjusted DCE QFI with combined cavity + qubit dissipation.

Extends the single-qubit Lindblad analysis to include qubit relaxation
(T1) and pure dephasing (T_phi) in addition to cavity loss (kappa).

This addresses the editor's M-1: the original loss analysis only
included cavity loss, but physical USC platforms have qubit T1 in the
10-100 us range and T_phi in the 1-10 us range at the relevant operating
point. Adding qubit dissipation is the missing piece.
"""

import argparse
import json
import os
import time
import numpy as np
import qutip as qt

from config import DEFAULT_QUBIT_COHERENCE

ETA = 0.150
GD = 0.11923
WQ_WC = 1.027
WD_WC = 2.077
WC = 1.0
WQ = WQ_WC * WC
WD = WD_WC * WC
N_CAV = 20
T_D = 2 * np.pi / WD
T_MAX = 40 * T_D
DELTA = 0.001
COHERENCE = DEFAULT_QUBIT_COHERENCE
GAMMA_T1, GAMMA_TPHI = COHERENCE.simulation_rates()


def _ops():
    a = qt.tensor(qt.destroy(N_CAV), qt.qeye(2))
    ad = a.dag()
    sx = qt.tensor(qt.qeye(N_CAV), qt.sigmax())
    sz = qt.tensor(qt.qeye(N_CAV), qt.sigmaz())
    sm = qt.tensor(qt.qeye(N_CAV), qt.sigmam())
    sp = qt.tensor(qt.qeye(N_CAV), qt.sigmap())
    return a, ad, sx, sz, sm, sp


H0_STATIC = None
H_DRIVE = None


def _build_operators():
    global H0_STATIC, H_DRIVE
    if H0_STATIC is None:
        a, ad, sx, sz, sm, sp = _ops()
        H0_STATIC = 0.5 * WQ * sz + WC * ad * a
        H_DRIVE = sx * (a + ad)


def build_H(eta_val):
    _build_operators()
    def H_t(t, args=None):
        return H0_STATIC + (eta_val + GD * np.cos(WD * t)) * H_DRIVE
    return H_t


def get_ground_state():
    return qt.tensor(qt.basis(N_CAV, 0), qt.basis(2, 0))


def compute_qfi_with_qubit_loss(kappa_rel, gamma_rel, gamma_phi_rel, n_steps=500):
    """QFI under combined cavity loss + qubit T1 + pure dephasing.

    gamma_rel = qubit T1 rate / wc
    gamma_phi_rel = pure dephasing rate / wc
    """
    a, ad, sx, sz, sm, sp = _ops()

    c_ops = []
    if kappa_rel > 0:
        c_ops.append(np.sqrt(kappa_rel) * a)
    if gamma_rel > 0:
        c_ops.append(np.sqrt(gamma_rel) * sm)
    if gamma_phi_rel > 0:
        c_ops.append(np.sqrt(gamma_phi_rel / 2) * sz)

    psi0 = get_ground_state()
    H_0 = build_H(ETA)
    H_d = build_H(ETA + DELTA)

    tlist = np.linspace(0, T_MAX, n_steps)
    r0 = qt.mesolve(H_0, psi0, tlist, c_ops=c_ops, e_ops=[], args={})
    rd = qt.mesolve(H_d, psi0, tlist, c_ops=c_ops, e_ops=[], args={})

    qfi = np.zeros(n_steps)
    is_pure = (kappa_rel == 0 and gamma_rel == 0 and gamma_phi_rel == 0)
    for i in range(n_steps):
        rho0 = r0.states[i]
        rhod = rd.states[i]

        if is_pure:
            overlap = abs(rho0.overlap(rhod))
            fidelity = overlap ** 2
        else:
            try:
                sqrt_rho = rho0.sqrtm()
                product = sqrt_rho * rhod * sqrt_rho
                evals = product.eigenenergies()
                fidelity = sum(np.sqrt(max(0, v.real)) for v in evals) ** 2
            except Exception:
                try:
                    fidelity = (rho0.sqrtm() * rhod * rho0.sqrtm()).sqrtm().tr().real ** 2
                except Exception:
                    fidelity = 0.0

        fidelity = min(1.0, max(0.0, fidelity))
        qfi[i] = 8.0 * (1.0 - np.sqrt(fidelity)) / DELTA ** 2

    return tlist, qfi


def parse_args():
    """Parse command-line options for the qubit-loss producer.

    Returns:
        argparse.Namespace: Parsed options. The producer remains backward
            compatible: without options it executes the retained benchmark
            calculation and writes the JSON artifact.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args()


def main():
    """Run the retained single-qubit qubit-loss stress-test calculation.

    Returns:
        None: The producer writes its JSON artifact and prints a summary.
    """
    parse_args()
    print("=== M-B-18: Loss-Adjusted DCE QFI (cavity + qubit dissipation) ===")
    print(f"Operating point: eta={ETA}, gd={GD}, wq/wc={WQ_WC}, wd/wc={WD_WC}")
    print(f"wc/2pi = 6 GHz reference; t_final = 40*T_d ~ 121 in wc^-1 units")
    print()

    # The simulation uses a deliberately conservative coherence benchmark,
    # not a claim about current record transmon performance. The active rates
    # are derived centrally from the declared SI coherence times and reference
    # frequency in config.py.
    # Cavity: kappa/wc = 4.1e-2 (physical CPW)

    # Operating-point configurations
    configs = [
        # (label, kappa/wc, gamma_rel, gamma_phi_rel)
        ("pure (no loss)",          0,         0,         0),
        ("cavity only (kappa=1e-4)",  1e-4,     0,         0),
        ("cavity only (kappa=1e-2)",  1e-2,     0,         0),
        ("cavity only (kappa=4.1e-2)", 4.1e-2,  0,         0),
        (f"qubit T1 only (T1={COHERENCE.T1_us:g} us)", 0, GAMMA_T1, 0),
        (f"qubit T_phi only ({COHERENCE.Tphi_us:g} us)", 0, 0, GAMMA_TPHI),
        (f"T1+T_phi ({COHERENCE.T1_us:g} us,{COHERENCE.Tphi_us:g} us)", 0, GAMMA_T1, GAMMA_TPHI),
        ("cav+qubit (physical)",    4.1e-2,    GAMMA_T1, GAMMA_TPHI),
        ("3D cav (kappa=1e-4) + qubit T1", 1e-4, GAMMA_T1, 0),
    ]

    n_steps = 500
    results = []

    for label, kappa, gamma, gphi in configs:
        t0 = time.time()
        tlist, qfi = compute_qfi_with_qubit_loss(kappa, gamma, gphi, n_steps=n_steps)
        elapsed = time.time() - t0

        qfi_peak = float(np.max(qfi))
        peak_idx = int(np.argmax(qfi))
        peak_time = tlist[peak_idx]
        n_ss = max(1, len(qfi) // 5)
        qfi_ss = float(np.mean(qfi[-n_ss:]))

        results.append({
            "label": label,
            "kappa_rel": float(kappa),
            "gamma_rel_T1": float(gamma),
            "gamma_rel_Tphi": float(gphi),
            "qfi_peak": qfi_peak,
            "qfi_ss": qfi_ss,
            "peak_time_periods": float(peak_time / T_D),
            "elapsed_s": float(elapsed),
        })
        print(f"  {label:35s}: pk={qfi_peak:10.1f}, ss={qfi_ss:10.1f}, {elapsed:.1f}s")

    output = {
        "operating_point": {
            "eta": ETA, "gd": GD, "wq_wc": WQ_WC, "wd_wc": WD_WC,
            "n_cav": N_CAV, "n_drive_periods": 40, "delta": DELTA,
        },
        "loss_configs": results,
        "physical_reference_at_6GHz": {
            "T1_us": COHERENCE.T1_us,
            "gamma_rel_T1": GAMMA_T1,
            "Tphi_us": COHERENCE.Tphi_us,
            "gamma_rel_Tphi": GAMMA_TPHI,
            "rate_convention": COHERENCE.rate_convention,
            "rate_provenance": "computed from declared SI coherence times and reference angular frequency",
            "cited_cpw_linewidth_MHz": 248.0,
            "simulation_kappa_rel": 4.1e-2,
            "simulation_kappa_2pi_MHz": 246.0,
            "reference_frequency_GHz": COHERENCE.reference_frequency_GHz,
            "benchmark_label": COHERENCE.benchmark_label,
            "current_record_t1_max_ms": COHERENCE.current_record_t1_max_ms,
            "current_record_citation_key": COHERENCE.current_record_citation_key,
            "current_record_citation_status": COHERENCE.literature_status,
            "background_citation_key": COHERENCE.background_citation_key,
        },
        "method": "Lindblad master equation with combined cavity loss (sqrt(kappa) a), qubit relaxation (sqrt(gamma) sigma-), and pure dephasing (sqrt(gamma_phi/2) sigma_z). Mixed-state QFI via Uhlmann fidelity (sqrtm) for non-pure states.",
        "note": "The CPW reference is approximately 248 MHz, while the normalized simulation point kappa/wc=0.041 at wc/2pi=6 GHz corresponds to 246 MHz. T1=30 us and T_phi=5 us are deliberately conservative benchmark assumptions, not state-of-the-art claims. A 2025 primary source records transmon T1 up to 1.68 ms. The active qubit rates are now computed literally from the declared SI times and angular cavity frequency; the resulting rows are a conservative coherence benchmark, not a record-device prediction. Adding qubit dissipation is evaluated together with cavity loss in the Lindblad model.",
    }

    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "data", "dce_loss_with_qubit.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
