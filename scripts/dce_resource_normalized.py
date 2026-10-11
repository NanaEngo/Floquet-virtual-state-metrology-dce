#!/usr/bin/env python3
"""
M-B-15: Resource-normalized DCE QFI.

Computes:
  (a) QFI per pump-photon budget (gd^2 over the drive window)
  (b) QFI per intracavity photon (n_peak and n_avg)
  (c) Squeezed-state probe benchmark at matched photon budget

The DCE pump is parameterized by gd (modulation amplitude); the time-averaged
pump photon flux scales as gd^2. The intracavity photon budget is given
by <n> computed from the state.

The squeezed-state benchmark is a single-mode probe with squeezing r, which
consumes n_sq = sinh^2(r) photons. The QFI of a pure squeezed state for
parameter encoding U(theta) = exp(-i theta a^dag a) is F_Q = 4 sinh^2(2r)
+ 4 n_sq (linear in n_sq at the SQL).
"""

import json
import time
import numpy as np
import qutip as qt

ETA = 0.150
GD_PEAK = 0.11923
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


def build_H(eta_val, gd_val):
    _build_operators()
    def H_t(t, args=None):
        return H0_STATIC + (eta_val + gd_val * np.cos(WD * t)) * H_DRIVE
    return H_t


def get_ground_state():
    return qt.tensor(qt.basis(N_CAV, 0), qt.basis(2, 0))


def compute_dce_trace(eta_val, gd_val, n_steps=500):
    """Compute DCE QFI and photon number traces at (eta, gd)."""
    a, ad, sx = _ops()

    H_0 = build_H(eta_val, gd_val)
    H_d = build_H(eta_val + DELTA, gd_val)
    psi0 = get_ground_state()

    tlist = np.linspace(0, T_MAX, n_steps)
    result_0 = qt.mesolve(H_0, psi0, tlist, c_ops=[], e_ops=[], args={})
    result_d = qt.mesolve(H_d, psi0, tlist, c_ops=[], e_ops=[], args={})

    qfi = np.zeros(n_steps)
    n_phot = np.zeros(n_steps)
    for i in range(n_steps):
        s0 = result_0.states[i]
        sd = result_d.states[i]
        overlap = abs(s0.overlap(sd))
        qfi[i] = 8.0 * (1.0 - np.sqrt(overlap ** 2)) / DELTA ** 2
        n_phot[i] = qt.expect(a.dag() * a, s0)

    return tlist, qfi, n_phot


def compute_squeezed_benchmark(n_target):
    """QFI of a single-mode squeezed vacuum at fixed photon budget n_target.

    Pure squeezed vacuum |0; r> has n = sinh^2(r) photons.
    QFI for phase encoding: F_Q = 4 * (2 n_target + 1)^2 / 2 ... but the
    practical benchmark is the linear SQL: F_Q = 4 n_target for coherent
    states, and F_Q ~ 4 n (sinh 2r)^2 = 4 n * (1 + 2 n) for squeezed
    states at the same n. We report both SQL and squeezed-state QFI
    at matched n.
    """
    sql_qfi = 4.0 * n_target
    if n_target <= 0:
        return sql_qfi, sql_qfi
    r = np.arcsinh(np.sqrt(n_target))
    sq_qfi = 4.0 * n_target * (1.0 + 2.0 * n_target)
    return sql_qfi, sq_qfi


def main():
    print("=== M-B-15: Resource-Normalized DCE QFI ===")
    print()

    # Sweep gd around the published optimum
    gd_values = [0.05, 0.10, 0.11923, 0.15, 0.20, 0.30, 0.50]
    n_steps = 500
    results = []

    for gd in gd_values:
        t0 = time.time()
        tlist, qfi, n_phot = compute_dce_trace(ETA, gd, n_steps=n_steps)
        elapsed = time.time() - t0

        qfi_peak = float(np.max(qfi))
        qfi_peak_idx = int(np.argmax(qfi))
        n_at_peak = float(n_phot[qfi_peak_idx])
        n_avg = float(np.mean(n_phot))
        n_avg_last10 = float(np.mean(n_phot[-n_steps // 10:]))

        # Pump-photon proxy: gd^2 (time-averaged modulation energy)
        pump_proxy = gd ** 2

        # Normalized QFI metrics
        qfi_per_pump = qfi_peak / pump_proxy
        qfi_per_n_peak = qfi_peak / max(n_at_peak, 1e-6)
        qfi_per_n_avg = qfi_peak / max(n_avg, 1e-6)

        results.append({
            "gd": float(gd),
            "pump_proxy_gd_sq": float(pump_proxy),
            "qfi_peak": qfi_peak,
            "n_at_peak": n_at_peak,
            "n_avg": n_avg,
            "n_avg_last10": n_avg_last10,
            "qfi_per_pump_proxy": qfi_per_pump,
            "qfi_per_n_peak": qfi_per_n_peak,
            "qfi_per_n_avg": qfi_per_n_avg,
            "elapsed_s": float(elapsed),
        })
        print(f"  gd={gd:.4f}: QFI_pk={qfi_peak:10.1f}, n_peak={n_at_peak:5.2f}, n_avg={n_avg:5.2f}, "
              f"QFI/n_avg={qfi_per_n_avg:9.1f}, QFI/pump={qfi_per_pump:9.1f}, {elapsed:.1f}s")

    # Squeezed-state benchmark at matched photon budget
    # Compare DCE at peak (gd=0.11923) with squeezed probe at same n
    dce_peak = next(r for r in results if abs(r["gd"] - GD_PEAK) < 1e-6)
    n_match = dce_peak["n_at_peak"]
    sql_qfi, sq_qfi = compute_squeezed_benchmark(n_match)

    print()
    print(f"=== Comparison at n_target = {n_match:.2f} photons (DCE peak) ===")
    print(f"  DCE peak QFI:           {dce_peak['qfi_peak']:.1f}")
    print(f"  Coherent SQL (4n):      {sql_qfi:.1f}")
    print(f"  Squeezed-state QFI:     {sq_qfi:.1f}")
    print(f"  DCE / SQL ratio:        {dce_peak['qfi_peak']/sql_qfi:.2f}x")
    print(f"  DCE / Squeezed ratio:   {dce_peak['qfi_peak']/sq_qfi:.2f}x")

    # Also compare at n_avg
    n_avg_match = dce_peak["n_avg"]
    sql_qfi_avg, sq_qfi_avg = compute_squeezed_benchmark(n_avg_match)
    print()
    print(f"=== Comparison at n_target = {n_avg_match:.2f} photons (DCE average) ===")
    print(f"  Coherent SQL (4n):      {sql_qfi_avg:.1f}")
    print(f"  Squeezed-state QFI:     {sq_qfi_avg:.1f}")
    print(f"  DCE / SQL ratio:        {dce_peak['qfi_peak']/sql_qfi_avg:.2f}x")
    print(f"  DCE / Squeezed ratio:   {dce_peak['qfi_peak']/sq_qfi_avg:.2f}x")

    output = {
        "operating_point": {
            "eta": ETA, "wq_wc": WQ_WC, "wd_wc": WD_WC,
            "n_cav": N_CAV, "n_drive_periods": 40, "delta": DELTA,
        },
        "gd_sweep": results,
        "squeezed_benchmark": {
            "n_target_at_peak": n_match,
            "sql_qfi_4n": sql_qfi,
            "squeezed_state_qfi_4n_1_2n": sq_qfi,
            "n_target_at_avg": n_avg_match,
            "sql_qfi_avg": sql_qfi_avg,
            "squeezed_qfi_avg": sq_qfi_avg,
            "dce_peak_qfi": dce_peak["qfi_peak"],
            "dce_vs_sql_at_peak": dce_peak["qfi_peak"] / sql_qfi,
            "dce_vs_squeezed_at_peak": dce_peak["qfi_peak"] / sq_qfi,
        },
        "method": {
            "pump_proxy": "gd^2 (time-averaged modulation energy)",
            "squeezed_formula": "F_Q = 4 n (1 + 2n) for pure squeezed vacuum at n=sinh^2(r) photons",
            "sql_formula": "F_Q = 4 n for coherent state (shot noise limit)",
        },
        "note": "The DCE QFI advantage at matched photon budget reflects the parametric amplification gain, but the same photon budget could be invested in a squeezed-state probe. The benchmark answers: at the same intracavity photon cost, is the DCE protocol better than a squeezed-state probe?",
    }

    outpath = "Q1_Papers/manuscripts/track_B_metrology/data/dce_resource_normalized.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
