#!/usr/bin/env python3
"""Photon-number-resolved QFI for Dicke+DCE rows (B-S2).

Replays the canonical dicke_dce_loss_adjusted.py row protocol (same
builders, tolerances, DELTA) and recomputes the fidelity-based QFI peak
with both states truncated to Fock levels 0..ncut-1 (trace-renormalized).
The saturation curve QFI(ncut) shows whether the physical-point QFI lives
in the bulk low-n sectors (real effect) or the truncation-sensitive tail.

Gate: at full cutoff the peak must reproduce the archived canonical row
within 2%; aborts otherwise. Rows: kappa in {0, 1e-2, 4.1e-2} at
N_CAV in {6, 12} by default; --smoke runs one reduced row.
Output: data/photon_resolved_qfi.json (checkpointed per row).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import qutip as qt

import dicke_dce_loss_adjusted as prod

from trackb_qutip_utils import density_matrix

ROOT = Path(__file__).resolve().parents[1]
CANON = ROOT / "data" / "dicke_dce_loss_adjusted.json"
OUT = ROOT / "data" / "photon_resolved_qfi.json"
PEAK_TOL = 0.02


def sector_qfi(states_0, states_d, ncut, n_cav, delta, window=slice(None)):
    """Peak fidelity-QFI with both states restricted to Fock < ncut.

    Args:
        states_0: Nominal-parameter states (list of qt.Qobj).
        states_d: Detuned-parameter states (list of qt.Qobj).
        ncut: Retained Fock levels 0..ncut-1 (int).
        n_cav: Full cavity truncation (int).
        delta: Finite-difference step (dimensionless).
        window: Time-index slice evaluated (default all steps; pass a
            peak-centered window for speed — the full peak itself is
            always computed on the complete trace by the caller).

    Returns:
        Tuple (peak_qfi, retained_weight) with peak over the window
        (rad^-2, same unit as the producer) and mean retained trace
        weight (dimensionless).
    """
    dim_spin = int(states_0[0].shape[0] // n_cav)
    proj = qt.Qobj(np.diag([1.0] * ncut + [0.0] * (n_cav - ncut)))
    pop = qt.tensor(proj, qt.qeye(dim_spin))
    peak, weights = 0.0, []
    for s0, sd in zip(states_0[window], states_d[window]):
        q0, qd = density_matrix(s0), density_matrix(sd)
        r0 = pop * q0 * pop
        rd = pop * qd * pop
        w0, wd = float(r0.tr().real), float(rd.tr().real)
        weights.append(0.5 * (w0 + wd))
        if w0 <= 0.0 or wd <= 0.0:
            continue
        r0 = r0 / w0
        rd = rd / wd
        root = r0.sqrtm()
        evals = (root * rd * root).eigenenergies()
        fidelity = float(np.sum(np.sqrt(np.clip(evals.real, 0.0, None))) ** 2)
        fidelity = min(1.0, max(0.0, fidelity))
        peak = max(peak, 8.0 * (1.0 - np.sqrt(fidelity)) / delta ** 2)
    return peak, float(np.mean(weights))


def run_row(n_cav, kappa_rel, n_steps, cutoffs):
    """Replay one canonical row and resolve QFI by photon cutoff.

    Args:
        n_cav: Cavity truncation (int).
        kappa_rel: Dimensionless kappa/omega_c.
        n_steps: Output times over the drive window (int).
        cutoffs: Fock cutoffs to evaluate (list of int, <= n_cav).

    Returns:
        dict: Row result with per-cutoff peaks, weights, and gate check.
    """
    prod.N_CAV = n_cav
    prod.N_MAX = n_cav
    # The producer caches static Hamiltonian pieces in module globals on
    # first use: reset the cache whenever the truncation changes, otherwise
    # a later row reuses stale-dimension operators (dim mismatch crash).
    prod.H0_STATIC = None
    prod.H_DRIVE = None
    # Full-row replay through the canonical producer (validates the path;
    # the replay gate in main() checks the peak against the archive).
    _, qfi, n_phot, _ = prod.compute_dicke_dce_qfi(kappa_rel, n_steps=n_steps)
    # Regenerate the state pair with identical solver settings for the
    # sector analysis (compute_dicke_dce_qfi does not return states).
    a, ad, jx, jz = prod.build_operators()
    c_ops = [] if kappa_rel <= 0 else [np.sqrt(kappa_rel) * a]
    psi0 = prod.get_ground_state()
    g0 = prod.ETA * prod.WC
    h0 = prod.build_H(prod.ETA, g0 * prod.GD_OVER_G0)
    hd = prod.build_H(prod.ETA + prod.DELTA, g0 * prod.GD_OVER_G0)
    tlist = np.linspace(0, prod.T_MAX, n_steps)
    options = {"store_states": True, "atol": prod.SOLVER_ATOL,
               "rtol": prod.SOLVER_RTOL, "nsteps": 100000}
    states_0 = qt.mesolve(h0, psi0, tlist, c_ops=c_ops, e_ops=[],
                          options=options).states
    states_d = qt.mesolve(hd, psi0, tlist, c_ops=c_ops, e_ops=[],
                          options=options).states
    full_peak = float(np.max(qfi))
    peak_idx = int(np.argmax(qfi))
    # Sector eigendecompositions dominate cost (sqrtm + full eig per
    # step per cutoff); evaluate on ±15 steps around the full peak
    # instead of all 200. The reported full peak stays exact.
    lo, hi = max(0, peak_idx - 15), peak_idx + 16
    window = slice(lo, hi)
    curve = []
    for ncut in cutoffs:
        peak, weight = sector_qfi(states_0, states_d, ncut, n_cav,
                                  prod.DELTA, window)
        curve.append({"ncut": ncut, "peak_qfi": peak,
                      "retained_weight": weight,
                      "fraction_of_full": peak / full_peak if full_peak > 0 else 0.0})
    return {"n_cav": n_cav, "kappa_rel": kappa_rel, "n_steps": n_steps,
            "full_peak_qfi": full_peak,
            "sector_window": [lo, hi],
            "mean_photons": float(np.max(n_phot)), "curve": curve}


def main() -> int:
    """Run the configured rows with per-row checkpointing."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="one reduced row (N_CAV=4, kappa=1e-2, 50 steps)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    if args.smoke:
        plan = [(4, 1e-2, 50, [2, 4])]
    else:
        plan = [(6, k, 200, [2, 3, 4, 5, 6]) for k in (0.0, 1e-2, 4.1e-2)]
        plan += [(12, k, 200, [4, 6, 8, 10, 12]) for k in (1e-2, 4.1e-2)]
    canon = {r["kappa_rel"]: r
             for r in json.loads(CANON.read_text())["kappa_sweep"]}
    try:
        rows = json.load(open(args.out))["rows"]
        done = {(r["n_cav"], r["kappa_rel"]) for r in rows}
    except (FileNotFoundError, KeyError):
        rows, done = [], set()
    for n_cav, kappa_rel, n_steps, cutoffs in plan:
        if (n_cav, kappa_rel) in done:
            print(f"skip complete row N_CAV={n_cav} kappa={kappa_rel}")
            continue
        print(f"row N_CAV={n_cav} kappa={kappa_rel} ...", flush=True)
        row = run_row(n_cav, kappa_rel, n_steps, cutoffs)
        if not args.smoke and n_cav == 6 and kappa_rel in canon:
            want = float(canon[kappa_rel]["qfi_peak"])
            got = row["full_peak_qfi"]
            if abs(got - want) > PEAK_TOL * abs(want):
                print(f"REPLAY GATE FAIL: peak {got} vs canonical {want}",
                      file=sys.stderr)
                return 1
            print(f"  replay gate OK ({got:.1f} vs {want:.1f})")
        rows.append(row)
        args.out.write_text(json.dumps({"rows": rows}, indent=2) + "\n")
        for c in row["curve"]:
            print(f"  ncut={c['ncut']}: peak={c['peak_qfi']:.1f} "
                  f"frac={c['fraction_of_full']:.3f} w={c['retained_weight']:.4f}")
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
