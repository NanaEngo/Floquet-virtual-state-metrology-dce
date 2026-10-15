#!/usr/bin/env python3
"""Kappa crossover sweep at fixed single-qubit DCE protocol (B-S3).

Log-spaced kappa/omega_c sweep through the single-qubit DCE row protocol
(dce_loss_adjusted_qfi), reporting peak-QFI retention vs the lossless row
and locating the crossover points where retention crosses 80 % and 50 %
(the paper's self-declared target metric). Checkpointed per kappa;
HPC-scale at full fidelity; --smoke runs two reduced points.
Output: data/kappa_crossover_sweep.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import dce_loss_adjusted_qfi as row

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "kappa_crossover_sweep.json"


def run_point(kappa_rel, n_steps):
    """Replay one kappa point and return peak QFI (rad^-2).

    Args:
        kappa_rel: Dimensionless kappa/omega_c.
        n_steps: Output times over the drive window (int).

    Returns:
        Peak fidelity-based QFI over the window (float).
    """
    _, qfi = row.compute_loss_adjusted_qfi(kappa_rel, n_steps=n_steps)
    return float(np.max(qfi))


def cross(values, threshold):
    """Log-interpolate the kappa where retention crosses threshold.

    Args:
        values: List of (kappa, retention) sorted by ascending kappa.
        threshold: Retention fraction in (0, 1).

    Returns:
        Crossing kappa (float) or None when never crossed in range.
    """
    for (k0, r0), (k1, r1) in zip(values, values[1:]):
        if r0 >= threshold >= r1 and r0 != r1:
            f = (r0 - threshold) / (r0 - r1)
            return float(np.exp(np.log(k0) + f * (np.log(k1) - np.log(k0))))
    return None


def main() -> int:
    """Run the kappa sweep with per-point checkpointing."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    if args.smoke:
        kappas, n_steps = [0.0, 1e-3], 60
    else:
        kappas = [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 2e-2,
                  4.1e-2]
        n_steps = 500
    try:
        points = json.load(open(args.out))["points"]
        done = {p["kappa_rel"] for p in points}
    except (FileNotFoundError, KeyError):
        points, done = [], set()
    for kappa_rel in kappas:
        if kappa_rel in done:
            print(f"skip complete kappa={kappa_rel}")
            continue
        print(f"kappa={kappa_rel} ...", flush=True)
        points.append({"kappa_rel": kappa_rel,
                       "peak_qfi": run_point(kappa_rel, n_steps),
                       "n_steps": n_steps})
        args.out.write_text(json.dumps({"points": points}, indent=2) + "\n")
    pure = next(p["peak_qfi"] for p in points if p["kappa_rel"] == 0.0)
    curve = sorted(((p["kappa_rel"], p["peak_qfi"] / pure) for p in points
                    if p["kappa_rel"] > 0.0))
    out = {"points": points, "pure_peak_qfi": pure,
           "kappa_star_80": cross(curve, 0.80),
           "kappa_star_50": cross(curve, 0.50)}
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"kappa*80={out['kappa_star_80']} kappa*50={out['kappa_star_50']}")
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
