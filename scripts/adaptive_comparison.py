#!/usr/bin/env python3
"""
M-B-17: Adaptive protocol comparison.

Computes formal uncertainty-bound references for coherent-state phase
estimation as a function of mean photon number and records them beside the
DCE QFI at the stated photon budgets.

The adaptive entries are uncertainty bounds or formal reference expressions,
not QFI values. They are retained for context only and are not divided into
DCE-to-adaptive enhancement ratios.
"""

import json
import numpy as np


def coherent_sql(n):
    """SQL QFI for coherent state with mean photon number n."""
    return 4.0 * n


def helstrom_reference(n):
    """Return the formal single-mode Helstrom reference expression.

    This quantity is stored as a reference bound, not as an adaptive QFI.
    """
    if n <= 0:
        return 0.0
    return 1.0 / n ** 2 * (4.0 * n + 1.0)


def adaptive_uncertainty_bound(n):
    """Return the formal adaptive uncertainty-bound reference ``1/n``.

    The value is not a QFI and is not directly comparable to the DCE QFI.
    """
    if n <= 0:
        return 0.0
    return 1.0 / n


def main():
    # Reference budgets: DCE peak at n ~ 9, time-average budget at n ~ 5.7;
    # the latter reuses the peak QFI and is explicitly not a matched probe.
    n_values = [0.1, 1.0, 5.0, 5.72, 9.0, 9.01, 20.0, 50.0]

    # DCE peak QFI from resource normalization
    dce_peak_qfi = 170902.7  # at n_peak = 9.01, gd=0.11923
    dce_at_avg = 170902.7  # peak reused at the time-average photon budget; not matched

    print("=== Adaptive Protocol Comparison ===")
    print()
    print(f"{'n_target':>10} {'SQL QFI':>12} {'Adaptive bound':>20} {'Formal ref.':>12} {'DCE reference':>24}")
    print("-" * 80)

    rows = []
    for n in n_values:
        sql = coherent_sql(n)
        adapt = adaptive_uncertainty_bound(n)
        helm = helstrom_reference(n)
        if abs(n - 9.01) < 0.1:
            dce = dce_peak_qfi
            dce_basis = "peak_qfi_at_peak_photon_budget"
            dce_label = f"{dce:.0f} (peak budget)"
        elif abs(n - 5.72) < 0.1:
            dce = dce_at_avg
            dce_basis = "peak_qfi_reused_at_time_average_budget_not_matched"
            dce_label = f"{dce:.0f} (reused peak)"
        else:
            dce = None
            dce_basis = None
            dce_label = "n/a"
        print(f"{n:10.2f} {sql:12.1f} {adapt:20.4f} {helm:12.2f} {dce_label:>24}")
        rows.append({
            "n": n,
            "sql_qfi": sql,
            "adaptive_uncertainty_bound": adapt,
            "helstrom_reference": helm,
            "dce_qfi_reference": dce,
            "dce_reference_basis": dce_basis,
        })

    # Headline comparison
    print()
    print("=== Headline Comparison ===")
    print(f"At n_target = 9.01 photons (DCE peak photon budget):")
    print(f"  Coherent SQL (4n):                {coherent_sql(9.01):.1f}")
    print(f"  Adaptive uncertainty bound (1/n): {adaptive_uncertainty_bound(9.01):.4f}")
    print(f"  Helstrom reference expression:    {helstrom_reference(9.01):.2f}")
    print(f"  DCE peak QFI:                    {dce_peak_qfi:.1f}")
    print(f"  DCE / SQL-QFI ratio:             {dce_peak_qfi/coherent_sql(9.01):.1f}x")
    print("  DCE/adaptive ratios:             not computed (different estimands)")
    print("  DCE/formal-reference ratio:      not computed (different estimands)")
    print()
    print("Caveat: The adaptive Bayesian CRB assumes ideal measurement and "
          "feedback at the same intracavity photon number; the SQL assumes "
          "ideal homodyne at the same photon number. The DCE QFI assumes "
          "perfect fidelity under unitary evolution (no loss). Under "
          "physical cavity loss (kappa/2pi ~ 248 MHz), the DCE QFI "
          "collapses to <1% of the pure-state value, while the adaptive "
          "protocol is more robust to loss because it does not require "
          "parametric amplification of a vacuum.")

    output = {
        "schema": "adaptive_comparison_v2_estimands_explicit",
        "n_sweep": rows,
        "headline": {
            "n_target": 9.01,
            "sql_qfi": coherent_sql(9.01),
            "adaptive_uncertainty_bound": adaptive_uncertainty_bound(9.01),
            "helstrom_reference": helstrom_reference(9.01),
            "dce_qfi": dce_peak_qfi,
            "dce_vs_sql_qfi": dce_peak_qfi / coherent_sql(9.01),
        },
        "method": {
            "sql_qfi": "F_Q = 4 n (coherent-state QFI reference)",
            "adaptive_uncertainty_bound": "uncertainty bound ~ 1/n (known amplitude, optimal feedback assumptions)",
            "helstrom_reference": "formal reference expression (4n+1)/n^2; not an adaptive QFI",
        },
        "caveat": "Adaptive entries are uncertainty-bound or formal reference expressions, whereas DCE entries are pure-state QFI values. No DCE-to-adaptive ratio is computed. The n=5.72 row reuses the peak QFI only as a label for the time-average photon budget; it is not a matched-probe calculation. A meaningful device comparison requires a common estimation task, prior, observation time, loss model, drive budget, and readout (see SI).",
    }

    outpath = "Q1_Papers/manuscripts/track_B_metrology/data/adaptive_comparison.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
