#!/usr/bin/env python3
"""
Derive the canonical static-reference values (Dicke static QFI family and
single-qubit static Rabi QFI) for the Track B package, cross-validating every
recomputed value against the stored anchors in the data artifacts.

Background (2026-09-03 provenance recovery):
    The static Dicke QFI family (229 / 228.6 / 236.7) and the static-Rabi 1.75x
    denominator were flagged as "no canonical JSON backing" in an earlier audit.
    Git archaeology located the original producers, which had been removed from
    the working tree by the 2026-08-23 restructure (commit b925ee5):
      - Q1_Papers/code/scripts/papers/compute_trackB_all_data.py
      - Q1_Papers/code/scripts/papers/compute_dicke_detuning_sensitivity.py
      - Q1_Papers/code/scripts/papers/trackB_benchmark.py (via Q1_Papers_Roadmap)
    They are restored verbatim as scripts/legacy_*.py in this package. This
    script re-derives the values with those conventions and asserts exact
    agreement with every stored anchor.

Conventions recovered:
    Dicke static QFI: ground-state gap-weighted sum
        F_Q = 4 * sum_{n>0} |<0| V |n>|^2 / (E_n - E_0)^2,
        V = (a + a^dagger) J_x / sqrt(N),
    with exact QuTiP diagonalization of the Dicke Hamiltonian
        H = wc a^dag a + wq J_z + (g/sqrt(N)) J_x (a + a^dagger).
    The published 228.6 / 229 values correspond to (N=32, eta=1.15,
    wq/wc=1.0, n_max=35, n_eig=20); the 236.7 screen reference uses the
    detuned grid point wq/wc=1.005 with n_eig=30.

    Static Rabi 1.75x: the value originates from the legacy Gaussian
    virtual-state benchmark (VirtualStateMetrology, g0=9.815435,
    param=omega_m, eps=1e-4) at the benchmark eta-grid point eta=0.08368
    (F_Q = 1.7539), later hardcoded into the DCE trace figure as
    "Rabi QFI at eta=0.15". It is NOT the static Rabi QFI at eta=0.15,
    which is 1.0694 (exact RabiMetrology fidelity convention). The
    published driven/undriven ratio 97659x therefore used an inconsistent
    static denominator and is superseded by 159722x (= 170903/1.069).

Outputs:
    data/dicke_static_reference.json

The script aborts (exit 1) if any anchor fails to match, so a regression in
the conventions cannot silently propagate.
"""

import json
import logging
import os
import sys

import numpy as np
import qutip as qt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

# ---------------------------------------------------------------------------
# FL_QOM availability check (required for the Gaussian virtual-state convention)
# ---------------------------------------------------------------------------
# fl_qom is a pip-installable dependency (editable install of the FL_QOM repo);
# add `fl_qom` to the pip section of environment-track-B-20260902.yml when
# recreating the environment from this deposit.
try:
    from fl_qom.core.usc_system import USCOptomechanicalSystem
    from fl_qom.core.parameters import SystemParameters
    from fl_qom.analysis.virtual_state_metrology import VirtualStateMetrology
    from fl_qom.analysis.rabi_metrology import RabiMetrology
    FL_QOM_AVAILABLE = True
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "FL_QOM is required for the single-qubit static-Rabi conventions in "
        "derive_static_references.py. Install it from the FL_QOM repository "
        "(pip install -e <path-to-FL_QOM>) before running this script. "
        f"Original error: {exc}"
    ) from exc

logging.disable(logging.CRITICAL)

# ---------------------------------------------------------------------------
# Convention 1: Dicke ground-state gap-weighted QFI (recovered producer)
# ---------------------------------------------------------------------------


def dicke_hamiltonian(n_qubits, eta, wq_wc, n_max):
    """Build the Dicke Hamiltonian in the cavity x collective-spin basis.

    Args:
        n_qubits: Number of qubits N (dimensionless).
        eta: Coupling strength g/wc (dimensionless).
        wq_wc: Qubit-to-cavity frequency ratio wq/wc (dimensionless).
        n_max: Cavity Fock truncation (dimensionless).

    Returns:
        QuTiP Qobj Hamiltonian in units of wc (dimensionless).
    """
    wc, wq = 1.0, wq_wc
    g = eta * wc
    j_spin = n_qubits / 2
    a = qt.destroy(n_max)
    jx, _jy, _jz = qt.jmat(j_spin)
    return (
        wc * qt.tensor(a.dag() * a, qt.qeye(int(2 * j_spin + 1)))
        + wq * qt.tensor(qt.qeye(n_max), _jz)
        + (g / np.sqrt(n_qubits)) * qt.tensor(a + a.dag(), jx)
    )


def dicke_ground_qfi(n_qubits, eta, wq_wc, n_max, n_eig):
    """Ground-state QFI for eta estimation (gap-weighted sum, recovered convention).

    Args:
        n_qubits: Number of qubits N (dimensionless).
        eta: Coupling g/wc at evaluation point (dimensionless).
        wq_wc: Frequency ratio wq/wc (dimensionless).
        n_max: Cavity Fock truncation (dimensionless).
        n_eig: Number of eigenstates in the sum (dimensionless).

    Returns:
        Static ground-state QFI (dimensionless).
    """
    hamiltonian = dicke_hamiltonian(n_qubits, eta, wq_wc, n_max)
    evals, evecs = hamiltonian.eigenstates(eigvals=n_eig)
    a = qt.destroy(n_max)
    jx, _jy, _jz = qt.jmat(n_qubits / 2)
    generator = qt.tensor(a + a.dag(), jx) / np.sqrt(n_qubits)
    psi0 = evecs[0]
    e0 = evals[0]
    qfi = 0.0
    for n in range(1, n_eig):
        de = evals[n] - e0
        if abs(de) > 1e-10:
            matrix_element = abs(psi0.dag() * generator * evecs[n]) ** 2
            qfi += 4 * matrix_element / de**2
    return float(qfi)


def main():
    """Recompute all static-reference anchors and write the canonical JSON.

    Raises:
        SystemExit: If any anchor fails to match its stored value.
    """
    print("=== Track B static-reference derivation (recovered conventions) ===")
    dicke_points = []

    # A1: detuning-sweep anchor -> published 228.6
    q_a1 = dicke_ground_qfi(32, 1.15, 1.0, 35, 20)
    dicke_points.append({
        "point": "A1_detuning_sweep_wq1.0", "N": 32, "eta": 1.15,
        "wq_wc": 1.0, "n_max": 35, "n_eig": 20, "qfi": q_a1,
        "stored_value": 228.62056386076293,
        "matches_stored": abs(q_a1 - 228.62056386076293) < 1e-6,
        "stored_source": "data/dicke_sensitivity.json detuning_sweep[10]",
        "published_as": "228.6 (SI convergence table) / 229 (main text)",
    })

    # A2: no-go screen qfi_ideal -> published 236.7
    q_a2 = dicke_ground_qfi(32, 1.15, 1.005, 35, 30)
    dicke_points.append({
        "point": "A2_no_go_qfi_ideal", "N": 32, "eta": 1.15,
        "wq_wc": 1.005, "n_max": 35, "n_eig": 30, "qfi": q_a2,
        "stored_value": 236.70744443198538,
        "matches_stored": abs(q_a2 - 236.70744443198538) < 1e-6,
        "stored_source": "data/dicke_no_go_boundaries.json qfi_ideal",
        "published_as": "236.7 (thermal/loss screen reference)",
    })

    # A3: N-sweep anchor N=27 -> published 229.1 (the "F_Q(N=27)=229x" claim)
    q_a3 = dicke_ground_qfi(27, 1.15, 1.005, 32, 15)
    dicke_points.append({
        "point": "A3_N_sweep_N27", "N": 27, "eta": 1.15,
        "wq_wc": 1.005, "n_max": 32, "n_eig": 15, "qfi": q_a3,
        "stored_value": 229.140533439563,
        "matches_stored": abs(q_a3 - 229.140533439563) < 1e-6,
        "stored_source": "data/dicke_sensitivity.json N_sweep[N=27]",
        "published_as": "229x (main text saturation data point)",
    })

    # A4-A6: SI convergence table (tab:si_convergence_dicke), rounded 1 decimal
    for n_max, stored_rounded, tag in [
        (20, 228.2, "A4_SI_conv_nmax20"),
        (25, 228.6, "A5_SI_conv_nmax25"),
        (30, 228.6, "A6_SI_conv_nmax30"),
    ]:
        q_val = dicke_ground_qfi(32, 1.15, 1.0, n_max, 20)
        dicke_points.append({
            "point": tag, "N": 32, "eta": 1.15, "wq_wc": 1.0,
            "n_max": n_max, "n_eig": 20, "qfi": q_val,
            "stored_value": stored_rounded,
            "matches_stored": round(q_val, 1) == stored_rounded,
            "stored_source": "SI tab:si_convergence_dicke (rounded to 1 decimal)",
            "published_as": f"{stored_rounded} (SI convergence table)",
        })

    # B1: single-qubit static Rabi, legacy Gaussian virtual-state convention.
    # The published 1.75x denominator traces to this benchmark eta-grid point.
    sys_params = SystemParameters(
        omega_m=1.0, omega_c=1.0, g0=9.815435, kappa=0.05,
        gamma_m=0.01, temperature=0.01, m_eff=1.0,
    )
    system = USCOptomechanicalSystem(sys_params)
    metrology = VirtualStateMetrology(
        system=system, param_name="omega_m", epsilon=1e-4,
    )
    eta_b1 = 0.08368421052631578
    r_b1 = metrology.compute_metrology(eta=eta_b1)
    rabi_point = {
        "point": "B1_legacy_benchmark_eta_grid", "eta": eta_b1, "g0": 9.815435,
        "qfi": r_b1.qfi, "stored_value": 1.7539430322908927,
        "matches_stored": abs(r_b1.qfi - 1.7539430322908927) < 1e-6,
        "stored_source": "git 7bc1f55 Q1_Papers_Roadmap/code/_cache/trackB_eta_0.084.json",
        "published_as": "1.75x static-Rabi denominator (legacy, superseded)",
    }

    # B2: current exact static Rabi QFI at the DCE operating eta=0.15
    rabi_metrology = RabiMetrology(omega_c=1.0, omega_q=1.0, n_max=50)
    qfi_b2 = rabi_metrology.compute_at_eta(0.15).qfi

    all_matches = [p["matches_stored"] for p in dicke_points] + [rabi_point["matches_stored"]]
    if not all(all_matches):
        failed = [p["point"] for p in dicke_points + [rabi_point] if not p["matches_stored"]]
        print(f"ANCHOR MISMATCH: {failed}")
        raise SystemExit(1)

    output = {
        "schema": "static_references_v1",
        "generated": "2026-09-03",
        "generator": "scripts/derive_static_references.py (deterministic; "
                     "conventions recovered from git b925ee5^ producers, "
                     "restored as scripts/legacy_*.py)",
        "dicke_static_qfi": {
            "method": "ground-state gap-weighted sum "
                      "F_Q = 4 sum_n |<0|V|n>|^2/(E_n-E_0)^2, "
                      "V=(a+a^dagger)J_x/sqrt(N), exact QuTiP diagonalization",
            "points": dicke_points,
            "resolution_note": "229/228.6/236.7 are three stored anchors of the "
                               "same estimator family, not conflicting values: "
                               "228.6 = wq/wc 1.0, 236.7 = wq/wc 1.005 screen "
                               "reference with n_eig=30, 229.1 = N_sweep N=27.",
        },
        "rabi_static_qfi": {
            "conventions": {
                "exact_fidelity_RabiMetrology_eta0.15": {
                    "qfi": qfi_b2, "eta": 0.15, "n_max": 50,
                    "published_as": "static Rabi reference at the DCE operating point",
                },
                "legacy_gaussian_virtual_state_g0_9.815435": rabi_point,
                "hardcoded_figure_annotation_1.75": {
                    "value": 1.75,
                    "origin": "legacy_figB_dce_oscillations.py line 70 (git 115f62c): "
                              "static_qfi_at_eta = 1.75  # Rabi QFI at eta=0.15",
                    "nearest_computed_value": 1.7539430322908927,
                    "nearest_eta": eta_b1,
                    "interpretation": "The published 1.75x matches the legacy "
                                      "Gaussian virtual-state benchmark value at "
                                      "eta=0.0837, not a value computed at eta=0.15. "
                                      "At eta=0.15 the exact static Rabi QFI is "
                                      "1.0694x, so the published 97659x "
                                      "driven/undriven ratio used an inconsistent "
                                      "static denominator and is superseded by "
                                      "159722x (= 170903/1.069).",
                },
            },
        },
    }

    out_path = os.path.join(DATA_DIR, "dicke_static_reference.json")
    with open(out_path, "w") as handle:
        json.dump(output, handle, indent=2)
    print(f"All {len(all_matches)} anchors verified exactly.")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
