#!/usr/bin/env python3
"""Joint realistic-device row at the single-qubit DCE optimum (B-S1).

Runs the driven-DCE protocol once pure (T=0, no loss) and once joint
(T=20 mK, physical cavity loss, qubit T1), reusing the
thermal_competition machinery (_operators, _propagate, _qfi_trace),
and reports the end-to-end QFI retention.

Scope honesty: the thermal machinery builds two-level Pauli operators
only, so this row is joint in (temperature x loss x T1) at d=2. The
three-level extension needs the multilevel effective-spectrum
machinery merged in (declared follow-up); level-2 leakage is therefore
not measured here.

Output: data/joint_stressor_row.json. HPC-scale at full fidelity;
--smoke runs a reduced row for path validation.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import qutip as qt

import thermal_competition as tc
from config import DEFAULT_THERMAL

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "joint_stressor_row.json"

ETA_OPT = 0.150
KAPPA_PHYS = 4.1e-2
TEMP_MK = 20.0
CAV_GHZ = 6.0


def run_protocol(config, parameters):
    """Propagate reference + detuned rows and return peak QFI (rad^-2).

    Args:
        config: ThermalConfig (already specialized for this row).
        parameters: Row parameter mapping (protocol, temperatures, loss).

    Returns:
        Peak QFI over the window (float).
    """
    operators = tc._operators(config.n_cav)
    eta = float(parameters["eta"])
    reference = tc._propagate(parameters, eta, operators, config)
    perturbed = tc._propagate(parameters, eta + config.delta_eta,
                              operators, config)
    qfi = tc._qfi_trace(reference["states"], perturbed["states"],
                        config.delta_eta)
    return float(np.max(qfi))


def main() -> int:
    """Run the pure reference row and the joint realistic row."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    if args.smoke:
        config = dataclasses.replace(
            DEFAULT_THERMAL, n_cav=4, n_drive_periods=2, n_time_points=20)
        gamma1 = 1e-4
    else:
        config = dataclasses.replace(DEFAULT_THERMAL, n_cav=10,
                                      n_drive_periods=20, n_time_points=200)
        gamma1 = 1e-4  # deliberate conservative qubit-T1 benchmark
    base = {"protocol": "driven_dce", "eta": ETA_OPT,
            "init_kind": "vacuum",
            "cavity_frequency_GHz": CAV_GHZ,
            "qubit_frequency_GHz": CAV_GHZ,
            "kappa_rel": 0.0, "temperature_mK": 0.0, "gamma1_rel": 0.0}
    pure_peak = run_protocol(config, base)
    print(f"pure peak QFI = {pure_peak:.3g}", flush=True)
    joint = dict(base, temperature_mK=TEMP_MK, kappa_rel=KAPPA_PHYS,
                 gamma1_rel=gamma1)
    joint_peak = run_protocol(config, joint)
    retention = joint_peak / pure_peak if pure_peak > 0 else 0.0
    print(f"joint peak QFI = {joint_peak:.3g} retention = {retention:.3%}",
          flush=True)
    payload = {
        "status": "JOINT_STRESSOR_ROW",
        "operating_point": {"eta": ETA_OPT, "temperature_mK": TEMP_MK,
                            "d_qubit": 2, "kappa_rel": KAPPA_PHYS,
                            "gamma1_rel": gamma1,
                            "cavity_GHz": CAV_GHZ, "n_cav": config.n_cav,
                            "periods": config.n_drive_periods},
        "pure_peak_qfi": pure_peak,
        "joint_peak_qfi": joint_peak,
        "end_to_end_retention": retention,
        "caveat": "joint in (temperature x cavity loss x qubit T1) at "
                  "d=2; the d=3 multilevel merge is a declared follow-up",
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
