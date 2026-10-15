#!/usr/bin/env python3
"""Merge per-kappa N_cav=10,12 recomputation caches into single JSON artifacts."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE / "data" / "_cache"
OUT_DIR = BASE / "data"

def merge_n_cav(n_cav: int):
    cache = CACHE_DIR / f"dicke_dce_loss_adjusted_n{n_cav}"
    rows = []
    for k in range(5):
        f = cache / f"kappa_{k}.json"
        with open(f) as fh:
            d = json.load(fh)
        rows.append(d["kappa_sweep"][0])
    rows.sort(key=lambda r: r["kappa_rel"])

    first = CACHE_DIR / f"dicke_dce_loss_adjusted_n{n_cav}" / "kappa_0.json"
    with open(first) as fh:
        d0 = json.load(fh)
    op = d0["operating_point"]
    caveat = d0.get("caveat", "")
    note = d0.get("note", "")

    artifact = {
        "schema": "trackB_dicke_dce_loss_adjusted",
        "operating_point": op,
        "kappa_sweep": rows,
        "caveat": caveat,
        "note": f"Merged per-kappa recomputation for N_cav={n_cav}.",
    }
    out = OUT_DIR / f"dicke_dce_loss_adjusted_n{n_cav}.json"
    with open(out, "w") as fh:
        json.dump(artifact, fh, indent=2)
    print(f"Wrote {out} ({len(rows)} kappa rows)")
    for r in rows:
        print(f"  k={r['kappa_rel']:.4f}  QFI_pk={r['qfi_peak']:.1f}  "
              f"QFI_late={r['qfi_ss_mean']:.1f}  n_pk={r['n_peak']:.2f}  "
              f"min_eig={r['physicality']['min_eigenvalue']:.1e}")

if __name__ == "__main__":
    merge_n_cav(10)
    merge_n_cav(12)
