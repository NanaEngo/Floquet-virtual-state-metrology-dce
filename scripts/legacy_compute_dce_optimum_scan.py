"""
Compute DCE optimum scan: exact QFI via fidelity method F_Q = 8(1-|F|)/δ²
Two solves per point: at η and η+δ. Gauge-invariant, exact for small δ.
"""
import json, sys, os, warnings, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import qutip as qt

N_cav = 20
a = qt.destroy(N_cav)
I2 = qt.qeye(2); I_cav = qt.qeye(N_cav)
sx_full = qt.tensor(I_cav, qt.sigmax())
sz_full = qt.tensor(I_cav, qt.sigmaz())
n_op = qt.tensor(a.dag()*a, I2)
a_full = qt.tensor(a, I2)
H_int = sx_full * (a_full + a_full.dag())
wc = 1.0; wq_r = 1.027; wd_r = 2.077
T_d = 2*np.pi/(wd_r * wc)
d_eta = 0.001

def compute_qfi(eta, gd, tlist):
    H0 = 0.5*(wq_r*wc)*sz_full + wc*qt.tensor(a.dag()*a, I2)
    psi0 = qt.tensor(qt.basis(N_cav,0), qt.basis(2,0))
    def H_t(t, args):
        return H0 + (eta + gd*np.cos(wd_r*wc*t))*H_int
    result = qt.mesolve(H_t, psi0, tlist, c_ops=[], e_ops=[],
                        options=dict(nsteps=50000, store_states=True))
    return result.states

tlist = np.linspace(0, 40*T_d, 500)
print(f"Computing DCE optimum scan: fidelity-based QFI (δ={d_eta})")
print(f"Grid: {len(np.linspace(0.05,0.30,16))}η × {len(np.linspace(0.05,0.50,14))}gd")
print(f"Each point: 2× mesolve (η and η+δ), {len(tlist)} time steps")

eta_vals = np.linspace(0.05, 0.30, 16)
gd_vals = np.linspace(0.05, 0.50, 14)
data = []
total = len(eta_vals)*len(gd_vals)
t0 = time.time()

for i, eta in enumerate(eta_vals):
    for j, gd in enumerate(gd_vals):
        try:
            states_eta = compute_qfi(eta, gd, tlist)
            states_etad = compute_qfi(eta+d_eta, gd, tlist)
            qfi_t = np.array([8*(1-abs(states_eta[k].overlap(states_etad[k])))/d_eta**2
                              for k in range(len(tlist))])
            n_max = float(np.max([(s.dag()*n_op*s).real for s in states_eta]))
            data.append({"eta": float(eta), "gd": float(gd),
                         "qfi_peak": float(np.max(qfi_t)),
                         "qfi_ss": float(np.mean(qfi_t[-100:])),
                         "qfi_mean": float(np.mean(qfi_t)),
                         "n_virt_max": n_max})
        except Exception as e:
            print(f"  ERROR at η={eta:.3f}, gd={gd:.3f}: {e}")
        idx = i*len(gd_vals)+j+1
        if idx % 16 == 0 or idx == total:
            elapsed = time.time()-t0
            rate = idx/elapsed if elapsed > 0 else 0
            eta_rem = (total-idx)/rate if rate > 0 else 0
            print(f"  [{idx}/{total}] {idx/total*100:.0f}% | {rate:.1f} pts/s | ETA {eta_rem:.0f}s")

best = max(data, key=lambda d: d['qfi_peak'])
print(f"\nBest: η={best['eta']:.3f}, gd={best['gd']:.3f}, QFI_peak={best['qfi_peak']:.1f}")
print(f"Total time: {time.time()-t0:.0f}s")

with open(ROOT / "data" / "dce_optimum_scan.json", "w") as f:
    json.dump(data, f, indent=2)
print(f"Saved {len(data)} points to dce_optimum_scan.json")
