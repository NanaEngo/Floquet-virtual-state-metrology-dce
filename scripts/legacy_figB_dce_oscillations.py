"""
Figure: DCE time trace at the rounded representative point (eta=0.150, gd=0.119).
The trace is distinct from the fixed-detuning scan maximum and is stored in dce_final_results.json.
"""
import json, warnings, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import qutip as qt

N_fock = 20
a = qt.destroy(N_fock)
I2 = qt.qeye(2); I_cav = qt.qeye(N_fock)
a_full = qt.tensor(a, I2)
sx_full = qt.tensor(I_cav, qt.sigmax())
sz_full = qt.tensor(I_cav, qt.sigmaz())
n_op = a_full.dag() * a_full
wc = 1.0
gs_init = qt.tensor(qt.basis(N_fock, 0), qt.basis(2, 0))

eta, gd, wq_r, wd_r = 0.150, 0.119, 1.027, 2.077
H0 = 0.5 * (wq_r * wc) * sz_full + wc * a_full.dag() * a_full
H_drive = sx_full * (a_full + a_full.dag())

def build_H(eta_val):
    def H_t(t, args=None):
        return H0 + (eta_val + gd * np.cos(wd_r * wc * t)) * H_drive
    return H_t

T = 2 * np.pi / (wd_r * wc)
tlist = np.linspace(0, 40 * T, 500)

delta = 0.001
opts = dict(nsteps=50000, store_states=True)
result_0 = qt.mesolve(build_H(eta), gs_init, tlist, c_ops=[], e_ops=[], args={}, options=opts)
result_d = qt.mesolve(build_H(eta + delta), gs_init, tlist, c_ops=[], e_ops=[], args={}, options=opts)
states_0 = result_0.states
states_d = result_d.states

qfi_fidelity = np.array([
    8.0 * (1.0 - abs((s0.overlap(sd)))) / delta**2
    for s0, sd in zip(states_0, states_d)
])
qfi_fidelity = np.maximum(qfi_fidelity.real, 0.0)
nv_t = np.array([float(np.real(qt.expect(n_op, s))) for s in states_0])

periods = np.arange(len(tlist)) * 40.0 / len(tlist)
peak_qfi = float(np.max(qfi_fidelity))
peak_idx = int(np.argmax(qfi_fidelity))
ss_mask = tlist >= 30 * T
ss_mean = float(np.mean(qfi_fidelity[ss_mask]))
print(f"Peak QFI (fidelity): {peak_qfi:.0f}× at period {periods[peak_idx]:.1f}")
print(f"SS mean QFI (t>30T): {ss_mean:.0f}×")

try:
    import scienceplots
    plt_style = ["science", "nature", "no-latex"]
except ImportError:
    plt_style = ["seaborn-v0_8-whitegrid"]

import matplotlib.pyplot as plt

with plt.style.context(plt_style):
    plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16, 'legend.fontsize': 12, 'xtick.labelsize': 14, 'ytick.labelsize': 14})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    ax1.plot(periods, qfi_fidelity, "C0-", lw=1.2)
    static_qfi_at_eta = 1.75  # Rabi QFI at η=0.15
    ax1.axhline(static_qfi_at_eta, color="gray", ls="--", lw=0.8,
                label=f"Static QFI at η=0.15 ({static_qfi_at_eta:.2f}×)")
    ax1.annotate(f"Peak: {peak_qfi:.0f}×\nat period {periods[peak_idx]:.0f}",
                 xy=(periods[peak_idx], peak_qfi),
                 xytext=(periods[peak_idx] + 8, peak_qfi * 0.5),
                 arrowprops=dict(facecolor="C3", arrowstyle="->", lw=0.8), color="C3")
    ax1.axhspan(ss_mean * 0.95, ss_mean * 1.05, alpha=0.15, color="C2",
                label=f"SS avg: {ss_mean:.0f}×")
    ax1.set_ylabel(r"QFI ($\times$ coherent baseline)")
    ax1.tick_params(labelsize=11)
    ax1.legend(fontsize=12)
    ax1.set_yscale("symlog", linthresh=10)

    ax2.plot(periods, nv_t, "C1-", lw=1.2)
    ax2.axhline(1.75, color="gray", ls="--", lw=0.8,
                label=f"Static n_virt at η=0.15")
    ax2.set_xlabel("Drive periods")
    ax2.set_ylabel(r"Photon number $\langle a^\dagger a \rangle$")
    ax2.tick_params(labelsize=11)
    ax2.legend(fontsize=12)

    # fig.text(0.02, 0.5,
    #          rf"$\eta={eta:.3f},\; g_d/g_0={gd:.3f}$"
    #          "\n"
    #          rf"$\omega_q/\omega_c={wq_r:.1f},\; \omega_d/\omega_c={wd_r:.1f}$", bbox=dict(facecolor="wheat", alpha=0.5, boxstyle="round"))

    plt.tight_layout()
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / "figB_dce_oscillations.pdf", dpi=300)
    plt.savefig(out / "figB_dce_oscillations.png", dpi=150)
    print(f"Saved figB_dce_oscillations.pdf/png")

output = {
    "eta": eta, "gd": gd, "wq_wc": wq_r, "wd_wc": wd_r,
    "qfi_peak": float(peak_qfi), "qfi_ss_mean": float(ss_mean),
    "n_virt_max": float(np.max(nv_t)),
    "method": "fidelity-based (delta=0.001)"
}
with open(ROOT / "data" / "dce_final_results.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"Saved dce_final_results.json")
