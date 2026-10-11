"""
Compute Dicke model QFI sensitivity to detuning ωq/ωc.
Uses QuTiP for exact diagonalization with proper spin-J operators.
Sweeps over ωq/ωc ∈ [0.9, 1.1] at fixed N=32, η=1.15.
"""
import json, sys, os
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import qutip as qt

def dicke_hamiltonian(N, eta, wq_wc, n_max=50):
    """
    Dicke Hamiltonian: H = wc*a†a + wq*Jz + (g/√N)*Jx*(a + a†)
    where g = eta * wc, J = N/2
    """
    wc = 1.0  # Reference frequency
    wq = wq_wc * wc
    g = eta * wc
    J = N / 2  # Total spin
    
    # Cavity operators (n_max dimensional)
    a = qt.destroy(n_max)
    ad = a.dag()
    
    # Collective spin operators (2J+1 = N+1 dimensional)
    Jx, Jy, Jz = qt.jmat(J)
    
    # Tensor product: cavity ⊗ spin
    H = (wc * qt.tensor(ad * a, qt.qeye(int(2*J+1))) + 
         wq * qt.tensor(qt.qeye(n_max), Jz) + 
         (g / np.sqrt(N)) * qt.tensor(a + ad, Jx))
    
    return H

def compute_dicke_qfi(N, eta, wq_wc, n_max=50, n_eigenstates=20):
    """
    Compute static QFI for Dicke model at given detuning.
    Uses the formula: F_Q = 4 * ∑_{n>0} |⟨n|∂θH|0⟩|^2 / (E_n - E_0)^2
    """
    wc = 1.0
    g = eta * wc
    J = N / 2
    
    # Get Hamiltonian and eigenstates
    H = dicke_hamiltonian(N, eta, wq_wc, n_max)
    evals, evecs = H.eigenstates(eigvals=n_eigenstates)
    psi0 = evecs[0]
    E0 = evals[0]
    
    # Interaction Hamiltonian (derivative w.r.t. eta)
    # H_int = ∂H/∂η = (wc/√N) * Jx * (a + a†)
    a = qt.destroy(n_max)
    ad = a.dag()
    Jx, Jy, Jz = qt.jmat(J)
    H_int = (wc / np.sqrt(N)) * qt.tensor(a + ad, Jx)
    
    # Compute QFI using sum over excited states
    qfi = 0.0
    for n in range(1, n_eigenstates):
        psi_n = evecs[n]
        E_n = evals[n]
        delta_E = E_n - E0
        if delta_E > 1e-10:  # Avoid division by zero
            # Matrix element: ⟨n|H_int|0⟩
            # psi_n.dag() * H_int * psi0 returns a complex scalar
            matrix_element = psi_n.dag() * H_int * psi0
            qfi += 4 * abs(matrix_element)**2 / delta_E**2
    
    return qfi.real

# Parameters
N = 32
eta = 1.15
n_max = 35  # Fock space truncation
n_eigenstates = 20  # Number of eigenstates to include

print("=" * 60)
print("Dicke Model Detuning Sensitivity Analysis")
print("=" * 60)
print(f"N = {N}, η = {eta}, n_max = {n_max}")
print(f"Sweeping ωq/ωc ∈ [0.9, 1.1]")
print()

# Detuning sweep
wq_wc_vals = np.linspace(0.9, 1.1, 21)
QFI_detuning = []

print("Computing QFI for each detuning...")
for i, wq_wc in enumerate(wq_wc_vals):
    try:
        qfi = compute_dicke_qfi(N, eta, wq_wc, n_max, n_eigenstates)
        QFI_detuning.append(qfi)
        print(f"  [{i+1}/21] ωq/ωc = {wq_wc:.3f}: QFI = {qfi:.1f}×")
    except Exception as e:
        print(f"  [{i+1}/21] ωq/ωc = {wq_wc:.3f}: ERROR - {e}")
        import traceback
        traceback.print_exc()
        QFI_detuning.append(0.0)

QFI_detuning = np.array(QFI_detuning)

# Load existing sensitivity data, or create new
sens_path = ROOT / "data" / "dicke_sensitivity.json"
if sens_path.exists():
    with open(sens_path) as f:
        sens_data = json.load(f)
else:
    sens_data = {"N": N, "eta": eta, "n_max": n_max}

# Update with detuning sweep
sens_data["detuning_sweep"] = {
    "wq_wc": wq_wc_vals.tolist(),
    "qfi": QFI_detuning.tolist()
}

# Save updated data
with open(ROOT / "data" / "dicke_sensitivity.json", "w") as f:
    json.dump(sens_data, f, indent=2)

print()
print("=" * 60)
print("DETUNING SENSITIVITY COMPLETE")
print("=" * 60)
if len(QFI_detuning) > 0 and np.max(QFI_detuning) > 0:
    peak_idx = np.argmax(QFI_detuning)
    print(f"Peak QFI: {QFI_detuning[peak_idx]:.1f}× at ωq/ωc = {wq_wc_vals[peak_idx]:.3f}")
    print(f"QFI at ωq/ωc = 0.9: {QFI_detuning[0]:.1f}× ({QFI_detuning[0]/QFI_detuning[peak_idx]*100:.1f}%)")
    print(f"QFI at ωq/ωc = 1.1: {QFI_detuning[-1]:.1f}× ({QFI_detuning[-1]/QFI_detuning[peak_idx]*100:.1f}%)")
    
    # Check robustness
    robust_mask = QFI_detuning > 0.9 * QFI_detuning[peak_idx]
    if np.any(robust_mask):
        wq_robust = wq_wc_vals[robust_mask]
        print(f"Robust range (>90% peak): ωq/ωc ∈ [{wq_robust[0]:.3f}, {wq_robust[-1]:.3f}]")
else:
    print("No valid QFI values computed")
print()
print(f"Results saved to {ROOT / 'data' / 'dicke_sensitivity.json'}")
