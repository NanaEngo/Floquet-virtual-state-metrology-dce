"""Shared QuTiP state kernels for Track B screen producers (B-2).

Single source of truth for the pure helpers duplicated across the screen
scripts (density normalization, Uhlmann fidelity, purity). Only functions
with byte-identical bodies were centralized; screen-specific acceptance
criteria (_physicality_summary variants, _qfi_trace validation levels)
stay local on purpose because their thresholds feed different JSON schemas.

Two fidelity flavors are kept because call sites depend on their exact
numerics: squared_fidelity (general Uhlmann form) and
squared_fidelity_pure (ket-overlap shortcut for pure states).
"""
from __future__ import annotations

import numpy as np
import qutip as qt

__all__ = [
    "density_matrix",
    "squared_fidelity",
    "squared_fidelity_pure",
    "purity",
]


def density_matrix(state: qt.Qobj) -> qt.Qobj:
    """Return a unit-trace density operator for a ket or density operator.

    Args:
        state: QuTiP state, dimensionless.

    Returns:
        qt.Qobj: Unit-trace density operator, dimensionless.
    """
    rho = state.proj() if state.isket else state
    trace = float(np.real(rho.tr()))
    if trace <= 0.0:
        raise ValueError("State trace is non-positive.")
    return rho / trace


def squared_fidelity(state_a: qt.Qobj, state_b: qt.Qobj) -> float:
    """Compute squared Uhlmann fidelity for two states.

    Args:
        state_a: First ket or density operator, dimensionless.
        state_b: Second ket or density operator, dimensionless.

    Returns:
        float: Squared Uhlmann fidelity in [0, 1], dimensionless.
    """
    rho_a = density_matrix(state_a)
    rho_b = density_matrix(state_b)
    root_a = rho_a.sqrtm()
    product = root_a * rho_b * root_a
    eigenvalues = np.real_if_close(product.eigenenergies()).real
    root_fidelity = float(np.sum(np.sqrt(np.clip(eigenvalues, 0.0, None))))
    return float(np.clip(root_fidelity**2, 0.0, 1.0))


def squared_fidelity_pure(state_a: qt.Qobj, state_b: qt.Qobj) -> float:
    """Compute squared fidelity with a ket-overlap shortcut for pure states.

    Bit-identical to squared_fidelity on pure states up to floating-point
    evaluation order; mixed states fall back to the general form. Kept as
    a separate function because existing screens depend on its exact
    numerics.

    Args:
        state_a: First ket or density operator, dimensionless.
        state_b: Second ket or density operator, dimensionless.

    Returns:
        float: Squared fidelity in [0, 1], dimensionless.
    """
    if state_a.isket and state_b.isket:
        overlap = abs(complex(state_a.dag() * state_b)) ** 2
        return float(np.clip(overlap, 0.0, 1.0))
    return squared_fidelity(state_a, state_b)


def purity(rho: qt.Qobj) -> float:
    """Compute the purity of a density operator.

    Args:
        rho: Unit-trace density operator, dimensionless.

    Returns:
        float: Purity Tr(rho^2), dimensionless, clipped to [0, 1].
    """
    return float(np.clip(np.real((rho * rho).tr()), 0.0, 1.0))
