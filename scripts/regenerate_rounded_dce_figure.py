#!/usr/bin/env python3
"""Regenerate the rounded-point single-qubit DCE dynamics figure.

The producer reproduces the canonical rounded-point trace using the active Track
B convention and exports the QFI and intracavity-photon panels in one row without
modifying the canonical numerical summary artifacts.

Attributes:
    OUTPUT_DIR: Active Track B figure directory.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

try:
    from config import DEFAULT_ROUNDED_DCE_FIGURE, RoundedDCEFigureConfig
except ImportError as exc:  # pragma: no cover - environment-specific
    raise ImportError(
        "Track B configuration is unavailable; run this script from its scripts directory."
    ) from exc


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "figures"


def parse_args() -> argparse.Namespace:
    """Parse regeneration paths.

    Returns:
        argparse.Namespace: Output directory and summary-artifact paths.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def build_operators(config: RoundedDCEFigureConfig) -> tuple[qt.Qobj, qt.Qobj, qt.Qobj, qt.Qobj]:
    """Build the tensor-ordered Rabi operators.

    Args:
        config: Centralized rounded-trace configuration; cavity dimension is
            dimensionless.

    Returns:
        tuple[qt.Qobj, qt.Qobj, qt.Qobj, qt.Qobj]: Annihilation, number,
        qubit-\(\sigma_x\), and static Hamiltonian operators in normalized
        cavity-frequency units.
    """
    a = qt.tensor(qt.destroy(config.n_cav), qt.qeye(2))
    number = a.dag() * a
    sx = qt.tensor(qt.qeye(config.n_cav), qt.sigmax())
    sz = qt.tensor(qt.qeye(config.n_cav), qt.sigmaz())
    static = 0.5 * config.wq_wc * sz + number
    return a, number, sx, static


def solve_trace(config: RoundedDCEFigureConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute paired unitary DCE trajectories and observables.

    Args:
        config: Centralized rounded-point parameters; normalized frequencies and
            couplings are dimensionless, and duration is in drive periods.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Drive periods, fidelity QFI,
        and intracavity photon number, all dimensionless under the package
        normalization.
    """
    annihilation, number, sx, static = build_operators(config)
    interaction = sx * (annihilation + annihilation.dag())
    initial = qt.tensor(qt.basis(config.n_cav, 0), qt.basis(2, 0))
    period = 2.0 * math.pi / config.wd_wc
    times = np.linspace(0.0, config.n_drive_periods * period, config.n_time_points)

    def make_hamiltonian(eta: float) -> qt.QobjEvo:
        """Build the time-dependent Hamiltonian for one coupling value.

        Args:
            eta: Static normalized coupling g0/wc, dimensionless.

        Returns:
            qt.QobjEvo: Driven Hamiltonian in normalized angular-frequency units.
        """
        def coefficient(time_normalized: float, _args: dict | None = None) -> float:
            """Return the normalized DCE coupling coefficient.

            Args:
                time_normalized: Time in inverse cavity angular-frequency units.
                _args: Optional QuTiP callback arguments, unused.

            Returns:
                float: Dimensionless instantaneous coupling coefficient.
            """
            return eta + config.gd * math.cos(config.wd_wc * time_normalized)

        return qt.QobjEvo([static, [interaction, coefficient]])

    options = {
        "atol": 1.0e-10,
        "rtol": 1.0e-8,
        "method": "adams",
        "nsteps": config.solver_nsteps,
        "store_states": True,
        "progress_bar": "",
    }
    reference = qt.mesolve(make_hamiltonian(config.eta), initial, times, c_ops=[], e_ops=[number], options=options)
    perturbed = qt.mesolve(make_hamiltonian(config.eta + config.delta_eta), initial, times, c_ops=[], e_ops=[], options=options)
    qfi = np.asarray(
        [8.0 * (1.0 - abs(complex(left.overlap(right)))) / config.delta_eta**2 for left, right in zip(reference.states, perturbed.states)],
        dtype=float,
    )
    qfi = np.maximum(qfi, 0.0)
    photons = np.asarray(reference.expect[0], dtype=float)
    return times / period, qfi, photons


def configure_style() -> None:
    """Apply the repository's publication style with a fallback."""
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "nature", "no-latex"])
    except ImportError:
        plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 10, "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8})


def regenerate(config: RoundedDCEFigureConfig, output_dir: Path) -> None:
    """Regenerate the side-by-side figure without changing numerical artifacts.

    Args:
        config: Centralized rounded-point parameters.
        output_dir: Destination for PDF and PNG figures.
    """
    periods, qfi, photons = solve_trace(config)
    configure_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, (qfi_axis, photon_axis) = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=True, constrained_layout=True)
    qfi_axis.plot(periods, qfi, color="C0", linewidth=1.15)
    qfi_axis.set_yscale("symlog", linthresh=10.0)
    qfi_axis.set_xlabel("Drive periods")
    qfi_axis.set_ylabel(r"QFI ($\times$ coherent baseline)")
    qfi_axis.set_title("(a) QFI")
    qfi_axis.set_xlim(float(periods[0]), float(periods[-1]))
    photon_axis.plot(periods, photons, color="C1", linewidth=1.15)
    photon_axis.set_xlabel("Drive periods")
    photon_axis.set_ylabel(r"Photon number $\langle a^\dagger a \rangle$")
    photon_axis.set_title("(b) Intracavity photons")
    pdf_path = output_dir / "figB_dce_oscillations.pdf"
    png_path = output_dir / "figB_dce_oscillations.png"
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(png_path, format="png", dpi=300)
    plt.close(fig)
    print(f"Regenerated {pdf_path}")
    print(f"Regenerated {png_path}")


def main() -> None:
    """Regenerate the rounded-point figure from the centralized configuration."""
    args = parse_args()
    regenerate(DEFAULT_ROUNDED_DCE_FIGURE, args.output_dir)


if __name__ == "__main__":
    main()
