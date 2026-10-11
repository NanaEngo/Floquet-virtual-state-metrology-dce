#!/usr/bin/env python3
"""Regenerate the Track B finite-size Dicke scaling figure (two panels).

Panel (a) shows the static ground-state gap-weighted QFI versus emitter
count N from the canonical finite-size artifact (extended grid, N<=64) with
the declared power-law fit over N in [6, 18].  Panel (b) shows the static
QFI versus coupling eta at fixed N from the canonical eta-sweep artifact
with the thermodynamic critical coupling marked.  Both panels read the
canonical QuTiP-generated JSON artifacts and change only figure
presentation; they do not recompute or alter the underlying data.

Attributes:
    FINITE_SIZE_PATH: Canonical finite-size scaling artifact path.
    ETA_SWEEP_PATH: Canonical Dicke eta-sweep artifact path.
    OUTPUT_DIR: Directory receiving the regenerated PDF and PNG figures.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Pin the embedded PDF timestamp so regenerations are byte-identical.
os.environ.setdefault("SOURCE_DATE_EPOCH", "1757000000")

import matplotlib.pyplot as plt
import numpy as np


FINITE_SIZE_PATH = Path(__file__).resolve().parents[1] / "data" / "finite_size_resource_scaling.json"
ETA_SWEEP_PATH = Path(__file__).resolve().parents[1] / "data" / "dicke_eta_sweeps.json"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "figures"

FIT_N_MIN = 6
FIT_N_MAX = 18
FIT_FAMILY_WQ_WC = 1.005


def parse_args() -> argparse.Namespace:
    """Parse figure regeneration paths.

    Returns:
        argparse.Namespace: Optional artifact and output directory paths.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finite-size", type=Path, default=FINITE_SIZE_PATH, help="Canonical finite-size JSON artifact.")
    parser.add_argument("--eta-sweep", type=Path, default=ETA_SWEEP_PATH, help="Canonical eta-sweep JSON artifact.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Figure output directory.")
    return parser.parse_args()


def load_finite_size_family(path: Path, wq_wc: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load the static QFI-versus-N family from the finite-size artifact.

    Args:
        path: JSON artifact containing the finite-size scaling rows.
        wq_wc: Qubit-to-cavity ratio selecting the family, dimensionless.

    Returns:
        tuple[np.ndarray, np.ndarray, dict]: Emitter counts N, static QFI
        (high truncation, dimensionless), and the matching fit record.

    Raises:
        ValueError: If the family is incomplete, non-finite, or unordered.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        row
        for row in payload["rows"]
        if row["row_type"] == "static"
        and abs(row["wq_wc"] - wq_wc) <= 1.0e-9
        and row["truncation_converged"]
    ]
    rows.sort(key=lambda row: row["N"])
    sizes = np.array([row["N"] for row in rows], dtype=float)
    qfi = np.array([row["qfi_high_truncation"] for row in rows], dtype=float)
    if sizes.size < 2 or not np.all(np.isfinite(qfi)):
        raise ValueError("Finite-size family is incomplete or non-finite.")
    if np.any(np.diff(sizes) <= 0.0):
        raise ValueError("Emitter counts must be strictly increasing.")
    families = payload["validation"]["scaling_fits"]["families"]
    fit_record = next(
        (f for f in families if abs(f["wq_wc"] - wq_wc) <= 1.0e-9),
        None,
    )
    if fit_record is None:
        raise ValueError("No scaling fit stored for the requested family.")
    return sizes, qfi, fit_record


def load_eta_sweeps(path: Path) -> list[dict]:
    """Load and validate the eta-sweep rows.

    Args:
        path: JSON artifact containing the QFI-versus-eta curves.

    Returns:
        list[dict]: Rows sorted by emitter count N.

    Raises:
        ValueError: If any row is incomplete, non-finite, or fails its
        anchor cross-check against the finite-size grid.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    validation = payload["validation"]
    if not validation.get("all_rows_complete") or not validation.get("all_anchors_within_tolerance"):
        raise ValueError("Eta-sweep artifact is incomplete or failed its anchor gate.")
    rows = sorted(payload["rows"], key=lambda row: row["N"])
    for row in rows:
        eta = np.asarray(row["eta_values"], dtype=float)
        qfi = np.asarray(row["qfi"], dtype=float)
        if eta.size != qfi.size or eta.size < 2 or not np.all(np.isfinite(qfi)):
            raise ValueError("Eta-sweep row is incomplete or non-finite.")
    return rows


def configure_style() -> None:
    """Apply the repository's preferred publication style when available."""
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "nature", "no-latex"])
    except ImportError:
        plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "savefig.dpi": 300,
        }
    )


def regenerate(finite_size_path: Path, eta_sweep_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """Create the two-panel finite-size scaling figure.

    Args:
        finite_size_path: Canonical finite-size artifact path.
        eta_sweep_path: Canonical eta-sweep artifact path.
        output_dir: Destination directory for the PDF and PNG outputs.

    Returns:
        tuple[Path, Path]: Generated PDF and PNG paths.
    """
    sizes, qfi_n, fit_record = load_finite_size_family(finite_size_path, FIT_FAMILY_WQ_WC)
    sweep_rows = load_eta_sweeps(eta_sweep_path)
    configure_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, (n_axis, eta_axis) = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)

    # Panel (a): QFI versus N with the declared power-law fit over [6, 18].
    n_axis.plot(sizes, qfi_n, marker="o", markersize=3.2, linewidth=1.0, color="C0", label="exact $F_Q$ (grid)")
    power_fit = fit_record["power_fit"]
    alpha = float(power_fit["alpha"])
    alpha_err = float(power_fit["alpha_standard_error"])
    prefactor = float(power_fit["prefactor_10^intercept"])
    fit_mask = (sizes >= FIT_N_MIN) & (sizes <= FIT_N_MAX)
    fit_line = prefactor * sizes[fit_mask] ** alpha
    n_axis.plot(sizes[fit_mask], fit_line, linestyle="--", linewidth=1.0, color="C3", label=rf"fit $N^{{\alpha}}$, $\alpha={alpha:.2f}\pm{alpha_err:.2f}$")
    n_axis.set_xscale("log")
    n_axis.set_yscale("log")
    n_axis.set_xlabel(r"Qubits $N$")
    n_axis.set_ylabel(r"QFI ($\times$ coherent baseline)")
    n_axis.set_title("(a) Static QFI vs $N$")
    n_axis.legend(loc="lower right", frameon=False)

    # Panel (b): QFI versus eta at fixed N from the eta-sweep artifact.
    for index, row in enumerate(sweep_rows):
        eta = np.asarray(row["eta_values"], dtype=float)
        qfi = np.asarray(row["qfi"], dtype=float)
        eta_axis.plot(eta, qfi, linewidth=1.0, color=f"C{index}", label=rf"$N={row['N']}$")
        anchor = row["anchor"]
        eta_axis.plot(anchor["eta"], anchor["qfi"], marker="s", markersize=3.5, color=f"C{index}", linestyle="none")
    eta_axis.axvline(0.5, linestyle=":", linewidth=0.9, color="k")
    eta_axis.text(0.51, 0.04, r"$\eta_c=0.5$", transform=eta_axis.get_xaxis_transform(), fontsize=7.5, rotation=90, va="bottom")
    eta_axis.set_yscale("log")
    eta_axis.set_xlabel(r"Coupling $\eta = g_0/\omega_c$")
    eta_axis.set_ylabel(r"QFI ($\times$ coherent baseline)")
    eta_axis.set_title(r"(b) Static QFI vs $\eta$")
    eta_axis.legend(loc="upper left", frameon=False)

    pdf_path = output_dir / "figB_dicke_scaling.pdf"
    png_path = output_dir / "figB_dicke_scaling.png"
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(png_path, format="png", dpi=300)
    plt.close(fig)
    return pdf_path, png_path


def main() -> None:
    """Regenerate the canonical two-panel scaling figure and report outputs."""
    args = parse_args()
    pdf_path, png_path = regenerate(args.finite_size, args.eta_sweep, args.output_dir)
    print(f"Regenerated {pdf_path}")
    print(f"Regenerated {png_path}")


if __name__ == "__main__":
    main()
