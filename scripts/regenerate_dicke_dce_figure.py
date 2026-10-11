#!/usr/bin/env python3
"""Regenerate the Track B Dicke+DCE dynamics figure in a one-row layout.

The producer reads the canonical QuTiP-generated JSON artifact and changes only
figure presentation; it does not recompute or alter the underlying data.

Attributes:
    DATA_PATH: Canonical Dicke+DCE time-trace artifact path.
    OUTPUT_DIR: Directory receiving the regenerated PDF and PNG figures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "dicke_dce_oscillations.json"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "figures"


def parse_args() -> argparse.Namespace:
    """Parse figure regeneration paths.

    Returns:
        argparse.Namespace: Optional input JSON and output directory paths.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Canonical Dicke+DCE JSON artifact.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Figure output directory.")
    return parser.parse_args()


def load_trace(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load and validate the canonical Dicke+DCE trace.

    Args:
        path: JSON artifact containing periods, QFI, photon number, and params.
            The periods are dimensionless drive periods, QFI is baseline-relative,
            and photon number is dimensionless.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, dict]: Drive periods, QFI,
        photon number, and parameter metadata.

    Raises:
        ValueError: If required arrays are missing, mismatched, or non-finite.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    periods = np.asarray(payload["periods"], dtype=float)
    qfi = np.asarray(payload["qfi"], dtype=float)
    photon_number = np.asarray(payload["n_virt"], dtype=float)
    if not (len(periods) == len(qfi) == len(photon_number)):
        raise ValueError("Dicke+DCE arrays have inconsistent lengths.")
    if len(periods) < 2 or not np.all(np.isfinite(np.vstack((periods, qfi, photon_number)))):
        raise ValueError("Dicke+DCE artifact contains insufficient or non-finite data.")
    if np.any(np.diff(periods) <= 0.0):
        raise ValueError("Drive-period samples must be strictly increasing.")
    return periods, qfi, photon_number, payload["params"]


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


def regenerate(data_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """Create the side-by-side Dicke+DCE QFI/photon-number figure.

    Args:
        data_path: Canonical JSON trace path.
        output_dir: Destination directory for the PDF and PNG outputs.

    Returns:
        tuple[Path, Path]: Generated PDF and PNG paths.
    """
    periods, qfi, photon_number, params = load_trace(data_path)
    configure_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, (qfi_axis, photon_axis) = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.0),
        sharex=True,
        constrained_layout=True,
    )

    qfi_axis.plot(periods, qfi, color="C0", linewidth=1.15)
    qfi_axis.set_yscale("symlog", linthresh=10.0)
    qfi_axis.set_xlabel("Drive periods")
    qfi_axis.set_ylabel(r"QFI ($\times$ coherent baseline)")
    qfi_axis.set_title("(a) QFI")
    qfi_axis.set_xlim(float(periods[0]), float(periods[-1]))
    qfi_axis.axvline(float(periods[np.argmax(qfi)]), color="C3", linestyle=":", linewidth=0.8)

    photon_axis.plot(periods, photon_number, color="C1", linewidth=1.15)
    photon_axis.set_xlabel("Drive periods")
    photon_axis.set_ylabel(r"Photon number $\langle a^\dagger a \rangle$")
    photon_axis.set_title("(b) Intracavity photons")

    pdf_path = output_dir / "figB_dicke_dce_oscillations.pdf"
    png_path = output_dir / "figB_dicke_dce_oscillations.png"
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(png_path, format="png", dpi=300)
    plt.close(fig)
    return pdf_path, png_path


def main() -> None:
    """Regenerate the canonical one-row composite figure and report outputs."""
    args = parse_args()
    pdf_path, png_path = regenerate(args.data, args.output_dir)
    print(f"Regenerated {pdf_path}")
    print(f"Regenerated {png_path}")


if __name__ == "__main__":
    main()
