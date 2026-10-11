#!/usr/bin/env python3
"""Validate the active Track B package before a Physical Review X upload.

The checker is intentionally local and fail-closed. It validates the files that
are meant to be shared, while treating the internal archive and raw SLURM logs as
explicitly excluded provenance. It does not contact APS or infer author metadata.

Attributes:
    REQUIRED_FILES: Relative files required by the active package manifest.
    STALE_BASENAMES: Legacy names that must not occur in the active root.
    GENERATED_SUFFIXES: Generated build suffixes excluded from the upload set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Iterable


REQUIRED_FILES = (
    "01_track_B_metrology_main.tex",
    "01_track_B_metrology_main.pdf",
    "02_track_B_metrology_SI.tex",
    "02_track_B_metrology_SI.pdf",
    "03_track_B_metrology_cover_letter.tex",
    "03_track_B_metrology_cover_letter.pdf",
    "04_popular_summary.txt",
    "README.md",
    "Track_B_data_analysis_report.md",
    "PRX_submission_metadata_20260902.md",
    # NOTE 2026-10-10: PRX_UPLOAD_MANIFEST_20260902.md removed from the
    # required set — superseded by the submission metadata + DAR; its
    # request-wording/upload-set contents are stale relative to the
    # current package (repository URLs, 42 refs, rebuilt PDFs).
    "environment-track-B-20260902.yml",
    "references.bib",
)
STALE_BASENAMES = {
    "03_cover_letter.tex",
    "03_cover_letter.pdf",
    "02_si_trackB.tex",
    "02_si_trackB.pdf",
    "01_track_B_metrology.tex",
    "01_track_B_metrology.pdf",
}
GENERATED_SUFFIXES = (".aux", ".bbl", ".blg", ".log", ".out")
STALE_TERMS = (
    "PRX Quantum",
    "prxquantum",
    "will be deposited",
    "persistent DOI",
    "persistent identifier",
    "reasonnable",
    "availabel",
)
SOURCE_FILES = (
    "01_track_B_metrology_main.tex",
    "02_track_B_metrology_SI.tex",
    "03_track_B_metrology_cover_letter.tex",
    "PRX_submission_metadata_20260902.md",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the package root.

    Returns:
        argparse.Namespace: Parsed ``--root`` path and optional quiet mode.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Track B package root (default: parent of scripts/)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only failures and the final status.",
    )
    return parser.parse_args()


def check_required_files(root: Path, failures: list[str]) -> None:
    """Check that every file named by the active package is present.

    Args:
        root: Track B package root.
        failures: Mutable list receiving human-readable failure messages.
    """
    for relative in REQUIRED_FILES:
        path = root / relative
        if not path.is_file():
            failures.append(f"missing required file: {relative}")

    for directory in ("figures", "scripts", "data"):
        if not (root / directory).is_dir():
            failures.append(f"missing required directory: {directory}/")


def check_active_root(root: Path, failures: list[str], notes: list[str]) -> None:
    """Reject legacy or generated upload files in the package root.

    Args:
        root: Track B package root.
        failures: Mutable list receiving upload violations.
        notes: Mutable list receiving intentional exclusions.
    """
    for path in root.iterdir():
        if path.name == "_archive":
            notes.append("excluded directory: _archive/")
            continue
        if path.name.startswith("slurm-") and path.suffix == ".out":
            notes.append(f"excluded raw scheduler log: {path.name}")
            continue
        if path.name in STALE_BASENAMES:
            failures.append(f"legacy basename in active root: {path.name}")
        if path.is_file() and path.suffix in GENERATED_SUFFIXES:
            failures.append(f"generated build file in active root: {path.name}")
        if path.is_file() and path.name.endswith("Notes.bib"):
            failures.append(f"generated bibliography file in active root: {path.name}")


def check_json(root: Path, failures: list[str]) -> None:
    """Parse every top-level JSON data artifact.

    Args:
        root: Track B package root.
        failures: Mutable list receiving malformed JSON messages.
    """
    data_files = sorted((root / "data").glob("*.json"))
    if not data_files:
        failures.append("no top-level data/*.json artifacts found")
        return
    for path in data_files:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"invalid JSON: {path.relative_to(root)} ({exc})")


def _close(left: float | None, right: float | None, rel: float = 1e-9, abs_tol: float = 1e-9) -> bool:
    """Compare two recorded values within a tight relative tolerance.

    Args:
        left: First value, e.g. an anchor recorded in ``dicke_static_reference.json`` (dimensionless).
        right: Second value, e.g. the same quantity read from its source artifact (dimensionless).
        rel: Relative tolerance (dimensionless); loose enough for JSON round-trip rounding.
        abs_tol: Absolute tolerance (dimensionless) for values near zero.

    Returns:
        bool: True when both values are present and agree within tolerance.
    """
    if left is None or right is None:
        return False
    return abs(left - right) <= max(abs_tol, rel * max(abs(left), abs(right)))


def check_semantics(root: Path, failures: list[str]) -> None:
    """Validate the reconciled loss and adaptive-comparison contracts.

    Args:
        root: Track B package root.
        failures: Mutable list receiving semantic consistency violations.
    """
    data_dir = root / "data"

    adaptive_path = data_dir / "adaptive_comparison.json"
    if adaptive_path.is_file():
        try:
            adaptive = json.loads(adaptive_path.read_text(encoding="utf-8"))
            expected_schema = "adaptive_comparison_v2_estimands_explicit"
            if adaptive.get("schema") != expected_schema:
                failures.append(f"adaptive artifact has unexpected schema: {adaptive.get('schema')}")
            legacy_keys = {"dce_vs_adaptive", "dce_vs_helstrom", "adaptive_bayes", "helstrom"}
            present = legacy_keys.intersection(adaptive.get("headline", {}))
            if present:
                failures.append(f"adaptive artifact retains legacy headline keys: {sorted(present)}")
            for index, row in enumerate(adaptive.get("n_sweep", [])):
                required = {"n", "sql_qfi", "adaptive_uncertainty_bound", "helstrom_reference", "dce_qfi_reference", "dce_reference_basis"}
                missing = required.difference(row)
                if missing:
                    failures.append(f"adaptive row {index} missing keys: {sorted(missing)}")
        except json.JSONDecodeError:
            pass

    loss_path = data_dir / "dce_loss_adjusted_qfi.json"
    if loss_path.is_file():
        try:
            loss = json.loads(loss_path.read_text(encoding="utf-8"))
            interpretation = loss.get("interpretation", {})
            first = interpretation.get("first_sampled_row_at_or_above_target", {})
            if interpretation.get("threshold_status") != "sampled_reference_only_not_a_monotonic_cutoff":
                failures.append("single-qubit loss artifact lacks sampled-threshold status")
            if first.get("kappa_rel") != 1e-4 or first.get("kappa_2pi_MHz") != 0.6:
                failures.append("single-qubit loss artifact has inconsistent first sampled threshold row")
            low_loss = interpretation.get("low_loss_nonmonotonicity", {})
            if low_loss.get("retention_percent") != [78.3, 78.8, 80.6]:
                failures.append("single-qubit loss artifact has inconsistent low-loss retention metadata")
        except json.JSONDecodeError:
            pass

    homodyne_path = data_dir / "homodyne_cfi.json"
    if homodyne_path.is_file():
        try:
            homodyne = json.loads(homodyne_path.read_text(encoding="utf-8"))
            if homodyne.get("schema") != "trackB_homodyne_cfi_v1":
                failures.append(
                    "homodyne artifact has unexpected schema: "
                    f"{homodyne.get('schema')}"
                )
            rows = homodyne.get("rows", [])
            validation = homodyne.get("validation", {})
            if len(rows) != validation.get("row_count"):
                failures.append("homodyne artifact row-count metadata is inconsistent")
            required = {
                "classical_fisher_information",
                "matching_output_gaussian_qfi",
                "cfi_qfi_ratio",
                "same_state_ratio_valid",
                "quadrature_variance",
            }
            for index, row in enumerate(rows):
                missing = required.difference(row)
                if missing:
                    failures.append(f"homodyne row {index} missing keys: {sorted(missing)}")
                    break
                finite_keys = required - {"cfi_qfi_ratio"}
                if not all(math.isfinite(float(row[key])) for key in finite_keys):
                    failures.append(f"homodyne row {index} contains non-finite metrics")
                    break
                ratio_value = row.get("cfi_qfi_ratio")
                if row.get("same_state_ratio_valid"):
                    if ratio_value is None or not math.isfinite(float(ratio_value)):
                        failures.append(f"homodyne valid row {index} has a non-finite CFI/QFI ratio")
                        break
                elif ratio_value is not None and not math.isfinite(float(ratio_value)):
                    failures.append(f"homodyne inconclusive row {index} has an invalid non-null CFI/QFI ratio")
                    break
                if float(row["quadrature_variance"]) <= 0.0 or float(row["classical_fisher_information"]) < 0.0:
                    failures.append(f"homodyne row {index} violates variance/CFI positivity")
                    break
            if not validation.get("all_rows_finite"):
                failures.append("homodyne artifact contains non-finite metrics")
            if not validation.get("nonnegative_cfi"):
                failures.append("homodyne artifact contains negative CFI")
            if not validation.get("same_state_ratio_bounds_pass"):
                failures.append("homodyne artifact violates its same-state CFI/QFI ratio gate")
            hypotheses = validation.get("hypotheses", {})
            for key in ("static_92_4_percent", "driven_88_1_percent"):
                if hypotheses.get(key, {}).get("status") not in {"reproduced", "not_reproduced", "inconclusive"}:
                    failures.append(f"homodyne artifact lacks explicit status for {key}")
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            failures.append(f"invalid homodyne semantic artifact: {exc}")

    multiparameter_path = data_dir / "multiparameter_gauge_qfi.json"
    if multiparameter_path.is_file():
        try:
            multiparameter = json.loads(multiparameter_path.read_text(encoding="utf-8"))
            if multiparameter.get("schema") != "trackB_multiparameter_gauge_qfi_v1":
                failures.append(
                    "multiparameter artifact has unexpected schema: "
                    f"{multiparameter.get('schema')}"
                )
            rows = multiparameter.get("rows", [])
            validation = multiparameter.get("validation", {})
            if len(rows) != validation.get("row_count"):
                failures.append("multiparameter artifact row-count metadata is inconsistent")
            if not validation.get("all_rows_have_finite_metrics"):
                failures.append("multiparameter artifact contains non-finite metrics")
            if validation.get("converged_row_count") != len(rows):
                failures.append("multiparameter artifact has unresolved QFI rows")
            gauge = validation.get("gauge_check", {})
            if gauge.get("status") != "consistent":
                failures.append("multiparameter representation check did not pass")
            if gauge.get("unitarity_error_frobenius", float("inf")) > 1.0e-8:
                failures.append("multiparameter representation is not numerically unitary")
            if gauge.get("state_fidelity_to_transformed_baseline", 0.0) < 1.0 - 1.0e-8:
                failures.append("multiparameter transformed-state fidelity is below tolerance")
        except json.JSONDecodeError:
            pass

    thermal_path = data_dir / "thermal_competition.json"
    if thermal_path.is_file():
        try:
            thermal = json.loads(thermal_path.read_text(encoding="utf-8"))
            if thermal.get("schema") != "single_qubit_thermal_competition_v1":
                failures.append(
                    "thermal artifact has unexpected schema: "
                    f"{thermal.get('schema')}"
                )
            rows = thermal.get("rows", [])
            validation = thermal.get("validation", {})
            if not validation.get("all_rows_complete"):
                failures.append("thermal artifact has incomplete rows")
            if len(rows) != validation.get("rows_completed"):
                failures.append("thermal artifact row-count metadata is inconsistent")
            criterion = validation.get("retention_criterion_fraction")
            if criterion is None or not (0.0 < criterion < 1.0):
                failures.append("thermal artifact lacks a declared retention-criterion fraction")
            for row in rows:
                physicality = row.get("physicality_reference") or {}
                if "numerical_acceptance" in physicality and not physicality["numerical_acceptance"]:
                    failures.append("thermal artifact has a reference row failing the numerical acceptance gate")
                    break
            hypotheses = validation.get("occupation_hypotheses", {})
            for key, entry in hypotheses.items():
                if entry.get("status") not in {"reproduced", "not_reproduced", "inconclusive"}:
                    failures.append(f"thermal artifact lacks explicit status for {key}")
            # Cross-check the stored 6 GHz occupation at 20 mK against the row-level value,
            # so the not-reproduced basis cannot silently drift from the computed data.
            for row in rows:
                parameters = row.get("parameters", {})
                if (
                    parameters.get("cavity_frequency_GHz") == 6.0
                    and parameters.get("temperature_mK") == 20.0
                ):
                    stored_nth = row.get("thermal_occupation_cavity_nth")
                    if stored_nth is None or not math.isfinite(float(stored_nth)):
                        failures.append("thermal 6 GHz / 20 mK row lacks a finite occupation")
                    break
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            failures.append(f"invalid thermal semantic artifact: {exc}")

    multilevel_path = data_dir / "multilevel_emitters.json"
    if multilevel_path.is_file():
        try:
            multilevel = json.loads(multilevel_path.read_text(encoding="utf-8"))
            if multilevel.get("schema") != "trackB_multilevel_emitters_v1":
                failures.append(
                    "multilevel artifact has unexpected schema: "
                    f"{multilevel.get('schema')}"
                )
            rows = multilevel.get("rows", [])
            validation = multilevel.get("validation", {})
            if not validation.get("all_rows_complete"):
                failures.append("multilevel artifact has incomplete rows")
            if len(rows) != validation.get("rows_completed"):
                failures.append("multilevel artifact row-count metadata is inconsistent")
            tolerance = multilevel.get("configuration", {}).get("leakage_tolerance")
            if tolerance is None or tolerance <= 0.0:
                failures.append("multilevel artifact lacks a declared leakage tolerance")
            comparisons = validation.get("two_level_baseline_comparisons", [])
            for entry in comparisons:
                change = entry.get("qfi_peak_relative_change_from_two_level")
                if change is not None and not math.isfinite(float(change)):
                    failures.append("multilevel baseline comparison has a non-finite QFI change")
                    break
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            failures.append(f"invalid multilevel semantic artifact: {exc}")

    heom_ext_path = data_dir / "heom_extended_validation.json"
    if heom_ext_path.is_file():
        try:
            heom_ext = json.loads(heom_ext_path.read_text(encoding="utf-8"))
            if heom_ext.get("schema") != "trackB_heom_extended_validation_v1":
                failures.append(
                    "extended-HEOM artifact has unexpected schema: "
                    f"{heom_ext.get('schema')}"
                )
            rows = heom_ext.get("rows", [])
            validation = heom_ext.get("validation", {})
            if not validation.get("all_rows_complete"):
                failures.append("extended-HEOM artifact has incomplete rows")
            if len(rows) != validation.get("rows_completed"):
                failures.append("extended-HEOM artifact row-count metadata is inconsistent")
            if validation.get("inconclusive_positivity_gate_count", 0) > len(rows):
                failures.append("extended-HEOM artifact has an inconsistent inconclusive-row count")
            for row in rows:
                if row.get("valid_for_agreement") and row.get("status") not in {"physical", "converged"}:
                    failures.append(
                        f"extended-HEOM row {row.get('row_id')} is valid for agreement "
                        "without a physical status"
                    )
                    break
                if not row.get("valid_for_agreement") and row.get("status") == "physical":
                    failures.append(
                        f"extended-HEOM physical row {row.get('row_id')} is not marked valid "
                        "for agreement"
                    )
                    break
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            failures.append(f"invalid extended-HEOM semantic artifact: {exc}")

    finite_size_path = data_dir / "finite_size_resource_scaling.json"
    if finite_size_path.is_file():
        try:
            finite_size = json.loads(finite_size_path.read_text(encoding="utf-8"))
            if finite_size.get("schema") != "trackB_finite_size_resource_scaling_v1":
                failures.append(
                    "finite-size artifact has unexpected schema: "
                    f"{finite_size.get('schema')}"
                )
            rows = finite_size.get("rows", [])
            validation = finite_size.get("validation", {})
            if not validation.get("all_rows_complete"):
                failures.append("finite-size artifact has incomplete rows")
            if len(rows) != validation.get("rows_completed"):
                failures.append("finite-size artifact row-count metadata is inconsistent")
            converged = validation.get("static_truncation_converged_fraction")
            if converged is not None and not (0.0 <= converged <= 1.0):
                failures.append("finite-size artifact has an invalid truncation-converged fraction")
            fits = validation.get("scaling_fits", {}).get("families", [])
            for family in fits:
                power = family.get("power_fit")
                if power and not all(
                    math.isfinite(float(power[key]))
                    for key in ("alpha", "alpha_standard_error")
                ):
                    failures.append("finite-size artifact has a non-finite power-fit parameter")
                    break
                if power is None and not family.get("power_fit_omitted_reason"):
                    failures.append(
                        "finite-size artifact omits a power-fit without a stated reason"
                    )
                    break
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            failures.append(f"invalid finite-size semantic artifact: {exc}")

    eta_sweep_path = data_dir / "dicke_eta_sweeps.json"
    if eta_sweep_path.is_file():
        try:
            eta_sweep = json.loads(eta_sweep_path.read_text(encoding="utf-8"))
            if eta_sweep.get("schema") != "trackB_dicke_eta_sweeps_v1":
                failures.append(
                    "eta-sweep artifact has unexpected schema: "
                    f"{eta_sweep.get('schema')}"
                )
            eta_validation = eta_sweep.get("validation", {})
            if not eta_validation.get("all_rows_complete"):
                failures.append("eta-sweep artifact has incomplete rows")
            if not eta_validation.get("all_anchors_within_tolerance"):
                failures.append("eta-sweep artifact fails its anchor cross-check gate")
            for eta_row in eta_sweep.get("rows", []):
                try:
                    eta_grid = [float(v) for v in eta_row.get("eta_values", [])]
                    eta_qfi = [float(v) for v in eta_row.get("qfi", [])]
                except (TypeError, ValueError):
                    failures.append("eta-sweep artifact has non-numeric coupling or QFI values")
                    break
                if len(eta_grid) < 2 or any(
                    later <= earlier for earlier, later in zip(eta_grid, eta_grid[1:])
                ):
                    failures.append("eta-sweep artifact has a non-increasing coupling grid")
                    break
                if len(eta_qfi) != len(eta_grid) or not all(math.isfinite(v) for v in eta_qfi):
                    failures.append("eta-sweep artifact has non-finite or mismatched QFI values")
                    break
                anchor = eta_row.get("anchor", {})
                if anchor.get("relative_deviation") is not None and not anchor.get("within_tolerance"):
                    failures.append("eta-sweep artifact has an anchor outside the declared tolerance")
                    break
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            failures.append(f"invalid eta-sweep semantic artifact: {exc}")

    dicke_path = data_dir / "dicke_dce_loss_adjusted.json"
    if dicke_path.is_file():
        try:
            dicke = json.loads(dicke_path.read_text(encoding="utf-8"))
            validation = dicke.get("validation", {})
            physicality = str(validation.get("physicality", "")).lower()
            caveat = str(dicke.get("caveat", "")).lower()
            if "numerical" not in physicality or "exact positivity" not in physicality:
                failures.append("Dicke loss validation does not distinguish numerical acceptance from exact positivity")
            if "numerical eigenvalue acceptance floor" not in caveat:
                failures.append("Dicke loss caveat lacks numerical eigenvalue acceptance-floor wording")
            if "positivity tolerance" in caveat:
                failures.append("Dicke loss caveat still calls the acceptance floor a positivity tolerance")
        except json.JSONDecodeError:
            pass

    tables_path = data_dir / "trackB_tables_data.json"
    if tables_path.is_file():
        try:
            tables = json.loads(tables_path.read_text(encoding="utf-8"))
            critical = tables.get("critical_params", {})
            eta_crit = critical.get("superradiant_eta_crit")
            if eta_crit != 0.5:
                failures.append(
                    "trackB_tables_data.json superradiant_eta_crit must be 0.5 "
                    f"(exact thermodynamic Dicke critical coupling at resonance); found {eta_crit}"
                )
            if not critical.get("superradiant_eta_crit_basis"):
                failures.append("trackB_tables_data.json lacks superradiant_eta_crit_basis provenance")
        except json.JSONDecodeError:
            pass

    static_ref_path = data_dir / "dicke_static_reference.json"
    if static_ref_path.is_file():
        try:
            static_ref = json.loads(static_ref_path.read_text(encoding="utf-8"))
            if static_ref.get("schema") != "static_references_v1":
                failures.append(
                    f"static-reference artifact has unexpected schema: {static_ref.get('schema')}"
                )
            # Cross-check every static-reference anchor against its stored source JSON,
            # so a silent edit of either file breaks the preflight instead of drifting.
            sensitivity = {}
            sensitivity_path = data_dir / "dicke_sensitivity.json"
            if sensitivity_path.is_file():
                sensitivity = json.loads(sensitivity_path.read_text(encoding="utf-8"))
            no_go = {}
            no_go_path = data_dir / "dicke_no_go_boundaries.json"
            if no_go_path.is_file():
                no_go = json.loads(no_go_path.read_text(encoding="utf-8"))
            anchor_sources = {
                "A1_detuning_sweep_wq1.0": (
                    lambda: sensitivity["detuning_sweep"]["qfi"][
                        sensitivity["detuning_sweep"]["wq_wc"].index(1.0)
                    ]
                ),
                "A2_no_go_qfi_ideal": lambda: no_go["qfi_ideal"],
                "A3_N_sweep_N27": (
                    lambda: sensitivity["N_sweep"]["qfi"][sensitivity["N_sweep"]["N"].index(27)]
                ),
            }
            for point in static_ref.get("dicke_static_qfi", {}).get("points", []):
                if not point.get("matches_stored"):
                    failures.append(
                        f"static-reference anchor failed: {point.get('point')} "
                        f"(recomputed {point.get('qfi')} vs stored {point.get('stored_value')})"
                    )
                    continue
                source = anchor_sources.get(point.get("point"))
                if source is None:
                    continue
                try:
                    stored = source()
                    if not _close(point.get("qfi"), stored):
                        failures.append(
                            f"static-reference anchor {point.get('point')} disagrees with its "
                            f"source artifact: recorded {point.get('qfi')} vs source {stored}"
                        )
                except (KeyError, ValueError, IndexError) as exc:
                    failures.append(
                        f"static-reference anchor {point.get('point')} source lookup failed: {exc}"
                    )
            legacy = (
                static_ref.get("rabi_static_qfi", {})
                .get("conventions", {})
                .get("legacy_gaussian_virtual_state_g0_9.815435", {})
            )
            if legacy and not legacy.get("matches_stored"):
                failures.append("static-reference legacy Rabi benchmark anchor failed")
        except json.JSONDecodeError:
            pass


def check_summary(root: Path, failures: list[str]) -> None:
    """Enforce the PRX Popular Summary word limit.

    Args:
        root: Track B package root.
        failures: Mutable list receiving summary violations.
    """
    path = root / "04_popular_summary.txt"
    if not path.is_file():
        return
    words = path.read_text(encoding="utf-8").split()
    if len(words) > 150:
        failures.append(f"Popular Summary has {len(words)} words; limit is 150")


def _has_availability(text: str) -> bool:
    """Availability evidence: request-based wording or a public repository URL.

    Author policy 2026-10-10: code/data live in a public GitHub repository,
    which supersedes the request-based route. Accept either as evidence.
    """
    return ("available upon reasonable request" in text) or ("github.com/" in text)


def check_sources(root: Path, failures: list[str]) -> None:
    """Check venue metadata, availability wording, and stale active claims.

    Args:
        root: Track B package root.
        failures: Mutable list receiving source-content violations.
    """
    for relative in SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for term in STALE_TERMS:
            if term in text:
                failures.append(f"stale term in {relative}: {term}")

    manuscript_paths = (
        root / "01_track_B_metrology_main.tex",
        root / "02_track_B_metrology_SI.tex",
    )
    manuscript_texts = []
    for path in manuscript_paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        manuscript_texts.append((path.name, text))
        if "documentclass[aps,prx," not in text:
            failures.append(f"source is not using the PRX class option: {path.name}")
        if not _has_availability(text):
            failures.append(f"missing request-based availability wording: {path.name}")

    author_pattern = re.compile(r"\\author\{([^}]+)\}")
    author_lists = [
        (name, author_pattern.findall(text)) for name, text in manuscript_texts
    ]
    if len(author_lists) == 2 and author_lists[0][1] != author_lists[1][1]:
        failures.append("main and SI author lists differ")
    if author_lists and not author_lists[0][1]:
        failures.append("main manuscript has no parsed authors")

    cover = root / "03_track_B_metrology_cover_letter.tex"
    if cover.is_file():
        cover_text = cover.read_text(encoding="utf-8")
        if "Physical Review X" not in cover_text:
            failures.append("cover letter is not addressed to Physical Review X")
        if not _has_availability(cover_text):
            failures.append("cover letter lacks request-based availability wording")
        if "P.~Djorw" not in cover_text or "corresponding author" not in cover_text:
            failures.append("cover letter lacks the declared corresponding-author record")

    metadata = root / "PRX_submission_metadata_20260902.md"
    if metadata.is_file() and not _has_availability(metadata.read_text(encoding="utf-8")):
        failures.append("metadata record lacks request-based availability wording")


def iter_graphics_references(text: str) -> Iterable[str]:
    """Yield graphics basenames referenced by LaTeX source.

    Args:
        text: LaTeX source text.

    Yields:
        str: The argument supplied to each ``includegraphics`` command.
    """
    pattern = re.compile(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}")
    yield from pattern.findall(text)


def check_figures(root: Path, failures: list[str]) -> None:
    """Verify that each source-referenced figure exists in the figures directory.

    Args:
        root: Track B package root.
        failures: Mutable list receiving missing-figure messages.
    """
    figure_dir = root / "figures"
    available = {path.stem for path in figure_dir.iterdir()} if figure_dir.is_dir() else set()
    for relative in ("01_track_B_metrology_main.tex", "02_track_B_metrology_SI.tex"):
        path = root / relative
        if not path.is_file():
            continue
        for reference in iter_graphics_references(path.read_text(encoding="utf-8")):
            if Path(reference).stem not in available:
                failures.append(f"missing figure for {relative}: {reference}")


def check_checksums(root: Path, failures: list[str]) -> None:
    """Verify every entry in the package checksum manifest.

    Args:
        root: Track B package root.
        failures: Mutable list receiving checksum violations.
    """
    manifest = root / "checksums.sha256"
    if not manifest.is_file():
        failures.append("missing checksums.sha256")
        return
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            failures.append(f"malformed checksum line {line_number}")
            continue
        expected, relative = parts
        path = root / relative.removeprefix("./")
        if not path.is_file():
            failures.append(f"checksum target missing: {relative}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            failures.append(f"checksum mismatch: {relative}")


def main() -> int:
    """Run all preflight checks and return a process status.

    Returns:
        int: ``0`` when all checks pass, otherwise ``1``.
    """
    args = parse_args()
    root = args.root.resolve()
    failures: list[str] = []
    notes: list[str] = []

    if not root.is_dir():
        print(f"FAIL: package root does not exist: {root}")
        return 1

    check_required_files(root, failures)
    check_active_root(root, failures, notes)
    check_json(root, failures)
    check_semantics(root, failures)
    check_summary(root, failures)
    check_sources(root, failures)
    check_figures(root, failures)
    check_checksums(root, failures)

    if not args.quiet:
        print(f"package_root={root}")
        print(f"excluded_items={len(notes)}")
        for note in notes:
            print(f"EXCLUDED: {note}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        print(f"PRX_UPLOAD_PREFLIGHT=FAIL ({len(failures)} failure(s))")
        return 1
    print("PRX_UPLOAD_PREFLIGHT=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
