#!/usr/bin/env python3
"""Generate bounded SLURM plans for Track B without submitting jobs.

The planner emits a deterministic shell script and JSON task manifest.  It
never calls ``sbatch``.  The generated array is one logical row per task and
uses the resource profile declared in ``scripts/config.py``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

try:
    from config import (
        DEFAULT_CAVITY_HEOM,
        DEFAULT_DICKE_LOSS_EXTENSION,
        DEFAULT_MULTILEVEL_DRIVE_SWEEP,
        DEFAULT_RESOURCE_MATCHED,
    )
    from trackb_recompute_utils import atomic_write_json, stable_row_id
except ImportError as exc:  # pragma: no cover
    raise ImportError("Run this planner from the Track B scripts directory.") from exc


STUDIES = {
    "dicke-extension": (DEFAULT_DICKE_LOSS_EXTENSION, "dicke_dce_loss_adjusted_slurm.py"),
    "multilevel-sweep": (DEFAULT_MULTILEVEL_DRIVE_SWEEP, "multilevel_drive_sweep.py"),
    "cavity-heom": (DEFAULT_CAVITY_HEOM, "cavity_heom_validation.py"),
    "resource-matched": (DEFAULT_RESOURCE_MATCHED, "resource_matched_comparison.py"),
}


def _rows_for_study(name: str) -> list[dict[str, Any]]:
    """Return deterministic task rows for a supported study."""
    if name == "dicke-extension":
        config = DEFAULT_DICKE_LOSS_EXTENSION
        return [
            {"n_cav": int(n_cav), "kappa_rel": float(kappa)}
            for n_cav in config.cavity_truncations
            for kappa in config.kappa_values
        ]
    if name == "multilevel-sweep":
        config = DEFAULT_MULTILEVEL_DRIVE_SWEEP
        return [
            {"model": model, "gd": float(gd), "kappa_rel": float(kappa), "d_e": int(levels)}
            for model in config.models
            for levels in config.emitter_levels
            for gd in config.drive_coefficients
            for kappa in config.kappa_values
        ]
    if name == "cavity-heom":
        config = DEFAULT_CAVITY_HEOM
        return [
            {"depth": int(depth), "lam": float(lam), "gamma": float(gamma), "temperature": float(temp), "t_final": float(t_final)}
            for depth in config.hierarchy_depths
            for lam in config.bath_couplings
            for gamma in config.bath_cutoffs
            for temp in config.temperatures
            for t_final in config.final_times
        ]
    if name == "resource-matched":
        config = DEFAULT_RESOURCE_MATCHED
        return [
            {"scenario": scenario, "kappa_rel": float(kappa), "observation_periods": int(periods)}
            for scenario in config.scenarios
            for kappa in config.loss_values
            for periods in config.observation_periods
        ]
    raise ValueError(f"Unsupported study: {name}")


def _profile_for_study(name: str) -> dict[str, Any]:
    """Return the configured bounded resource profile."""
    config = STUDIES[name][0]
    return asdict(config.resources)


def _render_script(name: str, rows: list[dict[str, Any]], output_dir: Path) -> str:
    """Render a non-submitting SLURM array script."""
    profile = _profile_for_study(name)
    launcher = STUDIES[name][1]
    scripts_dir = Path(__file__).resolve().parent
    partition = f"#SBATCH --partition={profile['partition']}\n" if profile["partition"] else ""
    task_lines = []
    for index, row in enumerate(rows):
        task_lines.append(f"# {index}: {stable_row_id(row)} {json.dumps(row, sort_keys=True)}")
    if name == "dicke-extension":
        command = 'python "$SCRIPT_DIR/dicke_dce_loss_adjusted_slurm.py" --n-cav "${NCAV[$SLURM_ARRAY_TASK_ID]}" --kappa-index "${KIDX[$SLURM_ARRAY_TASK_ID]}" --checkpoint-dir "$CHECKPOINT_DIR"'
        arrays = "NCAV=(16 16 16 20 20 20)\nKIDX=(0 1 2 0 1 2)"
    else:
        command = f'python "$SCRIPT_DIR/{launcher}" --row-index "$SLURM_ARRAY_TASK_ID" --output "$CHECKPOINT_DIR/task_${{SLURM_ARRAY_TASK_ID}}.json" --checkpoint-dir "$CHECKPOINT_DIR/rows"'
        arrays = "# The selected producer maps --row-index to its deterministic manifest row."
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=trackb-{name}
#SBATCH --array=0-{len(rows) - 1}%{profile['max_concurrent']}
#SBATCH --cpus-per-task={profile['cpus']}
#SBATCH --mem={profile['memory_gb']}G
#SBATCH --time={profile['walltime_minutes']}
{partition}set -euo pipefail

# Generated plan only: this file does not submit itself.
SCRIPT_DIR="{scripts_dir}"
CHECKPOINT_DIR="{output_dir / '_cache' / name}"
{arrays}
# mamba run -n qom is required by the repository environment.
mamba run -n qom {command}

# Planned task rows:
{chr(10).join(task_lines)}
"""


def parse_args() -> argparse.Namespace:
    """Parse planner options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=tuple(STUDIES), default="dicke-extension")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "_cache" / "hpc_plans")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--script", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    """Write a deterministic plan and never invoke sbatch."""
    args = parse_args()
    rows = _rows_for_study(args.study)
    profile = _profile_for_study(args.study)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.manifest or output_dir / f"{args.study}.json"
    script = args.script or output_dir / f"{args.study}.sh"
    payload = {
        "schema": "trackB_hpc_plan_v1",
        "study": args.study,
        "submission": {"automatic_submission": False, "command": "not invoked"},
        "resource_profile": profile,
        "array": {"start": 0, "end": len(rows) - 1, "max_concurrent": profile["max_concurrent"]},
        "rows": [
            {"array_index": index, "row_id": stable_row_id(row), "parameters": row}
            for index, row in enumerate(rows)
        ],
    }
    atomic_write_json(manifest, payload)
    script.write_text(_render_script(args.study, rows, output_dir), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest), "script": str(script), "rows": len(rows), "automatic_submission": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
