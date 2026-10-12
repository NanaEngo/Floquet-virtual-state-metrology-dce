#!/usr/bin/env python3
"""Non-regression gate for trackb_recompute_utils (B-1, frozen API).

Pins exact semantics of every public helper: golden hash vector for
stable_row_id, atomic-write roundtrip, schema/finite/gate rejections,
checkpoint merge errors, and the __all__ surface itself. Exits non-zero
on any deviation.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import trackb_recompute_utils as utils

EXPECTED_API = [
    "stable_row_id",
    "atomic_write_json",
    "require_schema",
    "assert_finite_json",
    "numerical_gate",
    "merge_checkpoints",
    "failure_row",
]


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record one gate assertion (aborts the gate on failure).

    Args:
        name: Assertion label.
        condition: True when the behavior holds.
        detail: Extra context printed on failure.
    """
    print(f"{'OK  ' if condition else 'FAIL'} {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        raise SystemExit(f"API GATE FAIL: {name} {detail}")


def main() -> None:
    """Run every pinned behavior check."""
    check("api_surface", list(utils.__all__) == EXPECTED_API, str(sorted(utils.__all__)))

    # stable_row_id: golden vector from the documented algorithm
    canonical = json.dumps({"a": 1, "b": [1.5, "x"]}, sort_keys=True,
                           separators=(",", ":"), ensure_ascii=True)
    golden = "row_" + hashlib.sha256(canonical.encode()).hexdigest()[:16]
    check("row_id_golden", utils.stable_row_id({"b": [1.5, "x"], "a": 1}) == golden, golden)
    check("row_id_prefix", utils.stable_row_id({"a": 1}, prefix="chk").startswith("chk_"))
    check("row_id_distinct", utils.stable_row_id({"a": 1}) != utils.stable_row_id({"a": 2}))

    # atomic_write_json roundtrip + parent creation + NaN refusal
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "sub" / "row.json"
        utils.atomic_write_json(dest, {"row_id": "row_x", "v": [1, 2.5]})
        check("atomic_roundtrip", json.loads(dest.read_text()) == {"row_id": "row_x", "v": [1, 2.5]})
        try:
            utils.atomic_write_json(dest, {"v": float("nan")})
            check("atomic_rejects_nan", False)
        except (TypeError, ValueError):
            check("atomic_rejects_nan", True)

    # require_schema
    utils.require_schema({"schema": "s1"}, "s1")
    check("schema_accept", True)
    try:
        utils.require_schema({"schema": "s2"}, "s1")
        check("schema_reject", False)
    except ValueError:
        check("schema_reject", True)

    # assert_finite_json
    utils.assert_finite_json({"a": 1.0, "b": {"c": [1, 2]}})
    check("finite_accept", True)
    try:
        utils.assert_finite_json({"a": float("inf")})
        check("finite_reject_inf", False)
    except ValueError:
        check("finite_reject_inf", True)
    utils.assert_finite_json({"a": float("nan")}, ignored_keys={"a"})
    check("finite_ignore_key", True)

    # numerical_gate boundaries
    good = utils.numerical_gate(1e-12, 1e-12, -1e-6, -1e-4)
    check("gate_accept", good["numerical_acceptance"] is True)
    bad = utils.numerical_gate(1e-12, 1e-12, -1e-3, -1e-4)
    check("gate_reject_floor", bad["numerical_acceptance"] is False)
    bad2 = utils.numerical_gate(float("nan"), 0.0, 0.0, -1e-4)
    check("gate_reject_nan", bad2["numerical_acceptance"] is False)

    # merge_checkpoints: happy path, missing, duplicate, unexpected
    with tempfile.TemporaryDirectory() as tmp:
        cdir = Path(tmp)
        (cdir / "row_aaa.json").write_text(json.dumps({"row_id": "row_aaa", "v": 1}))
        (cdir / "row_bbb.json").write_text(json.dumps({"row_id": "row_bbb", "v": 2}))
        rows, missing = utils.merge_checkpoints(cdir, {"row_aaa", "row_bbb", "row_ccc"})
        check("merge_rows", [r["row_id"] for r in rows] == ["row_aaa", "row_bbb"], str(rows))
        check("merge_missing", missing == ["row_ccc"], str(missing))
        (cdir / "row_dup.json").write_text(json.dumps({"row_id": "row_aaa"}))
        try:
            utils.merge_checkpoints(cdir, {"row_aaa", "row_bbb"})
            check("merge_reject_duplicate", False)
        except ValueError:
            check("merge_reject_duplicate", True)

    # failure_row structure
    fr = utils.failure_row("row_x", {"a": 1}, RuntimeError("boom"))
    check("failure_row", fr["status"] == "failed" and fr["error_type"] == "RuntimeError"
          and fr["row_id"] == "row_x" and "boom" in fr["error_message"], str(fr))

    print("TRACKB UTILS GATE: all PASS.")


if __name__ == "__main__":
    main()
