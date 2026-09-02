#!/usr/bin/env python3
"""Verify the standalone campaign bundle and its published formal results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BUNDLE_MANIFEST = ROOT / "bundle_manifest.json"
FORMAL_RESULTS = ROOT / "formal_results"
ANALYZER_NAME = "analyze_npu32_ssu345_timeline64.py"
ANALYZER_SHA256 = "bfcbec54b2f101e5970074b1b717e1544d900e0a3beb67656c265dce4630fa94"
DATA_SHA256 = "fd197b79865b4c1f42d400100c5e05349ca1ba5f2d42b904af8a1759aabeb04b"
EXCLUDED_PARTS = {"__pycache__", ".venv", "raw-private", "reproduced_results"}

EXPECTED_POINTS = {
    (3, "baseline"): (0.5818945783160964, 0.4002450033248864, 2877),
    (3, "layer_once_ttl_5ms"): (0.5831890237000092, 0.6633794789098499, 2887),
    (3, "adaptive_t0_i100ms"): (0.5892580540367265, 0.7249802060819395, 2875),
    (4, "baseline"): (0.7647258646576165, 0.5305620336207514, 3808),
    (4, "layer_once_ttl_5ms"): (0.7633041771605574, 0.7531514520001941, 3805),
    (4, "adaptive_t0_i100ms"): (0.7607061787958205, 0.7872048706140216, 3790),
    (5, "baseline"): (0.8992261694298254, 0.7000394897953395, 4459),
    (5, "layer_once_ttl_5ms"): (0.8920170417534724, 0.8506696533916206, 4445),
    (5, "adaptive_t0_i100ms"): (0.8961180523215756, 0.849595850518655, 4456),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def included_files() -> list[Path]:
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path == BUNDLE_MANIFEST:
            continue
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(ROOT).as_posix())


def manifest_rows() -> list[dict[str, object]]:
    return [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in included_files()
    ]


def write_manifest() -> None:
    payload = {
        "schema": "npu32_ssu345_timeline64_standalone_bundle_v1",
        "formal_source_commit": "79212f1",
        "analyzer_sha256": ANALYZER_SHA256,
        "raw_inputs_included": False,
        "files": manifest_rows(),
    }
    temporary = BUNDLE_MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(BUNDLE_MANIFEST)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"verification failed: {message}")


def verify_bundle_manifest() -> int:
    require(BUNDLE_MANIFEST.is_file(), "bundle_manifest.json is missing")
    payload = json.loads(BUNDLE_MANIFEST.read_text(encoding="utf-8"))
    require(
        payload.get("schema") == "npu32_ssu345_timeline64_standalone_bundle_v1",
        "bundle manifest schema mismatch",
    )
    rows = payload.get("files")
    require(isinstance(rows, list), "bundle manifest files are missing")
    expected = {row["path"]: row for row in rows}
    actual_paths = {
        path.relative_to(ROOT).as_posix(): path for path in included_files()
    }
    require(set(expected) == set(actual_paths), "bundle file set mismatch")
    for relative, path in actual_paths.items():
        row = expected[relative]
        require(path.stat().st_size == row["size_bytes"], f"size mismatch: {relative}")
        require(sha256(path) == row["sha256"], f"SHA-256 mismatch: {relative}")
    return len(rows)


def verify_formal_manifest() -> int:
    manifest_path = FORMAL_RESULTS / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("contains_hostname") is False, "formal manifest exposes hostname")
    require(manifest.get("contains_token") is False, "formal manifest exposes token")
    require(manifest.get("raw_inputs_included") is False, "formal results contain raw inputs")
    require(manifest.get("analyzer_sha256") == ANALYZER_SHA256, "formal analyzer SHA mismatch")
    rows = manifest.get("files")
    require(isinstance(rows, list) and len(rows) == 125, "formal manifest row count mismatch")
    expected = {row["path"]: row for row in rows}
    actual = {
        path.relative_to(FORMAL_RESULTS).as_posix(): path
        for path in FORMAL_RESULTS.rglob("*")
        if path.is_file() and path != manifest_path
    }
    require(set(expected) == set(actual), "formal result file set mismatch")
    limit = int(manifest["strict_max_file_size_bytes_exclusive"])
    for relative, path in actual.items():
        row = expected[relative]
        require(path.stat().st_size == row["size_bytes"], f"formal size mismatch: {relative}")
        require(path.stat().st_size < limit, f"formal file exceeds size limit: {relative}")
        require(sha256(path) == row["sha256"], f"formal SHA mismatch: {relative}")
    return len(rows) + 1


def verify_scientific_summary() -> None:
    validation = json.loads((FORMAL_RESULTS / "validation.json").read_text(encoding="utf-8"))
    require(validation.get("passed") is True, "formal validation did not pass")
    checks = validation.get("checks")
    require(isinstance(checks, dict) and len(checks) == 42, "validation check count mismatch")
    require(all(value is True for value in checks.values()), "a formal validation check is false")
    require(validation.get("analyzer_sha256") == ANALYZER_SHA256, "validation analyzer SHA mismatch")

    results = json.loads((FORMAL_RESULTS / "results.json").read_text(encoding="utf-8"))
    require(
        results.get("all_strategy_utilization_deltas_explained_by_io_barrier_when_other_zero")
        is True,
        "state-duration causal closure is false",
    )
    rows = results.get("whole_64s")
    require(isinstance(rows, list) and len(rows) == 9, "whole_64s shape mismatch")
    observed = {(int(row["num_ssu"]), str(row["case"])): row for row in rows}
    require(set(observed) == set(EXPECTED_POINTS), "whole_64s keys mismatch")
    for key, (expected_util, expected_alpha2, expected_count) in EXPECTED_POINTS.items():
        row = observed[key]
        require(
            math.isclose(float(row["npu_utilization"]), expected_util, rel_tol=0.0, abs_tol=1e-14),
            f"utilization mismatch: {key}",
        )
        require(
            math.isclose(
                float(row["equal_observed_npu_slo_alpha2"]),
                expected_alpha2,
                rel_tol=0.0,
                abs_tol=1e-14,
            ),
            f"alpha2 mismatch: {key}",
        )
        require(int(row["request_count"]) == expected_count, f"request count mismatch: {key}")

    require(sha256(ROOT / ANALYZER_NAME) == ANALYZER_SHA256, "root analyzer SHA mismatch")
    require(
        sha256(FORMAL_RESULTS / ANALYZER_NAME) == ANALYZER_SHA256,
        "published analyzer SHA mismatch",
    )
    require(sha256(ROOT / "data") == DATA_SHA256, "authenticated workload data SHA mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="regenerate bundle_manifest.json from the current public files",
    )
    args = parser.parse_args()
    if args.write_manifest:
        write_manifest()
    bundle_count = verify_bundle_manifest()
    formal_count = verify_formal_manifest()
    verify_scientific_summary()
    print(
        json.dumps(
            {
                "passed": True,
                "bundle_managed_files": bundle_count,
                "formal_result_files": formal_count,
                "formal_validation_checks": 42,
                "scientific_points": 9,
                "analyzer_sha256": ANALYZER_SHA256,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
