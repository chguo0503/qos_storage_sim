"""Verify final read-only aggregation and package source/inputs, without rerunning simulations."""
from __future__ import annotations

import csv
import hashlib
import json
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
INITIAL_ENTRY_COUNT = 103


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def is_current_artifact(path):
    # Match analyze_results.py, including interrupted jobs retained for history.
    return not any(part in ("archive", "_archive") or part.startswith("interrupted_attempt_")
                   for part in path.parts)


def local_path(value, root):
    path = Path(value)
    path = (path if path.is_absolute() else root / path).resolve()
    path.relative_to(root.resolve())
    return path


def discover_results(base):
    return {p.resolve() for p in base.glob("**/runs/*/*/*.json.gz") if is_current_artifact(p)} | {
        p.resolve() for p in (base / "pilot").rglob("*.json.gz")
        if "inputs" not in p.relative_to(base).parts and is_current_artifact(p)}


def discover_planned_jobs(base):
    """Mirror the analysis plan rules, retaining duplicate declarations for checks."""
    expected = {}
    plans = set(base.glob("**/*_plan.json")) | set(base.glob("**/plan.json"))
    for path in sorted(p for p in plans if p.parent != base and is_current_artifact(p)):
        plan = json.loads(path.read_text())
        jobs = plan.get("jobs", [])
        inputs = plan.get("inputs", [])
        if not jobs and isinstance(inputs, list) and (path.parent / "runs").is_dir():
            jobs = [{"input": item["label"], "strategy": strategy}
                    for item in inputs for strategy in plan.get("strategies", [])]
        fingerprints = {item["label"]: item.get("input_fingerprint")
                        for item in inputs if isinstance(item, dict) and "label" in item}
        for job in jobs:
            key = (str(path.parent.relative_to(base)), job["input"], job.get("variant", job["strategy"]))
            expected.setdefault(key, []).append({"plan": str(path), "strategy": job["strategy"],
                                                "input_fingerprint": fingerprints.get(job["input"])})
    return expected


def verify_snapshot(audit, rows, failures, initial, *, base, root):
    """Read and verify; no finalization files or archive are written here."""
    base, root = base.resolve(), root.resolve()
    incomplete = [r for r in rows if r["status"] != "complete"]
    changed = [name for name, sha in initial.items()
               if not local_path(name, root).is_file() or digest(local_path(name, root)) != sha]
    files, inputs, errors, audited_paths = [], {}, [], set()
    for row in audit["results"]:
        path = local_path(row["result"], root)
        if path in audited_paths:
            errors.append({"path": str(path), "error": "Duplicate audited result"})
        audited_paths.add(path)
        if not path.is_file() or digest(path) != row["result_sha256"]:
            errors.append({"path": str(path), "error": "Result missing or changed after aggregation"})
        else:
            files.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                          "sha256": row["result_sha256"], "audit_pass": row["all_checks_pass"]})
        manifest = row["manifest"]
        path = local_path(manifest["path"], root)
        if path in inputs and inputs[path] != manifest["sha256"]:
            errors.append({"path": str(path), "error": "Conflicting audited manifest hashes"})
        inputs[path] = manifest["sha256"]
    for path, sha in inputs.items():
        if not path.is_file() or digest(path) != sha:
            errors.append({"path": str(path), "error": "Manifest missing or changed after aggregation"})

    current_paths = discover_results(base)
    complete_rows = [r for r in rows if r["status"] == "complete"]
    summary_paths = {local_path(r["result"], root) for r in complete_rows}
    by_path = {local_path(r["result"], root): r for r in audit["results"]}
    exports_match = all(local_path(r["result"], root) in by_path and
                        r["result_sha256"] == by_path[local_path(r["result"], root)]["result_sha256"] and
                        str(r["audit_pass"]).lower() == "true" for r in complete_rows)
    exported_windows = {}
    for row in complete_rows:
        exported_windows.setdefault(local_path(row["result"], root), Counter())[
            (float(row["start_ms"]), float(row["end_ms"]))] += 1
    for path, result_audit in by_path.items():
        audited_windows = Counter((float(w["start_ms"]), float(w["end_ms"])) for w in result_audit["windows"])
        exports_match = exports_match and exported_windows.get(path, Counter()) == audited_windows
    jobs = {}
    for row in complete_rows:
        jobs.setdefault((row["suite"], row["input"], row["variant"]), []).append(row)
    expected = discover_planned_jobs(base)
    missing_jobs, mismatched_jobs = [], []
    for key, declarations in expected.items():
        if key not in jobs:
            missing_jobs.append({"job": list(key), "plans": [d["plan"] for d in declarations]})
            continue
        found = jobs[key]
        if len({r["result"] for r in found}) != 1 or any(
                r["strategy"] != d["strategy"] or
                (d["input_fingerprint"] and r["input_fingerprint"] != d["input_fingerprint"])
                for r in found for d in declarations):
            mismatched_jobs.append({"job": list(key), "plans": declarations})
    checks = {
        "all_result_audits_pass": bool(audit["results"]) and audit["all_result_audits_pass"] and all(r["all_checks_pass"] for r in audit["results"]),
        "all_result_exports_pass": audit["all_result_exports_pass"] and exports_match and summary_paths == audited_paths,
        "all_matched_groups_pass": audit["all_matched_groups_pass"],
        "no_incomplete_plan_rows": not incomplete,
        "all_current_planned_jobs_collected": not missing_jobs and not mismatched_jobs,
        "no_failure_records": not failures,
        "initial_103_source_data_entries_present": len(initial) == INITIAL_ENTRY_COUNT and "data" in initial,
        "initial_sources_and_data_unchanged": not changed,
        "all_audited_results_and_inputs_rehashed": not errors,
        "current_results_match_audit_snapshot": current_paths == audited_paths,
        "snapshot_counts_match": len(files) == audit["result_files_read"] == audit["result_files_in_snapshot"]
                                 and len(rows) == audit["summary_rows"] and len(failures) == audit["failure_record_count"],
        "current_analysis_source_matches_snapshot": digest(base / "analyze_results.py") == audit["analysis_source_sha256"],
    }
    verification = {
        "at_utc": datetime.now(timezone.utc).isoformat(), "checks": checks,
        "all_checks_pass": all(checks.values()), "initial_entry_count": len(initial),
        "changed_initial_entries": changed, "incomplete_rows": incomplete,
        "failure_records": failures, "result_files": files, "file_errors": errors,
        "missing_current_planned_jobs": missing_jobs, "mismatched_current_planned_jobs": mismatched_jobs,
        "results_added_since_aggregation": sorted(str(p) for p in current_paths - audited_paths),
        "results_removed_since_aggregation": sorted(str(p) for p in audited_paths - current_paths),
        "audited_input_files": [{"path": str(p.relative_to(root)), "sha256": sha} for p, sha in sorted(inputs.items())],
    }
    return verification, set(inputs)


def create_archive(archive, source_files, source_index, index_path, root):
    """Commit only an archive whose actual contents match the recorded index."""
    temporary = archive.with_name(archive.name + ".tmp")
    expected = {row["path"]: row["sha256"] for row in source_index}
    expected[str(index_path.relative_to(root))] = digest(index_path)
    try:
        with tarfile.open(temporary, "w:gz", dereference=True) as tar:
            for path in sorted(source_files):
                tar.add(path, arcname=str(path.relative_to(root)), recursive=False)
            tar.add(index_path, arcname=str(index_path.relative_to(root)), recursive=False)
        seen = set()
        with tarfile.open(temporary, "r:gz") as tar:
            for member in tar:
                if not member.isfile() or member.name in seen or member.name not in expected:
                    raise AssertionError(f"Unexpected archive entry: {member.name}")
                seen.add(member.name)
                h = hashlib.sha256()
                with tar.extractfile(member) as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        h.update(chunk)
                if h.hexdigest() != expected[member.name]:
                    raise AssertionError(f"Source changed while packaging: {member.name}")
        assert seen == set(expected), "Archive is missing indexed files"
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    audit = json.loads((BASE / "analysis/matched_audit.json").read_text())
    rows = list(csv.DictReader((BASE / "analysis/summary.csv").open()))
    failures = json.loads((BASE / "analysis/failures.json").read_text())
    initial = json.loads((BASE / "initial_source_hashes.json").read_text())
    verification, inputs = verify_snapshot(audit, rows, failures, initial, base=BASE, root=ROOT)
    write(BASE / "final_verification.json", verification)
    assert verification["all_checks_pass"], "Final verification incomplete; see final_verification.json"
    files = verification["result_files"]

    result_rows = {r["result"]: r for r in rows if r["status"] == "complete"}
    suite_counts = Counter(r["suite"] for r in result_rows.values())
    source_files = set(ROOT.glob("*.py")) | {ROOT / "data"}
    source_files.update(BASE.rglob("*.py"))
    source_files.update(inputs)
    source_files.add(BASE / "initial_source_hashes.json")
    for path in BASE.rglob("*.json"):
        if "plan" in path.stem or "spec" in path.stem:
            source_files.add(path)
    source_files = {p for p in source_files if p.is_file() and is_current_artifact(p) and "__pycache__" not in p.parts}
    assert inputs <= source_files, "An audited input would be omitted from the package"
    source_index = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p), "bytes": p.stat().st_size}
                    for p in sorted(source_files)]
    # The archive must contain the audited input bytes and initial source bytes,
    # even if a file changes between verification and building this index.
    protected_hashes = dict(initial)
    protected_hashes.update({row["path"]: row["sha256"] for row in verification["audited_input_files"]})
    indexed_hashes = {row["path"]: row["sha256"] for row in source_index}
    assert all(indexed_hashes.get(name) == sha for name, sha in protected_hashes.items()), \
        "A frozen source or audited input changed before packaging"
    write(BASE / "final_source_index.json", source_index)
    archive = BASE / "final_source_and_inputs.tar.gz"
    create_archive(archive, source_files, source_index, BASE / "final_source_index.json", ROOT)
    now = datetime.now(timezone.utc)
    start = datetime(2026, 9, 6, 15, 50, tzinfo=timezone.utc)
    metadata = {
        "started_utc": start.isoformat(), "finalized_utc": now.isoformat(),
        "elapsed_hours": (now - start).total_seconds() / 3600,
        "user_deadline_utc": "2026-09-06T21:50:00+00:00",
        "before_user_deadline": (now - start).total_seconds() <= 6 * 3600,
        "all_final_checks_pass": True, "initial_source_data_entries_unchanged": len(initial),
        "audited_result_file_count": len(files), "result_files_by_suite": dict(sorted(suite_counts.items())),
        "distinct_result_sha256_count": len({f["sha256"] for f in files}),
        "unique_input_fingerprint_count": len({r["input_fingerprint"] for r in result_rows.values()}),
        "counting_scope": "Completed new-study result artifacts; includes explicit bridge/repeat experiments and pilot/grid jobs. Original experiment reanalysis is separate. Formal local offload copies installed at their original job paths are counted once. Historical interrupted attempts and preparation-only candidates are excluded.",
        "archive_scope": "Root Python sources and data, study Python helpers, audited frozen input manifests and plan/spec JSON. Raw simulation results and rendered report remain in the study directory; this is a source/input package, not a full result archive.",
        "source_archive": {"path": archive.name, "sha256": digest(archive), "bytes": archive.stat().st_size,
                           "source_and_input_file_count": len(source_files)},
        "final_report": {"markdown": "docs/report.md", "pdf": "report.pdf"},
        "audits": ["final_verification.json", "analysis/matched_audit.json", "analysis/failures.json"],
    }
    write(BASE / "completion.json", metadata)
    print(json.dumps({k: metadata[k] for k in ("elapsed_hours", "before_user_deadline", "all_final_checks_pass", "audited_result_file_count", "unique_input_fingerprint_count")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
