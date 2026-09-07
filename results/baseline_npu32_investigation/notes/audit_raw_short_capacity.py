#!/usr/bin/env python3
"""Independently check the raw-three-short full-population byte bound; no simulation."""
import ast
import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NOTES = Path(__file__).resolve().parent


def main():
    evidence = NOTES / "raw_varied_allshort_capacity.json"
    claim = json.loads(evidence.read_text())
    source = ROOT / claim["source_manifest"]
    manifest = json.loads(gzip.decompress(source.read_bytes()))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == claim["source_manifest_file_sha256"]
    assert manifest["input_fingerprint"] == claim["source_input_fingerprint"]
    raw = ast.literal_eval((ROOT / "data").read_text())
    keys = [(p["seq_len_k"], p["nql"]) for p in claim["profiles"]]
    compute = sum(raw[key][1] / 1e6 for key in keys)
    volume = sum(raw[key][3] for key in keys)
    rho = 32 * volume / compute / (8 * 40)
    assert math.isclose(rho, claim["aggregate_disk_rho"], rel_tol=1e-14)
    assert math.isclose(1/rho, claim["full_population_mean_npu_utilization_upper_bound_from_total_bytes"], rel_tol=1e-14)
    short = [r for r in manifest["requests"] if r["load"]["role"] == "short"]
    total_c = sum(8*r["load"]["per_layer_us"]/1000 for r in short)
    total_v = sum(8*sum(v for layer in manifest["placements"][r["placement_index"]] for _,v in layer) for r in short)
    lower_ms = 1000*total_v/(8*40)
    upper_u = total_c/(32*lower_ms)
    assert math.isclose(upper_u, 1/rho, rel_tol=1e-13)
    result = {"audit_passed": True, "audit_method": __doc__,
              "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "checked_evidence_path": str(evidence.relative_to(ROOT)),
              "checked_evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
              "source_manifest_sha256": claim["source_manifest_file_sha256"],
              "raw_keys": keys, "short_requests": len(short),
              "retained_short_compute_ms": total_c, "retained_short_read_gib": total_v,
              "minimum_full_makespan_ms_from_total_bytes": lower_ms,
              "pool_gib_s_per_npu": volume/compute, "aggregate_rho": rho,
              "full_run_fleet_compute_fraction_upper_bound": upper_u,
              "scope": "Full completed population with all reads inside the run; not an arbitrary intermediate-window bound; not a simulated result"}
    (NOTES / "raw_allshort_capacity_independent.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
