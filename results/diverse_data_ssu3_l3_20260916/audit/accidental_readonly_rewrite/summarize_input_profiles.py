#!/usr/bin/env python3
"""Report frozen input multiplicities and initial legal-pool decisions."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT)]
from run_baseline_npu32_stress import load_manifest
from policy_logic import category_path_ids, hardware_view
from continuous_prefill_client import ISSUE_INTERVAL_US, static_qos_config
from policy import RequestBudget, VARIANTS, choose_path_pool


def main():
    inputs = {}
    checks = []
    for path in sorted((HERE / "inputs").glob("*.json.gz")):
        requests, metadata = load_manifest(path)
        checks.append(dict(file=str(path.relative_to(HERE)),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            input_fingerprint=metadata["input_fingerprint"], request_count=len(requests),
            profiles_per_npu=len(metadata["profiles"]),
            pure_compute_ms_per_npu=metadata["actual_pure_compute_ms_per_npu"],
            source_data_sha256=metadata["source_data_sha256"]))
        if path.name in ("semi_seed7.json.gz", "full_seed7.json.gz"):
            inputs[metadata["scenario_candidate"]] = requests, metadata
    (HERE / "input_checks.json").write_text(json.dumps(dict(
        checks="All listed manifests restored and fingerprints checked; direct raw profiles, exact blocks; no SSD simulation in this audit",
        manifests=checks), indent=2) + "\n")
    qos = hardware_view(static_qos_config())
    counts = {scenario: Counter((q.load["seq_len_k"],q.load["nql"]) for q in requests if q.npu_id == 0)
              for scenario,(requests,_) in inputs.items()}
    rows = []
    for key in sorted(counts["semi"]):
        q = next(q for q in inputs["semi"][0] if q.npu_id==0 and (q.load["seq_len_k"],q.load["nql"])==key)
        c_ms = q.load["per_layer_us"]/1000
        v = q.load["per_layer_kv_gb"]
        disk_work = [sum(size for d,size in q.placement[0] if d==disk) for disk in range(3)]
        budget = RequestBudget(12*c_ms,8*c_ms,8,tuple(disk_work))
        pool = category_path_ids(q.load["category"],qos)
        decisions = {name: choose_path_pool(pool,qos,budget=budget,config=VARIANTS[name])
                     for name in ("mild","aggressive","static")}
        # A serial-stage feasible schedule upper-bounds isolated read latency:
        # submit every command, read each disk in parallel, then transmit all
        # bytes over this NPU's receive link. The native pipeline can overlap
        # these stages. Zero modeled control/CPU/dispatch latency is assumed.
        issue_upper_ms = len(q.placement[0])*ISSUE_INTERVAL_US/1000
        read_lower_ms = 1000*max(max(disk_work)/40,v/50)
        read_upper_ms = issue_upper_ms+1000*max(disk_work)/40+1000*v/50
        request_upper_ms = read_upper_ms+8*c_ms+7*max(0.,read_upper_ms-c_ms)
        slo_ms = 12*c_ms
        assert request_upper_ms <= slo_ms
        rows.append(dict(profile=f"{key[0]}:{key[1]}",total_length_k=key[0],miss_tokens=key[1],
                         category=q.load["category"],layer_V_GiB=v,layer_C_ms=c_ms,B_GiB_s=v/(c_ms/1000),
                         semi_count_per_npu=counts["semi"][key],full_count_per_npu=counts["full"][key],
                         original_legal_pool_width=len(pool),
                         initial_mild_pool_width=len(decisions["mild"].path_ids),
                         initial_aggressive_pool_width=len(decisions["aggressive"].path_ids),
                         static_pool_width=len(decisions["static"].path_ids),
                         initial_mode=decisions["static"].mode,
                         initial_waiting_budget_per_layer_ms=.5*c_ms,
                         isolated_layer_read_fluid_lower_bound_ms=read_lower_ms,
                         isolated_layer_read_conservative_upper_bound_ms=read_upper_ms,
                         isolated_cold_request_conservative_upper_bound_ms=request_upper_ms,
                         request_SLO15_ms=slo_ms,
                         isolated_request_SLO15_feasible_in_model=True,
                         pre_admission_L0_pool="full legal pool; unknown deadline falls back to Once"))
    with (HERE / "input_profile_summary.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(f"Validated {len(checks)} manifests; wrote {len(rows)} profile rows; all isolated conservative bounds fit SLO1.5")
    print("Decision counts:",dict(Counter((r["original_legal_pool_width"],r["static_pool_width"]) for r in rows)))


if __name__ == "__main__":
    main()
