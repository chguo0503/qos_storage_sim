#!/usr/bin/env python3
"""Read-only full-run compute/stall/start/gap/tail accounting for policy extension."""
from collections import defaultdict
from pathlib import Path
import math

import aggregate_mixed_profiles as agg


def main():
    output = agg.BASE / "notes/mixed_policy_audit"
    source = output / "audit.json"
    study = agg.read(source)
    rows = []
    for record in study["results"]:
        if record["status"] != "complete":
            continue
        path = Path(record["path"])
        assert agg.sha(path) == record["sha256"]
        x = agg.read(path)
        summary = x["summary"]
        by_card = defaultdict(list)
        for batch in summary["microbatch_metrics"]:
            by_card[batch["npu_id"]].append(batch)
        makespan, npu_count = summary["makespan_ms"],summary["num_npu"]
        start_idle = gap_idle = tail_idle = active = compute = 0.0
        for n in range(npu_count):
            batches = sorted(by_card[n],key=lambda b:b["admission_time_ms"])
            assert batches
            start_idle += batches[0]["admission_time_ms"]
            tail_idle += makespan - batches[-1]["completion_time_ms"]
            for left,right in zip(batches,batches[1:]):
                gap = right["admission_time_ms"]-left["completion_time_ms"]
                assert gap >= -1e-7
                gap_idle += max(0.0,gap)
            active += sum(b["completion_time_ms"]-b["admission_time_ms"] for b in batches)
            compute += sum(b["compute_busy_ms"] for b in batches)
        idle = start_idle+gap_idle+tail_idle
        assert math.isclose(npu_count*makespan,active+idle,rel_tol=1e-10,abs_tol=2e-6)
        assert math.isclose(active,record["all_requests"]["active_ms"],rel_tol=1e-10,abs_tol=2e-6)
        assert math.isclose(compute,record["all_requests"]["compute_ms"],rel_tol=1e-10,abs_tol=2e-6)
        rows.append({"label":record["label"],"strategy":record["strategy"],"assignment":record["assignment"],
                     "makespan_ms":makespan,"total_fleet_budget_npu_ms":npu_count*makespan,
                     "compute_npu_ms":compute,"stall_npu_ms":active-compute,
                     "start_idle_npu_ms":start_idle,"between_requests_idle_npu_ms":gap_idle,
                     "tail_idle_npu_ms":tail_idle,"all_idle_npu_ms":idle,
                     "min_assigned_compute_ms":min(record["assigned_ideal_compute_ms_by_npu"]),
                     "max_assigned_compute_ms":max(record["assigned_ideal_compute_ms_by_npu"]),
                     "source_path":str(path),"source_sha256":record["sha256"]})
    agg.csv_dump(output/"full_idle_decomposition.csv",rows)
    agg.dump(output/"full_idle_decomposition.json",{"script_sha256":agg.sha(Path(__file__)),
              "source_audit_sha256":agg.sha(source),"method":__doc__,
              "identity":"32*makespan = compute + stall + start_idle + between_request_idle + tail_idle; actual execution NPU",
              "rows":rows})
    print({"rows":len(rows),"max_start_idle_npu_ms":max(r["start_idle_npu_ms"] for r in rows),
           "max_between_requests_idle_npu_ms":max(r["between_requests_idle_npu_ms"] for r in rows),
           "output":str(output/"full_idle_decomposition.csv")})


if __name__ == "__main__":
    main()
