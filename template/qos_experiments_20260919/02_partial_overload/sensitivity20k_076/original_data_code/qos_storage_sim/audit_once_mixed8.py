#!/usr/bin/env python3
"""Read-only audit of the existing FIFO and 5-ms Once mixed-eight-NPU runs.

This never launches the simulator or modifies its original sources. It reuses
the independent raw interval audit, then verifies the paired frozen inputs and
records the remaining stalls directly from raw layer timing intervals.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

from audit_fifo_mixed8 import audit_run, read_json, require

SOURCE = Path(__file__).resolve().parent
OUT = SOURCE / "results/fifo_mixed_unique_20260914"


def locate(stage, policy):
    found = []
    for directory in sorted(stage.iterdir()):
        if not directory.is_dir() or not (directory / "metrics.json").exists():
            continue
        if not (directory / "result.json.gz").exists():
            continue
        result = read_json(directory / "result.json.gz")
        if result.get("probe_policy") == policy:
            found.append(directory)
    require(len(found) == 1, f"Expected one completed {policy} case under {stage}; found {len(found)}")
    return found[0]


def raw_stall_examples(result, manifest, window):
    loads = {r["request_id"]: r["load"] for r in manifest["requests"]}
    groups = collections.defaultdict(list)
    for batch in result["summary"]["microbatch_metrics"]:
        rid = batch["member_request_ids"][0]
        previous_end = batch["admission_time_ms"]
        for layer in batch["layer_metrics"]:
            start, end = previous_end, layer["compute_start_ms"]
            previous_end = layer["compute_end_ms"]
            clipped = max(0.0, min(end, window[1]) - max(start, window[0]))
            if clipped <= 1e-9:
                continue
            key = loads[rid]["role"] + ("_L0" if layer["layer"] == 0 else "_L1_L7")
            groups[key].append(dict(npu_id=batch["npu_id"], request_id=rid,
                layer=layer["layer"], waiting_start_ms=start,
                compute_start_ms=end, io_ready_time_ms=layer["io_ready_time_ms"],
                window_stall_card_ms=clipped, untrimmed_stall_ms=end-start))
    return {key: dict(positive_layer_stall_count=len(rows),
        total_window_stall_card_ms=sum(r["window_stall_card_ms"] for r in rows),
        max_untrimmed_stall_ms=max(r["untrimmed_stall_ms"] for r in rows),
        largest_examples=sorted(rows, key=lambda r: -r["window_stall_card_ms"])[:12])
        for key, rows in sorted(groups.items())}


def configuration_document():
    reference = read_json(SOURCE / "reference_command.json")
    hashes = reference["core_source_sha256"]
    checks = {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == digest
              for name, digest in hashes.items()}
    require(all(checks.values()), "Reference Once sources no longer match current source files.")
    return dict(strategy="once", reference_strategy=reference["strategy"],
        reference_name="baseline_ab128_32_ratio12_20260912/once_per_layer_ssu3_seed7",
        reference_command_file="reference_command.json",
        reference_core_source_sha256=hashes,
        reference_source_hash_checks=checks,
        all_29_reference_core_hashes_match=True,
        state_collection=dict(interval_ms=5.0, times="0,5,10,... ms",
            client_reads="Latest periodic snapshot; no client-triggered SSU refresh",
            source="shared_ssu_state.py:29,42; run_coflow_experiments.py:152,182"),
        routing=dict(unit="One complete request-layer-SSU IO plan",
            per_layer_can_use_multiple_paths=True,
            candidate_pool="Only paths belonging to the request QoS category",
            score="Predicted completion time from snapshot backlog and static CIR/PIR/WRR estimate",
            projection="Local shadow adds each IO planned in this same layer",
            other_client_global_reservations_used=False,
            source="shared_path_sim_adapter.py:187-202; policy_logic.py:219-294,439-471"),
        static_qos=dict(path_count=256, group_count=8,
            category_order=["SS","SL","LS","LL"],
            category_total_cir_gib_s=[20,6,8,6],
            category_path_counts=[96,32,96,32],
            category_paths_per_group=[12,4,12,4],
            path_pir="unlimited", path_weights=1.0, group_weights=1.0,
            runtime_cir_writes=0, unused_capacity="Redistributed among backlogged paths",
            source="strategy_profiles.py:20-43; sim.py:686-715"),
        current_input_categories=dict(S="SL",L="LL",
            rationale="Both NQL values exceed the native long-NQL threshold 512; total-length threshold is 80K",
            S_path_offsets_in_each_group=[12,13,14,15],
            L_path_offsets_in_each_group=[28,29,30,31],
            each_class_path_count=32, each_class_cir_gib_s=6,
            each_path_cir_gib_s=0.1875,
            source="sim.py:45-46,170-180; policy_logic.py:200-209"),
        execution=dict(npu_assignment="Frozen manifest NPU binding; no reassignment or request reordering",
            submission_io_size_kib=176, submission_batch_size=1,
            per_npu_issue_interval_us=0.1, ssd_command_service="One nonpreemptive command at a time",
            per_path_queue="FIFO", disk_bandwidth_gib_s=40,
            npu_receive_bandwidth_gib_s=50, layers=8, batch_size=1,
            cross_request_layer0_prefetch=True,
            source="continuous_prefill_client.py:22-23; sim.py:1113-1137; run_coflow_experiments.py:178-184"),
        excluded_variants=dict(once_native="Different runner uses pressure_ttl_ms=0",
            new_once="Uses other-client reservations; not selected",
            short_first="Diagnostic per-layer-size priority; not Once"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fifo-stage", type=Path, default=OUT / "formal_random_s1")
    parser.add_argument("--once-stage", type=Path, default=OUT / "formal_random_s1_once")
    parser.add_argument("--window", nargs=2, type=float, default=[2000.0,4000.0])
    parser.add_argument("--configuration-only", action="store_true")
    args = parser.parse_args()
    args.once_stage.mkdir(parents=True, exist_ok=True)
    config_path = args.once_stage / "once_configuration.json"
    config = configuration_document()
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    if args.configuration_only:
        print(json.dumps(dict(configuration=str(config_path), all_reference_hashes_match=True)))
        return
    document = dict(method="Independent raw frozen manifest plus microbatch layer interval audit; no simulator rerun.",
                    window_ms=args.window, policies={})
    raw = {}
    for policy, stage in (("fifo",args.fifo_stage),("once",args.once_stage)):
        directory = locate(stage, policy)
        audited, result, manifest = audit_run(directory, args.window)
        audited["raw_positive_stall_groups"] = raw_stall_examples(result, manifest, args.window)
        audited["input_directory"] = str(directory)
        document["policies"][policy] = audited
        raw[policy] = result, manifest
    fifo, fmanifest = raw["fifo"]
    once, omanifest = raw["once"]
    checks = dict(requests_exactly_identical=fmanifest["requests"] == omanifest["requests"],
        placements_exactly_identical=fmanifest["placements"] == omanifest["placements"],
        input_fingerprint_equal=fifo["input_fingerprint"] == once["input_fingerprint"],
        logical_input_fingerprint_equal=fifo["logical_input_fingerprint"] == once["logical_input_fingerprint"],
        execution_placement_fingerprint_equal=fifo["execution_placement_fingerprint"] == once["execution_placement_fingerprint"],
        core_and_policy_sha256_equal=fifo["core_and_policy_sha256"] == once["core_and_policy_sha256"],
        current_core_files_match_recorded_hashes=all(
            hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == digest
            for name, digest in once["core_and_policy_sha256"].items()),
        once_uses_5ms_collector=once["collector_interval_ms"] == 5,
        once_has_no_runtime_cir_writes=not once["adapter_statistics"]["cir_write_events"],
        once_has_no_npu_reassignments=not once["assignment_log"],
        all_29_reference_core_hashes_match=all(config["reference_source_hash_checks"].values()))
    require(all(checks.values()), f"Paired validation failed: {checks}")
    document["paired_checks"] = checks
    a, b = document["policies"]["fifo"], document["policies"]["once"]
    document["window_difference"] = dict(fleet_improvement_pp=b["U_percent"]-a["U_percent"],
        short_stall_saved_card_ms=a["by_role"]["S"]["stall_ms"]-b["by_role"]["S"]["stall_ms"],
        long_stall_added_card_ms=b["by_role"]["L"]["stall_ms"]-a["by_role"]["L"]["stall_ms"],
        net_stall_saved_card_ms=a["stall_card_ms"]-b["stall_card_ms"])
    document["interpretation_limits"] = [
        "Window admission cohorts may differ even with identical frozen request decks.",
        "Layer timing locates exposed stalls but does not identify every preceding SSD command.",
        "S-to-L physical read lower bounds are only part of actual waiting; additional waiting is not automatically unavoidable.",
        "Nominal admitted-request D/C demand does not separately add a next-request L0 prefetch burst."]
    path = args.once_stage / "once_comparison_audit.json"
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(output=str(path), paired_checks=checks,
        window_difference=document["window_difference"],
        once_U_percent=b["U_percent"], once_by_role=b["by_role"],
        once_SLO=b["SLO_1p5_admission_cohort"],
        once_layer_stalls=b["window_stall_by_role_and_layer_card_ms"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
