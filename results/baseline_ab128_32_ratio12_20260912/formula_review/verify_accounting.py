#!/usr/bin/env python3
"""Offline reproduction of tutorial equations 10, 17--25 for six complete logs.

Standard library only; no simulator imports, new simulation, or block-trace reads.
Run: python -B results/baseline_ab128_32_ratio12_20260912/formula_review/verify_accounting.py
Writes only sibling accounting_checks.json.
"""
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY, ROOT = HERE.parent, HERE.parents[2]
OUT = HERE / "accounting_checks.json"
N, LAYERS, DISK_GIB_S, ALPHA = 32, 8, 40.0, 1.5
PROFILE_KEYS = {"A": (128, 256), "B": (32, 4096)}
CHECKS = []


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    with (gzip.open if str(path).endswith(".gz") else open)(path, "rt") as stream:
        return json.load(stream)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def check(name, passed, **detail):
    CHECKS.append(dict(name=name, passed=bool(passed), **detail))


def near(name, actual, expected, abs_tol=1e-6, rel_tol=1e-10):
    check(name, math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol),
          actual=actual, expected=expected, error=actual-expected,
          absolute_tolerance=abs_tol, relative_tolerance=rel_tol)


def overlap(a, b, left, right):
    return max(0.0, min(b, right)-max(a, left))


def cohort(requests, info, right, existing=None, label=""):
    ids = sorted(r["request_id"] for r in requests)
    out = dict(count=len(ids), request_ids_sha256=canonical_sha(ids),
               completion_after_window_count=sum(r["completion_time_ms"] > right for r in requests))
    for clock in ("admission", "arrival"):
        successes = [r["request_id"] for r in requests
                     if r["completion_time_ms"]-r[clock+"_time_ms"] <=
                     ALPHA*LAYERS*info[r["request_id"]]["C_ms"]+1e-9]
        out[clock+"_clock"] = dict(passed=len(successes), count=len(ids),
            slo_percent=100*len(successes)/len(ids) if ids else None,
            passing_ids_sha256=canonical_sha(sorted(successes)))
        if existing is not None:
            old = existing[clock+"_clock"]
            check(label+"."+clock+".counts", old["count"] == len(ids) and old["passed"] == len(successes))
            if ids:
                near(label+"."+clock+".rate", out[clock+"_clock"]["slo_percent"], old["slo_percent"])
            else:
                check(label+"."+clock+".empty", old["slo_percent"] is None)
    if existing is not None:
        check(label+".population", existing["request_ids"] == ids and existing["count"] == len(ids) and
              existing["completion_after_window_count"] == out["completion_after_window_count"])
    return out


def one_case(ssu, order, profiles):
    label = f"ssu{ssu}_{order}_k1_sync_seed7"
    folder = STUDY/"validation20s"/"runs"/label/"baseline"
    paths = {"manifest": folder/"manifest.json.gz", "result": folder/"result.json.gz",
             "command": folder/"command.json", "existing_audit": folder/"audit"/"audit.json"}
    sources = {k: dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for k, p in paths.items()}
    m, raw, command, old = (read(paths[k]) for k in ("manifest", "result", "command", "existing_audit"))
    s, meta = raw["summary"], m["metadata"]
    check(label+".complete", command["status"] == "complete" and command["returncode"] == 0 and
          command["completed_simulation"] and old["technical_passed"] and
          all(v is True for v in s["invariants"].values()))
    check(label+".configuration", raw["strategy"] == "baseline" and raw["submit_seed"] == 7 and
          meta["order"] == order and meta["explicit_A_count"] == 40 and s["num_npu"] == N and
          s["num_ssu"] == ssu and s["n_layers"] == LAYERS and s["batch_size"] == 1 and
          meta["disk_bw_gib_s"] == DISK_GIB_S and meta["npu_bw_gib_s"] == 50)
    check(label+".source_references",
          old["source"]["manifest_sha256"] == sources["manifest"]["sha256"] and
          old["source"]["result_sha256"] == command["output_sha256"] == sources["result"]["sha256"] and
          meta["source_data_sha256"] == command["source_data_sha256"] == sha(ROOT/"data") and
          m["input_fingerprint"] == raw["input_fingerprint"] == command["input_fingerprint"])
    check(label+".core_hashes", raw["core_and_policy_sha256"] == command["core_source_sha256"] and
          all(sha(ROOT/path) == digest for path, digest in raw["core_and_policy_sha256"].items()))
    sources["recorded_core_and_policy_sha256"] = raw["core_and_policy_sha256"]
    sources["recorded_stress_runner_sha256"] = raw["stress_runner_sha256"]
    info, identity, disk_bytes, counts = {}, [], [0]*ssu, defaultdict(Counter)
    for r in m["requests"]:
        load, rid = r["load"], r["request_id"]
        role = next(k for k, key in PROFILE_KEYS.items() if key == (load["seq_len_k"], load["nql"]))
        p = profiles[role]
        assert load["role"] == role and load["per_layer_us"]/1000 == p["C_ms"]
        assert load["per_layer_kv_gb"] == p["D_gib"] and r["arrival_time_ms"] == 0
        placement = m["placements"][r["placement_index"]]
        # A single stored vector is reused for all eight layers in this schema.
        assert len(placement) in (1, LAYERS)
        vectors = placement*LAYERS if len(placement) == 1 else placement
        for vector in vectors:
            layer_bytes = 0
            for disk, volume in vector:
                size = int(volume*2**30)
                assert size == volume*2**30 == 176*1024 and 0 <= disk < ssu
                disk_bytes[disk] += size
                layer_bytes += size
            assert layer_bytes == p["D_bytes"]
        info[rid] = dict(role=role, C_ms=p["C_ms"], npu=r["npu_id"])
        counts[r["npu_id"]][role] += 1
        identity.append([load["original_request_id"], r["npu_id"], r["arrival_time_ms"],
                         list(PROFILE_KEYS[role]), p["C_ms"], p["D_gib"], canonical_sha(vectors)])
    check(label+".population", len(info) == len(m["requests"]) == 3840 and
          set(counts) == set(range(N)) and all(x == {"A":40, "B":80} for x in counts.values()) and
          len({x[0] for x in identity}) == 3840)
    requests = s["request_metrics"]
    req_by_id = {r["request_id"]:r for r in requests}
    check(label+".complete_request_identity", len(requests) == len(req_by_id) == 3840 and set(req_by_id) == set(info))
    layers, compute_errors, batch_ids, clock_errors = [], [], [], []
    for batch in s["microbatch_metrics"]:
        assert len(batch["member_request_ids"]) == 1
        rid = batch["member_request_ids"][0]
        batch_ids.append(rid)
        request, p = req_by_id[rid], info[rid]
        previous = request["admission_time_ms"]
        assert batch["npu_id"] == request["npu_id"] == p["npu"]
        metrics = sorted(batch["layer_metrics"], key=lambda x:x["layer"])
        assert [x["layer"] for x in metrics] == list(range(LAYERS))
        for layer in metrics:
            cs, ce = layer["compute_start_ms"], layer["compute_end_ms"]
            assert previous <= cs+1e-7 and cs <= ce
            compute_errors.extend((ce-cs-p["C_ms"], layer["compute_duration_ms"]-p["C_ms"]))
            layers.append(dict(npu=p["npu"], role=p["role"], request_id=rid,
                               layer=layer["layer"], compute_start=cs, compute_end=ce, wait_start=previous))
            previous = ce
        clock_errors.extend((request["completion_time_ms"]-previous,
            request["processing_latency_ms"]-(request["completion_time_ms"]-request["admission_time_ms"]),
            request["latency_ms"]-(request["completion_time_ms"]-request["arrival_time_ms"]),
            request["admission_wait_ms"]-(request["admission_time_ms"]-request["arrival_time_ms"])))
    check(label+".complete_layer_population", Counter(batch_ids) == Counter(info.keys()) and
          len(layers) == s["completed_batch_layers"] == 30720)
    check(label+".individual_compute_and_clock_fields", max(map(abs, compute_errors)) < 1e-7 and
          max(map(abs, clock_errors)) < 1e-7,
          max_compute_error_ms=max(map(abs, compute_errors)), max_clock_error_ms=max(map(abs, clock_errors)))
    T = s["makespan_ms"]
    near(label+".makespan", T, max(r["completion_time_ms"] for r in requests))
    for npu in range(N):
        lane = sorted((r for r in requests if r["npu_id"] == npu), key=lambda r:r["admission_time_ms"])
        assert all(a["completion_time_ms"] <= b["admission_time_ms"]+1e-7 for a,b in zip(lane,lane[1:]))
    WD = sum(p["request_count"]*LAYERS*p["D_gib"] for p in profiles.values())
    WC = math.fsum(p["request_count"]*LAYERS*p["C_ms"] for p in profiles.values())
    actual_C = math.fsum(l["compute_end"]-l["compute_start"] for l in layers)
    actual_ssd = [d["completed_gb"] for d in sorted(s["disk_stats"], key=lambda d:d["ssu_id"])]
    check(label+".eq17_19_exact_bytes", [b/2**30 for b in disk_bytes] == actual_ssd and
          WD == sum(actual_ssd) == s["expected_read_gb"] == s["completed_read_gb"] and
          sum(disk_bytes) == WD*2**30 and s["completed_blocks"] == sum(disk_bytes)//(176*1024))
    near(label+".eq18_compute", actual_C, WC)
    weighted_C = math.fsum(p["request_count"]*LAYERS*p["D_gib"]/p["nominal_gib_s"]*1000 for p in profiles.values())
    near(label+".eq18_weighted_io", weighted_C, WC)
    disk_lower = max(actual_ssd)/DISK_GIB_S*1000
    check(label+".eq20_lower_bound", T >= disk_lower and T >= WC/N)
    near(label+".eq21_full_U_percent", 100*WC/(N*T), s["fleet_npu_compute_utilization"]*100, abs_tol=1e-8)
    for d in s["disk_stats"]:
        near(label+f".ssd{d['ssu_id']}_service_time", d["utilization"]*T,
             d["completed_gb"]/DISK_GIB_S*1000)

    old_windows = {(w["start_ms"],w["end_ms"]):w for w in old["windows"]}
    raw_windows = {(w["start_ms"],w["end_ms"]):w for w in raw["windows"]}
    selected = set(old_windows)|set(raw_windows)|{(0.0,T),(2000.0,12000.0),(2000.0,20000.0)}
    selected |= {(float(t),float(t+2000)) for t in range(2000,20000,2000)}
    windows = []
    for left,right in sorted(selected):
        assert 0 <= left < right <= T+1e-7
        prefix = label+f".window_{left:g}_{right:g}"
        duration = right-left
        card_C, card_active, class_C = [], [], {role:[] for role in profiles}
        for npu in range(N):
            ll = [l for l in layers if l["npu"] == npu]
            card_C.append(math.fsum(overlap(l["compute_start"],l["compute_end"],left,right) for l in ll))
            card_active.append(math.fsum(overlap(r["admission_time_ms"],r["completion_time_ms"],left,right)
                                         for r in requests if r["npu_id"] == npu))
            for role in profiles:
                class_C[role].append(math.fsum(overlap(l["compute_start"],l["compute_end"],left,right)
                                              for l in ll if l["role"] == role))
        compute, active = math.fsum(card_C), math.fsum(card_active)
        stall = math.fsum(overlap(l["wait_start"],l["compute_start"],left,right) for l in layers)
        idle = N*duration-active
        near(prefix+".compute_stall_active", compute+stall, active)
        near(prefix+".eq10_two_forms", compute/(N*duration), math.fsum(c/duration for c in card_C)/N)
        window = dict(start_ms=left,end_ms=right,full_run=(left == 0 and right == T),
            compute_NPU_ms=compute,io_stall_NPU_ms=stall,idle_NPU_ms=idle,active_NPU_ms=active,
            device_utilization_percent=100*compute/(N*duration),compute_ms_by_npu=card_C,
            active_ms_by_npu=card_active,idle_ms_by_npu=[duration-a for a in card_active],
            all_32_active=all(math.isclose(a,duration,abs_tol=1e-7) for a in card_active),
            both_profiles_positive_compute_cards=sum(all(class_C[r][n] > 0 for r in profiles) for n in range(N)),
            class_compute_ms_by_npu=class_C,cohorts={})
        previous_window = old_windows.get((left,right))
        for name,clock in (("window_admissions","admission"),("window_arrivals","arrival")):
            members = [r for r in requests if left <= r[clock+"_time_ms"] < right]
            window["cohorts"][name] = cohort(members,info,right,
                previous_window["cohorts"][name] if previous_window else None,prefix+"."+name)
        if previous_window:
            for field,own in (("compute_ms",compute),("active_ms",active),("idle_ms",idle),
                              ("io_stall_ms",stall),("device_utilization_percent",window["device_utilization_percent"])):
                near(prefix+".existing_audit."+field,own,previous_window[field])
            check(prefix+".existing_audit.scientific_flags",window["all_32_active"] == previous_window["all_32_active"] and
                  window["both_profiles_positive_compute_cards"] == previous_window["mixed_card_count"])
        if (left,right) in raw_windows:
            near(prefix+".raw_device_U",window["device_utilization_percent"],
                 100*raw_windows[(left,right)]["mean_npu_utilization"])
            for npu,value in enumerate(card_C):
                near(prefix+f".raw_compute_npu{npu}",value,raw_windows[(left,right)]["compute_ms_by_npu"][npu])
        windows.append(window)
    full_cohort = cohort(requests,info,T,old["full_population_SLO"],label+".full_population")
    examples = []
    for role in profiles:
        r = min((r for r in requests if 2000 <= r["admission_time_ms"] < 4000 and info[r["request_id"]]["role"] == role),
                key=lambda r:(r["npu_id"],r["request_id"]))
        t0,a,f = (r[k] for k in ("arrival_time_ms","admission_time_ms","completion_time_ms"))
        limit = ALPHA*LAYERS*profiles[role]["C_ms"]
        examples.append(dict(role=role,request_id=r["request_id"],npu_id=r["npu_id"],arrival_ms=t0,
            admission_ms=a,completion_ms=f,pre_admission_queue_ms=a-t0,post_admission_ms=f-a,
            arrival_to_prefill_ms=f-t0,threshold_ms=limit,admission_pass=f-a <= limit+1e-9,
            arrival_pass=f-t0 <= limit+1e-9,eq23_error_ms=(f-t0)-((a-t0)+(f-a))))
    return dict(label=label,num_ssu=ssu,seed=7,order=order,strategy="baseline",sources=sources,
        input_fingerprint=m["input_fingerprint"],same_population_placement_sha256=canonical_sha(sorted(identity)),
        request_count=len(requests),layer_count=len(layers),block_count=s["completed_blocks"],
        eq17=dict(W_D_gib=WD,W_D_bytes=sum(disk_bytes)),
        eq18=dict(W_C_NPU_ms=WC,measured_compute_NPU_ms=actual_C,error_NPU_ms=actual_C-WC,
                  weighted_service_volume_NPU_ms=weighted_C,per_card_ideal_compute_ms=WC/N),
        eq19=dict(per_ssu_input_bytes=disk_bytes,per_ssu_completed_gib=actual_ssd,
                  ssd_service_integral_gib=sum(actual_ssd),NPU_link_completed_gib=s["completed_read_gb"],
                  byte_error=sum(disk_bytes)-int(sum(actual_ssd)*2**30),
                  integral_source="Complete-run disk_stats.completed_gb counters; no sampled trace used"),
        eq20=dict(disk_capacity_gib_s=DISK_GIB_S,aggregate_service_lower_bound_ms=WD/(ssu*DISK_GIB_S)*1000,
            per_ssu_service_lower_bound_ms=[v/DISK_GIB_S*1000 for v in actual_ssd],
            strongest_disk_lower_bound_ms=disk_lower,per_card_compute_lower_bound_ms=WC/N,
            combined_lower_bound_ms=max(disk_lower,WC/N),observed_makespan_ms=T,
            full_U_upper_percent=min(100.0,100*WC/(N*disk_lower))),
        eq21=dict(W_C_NPU_ms=WC,N=N,T_end_ms=T,recomputed_full_device_U_percent=100*WC/(N*T),
            logged_full_device_U_percent=s["fleet_npu_compute_utilization"]*100,
            error_pp=100*WC/(N*T)-100*s["fleet_npu_compute_utilization"],
            per_ssu_utilization_percent=[d["utilization"]*100 for d in s["disk_stats"]]),
        eq10_windows=windows,eq22_25=dict(alpha=ALPHA,n_layers=LAYERS,numerical_threshold_tolerance_ms=1e-9,
            thresholds_ms={r:ALPHA*LAYERS*p["C_ms"] for r,p in profiles.items()},
            full_population=full_cohort,examples=examples))


def main():
    data = ast.literal_eval((ROOT/"data").read_text())
    profiles = {}
    for role,key in PROFILE_KEYS.items():
        row = data[key]
        profiles[role] = dict(raw_data_key=list(key),raw_data_row=list(row),request_count=1280 if role == "A" else 2560,
            per_card_request_count=40 if role == "A" else 80,C_ms=row[1]/1000,D_gib=row[3],D_bytes=int(row[3]*2**30),
            nominal_gib_s=row[0],total_tokens=key[0]*1024,new_tokens=key[1],hit_tokens=key[0]*1024-key[1])
        near("profile_"+role+".D_over_C",row[3]/(row[1]/1e6),row[0])
    report = dict(schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),no_new_simulation=True,
        no_trace_read=True,script_sha256=sha(__file__),source_data=dict(path="data",sha256=sha(ROOT/"data")),
        tutorial=dict(path="docs/L1_L2_L3_NPU_图解教程.md",sha256=sha(ROOT/"docs/L1_L2_L3_NPU_图解教程.md")),
        formulas=["10","17","18a","18b","19","20","21","22","23","24","25"],
        units=dict(storage="GiB (2^30 bytes), despite legacy gb field names",time="ms",compute_work="NPU*ms"),
        scope=dict(cases="Six validation20s seed7 Baseline runs: 3/4/6 SSUs x random/ordered k1_sync",
            windows="Union of existing audited and raw windows, full run, [2,12), [2,20), and every 2s interval from2to20s",
            comparison="L2 queue order changes; Baseline L3 policy stays fixed",
            SLO="Prefill completion proxy, not first-token measurement; window admissions followed to completion",
            capacity="Eq20 is a total physical service-work lower bound, not an instantaneous nominal underload certificate",
            weighting="Each profile's SSD workload divided by its own D/C, never one common weight for mixed profiles",
            input_lengths="Raw data keys denote total input length, not hit-prefix length plus new query"),
        profiles=profiles,cases=[])
    for ssu in (3,4,6):
        for order in ("random","ordered"):
            report["cases"].append(one_case(ssu,order,profiles))
    pairs = []
    for ssu in (3,4,6):
        pair = [c for c in report["cases"] if c["num_ssu"] == ssu]
        passed = (pair[0]["same_population_placement_sha256"] == pair[1]["same_population_placement_sha256"] and
                  pair[0]["eq17"] == pair[1]["eq17"] and pair[0]["eq18"]["W_C_NPU_ms"] == pair[1]["eq18"]["W_C_NPU_ms"] and
                  pair[0]["eq19"]["per_ssu_input_bytes"] == pair[1]["eq19"]["per_ssu_input_bytes"])
        check(f"ssu{ssu}.same_population_work_placement",passed)
        pairs.append(dict(num_ssu=ssu,passed=passed,population_placement_sha256=pair[0]["same_population_placement_sha256"]))
    report.update(pairs=pairs,checks=CHECKS,failed_checks=[c for c in CHECKS if not c["passed"]],
                  check_count=len(CHECKS),case_count=len(report["cases"]),
                  window_count=sum(len(c["eq10_windows"]) for c in report["cases"]))
    report["passed"] = not report["failed_checks"]
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:report[k] for k in ("passed","case_count","window_count","check_count")},ensure_ascii=False))
    assert report["passed"], f"{len(report['failed_checks'])} failed checks; see {OUT}"


if __name__ == "__main__":
    main()
