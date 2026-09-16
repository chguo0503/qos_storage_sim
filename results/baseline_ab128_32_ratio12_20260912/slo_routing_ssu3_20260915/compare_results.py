#!/usr/bin/env python3
"""Recompute SSU3 Random cohorts from complete logs; previews stay separate.

Run directly, without arguments. Only this new study directory is written.
Repeated controls are checked against frozen references and counted once.
Macro means give each seed equal weight; pooled rates give each request equal
weight. Window admission populations differ across strategies despite identical
full input. This is admission-to-prefill-completion SLO, not arrival TTFT.
"""

from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[1]
WINDOWS = (("warm_2_4s", 2000., 4000.), ("long_2_20s", 2000., 20000.))
SOURCES = {}
CONTROLS = []


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    data = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))] = sha_bytes(data)
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def digest(path):
    return SOURCES[str(path.relative_to(ROOT))]


def clip(a, z, left, right):
    return max(0., min(z, right) - max(a, left))


def close(a, b):
    assert math.isclose(a, b, rel_tol=0., abs_tol=1e-7), (a, b)


def stats(rows, requests):
    passed = sum(r["completion_time_ms"] - r["admission_time_ms"] <=
                 1.5 * 8 * requests[r["request_id"]]["load"]["per_layer_us"] / 1000 + 1e-9
                 for r in rows)
    return dict(passed=passed, count=len(rows),
                percent=100 * passed / len(rows) if rows else None)


def flatten_slo(overall, by_role):
    out = dict(slo_passed=overall["passed"], slo_count=overall["count"],
               slo_percent=overall["percent"])
    for role in ("A", "B"):
        out.update({f"{role}_{k}": v for k, v in by_role[role].items()})
    out["B_admission_fraction_percent"] = 100 * by_role["B"]["count"] / overall["count"] if overall["count"] else None
    return out


def load_case(case, reference_commands, origin):
    command = read(case / "command.json")
    if command.get("status") != "complete" or command.get("smoke"):
        return None
    assert command["completed_simulation"] and command["assignment"] == "fixed"
    manifest = read(case / "manifest.json.gz")
    raw = read(case / "result.json.gz")
    assert digest(case / "manifest.json.gz") == command["manifest_sha256"]
    assert digest(case / "result.json.gz") == command["output_sha256"]
    meta = manifest["metadata"]
    seed = meta["seed"]
    ref = reference_commands[seed]
    assert command["core_source_sha256"] == ref["core_source_sha256"]
    assert command["manifest_sha256"] == ref["manifest_sha256"]
    assert (meta["num_npu"], meta["num_ssu"], meta["n_layers"], meta["order"]) == (32, 3, 8, "random")
    assert raw["input_fingerprint"] == manifest["input_fingerprint"]
    assert all(raw["summary"]["invariants"].values())
    assert raw["adapter_statistics"]["reorder_calls"] == 0
    assert raw["adapter_statistics"]["cir_write_events"] == []
    assert raw["adapter_statistics"]["assignment_count"] == 0
    assert raw["modeled_control_latency_ms"] == 0
    assert raw["summary"]["batch_size"] == 1
    if "extension_source_sha256" in command:
        assert command["extension_source_sha256"] == raw["routing_extension"]["source_sha256"]
        for name, expected in command["extension_source_sha256"].items():
            source = HERE/name
            actual = sha_bytes(source.read_bytes())
            assert actual == expected, name
            SOURCES[str(source.relative_to(ROOT))] = actual
    requests = {r["request_id"]: r for r in manifest["requests"]}
    complete = raw["summary"]["request_metrics"]
    assert len(complete) == len(requests) == 3840
    assert {r["request_id"] for r in complete} == set(requests)
    for r in complete:
        assert math.isfinite(r["completion_time_ms"]) and r["arrival_time_ms"] == 0
        close(r["own_compute_ms"], 8 * requests[r["request_id"]]["load"]["per_layer_us"] / 1000)
    for npu in range(32):
        actual = [r["request_id"] for r in sorted(complete, key=lambda r:r["admission_time_ms"]) if requests[r["request_id"]]["npu_id"] == npu]
        expected = [r["request_id"] for r in manifest["requests"] if r["npu_id"] == npu]
        assert actual == expected
    makespan = max(r["completion_time_ms"] for r in complete)
    close(makespan, raw["summary"]["makespan_ms"])
    full_compute = math.fsum(r["own_compute_ms"] for r in complete)
    full_u = 100 * full_compute / (32 * makespan)
    close(full_u, 100 * raw["summary"]["fleet_npu_compute_utilization"])
    rows = []
    for name, left, right in (*WINDOWS, ("full_population", 0., makespan)):
        cohort = complete if name == "full_population" else [r for r in complete if left <= r["admission_time_ms"] < right]
        overall = stats(cohort, requests)
        by_role = {role:stats([r for r in cohort if requests[r["request_id"]]["load"]["role"] == role], requests) for role in ("A", "B")}
        # Actual completions per wall-second are a different population from
        # requests admitted in the window and followed beyond its end.
        finished_in_window = complete if name == "full_population" else [
            r for r in complete if left <= r["completion_time_ms"] < right]
        timely_finishes = stats(finished_in_window, requests)
        compute = math.fsum(clip(l["compute_start_ms"], l["compute_end_ms"], left, right)
                            for b in raw["summary"]["microbatch_metrics"] for l in b["layer_metrics"])
        u = 100 * compute / (32 * (right - left))
        if name != "full_population":
            stored = next(w for w in raw["windows"] if w["start_ms"] == left and w["end_ms"] == right)
            close(u, 100 * stored["mean_npu_utilization"])
            all_active = stored["all_npus_active_whole_window"]
        else:
            close(u, full_u)
            all_active = False  # finite run includes asynchronous tail drain
        if name == "warm_2_4s":
            stored = raw["slo"]["window_admissions"]["admission"]
            assert (overall["passed"], overall["count"]) == (stored["passed"], stored["count"])
        rows.append(dict(policy=command["strategy"], seed=seed, window=name,
            start_ms=left, end_ms=right, U_percent=u, **flatten_slo(overall, by_role),
            completed_after_window_count=sum(r["completion_time_ms"] > right for r in cohort),
            admission_cohort_passed_per_window_second=overall["passed"] / ((right-left)/1000),
            completion_window_count=timely_finishes["count"],
            completion_window_timely_count=timely_finishes["passed"],
            completion_window_timely_per_second=timely_finishes["passed"]/((right-left)/1000),
            full_run_U_percent=full_u, makespan_ms=makespan,
            all_npus_active_whole_window=all_active,
            manifest_sha256=command["manifest_sha256"], result_sha256=command["output_sha256"],
            origin=origin, case=str(case.relative_to(ROOT)),
            request_ids=sorted(r["request_id"] for r in cohort)))
    return rows


def assert_reproduction(previous, repeated):
    for old, new in zip(previous, repeated):
        assert (old["policy"],old["seed"],old["window"]) == (new["policy"],new["seed"],new["window"])
        for field in ("U_percent", "full_run_U_percent", "makespan_ms", "slo_percent", "A_percent", "B_percent"):
            close(old[field], new[field])
        for field in ("slo_passed", "slo_count", "A_passed", "A_count", "B_passed", "B_count", "request_ids"):
            assert old[field] == new[field], field
    CONTROLS.append(dict(policy=repeated[0]["policy"], seed=repeated[0]["seed"],
        reference_case=previous[0]["case"], repeated_case=repeated[0]["case"],
        all_cohorts_and_npu_metrics_reproduced=True))


def deltas(rows, controls=None):
    indexed = {(r["policy"], r["seed"], r["window"]):r for r in (controls or rows)}
    out = []
    for r in rows:
        for control in ("baseline", "once"):
            b = indexed.get((control, r["seed"], r["window"]))
            if b is None or r["policy"] == control:
                continue
            assert r["manifest_sha256"] == b["manifest_sha256"]
            delta = dict(policy=r["policy"], control=control, seed=r["seed"], window=r["window"],
                         same_full_input=True, same_window_request_cohort=r["request_ids"] == b["request_ids"])
            for field in ("U_percent", "slo_percent", "A_percent", "B_percent", "B_admission_fraction_percent"):
                delta[field.replace("_percent", "_delta_pp")] = r[field] - b[field] if r[field] is not None and b[field] is not None else None
            out.append(delta)
    return out


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["policy"], row["window"])].append(row)
    summaries = []
    for (policy, window), part in sorted(groups.items()):
        assert len({r["seed"] for r in part}) == len(part)
        row = dict(policy=policy, window=window, seed_count=len(part), seeds=[r["seed"] for r in part])
        for field in ("U_percent", "slo_percent", "A_percent", "B_percent", "B_admission_fraction_percent", "full_run_U_percent", "makespan_ms", "completion_window_timely_per_second"):
            valid = [r[field] for r in part if r[field] is not None]
            row["macro_mean_" + field] = math.fsum(valid)/len(valid) if valid else None
        for prefix in ("slo", "A", "B"):
            count = sum(r[prefix+"_count"] for r in part)
            passed = sum(r[prefix+"_passed"] for r in part)
            row.update({f"pooled_{prefix}_count":count, f"pooled_{prefix}_passed":passed,
                        f"pooled_{prefix}_percent":100*passed/count if count else None})
        summaries.append(row)
    return summaries


def aggregate_deltas(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["policy"], r["control"], r["window"])].append(r)
    out = []
    for (policy, control, window), part in sorted(groups.items()):
        row = dict(policy=policy, control=control, window=window, paired_seed_count=len(part), seeds=[r["seed"] for r in part])
        for field in part[0]:
            if field.endswith("_delta_pp"):
                valid = [r[field] for r in part if r[field] is not None]
                row["macro_mean_"+field] = math.fsum(valid)/len(valid) if valid else None
        out.append(row)
    return out


def write_json(name, obj):
    (HERE/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def write_csv(name, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r if k != "request_ids"))
    with (HERE/name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items() if k in fields} for r in rows)


def main():
    reference_cases = {seed:STUDY/"validation20s/runs"/f"ssu3_random_k1_sync_seed{seed}"/"baseline" for seed in (7,19,43)}
    commands = {seed:read(case/"command.json") for seed,case in reference_cases.items()}
    # Recorded cores must agree and still be present unchanged in this checkout.
    core = commands[7]["core_source_sha256"]
    for command in commands.values():
        assert command["core_source_sha256"] == core
    for name, expected in core.items():
        actual = sha_bytes((ROOT/name).read_bytes())
        assert actual == expected, name
        SOURCES[name] = actual
    selected = {}
    for seed, case in reference_cases.items():
        selected[("baseline",seed)] = load_case(case,commands,"frozen_reference")
    once_case = STUDY/"once_per_layer_ssu3_seed7/runs/random/once"
    selected[("once",7)] = load_case(once_case,commands,"frozen_reference")
    for policy, u, passed, count in (("baseline",90.67794278896096,264,341), ("once",91.24281038614289,273,343)):
        r = selected[(policy,7)][0]
        close(r["U_percent"],u)
        assert (r["slo_passed"],r["slo_count"]) == (passed,count)
    ignored = []
    previews = []
    for path in sorted((HERE/"runs").glob("*/command.json")):
        command = read(path)
        case = path.parent
        if command.get("smoke"):
            ignored.append(dict(case=str(case.relative_to(ROOT)),reason="smoke"))
            continue
        if command.get("status") == "complete":
            rows = load_case(case,commands,"new_full_run")
            key = (rows[0]["policy"],rows[0]["seed"])
            if key in selected:
                assert_reproduction(selected[key],rows)
            selected[key] = rows
            continue
        ignored.append(dict(case=str(case.relative_to(ROOT)),reason=command.get("status")))
        preview_path = case/"warm_preview.json"
        if not preview_path.exists() or command.get("status") not in ("running",):
            continue
        p = read(preview_path)
        assert p["status"] == "exact_warm_preview_full_run_continues" and p["window_ms"] == [2000.,4000.]
        assert p["observed_at_ms"] >= 4500.
        manifest = read(case/"manifest.json.gz")
        assert digest(case/"manifest.json.gz") == command["manifest_sha256"] == commands[p["seed"]]["manifest_sha256"]
        reqs = {r["request_id"]:r for r in manifest["requests"]}
        assert len(set(p["request_ids"])) == p["slo"]["overall"]["count"]
        assert set(p["request_ids"]) <= set(reqs)
        for role in ("A","B"):
            assert sum(reqs[r]["load"]["role"] == role for r in p["request_ids"]) == p["slo"]["by_role"][role]["count"]
        previews.append(dict(policy=p["policy"],seed=p["seed"],window="warm_2_4s",start_ms=2000.,end_ms=4000.,
            U_percent=p["U_percent"],**flatten_slo(p["slo"]["overall"],p["slo"]["by_role"]),
            manifest_sha256=command["manifest_sha256"],case=str(case.relative_to(ROOT)),request_ids=p["request_ids"],
            status="finalized_warm_cohort_but_full_run_incomplete",observed_at_ms=p["observed_at_ms"],
            SSD_GiB_s=p["SSD_GiB_s"]))
    rows = [r for key in sorted(selected) for r in selected[key]]
    pairs = deltas(rows)
    means = aggregate(rows)
    paired_means = aggregate_deltas(pairs)
    audit = dict(created_utc=datetime.now(timezone.utc).isoformat(),builder_sha256=sha_bytes(Path(__file__).read_bytes()),
        all_checks_passed=True, no_new_simulation=True, num_npu=32,num_ssu=3,order="random",alpha=1.5,
        cohort="Admission in [start,end), followed to completion; full_population includes all 3840 requests",
        metric="8-layer prefill completion minus admission; excludes pre-admission waiting; not measured first-token event",
        macro="Each seed has equal weight; compare paired means when available seed sets differ",
        pooled="Sum pass counts / sum admission counts; differs from equal-seed macro averaging",
        window_bias="Identical full input does not imply identical admitted request cohorts or A/B proportions",
        sources=SOURCES,controls=CONTROLS,ignored_runs=ignored,results=rows,
        paired_deltas=pairs,macro_and_pooled=means,paired_macro_deltas=paired_means)
    write_json("comparison.json",audit)
    write_csv("comparison.csv",rows)
    write_csv("paired_deltas.csv",pairs)
    write_csv("macro_summary.csv",means)
    write_csv("paired_macro_deltas.csv",paired_means)
    write_json("preview.json",dict(created_utc=audit["created_utc"],
        warning="Warm admission cohort reported complete by live runner; final full logs/invariants not yet available. Never pooled with complete runs.",
        sources=SOURCES,results=previews,paired_deltas=deltas(previews,rows)))
    write_csv("preview.csv",previews)
    print(json.dumps(dict(complete_policy_seed_pairs=len(selected),complete_rows=len(rows),previews=len(previews),
        reproduced_controls=len(CONTROLS),outputs=str(HERE)),ensure_ascii=False))


if __name__ == "__main__":
    main()
