"""Same requests/placement, independent per-card shuffle, frozen OD runtime.

Run from any working directory.  Only files under order_control are written.
The original request ID remains the placement identity; the new request ID is
only the FIFO position because the native arrival queue sorts by request ID.
"""
from pathlib import Path
from unittest.mock import patch
from collections import Counter
import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / "local_overload"
RUNTIME = REFERENCE / "runtime"
SOURCE_MANIFEST = REFERENCE / "inputs/a128_b128_r12.json.gz"
REFERENCE_RUN = REFERENCE / "runs/a128_b128_r12_od_baseline_full"
TARGET = HERE / "runs/shuffled_seed7_od_baseline_full"
WINDOWS = [(2000., 4000.), (4000., 8000.), (8000., 12000.)]
sys.path[:0] = [str(RUNTIME), str(HERE)]

from simulator.api import run_simulation
from simulator.core import continuous_batch_sim as core
from inputs.manifest import load_manifest, save_manifest, read_json, write_json
from metrics import summarize, exact_demand, live_summary, overlap


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_hashes():
    paths = [*sorted(RUNTIME.rglob("*.py")), RUNTIME / "data",
             SOURCE_MANIFEST, Path(__file__), HERE / "metrics.py"]
    return {str(p.relative_to(HERE.parent)): sha(p) for p in paths}


def identity_fingerprint(requests):
    """Ignore FIFO IDs only, retaining all physical/profile/arrival fields."""
    rows = []
    for q in requests:
        identity = int(q.load["original_request_id"])
        load = dict(q.load)
        load["request_id"] = identity
        rows.append([identity, q.npu_id, q.arrival_time_ms, load, q.placement])
    body = json.dumps(sorted(rows), separators=(",", ":"), sort_keys=True,
                      allow_nan=False).encode()
    return hashlib.sha256(body).hexdigest()


def prepare():
    old, meta = load_manifest(SOURCE_MANIFEST)
    assert all(q.arrival_time_ms == 0 for q in old)
    assert len({q.load["original_request_id"] for q in old}) == len(old)
    requests = []
    mapping = []
    queues = []
    for npu in range(32):
        original = sorted((q for q in old if q.npu_id == npu), key=lambda q: q.request_id)
        shuffled = list(original)
        random.Random(7 + npu).shuffle(shuffled)
        assert Counter(q.load["role"] for q in original) == {"A": 11, "B": 22}
        for position, q in enumerate(shuffled):
            rid = npu * 1_000_000 + position
            load = dict(q.load)
            load["request_id"] = rid
            updated = core.ContinuousBatchRequest.from_normalized(
                rid, npu, q.arrival_time_ms, load, q.placement)
            assert updated.placement == q.placement
            assert {k:v for k,v in updated.load.items() if k != "request_id"} == {
                k:v for k,v in q.load.items() if k != "request_id"}
            requests.append(updated)
            mapping.append(dict(npu_id=npu, fifo_position=position, request_id=rid,
                                original_request_id=q.load["original_request_id"],
                                original_fifo_id=q.request_id, role=q.load["role"]))
        queues.append(dict(npu_id=npu, random_seed=7+npu,
                           roles=[q.load["role"] for q in shuffled],
                           original_request_ids=[q.load["original_request_id"] for q in shuffled]))
    requests = tuple(requests)
    fp_old, fp_new = identity_fingerprint(old), identity_fingerprint(requests)
    assert fp_old == fp_new
    assert len({q.request_id for q in requests}) == len(old) == 1056
    assert core.continuous_batch_input_fingerprint(old) != core.continuous_batch_input_fingerprint(requests)
    metadata = dict(meta)
    metadata.update(label="a128_b128_r12_per_npu_shuffle_seed7",
                    order="independent_per_npu_random_shuffle", cycle=None,
                    queue_per_card=None, queue_by_npu=queues,
                    shuffle_algorithm="Python random.Random(7 + npu_id).shuffle(original per-card FIFO)",
                    source_manifest_sha256=sha(SOURCE_MANIFEST),
                    physical_identity_fingerprint=fp_new,
                    placement_identity="Unchanged original_request_id and verbatim immutable placement",
                    request_id_semantics="npu_id*1000000 + new FIFO position; does not rehash placement")
    input_dir = HERE / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    shuffled_path = input_dir / "a128_b128_r12_shuffle_seed7.json.gz"
    save_manifest(shuffled_path, requests, metadata)
    loaded, _ = load_manifest(shuffled_path)
    assert identity_fingerprint(loaded) == fp_old
    with (HERE / "request_identity_mapping.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(mapping[0]))
        writer.writeheader()
        writer.writerows(mapping)
    checks = dict(passed=True, original_manifest=str(SOURCE_MANIFEST),
                  original_manifest_sha256=sha(SOURCE_MANIFEST),
                  shuffled_manifest_sha256=sha(shuffled_path),
                  original_input_fingerprint=core.continuous_batch_input_fingerprint(old),
                  shuffled_input_fingerprint=core.continuous_batch_input_fingerprint(requests),
                  original_physical_identity_fingerprint=fp_old,
                  shuffled_physical_identity_fingerprint=fp_new,
                  request_count=len(requests), requests_per_card=33,
                  role_counts_per_card={"A":11,"B":22},
                  unchanged_fields="npu, arrival, original physical ID, load except FIFO ID, all block placements",
                  pure_compute_ms_per_card=meta["pure_compute_ms_per_card"],
                  expected_blocks=meta["expected_blocks"])
    write_json(HERE / "input_checks.json", checks)
    return requests, metadata, shuffled_path


def preview(ctx, requests):
    summary = live_summary(ctx)
    role_compute = [{"A":0., "B":0.} for _ in range(32)]
    byid = {q.request_id:q for q in requests}
    active = [0.]*32
    for r in summary["request_metrics"]:
        active[r["npu_id"]] += overlap(r["admission_time_ms"],r["completion_time_ms"],2000,4000)
    for b in summary["microbatch_metrics"]:
        q = byid[b["member_request_ids"][0]]
        for layer in b["layer_metrics"]:
            role_compute[q.npu_id][q.load["role"]] += overlap(layer["compute_start_ms"],layer["compute_end_ms"],2000,4000)
    return dict(start_ms=2000.,end_ms=4000.,
                U_percent=100*sum(sum(r.values()) for r in role_compute)/64000.,
                all_npus_active=all(abs(x-2000)<1e-7 for x in active),
                per_npu_role_compute_ms=role_compute,
                mixed_cards=sum(min(r.values())>0 for r in role_compute),
                demand=exact_demand(summary["request_metrics"],byid,2000.,4000.),
                scientific_status="preliminary, before full drain; no final SLO yet")


def run():
    requests, metadata, manifest = prepare()
    TARGET.mkdir(parents=True, exist_ok=False)
    (TARGET / "manifest.json.gz").write_bytes(manifest.read_bytes())
    before = protected_hashes()
    started = last_progress = time.perf_counter()
    completed = 0
    preview_done = False
    busy = [[0.]*3 for _ in WINDOWS]
    record = dict(status="running", pid=os.getpid(), argv=sys.argv,
                  cpu_affinity=sorted(os.sched_getaffinity(0)),
                  source_sha256=before, metadata=metadata,
                  manifest_sha256=sha(manifest),
                  input_fingerprint=core.continuous_batch_input_fingerprint(requests),
                  configuration=dict(strategy="od_baseline",num_npu=32,num_ssu=3,
                    n_layers=8,seed=7,disk_bw_gib_s=40,npu_bw_gib_s=50,
                    collector_interval_ms=5,cross_request_layer0_prefetch=True,
                    od_queue_depth_per_ssu=8192),
                  original_result=str(REFERENCE_RUN / "result.json.gz"))
    write_json(TARGET / "command.json", record)
    original_complete = core._register_complete

    def observe(ctx, flow):
        nonlocal completed, last_progress, preview_done
        completed += 1
        for k, (left,right) in enumerate(WINDOWS):
            busy[k][flow.disk_id] += overlap(flow.ssd_activation_time,flow.link_enqueue_time,left,right)
        ret = original_complete(ctx,flow)
        if not preview_done and ctx.current_time_ms >= 4010:
            pending = [f for n in ctx.npus for f in
                       ([n.link_active_flow] if n.link_active_flow else []) + list(n.link_pending)]
            if not any(f.ssd_activation_time < 4000 for f in pending):
                data = preview(ctx,requests)
                data["SSD_GiB_s"] = [v*40/2000 for v in busy[0]]
                write_json(TARGET / "warm_preview.json",data)
                print(json.dumps(dict(preview=True,U=data["U_percent"],mixed=data["mixed_cards"],
                    demand_mean=data["demand"]["per_disk_mean_GiB_s"],
                    overload_percent=data["demand"]["per_disk_overload_percent"])),flush=True)
                preview_done = True
        if completed % 25000 == 0 and time.perf_counter()-last_progress >= 15:
            progress = dict(simulation_ms=ctx.current_time_ms,completed_blocks=completed,
                            expected_blocks=metadata["expected_blocks"],
                            wall_seconds=time.perf_counter()-started)
            write_json(TARGET / "progress.json",progress)
            print(json.dumps(progress),flush=True)
            last_progress = time.perf_counter()
        return ret

    try:
        with patch.object(core,"_register_complete",observe):
            result = run_simulation(requests,**record["configuration"])
        summary = result["summary"]
        assert completed == summary["completed_blocks"] == metadata["expected_blocks"]
        assert all(summary["invariants"].values())
        analyses = []
        for k,(left,right) in enumerate(WINDOWS):
            row = summarize(summary,requests,left,right)
            row["SSD_GiB_s"] = [v*40/(right-left) for v in busy[k]]
            assert row["all_npus_active"]
            analyses.append(row)
        qdepth = summary["ssd_queue_depth"]
        assert max(map(max,qdepth["peak_outstanding_blocks_by_npu_ssu"])) <= 256
        assert qdepth["host_deferred_blocks_at_stop"] == qdepth["ssd_outstanding_blocks_at_stop"] == qdepth["link_outstanding_blocks_at_stop"] == 0
        assert identity_fingerprint(requests) == metadata["physical_identity_fingerprint"]
        result.update(metadata=metadata,analysis=analyses,source_sha256=before,
                      full_analysis=summarize(summary,requests,0.,summary["makespan_ms"],full=True))
        write_json(TARGET / "result.json.gz",result)
        record.update(status="complete",completed_simulation=True,completed_blocks=completed,
                      makespan_ms=summary["makespan_ms"],result_sha256=sha(TARGET/"result.json.gz"),
                      analysis=analyses)
    except Exception as exc:
        record.update(status="failed",error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        record.update(wall_seconds=time.perf_counter()-started,
                      all_protected_files_unchanged=protected_hashes()==before)
        write_json(TARGET/"command.json",record)
        assert record["all_protected_files_unchanged"]
    comparison()
    print(json.dumps(dict(status=record["status"],wall_seconds=record["wall_seconds"])),flush=True)


def comparison():
    paths = {"original_ABB":REFERENCE_RUN,"shuffled_per_npu_seed7":TARGET}
    rows = []
    for order,path in paths.items():
        if not (path/"result.json.gz").exists():
            continue
        raw = read_json(path/"result.json.gz")
        requests,_ = load_manifest(path/"manifest.json.gz")
        for left,right in WINDOWS:
            row = summarize(raw["summary"],requests,left,right)
            role = row["role_and_stall"]
            rows.append(dict(order=order,start_s=left/1000,end_s=right/1000,
                U_percent=row["U_percent"],SLO_1p5_percent=row["slo"]["percent"],
                admitted_requests=row["slo"]["count"],passed_requests=row["slo"]["passed"],
                all_npus_active=row["all_npus_active"],
                mixed_npus=role["npus_with_A_and_B_compute"],
                per_disk_mean_GiB_s=row["demand"]["per_disk_mean_GiB_s"],
                per_disk_max_GiB_s=row["demand"]["per_disk_max_GiB_s"],
                per_disk_overload_percent=row["demand"]["per_disk_overload_percent"],
                strict_underload=row["demand"]["strict_underload_all_disks"],
                io_stall_card_ms=role["io_stall"]["by_kind_card_ms"],
                per_npu_role_compute_ms=role["per_npu_role_compute_ms"],
                makespan_ms=raw["summary"]["makespan_ms"]))
    complete = len(rows) == 6
    write_json(HERE/"comparison.json",dict(complete=complete,rows=rows,
                input_checks=read_json(HERE/"input_checks.json")))
    if not rows:
        return
    cols = [k for k in rows[0] if k not in ("per_npu_role_compute_ms","io_stall_card_ms")]
    with (HERE/"comparison.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader()
        writer.writerows({k:json.dumps(row[k]) if isinstance(row[k],list) else row[k] for k in cols} for row in rows)
    lines = ["# 同一批请求的卡内随机顺序对照", "",
             "状态："+("两种顺序均完整排空。" if complete else "尚未取得两种顺序的全部正式结果，以下为已完成项。"), "",
             "32 NPU、3 SSU × 40 GiB/s；OD 每盘 8192 深度，每卡每盘固定 256；每请求 8 层。", "",
             "原始输入每卡 ABB 重复 11 次，共 33 请求（A=128K/miss256，B=128K/miss4096），每卡纯计算 16.901 秒。全部 arrival=0。对照只用 Random(7+npu_id) 独立打乱每卡这 33 个请求；改变 FIFO request_id，但每个 original_request_id 的画像、归属卡及完整落盘逐块保持不变。冻结运行时与原实验相同。", "",
             "|顺序|窗口 秒|NPU U|SLO×1.5|活跃卡|实际计算过A/B的卡|逐盘需求峰值 GiB/s|逐盘超40占比|", "|---|---|---:|---:|---|---:|---|---|"]
    for r in rows:
        lines.append(f"|{r['order']}|[{r['start_s']:g},{r['end_s']:g})|{r['U_percent']:.4f}%|{r['SLO_1p5_percent']:.4f}%|{'32' if r['all_npus_active'] else '不足32'}|{r['mixed_npus']}|" + "/".join(f"{x:.3f}" for x in r['per_disk_max_GiB_s']) + "|" + "/".join(f"{x:.3f}%" for x in r['per_disk_overload_percent']) + "|")
    lines += ["", "需求逐事件统计当前已接纳请求 V_i,s/C_i，等待期间仍计入；下一请求 L0 不额外叠加第二份画像需求。逐盘峰值和超限占比均来自实际各自执行轨迹。若超限，此对照不属于严格欠载反例。", "",
              "SLO 使用窗口内接纳请求、跟踪至最终完成，以接纳至 prefill 完成为耗时，自身八层纯计算时间 ×1.5 为门槛；不含接纳前排队，不是真实首 token 时间。NPU U 对窗口内所有计算片段积分。三个固定窗口均不因结果变动。", "",
              "随机对照只有一个输入种子，不可推断所有随机输入；本次保持有限请求总量不变，也不是 60 秒长期平台验证。完整原始输出、逐卡角色计算和逐盘事件区间保存在 runs/；comparison.json/csv 为复算结果；input_checks.json 和 request_identity_mapping.csv 证明只改变 L2 顺序。"]
    (HERE/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--summarize-only",action="store_true")
    args=parser.parse_args()
    comparison() if args.summarize_only else run()
