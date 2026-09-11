#!/usr/bin/env python3
"""Export the four frozen seed-7 role assignments, without running simulation.

Only writes input_assignment.md, input_assignment_by_npu.csv,
input_assignment_audit.json and input_sequences/*.csv in this directory.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(ROOT))
from run_baseline_npu32_stress import load_manifest

PROFILES = ((1, 128), (1, 256), (1, 384), (192, 768))
TYPES = ("S1", "S2", "S3", "L")
EXPECTED = dict(zip(TYPES, (6400, 6400, 6400, 256)))
SEED = 7
MUTABLE = {"request_id", "npu_id", "generation", "original_request_id", "source_original_npu_id"}


def read(path):
    with (gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)) as stream:
        return json.load(stream)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def kind(row):
    return TYPES[PROFILES.index((row["load"]["seq_len_k"], row["load"]["nql"]))]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def semantic(row, placements):
    return {"arrival_time_ms": row["arrival_time_ms"],
            "load": {k:v for k,v in row["load"].items() if k not in MUTABLE},
            "placement": placements[row["placement_index"]]}


def expected_allocation(source_rows, n_long):
    """Reproduce the documented runner rule independently on raw dictionaries."""
    lanes = [[] for _ in range(32)]
    for p, name in enumerate(TYPES):
        pool = sorted((r["request_id"] for r in source_rows if kind(r) == name))
        random.Random(SEED*1000003 + n_long*10007 + p*101 + 17).shuffle(pool)
        targets = list(range(n_long)) if name == "L" else list(range(n_long,32))
        for i, oid in enumerate(pool):
            lanes[targets[i % len(targets)]].append(oid)
    return lanes


def expected_order(original_ids, source_by_id, npu, order):
    if kind(source_by_id[original_ids[0]]) == "L":
        return sorted(original_ids)
    if order == "random":
        result = sorted(original_ids)
        random.Random(SEED*1000003 + npu*100003 + 71923).shuffle(result)
        return result
    queues = [sorted(oid for oid in original_ids if kind(source_by_id[oid]) == name) for name in TYPES[:3]]
    return [q[i] for i in range(max(map(len, queues))) for q in queues if i < len(q)]


def main():
    plan_path = HERE / "plan.json"
    source_path = BASE / "inputs" / "concurrency_l768_seed7.json.gz"
    plan = read(plan_path); source = read(source_path)
    # The normal loader independently recomputes the simulator fingerprint.
    load_manifest(source_path)
    source_by_id = {r["request_id"]:r for r in source["requests"]}
    assert len(source_by_id) == len(source["requests"]) == 19456
    assert Counter(kind(r) for r in source["requests"]) == EXPECTED
    items = [i for i in plan["inputs"] if i["seed"] == SEED]
    assert {(i["long_npu_count"], i["order"]) for i in items} == {(n,o) for n in (11,6) for o in ("random","round_robin")}
    monitored = {str(plan_path.relative_to(ROOT)):sha(plan_path), str(source_path.relative_to(ROOT)):sha(source_path),
                 str(Path(__file__).relative_to(ROOT)):sha(__file__)}
    monitored.update(plan["source_sha256"])
    source_checks = {name:sha(ROOT/name) == value and sha(HERE/"sources"/name) == value for name,value in plan["source_sha256"].items()}
    assert all(source_checks.values()), "Construction sources differ from frozen plan"
    all_lanes=[]; cases=[]; signatures={}; outputs=[]
    profile_rows=[]
    for name in TYPES:
        group=[r for r in source["requests"] if kind(r)==name]
        row=group[0]; load=row["load"]; placement=source["placements"][row["placement_index"]]
        volumes=[math.fsum(v for _,v in layer) for layer in placement]
        assert len(placement) in (1,8) and all(math.isclose(v,volumes[0],abs_tol=1e-12) for v in volumes)
        assert all(r["load"]["per_layer_us"] == load["per_layer_us"] for r in group)
        profile_rows.append({"type":name,"seq_len_k":load["seq_len_k"],"nql":load["nql"],"category":load["category"],
            "role":load["role"],"per_layer_compute_ms":load["per_layer_us"]/1000,
            "request_8layer_compute_ms":8*load["per_layer_us"]/1000,
            "per_layer_physical_read_mib":volumes[0]*1024,"blocks_per_layer":len(placement[0]),
            "global_request_count":len(group),"profile_construction":load["profile_construction"]})
    for item in items:
        label, n_long, order = item["label"], item["long_npu_count"], item["order"]
        path=HERE/"inputs"/(label+".json.gz"); data=read(path); meta=data["metadata"]
        load_manifest(path)
        monitored[str(path.relative_to(ROOT))]=sha(path)
        checks={"file_hash_matches_frozen_plan":sha(path)==item["manifest_sha256"],
            "fingerprint_matches_frozen_plan_and_metadata":data["input_fingerprint"]==item["input_fingerprint"]==meta["input_fingerprint"],
            "source_manifest_sha_and_fingerprint":meta["source_manifest_sha256"]==sha(source_path) and meta["source_input_fingerprint"]==source["input_fingerprint"],
            "dimensions_32_npu_6_ssu_8_layers":(meta["num_npu"],meta["num_ssu"],meta["n_layers"])==(32,6,8),
            "frozen_label_seed_ratio_order":(meta["label"],meta["seed"],meta["long_npu_count"],meta["order"])==(label,7,n_long,order),
            "whole_original_population_bijection":False, "unchanged_scientific_fields_and_exact_placement":True,
            "position_id_and_original_card_mapping":True, "all_arrival_zero":True,
            "exact_profile_stratified_allocation":True,"exact_documented_order":True,
            "per_card_role_and_counts_match_formula_and_metadata":True,"per_card_pure_compute_matches_metadata":True}
        lanes=defaultdict(list);seen=set();sequence=[];signature={};placement_hashes={i:digest(p) for i,p in enumerate(data["placements"])}
        for row in data["requests"]:
            oid=row["load"]["original_request_id"]
            assert oid not in seen and oid in source_by_id
            seen.add(oid); old=source_by_id[oid]
            checks["unchanged_scientific_fields_and_exact_placement"] &= semantic(row,data["placements"]) == semantic(old,source["placements"])
            checks["all_arrival_zero"] &= row["arrival_time_ms"]==0
            checks["position_id_and_original_card_mapping"] &= row["load"]["source_original_npu_id"]==old["npu_id"]
            lanes[row["npu_id"]].append(row)
        checks["whole_original_population_bijection"] = seen==set(source_by_id) and len(data["requests"])==19456
        assert Counter(kind(r) for r in data["requests"])==EXPECTED
        allocated=expected_allocation(source["requests"],n_long)
        for n in range(32):
            lane=sorted(lanes[n],key=lambda r:r["request_id"]); ids=[r["load"]["original_request_id"] for r in lane]
            role="long" if n<n_long else "short"; counts=Counter(kind(r) for r in lane)
            checks["exact_profile_stratified_allocation"] &= sorted(ids)==sorted(allocated[n])
            checks["exact_documented_order"] &= ids==expected_order(allocated[n],source_by_id,n,order)
            targets=n_long if role=="long" else 32-n_long; local=n if role=="long" else n-n_long
            q,rem=divmod(256 if role=="long" else 6400,targets)
            expected_counts={name:((q+(local<rem)) if (name=="L")== (role=="long") else 0) for name in TYPES}
            meta_lane=meta["per_npu_assignment"][n]
            checks["per_card_role_and_counts_match_formula_and_metadata"] &= (
                all(r["load"]["role"]==role for r in lane)
                and [counts[name] for name in TYPES]==[expected_counts[name] for name in TYPES]==meta_lane["profile_counts"]
                and meta_lane["npu_id"]==n and meta_lane["assigned_role"]==role and meta_lane["request_count"]==len(lane))
            pure=8*math.fsum(r["load"]["per_layer_us"]/1000 for r in lane)
            checks["per_card_pure_compute_matches_metadata"] &= math.isclose(pure,meta_lane["ideal_compute_ms"],abs_tol=1e-8)
            signature[n]={"original_ids":sorted(ids),"original_order":ids,"content_by_original_id":{},"profile_counts":dict(counts)}
            all_lanes.append({"label":label,"seed":7,"long_npus":n_long,"short_npus":32-n_long,"order":order,
                "npu_id":n,"assigned_role":role,**{name+"_count":counts[name] for name in TYPES},
                "request_count":len(lane),"pure_8layer_compute_ms":pure,
                "first_16_types":" ".join(kind(r) for r in lane[:16]),
                "first_16_original_request_ids":" ".join(str(x) for x in ids[:16]),
                "original_population_sha256":digest(sorted(ids)),"original_order_sha256":digest(ids)})
            for position,row in enumerate(lane):
                load=row["load"];oid=load["original_request_id"];placement=data["placements"][row["placement_index"]]
                checks["position_id_and_original_card_mapping"] &= (row["request_id"]==n*1000000+position==load["request_id"]
                    and load["generation"]==position and row["npu_id"]==load["npu_id"]==n)
                sig=digest({"arrival":row["arrival_time_ms"],"load":{k:v for k,v in load.items() if k not in MUTABLE},
                    "placement_sha256":placement_hashes[row["placement_index"]]})
                signature[n]["content_by_original_id"][oid]=sig
                volumes=[math.fsum(v for disk,v in placement[0] if disk==s) for s in range(6)]
                sequence.append({"label":label,"seed":7,"order":order,"npu_id":n,"assigned_role":role,
                    "queue_position_0based":position,"queue_position_1based":position+1,"type":kind(row),
                    "request_id":row["request_id"],"original_request_id":oid,"source_original_npu_id":load["source_original_npu_id"],
                    "seq_len_k":load["seq_len_k"],"nql":load["nql"],"category":load["category"],"arrival_ms":row["arrival_time_ms"],
                    "n_layers":8,"per_layer_compute_ms":load["per_layer_us"]/1000,"request_pure_compute_ms":8*load["per_layer_us"]/1000,
                    "per_layer_physical_read_mib":math.fsum(volumes)*1024,"blocks_per_layer":len(placement[0]),
                    "placement_sha256":placement_hashes[row["placement_index"]],
                    **{f"per_layer_ssu{s}_mib":v*1024 for s,v in enumerate(volumes)}})
        assert all(checks.values()), (label,[k for k,v in checks.items() if not v])
        csv_path=HERE/"input_sequences"/(label+".csv");write_csv(csv_path,sequence);outputs.append(csv_path)
        signatures[(n_long,order)]=signature
        cases.append({"label":label,"manifest":str(path.relative_to(HERE)),"manifest_sha256":sha(path),
            "input_fingerprint":data["input_fingerprint"],"request_count":len(sequence),"profile_counts":EXPECTED,
            "total_pure_compute_ms":math.fsum(r["request_pure_compute_ms"] for r in sequence),
            "all_checks_passed":all(checks.values()),"checks":checks,"queue_csv":str(csv_path.relative_to(HERE)),"queue_csv_sha256":sha(csv_path)})
        print(json.dumps({"label":label,"requests":len(sequence),"checks_passed":all(checks.values())}),flush=True)
    pairs=[]
    for n_long in (11,6):
        a,b=signatures[(n_long,"random")],signatures[(n_long,"round_robin")]
        rows=[{"npu_id":n,"same_original_identity_population":a[n]["original_ids"]==b[n]["original_ids"],
            "same_scientific_fields_and_placement":a[n]["content_by_original_id"]==b[n]["content_by_original_id"],
            "same_profile_counts":a[n]["profile_counts"]==b[n]["profile_counts"],
            "identical_original_identity_order":a[n]["original_order"]==b[n]["original_order"]} for n in range(32)]
        passed=all(r["same_original_identity_population"] and r["same_scientific_fields_and_placement"] and r["same_profile_counts"]
                   and (r["identical_original_identity_order"] if r["npu_id"]<n_long else True) for r in rows)
        assert passed
        pairs.append({"long_npus":n_long,"short_npus":32-n_long,"passed":passed,"per_npu":rows})
    per_npu_path=HERE/"input_assignment_by_npu.csv";write_csv(per_npu_path,all_lanes);outputs.append(per_npu_path)
    lines=["# 固定长短角色分卡：seed 7 的实际输入", "",
        "这里导出的是已经完成仿真的四份冻结输入：11 长卡 / 21 短卡、6 长卡 / 26 短卡，各有 random 和 round_robin 两种短卡队列顺序。Baseline 与 Once 使用同一份输入，因此每种输入只导出一次。没有生成新请求，也没有重新运行仿真。", "",
        "所有请求在 t=0 到达，每条请求运行 8 层。每张 NPU 一次处理一条请求，绑定后不跨卡迁移；32 张卡共享同一组 6 块 SSU，并非每类卡拥有专用磁盘。原来的混合实验同样固定 NPU 绑定，这次改变的是把长短角色分别集中到不同卡。短卡只承担 short 角色，但包含 S1、S2、S3 三种不同画像。", "",
        "## 四种请求到底是什么", "", "| 类型 | seq_len / NQL | 路由类别 | 单层 C ms | 一条请求纯计算 8C ms | 每层实际读取 MiB | 全局条数 |", "|---|---|---|---:|---:|---:|---:|"]
    for p in profile_rows:
        lines.append(f"| {p['type']} | {p['seq_len_k']}K / {p['nql']} | {p['category']} | {p['per_layer_compute_ms']:.9f} | {p['request_8layer_compute_ms']:.9f} | {p['per_layer_physical_read_mib']:.6f} | {p['global_request_count']:,} |")
    lines += ["", "全局共 19,456 条：S1/S2/S3 各 6,400 条，L 共 256 条。四份输入与原混合实验保持同一批请求身份、计算时间、读取量和到达时刻。S1–S3 是从原 data 的 32K/48K 外推到 1K 的构造画像；S3 另在 NQL256/512 间插值。L 是 192K 的 NQL512/1024 间插值得到 NQL768。它们不是从原 data 原封不动抽出的四行。", "",
        "## 卡号与配额", "", "| 方案 | NPU 卡号（从 0 开始） | 每卡 S1 / S2 / S3 / L | 每卡总条数 |", "|---|---|---|---:|",
        "| 11 长 / 21 短 | 0–2 | 0 / 0 / 0 / 24 | 24 |", "| 11 长 / 21 短 | 3–10 | 0 / 0 / 0 / 23 | 23 |",
        "| 11 长 / 21 短 | 11–26 | 305 / 305 / 305 / 0 | 915 |", "| 11 长 / 21 短 | 27–31 | 304 / 304 / 304 / 0 | 912 |",
        "| 6 长 / 26 短 | 0–3 | 0 / 0 / 0 / 43 | 43 |", "| 6 长 / 26 短 | 4–5 | 0 / 0 / 0 / 42 | 42 |",
        "| 6 长 / 26 短 | 6–9 | 247 / 247 / 247 / 0 | 741 |", "| 6 长 / 26 短 | 10–31 | 246 / 246 / 246 / 0 | 738 |", "",
        "配额来源是按画像分别分发：先把该画像的原请求 ID 排序，再独立打乱身份，随后从该角色最小卡号开始逐卡轮转，直到发完。例如 6,400=21×304+16，所以 21 张短卡中前 16 张每种短画像多一条；三个画像都从同一张短卡重新开始分发，所以余数落在相同卡上。256=11×23+3；6 卡方案则有 256=6×42+4、6,400=26×246+4。", "",
        "分配身份所用 Python RNG 种子为 `seed*1000003 + 长卡数*10007 + profile_index*101 + 17`，其中 profile_index 按 S1、S2、S3、L 为 0、1、2、3。本页 seed=7。先完成分卡，之后才选择 random 或 round_robin，因此两种顺序不会改变某张新卡分到哪些请求。", "",
        "random：先按原请求 ID 排好该短卡完整列表，再用 `Random(seed*1000003 + npu_id*100003 + 71923).shuffle(...)` 整体打乱一次，不是反复复制一个小段。round_robin：把该卡 S1/S2/S3 各自按原 ID 排序，依次从三个队列各取一条；本批每卡三类配额相等，因此类型序列完整重复 S1、S2、S3。两种模式下，长卡都按原 ID 排序：全是同一种 L，连身份顺序也相同。规则来源：[run_role_separated.py](run_role_separated.py#L57)。", "",
        "## 每卡队列与纯计算总时长", "",
        "下表是输入队首 16 条类型，**不是暖窗 2–4 秒内的前 16 条请求**。纯计算总时长是该卡完整队列所有请求的 8C 之和，不含 I/O 等待，不能当作实际完成时间。完整队列见文末 CSV。"]
    lookup={(r["long_npus"],r["order"],r["npu_id"]):r for r in all_lanes}
    for n_long in (11,6):
        lines += ["",f"### {n_long} 长卡 / {32-n_long} 短卡", "", "| NPU | 角色 | S1 / S2 / S3 / L | 完整队列纯 C ms | random 队首16条 | round_robin 队首16条 |", "|---:|---|---|---:|---|---|"]
        for n in range(32):
            a,b=lookup[(n_long,"random",n)],lookup[(n_long,"round_robin",n)]
            lines.append(f"| {n} | {'长' if a['assigned_role']=='long' else '短'} | {' / '.join(str(a[x+'_count']) for x in TYPES)} | {a['pure_8layer_compute_ms']:.9f} | {a['first_16_types']} | {b['first_16_types']} |")
    lines += ["", "## 共享盘与请求身份如何保留", "",
        "重新分卡没有重新计算落盘位置：每条请求的每层逐块 `(SSU编号, 大小)` 列表原样保留，176 KiB 块仍在原来的盘上。原条带偏移来自 `source_original_npu_id//4`，不是新卡号；因此一张新卡可以持有来自不同原卡、条带偏移不同的请求。磁盘仍是所有 NPU 共同竞争的 6 块盘。", "",
        "模拟器用 request_id 决定同到达时刻的卡内排队顺序，因此新 ID 编为 `新NPU*1000000 + 从0开始的位置`，generation 也更新为位置。`original_request_id` 保留科学上的原请求身份，`source_original_npu_id` 保留原卡号。random 与 round_robin 的模拟器输入指纹不同；本次按原身份核对，确认每张新卡的完整人口、所有计算字段、到达和逐块 placement 相同。不能把新 ID 相同直接当作同一个原请求。", "",
        "四份输入所有卡的完整纯计算总量均超过 4 秒。已有运行审计确认暖窗 [2,4)s 中 32 卡都持续 active，且每卡仅计算指定角色；本导出不重新计算运行性能。每卡只有一种角色是本干预的目标，并非要求每卡只有一个短画像。", "",
        "## 完整可复算文件", "", "- [每卡配额、前16条与纯C CSV](input_assignment_by_npu.csv)：128 行，4 份输入各 32 张卡。",
        "- [输入与配对审核 JSON](input_assignment_audit.json)：冻结文件哈希、输入指纹、原人口双射、独立重建分配/排序及 64 张卡的 random/RR 配对核对。"]
    for case in cases:
        lines.append(f"- [{case['label']} 完整队列]({case['queue_csv']})：19,456 行，按新卡号、队列位置排序；含原身份、原卡号、画像、C、逐盘 V 与 placement 哈希。")
    lines += ["", "复算命令：`python -B export_input_assignment.py`。四份完整队列合计 77,824 行（不含表头），没有对相同请求跨配置的重复出现去重。脚本只写本说明及其导出文件，不修改冻结 plan、输入、runner 或既有分析。", ""]
    markdown_path=HERE/"input_assignment.md";markdown_path.write_text("\n".join(lines),encoding="utf-8");outputs.append(markdown_path)
    unchanged=all(sha(ROOT/name)==value for name,value in monitored.items())
    assert unchanged
    audit={"generated_utc":datetime.now(timezone.utc).isoformat(),"seed":7,"all_checks_passed":unchanged and all(source_checks.values()) and all(c["all_checks_passed"] for c in cases) and all(p["passed"] for p in pairs),
        "scope":"Four already frozen input manifests only; no simulation and no performance re-aggregation.",
        "source_paths_relative_to":str(ROOT),"read_only_source_sha256":monitored,"construction_source_checks":source_checks,
        "sources_unchanged_during_export":unchanged,"profiles":profile_rows,"inputs":cases,"random_round_robin_per_card_pairs":pairs,
        "by_npu_csv_rows":len(all_lanes),"complete_queue_csv_rows":sum(c["request_count"] for c in cases),
        "outputs":{str(p.relative_to(HERE)):sha(p) for p in outputs}}
    (HERE/"input_assignment_audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"all_checks_passed":audit["all_checks_passed"],"inputs":len(cases),"per_npu_rows":len(all_lanes),"queue_rows":audit["complete_queue_csv_rows"]}),flush=True)


if __name__ == "__main__":
    main()
