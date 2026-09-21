#!/usr/bin/env python3
"""Independent, read-only audit of completed raw layer/request records.

Does not import the runner, its audit helpers, or the simulator. Source raw
artifacts are never modified. Missing/incomplete cases are reported, not passed.
"""
import argparse
import ast
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EPS = 1e-8


def read(path):
    with (gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)) as f:
        return json.load(f)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def overlap(a, b, start, end):
    return max(0., min(float(b), end) - max(float(a), start))


def audit_case(path):
    required = ("native_summary.json.gz", "manifest.json.gz", "config.json", "command.json", "metrics.json")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        return dict(status="pending", source=str(path.relative_to(ROOT)), missing=missing)
    original_sha = {name: sha(path / name) for name in required}
    s, manifest, cfg, command, reported = (read(path / name) for name in required)
    original_config_path = ROOT / command["argv"][command["argv"].index("--config") + 1]
    n, disks, layers = int(s["num_npu"]), int(s["num_ssu"]), int(s["n_layers"])
    end = float(s["makespan_ms"])
    meta = {int(row["request_id"]): row["load"] for row in manifest["requests"]}
    requests = s["request_metrics"]
    batches = s["microbatch_metrics"]
    request_ids = [int(row["request_id"]) for row in requests]
    checks = dict(completed_metrics_marker=reported.get("source_unchanged") is True,
        requests_all_drained=len(requests) == len(meta) == s["request_count"] == len(set(request_ids)),
        original_request_ids=set(request_ids) == set(meta),
        original_config_sha=sha(original_config_path) == command["config_sha256"] == reported["config_sha256"],
        normalized_config_equals_manifest=cfg == manifest["metadata"]["config"],
        all_arrivals_zero=all(row["arrival_time_ms"] == 0. for row in manifest["requests"]),
        all_invariants=all(s["invariants"].values()),
        eight_layers=layers == 8 and all(len(batch["layer_metrics"]) == layers for batch in batches),
        manifest_sha=sha(path / "manifest.json.gz") == command["manifest_sha256"] == reported["manifest_sha256"],
        fingerprints_agree=manifest["input_fingerprint"] == s["input_fingerprint"] == reported["input_fingerprint"],
        source_hash_records_agree=command["source_sha256"] == reported["source_sha256"])
    current_sources = {p: (ROOT / p).is_file() and sha(ROOT / p) == value
                       for p, value in command["source_sha256"].items()}
    checks["source_files_currently_match"] = all(current_sources.values())
    expected_blocks = 0
    for row in manifest["requests"]:
        placement = manifest["placements"][row["placement_index"]]
        # Every stored A/B request repeats the same one-layer placement 8 times.
        assert len(placement) == 1
        expected_blocks += layers * len(placement[0])
    checks["block_conservation"] = expected_blocks == s["submitted_blocks"] == s["completed_blocks"]
    active, computes, waits = [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)]
    finish = [0.] * n
    demand_events = defaultdict(lambda: {"start": [], "end": []})
    demand_events[0.]
    demand_events[end]
    rates_by_id = {}
    for row in requests:
        rid, card = int(row["request_id"]), int(row["npu_id"])
        a, b = float(row["admission_time_ms"]), float(row["completion_time_ms"])
        group = meta[rid]["profile_group"]
        active[card].append((a, b, group, rid))
        finish[card] = max(finish[card], b)
        assert meta[rid]["npu_id"] == card
        assert math.isclose(row["own_compute_ms"], layers * meta[rid]["per_layer_us"] / 1000, abs_tol=1e-6)
        rates_by_id[rid] = [v * 2**30 / 1e9 / (meta[rid]["per_layer_us"] / 1e6) for v in meta[rid]["disk_gib"]]
        demand_events[a]["start"].append(rid)
        demand_events[b]["end"].append(rid)
    for batch in batches:
        assert len(batch["member_request_ids"]) == 1
        rid = int(batch["member_request_ids"][0]);card = int(batch["npu_id"])
        group = meta[rid]["profile_group"]
        previous = float(batch["admission_time_ms"])
        for layer in batch["layer_metrics"]:
            a, b = float(layer["compute_start_ms"]), float(layer["compute_end_ms"])
            wait = float(layer["io_barrier_wait_ms"])
            assert math.isclose(b - a, meta[rid]["per_layer_us"] / 1000, abs_tol=1e-7)
            assert math.isclose(a - previous, wait, abs_tol=1e-7)
            computes[card].append((a, b, group, rid))
            if wait > EPS:
                kind = "internal" if int(layer["layer"]) > 0 else ("startup_L0" if meta[rid]["generation"] == 0 else "cross_request_L0")
                waits[card].append((a - wait, a, kind, group, rid))
            previous = b
        assert math.isclose(previous, batch["completion_time_ms"], abs_tol=1e-7)
    for card in range(n):
        active[card].sort();computes[card].sort()
        assert active[card][0][0] == 0.
        assert all(abs(left[1] - right[0]) < EPS for left, right in zip(active[card], active[card][1:]))
        assert all(left[1] <= right[0] + EPS for left, right in zip(computes[card], computes[card][1:]))
    points = sorted(demand_events)
    current, segments = {}, []
    for i, t in enumerate(points[:-1]):
        for rid in demand_events[t]["end"]:
            del current[rid]
        for rid in demand_events[t]["start"]:
            current[rid] = rates_by_id[rid]
        stop = points[i + 1]
        if stop <= t:
            continue
        rates = [math.fsum(values[d] for values in current.values()) for d in range(disks)]
        segments.append((t, stop, rates, len(current)))

    def demand(start, stop):
        integral = [0.] * disks
        low, high = [math.inf] * disks, [-math.inf] * disks
        over, at_cap = [0.] * disks, [0.] * disks
        any_over = any_at_cap = 0.
        violations = []
        active_min = n
        for a, b, rates, num_active in segments:
            dt = overlap(a, b, start, stop)
            if dt <= 0:
                continue
            active_min = min(active_min, num_active)
            flags, equal_flags = [], []
            for d, rate in enumerate(rates):
                integral[d] += dt * rate
                low[d], high[d] = min(low[d], rate), max(high[d], rate)
                flags.append(rate > 40 + EPS)
                equal_flags.append(rate >= 40 - EPS)
                over[d] += dt * flags[-1]
                at_cap[d] += dt * equal_flags[-1]
            any_over += dt * any(flags)
            any_at_cap += dt * any(equal_flags)
            if any(equal_flags):
                violations.append(dict(start_ms=max(a, start), end_ms=min(b, stop), demand_GB_s=rates))
        duration = stop - start
        return dict(mean_GB_s_by_ssu=[v / duration for v in integral], min_GB_s_by_ssu=low,
            max_GB_s_by_ssu=high, overload_ms_by_ssu=over,
            overload_fraction_by_ssu=[v / duration for v in over], at_or_above_capacity_ms_by_ssu=at_cap,
            any_ssu_overload_fraction=any_over / duration,
            any_ssu_at_or_above_capacity_fraction=any_at_cap / duration,
            strict_under40_all_disks=all(v < 40 - EPS for v in high),
            at_or_above_capacity_intervals=violations, minimum_admitted_active_card_count=active_min)

    def window(start, stop):
        if stop > end + EPS:
            return dict(start_ms=start, end_ms=stop, status="unavailable")
        duration = stop - start
        per_card = []
        stalls = {kind: {"card_ms": 0., "interval_count": 0, "A_card_ms": 0., "B_card_ms": 0.}
                  for kind in ("internal", "startup_L0", "cross_request_L0")}
        for card in range(n):
            active_ms = math.fsum(overlap(a, b, start, stop) for a, b, _, _ in active[card])
            group_compute = {g: math.fsum(overlap(a, b, start, stop) for a, b, group, _ in computes[card] if group == g)
                             for g in ("A", "B")}
            group_ids = {g: sorted({rid for a, b, group, rid in computes[card]
                                   if group == g and overlap(a, b, start, stop) > EPS}) for g in ("A", "B")}
            card_stall = 0.
            for a, b, kind, group, _rid in waits[card]:
                dt = overlap(a, b, start, stop)
                if dt > EPS:
                    stalls[kind]["card_ms"] += dt
                    stalls[kind]["interval_count"] += 1
                    stalls[kind][group + "_card_ms"] += dt
                    card_stall += dt
            compute_ms = math.fsum(group_compute.values())
            assert abs(active_ms - compute_ms - card_stall) < 1e-5
            per_card.append(dict(npu_id=card, U_percent=100 * compute_ms / duration,
                compute_ms=compute_ms, active_ms=active_ms, stall_ms=card_stall,
                group_compute_ms=group_compute, group_computing_request_count={g: len(ids) for g, ids in group_ids.items()},
                actual_compute_both_AB=all(value > EPS for value in group_compute.values()),
                active_whole_window=abs(active_ms - duration) < 1e-6))
        cohort = [row for row in requests if start <= row["admission_time_ms"] < stop]
        passed = [row for row in cohort if row["completion_time_ms"] - row["admission_time_ms"] <= 1.5 * row["own_compute_ms"] + 1e-9]
        return dict(status="complete", start_ms=start, end_ms=stop, duration_ms=duration,
            U_percent=100 * math.fsum(card["compute_ms"] for card in per_card) / (n * duration),
            SLO_1p5_percent=100 * len(passed) / len(cohort) if cohort else None,
            admitted_count=len(cohort), passed_count=len(passed),
            cohort_completed_after_window_count=sum(row["completion_time_ms"] > stop for row in cohort),
            all_cards_active=all(card["active_whole_window"] for card in per_card),
            all_cards_actual_AB_compute=all(card["actual_compute_both_AB"] for card in per_card),
            mixed_card_count=sum(card["actual_compute_both_AB"] for card in per_card),
            ends_before_first_npu_finishes=stop <= min(finish) + EPS,
            per_npu=per_card, stall=stalls, nominal_demand=demand(start, stop))

    windows = {"warm_2_4": window(2000., 4000.), "long_2_60": window(2000., 60000.),
               "late_20_60": window(20000., 60000.), "late_40_60": window(40000., 60000.),
               "long_2_90": window(2000., 90000.), "late_60_90": window(60000., 90000.),
               "pre_drain": window(0., min(finish)), "full": window(0., end)}
    bins = [window(float(a), min(float(a + 2000), end)) for a in range(0, math.ceil(end), 2000)]
    for row in bins:
        row["full_2second_window"] = abs(row["duration_ms"] - 2000) < EPS
    checks["warm_U_agrees_reported"] = abs(windows["warm_2_4"]["U_percent"] - reported["warm_U_percent"]) < 1e-8
    checks["warm_SLO_agrees_reported"] = abs(windows["warm_2_4"]["SLO_1p5_percent"] - reported["warm_SLO_1p5_percent"]) < 1e-8
    full_compute = math.fsum(row["compute_ms"] for row in windows["full"]["per_npu"])
    bins_compute = math.fsum(row["U_percent"] / 100 * n * row["duration_ms"] for row in bins)
    checks["per2s_compute_conservation"] = abs(full_compute - bins_compute) < 1e-5
    checks["per2s_SLO_population_conservation"] = sum(row["admitted_count"] for row in bins) == len(requests)
    checks["per2s_SLO_pass_conservation"] = sum(row["passed_count"] for row in bins) == windows["full"]["passed_count"]
    checks["raw_artifacts_unchanged"] = original_sha == {name: sha(path / name) for name in required}
    return dict(status="passed" if all(checks.values()) else "failed", source=str(path.relative_to(ROOT)),
        artifact_sha256=original_sha, manifest_sha256=original_sha["manifest.json.gz"],
        input_fingerprint=s["input_fingerprint"], strategy=command["strategy"], order=cfg["mode"],
        npu_count=n, ssu_count=disks, n_layers=layers, makespan_ms=end,
        request_count=len(requests), completed_blocks=expected_blocks,
        first_npu_finishes_ms=min(finish), final_completion_ms_by_npu=finish,
        per_card_input_role_counts=[dict(Counter(meta[rid]["profile_group"] for _, _, _, rid in intervals)) for intervals in active],
        checks=checks, source_hash_mismatches=[p for p, value in current_sources.items() if not value],
        exact_nominal_demand_segment_count=len(segments), windows=windows, consecutive_2s=bins,
        conclusion="An audited finite trajectory; a low early window alone does not establish long-lived low utilization.")



def audit_profiles(path):
    """Independently derive interpolation weights and exact bytes, without runner imports."""
    cfg=read(path/'config.json'); manifest=read(path/'manifest.json.gz');raw=read(path/'native_summary.json.gz')
    data=ast.literal_eval((ROOT/'data').read_text());data_sha=sha(ROOT/'data')
    command=read(path/'command.json'); input_path=ROOT/command['argv'][command['argv'].index('--config')+1]; original_cfg=read(input_path)
    assert all(cfg[k]==v for k,v in original_cfg.items())
    allowed_defaults={'disk_gb_s':40.0,'npu_gb_s':50.0,'input_counts':cfg['input_counts'],
        'input_counts_semantics':'per-card complete A/B-family counts; per-request profiles may vary'}
    assert all(k in allowed_defaults and cfg[k]==allowed_defaults[k] for k in cfg.keys()-original_cfg.keys())
    catalog=cfg['profiles_catalog'];ks=sorted({k for k,m in data});ms=sorted({m for k,m in data})
    physical=set(); upper=[];families={}; request_lookup={r['request_id']:r for r in manifest['requests']}
    for key,p in catalog.items():
        t,m=p['total_tokens'],p['nql']; construction=p['profile_construction']
        assert isinstance(t,int) and isinstance(m,int) and 32768<=t<=204800 and 64<=m<=4096
        kl=max(k for k in ks if k*1024<=t);kh=min(k for k in ks if k*1024>=t)
        ml=max(x for x in ms if x<=m);mh=min(x for x in ms if x>=m)
        wk=0. if kl==kh else (t/1024-kl)/(kh-kl);wm=0. if ml==mh else (m-ml)/(mh-ml)
        weights=defaultdict(float)
        for k,w1 in [(kl,1-wk),(kh,wk)]:
            for n,w2 in [(ml,1-wm),(mh,wm)]: weights[k,n]+=w1*w2
        weights={k:w for k,w in weights.items() if w>0}
        anchors=construction['anchors'];actual={(a['seq_len_k'],a['nql']):a['weight'] for a in anchors}
        assert set(weights)==set(actual) and len(actual)==len(anchors)
        assert all(abs(actual[k]-weights[k])<1e-12 and actual[k]>=0 for k in actual)
        assert abs(math.fsum(actual.values())-1)<1e-12
        for a in anchors:
            key_data=(a['seq_len_k'],a['nql']);assert a['compute_us']==data[key_data][1]
            if 'original_data_row' in a:assert tuple(a['original_data_row'])==data[key_data]
        expected_c=math.fsum(data[k][1]*w for k,w in weights.items());expected_v=(t-m)*1408/2**30
        assert abs(expected_c-p['compute_us'])<1e-8 and abs(expected_v-p['read_gib'])<1e-15
        assert p['ssd_prefix_tokens']==t-m and abs(p['B_gib_s']-expected_v/(expected_c/1e6))<1e-12
        assert construction['data_sha256']==data_sha and construction['compute_scale']==1 and construction['extrapolated'] is False
        physical.add((t,m));families[key]=p['family'];upper.append(expected_v*2**30/1e9/(expected_c/1e6))
    pure=[];counts=[];profile_counts=Counter()
    for n,seq in enumerate(cfg['per_npu_sequences']):
        counts.append(dict(Counter(families[k] for k in seq))); pure.append(8*math.fsum(catalog[k]['compute_us'] for k in seq)/1000)
        for pos,k in enumerate(seq):
            row=request_lookup[n*1000000+pos];load=row['load'];p=catalog[k];profile_counts[k]+=1
            assert load['profile_key']==k and load['profile_group']==p['family'] and row['npu_id']==n
            assert load['per_layer_us']==p['compute_us'] and load['total_tokens']==p['total_tokens'] and load['nql']==p['nql']
            assert row['arrival_time_ms']==0 and load['generation']==pos
            blocks=manifest['placements'][row['placement_index']][0]
            assert len(blocks)==math.ceil(p['ssd_prefix_tokens']/128)
            assert all(d==0 for d,v in blocks)
            assert all(v==128*1408/2**30 for d,v in blocks[:-1])
            assert abs(math.fsum(v for d,v in blocks)-p['read_gib'])<1e-14
            assert math.isclose(load['disk_gib'][0],p['read_gib'],rel_tol=0.,abs_tol=1e-14)
    bs={b['member_request_ids'][0]:b for b in raw['microbatch_metrics']}
    max_prefetch_error=0.;prefetch_checks=0
    for n,seq in enumerate(cfg['per_npu_sequences']):
        for pos in range(1,len(seq)):
            prev=bs[n*1000000+pos-1]['layer_metrics'][-1];cur=bs[n*1000000+pos]['layer_metrics'][0]
            max_prefetch_error=max(max_prefetch_error,abs(cur['io_start_time_ms']-prev['compute_start_ms']));prefetch_checks+=1
    assert max_prefetch_error<1e-8
    return dict(status='passed',data_sha256=data_sha,catalog_keys=len(catalog),unique_physical_profiles=len(physical),
        physical_profiles_by_family={f:len({(catalog[k]['total_tokens'],catalog[k]['nql']) for k in catalog if families[k]==f}) for f in ['A','B']},
        raw_grid_row_count=sum((p['total_tokens']/1024,p['nql']) in data for p in catalog.values()),
        interpolation_estimates_not_all_direct_measurements=True,per_card_family_counts=counts,
        all_cards_equal_AB_counts=all(c==counts[0] for c in counts),pure_compute_ms_by_card=pure,
        original_config_sha256=sha(input_path),normalized_config_sha256=sha(path/'config.json'),
        normalization_only_added_fields=sorted(cfg.keys()-original_cfg.keys()),
        any_progress_current_request_D_upper_GB_s=cfg['num_npu']*max(upper),
        crossrequest_L0_prefetch_checks=prefetch_checks,crossrequest_L0_start_max_error_ms=max_prefetch_error,
        request_count_by_profile_key=dict(profile_counts))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',default='full_input_aligned_A10B45')
    parser.add_argument('--strategies',nargs='+',default=['asu_baseline','once'])
    parser.add_argument('--output',type=Path)
    parser.add_argument('--markdown',type=Path,default=HERE/'AUDIT.md')
    parser.add_argument('--no-markdown',action='store_true')
    args=parser.parse_args();cases={}
    for policy in args.strategies:
        path=HERE/'runs'/args.case/policy
        try:
            item=audit_case(path)
            if item['status']=='passed':item['profile_validation']=audit_profiles(path)
            cases[policy]=item
        except Exception as exc:cases[policy]=dict(status='failed',error_type=type(exc).__name__,detail=str(exc))
    completed=[r for r in cases.values() if r.get('status')=='passed']
    comparison=bool(len(completed)==len(cases)>1 and len({r['manifest_sha256'] for r in completed})==1)
    status='passed' if len(completed)==len(cases) else 'pending_or_failed'
    result=dict(case=args.case,status=status,paired_byte_identical_manifest=comparison,audit_script_sha256=sha(Path(__file__)),
        original_runs_untouched=True,independent_of_runner_and_model=True,cases=cases,
        definitions=dict(U='Sum exact raw layer compute overlap / N / window duration',
            SLO='Admission cohort in half-open window; complete uncensored completion-admission / own_compute <=1.5',
            demand='Current admitted V_disk/C includes IO stalls, excludes extra next-request L0 term; decimal GB/s',
            pre_drain='[0,earliest final request completion among cards); SLO includes final completion beyond the window if admitted within it'))
    output=args.output or HERE/('audit_'+args.case+'.json');output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# 原生结果独立审计','',f'当前审计输入：`{args.case}`。脚本 `audit_native_aligned.py` 直接读取完整原生层、请求和冻结 manifest；不导入 runner、原统计模块或快速模型。','',
        '## 如何重算','',
        '- 利用率：逐层裁出与统计窗口相交的计算时长，求和后除以 `16 × 窗口长度`。I/O stall 留在分母中。',
        '- SLO×1.5：只按接纳时间选 `[start,end)` cohort；随后使用该请求在完整排空结果中的真实完成时间，比较 `completion-admission <= 1.5 × 8 × 每层计算时间`。不丢弃窗口结束后才完成的请求。',
        '- 欠载：用请求接纳和完成的全部事件切分时间，逐段累加当前请求的真实逐盘读取量/C，包括 stall。全程逐段检查 `<40`；沿用既有定义，不叠加下一请求首层的额外名义需求，因此不等同于任意瞬间物理提交没有突发。',
        '- 原配置 SHA 对照 command 记录；run/config.json 是补入默认容量和请求数后的规范化副本，逐项检查只增加允许的默认字段，并与 manifest metadata 完全相同。',
        '- 逐卡核对连续活跃、A/B实际计算覆盖、compute+stall时间守恒、原始块/请求守恒、冻结文件SHA、106个运行源文件SHA。另独立重算 data 锚点和插值权重、C、真实hit读取量及尾块，核对跨请求L0确实从上一请求末层计算开始。','',
        '## 重算结果','',
        '|策略|窗口（秒）|NPU U|SLO×1.5|达标/接纳|全卡活跃|全卡实际算过A/B|',
        '|---|---|---:|---:|---:|---|---|']
    for policy,r in cases.items():
        if r.get('status')!='passed':
            lines.append(f"|{policy}|尚未完成或审计未通过|—|—|—|—|—|");continue
        for key in ['warm_2_4','long_2_60','late_20_60','late_40_60','long_2_90','late_60_90','pre_drain','full']:
            w=r['windows'][key]
            if w['status']!='complete':
                lines.append(f'|{policy}|{key} 未覆盖|—|—|—|—|—|');continue
            lines.append(f"|{policy}|[{w['start_ms']/1000:.3f},{w['end_ms']/1000:.3f})|{w['U_percent']:.6f}%|{w['SLO_1p5_percent']:.6f}%|{w['passed_count']}/{w['admitted_count']}|{w['all_cards_active']}|{w['all_cards_actual_AB_compute']}|")
    for policy,r in cases.items():
        if r.get('status')!='passed':continue
        pv=r['profile_validation'];d=r['windows']['full']['nominal_demand']
        lines+=['',f"{policy}：{r['request_count']} 请求、{r['completed_blocks']} 完成块；首次有卡排空 {r['first_npu_finishes_ms']/1000:.6f} 秒，全部排空 {r['makespan_ms']/1000:.6f} 秒。全程逐盘需求峰值 {d['max_GB_s_by_ssu']} GB/s，超限时长 {d['overload_ms_by_ssu']} ms。输入 manifest SHA256：`{r['manifest_sha256']}`。",'',
            f"画像为 A/B **两个家族、多个物理画像**：{pv['catalog_keys']} 个目录键，{pv['unique_physical_profiles']} 个不同的总长/miss组合（按家族 {pv['physical_profiles_by_family']}）。不能写成只有两个固定请求画像。计算时间来自 data 网格内插值估计，而非每一个组合都有直接测量行。"]
    lines += ['', '## 证据边界','',
        '- 快速模型用于离线寻找输入，正式数值只来自本表引用的完整原生 raw；模型与其他轨迹的逐层一致校准不替代对新输入的原生验证。',
        '- 当前输入按ASU的确定性时序离线选择少数A画像来校正相位。最终运行不改变请求画像、不插入空闲、不应用规划hook；但它仍是针对固定配置构造的压力输入，不代表随机工作负载。',
        '- 该构造对同刻提交次序的敏感性必须用独立原生控制衡量；若只改变submit seed就恢复高U，便不能称其种子稳健。计算时延抖动的影响仍是另一项尚未测试的变量，不能把提交seed控制替代全部噪声测试。有限轨迹也不能证明无限时间低平台。',
        '- 本审计展示60秒后的窗口和首次排空前的窗口；若剩余未校正输入出现恢复，应保留披露，不能只展示低谷。若后续修订了剩余画像，必须作为新冻结输入、单独完整重放，不能覆盖本次结果。',
        '- 每个2秒窗口的U/SLO、全卡混合覆盖、逐盘需求、内部层/边界等待明细见同名JSON。宽窗SLO按请求计数重新求比，不能平均小窗口百分比。', '',
        f"当前策略配对完整且manifest逐字节一致：`{comparison}`。状态：`{status}`；缺失结果不记为通过。"]
    history_path=HERE/'audit_native_aligned_A10B45_60s.json'
    if args.case!='native_aligned_A10B45_60s' and history_path.is_file():
        historical=read(history_path).get('cases',{}).get('asu_baseline',{})
        if historical.get('status')=='passed':
            ws=historical['windows']
            lines += ['', '## 保留的过程记录', '',
                '上一版 `native_aligned_A10B45_60s` 只针对前60秒校正，完整原生重放已经显示后续恢复。它不是当前主输入，旧raw和审计JSON保持原样。', '',
                '|旧输入 ASU 窗口|U|SLO×1.5|', '|---|---:|---:|']
            for key in ['long_2_60','long_2_90','late_60_90','pre_drain']:
                w=ws[key]
                lines.append(f"|[{w['start_ms']/1000:.3f},{w['end_ms']/1000:.3f})|{w['U_percent']:.6f}%|{w['SLO_1p5_percent']:.6f}%|")
            lines += ['', '最终输入维持同一份4688请求和每卡53A/240B顺序，只修改31次A请求画像。受影响的旧轨迹首个L0预取在59.802756秒，包含7次在60秒前预取的请求；因此最终输入所有窗口都独立重算，没有沿用旧数字。', '',
                '另外，`phase_search/two_profile_tuned` 的75次有界快速筛选中，15条仅两个固定画像的60秒轨迹均在后期恢复到97.55%–99.14%。这是模型筛选记录，不应与本表的原生测量混写，也不能据此证明所有双画像输入均无解。']
    if args.case=='full_input_aligned_A10B45':
        control_names=['full_input_aligned_submitseed17','full_input_pool_random_inputseed7','full_input_pool_random_inputseed17','full_input_pool_random_inputseed27']
        control_rows=[]
        for name in control_names:
            path=HERE/('audit_'+name+'.json')
            if not path.is_file():continue
            audited=read(path).get('cases',{}).get('asu_baseline',{})
            if audited.get('status')!='passed':continue
            w=audited['windows']['long_2_60']
            control_rows.append(dict(case=name,U_percent=w['U_percent'],SLO_1p5_percent=w['SLO_1p5_percent'],passed=w['passed_count'],admitted=w['admitted_count'],warm_mixed_card_count=audited['windows']['warm_2_4']['mixed_card_count'],warm_nonmixed_npu_ids=[p['npu_id'] for p in audited['windows']['warm_2_4']['per_npu'] if not p['actual_compute_both_AB']],audit_json_sha256=sha(path)))
        if control_rows:
            result['independently_audited_ASU_controls']=control_rows
            output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
            lines += ['', '## 最终输入的敏感性与顺序控制（原生证据）', '',
                '四个控制已分别从完整原生层/request记录重算。三个random控制与最终输入的逐卡物理画像多重集、data来源和每卡数量完全一致，仿真提交seed保持7；它们只打乱每卡顺序。submitseed17控制保留最终请求序列及每个画像，只改变同刻I/O提交的打散种子。配置多重集证据见 `audit_full_input_config.json`。', '',
                '|ASU 控制|U [2,60)|SLO×1.5 [2,60)|达标/接纳|warm实际A/B卡数|', '|---|---:|---:|---:|---:|']
            for c in control_rows:
                lines.append(f"|{c['case']}|{c['U_percent']:.6f}%|{c['SLO_1p5_percent']:.6f}%|{c['passed']}/{c['admitted']}|{c['warm_mixed_card_count']}/16|")
            seed_control=next((c for c in control_rows if c['case']=='full_input_aligned_submitseed17'),None)
            if seed_control is not None and cases.get('asu_baseline',{}).get('status')=='passed':
                u=cases['asu_baseline']['windows']['long_2_60']['U_percent']
                lines += ['', f"仅改变提交seed，U就从 {u:.6f}% 恢复到 {seed_control['U_percent']:.6f}%。因此主case证明了固定配置下可复现的长期低利用率构造，但**没有证明对提交次序扰动稳健**。随机同池结果也不支持‘普通随机输入通常也这么差’。", '',
                    '随机inputseed7和27的warm各只有15卡实际计算A/B：分别是NPU id3与id11没有A计算（id从0开始）。这两组不能算满足‘warm全16卡混合’的正式输入，只作为顺序敏感性对照。随机inputseed17和提交submitseed17对照的warm均16卡混合；后者恢复高U不能归因于warm缺类。',
                    '各策略在同一时间窗接纳的请求集合可能不同；这里按各自完整admission cohort统计SLO，不把窗口样本视为逐请求一一配对。控制组在90秒前已经排空，因此其[2,90)标记为未覆盖，未通过补零制造低利用率。']
    if not args.no_markdown:
        args.markdown.write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=status,cases={p:dict(status=r.get('status'),first_drain=r.get('first_npu_finishes_ms'),windows={k:{f:v.get(f) for f in ['U_percent','SLO_1p5_percent','admitted_count','passed_count','all_cards_active','mixed_card_count']} for k,v in r.get('windows',{}).items()}) for p,r in cases.items()}),indent=2))

if __name__=='__main__':main()
