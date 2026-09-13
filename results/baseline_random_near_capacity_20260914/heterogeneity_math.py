#!/usr/bin/env python3
"""Frozen-work, fixed-coarse-order heterogeneity checks; mathematics only."""
import ast
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SEEDS=[7,19,43,67,101]
BLOCK_GIB=176/1048576


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def fit(raw,miss):
    rows=sorted((k[0],v[1]/1000) for k,v in raw.items() if k[1]==miss)
    mx=statistics.mean(x for x,y in rows);my=statistics.mean(y for x,y in rows)
    b=math.fsum((x-mx)*(y-my) for x,y in rows)/math.fsum((x-mx)**2 for x,y in rows)
    a=my-b*mx
    return dict(fixed_miss=miss,intercept_ms=a,slope_ms_per_K=b,source_total_K_range=[32,200],
        source_points=[dict(total_K=x,C_ms=y) for x,y in rows],
        max_abs_training_residual_ms=max(abs(y-a-b*x) for x,y in rows))


def main():
    protected=[HERE/n for n in ('long_scale_math.json','long_scale_math.py','constructed_candidates.json',
        'math_followup_constructed.py','low_b_short_math.json','low_b_short_math.py','prepare_context_scale.py','experiment.py')]
    before={str(p.relative_to(ROOT)):sha(p) for p in protected}
    raw=ast.literal_eval((ROOT/'data').read_text());data_sha=sha(ROOT/'data')
    fits={m:fit(raw,m) for m in (128,1024)}
    old_long=json.loads((HERE/'long_scale_math.json').read_text())['fits_by_fixed_miss']['1024']
    old_short=json.loads((HERE/'constructed_candidates.json').read_text())['fit']
    for miss,old in [(128,old_short),(1024,old_long)]:
        assert fits[miss]['intercept_ms']==old['intercept_ms'] and fits[miss]['slope_ms_per_K']==old['slope_ms_per_K']
    def profile(role,coarse,K,miss,fraction):
        f=fits[miss];c=(f['intercept_ms']+f['slope_ms_per_K']*K)*1000
        hit=K*1024-miss;assert hit%128==0
        blocks=hit//128;v=blocks*BLOCK_GIB
        method='affine_extrapolation_above_raw_maximum' if K>200 else 'affine_extrapolation_below_raw_minimum'
        assert K<32 or K>200
        source='constructed_candidates.json' if miss==128 else 'long_scale_math.json'
        return dict(role=role,coarse_role=coarse,total_k=K,nql=miss,compute_us=c,V_GiB=v,
            required_B_GiB_s=v*1e6/c,blocks_per_layer=blocks,fraction_of_coarse_role=fraction,
            category='LL' if coarse=='L' else 'SS',is_raw=False,source_ttft_ms=None,
            model_provenance=dict(source_data_sha256=data_sha,data_sha256=data_sha,
                compute_model=dict(method=method,fixed_miss=miss,intercept_ms=f['intercept_ms'],
                    slope_ms_per_K=f['slope_ms_per_K'],raw_support_total_K_range=[32,200],
                    frozen_source=source,frozen_source_sha256=sha(HERE/source)),
                V_model='Exact128-token cached blocks at176KiB perblock; no tail or padding',
                context_support='Both short10/12/14K andlong352/384/416K C are explicitly extrapolated; not measured hardware or validated maximum context.'))
    specifications=[
        ('hetero_S12_control_L384',[1,18],1,[profile('L','L',384,1024,[1,1]),profile('S','S',12,128,[1,1])]),
        ('hetero_S10_12_14_L384',[1,18],1,[profile('L','L',384,1024,[1,1])]+[profile(f'S{i+1}','S',K,128,[1,3]) for i,K in enumerate([10,12,14])]),
        ('hetero_L352_384_416_S10',[1,24],3,[profile(f'L{i+1}','L',K,1024,[1,3]) for i,K in enumerate([352,384,416])]+[profile('S','S',10,128,[1,1])]),
    ]
    selected=[];queue_details={}
    for name,coarse_counts,multiple,profiles in specifications:
        by_coarse={r:[p for p in profiles if p['coarse_role']==r] for r in ('L','S')}
        C={r:math.fsum(p['compute_us']*p['fraction_of_coarse_role'][0]/p['fraction_of_coarse_role'][1] for p in pp)/1000 for r,pp in by_coarse.items()}
        V={r:math.fsum(p['V_GiB']*p['fraction_of_coarse_role'][0]/p['fraction_of_coarse_role'][1] for p in pp) for r,pp in by_coarse.items()}
        cycle=8*(coarse_counts[0]*C['L']+coarse_counts[1]*C['S'])
        base_reps=math.ceil(22000/cycle);reps=math.ceil(base_reps/multiple)*multiple
        L,S=coarse_counts[0]*reps,coarse_counts[1]*reps;N=L+S
        for p in profiles:
            p['count_per_npu']=(L if p['coarse_role']=='L' else S)*p['fraction_of_coarse_role'][0]//p['fraction_of_coarse_role'][1]
            assert p['count_per_npu']*p['fraction_of_coarse_role'][1]==(L if p['coarse_role']=='L' else S)*p['fraction_of_coarse_role'][0]
        total_us=math.fsum(p['count_per_npu']*p['compute_us'] for p in profiles)
        total_v=math.fsum(p['count_per_npu']*p['V_GiB'] for p in profiles)
        blocks=sum(p['count_per_npu']*p['blocks_per_layer'] for p in profiles)
        pure_ms=8*total_us/1000;rho=32*total_v*1e6/(320*total_us)
        idealq=(V['L']-.01*C['L'])/(.01*C['S']-V['S'])
        profile_order=[]
        for original in range(N):
            coarse='L' if original<L else 'S';items=by_coarse[coarse]
            subtype=(original if coarse=='L' else original-L)%len(items)
            profile_order.append(profiles.index(items[subtype]))
        assert Counter(profile_order)=={i:p['count_per_npu'] for i,p in enumerate(profiles)}
        coverage=[];coarse_hashes={}
        for seed in SEEDS:
            missing=[];streams=[]
            for npu in range(32):
                ids=list(range(N));random.Random(seed+100003*npu).shuffle(ids)
                streams.append([[npu*1000000+position,npu*1000000+original,'L' if original<L else 'S'] for position,original in enumerate(ids)])
                t=0.;seen=set()
                for original in ids:
                    p=profiles[profile_order[original]];end=t+8*p['compute_us']/1000
                    if min(end,4000)>max(t,2000):seen.add(p['coarse_role'])
                    t=end
                    if t>=4000:break
                if seen!={'L','S'}:missing.append(npu)
            coarse_hashes[str(seed)]=hashlib.sha256(json.dumps(streams,separators=(',',':')).encode()).hexdigest()
            queue_details[name,seed]=streams
            coverage.append(dict(seed=seed,mixed_cards=32-len(missing),missing_cards=missing))
        selected.append(dict(name=name,num_npu=32,num_ssu=8,n_layers=8,family='data_affine_heterogeneity_control',
            coarse_counts=coarse_counts,coarse_roles=['L','S'],short_roles=[p['role'] for p in profiles if p['coarse_role']=='S'],
            long_roles=[p['role'] for p in profiles if p['coarse_role']=='L'],profiles=profiles,
            quota_repeat_multiple=multiple,ideal_continuous_nS_per_nL=idealq,rho_ideal=rho,
            minimum_22s_population=dict(L=L,S=S,repeats=reps,requests_per_card=N,total_requests=32*N,
                pure_compute_ms_per_card=pure_ms,per_layer_C_work_us_per_card=total_us,
                per_layer_V_work_GiB_per_card=total_v,expected_blocks_all_32=32*8*blocks),
            class_means=dict(C_ms=C,V_GiB=V,ratio_of_sums_B_GiB_s={r:1000*V[r]/C[r] for r in C},
                count_mean_B_GiB_s={r:statistics.mean(p['required_B_GiB_s'] for p in pp) for r,pp in by_coarse.items()}),
            ideal_short_compute_fraction=S*C['S']/(L*C['L']+S*C['S']),
            canonical_rule='Originalidentity indices0..N_L-1 areL; N_L..N_L+N_S-1 areS. Shuffle thefullidentity population once withRandom(seed+100003*npu).',
            subtype_rule='Use theoriginal identity, never queued position: L subtype=original_index%number_of_L_profiles; S subtype=(original_index-N_L)%number_of_S_profiles; profiles followlisted order. No extra random draws.',
            request_ID_rule='request_id=npu*1000000+queue_position; original_request_id=npu*1000000+original_index; arrivals all0.',
            coarse_input_sha256_by_seed=coarse_hashes,pure_compute_clock_warm_coverage=coverage,
            no_simulation_run=True))
    control,short_het,long_het=selected
    for seed in SEEDS:assert queue_details[control['name'],seed]==queue_details[short_het['name'],seed]
    for k in ('L','S','requests_per_card','total_requests','per_layer_C_work_us_per_card','per_layer_V_work_GiB_per_card','expected_blocks_all_32'):
        assert control['minimum_22s_population'][k]==short_het['minimum_22s_population'][k],k
    original=HERE/'inputs/context8_L384m1024_S10m128_ssu8_h22000_seed7.json.gz'
    with gzip.open(original,'rt') as f:old=json.load(f)
    old_streams=[[] for _ in range(32)]
    for r in old['requests']:
        load=r['load'];old_streams[r['npu_id']].append([r['request_id'],load['original_request_id'],load['role']])
    for stream in old_streams:stream.sort()
    assert old_streams==queue_details[long_het['name'],7]
    old_C=math.fsum(33*p['per_layer_compute_us'] if p['role']=='L' else 792*p['per_layer_compute_us'] for p in old['metadata']['profiles'])
    old_V=math.fsum(33*p['per_layer_kv_gib'] if p['role']=='L' else 792*p['per_layer_kv_gib'] for p in old['metadata']['profiles'])
    assert old_C==long_het['minimum_22s_population']['per_layer_C_work_us_per_card']
    assert old_V==long_het['minimum_22s_population']['per_layer_V_work_GiB_per_card']
    for c in selected:
        c['matched_control_name']='hetero_S12_control_L384' if c!=long_het else 'context8_L384m1024_S10m128'
        c['comparison_scope']='S12homogeneous vsS10/12/14heterogeneous, samecoarseIDs/order/totalC/V' if c!=long_het else 'Originalcontext384S10 vsL352/384/416heterogeneous, samecoarseIDs/order/totalC/V'
    result=dict(schema_version='request-heterogeneity-robustness-v1',status='Mathematical plan; no new manifest or simulation generated',
        family='data_affine_heterogeneity_control',selected_candidates=selected,fits={str(k):v for k,v in fits.items()},
        source_data_sha256=data_sha,original_context384_manifest=str(original.relative_to(ROOT)),
        original_context384_manifest_sha256=sha(original),original_context384_input_fingerprint=old['input_fingerprint'],
        paired_math_checks=dict(short_pair_full_population_total_C_us_equal=True,short_pair_full_population_total_V_GiB_equal=True,
            short_pair_all_5_seeds_coarse_ID_order_equal=True,long_pair_seed7_original_manifest_coarse_ID_order_equal=True,
            long_pair_full_population_total_C_us_equal=True,long_pair_full_population_total_V_GiB_equal=True,
            rounding_note='C equality above is checked on actualfloat compute_us weightedby integercounts; exactaffine identity establishes mathematicalequality. Pointwiseclass averageC may differ by~1e-16ms from summation rounding.'),
        mathematical_reason='AffineC andlinearV give [F(K-d)+F(K)+F(K+d)]/3=F(K). Counts andcoarseL/Sorderarefixed; within-classshapevariationis thecontrolledfactor.',
        protocol=dict(num_npu=32,num_ssu=8,per_ssu_GiB_s=40,npu_link_GiB_s=50,n_layers=8,
            seed7_pilot=True,predeclared_seeds=SEEDS,minimum_pure_compute_ms=22000,warm_ms=[2000,4000],long_ms=[2000,20000],
            random='Complete population independent shuffle; subtypeassigned byoriginalidentity aftercoarsepopulation construction. No changed coarseorder, repeateddeck, synchronization orseed rejection.'),
        limitations=[
            'This is a robustness check, not a further search for thelowestU. Report allthree preselected cases andtheir proper controls.',
            'S12 control changesmeanC,V,ratio andpopulationrelative tooldS10context384; do not call that aone-variablecomparison. Only S12vs10/12/14 andoldcontext384vs352/384/416 arematched.',
            'MeanC andmeanV arematched, andratioofsumV/sumC andidealrho arematched. Arithmeticmean(V/C) is nonlinear andnotgenerallymatched.',
            'Timing/admissionphase andwithin-windowmix may diverge, evenwith identicalqueueIDs/coarseorderandwholepopulationtotals. This is expected andmust be reported, not hidden by window orseed selection.',
            'Only coarseL/S needbothcompute on everyNPUinwarm. Requiringall three subtypes ineverywindow is astronger differentconstraint.',
            'Allprofilesin thesethree newcases are explicitlyextrapolated fromthe raw32..200K range. Totalshortinputis at least10K; originaldataevidence is notclaimed.',
            'QoScategorystaysLL forlongvariants andSS forshortvariants. Nointentional policy/pathclasschange ispartof thecomparison.',
            'Bothshortcaseshave idealrho1.002029, slightlyoverone; original/longheterogeneousrho.996878. No caseclaimsper-disk per-instantunderload.',
            'ExpectedphysicalblockcountandtotalKVworkarematched; cache-hit/missconstruction andcompute fitremainmodelassumptions.',
            'Futurehorizonchangesmustkeepfractional subtypequotasexact: longheterogeneous repeatsroundup toa multipleof3. Anewfullshuffle populationisnot acontinuationofoldinput.'],
        preserved_files_sha256=before,generator_sha256=sha(Path(__file__)))
    assert before=={str(p.relative_to(ROOT)):sha(p) for p in protected}
    (HERE/'heterogeneity_math.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# 请求画像多样性：总工作量与长短顺序保持不变','',
        '这是稳健性检查，不继续寻找更低U。只改变同一类内部的画像离散程度，保持每张卡总C、总V、请求数量及粗L/S队列和requestID一致。32NPU、8SSU×40GiB/s、单卡50GiB/s、8层，完整population独立Random；本文件没有生成输入或运行仿真。','',
        '| 新候选 | 画像及每卡数量 | 对应控制 | 每卡纯算秒 | rho |',
        '|---|---|---|---:|---:|']
    for c in selected:
        desc='；'.join(f"{p['role']}:{p['total_k']}K/miss{p['nql']}×{p['count_per_npu']}" for p in c['profiles'])
        lines.append(f"| {c['name']} | {desc} | {c['matched_control_name']} | {c['minimum_22s_population']['pure_compute_ms_per_card']/1000:.9f} | {c['rho_ideal']:.9f} |")
    lines+=['','前两组L:S=1:18，repeat35，每卡35L+630S。异质短有10/12/14K各210个；它们与同质S12成对比较。不能把这个新S12控制直接与旧S10实验当成只改多样性的对照。','',
        '第三组保持旧context384的1:24、33L+792S和同一粗队列，只把33L按原始身份分为352/384/416K各11个。短仍为10K/miss128；旧context384就是对应同质控制。','',
        '## 为什么总C和V能保持一致','',
        '```text','C(K)=a+b*K；V(K)=d*(1024*K-miss)',
        '[C(K-dK)+C(K)+C(K+dK)]/3 = C(K)',
        '[V(K-dK)+V(K)+V(K+dK)]/3 = V(K)','```','',
        '短画像中心12K、左右各差2K；长中心384K、左右各差32K。miss在各对内不变。实际用compute_us乘整数配额求和，前两新输入总C严格相等；第三与旧输入总C也严格相等。V来自完整176KiB块，块数和字节和同样严格相等。','',
        '| 画像 | 单层C ms | 单层V GiB | B=V/C GiB/s | 单层块数 |',
        '|---|---:|---:|---:|---:|']
    dedup={}
    for c in selected:
        for p in c['profiles']:dedup[p['total_k'],p['nql']]=p
    for key,p in sorted(dedup.items()):lines.append(f"| {key[0]}K/miss{key[1]} | {p['compute_us']/1000:.9f} | {p['V_GiB']:.12f} | {p['required_B_GiB_s']:.9f} | {p['blocks_per_layer']} |")
    lines+=['', '注意：总C与总V相同，保证时间加权理想B=ΣV/ΣC与rho相同；各请求B的算术平均并不必相同，因为V/C是比值。不要把“均C、均V相同”误写成“平均Bi也严格相同”。','',
        '## 怎样严格保留Random粗顺序','',
        '先创建原始identity：前NL个为长，后NS个为短。每张卡用`Random(seed+100003*npu)`打乱整个人口的identity一次，再按identity指定子画像。','',
        '- 请求ID始终为`npu*1000000+队列位置`；原始ID为`npu*1000000+原始identity`。',
        '- 短异质：`(原始identity-35)%3`依次对应10/12/14K。控制组这些identity都用12K。',
        '- 长异质：`原始identity%3`依次对应352/384/416K；原始identity>=33的短请求保持不变。',
        '- 不额外抽随机数，不按最终队列位置轮转子画像，因此不会把子画像变成运行中的固定周期。每卡完整粗L/S顺序及原始identity与对应控制逐项一致。','',
        'JSON已验证前两组五个固定seed的全部32卡粗顺序相同；第三组seed7与旧冻结manifest的26400条requestID/原始ID/粗角色逐项一致。未来增加seed时使用同一规则，不挑seed。','',
        '## 必须继续检查的变量','',
        '固定的是整个人口的工作量和输入顺序。不同C/V会改变完成相位，实际warm里完成了多少长短请求可以变化；不能再选一个刚好更差的窗口。继续用warm[2,4)s和长窗[2,20)s，并报告所有32卡是否都有粗长/短计算。无需强制每卡每个两秒窗都出现三种子画像，那是更强的新约束。','',
        '分别报告实际U、短内部层包含零的平均等待、长内部等待、首层交接、idle与TTFT代理SLO。短子型可单列分布；聚合要按实际周期数量/卡时间加权。所有短子型仍是SS，所有长子型仍是LL，没有通过改变QoS类别来制造差异。','',
        '所有新画像C都在原始32K—200K之外，是透明外推，不能叫原始data实测。短总输入最小10K；读取块模型保持不变。rho≈1不是逐盘逐时欠载，前两组rho约1.002还略高于1。','',
        '[完整计划、来源和控制核验](heterogeneity_math.json) · [数学复算脚本](heterogeneity_math.py)','']
    (HERE/'heterogeneity_math.md').write_text('\n'.join(lines))
    print(json.dumps(dict(selected=3,short_pair_total_C_V_and_all_five_coarse_queues_equal=True,
        long_pair_total_C_V_and_seed7_frozen_queue_equal=True,simulations=0,protected_unchanged=True)))
    for c in selected:print(c['name'],c['minimum_22s_population'],c['rho_ideal'])


if __name__=='__main__':main()
