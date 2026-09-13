#!/usr/bin/env python3
"""Plot original seed7 statistics with a verified same-input replay IO trace.

No simulation, new seed, frozen-source edit, or original-log replacement.
The historical renderers keep all their physical byte and cycle checks.
"""
from pathlib import Path
import gc
import json
import math
import sys

STUDY=Path(__file__).resolve().parents[4]
CASE=Path(__file__).resolve().parents[1]
REPLAY_ROOT=STUDY/'replays/context384_seed7'
sys.path.insert(0,str(STUDY))
import analyze as audit
import render
import zoom


def verified_replay():
    checks=REPLAY_ROOT/'replay_checks.json'
    record=audit.read(checks)
    assert record['passed'] and all(record['checks'].values())
    replay=Path(record['replay_case']).resolve()
    assert replay.is_relative_to(REPLAY_ROOT)
    provenance_path=REPLAY_ROOT/'replay_provenance.json'
    provenance=audit.read(provenance_path)
    assert Path(provenance['source_case']).resolve()==CASE
    original_command=audit.read(CASE/'command.json')
    command=audit.read(replay/'command.json')
    assert original_command['status']==command['status']=='complete'
    assert original_command['strategy']==command['strategy']=='baseline'
    assert original_command['core_source_sha256']==command['core_source_sha256']
    assert original_command['runner_sha256']==command['runner_sha256']
    assert provenance['source_manifest_sha256']==original_command['manifest_sha256']
    assert provenance['source_result_sha256']==original_command['output_sha256']
    sources={}
    for case,cmd in ((CASE,original_command),(replay,command)):
        for filename,key in (('manifest.json.gz','manifest_sha256'),('result.json.gz','output_sha256')):
            path=case/filename
            assert audit.sha(path)==cmd[key]
            sources[str(path)]=audit.sha(path)
        sources[str(case/'command.json')]=audit.sha(case/'command.json')
    assert audit.sha(replay/'manifest.json.gz')==audit.sha(CASE/'manifest.json.gz')
    assert audit.sha(replay/'trace.json.gz')==command['trace_sha256']
    # Recheck the equality contract now; a historical passed flag alone is not
    # enough if either underlying result was subsequently replaced.
    old=audit.read(CASE/'result.json.gz');new=audit.read(replay/'result.json.gz')
    for key in ('input_fingerprint','summary','windows','metadata','core_and_policy_sha256'):
        assert old[key]==new[key],key
    assert audit.read(CASE/'physical_service.json')==audit.read(replay/'physical_service.json')
    for p in (checks,provenance_path,CASE/'physical_service.json',replay/'physical_service.json',replay/'trace.json.gz'):
        sources[str(p)]=audit.sha(p)
    # The original zoom verifies a real analysis file alongside its input.
    # Preserve that contract using an explicit byte-for-byte reused analysis,
    # with a separate provenance record. This is not a new statistical sample.
    original_analysis=CASE/'analysis.json';reused=replay/'analysis.json'
    if reused.exists():
        assert audit.sha(reused)==audit.sha(original_analysis)
    else:
        reused.write_bytes(original_analysis.read_bytes())
    reuse_record=dict(original_analysis=str(original_analysis),sha256=audit.sha(original_analysis),
        reused_analysis=str(reused),reason='Original statistics reused after full replay equality verification.',
        no_new_seed=True,no_new_analysis=True,replay_checks_sha256=audit.sha(checks))
    (replay/'analysis_reuse.json').write_text(json.dumps(reuse_record,ensure_ascii=False,indent=2)+'\n')
    sources[str(original_analysis)]=audit.sha(original_analysis)
    sources[str(replay/'analysis_reuse.json')]=audit.sha(replay/'analysis_reuse.json')
    return replay,command,dict(no_new_seed=True,source_case=str(CASE),trace_case=str(replay),
        replay_checks_path=str(checks),replay_checks_sha256=audit.sha(checks),
        full_summary_windows_metadata_and_physical_service_equal=True,sources=sources)


def main():
    replay,command,replay_audit=verified_replay()
    gc.collect()
    out=CASE/'figures';man=audit.read(CASE/'manifest.json.gz');meta=man['metadata']
    assert meta['candidate']=='context8_L384m1024_S10m128'
    assert meta['num_ssu']==8 and meta['num_npu']==32 and meta['seed']==7
    assert meta['count_ratio']==[1,24]
    profiles={p['role']:p for p in meta['profiles']}
    assert all(p['construction']['method'].startswith('affine_extrapolation') for p in profiles.values())
    pure_us=sum(n*profiles[r]['per_layer_compute_us'] for n,r in zip(meta['count_ratio'],meta['role_names']))
    volume=sum(n*profiles[r]['per_layer_kv_gib'] for n,r in zip(meta['count_ratio'],meta['role_names']))
    rho=32*volume*1e6/pure_us/(8*40)
    assert math.isclose(rho,meta['ideal_load_ratio'],abs_tol=1e-12)
    notice=f'理想平均负载{100*rho:.2f}%（非逐盘逐时欠载保证）· 同seed等价重播trace'
    protected={str(p):audit.sha(p) for p in (Path(render.__file__),Path(zoom.__file__),Path(audit.__file__),STUDY/'replay_case.py')}
    builders=dict(protected);builders[str(Path(__file__))]=audit.sha(Path(__file__))
    original_save=render.save;original_trace=render.trace_evidence;display_checks=[]

    def replay_trace(case,manifest,raw,unused_command,left,right):
        assert case==CASE
        return original_trace(replay,manifest,raw,command,left,right)

    def annotated_save(fig,path):
        candidates=[t for t in fig.texts if ' NPU / ' in t.get_text()]
        assert len(candidates)==1
        text=candidates[0];text.set_text(text.get_text()+' · '+notice)
        for _ in range(48):
            fig.canvas.draw()
            if text.get_window_extent(fig.canvas.get_renderer()).x1<=fig.canvas.get_width_height()[0]*.97:
                break
            text.set_fontsize(text.get_fontsize()-.25)
        assert text.get_fontsize()>=10
        result=original_save(fig,path)
        display_checks.append(dict(path=str(path),subtitle=text.get_text(),subtitle_fontsize=text.get_fontsize(),
            all_C_extrapolated_explicit=True,mean_load_not_underload_guarantee=True,replay_trace_source_explicit=True))
        return result

    render.trace_evidence=replay_trace;render.save=annotated_save
    data=render.prepare(CASE,2000.,4000.)
    assert data['analysis']['short_roles']==['S']
    data['evidence']['same_seed_trace_replay']=replay_audit
    images=[render.fleet_bandwidth(data,out),render.timeline(data,out),render.history(data,out)]
    record=dict(all_checks_passed=True,no_new_simulation=True,no_new_seed=True,
        case=str(CASE),trace_source_case=str(replay),strategy='baseline',window_ms=[2000.,4000.],
        metadata_label=meta['label'],U_percent=data['window']['U_percent'],ideal_load_ratio=rho,
        evidence=data['evidence'],builders=builders,images=images)
    (out/'checks.json').write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    gc.collect()
    local=zoom.prepare(replay,(2000.,4000.))
    zoom_out=out/'internal_median_wait';zoom_out.mkdir(exist_ok=True)
    image=zoom.draw(local,zoom_out/'baseline_random_short_internal_cycle.png')
    evidence=local['evidence']
    evidence['builders'].update(builders)
    evidence.update(image=image,original_case=str(CASE),statistics_source_case=str(CASE),
        no_new_seed=True,ideal_load_ratio=rho,load_label=notice,same_seed_trace_replay=replay_audit)
    (zoom_out/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (zoom_out/'selected_cycle_blocks.json').write_text(json.dumps(dict(columns=render.COLS,
        target_rows=local['target'].tolist(),physical_ssd_rows=local['shown'].tolist()),ensure_ascii=False,indent=2)+'\n')
    e=evidence
    notes=['# 384K案例：同请求内部中位等待','',
        '[查看PNG](baseline_random_short_internal_cycle.png)','',
        '原实验seed7没有保留逐块日志，因此使用同输入、同seed的重播补充trace。完整summary（含所有层）、窗口、metadata、实际服务统计与原运行完全一致，并逐项重新核验SHA。重播没有作为新seed或新独立样本。','',
        '32 NPU、8 SSU×40 GiB/s；L384K/miss1024、S10K/miss128，两类C均外推。理想平均负载99.69%不保证逐盘逐时欠载。','',
        f'选择NPU{e["selected"]["npu"]}、请求{e["selected"]["request_id"]}。从{e["selection"]["eligible_cycles"]}个warm内完整、避开首尾且有等待的短层中选择中位附近；没有选择最坏层。',
        f'C={e["C_ms"]:.9f} ms；w={e["wait_ms"]:.9f} ms；D={e["D_ms"]:.9f} ms；V={e["V_MiB"]:.6f} MiB。',
        f'B={e["B_GiB_s"]:.9f} GiB/s；平均b={e["average_b_GiB_s"]:.9f} GiB/s；b/B=C/D={e["local_cycle_U_percent"]:.6f}%。这是完整周期记账关系，不是瞬时利用率或排队预测。','',
        '本卡NPU行与累计接收始终属于同一请求。SSU行包含其他卡的竞争IO；所有显示块与原manifest placement一致。',
        f'局部名义逐盘需求超限{e["local_nominal"]["any_ssu_over_capacity_ms"]:.9f} ms；不能只凭此图宣称严格欠载下FIFO损失。',
        '首层固有链路等待另见[损失分解](../loss_decomposition.md)，不把全部首层损失称作FIFO。','',
        '[选择规则、重播校验及全部来源SHA](evidence.json) · [实际块记录](selected_cycle_blocks.json)']
    (zoom_out/'README.md').write_text('\n'.join(notes)+'\n')
    assert len(display_checks)==4
    assert all(audit.sha(Path(p))==digest for p,digest in protected.items())
    (out/'display_audit.json').write_text(json.dumps(dict(all_checks_passed=True,
        original_renderers_unchanged=True,no_new_seed=True,builders=builders,images=display_checks,
        replay_verification=replay_audit),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(case=str(CASE),warm_U=data['window']['U_percent'],rho=rho,
        verified_cycles=data['evidence']['verified_complete_internal_cycles'],npu=e['selected']['npu'],
        C_ms=e['C_ms'],wait_ms=e['wait_ms'],D_ms=e['D_ms'],cycle_U=e['local_cycle_U_percent'],
        images=len(display_checks),replay_case=str(replay)),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
