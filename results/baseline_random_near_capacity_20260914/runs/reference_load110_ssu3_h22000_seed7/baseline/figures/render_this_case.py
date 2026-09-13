#!/usr/bin/env python3
"""Render this completed raw-data case with an explicit overload annotation.

Only display text is wrapped. The existing trace checks, accounting, selection,
and renderers are reused without changing their source or prior figures.
"""
from pathlib import Path
import json
import math
import sys

STUDY = Path(__file__).resolve().parents[4]
CASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY))
import analyze as audit
import render
import zoom


def main():
    out = CASE/'figures'
    man = audit.read(CASE/'manifest.json.gz')
    metadata = man['metadata']
    assert metadata['candidate'] == 'reference_load110'
    assert metadata['num_ssu'] == 3 and metadata['num_npu'] == 32
    assert metadata['count_ratio'] == [17, 31] and metadata['seed'] == 7
    profiles = {p['role']: p for p in metadata['profiles']}
    assert all(p['construction']['method'] == 'direct_data_row' for p in profiles.values())
    assert profiles['A']['per_layer_compute_us'] < profiles['B']['per_layer_compute_us']
    assert profiles['A']['per_layer_kv_gib'] > profiles['B']['per_layer_kv_gib']
    pure_us = sum(n*profiles[r]['per_layer_compute_us'] for n,r in zip(metadata['count_ratio'], metadata['role_names']))
    volume = sum(n*profiles[r]['per_layer_kv_gib'] for n,r in zip(metadata['count_ratio'], metadata['role_names']))
    rho = 32*volume*1e6/pure_us/(3*40)
    assert math.isclose(rho, metadata['ideal_load_ratio'], abs_tol=1e-12)
    notice = f'理想平均负载{100*rho:.2f}%（轻度过载）'
    protected = {str(p): audit.sha(p) for p in (Path(render.__file__), Path(zoom.__file__), Path(audit.__file__))}
    builders = dict(protected)
    builders[str(Path(__file__))] = audit.sha(Path(__file__))
    original_save = render.save
    display_checks = []

    def annotated_save(fig, path):
        configuration = [t for t in fig.texts if ' NPU / ' in t.get_text()]
        assert len(configuration) == 1
        text = configuration[0]
        assert '欠载' not in text.get_text()
        text.set_text(text.get_text()+' · '+notice)
        # The long case name and notice must remain inside the canvas.
        for _ in range(40):
            fig.canvas.draw()
            right = text.get_window_extent(fig.canvas.get_renderer()).x1
            if right <= fig.canvas.get_width_height()[0]*.965:
                break
            text.set_fontsize(text.get_fontsize()-.25)
        assert text.get_fontsize() >= 10
        check = dict(path=str(path),subtitle=text.get_text(),subtitle_fontsize=text.get_fontsize(),
                     rho=rho,overload_explicit=True)
        result = original_save(fig,path)
        display_checks.append(check)
        return result

    render.save = annotated_save
    data = render.prepare(CASE,2000.,4000.)
    assert data['analysis']['short_roles'] == ['A']
    images = [render.fleet_bandwidth(data,out), render.timeline(data,out), render.history(data,out)]
    record = dict(all_checks_passed=True,no_new_simulation=True,formats_created=['png'],
                  case=str(CASE),strategy='baseline',window_ms=[2000.,4000.],
                  metadata_label=metadata['label'],U_percent=data['window']['U_percent'],
                  evidence=data['evidence'],builders=builders,images=images,
                  ideal_load_ratio=rho,load_label=notice,
                  roles={'A':'短计算、大读取，原始 data (128,256)',
                         'B':'长计算、小读取，原始 data (32,4096)'})
    (out/'checks.json').write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    # Reuse the existing median selection, all physical assertions, and notes.
    sys.argv = [str(Path(zoom.__file__)),'--case',str(CASE)]
    zoom.main()
    evidence_path = out/'internal_median_wait/evidence.json'
    evidence = audit.read(evidence_path)
    evidence['builders'].update(builders)
    evidence.update(ideal_load_ratio=rho,load_label=notice,roles=record['roles'])
    evidence_path.write_text(json.dumps(evidence,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    notes = out/'internal_median_wait/README.md'
    notes.write_text(notes.read_text()+f'\n本批输入的 rho={rho:.9f}，{notice}；本例不能作为严格欠载或纯 FIFO 额外损失的独立证明。A 是短计算、大读取；B 是长计算、小读取，两者均直接来自原始 data。\n')
    assert all(audit.sha(Path(p)) == digest for p,digest in protected.items())
    assert len(display_checks) == 4
    (out/'display_audit.json').write_text(json.dumps(dict(all_checks_passed=True,
        original_renderers_unchanged=True,builders=builders,images=display_checks),
        ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(case=str(CASE),U_percent=data['window']['U_percent'],
        rho=rho,verified_cycles=data['evidence']['verified_complete_internal_cycles'],
        local_C_ms=evidence['C_ms'],local_wait_ms=evidence['wait_ms'],
        local_U_percent=evidence['local_cycle_U_percent'],images=len(display_checks)),ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
