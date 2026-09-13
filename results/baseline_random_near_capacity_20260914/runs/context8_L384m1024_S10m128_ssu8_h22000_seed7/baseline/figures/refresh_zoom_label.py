#!/usr/bin/env python3
"""Refresh one presentation label from the already verified local block extract.

No simulation or selection change. Original renderers and case wrapper stay frozen.
"""
from pathlib import Path
import json
import math
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[3]
sys.path.insert(0, str(STUDY))
import analyze as audit
import render
import zoom


def main():
    out = HERE / 'internal_median_wait'
    ep = out / 'evidence.json'
    bp = out / 'selected_cycle_blocks.json'
    e, blocks = audit.read(ep), audit.read(bp)
    before = dict(evidence_sha256=audit.sha(ep), block_extract_sha256=audit.sha(bp), image=e['image'])
    assert e['all_checks_passed'] and e['no_new_seed']
    for filename, digest in e['sources'].items():
        assert audit.sha(Path(filename)) == digest
    for filename, digest in e['builders'].items():
        assert audit.sha(Path(filename)) == digest
    assert blocks['columns'] == render.COLS
    man = audit.read(HERE.parent / 'manifest.json.gz')
    requests = {q['request_id']: q for q in man['requests']}
    target, shown = np.asarray(blocks['target_rows'], float), np.asarray(blocks['physical_ssd_rows'], float)
    rid, npu = e['selected']['request_id'], e['selected']['npu']
    layer = e['selected']['following']['layer']
    q = requests[rid]['load']
    for row in np.vstack([target, shown]):
        request = requests[int(row[0])]
        placements = man['placements'][request['placement_index']]
        disk, size = placements[0 if len(placements) == 1 else int(row[2])][int(row[3])]
        assert row[1] == request['npu_id'] and row[4] == disk and row[6] == size and row[5] == 0
    assert np.all(target[:, 0] == rid) and np.all(target[:, 1] == npu) and np.all(target[:, 2] == layer)
    assert len(target) == e['target_blocks'] and len(shown) == e['physical_rows_shown']
    audit.close(math.fsum(target[:, 6]), e['V_GiB'])
    left, right = e['left_ms'], e['right_ms']
    ssd = zoom.cumulative(target, 9, 10, man['metadata']['disk_bw_gib_s'], left, right)
    link = zoom.cumulative(target, 11, 12, man['metadata']['npu_bw_gib_s'], left, right)
    audit.close(ssd[1][-1], e['V_GiB'])
    audit.close(link[1][-1], e['V_GiB'])
    audit.close(float(np.interp(e['deadline_ms'], *link)) * 1024, e['received_at_compute_deadline_MiB'])
    data = dict(evidence=e, meta=man['metadata'], q=q, shown=shown, target=target,
                requests=requests, short_roles=set(e['selection']['short_roles']), ssd=ssd, link=link)
    protected = {str(Path(m.__file__)): audit.sha(Path(m.__file__)) for m in (audit, render, zoom)}
    original_save = render.save
    labels = {}

    def annotated_save(fig, path):
        subtitle = next(t for t in fig.texts if ' NPU / ' in t.get_text())
        subtitle.set_text(subtitle.get_text() + ' · ' + e['load_label'])
        for _ in range(48):
            fig.canvas.draw()
            if subtitle.get_window_extent(fig.canvas.get_renderer()).x1 <= fig.canvas.get_width_height()[0] * .97:
                break
            subtitle.set_fontsize(subtitle.get_fontsize() - .25)
        assert subtitle.get_fontsize() >= 10
        text = next(t for t in fig.texts if t.get_text().startswith('实际排队证据：'))
        labels['before'] = text.get_text()
        first = e['target_first_ssd_service_ms'] - left
        longs = [d['before_target_long_ssd_service_ms'] for d in e['disks']]
        labels['after'] = (f'实际排队证据：本层开始计算后第 {first:.3f} 毫秒，目标才获得 SSD 服务；'
                           f'此前各盘服务前方长请求约 {min(longs):.3f}～{max(longs):.3f} 毫秒。')
        text.set_text(labels['after'])
        assert 'ms' not in text.get_text()
        return original_save(fig, path)

    render.save = annotated_save
    e['image'] = zoom.draw(data, out / 'baseline_random_short_internal_cycle.png')
    e['builders'][str(Path(__file__).resolve())] = audit.sha(Path(__file__))
    e['presentation_label_refresh'] = dict(source_block_extract=str(bp), source_block_extract_sha256=audit.sha(bp),
                                            values_and_selection_unchanged=True, label=labels)
    ep.write_text(json.dumps(e, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    assert all(audit.sha(Path(p)) == digest for p, digest in protected.items())
    record = dict(all_checks_passed=True, no_simulation=True, no_new_seed=True,
                  values_and_selection_unchanged=True, full_trace_source_hashes_reverified=True,
                  local_placements_and_integrals_reverified=True, frozen_helpers_unchanged=protected,
                  before=before, after=e['image'], labels=labels, builder_sha256=audit.sha(Path(__file__)))
    (out / 'label_refresh_audit.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
