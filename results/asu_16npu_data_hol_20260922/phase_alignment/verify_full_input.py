#!/usr/bin/env python3
"""Replay the frozen, fully corrected finite input with the calibrated selector."""
import hashlib
import json
import math
import sys
from collections import Counter

from native_like_probe import HERE, simulate

sys.path.insert(0, str(HERE.parent))
from variant_runner import validate_config


def main():
    path = HERE / 'aligned_sequential_60000ms_full_input_native_like_config.json'
    raw = path.read_bytes()
    config = json.loads(raw)
    validate_config(config)
    catalog = config['profiles_catalog']
    seqs = config['per_npu_sequences']
    metrics, trace = simulate(catalog, seqs, right=200000., seed=config['seed'])
    finishes = [trace['ends'][n, len(s)-1] for n, s in enumerate(seqs)]
    first, last = min(finishes), max(finishes)

    def utilization(lo, hi):
        work = math.fsum(max(0., min(t+catalog[seqs[n][r]]['compute_us']/1000., hi)-max(t, lo))
                         for (n, r, layer), t in trace['phases'].items())
        return 100. * work / (len(seqs)*(hi-lo))

    phase_rows = [(*key, value) for key, value in sorted(trace['phases'].items())]
    standard = catalog['A']
    corrected = [p for p in catalog.values() if p['family']=='A' and
                 (p['total_tokens'], p['nql']) != (standard['total_tokens'], standard['nql'])]
    counts = [dict(Counter(catalog[k]['family'] for k in s)) for s in seqs]
    full_bins = [{'left_ms': left, 'right_ms': left+2000,
                  'U': utilization(left, left+2000)}
                 for left in range(2000, int(first)-1999, 2000)]
    metrics.update(
        U_all_busy_2s_to_first_idle=utilization(2000., first),
        U_finite_population_0_to_drain=utilization(0., last),
        U_2s_to_60s=utilization(2000., 60000.),
        U_after_60s_to_first_idle=utilization(60000., first),
        first_npu_finished_ms=first, last_npu_finished_ms=last,
        all_npus_completed=True, complete_2s_windows_while_all_busy=full_bins,
    )
    # Do not expose the 200 s selector horizon's idle-including aggregate as U.
    metrics.pop('U')
    metrics.pop('windows')
    metrics['selector_stop_ms'] = metrics.pop('right_ms')
    report = dict(
        source_config=path.name, source_config_sha256=hashlib.sha256(raw).hexdigest(),
        calibrated_selector_only=True, authoritative_results='native ASU/Once runs',
        metrics=metrics, family_counts_by_npu=counts,
        phase_count=len(phase_rows),
        phase_sha256=hashlib.sha256(json.dumps(phase_rows, separators=(',', ':')).encode()).hexdigest(),
        corrected_positions=len(corrected),
        corrected_unique_profiles=len({(p['total_tokens'], p['nql']) for p in corrected}),
        corrected_ranges={field: [min(p[field] for p in corrected), max(p[field] for p in corrected)]
                          for field in ('total_tokens', 'nql', 'compute_us', 'B_gib_s')},
    )
    out = HERE / 'aligned_sequential_60000ms_full_input_native_like_replay.json'
    out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in metrics.items() if k!='complete_2s_windows_while_all_busy'}, indent=2))
    print('phase_count', report['phase_count'])
    print('output', out)


if __name__ == '__main__':
    main()
