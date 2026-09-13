#!/usr/bin/env python3
"""External completion monitor; never runs simulations or alters the analyzer."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import os
import time
import analyze

HERE = Path(__file__).resolve().parent


def emit(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def short_roles(manifest):
    metadata = manifest['metadata']
    profiles = metadata['profiles']
    roles = {p['role'] for p in profiles}
    if roles == {'H', 'L'}:
        registry = analyze.read(HERE / 'adaptive_candidates.json')
        candidate = next(x for x in registry['selected_candidates']
                         if x['candidate_id'] == metadata['candidate'])
        return [candidate['shorter_compute_label']]
    if roles == {'A', 'B'}:
        return [min(profiles, key=lambda p: p['per_layer_compute_us'])['role']]
    return sorted(r for r in roles if r == 'S' or (r.startswith('S') and r[1:].isdigit()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hours', type=float, default=4.)
    parser.add_argument('--interval', type=float, default=15.)
    args = parser.parse_args()
    assert 0 < args.hours <= 6 and 0 < args.interval <= 60
    source = HERE / 'analyze.py'
    builder = analyze.sha(source)
    started = datetime.now(timezone.utc)
    deadline = time.monotonic() + 3600 * args.hours
    emit(dict(event='monitor_started', pid=os.getpid(), utc=started.isoformat(),
              seconds_limit=3600 * args.hours, analyzer_sha256=builder))
    while time.monotonic() < deadline:
        assert analyze.sha(source) == builder, 'Frozen analyzer changed; restart required.'
        for command_path in sorted((HERE / 'runs').glob('*/*/command.json')):
            command = analyze.read(command_path)
            if command.get('status') != 'complete':
                continue
            case = command_path.parent
            output = case / 'analysis.json'
            if output.exists() and analyze.read(output).get('builder_sha256') == builder:
                continue
            try:
                shorts = short_roles(analyze.read(case / 'manifest.json.gz'))
                data = analyze.analyze_case(case, short_roles=shorts)
                temporary = output.with_name(output.name + '.tmp')
                temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                                allow_nan=False) + '\n')
                temporary.replace(output)
                emit(dict(event='analysis_completed', case=case.parent.name,
                          strategy=data['strategy'], num_ssu=data['num_ssu'], short_roles=shorts,
                          windows=[dict(start_ms=w['start_ms'], end_ms=w['end_ms'],
                                        U_percent=w['U_percent'],
                                        long_short_mixed=w['long_short_mixed_card_count'],
                                        all_32_active=w['all_32_active'],
                                        short_mean_stall_ms=w.get('short_roles_internal_released', {}).get('stall_ms_including_zeros', {}).get('mean'),
                                        physical_disk_utilization_percent=w['physical'].get('per_ssu_utilization_percent'),
                                        nominal_over_capacity_percent=w['nominal']['any_ssu_over_capacity_percent'])
                                   for w in data['windows'][:2]]))
            except Exception as exc:
                emit(dict(event='analysis_failed', case=str(case),
                          error_type=type(exc).__name__, error=str(exc)))
        time.sleep(min(args.interval, max(0, deadline - time.monotonic())))
    emit(dict(event='monitor_finished', utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
