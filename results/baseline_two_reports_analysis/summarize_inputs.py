"""Read the supplied reports' CSVs; derive bandwidth demand and compute statistics.

No simulation, no alteration of supplied evidence. V/C is demand at continuous
compute, not observed SSD bandwidth. Input statistics cover the complete input;
utilization values cover each report's stated one-second warm window.
"""
from pathlib import Path
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'baseline_two_reports'
OUT = Path(__file__).resolve().parent
HASHES = {}


def mark(path):
    HASHES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def rows(path):
    path = mark(path)
    with (gzip.open(path, 'rt', encoding='utf-8-sig') if path.suffix == '.gz' else path.open(encoding='utf-8-sig')) as f:
        return list(csv.DictReader(f))


def read(path):
    return json.loads(mark(path).read_text())


def write_csv(name, data):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def summarize(case, group, data, group_type):
    c = [float(r['layer_compute_ms']) for r in data]
    v = [float(r['layer_kv_gib']) for r in data]
    bandwidth = [1000 * vv / cc for vv, cc in zip(v, c)]
    for r, cc, vv in zip(data, c, v):
        prefix = float(r['seq_len_k']) * 1024 - float(r['nql'])
        assert math.isclose(vv * 2**30, prefix * 1408, abs_tol=1e-5)
        assert cc > 0
    return dict(case=case, group_type=group_type, group=group, input_count=len(data),
                categories='/'.join(sorted({r['category'] for r in data})),
                mean_C_ms_per_layer=math.fsum(c)/len(c), min_C_ms_per_layer=min(c), max_C_ms_per_layer=max(c),
                mean_C_ms_per_8_layer_request=8*math.fsum(c)/len(c),
                mean_V_MiB_per_layer=1024*math.fsum(v)/len(v), min_V_MiB_per_layer=1024*min(v), max_V_MiB_per_layer=1024*max(v),
                mean_V_MiB_per_8_layer_request=8192*math.fsum(v)/len(v),
                demand_GiBps_ratio_of_sums=1000*math.fsum(v)/math.fsum(c),
                min_request_demand_GiBps=min(bandwidth), max_request_demand_GiBps=max(bandwidth),
                total_input_compute_NPU_ms=8*math.fsum(c), total_input_read_GiB=8*math.fsum(v))


def main():
    profiles, categories, capacity, utilization = [], [], [], []
    for case in ('high', 'low4', 'low5', 'low6'):
        data = rows(SRC/'data'/case/'all_inputs_and_times.csv.gz')
        for field, target in (('profile_id', profiles), ('category', categories)):
            groups = defaultdict(list)
            for r in data:
                groups[r[field]].append(r)
            target.extend(summarize(case, key, group, field) for key, group in sorted(groups.items()))
        stats = read(SRC/'data'/case/'statistics.json')
        assert len(data) == stats['input_count']
        assert stats['all_npus_active']
        assert math.isclose(stats['compute_ms'] + stats['stall_ms'], 32000, abs_tol=1e-6)
        for r in rows(SRC/'data'/case/'profile_summary.csv'):
            utilization.append(dict(case=case, group=r['label'], window_start_ms=stats['window_start_ms'], window_end_ms=stats['window_end_ms'],
                                    input_count=int(r['input_count']), warm_active_npus=int(r['warm_active_npus']),
                                    warm_compute_NPU_ms=float(r['compute_ms']), warm_stall_NPU_ms=float(r['stall_ms']),
                                    per_npu_conditional_U=float(r['mean_per_npu_conditional_utilization']),
                                    pooled_conditional_U=float(r['pooled_conditional_utilization']),
                                    occupancy_share=float(r['occupancy_share']), fleet_U=stats['fleet_utilization']))
        if case == 'high':
            continue
        meta = read(SRC/'sources'/case/'input_metadata.json')
        demands = {}
        for role in ('short', 'long'):
            lanes = [r for r in meta['lanes'] if r['role'] == role]
            actual = [r for r in data if r['role'] == role]
            per_lane = defaultdict(list)
            for r in actual:
                per_lane[int(r['npu_id'])].append(r)
            independently = math.fsum(math.fsum(float(r['layer_kv_gib']) for r in g)/math.fsum(float(r['layer_compute_ms']) for r in g)*1000 for g in per_lane.values())
            demands[role] = math.fsum(r['demand_gibps_mean_time_weighted'] for r in lanes)
            assert math.isclose(independently, demands[role], rel_tol=1e-10)
        assert math.isclose(sum(demands.values()), meta['ideal_time_weighted_total_demand_gibps'], rel_tol=1e-10)
        capacity.append(dict(case=case, num_ssu=meta['num_ssu'], n_short=meta['n_short'], n_long=meta['n_long'],
                             short_group_ideal_GiBps=demands['short'], long_group_ideal_GiBps=demands['long'],
                             ideal_total_GiBps=sum(demands.values()), disk_total_capacity_GiBps=40*meta['num_ssu'],
                             hottest_disk_ideal_GiBps=max(meta['ideal_time_weighted_demand_gibps_by_ssu'])))
    warm = rows(SRC/'data/high/warm_requests.csv')
    request = next(r for r in warm if r['request_id'] == '31000009')
    assert float(request['admission_ms']) >= 1000 and float(request['completion_ms']) <= 2000
    assert math.isclose(float(request['warm_compute_ms']), 8*float(request['layer_compute_ms']), rel_tol=1e-10)
    layers = [r for r in rows(SRC/'data/high/warm_layers.csv.gz') if r['request_id'] == '31000009']
    assert len(layers) == 8
    write_csv('profile_bandwidth_compute.csv', profiles)
    write_csv('category_bandwidth_compute.csv', categories)
    write_csv('group_capacity_demand.csv', capacity)
    write_csv('window_profile_utilization.csv', utilization)
    write_csv('long_request_example.csv', [request])
    write_csv('long_request_layers.csv', layers)
    sensitivity = []
    for family in ('1k_nql64', '1k_nql128', '1k_nql256', '1k_nql512', '32k_nql2048'):
        directory = SRC/'sources/fixed_long_sensitivity'/family
        meta, metrics = read(directory/'input_metadata.json'), read(directory/'metrics.json')
        for role in ('short', 'long'):
            lanes = [r for r in meta['lanes'] if r['role'] == role]
            count = sum(r['request_count'] for r in lanes)
            compute = math.fsum(r['ideal_compute_ms'] for r in lanes)
            volume = math.fsum(r['demand_gibps_mean_time_weighted']*r['ideal_compute_ms']/1000 for r in lanes)
            sensitivity.append(dict(short_family=family, role=role, num_ssu=meta['num_ssu'], n_npu=len(lanes), input_count=count,
                                    mean_C_ms_per_layer=compute/(8*count), mean_C_ms_per_request=compute/count,
                                    mean_V_MiB_per_layer=volume*1024/(8*count), demand_GiBps_ratio_of_sums=1000*volume/compute,
                                    group_ideal_GiBps=math.fsum(r['demand_gibps_mean_time_weighted'] for r in lanes),
                                    warm_U=metrics[role+'_utilization'], fleet_U=metrics['mean_utilization']))
    write_csv('fixed_long_sensitivity.csv', sensitivity)
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == digest for p, digest in HASHES.items())
    (OUT/'audit.json').write_text(json.dumps({'all_checks_pass': True, 'profile_rows': len(profiles), 'category_rows': len(categories), 'sensitivity_rows': len(sensitivity),
        'sources_unchanged': HASHES, 'bandwidth_definition': 'V/C at continuous compute; group demand sums per-NPU ratios over full input. Not measured SSD delivered bandwidth.',
        'scope': 'Read-only derivation of supplied baseline_two_reports; no simulation.', 'long_example': request}, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'profile_rows': len(profiles), 'category_rows': len(categories), 'all_checks_pass': True}))


if __name__ == '__main__':
    main()
