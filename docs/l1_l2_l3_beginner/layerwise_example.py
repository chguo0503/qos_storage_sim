"""Exact arithmetic check for a two-card teaching example, not a simulator run."""
import heapq
from fractions import Fraction


def timeline(priority, overrides=None):
    overrides = overrides or {}
    layers = 8
    compute_seconds = {'A': 2, 'B': 6}
    events, pending, computation, reads = [], [], [], []
    current = {n: -1 for n in compute_seconds}
    active = {n: False for n in compute_seconds}
    ready = {(n, 0) for n in compute_seconds}
    busy = False

    def start_compute(n, layer, t):
        current[n], active[n] = layer, True
        end = t + compute_seconds[n]
        computation.append(dict(npu=n, layer=layer, start=t, end=end))
        heapq.heappush(events, (end, 'compute', n, layer))
        if layer + 1 < layers:
            pending.append((t, n, layer + 1))

    def start_read(t):
        nonlocal busy
        if busy or not pending:
            return
        preferred = overrides.get(t, priority)
        item = min(pending, key=lambda p: (p[1] != preferred, p[0], p[1]))
        pending.remove(item)
        release, n, layer = item
        reads.append(dict(npu=n, layer=layer, release=release, start=t, end=t + 2))
        heapq.heappush(events, (t + 2, 'read', n, layer))
        busy = True

    for n in compute_seconds:
        start_compute(n, 0, 0)
    start_read(0)
    while events:
        t = events[0][0]
        batch = []
        while events and events[0][0] == t:
            batch.append(heapq.heappop(events))
        for _, kind, n, layer in batch:
            if kind == 'read':
                ready.add((n, layer))
                busy = False
            else:
                active[n] = False
        # All completions and newly enabled compute/IO releases precede selection.
        for n in compute_seconds:
            layer = current[n] + 1
            if not active[n] and layer < layers and (n, layer) in ready:
                start_compute(n, layer, t)
        start_read(t)

    assert len(computation) == 2 * layers
    assert len(reads) == 2 * (layers - 1)
    for before, after in zip(reads, reads[1:]):
        assert before['end'] <= after['start']
    lookup = {(x['npu'], x['layer']): x for x in computation}
    for x in reads:
        assert x['release'] == lookup[x['npu'], x['layer'] - 1]['start']
        assert x['start'] >= x['release']
        assert lookup[x['npu'], x['layer']]['start'] >= x['end']

    left, right = 6, 12
    overlap = lambda x: max(0, min(right, x['end']) - max(left, x['start']))
    c = {n: sum(overlap(x) for x in computation if x['npu'] == n) for n in compute_seconds}
    supplied = {n: Fraction(sum(overlap(x) for x in reads if x['npu'] == n), right-left)
                for n in compute_seconds}
    demand = {n: Fraction(2, compute_seconds[n]) for n in compute_seconds}
    estimated = sum(min(Fraction(1), supplied[n] / demand[n]) for n in demand) / 2
    actual = Fraction(sum(c.values()), 2 * (right-left))
    if not overrides:
        assert actual == estimated
        assert c == ({'A': 6, 'B': 0} if priority == 'A' else {'A': 4, 'B': 6})
    else:
        assert priority == 'B'
        assert c == ({'A': 4, 'B': 6} if overrides == {6: 'A'} else {'A': 6, 'B': 6})
    assert all(lookup[n, layers-1]['end'] > right for n in compute_seconds)
    extra_windows=[]
    for lo,hi in [(12,18),(6,18)]:
        totals={n:sum(max(0,min(hi,x['end'])-max(lo,x['start']))
                      for x in computation if x['npu']==n) for n in compute_seconds}
        extra_windows.append(dict(window=[lo,hi],compute_seconds=totals,
                                  actual_u=str(Fraction(sum(totals.values()),2*(hi-lo)))))
    if overrides == {6:'A',8:'A'}:
        assert actual == 1 and estimated == Fraction(5,6)
        assert [x['actual_u'] for x in extra_windows] == ['2/3','5/6']
    return dict(priority=priority, priority_overrides=overrides, compute=computation, reads=reads, window=[left,right],
                window_compute_seconds=c, actual_u=str(actual),
                window_mean_read_rate={n: str(v) for n,v in supplied.items()},
                reference_demand={n: str(v) for n,v in demand.items()},
                ideal_cycle_estimate=str(estimated),
                both_requests_unfinished_through_window=True, extra_windows=extra_windows)


def evaluate():
    return dict(teaching_example=True, real_simulator_run=False,
                initial_state='Both layer-0 inputs ready; both cards start computing at t=0.',
                num_npu=2, layers_per_request=8, read_units_per_layer=2,
                disk_units_per_second=1, compute_seconds={'A':2,'B':6},
                overload=True, total_reference_demand='4/3',
                io_rule='Nonpreemptive whole-layer reads; strict priority at dispatch; '
                        'same-time completions and new releases handled before dispatch.',
                cases=[timeline('A'), timeline('B')],
                local_alternatives=[timeline('B',{6:'A'}),timeline('B',{6:'A',8:'A'})])
