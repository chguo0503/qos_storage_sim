"""Alternative causal CIR budgets for the four-NPU experiments.

This is a small experimental extension of the repository's dedicated-Path
Scheme B, not a new SSD scheduler. It consumes only public request manifests.
No FIFO order, future arrivals or private virtual-finish state is inspected.
"""

from npu_stall_predictor import required_rate_gib_s


def allocate_utilization_rates(demands, capacity=40.0, floor_fraction=0.5):
    """Static-fluid utilization objective with a relative-service floor.

    If u_i=min(1,r_i/d_i), marginal gain is 1/d_i until r_i=d_i.
    First preserve floor_fraction of proportional sharing, then finish the
    lowest-demand lanes first. This maximizes sum(u_i) under those floors
    in that STATIC FLUID model only. Transitions and request latency can
    behave differently; a low floor may severely penalize high-demand jobs.
    """
    total = sum(demands)
    if total <= capacity:
        return tuple(demands)
    rates = [floor_fraction * capacity * d / total for d in demands]
    remaining = capacity - sum(rates)
    for i in sorted(range(len(demands)), key=lambda j: (demands[j], j)):
        added = min(demands[i] - rates[i], remaining)
        rates[i] += added
        remaining -= added
    return tuple(rates)


class LayerBudgetController:
    """Manifest-only controller, compatible with CIRControlConfig.

    budget='transition' gives a prefetched successor the currently computing
    last layer's C, not its own C. This is a release-time planning budget,
    not an absolute remaining deadline at later evaluations. Consequently
    it still is NOT a deadline guarantee. rate_mode='utility' is an explicit
    utilization/fairness experiment; 'proportional' preserves relative demand.
    """
    def __init__(self, *, budget="transition", rate_mode="proportional",
                 floor_fraction=0.5, capacity=40.0, paths=(0, 32, 64, 96)):
        self.budget = budget
        self.rate_mode = rate_mode
        self.floor_fraction = floor_fraction
        self.capacity = capacity
        self.paths = paths
        self.decisions = []

    def __call__(self, snapshot):
        from continuous_batch_sim import CIRControlDecision

        active = {r.npu_id: r for r in snapshot.active_requests if not r.prefetch_only}
        successors = {r.npu_id: r for r in snapshot.active_requests if r.prefetch_only}
        demands = [0.0] * snapshot.num_npu
        for npu_id in range(snapshot.num_npu):
            current, successor = active.get(npu_id), successors.get(npu_id)
            if self.budget == "transition" and successor is not None and current is not None:
                # Current's last layer is already in HBM; this lane's actual
                # outstanding read is the successor's Layer0.
                demands[npu_id] = required_rate_gib_s(
                    successor.next_layer_work_gb_by_ssu[0] / (176 * 1024 / 2**30),
                    current.per_layer_compute_ms)
            else:
                for request in (current, successor):
                    if request is not None:
                        demands[npu_id] = max(demands[npu_id], required_rate_gib_s(
                            request.next_layer_work_gb_by_ssu[0] / (176 * 1024 / 2**30),
                            request.per_layer_compute_ms))
        if self.rate_mode == "utility":
            rates = allocate_utilization_rates(demands, self.capacity, self.floor_fraction)
        else:
            scale = min(1.0, self.capacity / sum(demands)) if sum(demands) else 1.0
            rates = tuple(d * scale for d in demands)
        cirs = [0.0] * 256
        for path, rate in zip(self.paths, rates):
            cirs[path] = rate
        self.decisions.append({
            "time_ms": snapshot.time_ms, "demands": demands, "cirs": rates,
            "active_request_ids": [r.request_id for r in snapshot.active_requests],
            "budget": self.budget, "rate_mode": self.rate_mode,
            "floor_fraction": self.floor_fraction,
        })
        return CIRControlDecision((tuple(cirs),))
