"""Offline probe for bounded deadline-SRPT marginal desired CIR.

The probe reuses the existing ``POLICY_QOS_DYNAMIC_JOINT_CIR`` command data
plane without modifying it.  During one simulation call it temporarily
replaces ``sim.DynamicCIRController`` in the worker process.  Importing this
module, and invoking its CLI without ``--run``, never starts a simulation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Optional, Sequence

from experiment import _summary
import sim
from strategy_profiles import CLIENT_VARIANTS


SCHEMA_VERSION = 1
MARGINAL_MODE = "bounded_deadline_srpt_marginal"
NUM_NPU = 128
N_LAYERS = 16
SSU_LIST = (40, 56, 80)
SUPPORTED_SEEDS = (42, 43)
LS_RATIO = 0.5
PLACEMENT_SEED_OFFSET = 1_000_003
SUBMIT_ORDER_SEED_OFFSET = 2_000_003
ARRIVAL_DELAY_SEED_OFFSET = 3_000_003
DEFAULT_OUTPUT_PATH = Path("results/marginal_utility_probe/results.json")
DEFAULT_WORKERS = min(10, os.cpu_count() or 1)


@dataclass(frozen=True)
class MarginalUtilityConfig:
    """Fixed parameters for one bounded marginal-utility probe."""

    tau_ms: float = 1.0
    rmin_gbps: float = 5.0
    npu_cap_gbps: float = sim.NPU_BW_LIMIT

    def __post_init__(self):
        if self.tau_ms <= 0.0:
            raise ValueError("tau_ms must be positive")
        if not 0.0 <= self.rmin_gbps <= self.npu_cap_gbps:
            raise ValueError("rmin_gbps must lie inside the NPU rate bound")


DEFAULT_MARGINAL_CONFIG = MarginalUtilityConfig()


@dataclass(frozen=True)
class MarginalDynamicCIRConfig:
    """Duck-typed sim configuration carrying the truthful marginal mode."""

    routing_mode: str
    mode: str = MARGINAL_MODE
    paths_per_npu: int = 2

    def __post_init__(self):
        if self.routing_mode not in sim.DYNAMIC_ROUTE_MODES:
            raise ValueError("unknown dynamic Path routing mode")
        if self.mode != MARGINAL_MODE or self.paths_per_npu != 2:
            raise ValueError("marginal probe requires its own mode and two Paths")


@dataclass(frozen=True)
class MarginalDecision:
    """Unit-consistent components of one local desired-CIR decision."""

    slack_ms: float
    link_service_ms: float
    remaining_work_service_ms: float
    horizon_ms: float
    marginal_utility: float
    desired_cir_gbps: float


def marginal_desired_cir(
    *,
    now_ms: float,
    deadline_ms: float,
    link_backlog_gb: float,
    remaining_work_gb: float,
    config: MarginalUtilityConfig = DEFAULT_MARGINAL_CONFIG,
) -> MarginalDecision:
    """Compute ``D=rmin+(cap-rmin)*tau/(tau+H)`` using current state only.

    ``deadline_ms`` and ``tau_ms`` are already milliseconds.  Link backlog and
    remaining SSD work are converted from GB to milliseconds at the NPU cap,
    so ``H=slack+Q/50+W/50`` is dimensionally consistent.
    """

    slack_ms = max(0.0, float(deadline_ms) - float(now_ms))
    link_service_ms = (
        float(link_backlog_gb) / config.npu_cap_gbps * 1000.0
    )
    remaining_work_service_ms = (
        float(remaining_work_gb) / config.npu_cap_gbps * 1000.0
    )
    horizon_ms = slack_ms + link_service_ms + remaining_work_service_ms
    marginal_utility = config.tau_ms / (config.tau_ms + horizon_ms)
    desired_cir_gbps = config.rmin_gbps + (
        config.npu_cap_gbps - config.rmin_gbps
    ) * marginal_utility
    return MarginalDecision(
        slack_ms=slack_ms,
        link_service_ms=link_service_ms,
        remaining_work_service_ms=remaining_work_service_ms,
        horizon_ms=horizon_ms,
        marginal_utility=marginal_utility,
        desired_cir_gbps=desired_cir_gbps,
    )


def split_desired_cir_by_remaining_work(desired_cir_gbps, work_by_disk):
    """Split one NPU-local desired CIR exactly by ``W_disk / W``."""

    total_work_gb = sum(work_by_disk.values())
    return {
        disk_id: float(desired_cir_gbps) * work_gb / total_work_gb
        for disk_id, work_gb in work_by_disk.items()
    }


def marginal_disk_demands(
    *,
    now_ms,
    deadline_ms,
    link_backlog_gb,
    work_by_disk,
    config=DEFAULT_MARGINAL_CONFIG,
):
    """Return one pure decision and its proportional per-SSD desired rates."""

    decision = marginal_desired_cir(
        now_ms=now_ms,
        deadline_ms=deadline_ms,
        link_backlog_gb=link_backlog_gb,
        remaining_work_gb=sum(work_by_disk.values()),
        config=config,
    )
    demands = split_desired_cir_by_remaining_work(
        decision.desired_cir_gbps,
        work_by_disk,
    )
    return decision, demands


_BASE_DYNAMIC_CONTROLLER = sim.DynamicCIRController


class MarginalUtilityCIRController(_BASE_DYNAMIC_CONTROLLER):
    """Drop-in controller that advertises only current local marginal demand."""

    def __init__(self, npus, engine_config, marginal_config):
        super().__init__(npus, engine_config)
        self.marginal_config = marginal_config
        self.evaluation_count = 0
        self.min_desired_cir_gbps = float("inf")
        self.max_desired_cir_gbps = 0.0
        self.min_horizon_ms = float("inf")
        self.max_horizon_ms = 0.0

    def _decision(self, npu_id, state, current_time):
        decision = marginal_desired_cir(
            now_ms=current_time,
            deadline_ms=state.deadline_ms,
            link_backlog_gb=self._link_backlog_gb(npu_id, current_time),
            remaining_work_gb=state.remaining_ssd_gb_total,
            config=self.marginal_config,
        )
        self.evaluation_count += 1
        self.min_desired_cir_gbps = min(
            self.min_desired_cir_gbps,
            decision.desired_cir_gbps,
        )
        self.max_desired_cir_gbps = max(
            self.max_desired_cir_gbps,
            decision.desired_cir_gbps,
        )
        self.min_horizon_ms = min(self.min_horizon_ms, decision.horizon_ms)
        self.max_horizon_ms = max(self.max_horizon_ms, decision.horizon_ms)
        return decision

    def demands_for_layer(self, npu_id, layer, current_time):
        state = self.layers[(npu_id, layer)]
        decision = self._decision(npu_id, state, current_time)
        return split_desired_cir_by_remaining_work(
            decision.desired_cir_gbps,
            state.remaining_ssd_gb_by_disk,
        )

    def demand_for_disk(self, npu_id, layer, disk_id, current_time):
        state = self.layers[(npu_id, layer)]
        decision = self._decision(npu_id, state, current_time)
        return (
            decision.desired_cir_gbps
            * state.remaining_ssd_gb_by_disk[disk_id]
            / state.remaining_ssd_gb_total
        )

    def probe_telemetry(self):
        return {
            "desired_cir_evaluations": self.evaluation_count,
            "min_desired_cir_gbps": self.min_desired_cir_gbps,
            "max_desired_cir_gbps": self.max_desired_cir_gbps,
            "min_horizon_ms": self.min_horizon_ms,
            "max_horizon_ms": self.max_horizon_ms,
        }


@contextmanager
def marginal_controller_override(config=DEFAULT_MARGINAL_CONFIG):
    """Temporarily install the probe controller in the current worker only."""

    original = sim.DynamicCIRController
    created = []

    class BoundMarginalController(MarginalUtilityCIRController):
        def __init__(self, npus, engine_config):
            super().__init__(npus, engine_config, config)
            created.append(self)

    BoundMarginalController.__name__ = "BoundMarginalUtilityCIRController"
    sim.DynamicCIRController = BoundMarginalController
    try:
        yield created
    finally:
        sim.DynamicCIRController = original


@dataclass(frozen=True)
class ProbeStrategy:
    name: str
    routing_mode: str

    def config(self, marginal_config):
        return {
            "name": self.name,
            "policy": sim.POLICY_QOS_DYNAMIC_JOINT_CIR,
            "description": (
                "NPU-local bounded deadline-SRPT marginal desired CIR"
            ),
            "client_variant": "refresh8_batch8",
            "dynamic_mode": MARGINAL_MODE,
            "routing_mode": self.routing_mode,
            "dynamic_cir": marginal_control_metadata(
                marginal_config,
                self.routing_mode,
            ),
        }


def strategies():
    return (
        ProbeStrategy("dynamic_marginal_fixed_path", sim.DYNAMIC_ROUTE_FIXED),
        ProbeStrategy("joint_marginal_path_cir", sim.DYNAMIC_ROUTE_LEAST_WORK),
    )


def marginal_control_metadata(config, routing_mode):
    """Describe the actual probe controller without reusing a built-in label."""

    return {
        "mode": MARGINAL_MODE,
        "formula_version": "bounded_deadline_srpt_marginal_v1",
        "decision_owner": "npu_local_current_state",
        "future_information": "none",
        "formula": (
            "H=max(deadline-now,0)+1000*Q/50+1000*W/50; "
            "u=tau/(tau+H); D=rmin+(50-rmin)*u"
        ),
        "tau_ms": config.tau_ms,
        "rmin_gbps": config.rmin_gbps,
        "npu_cap_gbps": config.npu_cap_gbps,
        "units": {"time": "ms", "work": "GB", "rate": "GB/s"},
        "disk_split": "D * W_disk / W",
        "ssd_capacity_rule": "atomic proportional clamp of desired CIR sum to 40GBps",
        "configuration_latency_ms": 0.0,
        "routing_mode": routing_mode,
        "paths_per_npu_per_ssu": 2,
        "apply_boundary": "before_next_nonpreemptive_ssd_command",
    }


def runtime_for_seed(seed):
    return {
        "num_npu": NUM_NPU,
        "n_layers": N_LAYERS,
        "ssu_list": list(SSU_LIST),
        "ls_ratio": LS_RATIO,
        "seeds": {
            "workload": int(seed),
            "placement": int(seed + PLACEMENT_SEED_OFFSET),
            "submit_order": int(seed + SUBMIT_ORDER_SEED_OFFSET),
            "arrival_delay": int(seed + ARRIVAL_DELAY_SEED_OFFSET),
        },
        "arrival_delay_ms": [0.0, sim.ARRIVAL_DELAY_MAX_MS],
        "disk_bw_gbps": sim.DISK_BW,
        "npu_bw_limit_gbps": sim.NPU_BW_LIMIT,
    }


def _code_fingerprint():
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for name in (
        "sim.py",
        "advanced_policies.py",
        "experiment.py",
        "strategy_profiles.py",
        "marginal_utility_probe.py",
    ):
        digest.update(name.encode("utf-8"))
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def _data_fingerprint(table):
    rows = [[list(key), list(table[key])] for key in sorted(table)]
    return hashlib.sha256(
        json.dumps(rows, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def experiment_spec(table, seeds, selected, marginal_config):
    return {
        "schema_version": SCHEMA_VERSION,
        "code_fingerprint": _code_fingerprint(),
        "data_fingerprint": _data_fingerprint(table),
        "seeds": [int(seed) for seed in seeds],
        "runtime": {
            "num_npu": NUM_NPU,
            "n_layers": N_LAYERS,
            "ssu_list": list(SSU_LIST),
            "ls_ratio": LS_RATIO,
            "arrival_delay_ms": [0.0, sim.ARRIVAL_DELAY_MAX_MS],
            "disk_bw_gbps": sim.DISK_BW,
            "npu_bw_limit_gbps": sim.NPU_BW_LIMIT,
        },
        "backend": {
            "ssd": "one nonpreemptive command, size/40GBps",
            "npu": "one FCFS receive command per NPU, size/50GBps",
            "visible_after": "NPU receive completion",
        },
        "marginal_config": asdict(marginal_config),
        "selected_strategies": [
            strategy.config(marginal_config) for strategy in selected
        ],
    }


def prepare_inputs(table, seed, num_ssu):
    run_config = runtime_for_seed(seed)
    seeds = run_config["seeds"]
    return sim.prepare_simulation_inputs(
        table,
        total_requests=NUM_NPU,
        n_layers=N_LAYERS,
        num_disk=num_ssu,
        ls_ratio=LS_RATIO,
        workload_seed=seeds["workload"],
        placement_seed=seeds["placement"],
        arrival_delay_seed=seeds["arrival_delay"],
        arrival_delay_max_ms=run_config["arrival_delay_ms"][1],
    )


def _actual_marginal_telemetry(
    scheduler_telemetry,
    controller,
    marginal_config,
    routing_mode,
):
    telemetry = marginal_control_metadata(marginal_config, routing_mode)
    telemetry.update(
        {
            "epochs": scheduler_telemetry["epochs"],
            "final_total_cir_gbps": scheduler_telemetry[
                "final_total_cir_gbps"
            ],
            "max_total_cir_gbps": scheduler_telemetry[
                "max_total_cir_gbps"
            ],
        }
    )
    telemetry.update(controller.probe_telemetry())
    return telemetry


def run_case(
    table,
    seed,
    num_ssu,
    strategy,
    marginal_config,
    prepared_inputs=None,
):
    """Run one explicit formal case; callers decide when CPU work is allowed."""

    started = time.perf_counter()
    run_config = runtime_for_seed(seed)
    prepared = (
        prepare_inputs(table, seed, num_ssu)
        if prepared_inputs is None
        else prepared_inputs
    )
    engine_config = MarginalDynamicCIRConfig(
        routing_mode=strategy.routing_mode,
    )
    with marginal_controller_override(marginal_config) as controllers:
        _, full = sim.simulate_continuous(
            table,
            policy=sim.POLICY_QOS_DYNAMIC_JOINT_CIR,
            num_npu=NUM_NPU,
            num_disk=num_ssu,
            n_layers=N_LAYERS,
            ls_ratio=LS_RATIO,
            submit_order_seed=run_config["seeds"]["submit_order"],
            prepared_inputs=prepared,
            client_io_config=CLIENT_VARIANTS["refresh8_batch8"],
            dynamic_cir_config=engine_config,
        )
    controller = controllers[0]
    telemetry = _actual_marginal_telemetry(
        full["dynamic_cir_control"],
        controller,
        marginal_config,
        strategy.routing_mode,
    )
    summary = _summary(full)
    request_metrics = summary.pop("request_metrics")
    summary["dynamic_cir_control"] = telemetry
    assert all(summary["invariants"].values())
    assert summary["workload_fingerprint"] == prepared.workload_hash
    assert summary["placement_hash"] == prepared.placement_hash
    assert telemetry["max_total_cir_gbps"] <= sim.DISK_BW
    return {
        "seed": int(seed),
        "num_ssu": int(num_ssu),
        "strategy": strategy.name,
        "config": strategy.config(marginal_config),
        "seeds": run_config["seeds"],
        "workload_fingerprint": prepared.workload_hash,
        "placement_hash": prepared.placement_hash,
        "summary": summary,
        "request_metrics": request_metrics,
        "wall_time_s": time.perf_counter() - started,
    }


_WORKER_TABLE = None
_WORKER_CONFIG = None
_WORKER_STRATEGIES = None
_WORKER_PREPARED_KEY = None
_WORKER_PREPARED_VALUE = None


def _init_worker(table, marginal_config, selected):
    global _WORKER_TABLE, _WORKER_CONFIG, _WORKER_STRATEGIES
    global _WORKER_PREPARED_KEY, _WORKER_PREPARED_VALUE
    _WORKER_TABLE = table
    _WORKER_CONFIG = marginal_config
    _WORKER_STRATEGIES = {strategy.name: strategy for strategy in selected}
    _WORKER_PREPARED_KEY = None
    _WORKER_PREPARED_VALUE = None


def _worker(task):
    global _WORKER_PREPARED_KEY, _WORKER_PREPARED_VALUE
    seed, num_ssu, strategy_name = task
    prepared_key = (seed, num_ssu)
    if prepared_key != _WORKER_PREPARED_KEY:
        _WORKER_PREPARED_VALUE = prepare_inputs(
            _WORKER_TABLE,
            seed,
            num_ssu,
        )
        _WORKER_PREPARED_KEY = prepared_key
    return run_case(
        _WORKER_TABLE,
        seed,
        num_ssu,
        _WORKER_STRATEGIES[strategy_name],
        _WORKER_CONFIG,
        prepared_inputs=_WORKER_PREPARED_VALUE,
    )


def write_json_checkpoint(path, value):
    """Atomically replace one JSON checkpoint."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _selected_strategies(route_selection):
    available = strategies()
    if route_selection == "all":
        return available
    names = {name.strip() for name in route_selection.split(",")}
    selected = tuple(
        strategy
        for strategy in available
        if strategy.name in names or strategy.routing_mode in names
    )
    if not selected:
        raise ValueError("route selection did not match a probe strategy")
    return selected


def execution_plan(
    *,
    seeds=SUPPORTED_SEEDS,
    route_selection="all",
    output_path=DEFAULT_OUTPUT_PATH,
    marginal_config=DEFAULT_MARGINAL_CONFIG,
):
    seeds = tuple(seeds)
    selected = _selected_strategies(route_selection)
    return {
        "execute": False,
        "output_path": str(output_path),
        "seeds": [int(seed) for seed in seeds],
        "num_npu": NUM_NPU,
        "n_layers": N_LAYERS,
        "ssu_list": list(SSU_LIST),
        "strategies": [strategy.name for strategy in selected],
        "case_count": len(seeds) * len(SSU_LIST) * len(selected),
        "marginal_control": marginal_control_metadata(
            marginal_config,
            "selected_per_strategy",
        ),
        "instruction": "pass --run to start the formal simulations",
    }


def run_probe(
    output_path=DEFAULT_OUTPUT_PATH,
    *,
    seeds=SUPPORTED_SEEDS,
    workers=DEFAULT_WORKERS,
    route_selection="all",
    marginal_config=DEFAULT_MARGINAL_CONFIG,
    rerun=False,
):
    """Explicitly execute the process-parallel matrix with JSON checkpoints."""

    output_path = Path(output_path)
    seeds = tuple(int(seed) for seed in seeds)
    selected = _selected_strategies(route_selection)
    table = sim.load_bw_table_cache(num_npu=NUM_NPU)
    spec = experiment_spec(table, seeds, selected, marginal_config)
    rows = {}
    if output_path.exists() and not rerun:
        cached = json.loads(output_path.read_text(encoding="utf-8"))
        if cached["experiment"] != spec:
            raise ValueError("checkpoint experiment does not match this probe")
        rows = {
            (row["seed"], row["num_ssu"], row["strategy"]): row
            for row in cached["results"]
        }

    tasks = [
        (seed, num_ssu, strategy.name)
        for seed in seeds
        for num_ssu in SSU_LIST
        for strategy in selected
        if (seed, num_ssu, strategy.name) not in rows
    ]

    def checkpoint():
        ordered = [
            rows[(seed, num_ssu, strategy.name)]
            for seed in seeds
            for num_ssu in SSU_LIST
            for strategy in selected
            if (seed, num_ssu, strategy.name) in rows
        ]
        write_json_checkpoint(
            output_path,
            {
                "schema_version": SCHEMA_VERSION,
                "complete": len(ordered)
                == len(seeds) * len(SSU_LIST) * len(selected),
                "experiment": spec,
                "selected_strategies": [
                    strategy.name for strategy in selected
                ],
                "results": ordered,
            },
        )

    if tasks:
        worker_count = min(max(1, int(workers)), len(tasks))
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_worker,
            initargs=(table, marginal_config, selected),
        ) as pool:
            futures = {pool.submit(_worker, task): task for task in tasks}
            for future in as_completed(futures):
                row = future.result()
                key = (row["seed"], row["num_ssu"], row["strategy"])
                rows[key] = row
                checkpoint()
                print(
                    "seed=%d SSU=%d %-24s request=%7.3f%% fleet=%7.3f%% wall=%6.1fs"
                    % (
                        row["seed"],
                        row["num_ssu"],
                        row["strategy"],
                        100.0
                        * row["summary"]["avg_request_compute_fraction"],
                        100.0
                        * row["summary"]["fleet_npu_compute_utilization"],
                        row["wall_time_s"],
                    ),
                    flush=True,
                )
    checkpoint()
    return json.loads(output_path.read_text(encoding="utf-8"))


def _parse_ints(value):
    return tuple(int(item.strip()) for item in value.split(","))


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--seeds", type=_parse_ints, default=SUPPORTED_SEEDS)
    parser.add_argument("--routes", default="all")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--tau-ms", type=float, default=1.0)
    parser.add_argument("--rmin-gbps", type=float, default=5.0)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args(argv)
    config = MarginalUtilityConfig(
        tau_ms=args.tau_ms,
        rmin_gbps=args.rmin_gbps,
    )
    if not args.run:
        print(
            json.dumps(
                execution_plan(
                    seeds=args.seeds,
                    route_selection=args.routes,
                    output_path=args.output,
                    marginal_config=config,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    result = run_probe(
        args.output,
        seeds=args.seeds,
        workers=args.workers,
        route_selection=args.routes,
        marginal_config=config,
        rerun=args.rerun,
    )
    print("complete:", result["complete"])
    print("rows:", len(result["results"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
