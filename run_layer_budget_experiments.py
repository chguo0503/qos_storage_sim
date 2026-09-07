"""Compare additional manifest-only CIR budgets on the identical raw-data trace."""

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from layer_budget_controller import LayerBudgetController
from deadline_qos_controller import DeadlineController, deadline_control_events
from npu_request_assignment import assign_requests_on_arrival
from run_stall_policy_experiments import build_variable_requests, run_case, summarize_complete_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--mix", choices=("long", "broad"), default="long")
    parser.add_argument("--burst-size", type=int, default=1)
    parser.add_argument("--duration-ms", type=float, default=2000)
    parser.add_argument("--initial-backlog", type=int, default=6)
    parser.add_argument("--modes", nargs="+", default=["transition", "transition_utility"])
    parser.add_argument("--assignment", choices=("none", "fluid", "compute", "pipeline"), default="none")
    parser.add_argument("--output", type=Path, default=Path("results/stall_prediction_experiments/data"))
    args = parser.parse_args()
    requests, profiles, metadata = build_variable_requests(
        seed=args.seed, mix=args.mix, burst_size=args.burst_size,
        duration_ms=args.duration_ms, initial_backlog_per_npu=args.initial_backlog)
    args.output.mkdir(parents=True, exist_ok=True)
    for mode in args.modes:
        is_deadline = mode in ("edf", "least_slack")
        controller = (DeadlineController(mode) if is_deadline else LayerBudgetController(
            budget="transition" if mode.startswith("transition") else "own",
            rate_mode="utility" if "utility" in mode else "proportional"))
        # Dependency injection into the experiment harness, never the SSD.
        # Run each process independently; patch restores the original factory.
        with patch("run_stall_policy_experiments.ManifestCIRController", return_value=controller), \
             (deadline_control_events(controller) if is_deadline else nullcontext()), \
             (assign_requests_on_arrival(policy=args.assignment) if args.assignment != "none"
              else nullcontext([])) as assignment_decisions:
            result = run_case(profiles, "dedicated_demand", seed=args.seed,
                              requests=requests, complete_all=True)
        result["strategy"] = "dedicated_" + mode
        if args.assignment != "none":
            result["strategy"] += "_assign_" + args.assignment
            result["assignment_decisions"] = assignment_decisions
        result["controller_model"] = ("deadline_qos_controller: observed deadlines via CIR; not exact EDF"
                                      if is_deadline else "layer_budget_controller: conditional fluid budget")
        result["controller_information"] = ("arrived manifests plus NPU observed compute deadlines and unfinished read counts; no SSD FIFO"
                                            if is_deadline else "arrived manifests only; queue counts not used by this decision")
        for filename in ("layer_budget_controller.py", "deadline_qos_controller.py", "npu_request_assignment.py"):
            result["implementation_sha256"][filename] = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
        result["implementation_sha256"]["run_layer_budget_experiments.py"] = hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest()
        result["variable_workload_metadata"] = metadata
        result["common_absolute_window"] = summarize_complete_run(result)
        name = f"variable_budget_{args.mix}_{mode}_seed{args.seed}_burst{args.burst_size}"
        if args.assignment != "none":
            name += "_assign_" + args.assignment
        path = args.output / (name + ".json")
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"name": name, "wall_seconds": result["wall_seconds"],
                          "window": result["common_absolute_window"]}), flush=True)


if __name__ == "__main__":
    main()
