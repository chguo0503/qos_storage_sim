"""Read-only capacity and placement preflight for the 128-NPU report curve.

The provenance emitted here authenticates the contents of the tracked ``data``
file and the tables derived from it.  It is intentionally not described as a
complete source-code closure; report runners provide that stronger provenance.
"""

from __future__ import annotations

import argparse
import json
import time

from authenticated_workload_inputs import load_authenticated_bw_table
from policy_logic import GROUP_COUNT, MAX_NPU, PATH_COUNT, PATHS_PER_GROUP
from random_steady_state_workload import (
    IID_UNIFORM_PROFILE_CATALOG_V1,
    build_steady_state_profile_schedule,
    prepare_random_steady_state_workload,
)
from scheme_b_prefill import cold_start_hybrid_path_id


NUM_NPU = 128
N_LAYERS = 16
SCIENTIFIC_PREFIX_REQUESTS_PER_NPU = 32
DEFAULT_SSUS = (8, 12, 16, 20, 24, 32, 40, 48, 72)
PREFLIGHT_SCHEMA_VERSION = 2


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--requests-per-npu", type=int, default=128)
    parser.add_argument("--ssu", type=int, action="append")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run lightweight schema/count tests without loading data",
    )
    return parser.parse_args(argv)


def _selected_statistics(workload, num_ssu):
    stats = workload.statistics
    if len(stats["demand_gbps_by_ssu"]) != num_ssu:
        raise RuntimeError("materialized workload has the wrong SSU vector length")
    return {
        "workload_hash": workload.workload_hash,
        "placement_hash": workload.placement_hash,
        "trace_hash": workload.trace_hash,
        "fleet_demand_gbps": stats["fleet_demand_gbps"],
        "capacity_knee_ssu": stats["capacity_knee_ssu"],
        "global_load_ratio": stats["fleet_demand_gbps"] / (40.0 * num_ssu),
        "capacity_to_demand_ratio": (40.0 * num_ssu) / stats["fleet_demand_gbps"],
        "max_ssu_demand_gbps": stats["max_ssu_demand_gbps"],
        "ssu_over_40_count": stats["ssu_over_40_count"],
        "demand_gbps_by_ssu": stats["demand_gbps_by_ssu"],
        "per_npu_raw_demand_gbps": stats["per_npu_raw_demand_gbps"],
        "per_npu_ms_per_gb": stats["per_npu_ms_per_gb"],
        "fleet_category_counts": stats["fleet_category_counts_all"],
        "profiles_used": stats["profiles_used"],
        "prefix_32_assignment_hash": stats["prefix_32_assignment_hash"],
        "full_assignment_hash": stats["full_assignment_hash"],
    }


def _assignment_counts(backing_requests_per_npu):
    backing_requests_per_npu = int(backing_requests_per_npu)
    if backing_requests_per_npu < SCIENTIFIC_PREFIX_REQUESTS_PER_NPU:
        raise ValueError("requests-per-npu must retain the scientific prefix of 32")
    return {
        "backing_requests_per_npu": backing_requests_per_npu,
        "total_assignment_count": NUM_NPU * backing_requests_per_npu,
        "scientific_prefix_requests_per_npu": (SCIENTIFIC_PREFIX_REQUESTS_PER_NPU),
        "scientific_prefix_total_assignment_count": (
            NUM_NPU * SCIENTIFIC_PREFIX_REQUESTS_PER_NPU
        ),
    }


def _result_row(
    num_ssu,
    backing_requests_per_npu,
    full_backing_statistics,
    scientific_prefix_statistics,
):
    """Build a request-count-neutral v2 result row.

    V1 called the full view ``backing_128`` even when the CLI selected another
    backing length.  That misleading key is intentionally not retained as an
    alias; the top-level schema/version marker distinguishes historical output.
    """

    counts = _assignment_counts(backing_requests_per_npu)
    return {
        "num_ssu": int(num_ssu),
        "full_backing": {
            **full_backing_statistics,
            "requests_per_npu": counts["backing_requests_per_npu"],
            "total_assignment_count": counts["total_assignment_count"],
        },
        "scientific_prefix": {
            **scientific_prefix_statistics,
            "requests_per_npu": counts["scientific_prefix_requests_per_npu"],
            "total_assignment_count": counts[
                "scientific_prefix_total_assignment_count"
            ],
        },
    }


def _data_content_authentication(catalog_provenance):
    """Describe exactly what the loader authenticated, without overclaiming."""

    required = {
        "source",
        "source_sha256",
        "catalog_hash",
        "table_fingerprint",
        "profile_count",
        "cache_path",
        "cache_present",
        "cache_verified_equal",
    }
    if not isinstance(catalog_provenance, dict) or not required <= set(
        catalog_provenance
    ):
        raise RuntimeError("authenticated data provenance is incomplete")
    if catalog_provenance["source"] != "data":
        raise RuntimeError("capacity preflight must authenticate the data file")
    return {
        "scope": "data_file_content_and_derived_table_hashes_only",
        "is_complete_source_closure": False,
        "data_file": catalog_provenance["source"],
        "data_content_sha256": catalog_provenance["source_sha256"],
        "catalog_hash": catalog_provenance["catalog_hash"],
        "table_fingerprint": catalog_provenance["table_fingerprint"],
        "profile_count": catalog_provenance["profile_count"],
        "cache_validation": {
            "cache_path": catalog_provenance["cache_path"],
            "cache_present": catalog_provenance["cache_present"],
            "cache_verified_equal": catalog_provenance["cache_verified_equal"],
        },
    }


def _self_test():
    counts_128 = _assignment_counts(128)
    if counts_128 != {
        "backing_requests_per_npu": 128,
        "total_assignment_count": 16_384,
        "scientific_prefix_requests_per_npu": 32,
        "scientific_prefix_total_assignment_count": 4_096,
    }:
        raise AssertionError("128-request assignment counts changed")
    counts_256 = _assignment_counts(256)
    if counts_256["total_assignment_count"] != 32_768:
        raise AssertionError("256-request assignment count changed")

    row = _result_row(8, 256, {"marker": "full"}, {"marker": "prefix"})
    if set(row) != {"num_ssu", "full_backing", "scientific_prefix"}:
        raise AssertionError("preflight row schema is not backing-neutral")
    if "backing_128" in row or "scientific_prefix_32" in row:
        raise AssertionError("preflight row retained a request-count-specific key")
    if row["full_backing"]["total_assignment_count"] != 32_768:
        raise AssertionError("full-backing row count changed")
    if row["scientific_prefix"]["total_assignment_count"] != 4_096:
        raise AssertionError("scientific-prefix row count changed")

    authentication = _data_content_authentication(
        {
            "source": "data",
            "source_sha256": "a" * 64,
            "catalog_hash": "b" * 64,
            "table_fingerprint": "c" * 64,
            "profile_count": 84,
            "cache_path": "results/example.npz",
            "cache_present": False,
            "cache_verified_equal": False,
        }
    )
    if authentication["is_complete_source_closure"] is not False:
        raise AssertionError("data authentication overclaims source closure")
    if authentication["data_content_sha256"] != "a" * 64:
        raise AssertionError("data content hash was not preserved")


def main(argv=None):
    args = parse_args(argv)
    if args.self_test:
        _self_test()
        print(
            json.dumps(
                {
                    "self_test": "passed",
                    "preflight_schema_version": PREFLIGHT_SCHEMA_VERSION,
                },
                sort_keys=True,
            )
        )
        return 0
    assignment_counts = _assignment_counts(args.requests_per_npu)
    ssus = tuple(sorted(set(args.ssu or DEFAULT_SSUS)))
    if not ssus or any(ssu <= 0 for ssu in ssus):
        raise ValueError("SSU counts must be positive")

    paths = tuple(cold_start_hybrid_path_id(npu) for npu in range(NUM_NPU))
    if MAX_NPU != NUM_NPU or len(set(paths)) != NUM_NPU:
        raise RuntimeError("128-NPU dedicated Path mapping is not one-to-one")
    if min(paths) < 0 or max(paths) >= PATH_COUNT:
        raise RuntimeError("128-NPU dedicated Path mapping exceeds the Path ABI")
    paths_by_group = {
        group: tuple(path for path in paths if path // PATHS_PER_GROUP == group)
        for group in range(GROUP_COUNT)
    }
    if any(
        len(group_paths) != 16
        or {path % PATHS_PER_GROUP for path in group_paths} != set(range(16, 32))
        for group_paths in paths_by_group.values()
    ):
        raise RuntimeError("128-NPU dedicated Paths violate the group-local ABI")
    if 0 in paths:
        raise RuntimeError("warm dedicated Paths must not consume reserved Path 0")

    table, catalog_provenance = load_authenticated_bw_table(NUM_NPU)
    schedule = build_steady_state_profile_schedule(
        table,
        mode=IID_UNIFORM_PROFILE_CATALOG_V1,
        seed=args.seed,
        num_npu=NUM_NPU,
        requests_per_npu=args.requests_per_npu,
    )
    prefix_schedule = build_steady_state_profile_schedule(
        table,
        mode=IID_UNIFORM_PROFILE_CATALOG_V1,
        seed=args.seed,
        num_npu=NUM_NPU,
        requests_per_npu=SCIENTIFIC_PREFIX_REQUESTS_PER_NPU,
    )
    if len(schedule.assignments) != assignment_counts["total_assignment_count"]:
        raise RuntimeError("backing assignment count differs from the declared total")
    if (
        len(prefix_schedule.assignments)
        != assignment_counts["scientific_prefix_total_assignment_count"]
    ):
        raise RuntimeError(
            "scientific-prefix assignment count differs from the declared total"
        )
    if schedule.prefix_32_assignment_hash != prefix_schedule.full_assignment_hash:
        raise RuntimeError("backing schedule does not preserve the first-32 prefix")

    started = time.perf_counter()
    rows = []
    for num_ssu in ssus:
        workload = prepare_random_steady_state_workload(
            table, schedule=schedule, num_ssu=num_ssu, n_layers=N_LAYERS
        )
        prefix = prepare_random_steady_state_workload(
            table, schedule=prefix_schedule, num_ssu=num_ssu, n_layers=N_LAYERS
        )
        rows.append(
            _result_row(
                num_ssu,
                args.requests_per_npu,
                _selected_statistics(workload, num_ssu),
                _selected_statistics(prefix, num_ssu),
            )
        )

    print(
        json.dumps(
            {
                "preflight_schema_version": PREFLIGHT_SCHEMA_VERSION,
                "preflight": "128npu_random_real_data_capacity_v2",
                "num_npu": NUM_NPU,
                "n_layers": N_LAYERS,
                "seed": args.seed,
                **assignment_counts,
                "data_content_authentication": _data_content_authentication(
                    catalog_provenance
                ),
                "schedule_fingerprints": schedule.as_fingerprint_dict(),
                "path_abi": {
                    "path_count": PATH_COUNT,
                    "path_min": min(paths),
                    "path_max": max(paths),
                    "unique_path_count": len(set(paths)),
                    "group_count": GROUP_COUNT,
                    "paths_per_group": PATHS_PER_GROUP,
                    "path_count_by_group": {
                        str(group): len(group_paths)
                        for group, group_paths in paths_by_group.items()
                    },
                    "mapping": list(paths),
                },
                "rows": rows,
                "wall_time_s": time.perf_counter() - started,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
