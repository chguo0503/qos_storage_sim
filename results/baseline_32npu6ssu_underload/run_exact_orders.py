#!/usr/bin/env python3
"""Use exactly matched per-packet profile counts to test synchronized bursts."""
from pathlib import Path
import shutil
import run_orders
from exact_order_design import exact_order_indices, describe_exact_order

HERE = Path(__file__).resolve().parent
ORIGINAL_SAVE = run_orders.save_manifest
SOURCE_NAMES = ("run_orders.py", "order_design.py", "run_exact_orders.py", "exact_order_design.py")


def save_exact(path, requests, metadata):
    metadata.update(
        order_source_sha256={name: run_orders.sha(HERE / name) for name in SOURCE_NAMES},
        order_constructor_override="run_exact_orders installs exact-packet input permutation functions process-locally; simulator is unchanged",
        shuffle_rule="Original identities are shuffled within each profile, then every card follows identical exact-quota profile packets with group-specific pure-compute phase.",
        sampling_caveat="Deliberately synchronized profile order within each NPU group; requests remain mixed short/long and preserve the random reference population. This is a constructed worst-order probe, not a production arrival model.",
    )
    return ORIGINAL_SAVE(path, requests, metadata)


if __name__ == "__main__":
    for name in ("run_exact_orders.py", "exact_order_design.py"):
        target = HERE / "sources" / "exact_order_intervention" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert run_orders.sha(target) == run_orders.sha(HERE / name)
        else:
            shutil.copyfile(HERE / name, target)
    run_orders.order_indices = exact_order_indices
    run_orders.describe_order = describe_exact_order
    run_orders.save_manifest = save_exact
    run_orders.MODES = ("exact_cohort2", "exact_cohort4", "exact_burst")
    run_orders.main()
