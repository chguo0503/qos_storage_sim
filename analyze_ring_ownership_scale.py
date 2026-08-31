"""Audit consistent-hash ring ownership skew as the SSU count changes.

The placement ring owns an exact interval between every adjacent virtual node.
This script integrates those intervals (it does not Monte-Carlo sample block
keys), normalizes each SSU's ownership by the ideal ``1 / num_ssu`` share, and
writes a small reproducible JSON/Markdown report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics

import sim


ROOT = Path(__file__).resolve().parent
DEFAULT_SSU_LIST = (10, 18, 24, 40, 70)
DEFAULT_VNODE_SENSITIVITY = (256, 1024, 4096)
DEFAULT_SENSITIVITY_SSU_LIST = (40, 70)
OUTPUT_DIR = ROOT / "results" / "ring_ownership_scale"
OUTPUT = OUTPUT_DIR / "analysis.json"
REPORT = OUTPUT_DIR / "analysis.md"
SCHEMA_VERSION = 1
SOURCE_FILES = ("sim.py", Path(__file__).name)
RING_SIZE = 1 << 256


def _source_fingerprint():
    digest = hashlib.sha256(b"ring-ownership-scale-v1\0")
    for name in SOURCE_FILES:
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def ownership_by_ssu(
    num_ssu: int,
    virtual_nodes_per_ssu: int = sim.BLOCK_RING_VIRTUAL_NODES,
) -> tuple[float, ...]:
    """Return each SSU's exact fraction of the SHA-256 hash ring."""
    if int(num_ssu) <= 0:
        raise ValueError("num_ssu must be positive")
    if int(virtual_nodes_per_ssu) <= 0:
        raise ValueError("virtual_nodes_per_ssu must be positive")
    entries = sorted(
        (
            sim._sha256_ring_position(
                b"qos_storage_sim:block_ring_hash:vnode:v1\0",
                disk_id,
                virtual_node,
            ),
            disk_id,
        )
        for disk_id in range(int(num_ssu))
        for virtual_node in range(int(virtual_nodes_per_ssu))
    )
    positions = tuple(position for position, _ in entries)
    disk_ids = tuple(disk_id for _, disk_id in entries)
    owned_intervals = [0] * int(num_ssu)
    previous = positions[-1] - RING_SIZE
    for position, disk_id in zip(positions, disk_ids):
        owned_intervals[disk_id] += position - previous
        previous = position
    return tuple(interval / RING_SIZE for interval in owned_intervals)


def analyze_ssu(
    num_ssu: int,
    virtual_nodes_per_ssu: int = sim.BLOCK_RING_VIRTUAL_NODES,
) -> dict:
    shares = ownership_by_ssu(num_ssu, virtual_nodes_per_ssu)
    normalized = tuple(float(num_ssu) * share for share in shares)
    min_id = min(range(num_ssu), key=normalized.__getitem__)
    max_id = max(range(num_ssu), key=normalized.__getitem__)
    return {
        "num_ssu": int(num_ssu),
        "virtual_nodes_per_ssu": int(virtual_nodes_per_ssu),
        "ownership_shares": list(shares),
        "normalized_ownership": list(normalized),
        "normalized_mean": statistics.fmean(normalized),
        "normalized_min": normalized[min_id],
        "normalized_min_ssu_id": min_id,
        "normalized_max": normalized[max_id],
        "normalized_max_ssu_id": max_id,
        "normalized_population_stddev": statistics.pstdev(normalized),
        "conservation_error": abs(sum(shares) - 1.0),
    }


def analyze(
    ssu_list=DEFAULT_SSU_LIST,
    *,
    vnode_sensitivity=DEFAULT_VNODE_SENSITIVITY,
    sensitivity_ssu_list=DEFAULT_SENSITIVITY_SSU_LIST,
) -> dict:
    requested = tuple(dict.fromkeys(int(value) for value in ssu_list))
    if not requested or any(value <= 0 for value in requested):
        raise ValueError("ssu_list must contain positive values")
    rows = [analyze_ssu(num_ssu) for num_ssu in requested]
    vnode_values = tuple(
        dict.fromkeys(int(value) for value in vnode_sensitivity)
    )
    sensitivity_ssus = tuple(
        dict.fromkeys(int(value) for value in sensitivity_ssu_list)
    )
    if not vnode_values or any(value <= 0 for value in vnode_values):
        raise ValueError("vnode_sensitivity must contain positive values")
    if not sensitivity_ssus or any(value <= 0 for value in sensitivity_ssus):
        raise ValueError("sensitivity_ssu_list must contain positive values")
    vnode_rows = [
        analyze_ssu(num_ssu, vnode_count)
        for num_ssu in sensitivity_ssus
        for vnode_count in vnode_values
    ]
    by_ssu = {row["num_ssu"]: row for row in rows}
    comparisons = []
    for smaller, larger in ((10, 40), (18, 70)):
        if smaller not in by_ssu or larger not in by_ssu:
            continue
        low = by_ssu[smaller]
        high = by_ssu[larger]
        comparisons.append(
            {
                "from_ssu": smaller,
                "to_ssu": larger,
                "max_excess_change": (
                    high["normalized_max"] - low["normalized_max"]
                ),
                "min_deficit_change": (
                    (1.0 - high["normalized_min"])
                    - (1.0 - low["normalized_min"])
                ),
                "stddev_change": (
                    high["normalized_population_stddev"]
                    - low["normalized_population_stddev"]
                ),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "method": "exact_sha256_ring_interval_integration",
        "source_fingerprint": _source_fingerprint(),
        "source_files": list(SOURCE_FILES),
        "ring_virtual_nodes_per_ssu": sim.BLOCK_RING_VIRTUAL_NODES,
        "ssu_list": list(requested),
        "rows": rows,
        "scale_comparisons": comparisons,
        "vnode_sensitivity": {
            "method": "counterfactual_exact_ring_interval_integration",
            "ssu_list": list(sensitivity_ssus),
            "virtual_nodes_per_ssu": list(vnode_values),
            "rows": vnode_rows,
            "scope": (
                "placement-only counterfactual; the completed simulations and "
                "their fingerprints remain fixed at 256 virtual nodes per SSU"
            ),
            "recommendation": (
                "Increasing virtual nodes is a placement-balance candidate for "
                "a new experiment, not a reinterpretation of current results"
            ),
        },
        "conclusion": (
            "Keeping NPU:SSU capacity ratio fixed does not keep fixed-placement "
            "hotspot severity fixed: the maximum ownership rises from 1.0555x "
            "at SSU10 to 1.1696x at SSU40, and from 1.0592x at SSU18 to "
            "1.1504x at SSU70. Thus 32-NPU ratio points are capacity brackets, "
            "not placement-equivalent predictions for 128 NPU."
        ),
    }


def _pct_delta(normalized: float) -> str:
    return f"{100.0 * (normalized - 1.0):+.2f}%"


def render_markdown(result: dict) -> str:
    lines = [
        "# Consistent-hash ring ownership scale audit",
        "",
        "Method: exact integration of all adjacent SHA-256 ring intervals; no "
        "block-key sampling. Ownership is normalized so an ideally balanced SSU "
        "has `1.0x`.",
        "",
        f"Virtual nodes per SSU: `{result['ring_virtual_nodes_per_ssu']}`",
        "",
        "| SSU count | Min ownership | Min SSU | Max ownership | Max SSU | "
        "Population stddev | Conservation error |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["rows"]:
        lines.append(
            "| {num_ssu} | {minimum:.6f}x ({min_delta}) | {min_id} | "
            "{maximum:.6f}x ({max_delta}) | {max_id} | {stddev:.6f} | "
            "{error:.3e} |".format(
                num_ssu=row["num_ssu"],
                minimum=row["normalized_min"],
                min_delta=_pct_delta(row["normalized_min"]),
                min_id=row["normalized_min_ssu_id"],
                maximum=row["normalized_max"],
                max_delta=_pct_delta(row["normalized_max"]),
                max_id=row["normalized_max_ssu_id"],
                stddev=row["normalized_population_stddev"],
                error=row["conservation_error"],
            )
        )
    lines.extend(
        [
            "",
            "## Virtual-node sensitivity (placement-only counterfactual)",
            "",
            result["vnode_sensitivity"]["scope"],
            "",
            "| SSU count | Virtual nodes / SSU | Min ownership | Max ownership | "
            "Population stddev |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["vnode_sensitivity"]["rows"]:
        lines.append(
            f"| {row['num_ssu']} | {row['virtual_nodes_per_ssu']} | "
            f"{row['normalized_min']:.6f}x | {row['normalized_max']:.6f}x | "
            f"{row['normalized_population_stddev']:.6f} |"
        )
    lines.extend(
        [
            "",
            result["vnode_sensitivity"]["recommendation"],
            "",
            "## Scale conclusion",
            "",
            result["conclusion"],
            "",
            "In the completed 128-NPU SSU40 row, the persistent queue hotspot is "
            "SSU4, which is also the exact ring-ownership maximum (1.169584x). "
            "This alignment is evidence that the observed scale loss is driven "
            "by fixed placement, although it does not by itself prove every "
            "end-to-end latency effect.",
            "",
            f"Source fingerprint: `{result['source_fingerprint']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def run(
    output=OUTPUT,
    report=REPORT,
    *,
    ssu_list=DEFAULT_SSU_LIST,
    vnode_sensitivity=DEFAULT_VNODE_SENSITIVITY,
    sensitivity_ssu_list=DEFAULT_SENSITIVITY_SSU_LIST,
):
    result = analyze(
        ssu_list,
        vnode_sensitivity=vnode_sensitivity,
        sensitivity_ssu_list=sensitivity_ssu_list,
    )
    _write(
        Path(output),
        json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n",
    )
    _write(Path(report), render_markdown(result))
    return Path(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--ssu", action="append", type=int)
    parser.add_argument("--vnode", action="append", type=int)
    parser.add_argument("--sensitivity-ssu", action="append", type=int)
    args = parser.parse_args(argv)
    run(
        args.output,
        args.report,
        ssu_list=tuple(args.ssu or DEFAULT_SSU_LIST),
        vnode_sensitivity=tuple(args.vnode or DEFAULT_VNODE_SENSITIVITY),
        sensitivity_ssu_list=tuple(
            args.sensitivity_ssu or DEFAULT_SENSITIVITY_SSU_LIST
        ),
    )


if __name__ == "__main__":
    main()
