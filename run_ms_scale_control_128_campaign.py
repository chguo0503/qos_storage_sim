"""Run a resource-bounded queue of selected 128-NPU experiment shards."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from ms_scale_control_experiment import (
    SELECTED128_EXPECTED_RUNTIME_IDENTITY,
    SELECTED128_FORMAL_MAX_WORKERS,
    SELECTED128_SSUS,
    THREAD_LIMIT_ENVIRONMENT,
    _load_campaign_spec,
    acquire_output_lock,
    current_runtime_merge_identity,
    validate_selected128_campaign_document,
    validate_selected128_formal_payload,
)


ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "ms_scale_control_experiment.py"
DEFAULT_CAMPAIGN_SPEC = ROOT / "campaigns/selected128_alpha_tuned_v1.json"
DEFAULT_OUTPUT_DIR = ROOT / "results/ms_scale_control/selected128_alpha_tuned_v1_raw"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--campaign-spec", type=Path, default=DEFAULT_CAMPAIGN_SPEC)
    parser.add_argument("--ssu", action="append", type=int, required=True)
    parser.add_argument(
        "--max-workers", type=int, default=SELECTED128_FORMAL_MAX_WORKERS
    )
    return parser.parse_args(argv)


def _relative_or_resolved(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _validate_environment():
    invalid = {
        name: os.environ.get(name)
        for name in THREAD_LIMIT_ENVIRONMENT
        if os.environ.get(name) != "1"
    }
    if invalid:
        raise RuntimeError(f"thread limits must all equal 1: {invalid}")


def _validate_result(path: Path, expected_ssu: int, campaign_sha256: str):
    payload = json.loads(path.read_text(encoding="utf-8"))
    identity = validate_selected128_formal_payload(
        payload,
        expected_ssu=expected_ssu,
        expected_campaign_sha256=campaign_sha256,
    )
    return payload, identity


def main(argv=None):
    args = parse_args(argv)
    _validate_environment()
    if not 1 <= args.max_workers <= SELECTED128_FORMAL_MAX_WORKERS:
        raise ValueError(
            f"max-workers must be in [1, {SELECTED128_FORMAL_MAX_WORKERS}]"
        )
    if current_runtime_merge_identity() != SELECTED128_EXPECTED_RUNTIME_IDENTITY:
        raise RuntimeError("wrapper runtime differs from the frozen campaign runtime")
    ssus = tuple(dict.fromkeys(args.ssu))
    if any(ssu not in SELECTED128_SSUS for ssu in ssus):
        raise ValueError(f"SSUs must be selected from {SELECTED128_SSUS}")
    campaign_spec = _relative_or_resolved(args.campaign_spec)
    output_dir = _relative_or_resolved(args.output_dir)
    campaign_document, campaign_authentication = _load_campaign_spec(campaign_spec)
    validate_selected128_campaign_document(campaign_document)
    campaign_sha256 = campaign_authentication["sha256"]
    output_dir.mkdir(parents=True, exist_ok=True)

    shared_identity = None
    for ssu in ssus:
        output = output_dir / f"selected128_seed42_ssu{ssu}.json"
        command = [
            sys.executable,
            str(RUNNER),
            "--definition",
            "selected128",
            "--campaign-spec",
            str(campaign_spec),
            "--output",
            str(output),
            "--ssu",
            str(ssu),
            "--seed",
            "42",
            "--requests-per-npu",
            "128",
            "--warmup-requests",
            "8",
            "--settle-ms",
            "500",
            "--measurement-ms",
            "8000",
            "--block-ms",
            "500",
            "--max-workers",
            str(args.max_workers),
            "--mp-start-method",
            "spawn",
        ]
        print(
            json.dumps(
                {"event": "campaign_shard_start", "ssu": ssu, "output": str(output)},
                separators=(",", ":"),
            ),
            flush=True,
        )
        with acquire_output_lock(output, owner="launcher"):
            subprocess.run(command, cwd=ROOT, check=True)
            payload, identity = _validate_result(output, ssu, campaign_sha256)
        if shared_identity is None:
            shared_identity = identity
        elif identity != shared_identity:
            raise RuntimeError("completed shards do not share source/config/campaign")
        print(
            json.dumps(
                {
                    "event": "campaign_shard_complete",
                    "ssu": ssu,
                    "result_count": len(payload["results"]),
                    "source_fingerprint": payload["source_fingerprint"],
                    "config_fingerprint": payload["config_fingerprint"],
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
