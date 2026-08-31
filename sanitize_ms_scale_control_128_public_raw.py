"""Publish privacy-sanitized selected128 shards without changing raw evidence.

The sanitizer validates all seven private shards through the runner's canonical
formal validator and the independent summary builder before changing anything.
It then removes only process/host identifiers and nulls the non-scientific input
cache path.  Sanitized shards must pass both validators again before an atomic
directory publish makes them visible.

This file imports the runner; the runner never imports this file, so it does not
expand the experiment source-fingerprint closure.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from tempfile import TemporaryDirectory
from typing import Mapping, Sequence

import build_ms_scale_control_128_selected_summary as summary_builder
from ms_scale_control_experiment import (
    SELECTED128_SSUS,
    validate_selected128_campaign_document,
    validate_selected128_formal_payload,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CAMPAIGN_SPEC = Path("campaigns/selected128_alpha_tuned_v2.json")
DEFAULT_RAW_DIR = Path("results/ms_scale_control/selected128_alpha_tuned_v2_raw")
DEFAULT_OUTPUT_DIR = Path(
    "results/ms_scale_control/selected128_alpha_tuned_v2_public_raw"
)
MANIFEST_RELATIVE_PATH = Path("publication") / "manifest.json"
PUBLICATION_SCHEMA_VERSION = 1
EXPECTED_RESULT_COUNT = 10
HOME_PATH_MARKER = "/" + "home" + "/"

SECRET_PATTERNS = (
    (
        "GitHub credential prefix",
        re.compile("(?i)(?:gh" + "p_|github_" + "pat_|gh[ousr]_)"),
    ),
    (
        "private-key marker",
        re.compile("-----BEGIN " + r"(?:[A-Z ]+ )?PRIVATE KEY-----"),
    ),
    ("AWS access-key prefix", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    (
        "credentialed URL",
        re.compile(r"(?i)https?://[^\s/:@]+:[^\s/@]+@"),
    ),
    ("sshpass invocation", re.compile(r"(?i)\bsshpass\b")),
)
SENSITIVE_KEYS = frozenset(
    {
        "password",
        "passwd",
        "secret",
        "token",
        "api_key",
        "access_token",
        "auth_token",
        "credential",
        "private_key",
    }
)
PRIVACY_KEYS = frozenset(("hostname", "pid"))


class SanitizationError(ValueError):
    """Raised when private input cannot produce a validated public artifact."""


@dataclass(frozen=True)
class CampaignInfo:
    path: Path
    sha256: str
    size_bytes: int
    builder_document: summary_builder.CampaignDocument


@dataclass(frozen=True)
class ShardInfo:
    path: Path
    payload: dict[str, object]
    sha256: str
    size_bytes: int
    num_ssu: int
    formal_identity: dict[str, object]


def _require(condition: object, message: str) -> None:
    if not condition:
        raise SanitizationError(message)


def _reject_duplicate_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise SanitizationError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str):
    raise SanitizationError(f"non-finite JSON constant: {value}")


def _decode_json(raw: bytes, context: str) -> dict[str, object]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SanitizationError(f"{context}: invalid UTF-8 JSON: {error}") from error
    _require(isinstance(value, dict), f"{context}: top level must be an object")
    return value


def _encode_json(value: object, *, pretty: bool = False) -> bytes:
    options = {
        "ensure_ascii": False,
        "allow_nan": False,
        "sort_keys": True,
    }
    if pretty:
        text = json.dumps(value, indent=2, **options)
    else:
        text = json.dumps(value, separators=(",", ":"), **options)
    return (text + "\n").encode("utf-8")


def _canonical_hash(value: object, namespace: bytes) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(namespace + encoded).hexdigest()


def _resolve_input(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = PROJECT_ROOT / expanded
    return expanded.resolve()


def _portable_project_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError as error:
        raise SanitizationError(
            "public/campaign paths must stay inside the project"
        ) from error


def _read_campaign(path: Path) -> CampaignInfo:
    resolved = _resolve_input(path)
    _portable_project_path(resolved)
    try:
        raw = resolved.read_bytes()
    except OSError as error:
        raise SanitizationError(f"cannot read frozen campaign: {error}") from error
    document = _decode_json(raw, "campaign")
    try:
        validate_selected128_campaign_document(document)
    except (KeyError, TypeError, ValueError) as error:
        raise SanitizationError(f"frozen campaign is invalid: {error}") from error
    digest = sha256(raw).hexdigest()
    builder_document = summary_builder.CampaignDocument(
        path=resolved,
        sha256=digest,
        size_bytes=len(raw),
    )
    return CampaignInfo(resolved, digest, len(raw), builder_document)


def _discover_shards(raw_dir: Path, explicit: Sequence[Path]) -> list[Path]:
    if explicit:
        paths = [_resolve_input(path) for path in explicit]
    else:
        resolved = _resolve_input(raw_dir)
        _require(resolved.is_dir(), "private raw directory does not exist")
        paths = sorted(path.resolve() for path in resolved.glob("*.json"))
    _require(
        len(paths) == len(SELECTED128_SSUS),
        f"expected seven private shards, found {len(paths)}",
    )
    _require(len(set(paths)) == len(paths), "private shard paths are duplicated")
    for path in paths:
        _require(path.is_file(), "a private shard is not a regular file")
    return paths


def _selected_ssu(payload: Mapping[str, object], context: str) -> int:
    selected = payload.get("selected_ssus")
    _require(
        isinstance(selected, list)
        and len(selected) == 1
        and type(selected[0]) is int
        and selected[0] in SELECTED128_SSUS,
        f"{context}: expected exactly one frozen SSU",
    )
    return selected[0]


def _validate_scientific_authentication_linkage(
    payload: Mapping[str, object], context: str
) -> None:
    top_level = payload.get("input_authentication")
    experiment_spec = payload.get("experiment_spec")
    source_manifest = payload.get("source_manifest")
    _require(isinstance(top_level, dict), f"{context}: input authentication missing")
    _require(isinstance(experiment_spec, dict), f"{context}: experiment spec missing")
    workload = experiment_spec.get("workload")
    _require(isinstance(workload, dict), f"{context}: workload spec missing")
    _require(
        top_level == workload.get("authentication"),
        f"{context}: top-level and authenticated workload inputs differ",
    )
    _require(isinstance(source_manifest, dict), f"{context}: source manifest missing")
    _require(
        top_level.get("source_sha256") == source_manifest.get("data"),
        f"{context}: authenticated data SHA differs from source manifest",
    )


def _read_private_shards(
    paths: Sequence[Path], campaign: CampaignInfo
) -> list[ShardInfo]:
    shards = []
    identities = []
    seen_ssus = set()
    for index, path in enumerate(paths):
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise SanitizationError(
                f"private shard {index + 1}: cannot read: {error}"
            ) from error
        payload = _decode_json(raw, f"private shard {index + 1}")
        num_ssu = _selected_ssu(payload, f"private shard {index + 1}")
        _require(num_ssu not in seen_ssus, f"duplicate private shard for SSU{num_ssu}")
        try:
            identity = validate_selected128_formal_payload(
                payload,
                expected_ssu=num_ssu,
                expected_campaign_sha256=campaign.sha256,
            )
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise SanitizationError(
                f"private SSU{num_ssu}: canonical validation failed: {error}"
            ) from error
        _require(isinstance(identity, dict), "formal validator returned no identity")
        _validate_scientific_authentication_linkage(payload, f"private SSU{num_ssu}")
        digest = sha256(raw).hexdigest()
        shards.append(ShardInfo(path, payload, digest, len(raw), num_ssu, identity))
        identities.append(identity)
        seen_ssus.add(num_ssu)
    _require(seen_ssus == set(SELECTED128_SSUS), "private SSU grid is incomplete")
    _require(
        all(identity == identities[0] for identity in identities[1:]),
        "private shards do not share formal source/config/campaign identity",
    )
    return sorted(shards, key=lambda shard: shard.num_ssu)


def _builder_validate(
    shards: Sequence[ShardInfo],
    campaign: CampaignInfo,
    *,
    use_actual_paths: bool,
) -> None:
    documents = []
    for shard in shards:
        path = (
            shard.path
            if use_actual_paths
            else PROJECT_ROOT
            / "results"
            / ".selected128-sanitizer-private-virtual"
            / f"selected128_seed42_ssu{shard.num_ssu}.json"
        )
        documents.append(
            summary_builder.ShardDocument(
                path=path,
                payload=shard.payload,
                sha256=shard.sha256,
                size_bytes=shard.size_bytes,
            )
        )
    try:
        rows, _provenance = summary_builder._validate_shards(
            documents, campaign.builder_document
        )
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise SanitizationError(f"summary builder rejected shards: {error}") from error
    _require(len(rows) == 70, "summary builder did not validate the 70-row grid")


def _pop_runtime_privacy(runtime: object, context: str) -> list[dict[str, object]]:
    _require(isinstance(runtime, dict), f"{context}: runtime is missing")
    hostname = runtime.pop("hostname", None)
    pid = runtime.pop("pid", None)
    _require(
        isinstance(hostname, str) and bool(hostname),
        f"{context}: expected an original hostname",
    )
    _require(type(pid) is int and pid > 0, f"{context}: expected an original PID")
    return [
        {
            "json_path": f"{context}.hostname",
            "operation": "remove",
            "occurrences": 1,
        },
        {
            "json_path": f"{context}.pid",
            "operation": "remove",
            "occurrences": 1,
        },
    ]


def _sanitize_payload(
    original: Mapping[str, object], num_ssu: int
) -> tuple[dict[str, object], list[dict[str, object]]]:
    public = copy.deepcopy(original)
    transforms = _pop_runtime_privacy(public.get("runtime"), "runtime")

    results = public.get("results")
    _require(
        isinstance(results, list) and len(results) == EXPECTED_RESULT_COUNT,
        f"SSU{num_ssu}: expected ten result rows",
    )
    row_hostname_count = 0
    row_pid_count = 0
    for index, row in enumerate(results):
        _require(isinstance(row, dict), f"SSU{num_ssu}: malformed result row")
        row_transforms = _pop_runtime_privacy(
            row.get("runtime"), f"results[{index}].runtime"
        )
        row_hostname_count += row_transforms[0]["occurrences"]
        row_pid_count += row_transforms[1]["occurrences"]
    transforms.extend(
        [
            {
                "json_path": "results[*].runtime.hostname",
                "operation": "remove",
                "occurrences": row_hostname_count,
            },
            {
                "json_path": "results[*].runtime.pid",
                "operation": "remove",
                "occurrences": row_pid_count,
            },
        ]
    )

    loader_environment = public.get("input_loader_environment")
    _require(
        isinstance(loader_environment, dict),
        f"SSU{num_ssu}: input loader environment is missing",
    )
    _require(
        "cache_path" in loader_environment,
        f"SSU{num_ssu}: loader cache path field is missing",
    )
    cache_path = loader_environment["cache_path"]
    _require(
        cache_path is None or isinstance(cache_path, str),
        f"SSU{num_ssu}: loader cache path has the wrong type",
    )
    loader_environment["cache_path"] = None
    transforms.append(
        {
            "json_path": "input_loader_environment.cache_path",
            "operation": "replace_with_null",
            "occurrences": int(cache_path is not None),
        }
    )
    return public, transforms


def _is_absolute_or_home_path(value: str) -> bool:
    normalized = value.replace("\\", "/")
    return (
        normalized.startswith("/")
        or normalized.startswith("~/")
        or HOME_PATH_MARKER in normalized
        or re.match(r"^[A-Za-z]:/", normalized) is not None
        or normalized.lower().startswith("file://")
    )


def _audit_public_value(value: object, context: str = "public") -> None:
    if isinstance(value, dict):
        for index, (key, item) in enumerate(value.items()):
            _require(isinstance(key, str), f"{context}: non-string mapping key")
            normalized_key = key.lower()
            _require(
                normalized_key not in SENSITIVE_KEYS,
                f"{context}: sensitive-key category at mapping index {index}",
            )
            _require(
                normalized_key not in PRIVACY_KEYS,
                f"{context}: unsanitized privacy key at mapping index {index}",
            )
            if normalized_key == "cache_path":
                _require(item is None, f"{context}: cache path was not nulled")
            _audit_public_value(item, f"{context}[{index}]")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _audit_public_value(item, f"{context}[{index}]")
        return
    if not isinstance(value, str):
        return
    _require(
        not _is_absolute_or_home_path(value),
        f"{context}: absolute/home path rejected",
    )
    for category, pattern in SECRET_PATTERNS:
        _require(pattern.search(value) is None, f"{context}: rejected {category}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), 0o644)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        finally:
            raise


def _assert_private_sources_unchanged(shards: Sequence[ShardInfo]) -> None:
    for shard in shards:
        try:
            raw = shard.path.read_bytes()
        except OSError as error:
            raise SanitizationError(
                f"private SSU{shard.num_ssu}: cannot re-read source: {error}"
            ) from error
        _require(
            len(raw) == shard.size_bytes and sha256(raw).hexdigest() == shard.sha256,
            f"private SSU{shard.num_ssu}: source bytes changed during publication",
        )


def _public_name(num_ssu: int) -> str:
    return f"selected128_seed42_ssu{num_ssu}.json"


def _published_shard_from_file(
    path: Path,
    num_ssu: int,
    campaign: CampaignInfo,
    expected_identity: Mapping[str, object],
    expected_payload: Mapping[str, object],
) -> ShardInfo:
    raw = path.read_bytes()
    payload = _decode_json(raw, f"public SSU{num_ssu}")
    _require(
        payload == expected_payload,
        f"public SSU{num_ssu}: serialization changed payload values",
    )
    _audit_public_value(payload, f"public SSU{num_ssu}")
    _validate_scientific_authentication_linkage(payload, f"public SSU{num_ssu}")
    try:
        identity = validate_selected128_formal_payload(
            payload,
            expected_ssu=num_ssu,
            expected_campaign_sha256=campaign.sha256,
        )
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise SanitizationError(
            f"public SSU{num_ssu}: canonical validation failed: {error}"
        ) from error
    _require(
        identity == expected_identity,
        f"public SSU{num_ssu}: formal identity changed during sanitization",
    )
    return ShardInfo(
        path,
        payload,
        sha256(raw).hexdigest(),
        len(raw),
        num_ssu,
        identity,
    )


def _manifest(
    private_shards: Sequence[ShardInfo],
    public_shards: Sequence[ShardInfo],
    transforms_by_ssu: Mapping[int, Sequence[Mapping[str, object]]],
    campaign: CampaignInfo,
    output_dir: Path,
) -> dict[str, object]:
    private_by_ssu = {shard.num_ssu: shard for shard in private_shards}
    public_by_ssu = {shard.num_ssu: shard for shard in public_shards}
    identity = private_shards[0].formal_identity
    output_relative = _portable_project_path(output_dir)
    return {
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "artifact": "selected128_public_raw_sanitization_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True,
        "num_npu": 128,
        "ssu_counts": list(SELECTED128_SSUS),
        "shard_count": len(SELECTED128_SSUS),
        "campaign": {
            "path": _portable_project_path(campaign.path),
            "sha256": campaign.sha256,
            "size_bytes": campaign.size_bytes,
        },
        "formal_identity": {
            "source_fingerprint": identity["source_fingerprint"],
            "config_fingerprint": identity["config_fingerprint"],
            "campaign_spec_sha256": identity["campaign_spec_sha256"],
            "runtime_identity": identity["runtime_identity"],
        },
        "privacy_policy": {
            "version": 1,
            "scientific_fields_changed": False,
            "private_source_names_published": False,
            "runner_lock_sidecars_published": False,
            "service_logs_published": False,
            "transforms": [
                {
                    "json_path": "runtime.hostname",
                    "operation": "remove",
                    "reason": "host identifier",
                },
                {
                    "json_path": "runtime.pid",
                    "operation": "remove",
                    "reason": "ephemeral process identifier",
                },
                {
                    "json_path": "results[*].runtime.hostname",
                    "operation": "remove",
                    "reason": "host identifier",
                },
                {
                    "json_path": "results[*].runtime.pid",
                    "operation": "remove",
                    "reason": "ephemeral process identifier",
                },
                {
                    "json_path": "input_loader_environment.cache_path",
                    "operation": "replace_with_null",
                    "reason": "non-scientific local filesystem location",
                },
            ],
        },
        "validation": {
            "private_canonical_formal_validator": True,
            "private_summary_builder": True,
            "public_canonical_formal_validator": True,
            "public_summary_builder": True,
            "private_source_bytes_unchanged": True,
            "public_path_and_secret_scan": True,
            "validated_result_rows": 70,
        },
        "files": [
            {
                "num_ssu": num_ssu,
                "original_sha256": private_by_ssu[num_ssu].sha256,
                "original_size_bytes": private_by_ssu[num_ssu].size_bytes,
                "public_path": (f"{output_relative}/{_public_name(num_ssu)}"),
                "public_sha256": public_by_ssu[num_ssu].sha256,
                "public_size_bytes": public_by_ssu[num_ssu].size_bytes,
                "applied_transforms": list(transforms_by_ssu[num_ssu]),
            }
            for num_ssu in SELECTED128_SSUS
        ],
    }


def _publish(
    private_shards: Sequence[ShardInfo],
    campaign: CampaignInfo,
    output_dir: Path,
) -> dict[str, object]:
    output_dir = _resolve_input(output_dir)
    _portable_project_path(output_dir)
    _require(output_dir != PROJECT_ROOT, "public output cannot be the project root")
    _require(not output_dir.exists(), "public output directory already exists")
    for shard in private_shards:
        _require(
            output_dir not in shard.path.parents,
            "private input cannot be nested below the public output",
        )

    _builder_validate(private_shards, campaign, use_actual_paths=False)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            dir=output_dir.parent,
            prefix=f".{output_dir.name}.staging-",
        )
    )
    try:
        public_shards = []
        transforms_by_ssu = {}
        for private in private_shards:
            public_payload, transforms = _sanitize_payload(
                private.payload, private.num_ssu
            )
            _audit_public_value(public_payload, f"public SSU{private.num_ssu}")
            try:
                identity = validate_selected128_formal_payload(
                    public_payload,
                    expected_ssu=private.num_ssu,
                    expected_campaign_sha256=campaign.sha256,
                )
            except (AttributeError, KeyError, TypeError, ValueError) as error:
                raise SanitizationError(
                    f"public SSU{private.num_ssu}: canonical validation failed: {error}"
                ) from error
            _require(
                identity == private.formal_identity,
                f"SSU{private.num_ssu}: formal identity changed",
            )
            public_path = staging / _public_name(private.num_ssu)
            _atomic_write_bytes(public_path, _encode_json(public_payload))
            public_shards.append(
                _published_shard_from_file(
                    public_path,
                    private.num_ssu,
                    campaign,
                    private.formal_identity,
                    public_payload,
                )
            )
            transforms_by_ssu[private.num_ssu] = transforms

        _builder_validate(public_shards, campaign, use_actual_paths=True)
        _assert_private_sources_unchanged(private_shards)
        manifest = _manifest(
            private_shards,
            public_shards,
            transforms_by_ssu,
            campaign,
            output_dir,
        )
        _audit_public_value(manifest, "publication manifest")
        manifest_path = staging / MANIFEST_RELATIVE_PATH
        _atomic_write_bytes(manifest_path, _encode_json(manifest, pretty=True))
        manifest_round_trip = _decode_json(
            manifest_path.read_bytes(), "publication manifest"
        )
        _audit_public_value(manifest_round_trip, "publication manifest")
        _require(manifest_round_trip == manifest, "manifest round trip changed")
        _require(
            len(list(staging.glob("*.json"))) == len(SELECTED128_SSUS),
            "public raw root must contain exactly seven JSON shards",
        )
        _assert_private_sources_unchanged(private_shards)
        os.chmod(staging, 0o755)
        os.chmod(manifest_path.parent, 0o755)
        _require(not output_dir.exists(), "public output appeared during staging")
        staging.rename(output_dir)
        _fsync_directory(output_dir.parent)
        return manifest
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _synthetic_private_payload(payload: Mapping[str, object], index: int):
    value = json.loads(json.dumps(payload))
    spec = value["experiment_spec"]
    spec["workload"]["authentication"] = copy.deepcopy(value["input_authentication"])
    config_fingerprint = _canonical_hash(spec, b"ms-scale-control-config:v1\0")
    value["config_fingerprint"] = config_fingerprint
    value["ending_config_fingerprint"] = config_fingerprint
    for row in value["results"]:
        row["config_fingerprint"] = config_fingerprint
        row["case_fingerprint"] = _canonical_hash(
            {
                "case": row["case_spec"],
                "num_ssu": row["num_ssu"],
                "source_fingerprint": row["source_fingerprint"],
                "config_fingerprint": config_fingerprint,
            },
            b"ms-scale-control-case:v1\0",
        )
    runtimes = [value["runtime"]] + [row["runtime"] for row in value["results"]]
    for runtime_index, runtime in enumerate(runtimes):
        runtime["hostname"] = f"private-host-{index}"
        runtime["pid"] = 10_000 + 100 * index + runtime_index
    value["input_loader_environment"] = {
        "cache_path": "/" + "home" + "/private/cache/catalog.npy",
        "cache_present": True,
        "cache_verified_equal": True,
    }
    return value


def _expect_audit_rejection(value: object, expected: str) -> None:
    try:
        _audit_public_value(value, "negative self-test")
    except SanitizationError as error:
        _require(expected in str(error), "negative self-test rejected for wrong reason")
    else:
        raise SanitizationError("negative self-test unexpectedly passed")


def _self_test() -> dict[str, object]:
    # Also validate the real frozen campaign used by the production defaults.
    _read_campaign(DEFAULT_CAMPAIGN_SPEC)
    documents, builder_campaign = summary_builder._synthetic_documents()
    campaign = CampaignInfo(
        path=builder_campaign.path,
        sha256=builder_campaign.sha256,
        size_bytes=builder_campaign.size_bytes,
        builder_document=builder_campaign,
    )
    self_test_parent = PROJECT_ROOT / "results"
    self_test_parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        dir=self_test_parent, prefix=".selected128-sanitizer-selftest-"
    ) as temporary:
        temporary_root = Path(temporary)
        private_dir = temporary_root / "private"
        private_dir.mkdir()
        paths = []
        original_hashes = {}
        for index, document in enumerate(documents):
            num_ssu = document.payload["selected_ssus"][0]
            path = private_dir / _public_name(num_ssu)
            raw = _encode_json(_synthetic_private_payload(document.payload, index))
            _atomic_write_bytes(path, raw)
            paths.append(path)
            original_hashes[num_ssu] = sha256(raw).hexdigest()
        private_shards = _read_private_shards(paths, campaign)
        output_dir = temporary_root / "public_raw"
        manifest = _publish(private_shards, campaign, output_dir)
        _require(manifest["complete"] is True, "self-test manifest is incomplete")
        _require(
            len(manifest["files"]) == len(SELECTED128_SSUS),
            "self-test manifest shard count differs",
        )
        for file_record in manifest["files"]:
            _require(
                file_record["original_sha256"] != file_record["public_sha256"],
                "self-test raw-to-public SHA mapping did not change",
            )
            applied = {
                transform["json_path"]: transform["occurrences"]
                for transform in file_record["applied_transforms"]
            }
            _require(
                applied
                == {
                    "runtime.hostname": 1,
                    "runtime.pid": 1,
                    "results[*].runtime.hostname": EXPECTED_RESULT_COUNT,
                    "results[*].runtime.pid": EXPECTED_RESULT_COUNT,
                    "input_loader_environment.cache_path": 1,
                },
                "self-test applied-transform record differs",
            )
        _require(
            len(list(output_dir.glob("*.json"))) == len(SELECTED128_SSUS),
            "self-test public raw is not builder-discoverable",
        )
        for shard in private_shards:
            _require(
                sha256(shard.path.read_bytes()).hexdigest()
                == original_hashes[shard.num_ssu],
                f"self-test changed private SSU{shard.num_ssu}",
            )
        _expect_audit_rejection(
            {"leak": "/" + "home" + "/private/result"}, "absolute/home path"
        )
        _expect_audit_rejection(
            {"leak": "gh" + "p_" + "x" * 36}, "GitHub credential prefix"
        )
        bad_authentication = copy.deepcopy(private_shards[0].payload)
        replacement_digest = sha256(b"different authenticated input").hexdigest()
        bad_authentication["input_authentication"]["source_sha256"] = replacement_digest
        bad_authentication["experiment_spec"]["workload"]["authentication"][
            "source_sha256"
        ] = replacement_digest
        try:
            _validate_scientific_authentication_linkage(
                bad_authentication, "negative self-test"
            )
        except SanitizationError as error:
            _require(
                "source manifest" in str(error),
                "scientific-auth negative test rejected for the wrong reason",
            )
        else:
            raise SanitizationError(
                "scientific-auth negative self-test unexpectedly passed"
            )
    return {
        "self_test": "passed",
        "validated_private_shards": len(SELECTED128_SSUS),
        "validated_public_shards": len(SELECTED128_SSUS),
        "validated_result_rows": 70,
        "private_bytes_unchanged": True,
        "builder_rediscovery_json_count": len(SELECTED128_SSUS),
        "path_rejection": True,
        "secret_rejection": True,
        "scientific_auth_linkage_rejection": True,
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-spec",
        type=Path,
        default=DEFAULT_CAMPAIGN_SPEC,
        help="exact frozen campaign JSON used to authenticate every shard",
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument(
        "--shard",
        action="append",
        type=Path,
        help="explicit private shard path; repeat exactly seven times",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.self_test:
        print(json.dumps(_self_test(), ensure_ascii=False, indent=2))
        return 0
    campaign = _read_campaign(args.campaign_spec)
    paths = _discover_shards(args.raw_dir, tuple(args.shard or ()))
    private_shards = _read_private_shards(paths, campaign)
    manifest = _publish(private_shards, campaign, args.output_dir)
    print(
        json.dumps(
            {
                "complete": True,
                "public_raw": _portable_project_path(_resolve_input(args.output_dir)),
                "manifest": (
                    _portable_project_path(_resolve_input(args.output_dir))
                    + "/"
                    + MANIFEST_RELATIVE_PATH.as_posix()
                ),
                "shard_count": manifest["shard_count"],
                "private_source_bytes_unchanged": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SanitizationError as error:
        raise SystemExit(f"ERROR: {error}") from error
