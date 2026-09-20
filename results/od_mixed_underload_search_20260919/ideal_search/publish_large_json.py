"""Losslessly package, verify, or restore the three archived large JSON files.

The original JSON bytes are never parsed, reformatted, deleted, or overwritten.
Default operation is read-only verification. All paths are relative to this file.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import os
import shutil
import tempfile

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "large_json_archives.json"
NAMES = ("safe_valid.json", "resonance_valid.json", "expanded_results.json")
CHUNK = 1024 * 1024


def fingerprint(stream):
    digest = hashlib.sha256()
    size = 0
    for block in iter(lambda: stream.read(CHUNK), b""):
        digest.update(block)
        size += len(block)
    return dict(bytes=size, sha256=digest.hexdigest())


def file_fingerprint(path):
    with path.open("rb") as stream:
        return fingerprint(stream)


def install_without_overwrite(temporary, target):
    """Create atomically; existing identical files are left untouched."""
    try:
        os.link(temporary, target)
    except FileExistsError:
        if file_fingerprint(temporary) != file_fingerprint(target):
            raise FileExistsError(f"Refusing to overwrite different existing file: {target}")


def pack():
    rows = []
    for name in NAMES:
        original = HERE / name
        archive = original.with_suffix(original.suffix + ".gz")
        original_info = file_fingerprint(original)
        with tempfile.NamedTemporaryFile(dir=HERE, prefix=name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            try:
                with gzip.GzipFile(filename="", fileobj=handle, mode="wb", compresslevel=9, mtime=0) as compressed:
                    with original.open("rb") as stream:
                        shutil.copyfileobj(stream, compressed, CHUNK)
                handle.flush()
                with gzip.open(temporary, "rb") as decompressed:
                    if fingerprint(decompressed) != original_info:
                        raise ValueError(f"Round-trip byte validation failed: {name}")
                if file_fingerprint(original) != original_info:
                    raise ValueError(f"Original changed while compressing: {name}")
                install_without_overwrite(temporary, archive)
            finally:
                temporary.unlink(missing_ok=True)
        rows.append(dict(original=name, original_bytes=original_info["bytes"],
                         original_sha256=original_info["sha256"], archive=archive.name,
                         archive_bytes=archive.stat().st_size,
                         archive_sha256=file_fingerprint(archive)["sha256"]))
    content = dict(schema_version=1, format="gzip", compression_level=9,
                   gzip_mtime=0, gzip_original_filename="", preserves_original_bytes=True,
                   originals_kept_local=True, artifacts=rows)
    payload = (json.dumps(content, indent=2, sort_keys=True) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=HERE, prefix="large_json_archives.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(payload)
            handle.flush()
            install_without_overwrite(temporary, MANIFEST)
        finally:
            temporary.unlink(missing_ok=True)
    return verify()


def verified_rows():
    manifest = json.loads(MANIFEST.read_text())
    if manifest["schema_version"] != 1 or manifest["format"] != "gzip":
        raise ValueError("Unsupported archive manifest")
    rows = manifest["artifacts"]
    if [row["original"] for row in rows] != list(NAMES):
        raise ValueError("Unexpected original file list")
    for row in rows:
        if row["archive"] != row["original"] + ".gz":
            raise ValueError("Unexpected archive path")
        archive = HERE / row["archive"]
        if file_fingerprint(archive) != dict(bytes=row["archive_bytes"], sha256=row["archive_sha256"]):
            raise ValueError(f"Archive SHA/size mismatch: {archive.name}")
        with gzip.open(archive, "rb") as stream:
            if fingerprint(stream) != dict(bytes=row["original_bytes"], sha256=row["original_sha256"]):
                raise ValueError(f"Decompressed bytes mismatch: {archive.name}")
        yield row


def verify():
    rows = list(verified_rows())
    for row in rows:
        original = HERE / row["original"]
        if original.exists() and file_fingerprint(original) != dict(bytes=row["original_bytes"], sha256=row["original_sha256"]):
            raise ValueError(f"Local original differs from published snapshot: {original}")
    return dict(status="verified", archives=len(rows),
                original_bytes=sum(r["original_bytes"] for r in rows),
                archive_bytes=sum(r["archive_bytes"] for r in rows))


def restore(output_dir):
    rows = list(verified_rows())
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        target = output_dir / row["original"]
        expected = dict(bytes=row["original_bytes"], sha256=row["original_sha256"])
        if target.exists():
            if file_fingerprint(target) != expected:
                raise FileExistsError(f"Refusing to overwrite different existing file: {target}")
            continue
        with tempfile.NamedTemporaryFile(dir=output_dir, prefix=target.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            try:
                with gzip.open(HERE / row["archive"], "rb") as stream:
                    shutil.copyfileobj(stream, handle, CHUNK)
                handle.flush()
                if file_fingerprint(temporary) != expected:
                    raise ValueError(f"Restored bytes mismatch: {target.name}")
                install_without_overwrite(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
    return dict(status="restored_or_already_identical", files=len(rows), output_dir=str(output_dir))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("pack", "verify", "restore"), default="verify")
    parser.add_argument("--output-dir", type=Path, help="Restore destination; defaults to this archive directory")
    args = parser.parse_args()
    if args.output_dir is not None and args.action != "restore":
        parser.error("--output-dir is only valid with restore")
    result = pack() if args.action == "pack" else restore(args.output_dir or HERE) if args.action == "restore" else verify()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
