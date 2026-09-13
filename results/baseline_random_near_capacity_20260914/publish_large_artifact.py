#!/usr/bin/env python3
"""Lossless byte chunks for a frozen artifact above GitHub's per-file limit."""
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os

CHUNK = 64*1024*1024


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def pack(source):
    assert source.is_file() and source.stat().st_size >= 100*1024*1024
    out = source.with_name(source.name+'.parts')
    out.mkdir(exist_ok=False)
    original_sha = sha(source)
    pieces = []
    with source.open('rb') as stream:
        index = 0
        for block in iter(lambda:stream.read(CHUNK),b''):
            name = f'{index:03d}.chunk'
            (out/name).write_bytes(block)
            pieces.append({'name':name,'bytes':len(block),'sha256':hashlib.sha256(block).hexdigest()})
            index += 1
    assert sum(p['bytes'] for p in pieces) == source.stat().st_size
    assert sha(source) == original_sha
    manifest = {'created_utc':datetime.now(timezone.utc).isoformat(),
                'original_name':source.name,'original_bytes':source.stat().st_size,
                'original_sha256':original_sha,'pieces':pieces,
                'generator_sha256':sha(Path(__file__)),
                'method':'Concatenate chunks in listed order to reproduce original compressed bytes exactly. No decompression, recompression, sampling, or scientific change.'}
    path = out/'manifest.json'
    path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (out/'README.md').write_text('# 原始trace的无损分片\n\n'
        '原始压缩trace超过GitHub单文件100 MiB限制，因此按原始字节切为每片最多64 MiB。没有丢弃数据，也没有重新压缩。\n\n'
        '从项目根目录运行本研究的 `publish_large_artifact.py --restore 本目录/manifest.json`，将在上一级恢复原名文件；按分片和整体SHA256逐项验证。已存在且SHA正确的原文件不会被覆盖。\n\n'
        '原始输入、结果、绘图证据和命令中的SHA保持不变。具体字节数、顺序与SHA见 [manifest.json](manifest.json)。\n',encoding='utf-8')
    return path


def restore(manifest_path, output=None):
    data = json.loads(manifest_path.read_text())
    output = output or manifest_path.parent.parent/data['original_name']
    assert Path(data['original_name']).name == data['original_name']
    original = hashlib.sha256()
    total = 0
    # Validate every chunk even when the original is already present.
    for piece in data['pieces']:
        assert Path(piece['name']).name == piece['name']
        file = manifest_path.parent/piece['name']
        assert file.stat().st_size == piece['bytes'] and sha(file) == piece['sha256']
        with file.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):
                original.update(block);total += len(block)
    assert total == data['original_bytes'] and original.hexdigest() == data['original_sha256']
    if output.exists():
        assert sha(output) == data['original_sha256'], 'Refuse to overwrite a different existing file.'
        return output
    temporary = output.with_name(output.name+f'.{os.getpid()}.tmp')
    with temporary.open('xb') as stream:
        for piece in data['pieces']:
            with (manifest_path.parent/piece['name']).open('rb') as source:
                for block in iter(lambda:source.read(1024*1024),b''):
                    stream.write(block)
    assert sha(temporary) == data['original_sha256']
    temporary.replace(output)
    return output


def main():
    parser = ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--pack', type=Path)
    group.add_argument('--restore', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.pack:
        assert args.output is None
        print(pack(args.pack.resolve()))
    else:
        print(restore(args.restore.resolve(),args.output.resolve() if args.output else None))


if __name__ == '__main__':
    main()
