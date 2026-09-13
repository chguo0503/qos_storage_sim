#!/usr/bin/env python3
"""Run a frozen manifest in a separate output base with the unchanged runner."""
from argparse import ArgumentParser,Namespace
from pathlib import Path
import sys
import experiment as base


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output-base',type=Path,required=True)
    parser.add_argument('--strategy',choices=['baseline','once'],required=True)
    parser.add_argument('--trace',action='store_true')
    args=parser.parse_args()
    manifest=args.manifest.resolve(); output=args.output_base.resolve()
    assert output.is_relative_to(Path(__file__).resolve().parent)
    requests,meta=base.load_manifest(manifest)
    target=output/'runs'/meta['label']/args.strategy
    assert not target.exists(),'Use a new output case; no overwrite.'
    output.mkdir(parents=True,exist_ok=True)
    base.write_json(output/(args.strategy+'_runner_provenance.json'),dict(
        argv=sys.argv,manifest_sha256=base.sha(manifest),wrapper_sha256=base.sha(__file__),
        runner_sha256=base.sha(base.__file__),core_source_sha256=base.source_hashes(),
        note='Only output base changes. Runtime constants, event logic and frozen input are unchanged.'))
    base.HERE=output
    base.execute(Namespace(strategy=args.strategy,trace=args.trace),requests,meta,manifest)


if __name__=='__main__':main()
