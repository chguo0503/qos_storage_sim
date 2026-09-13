#!/usr/bin/env python3
"""Copy only completed remote cases and verify byte hashes before local audit."""
from pathlib import Path
import argparse
import hashlib
import json
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4*2**20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--remote-root', default='/home/chguo/qos_ab128_32_20260912/results/baseline_ab128_32_ratio12_20260912')
    p.add_argument('--host', default='192.168.31.126')
    p.add_argument('--socket', default='/tmp/qos-ab128-32-20260912.sock')
    p.add_argument('--stage', default='screen')
    p.add_argument('--audit', action='store_true')
    args = p.parse_args()
    code = ('from pathlib import Path; import json; root=Path('+repr(args.remote_root+'/'+args.stage)+'); '
            'rows=[json.loads(p.read_text()) for p in root.glob("runs/*/baseline/command.json")]; '
            'print(json.dumps([r for r in rows if r["status"]=="complete"]))')
    run = subprocess.run(['ssh', '-S', args.socket, args.host, 'python3 -c '+shlex.quote(code)],
                         check=True, capture_output=True, text=True)
    rows = json.loads(run.stdout)
    for row in rows:
        directory = HERE / args.stage / 'runs' / row['label'] / 'baseline'
        directory.mkdir(parents=True, exist_ok=True)
        files = [('manifest.json.gz', 'case_manifest_sha256'), ('result.json.gz', 'output_sha256')]
        if row.get('trace'):
            files.append(('trace.json.gz', 'trace_sha256'))
        remote_dir = args.remote_root+'/'+args.stage+'/runs/'+row['label']+'/baseline/'
        for filename, key in files:
            path = directory / filename
            if path.exists() and sha(path) == row[key]:
                continue
            tmp = path.with_name(path.name+'.download')
            subprocess.run(['scp', '-o', 'ControlPath='+args.socket,
                            args.host+':'+remote_dir+filename, str(tmp)], check=True,
                           stdout=subprocess.DEVNULL)
            assert sha(tmp) == row[key], (filename, 'download hash mismatch')
            tmp.replace(path)
        (directory/'command.json').write_text(json.dumps(row, indent=2)+'\n')
        audit = directory/'audit'
        audit_path = audit/'audit.json'
        if args.audit and (not audit_path.exists() or
                           json.loads(audit_path.read_text())['source']['analyzer_sha256'] != sha(HERE/'audit_results.py')):
            command = [sys.executable, '-B', str(HERE/'audit_results.py'),
                '--manifest', str(directory/'manifest.json.gz'), '--result', str(directory/'result.json.gz'),
                '--out', str(audit), '--window', '2000:6000']
            for window in row['windows']:
                command += ['--window', f'{window["start_ms"]}:{window["end_ms"]}']
            run = subprocess.run(command, capture_output=True, text=True)
            (directory/'audit.log').write_text(run.stdout+run.stderr)
            if run.returncode:
                print(run.stdout, run.stderr)
                raise RuntimeError('Audit failed: '+row['label'])
        print(json.dumps(dict(label=row['label'], verified=True, wall_seconds=row['wall_seconds'],
                              windows=row['windows'])), flush=True)
    print(json.dumps(dict(completed_remote_cases=len(rows))), flush=True)


if __name__ == '__main__':
    main()
