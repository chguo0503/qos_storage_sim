#!/usr/bin/env python3
"""Create an isolated source-only remote scratch; never launch a simulation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_PYTHON = '/tmp/qos_random_near_capacity_20260914_z34z48vi/venv/bin/python'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--host', default='chguo@192.168.31.126')
    ap.add_argument('--mux', default='/tmp/qos_od_baseline_20260918_mux')
    ap.add_argument('--python', default=DEFAULT_PYTHON)
    args = ap.parse_args()
    record_path = HERE/'remote_environment.json'
    assert not record_path.exists(), 'Preserve existing remote scratch; create a new named record manually if required'
    paths = sorted(ROOT.glob('*.py')) + [ROOT/'data']
    paths += [HERE/name for name in ('run_trial.py','metrics.py','launch_detached.py')]
    hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    archive = HERE/'remote_sources.tar.gz'
    with tarfile.open(archive,'x:gz') as tar:
        for p in paths:
            tar.add(p,arcname=str(p.relative_to(ROOT)))
    ssh = ['ssh','-S',args.mux,'-o','BatchMode=yes',args.host]
    def remote(code):
        result = subprocess.run(ssh+[shlex.join([args.python,'-c',code])],check=True,capture_output=True,text=True)
        return json.loads(result.stdout)
    scratch = remote("import tempfile,json; print(json.dumps(tempfile.mkdtemp(prefix='qos_od_mixed_underload_search_20260919_')))" )
    subprocess.run(['scp','-o',f'ControlPath={args.mux}','-o','BatchMode=yes',str(archive),args.host+':'+scratch+'/sources.tar.gz'],check=True)
    code = ('import tarfile,json,hashlib,sys,os; from pathlib import Path; '
            f"root=Path({scratch!r}); expected={hashes!r}; "
            "tarfile.open(root/'sources.tar.gz').extractall(root); "
            "actual={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in expected}; "
            "assert actual==expected; import numpy; "
            "print(json.dumps(dict(scratch=str(root),python=sys.executable,python_version=sys.version,"
            "numpy_version=numpy.__version__,cpu_count=os.cpu_count(),source_hashes_verified=True)))")
    info = remote(code)
    relative = HERE.relative_to(ROOT)
    probe = subprocess.run(ssh+[shlex.join([args.python,str(Path(scratch)/relative/'run_trial.py'),'--help'])],
                           check=True,capture_output=True,text=True)
    assert '--manifest' in probe.stdout and '--validate-only' in probe.stdout
    info.update(host=args.host,mux=args.mux,source_sha256=hashes,
                created_utc=datetime.now(timezone.utc).isoformat(),runner_import_check=True,
                launcher=str(Path(scratch)/relative/'launch_detached.py'),
                archive=str(archive.relative_to(ROOT)),simulations_started=False)
    with record_path.open('x') as f:
        json.dump(info,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in info.items() if k!='source_sha256'}),flush=True)


if __name__ == '__main__':
    main()
