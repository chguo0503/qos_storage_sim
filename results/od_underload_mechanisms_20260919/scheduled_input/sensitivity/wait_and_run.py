"""Wait for a fully synced immutable manifest, then launch one local replay."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
NAME = 'abb_interp_unique_50'
STATUS = HERE / 'launcher_status.json'


def write(value):
    temporary = STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(STATUS)


def main():
    started = time.time()
    status = dict(status='waiting_for_complete_manifest', watcher_pid=os.getpid(),
                  cpu=8, seed=19, name=NAME, created_unix_s=started)
    manifest = HERE.parent / 'inputs' / f'{NAME}.json.gz'
    planning = HERE.parent / 'planning' / NAME / 'command.json'
    while True:
        ready = False
        try:
            if manifest.exists() and planning.exists():
                p = json.loads(planning.read_text())
                ready = (p.get('status') == 'complete' and p.get('conditions_passed') is True and
                         hashlib.sha256(manifest.read_bytes()).hexdigest() == p.get('manifest_sha256'))
        except (OSError, ValueError):
            pass  # A concurrent sync may not have atomically installed all files yet.
        status.update(updated_unix_s=time.time(), wait_wall_seconds=time.time() - started)
        write(status)
        if ready:
            break
        time.sleep(15)
    command = ['taskset', '-c', '8', sys.executable, str(HERE / 'run_seed_sensitivity.py'),
               '--name', NAME, '--seed', '19']
    with (HERE / 'seed19.log').open('xb') as stream:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                   stderr=subprocess.STDOUT, close_fds=True, start_new_session=True)
        status.update(status='running', pid=process.pid, command=command, run_start_unix_s=time.time())
        write(status)
        while process.poll() is None:
            status.update(updated_unix_s=time.time(), run_wall_seconds=time.time() - status['run_start_unix_s'])
            write(status)
            time.sleep(15)
        status.update(status='complete' if process.returncode == 0 else 'failed',
                      returncode=process.returncode, updated_unix_s=time.time(),
                      run_wall_seconds=time.time() - status['run_start_unix_s'])
        write(status)


if __name__ == '__main__':
    main()
