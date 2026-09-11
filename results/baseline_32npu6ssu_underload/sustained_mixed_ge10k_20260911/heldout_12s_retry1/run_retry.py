#!/usr/bin/env python3
"""One authorized infrastructure retry; only subprocess session isolation is changed."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys
import time

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'run_candidates.py'
EXPECTED=json.loads((HERE/'runner_snapshot.json').read_text())['sha256']
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==EXPECTED
spec=importlib.util.spec_from_file_location('frozen_common_candidates',SOURCE)
common=importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)

# This module owns its Python interpreter. No shared/core source is patched.
original_popen=common.subprocess.Popen
def isolated_popen(*args,**kwargs):
    assert 'start_new_session' not in kwargs
    kwargs['start_new_session']=True
    return original_popen(*args,**kwargs)
common.subprocess.Popen=isolated_popen

rec=dict(status='running',pid=os.getpid(),session_id=os.getsid(0),started_unix=time.time(),
         common_runner_sha256=EXPECTED,orchestrator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         scope='Exactly two baseline jobs, same profiles/queues/windows/seeds as interrupted attempt; start_new_session=True for each simulator child.')
common.write_json(HERE/'driver_lifecycle.json',rec)
sys.argv=[str(SOURCE),'--spec',str(HERE/'spec.json'),'--out',str(HERE),'--run','--workers','2']
try:
    common.main()
except BaseException as exc:
    rec.update(status='failed',error=repr(exc),ended_unix=time.time())
    common.write_json(HERE/'driver_lifecycle.json',rec)
    raise
else:
    rec.update(status='complete',returncode=0,ended_unix=time.time())
    common.write_json(HERE/'driver_lifecycle.json',rec)
