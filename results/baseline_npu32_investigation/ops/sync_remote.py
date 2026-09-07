#!/usr/bin/env python3
"""Collect this study and atomically install completed offloaded jobs only.

Requires the study's pre-authenticated SSH control socket. No credentials stored.
Existing remote job directories are never replaced, including running jobs.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess

STUDY = Path(__file__).resolve().parents[1]
HOST = "192.168.31.126"
SOCKET = "/tmp/qos-study-20260907-ssh"
REMOTE = "/home/chguo/qos-baseline-study-20260907/results"
SSH = ["ssh", "-S", SOCKET, HOST]


def remote_python(code):
    return subprocess.check_output(SSH + ["python3 -c " + shlex.quote(code)], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pull", action="store_true")
    parser.add_argument("--push-offload", action="store_true")
    args = parser.parse_args()
    if args.push_offload:
        status = json.loads((STUDY / "formal/offload_status.json").read_text())
        for row in status["rows"]:
            if row["status"] not in ("complete", "existing"):
                continue
            relative = row["input"] + "/" + row["variant"]
            dest = REMOTE + "/formal/runs/" + relative
            if remote_python("import os; print(int(os.path.exists(" + repr(dest) + "))) ") == "1":
                continue
            src = STUDY / "formal/runs" / relative
            assert len(list(src.glob("*.json.gz"))) == 1
            staging = REMOTE + "/offload_transfer/" + relative
            subprocess.run(SSH + ["mkdir -p " + shlex.quote(staging)], check=True)
            subprocess.run(["rsync", "-a", "-e", "ssh -S " + SOCKET,
                            str(src) + "/", HOST + ":" + staging + "/"], check=True)
            # Linux RENAME_NOREPLACE makes the directory installation atomic:
            # either the whole completed job appears before remote execution,
            # or an already-started remote job wins and is left untouched.
            code = ("import os,ctypes; s=" + repr(staging) + "; d=" + repr(dest)
                    + "; os.makedirs(os.path.dirname(d),exist_ok=True); "
                    "c=ctypes.CDLL(None,use_errno=True); "
                    "r=c.renameat2(-100,s.encode(),-100,d.encode(),1); "
                    "print('installed' if r==0 else 'kept_remote_errno_'+str(ctypes.get_errno()))")
            outcome = remote_python(code)
            event = {"at_utc": datetime.now(timezone.utc).isoformat(),
                     "job": relative, "outcome": outcome}
            print(json.dumps(event), flush=True)
            with (STUDY / "ops/offload_transfers.jsonl").open("a") as log:
                log.write(json.dumps(event) + "\n")
    if args.pull:
        for stage in ("screen", "formal"):
            subprocess.run(["rsync", "-a", "--exclude=*.tmp", "-e", "ssh -S " + SOCKET,
                            HOST + ":" + REMOTE + "/" + stage + "/",
                            str(STUDY / stage) + "/"], check=True)
            state = json.loads((STUDY / stage / (stage + "_status.json")).read_text())
            print(json.dumps({"stage": stage, **{key: state[key] for key in
                             ("at_utc", "completed", "total", "failures")}}), flush=True)


if __name__ == "__main__":
    main()
