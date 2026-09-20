"""Read-only status and artifact sync for the uncapped OD ablation."""
from pathlib import Path
import json
import shlex
import subprocess
import sys
import time

LOCAL = Path(__file__).resolve().parents[1]
REMOTE = "/tmp/qos_scheduled_interp_20260920/results/od_underload_mechanisms_20260919/scheduled_input"
MUX = "/tmp/qos_qdepth_live_20260919_mux"
HOST = "chguo@192.168.31.126"
NAME = "abb_interp_unique_50"
SSH = ["ssh", "-S", MUX, "-o", "BatchMode=yes", HOST]
suffix = NAME + "_od_depth_none"
copied = False
for _ in range(1800):
    try:
        code = ("import json;from pathlib import Path;p=Path(" + repr(REMOTE)
                + ");f=p/'formal'/" + repr(suffix)
                + "/'command.json';print(f.read_text() if f.exists() else '{}')")
        command = json.loads(subprocess.check_output(SSH + ["python3 -c " + shlex.quote(code)], text=True))
        (LOCAL / "deployment/depth_none_status.json").write_text(json.dumps(command, indent=2) + "\n")
        if command.get("status") == "complete" and not copied:
            subprocess.run(["scp", "-q", "-r", "-o", "ControlPath=" + MUX,
                            HOST + ":" + REMOTE + "/formal/" + suffix,
                            str(LOCAL / "formal" / suffix)], check=True)
            copied = True
            print("SYNCED complete uncapped OD", flush=True)
        capped_command = LOCAL / "formal" / (NAME + "_od_baseline") / "command.json"
        if copied and capped_command.exists() and json.loads(capped_command.read_text()).get("status") == "complete":
            result = subprocess.run([sys.executable, str(LOCAL / "compare_depth.py"), "--name", NAME],
                                    capture_output=True, text=True)
            (LOCAL / "deployment/depth_compare.log").write_text(result.stdout + result.stderr)
            result.check_returncode()
            print("COMPARED both completed OD depths", flush=True)
            break
    except Exception as error:
        print(type(error).__name__ + ": " + str(error), flush=True)
    time.sleep(30)
