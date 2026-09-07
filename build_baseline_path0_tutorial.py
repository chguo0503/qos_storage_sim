#!/usr/bin/env python3
"""Validate and build the revised Chinese Path0 PDF, without rerunning the search."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

from baseline_path0_layout import DATA, DOCS, RESULT, ROOT

STEM = "baseline_path0_low_utilization_tutorial_v3_2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULT / f"{STEM}.pdf")
    args = parser.parse_args()
    for program in ("pandoc", "xelatex"):
        if shutil.which(program) is None:
            parser.error(f"Required program not found: {program}")
    if not (DATA / "tutorial_sensitivity_results.json").exists():
        parser.error("Run python run_baseline_path0_tutorial_sensitivity.py first")
    subprocess.run(
        [sys.executable, str(ROOT / "validate_baseline_path0_tutorial.py"),
         "--output", str(DATA / "tutorial_numeric_audit.json")], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / "make_baseline_path0_tutorial_assets.py")],
                   cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / "make_baseline_path0_beginner_assets.py")],
                   cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / "make_baseline_path0_read_event_assets.py")],
                   cwd=ROOT, check=True)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".building.pdf")
    subprocess.run(
        ["pandoc", str(DOCS / f"{STEM}.md"), "--standalone", "--number-sections",
         "--pdf-engine=xelatex", "--resource-path", str(DOCS), "--output", str(temporary)],
        cwd=ROOT, check=True)
    temporary.replace(output)
    print(f"Built {output}")


if __name__ == "__main__":
    main()
