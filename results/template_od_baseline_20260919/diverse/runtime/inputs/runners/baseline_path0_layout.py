"""Output locations: final PDFs at the top level, supporting files below."""

# Allow this file to be invoked directly from any working directory.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "results" / "baseline_4npu_ssu1_low_utilization"
DATA = RESULT / "data"
FIGURES = RESULT / "figures"
DOCS = RESULT / "docs"
