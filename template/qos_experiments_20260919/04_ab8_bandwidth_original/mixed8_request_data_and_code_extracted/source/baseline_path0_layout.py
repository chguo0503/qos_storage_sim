"""Output locations: final PDFs at the top level, supporting files below."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "results" / "baseline_4npu_ssu1_low_utilization"
DATA = RESULT / "data"
FIGURES = RESULT / "figures"
DOCS = RESULT / "docs"
