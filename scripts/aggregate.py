"""Aggregate results/ into tables and results/figures/ (see nashgat/report.py)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nashgat import report  # noqa: E402

if __name__ == "__main__":
    report.configure(ROOT / "results", ROOT / "results/figures", ROOT)
    report.main()
