"""Execute the Colab notebook locally (in the MODE set in the notebook) in an isolated work directory."""
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "notebooks/NashGAT_MARL_Research.ipynb"
work = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "nb_work").resolve()
work.mkdir(parents=True, exist_ok=True)
os.environ["NASHGAT_WORK"] = str(work)
os.environ["NASHGAT_DATA_ROOT"] = str(ROOT)
nb = nbformat.read(src, 4)
t0 = time.time()
client = NotebookClient(nb, timeout=7200, kernel_name="python3", resources={"metadata": {"path": str(work)}})
try:
    client.execute()
    status = "OK"
except Exception as e:  # keep the partially executed notebook for inspection
    status = f"FAILED: {type(e).__name__}: {str(e)[:2000]}"
out = ROOT / "notebooks/NashGAT_MARL_Research.executed_local.ipynb"
nbformat.write(nb, out)
print(status, f"{time.time() - t0:.0f}s ->", out)
