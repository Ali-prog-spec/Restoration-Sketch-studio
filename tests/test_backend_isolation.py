"""The backend container ships without PyTorch: importing the app must not import torch."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_backend_does_not_import_torch():
    code = ("import sys; sys.modules['torch'] = None; "  # any 'import torch' now raises ImportError
            "import backend.app.main as m; print('ok', len(m.app.routes))")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("ok")
