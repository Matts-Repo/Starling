"""starling must be importable and fully functional as a standalone package."""

import subprocess
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "starling"


def test_no_external_dfxm_imports_in_package():
    offenders = []
    for py in PKG.rglob("*.py"):
        src = py.read_text()
        if "import darling" in src or "from darling" in src:
            offenders.append(str(py))
    assert not offenders, f"unexpected external DFXM imports in: {offenders}"


def test_import_standalone():
    """starling works with no optional dependencies present."""
    code = """
import sys
sys.modules["darling"] = None  # raises ImportError on any 'import darling'
import numpy as np
import starling

data = (1000 * np.random.default_rng(0).random((16, 16, 20))).astype(np.uint16)
x = np.linspace(0.0, 1.0, 20)
out = starling.properties.fit_1D_gaussian(data, (x,), device="cpu")
assert out.shape == (16, 16, 6)
mean, cov = starling.properties.moments(data, np.array([x]), device="cpu")
assert mean.shape == (16, 16)
print("OK")
"""
    res = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert res.returncode == 0, res.stderr
    assert "OK" in res.stdout
