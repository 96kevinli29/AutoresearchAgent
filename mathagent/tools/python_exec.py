"""Python computation tool: run sympy/numpy snippets to compute & verify.

This is a *verification capability*, not a hardened sandbox. It runs the snippet
in a fresh subprocess with a wall-clock timeout. For untrusted/online use, swap
the subprocess for a container or a restricted runner (see M5).
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

_HEADER = textwrap.dedent(
    """
    import math, json
    import numpy as np
    import sympy as sp
    from sympy import *
    from sympy.abc import *  # predefine x, y, z, n, ... as symbols
    """
)


def run_python(code: str, timeout: float = 15.0) -> str:
    """Execute ``code`` with sympy/numpy preimported; return combined output."""
    script = _HEADER + "\n" + textwrap.dedent(code)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(script)
        path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"[python] TIMEOUT after {timeout}s"
    finally:
        Path(path).unlink(missing_ok=True)

    out = proc.stdout.strip()
    err = proc.stderr.strip()
    if proc.returncode != 0:
        return f"[python] exited {proc.returncode}\nSTDOUT:\n{out}\nSTDERR:\n{err}"
    return out if out else "[python] (no output — remember to print() results)"
