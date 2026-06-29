"""Lean verification tool (optional capability).

M1 ships a stub that reports Lean is unavailable. M3 wires this to a Lean REPL
(leanprover-community/repl) and checks that the proof compiles with no `sorry`.
The interface is fixed now so the agent/benchmark code does not change later.
"""

from __future__ import annotations

import shutil


def lean_available() -> bool:
    return shutil.which("lake") is not None or shutil.which("lean") is not None


def run_lean(code: str, timeout: float = 120.0) -> str:
    """Check a Lean snippet. Returns a verdict string the agent can read."""
    if not lean_available():
        return (
            "[lean] UNAVAILABLE — Lean toolchain not installed. "
            "Install via elan + mathlib (milestone M3) to enable formal verification."
        )
    # M3: send `code` to the Lean REPL, parse goals/errors, fail on `sorry`.
    return "[lean] NOT_IMPLEMENTED — REPL wiring lands in M3."
