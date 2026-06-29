"""Lean verification tool (optional capability).

M1 ships a stub that reports Lean is unavailable. M3 wires this to a Lean REPL
(leanprover-community/repl) and checks that the proof compiles with no `sorry`.
The interface is fixed now so the agent/benchmark code does not change later.
"""

from __future__ import annotations

import shutil


def lean_available() -> bool:
    return shutil.which("lake") is not None or shutil.which("lean") is not None


INSTALL_HINT = (
    "Lean is not installed. To enable formal verification, install it (one-time):\n"
    "  mathagent install-lean          # guided; add --run to execute\n"
    "or manually:\n"
    "  curl https://elan.lean-lang.org/elan-init.sh -sSf | sh -s -- -y\n"
    "  source $HOME/.elan/env && elan default leanprover/lean4:stable\n"
    "Mathlib (needed for nontrivial proofs) is a separate, heavy build — see README."
)


def run_lean(code: str, timeout: float = 120.0) -> str:
    """Check a Lean snippet. Returns a verdict string the agent can read."""
    if not lean_available():
        return "[lean] UNAVAILABLE — " + INSTALL_HINT
    # Lean REPL wiring (no-sorry compile check) is added when the user installs Lean.
    return (
        "[lean] toolchain found but the REPL bridge is not configured yet. "
        "Run `mathagent install-lean --run` to finish setup."
    )
