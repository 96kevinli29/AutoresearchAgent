"""Tool registry with capability flags.

Each verification capability (python, lean) is opt-in. The agent calls
``registry.run(name, body)`` from its text tool-protocol; unknown/disabled tools
return a readable error the model can react to.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .lean_verify import lean_available, run_lean
from .python_exec import run_python


@dataclass
class ToolRegistry:
    enable_python: bool = False
    enable_lean: bool = False
    python_timeout: float = 15.0
    lean_timeout: float = 120.0
    _log: list[dict] = field(default_factory=list)

    def descriptions(self) -> list[str]:
        """Human-readable lines describing enabled tools, for the system prompt."""
        lines = []
        if self.enable_python:
            lines.append(
                "<tool:python> ... </tool>  — run Python (sympy/numpy preimported); "
                "print() results to compute or check claims."
            )
        if self.enable_lean:
            lines.append(
                "<tool:lean> ... </tool>  — verify a Lean 4 statement/proof; "
                "a proof is accepted only if it compiles with no `sorry`."
            )
        return lines

    def run(self, name: str, body: str) -> str:
        name = name.strip().lower()
        if name == "python":
            if not self.enable_python:
                return "[python] disabled (enable with --python)"
            out = run_python(body, timeout=self.python_timeout)
        elif name == "lean":
            if not self.enable_lean:
                return "[lean] disabled (enable with --lean)"
            out = run_lean(body, timeout=self.lean_timeout)
        else:
            return f"[tool] unknown tool '{name}'. Available: python, lean."
        self._log.append({"tool": name, "input": body, "output": out})
        return out

    @property
    def calls(self) -> list[dict]:
        return self._log


def default_registry(python: bool = False, lean: bool = False) -> ToolRegistry:
    if lean and not lean_available():
        # keep the flag but the tool itself will report unavailability
        pass
    return ToolRegistry(enable_python=python, enable_lean=lean)
