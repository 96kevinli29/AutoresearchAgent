"""MathBenchmark — an A-Evolve `BenchmarkAdapter` for math tasks.

Task file format (JSONL), one object per line::

    {"id": "alg-1", "problem": "Find all real solutions of x^2-5x+6=0.",
     "answer": "2, 3", "check": "answer"}

``check`` is one of:
  - "answer": extract \\boxed{...} (or last "Answer:" line) and compare to ``answer``
  - "sympy":  ``answer`` is a python expr; success if sympy says pred == answer
  - "proof":  no closed-form answer; scored as 0/1 by an LLM/Lean grader (M2/M3).
              For now a proof task is "success" iff a <final> with non-trivial length exists.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from agent_evolve import BenchmarkAdapter, Feedback, Task, Trajectory

_BOXED_RE = re.compile(r"\\boxed\{")
_ANSWER_LINE_RE = re.compile(r"Answer:\s*\$?\\?boxed?\{?([^$\n}]+)", re.IGNORECASE)


def extract_boxed(text: str) -> str | None:
    """Extract the content of the last \\boxed{...}, brace-balanced."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        m = _ANSWER_LINE_RE.search(text)
        return m.group(1).strip() if m else None
    i = idx + len("\\boxed{")
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(c)
        i += 1
    return "".join(out).strip()


_LATEX_SPACES = ["\\left", "\\right", "\\,", "\\!", "\\:", "\\;", "\\quad", "\\qquad", "\\ "]
_FRAC_RE = re.compile(r"\\frac\{([^{}]*)\}\{([^{}]*)\}")


def clean_latex(s: str | None) -> str:
    """Turn a LaTeX answer fragment into a plain math expression string."""
    if s is None:
        return ""
    s = s.strip().strip("$").strip()
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    for _ in range(5):  # \frac{a}{b} -> ((a)/(b)), repeat for stacked fractions
        s2 = _FRAC_RE.sub(r"((\1)/(\2))", s)
        if s2 == s:
            break
        s = s2
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    for t in _LATEX_SPACES:
        s = s.replace(t, "")
    s = s.replace("^", "**").replace(" ", "")
    return s


def normalize(s: str | None) -> str:
    return clean_latex(s).lower().rstrip(".")


def _to_expr(s: str):
    import sympy as sp

    return sp.sympify(clean_latex(s))


def _expr_equal(a: str, b: str) -> bool:
    try:
        import sympy as sp

        return bool(sp.simplify(_to_expr(a) - _to_expr(b)) == 0)
    except Exception:
        return False


def _multiset_equal(pred: str, gold: str) -> bool:
    """Compare comma/semicolon-separated answer lists as multisets (order-free)."""
    pp = [p for p in re.split(r"[,;]", pred) if p.strip()]
    gg = [p for p in re.split(r"[,;]", gold) if p.strip()]
    if len(pp) != len(gg) or len(pp) < 2:
        return False
    used = [False] * len(pp)
    for g in gg:
        for i, p in enumerate(pp):
            if not used[i] and (normalize(p) == normalize(g) or _expr_equal(p, g)):
                used[i] = True
                break
        else:
            return False
    return True


def answers_match(pred: str | None, gold: str, check: str) -> bool:
    if pred is None:
        return False
    if normalize(pred) == normalize(gold):
        return True
    if _multiset_equal(pred, gold):
        return True
    return _expr_equal(pred, gold)


class MathBenchmark(BenchmarkAdapter):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._rows = [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def get_tasks(self, split: str = "train", limit: int | None = None) -> list[Task]:
        rows = self._rows
        # simple deterministic split: every 5th item is holdout ("test")
        if split == "test":
            rows = [r for i, r in enumerate(rows) if i % 5 == 0]
        elif split == "train":
            rows = [r for i, r in enumerate(rows) if i % 5 != 0]
        if limit:
            rows = rows[:limit]
        return [Task(id=r["id"], input=r["problem"], metadata=r) for r in rows]

    def evaluate(self, task: Task, trajectory: Trajectory) -> Feedback:
        meta = task.metadata
        check = meta.get("check", "answer")
        pred_raw = trajectory.output or ""
        pred = extract_boxed(pred_raw)

        if check == "proof":
            ok = bool(pred_raw) and len(pred_raw) > 80
            return Feedback(
                success=ok,
                score=1.0 if ok else 0.0,
                detail=json.dumps({"check": "proof", "len": len(pred_raw)}),
            )

        gold = str(meta.get("answer", ""))
        ok = answers_match(pred, gold, check)
        detail = {"check": check, "pred": pred, "gold": gold}
        return Feedback(success=ok, score=1.0 if ok else 0.0, detail=json.dumps(detail))
