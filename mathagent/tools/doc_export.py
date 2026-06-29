"""Export a solution to LaTeX (.tex / .pdf) and optionally Word (.docx)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

_TEMPLATE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{enumitem}
\newtheorem{theorem}{Theorem}
\newtheorem{lemma}{Lemma}
\title{%(title)s}
\date{}
\begin{document}
\maketitle
%(body)s
\end{document}
"""


def export(
    body_latex: str,
    out_dir: str | Path,
    name: str = "solution",
    title: str = "Solution",
    fmt: str = "latex",
) -> dict[str, str]:
    """Write the solution. ``fmt`` in {latex, pdf, docx}. Returns paths produced."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    produced: dict[str, str] = {}

    tex = out_dir / f"{name}.tex"
    tex.write_text(_TEMPLATE % {"title": title, "body": body_latex}, encoding="utf-8")
    produced["tex"] = str(tex)

    if fmt in ("pdf",):
        if shutil.which("pdflatex"):
            for _ in range(2):  # twice to resolve refs
                subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
                    cwd=out_dir,
                    capture_output=True,
                    text=True,
                )
            pdf = out_dir / f"{name}.pdf"
            if pdf.exists():
                produced["pdf"] = str(pdf)
        else:
            produced["warning"] = "pdflatex not found; produced .tex only"

    if fmt in ("docx",):
        try:
            import pypandoc  # provided by the [docx] extra (pypandoc-binary)

            docx = out_dir / f"{name}.docx"
            pypandoc.convert_file(str(tex), "docx", outputfile=str(docx))
            produced["docx"] = str(docx)
        except Exception as e:  # pragma: no cover - optional dependency
            produced["warning"] = f"docx export unavailable: {e}. Install extra: pip install '.[docx]'"

    return produced
