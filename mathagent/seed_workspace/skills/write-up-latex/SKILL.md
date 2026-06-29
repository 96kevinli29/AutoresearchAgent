---
name: write-up-latex
description: Render the final solution as clean, compilable LaTeX with proper math mode and a boxed answer.
---
# Write up in LaTeX

The `<final>...</final>` block becomes a LaTeX document body, so write valid LaTeX:

- Inline math `$...$`, display math `\[ ... \]` or `align*` for multi-line derivations.
- Use `\boxed{...}` for the final answer and end with `Answer: $\boxed{...}$`.
- For proofs, you may use the `theorem`/`proof` environments (amsthm is loaded).
- Keep each logical step on its own line; reference the plan's sub-goals.
- Do **not** include `\documentclass`, `\begin{document}`, or the preamble — only the body.

Self-contained and compilable beats clever. Avoid undefined macros.
