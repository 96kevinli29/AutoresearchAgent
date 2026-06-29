---
name: verify-with-python
description: Use the Python tool (sympy/numpy) to compute, simplify, and stress-test claims with concrete values before finalizing.
---
# Verify with Python

When a step involves a computation or a universally-quantified claim, check it:

- **Compute / simplify** with sympy: `solve`, `simplify`, `factor`, `integrate`,
  `summation`, `limit`, `series`. Always `print(...)` the result.
- **Find counterexamples**: loop over small concrete cases with numpy/plain Python and
  print any case where the claim fails. A single counterexample refutes a universal claim.
- **Sanity-check final answers** numerically (e.g. evaluate both sides at random points).

Example:
<tool:python>
from sympy import symbols, solve, Eq
x = symbols('x')
print(solve(Eq(x**2 - 5*x + 6, 0), x))
</tool>

Trust tool output over intuition. If the tool disagrees with your draft, the draft is wrong.
