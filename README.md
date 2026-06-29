# mathagent

An end-to-end **math problem-solving / proof agent**, built on the
[A-Evolve](https://github.com/A-EVO-Lab/a-evolve) agentic-evolution framework.

- **Brain:** any LLM via **LiteLLM** (Claude / OpenAI / DeepSeek / self-hosted vLLM).
- **Optional verification capabilities:** Python (sympy/numpy) computation, and Lean 4
  formal verification — each toggleable.
- **Output:** LaTeX (`.tex`/`.pdf`) and Word (`.docx`).
- **Self-improving:** the agent's prompt/skills/memory live in an evolvable `workspace/`
  that A-Evolve mutates against a math benchmark, keeping only changes that help (git-tracked).

## Layout

```
mathagent/            python package
  llm/                LiteLLMProvider (drop-in for A-Evolve's LLMProvider) + MockProvider
  agent/              MathAgent(BaseAgent): provider-agnostic tool loop
  tools/              python_exec, lean_verify (M3), doc_export, registry (capability flags)
  benchmarks/         MathBenchmark(BenchmarkAdapter): answer / sympy / proof checks
  engine/             MathEvolutionEngine (M2)
  cli.py              `mathagent solve|eval`
workspace/            EVOLVABLE: manifest, prompts/system.md, skills/*/SKILL.md, tools, memory
data/seed_problems.jsonl
```

## Quickstart

```bash
# 1. env (already bootstrapped with uv)
source .venv/bin/activate
uv pip install -e .

# 2. smoke test (no API key needed)
mathagent solve "Compute 2+2." --provider mock --format pdf

# 3. real run
cp .env.example .env   # set MATHAGENT_MODEL + API key
mathagent solve "Find all real roots of x^2-5x+6=0." --format pdf
mathagent eval data/seed_problems.jsonl

# 4. self-improve the workspace (verifier-grounded evolution, git-tracked)
mathagent evolve data/seed_problems.jsonl --cycles 5            # LLM-driven mutations
mathagent evolve data/seed_problems.jsonl --rule --cycles 4     # deterministic fallback
.venv/bin/python scripts/evolve_demo.py                         # no-API-key mechanics self-test

# 5. web UI / API (same core as the CLI) — "online use"
mathagent serve --port 8000                  # real provider (set .env)
mathagent serve --provider mock --port 8000  # no-key demo
#   GET /              chat UI (MathJax)
#   POST /api/solve    {problem, format, model?, enable_python?, enable_lean?}
#   GET /files/<name>  generated .tex/.pdf/.docx
```

## Evolution (M2)

`MathEvolutionEngine` turns verifier evidence (wrong/missing boxed answers, Python
tracebacks, later Lean errors) into a new reusable **skill**, then **self-gates**: it
re-scores a held-out split and keeps the change only on strict improvement, reverting
otherwise. The loop git-tags every cycle (`evo-N`) for a full audit trail. The demo
self-test lifts a stub agent from score **0.0 → 1.0** by evolving one skill.

## Milestones

- **M0** env bootstrap ✅
- **M1** CLI MVP: solve → Python-verify → LaTeX/PDF ✅
- **M2** evolution loop (verifier-grounded, self-gating, git-tracked) ✅
- **M4** web frontend (FastAPI + MathJax chat UI) sharing the same core ✅
- **M3** Lean verification (elan + mathlib + Lean REPL)
- **M5** docx export + packaging for local & online install
