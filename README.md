# mathagent

An end-to-end **math problem-solving & proof agent**. Type a problem, watch the
solution stream in, and download it as LaTeX / PDF / Word — from a clean web UI
or the command line.

- 🧠 **Any model** via [LiteLLM](https://github.com/BerriAI/litellm) / OpenRouter (Claude, GPT, DeepSeek, …) — one API key.
- ⚡ **Live & transparent**: the answer streams token-by-token with a running **token + cost + time** meter.
- 🔎 **Optional verification** (off by default): Python (sympy) computation and Lean 4 proof checking.
- 📄 **Output**: LaTeX, PDF, and Word `.docx`.
- 🧬 **Self-improving**: built on the [A-Evolve](https://github.com/A-EVO-Lab/a-evolve) agent framework — the agent's prompts/skills/memory live in a git-tracked workspace it can evolve against a benchmark.

---

## Quick start (≈30 seconds)

You need **Python 3.10+** and one API key (e.g. [OpenRouter](https://openrouter.ai/keys)).

```bash
git clone https://github.com/96kevinli29/AutoresearchAgent.git
cd AutoresearchAgent
bash install.sh                       # sets up a venv and installs everything
nano .env                             # paste your API key (file was created for you)
.venv/bin/mathagent serve             # open http://127.0.0.1:8000
```

That's it — open the URL and ask a question.

<details>
<summary>Prefer pip (no clone)?</summary>

```bash
pip install "git+https://github.com/96kevinli29/AutoresearchAgent.git"
export OPENROUTER_API_KEY=sk-or-...
export MATHAGENT_MODEL=openrouter/anthropic/claude-sonnet-4.6
mathagent serve
```
</details>

### `.env`
```ini
OPENROUTER_API_KEY=sk-or-...
MATHAGENT_MODEL=openrouter/anthropic/claude-sonnet-4.6
```
Other models: `openrouter/anthropic/claude-opus-4.8`, `openrouter/openai/gpt-5.5`,
`openrouter/deepseek/deepseek-v4-pro`, or any LiteLLM id. You can also switch model in the UI.

---

## Use it

**Web** — `mathagent serve` then open the URL. Enter a problem, pick a model, click **Solve**.
The solution streams in; download PDF/LaTeX/docx; switch to the **Process** tab to see every step.

**Command line**
```bash
mathagent solve "Find all real roots of x^2-5x+6=0." --format pdf
mathagent eval data/seed_problems.jsonl          # score a problem set
mathagent evolve data/seed_problems.jsonl        # self-improve the workspace
```

**Optional verification** (off by default — plain model solve out of the box):
```bash
mathagent solve "…" --python        # let the agent verify with sympy
mathagent solve "…" --lean          # formal check (run `mathagent install-lean` first)
```
In the web UI, tick the **Python** / **Lean** boxes. When enabled, the agent decides
on its own when to call them.

---

## How it works

It's a real agent (not a prompt wrapper), built on A-Evolve:

```
problem → MathAgent loop:  reason → (optional tool: Python/Lean) → observe → finalize
          reads its evolvable workspace (prompts · skills · memory)
          → output: LaTeX / PDF / docx
evolution engine: improves the workspace from verifier feedback (git-tracked)
```

| Layer | File | A-Evolve contract |
|---|---|---|
| Brain (any model) | `mathagent/llm/litellm_provider.py` | `LLMProvider` |
| Agent loop | `mathagent/agent/math_agent.py` | `BaseAgent` |
| Scoring | `mathagent/benchmarks/math_bench.py` | `BenchmarkAdapter` |
| Self-improvement | `mathagent/engine/math_evolution.py` | `EvolutionEngine` |
| Evolvable state | `workspace/` (prompts, skills, memory) | workspace contract |

---

## Deploy online

It's a standard FastAPI app (`mathagent.web.server:app`):
```bash
mathagent serve --host 0.0.0.0 --port 8000          # VM
# or: uvicorn mathagent.web.server:app --host 0.0.0.0 --port $PORT   # PaaS
```
Set your API key as an env var / secret. For public access add auth and keep
`--python` off (it runs model-written code).

On an HPC login node, reach the UI from your laptop with an SSH tunnel:
`ssh -L 8000:127.0.0.1:8000 <you@cluster>` then open `http://localhost:8000`.

---

## Notes
- PDF output needs a LaTeX install (`pdflatex`); `.docx` uses a bundled pandoc (no setup); `.tex` always works.
- Math renders with a vendored KaTeX (works offline).
- The first run creates `./workspace` from the bundled seed; edit those files to change the agent's behavior.
