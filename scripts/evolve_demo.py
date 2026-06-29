"""Deterministic self-test of the evolution mechanics — no API key needed.

A stub "brain" emits the correct word but only *boxes* it (so the benchmark can
extract the answer) once a `final-answer-discipline` skill has been evolved into
the workspace. So baseline scores 0, and a kept mutation should lift it to 1.0 —
exercising the full Solve -> evaluate -> failure-analysis -> mutate -> gate(holdout)
-> keep -> git-tag -> reload loop.

Run:  .venv/bin/python scripts/evolve_demo.py
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from agent_evolve import EvolveConfig, Evolver
from agent_evolve.llm.base import LLMProvider, LLMResponse

from mathagent.agent import MathAgent
from mathagent.benchmarks import MathBenchmark
from mathagent.engine import MathEvolutionEngine
from mathagent.tools import default_registry

ROOT = Path(__file__).resolve().parent.parent


class WorkspaceAwareStub(LLMProvider):
    """Boxes its answer only after `final-answer-discipline` is in the workspace."""

    def complete(self, messages, max_tokens=None, temperature=None, **kw):
        prompt = "\n".join(m.content for m in messages)
        m = re.search(r"word (\w+)", prompt)
        target = m.group(1) if m else "ANSWER"
        if "final-answer-discipline" in prompt:
            txt = f"<final>The requested word.\n\nAnswer: $\\boxed{{{target}}}$</final>"
        else:
            txt = f"<final>The answer is {target}.</final>"  # unboxed -> not extractable
        return LLMResponse(content=txt, usage={}, raw=None)

    def complete_with_tools(self, messages, tools, max_tokens=None, **kw):
        return self.complete(messages)


def main() -> int:
    work_root = ROOT / "evolution_workdir_demo"
    if work_root.exists():
        shutil.rmtree(work_root)
    work_ws = work_root / "workspace"
    shutil.copytree(ROOT / "workspace", work_ws)

    agent = MathAgent(work_ws, WorkspaceAwareStub(), tools=default_registry(python=True))
    bench = MathBenchmark(ROOT / "data" / "evolve_demo.jsonl")
    engine = MathEvolutionEngine(provider=None, holdout_limit=2)  # rule-based fallback
    config = EvolveConfig(batch_size=4, max_cycles=4, egl_window=2)

    evolver = Evolver(agent, bench, config=config, engine=engine, work_dir=str(work_root))
    result = evolver.run()

    print("\n===== DEMO RESULT =====")
    print("score history:", [round(s, 3) for s in result.score_history])
    print("converged:", result.converged)
    print("skills after:", [s.name for s in agent.workspace.list_skills()])
    mems = agent.workspace.read_all_memories(limit=20)
    evolved = [m.get("content", "") for m in mems if "[evolved]" in str(m.get("content", ""))]
    print("evolved memories:", evolved)

    hist = result.score_history
    assert hist and hist[0] == 0.0, f"expected baseline 0.0, got {hist}"
    assert max(hist) >= 1.0, f"expected a perfect cycle after evolution, got {hist}"
    assert any("final-answer-discipline" == s.name for s in agent.workspace.list_skills()), (
        "expected the evolved skill to be kept"
    )
    print("\nPASS: evolution lifted score from 0.0 to", max(hist))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
