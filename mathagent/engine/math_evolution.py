"""MathEvolutionEngine — verifier-grounded evolution for the math agent.

Our value-add over vanilla A-Evolve: the mutation signal is *grounded in the
verifiers*. Instead of treating a failure as a bare 0/1, we extract structured
evidence — wrong/missing boxed answers, Python tracebacks, (later) Lean errors —
and turn it into a new reusable skill (or prompt/memory note). Each mutation is
**self-gated**: we re-score a held-out task split and keep the change only if it
does not regress, reverting our file writes otherwise. The loop's git layer tags
every kept mutation for a full audit trail.

Two proposal modes:
  - LLM-driven (``provider`` given): the evolver model writes a skill grounded in
    the failure report. This is the real optimizer.
  - Rule-based (no provider): deterministic fallback keyed on the dominant failure
    mode — lets the whole evolve→gate→keep loop run/CI without an API key.
"""

from __future__ import annotations

import json
import re

from agent_evolve import EvolutionEngine, StepResult
from agent_evolve.llm.base import LLMMessage


def _reload_agent(trial) -> None:
    """Make the trial's agent reflect the current workspace files."""
    agent = getattr(trial, "_agent", None)
    if agent is not None:
        agent.reload_from_fs()


class MathEvolutionEngine(EvolutionEngine):
    def __init__(
        self,
        provider=None,
        holdout_split: str = "test",
        holdout_limit: int = 4,
        min_gain: float = 0.0,
        max_skill_chars: int = 1600,
    ) -> None:
        self.provider = provider
        self.holdout_split = holdout_split
        self.holdout_limit = holdout_limit
        self.min_gain = min_gain
        self.max_skill_chars = max_skill_chars

    # ---- verifier-grounded failure analysis -------------------------------

    @staticmethod
    def _tool_errors(obs) -> list[str]:
        errs = []
        for s in obs.trajectory.steps:
            out = str(s.get("output", ""))
            if s.get("type") == "tool" and ("exited" in out or "TIMEOUT" in out or "Error" in out):
                errs.append(out[:300])
        return errs

    def _failure_report(self, fails) -> list[dict]:
        report = []
        for o in fails:
            report.append(
                {
                    "task_id": o.task.id,
                    "problem": o.task.input[:400],
                    "feedback": o.feedback.detail,
                    "answer_tail": (o.trajectory.output or "")[-300:],
                    "tool_errors": self._tool_errors(o),
                }
            )
        return report

    def _holdout_score(self, trial) -> float:
        tasks = trial.get_tasks(split=self.holdout_split, limit=self.holdout_limit)
        if not tasks:
            tasks = trial.get_tasks(split="train", limit=self.holdout_limit)
        obs = trial.run_tasks(tasks)
        if not obs:
            return 0.0
        return sum(o.feedback.score for o in obs) / len(obs)

    # ---- the evolution step ----------------------------------------------

    def step(self, workspace, observations, history, trial) -> StepResult:
        fails = [o for o in observations if not o.feedback.success]
        if not fails:
            return StepResult(mutated=False, summary="no failures this cycle")

        report = self._failure_report(fails)

        # snapshot the pieces we might touch, so we can revert on a failed gate
        prev_prompt = workspace.read_prompt()
        prev_skills = {s.name for s in workspace.list_skills()}

        before = self._holdout_score(trial)  # gate baseline

        mutation = self._propose(workspace, report)
        if not mutation:
            return StepResult(mutated=False, summary="no mutation proposed")

        _reload_agent(trial)  # agent must see the mutation for the gate
        after = self._holdout_score(trial)

        if after > before + self.min_gain:  # keep only on strict improvement
            workspace.add_memory(
                {
                    "content": f"[evolved] {mutation['summary']} | holdout {before:.2f}->{after:.2f}",
                    "evidence": report[:3],
                },
                category="episodic",
            )
            return StepResult(
                mutated=True,
                summary=mutation["summary"],
                metadata={"holdout_before": before, "holdout_after": after, **mutation.get("meta", {})},
            )

        # gate failed → revert our writes
        self._revert(workspace, prev_prompt, prev_skills)
        _reload_agent(trial)
        return StepResult(
            mutated=False,
            summary=f"reverted ({mutation['summary']}): holdout {before:.2f}->{after:.2f}",
        )

    def _revert(self, workspace, prev_prompt: str, prev_skills: set) -> None:
        workspace.write_prompt(prev_prompt)
        for s in workspace.list_skills():
            if s.name not in prev_skills:  # only delete skills we added this cycle
                try:
                    workspace.delete_skill(s.name)
                except Exception:
                    pass

    # ---- proposal: LLM-driven, with a deterministic fallback --------------

    def _propose(self, workspace, report) -> dict | None:
        if self.provider is not None:
            out = self._propose_llm(workspace, report)
            if out:
                return out
        return self._propose_rule(workspace, report)

    def _propose_llm(self, workspace, report) -> dict | None:
        sys = (
            "You improve a math-solving agent. Write ONE new, reusable skill that would "
            "prevent the failures below. Ground every instruction in the specific feedback "
            "(wrong answers, tracebacks). Be concrete and general (not problem-specific).\n"
            "Respond EXACTLY as:\nSKILL_NAME: <kebab-case-name>\nSKILL_BODY:\n<markdown body>"
        )
        user = "Failures this cycle:\n" + json.dumps(report, indent=2)[:6000]
        try:
            resp = self.provider.complete(
                [LLMMessage("system", sys), LLMMessage("user", user)], max_tokens=1200
            )
        except Exception:
            return None
        m = re.search(
            r"SKILL_NAME:\s*([a-z0-9\-]+)\s*SKILL_BODY:\s*(.*)",
            resp.content or "",
            re.DOTALL | re.IGNORECASE,
        )
        if not m:
            return None
        name = m.group(1).strip().lower()
        body = m.group(2).strip()[: self.max_skill_chars]
        content = (
            f"---\nname: {name}\ndescription: Evolved skill addressing observed failures.\n---\n{body}\n"
        )
        workspace.write_skill(name, content)
        return {"summary": f"LLM-evolved skill '{name}'", "meta": {"mode": "llm", "skill": name}}

    def _propose_rule(self, workspace, report) -> dict | None:
        if any(r["tool_errors"] for r in report):
            name = "python-tool-discipline"
            body = (
                "# Python tool discipline\n\n"
                "Symbols `x, y, z, n, k` are predefined; define others with `symbols(...)`. "
                "Import what you use and always `print(...)` the result. If the tool returns a "
                "traceback, read it and fix the snippet before continuing — never ignore a tool error."
            )
        else:
            name = "final-answer-discipline"
            body = (
                "# Final-answer discipline\n\n"
                "Re-read exactly what the problem asks for (a list? a fraction? an integer? a set?) "
                "and match that format. Verify the value with the Python tool, then end with "
                "`Answer: $\\boxed{...}$`."
            )
        content = (
            f"---\nname: {name}\ndescription: {body.splitlines()[0].lstrip('# ').strip()}.\n---\n{body}\n"
        )
        workspace.write_skill(name, content)
        return {"summary": f"added skill '{name}'", "meta": {"mode": "rule", "skill": name}}
