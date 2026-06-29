"""MathAgent — an A-Evolve `BaseAgent` for math problem solving / proving.

Uses a provider-agnostic *text tool-protocol* (works on any LiteLLM backend,
no dependence on native function-calling schemas):

    <tool:python> ...code... </tool>     run a computation / find a counterexample
    <tool:lean>   ...lean... </tool>     verify formally (if enabled)
    <final> ...LaTeX solution... </final>  finish

The loop is error-driven (COPRA-style): tool output is fed back so the model can
react to failures and retry, up to ``max_steps``.
"""

from __future__ import annotations

import re

from agent_evolve import BaseAgent, Task, Trajectory
from agent_evolve.llm.base import LLMMessage, LLMProvider

from ..tools.registry import ToolRegistry

_TOOL_RE = re.compile(r"<tool:(\w+)>\s*(.*?)\s*</tool>", re.DOTALL)
_FINAL_RE = re.compile(r"<final>\s*(.*?)\s*</final>", re.DOTALL)


class MathAgent(BaseAgent):
    def __init__(
        self,
        workspace_dir,
        provider: LLMProvider,
        tools: ToolRegistry | None = None,
        max_steps: int = 8,
        max_tokens: int = 4096,
    ) -> None:
        self.provider = provider
        self.tools = tools or ToolRegistry()
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.last_usage: dict = {}
        super().__init__(workspace_dir)  # populates system_prompt, skills, memories

    # ---- prompt assembly (reads evolvable workspace state) ----------------

    def _skills_block(self) -> str:
        if not self.skills:
            return ""
        chunks = []
        for s in self.skills:
            try:
                content = self.workspace.read_skill(s.name)
            except Exception:
                content = s.description
            chunks.append(f"### Skill: {s.name}\n{content}")
        return "## Available skills (apply when relevant)\n\n" + "\n\n".join(chunks)

    def _memory_block(self) -> str:
        if not self.memories:
            return ""
        recent = self.memories[-8:]
        lines = []
        for m in recent:
            text = m.get("content") if isinstance(m, dict) else str(m)
            if text:
                lines.append(f"- {text}")
        if not lines:
            return ""
        return "## Lessons from past attempts\n" + "\n".join(lines)

    def _system(self) -> str:
        tool_lines = self.tools.descriptions()
        tool_help = ""
        if tool_lines:
            tool_help = (
                "\n\n## Tools\nEmit these blocks to use a tool; results are fed back:\n"
                + "\n".join(tool_lines)
                + "\nNever assert a computed result you have not checked with a tool when one is available."
            )
        finish = (
            "\n\n## Finishing\nWhen the solution is complete and verified, output the "
            "full self-contained solution in LaTeX inside <final>...</final>, "
            "ending with `Answer: $\\boxed{...}$` when the problem has a closed-form answer."
        )
        return (self.system_prompt or "You are a rigorous mathematician.") + tool_help + finish

    def _user(self, task: Task) -> str:
        parts = [f"## Problem\n{task.input}"]
        sb = self._skills_block()
        if sb:
            parts.append(sb)
        mb = self._memory_block()
        if mb:
            parts.append(mb)
        return "\n\n".join(parts)

    # ---- the solve loop ---------------------------------------------------

    def _track(self, usage: dict) -> None:
        t = self.last_usage
        for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cost_usd"):
            if usage.get(k):
                t[k] = t.get(k, 0) + usage[k]
        t["llm_calls"] = t.get("llm_calls", 0) + 1
        if usage.get("model"):
            t["model"] = usage["model"]

    def solve(self, task: Task) -> Trajectory:
        messages = [
            LLMMessage("system", self._system()),
            LLMMessage("user", self._user(task)),
        ]
        steps: list[dict] = []
        self.last_usage = {}

        def _traj(output: str) -> Trajectory:
            return Trajectory(
                task_id=task.id,
                output=output,
                steps=steps,
                conversation=[{"role": m.role, "content": m.content} for m in messages],
            )

        for turn in range(self.max_steps):
            resp = self.provider.complete(messages, max_tokens=self.max_tokens)
            text = resp.content or ""
            self._track(resp.usage or {})
            messages.append(LLMMessage("assistant", text))
            # record the model's reasoning for this turn (tool blocks stripped for display)
            reasoning = _TOOL_RE.sub("", _FINAL_RE.sub("", text)).strip()
            steps.append(
                {"type": "llm", "turn": turn + 1, "text": text,
                 "reasoning": reasoning, "usage": resp.usage or {}}
            )

            final = _FINAL_RE.search(text)
            if final:
                solution = final.group(1).strip()
                steps.append({"type": "final", "output": solution})
                return _traj(solution)

            calls = _TOOL_RE.findall(text)
            if not calls:
                # No tool, no <final>: treat the whole message as the answer.
                steps.append({"type": "final", "output": text.strip()})
                return _traj(text.strip())

            results = []
            for name, body in calls:
                out = self.tools.run(name, body)
                steps.append({"type": "tool", "tool": name, "input": body, "output": out})
                results.append(f"<tool:{name}> result:\n{out}")
            messages.append(
                LLMMessage(
                    "user",
                    "TOOL RESULTS:\n" + "\n\n".join(results) + "\n\nContinue, or give <final>.",
                )
            )

        return _traj(messages[-1].content.strip())
