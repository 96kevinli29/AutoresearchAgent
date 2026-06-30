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
_FINAL_OPEN_RE = re.compile(r"<final>\s*(.*)$", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:latex|tex|markdown)?")


def _extract_solution(text: str) -> str:
    """Robustly pull the final solution out of a model reply, even when the
    model (esp. reasoning models) forgets to close <final>, wraps in code
    fences, or leaves <think> chatter in the content."""
    m = _FINAL_RE.search(text)
    if m:
        t = m.group(1)
    else:
        m = _FINAL_OPEN_RE.search(text)  # opened <final> but never closed (long answer)
        t = m.group(1) if m else text
    t = _THINK_RE.sub("", t)
    t = _FENCE_RE.sub("", t).replace("```", "")
    return t.strip()


class MathAgent(BaseAgent):
    def __init__(
        self,
        workspace_dir,
        provider: LLMProvider,
        tools: ToolRegistry | None = None,
        max_steps: int = 8,
        max_tokens: int | None = None,  # None => uncapped output (hard problems)
        max_cost_usd: float | None = None,  # per-solve $ budget (None => no cap)
        max_total_tokens: int | None = None,  # per-solve token budget (None => no cap)
    ) -> None:
        self.provider = provider
        self.tools = tools or ToolRegistry()
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.max_cost_usd = max_cost_usd
        self.max_total_tokens = max_total_tokens
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
            "\n\n## Finishing — REQUIRED format\n"
            "Your message that contains the solution MUST put the complete, self-contained "
            "solution as valid LaTeX between the literal tags `<final>` and `</final>`, and "
            "MUST include the closing `</final>`.\n"
            "- Do not put scratch work or chain-of-thought outside the tags; only the clean solution goes inside.\n"
            "- Use LaTeX only (no Markdown, no ``` code fences).\n"
            "- It must compile on its own (define what you use; balanced braces and environments).\n"
            "- End with `Answer: $\\boxed{...}$` when the problem has a closed-form answer.\n"
            "Take as much space as the problem needs — there is no length limit."
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

    def _over_budget(self) -> str | None:
        u = self.last_usage
        if self.max_cost_usd and u.get("cost_usd", 0) >= self.max_cost_usd:
            return "cost"
        if self.max_total_tokens and u.get("total_tokens", 0) >= self.max_total_tokens:
            return "tokens"
        return None

    def _track(self, usage: dict) -> None:
        t = self.last_usage
        for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cost_usd"):
            if usage.get(k):
                t[k] = t.get(k, 0) + usage[k]
        t["llm_calls"] = t.get("llm_calls", 0) + 1
        if usage.get("model"):
            t["model"] = usage["model"]

    def solve(self, task: Task, on_event=None) -> Trajectory:
        """Solve ``task``. If ``on_event`` is given, it is called with a dict for
        each step as it happens (live streaming): an ``llm`` event after every
        model turn and a ``tool`` event after every tool run, each carrying the
        running token/cost usage so far under ``running``."""
        messages = [
            LLMMessage("system", self._system()),
            LLMMessage("user", self._user(task)),
        ]
        steps: list[dict] = []
        self.last_usage = {}

        def _emit(ev: dict) -> None:
            if on_event:
                ev["running"] = dict(self.last_usage)
                try:
                    on_event(ev)
                except Exception:
                    pass

        def _emit_token(piece: str) -> None:
            if on_event:
                try:
                    on_event({"ev": "token", "text": piece})
                except Exception:
                    pass

        def _traj(output: str) -> Trajectory:
            return Trajectory(
                task_id=task.id,
                output=output,
                steps=steps,
                conversation=[{"role": m.role, "content": m.content} for m in messages],
            )

        for turn in range(self.max_steps):
            tok_cb = _emit_token if on_event else None
            resp = self.provider.complete(
                messages, max_tokens=self.max_tokens, on_token=tok_cb
            )
            text = resp.content or ""
            self._track(resp.usage or {})
            messages.append(LLMMessage("assistant", text))
            # record the model's reasoning for this turn (tool blocks stripped for display)
            reasoning = _TOOL_RE.sub("", _FINAL_RE.sub("", text)).strip()
            steps.append(
                {"type": "llm", "turn": turn + 1, "text": text,
                 "reasoning": reasoning, "usage": resp.usage or {}}
            )
            _emit({"ev": "step", "type": "llm", "turn": turn + 1,
                   "reasoning": reasoning, "usage": resp.usage or {}})

            # Done when there's a closed <final>, or no tool call to run (in which
            # case _extract_solution handles an unclosed <final> / stray chatter).
            calls = _TOOL_RE.findall(text)
            if _FINAL_RE.search(text) or not calls:
                solution = _extract_solution(text)
                steps.append({"type": "final", "output": solution})
                return _traj(solution)

            # budget guard: stop before spending more if a per-solve cap is hit
            over = self._over_budget()
            if over:
                solution = _extract_solution(text) or text.strip()
                steps.append({"type": "final", "output": solution, "stopped": over})
                _emit({"ev": "step", "type": "budget", "reason": over})
                return _traj(solution)

            results = []
            for name, body in calls:
                _emit({"ev": "step", "type": "tool_start", "tool": name, "input": body})
                out = self.tools.run(name, body)
                steps.append({"type": "tool", "tool": name, "input": body, "output": out})
                _emit({"ev": "step", "type": "tool", "tool": name, "input": body, "output": out})
                results.append(f"<tool:{name}> result:\n{out}")
            messages.append(
                LLMMessage(
                    "user",
                    "TOOL RESULTS:\n" + "\n\n".join(results) + "\n\nContinue, or give <final>.",
                )
            )

        return _traj(_extract_solution(messages[-1].content))
