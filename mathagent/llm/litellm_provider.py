"""LiteLLM-backed LLM provider — a drop-in for A-Evolve's `LLMProvider`.

This honours the "multi-provider via LiteLLM" choice: the *same* provider powers
both the solver agent (`MathAgent`) and, optionally, the evolution engine.

Model names use LiteLLM's ``provider/model`` convention, e.g.
``anthropic/claude-opus-4-...``, ``openai/gpt-4o``, ``deepseek/deepseek-chat``.
Credentials are read from the usual env vars (``ANTHROPIC_API_KEY`` etc.).
"""

from __future__ import annotations

import os
from typing import Any

from agent_evolve.llm.base import LLMMessage, LLMProvider, LLMResponse


class LiteLLMProvider(LLMProvider):
    """Route completions through LiteLLM so any backend works unchanged."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **client_kwargs: Any,
    ) -> None:
        self.model = model or os.environ.get(
            "MATHAGENT_MODEL", "anthropic/claude-opus-4-6"
        )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client_kwargs = client_kwargs

    def _to_dicts(self, messages: list[LLMMessage]) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def complete(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        import litellm

        resp = litellm.completion(
            model=self.model,
            messages=self._to_dicts(messages),
            max_tokens=max_tokens or self.max_tokens,
            temperature=self.temperature if temperature is None else temperature,
            **{**self.client_kwargs, **kwargs},
        )
        choice = resp.choices[0]
        content = choice.message.content or ""
        usage = {}
        if getattr(resp, "usage", None):
            usage = {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
            }
        return LLMResponse(content=content, usage=usage, raw=resp)

    def complete_with_tools(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]],
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        import litellm

        resp = litellm.completion(
            model=self.model,
            messages=self._to_dicts(messages),
            tools=tools,
            max_tokens=max_tokens or self.max_tokens,
            **{**self.client_kwargs, **kwargs},
        )
        choice = resp.choices[0]
        content = choice.message.content or ""
        return LLMResponse(content=content, usage={}, raw=resp)


class MockProvider(LLMProvider):
    """Offline provider for smoke-testing the harness without an API key.

    Default behaviour drives the agent's text tool-protocol once: emit a
    ``<tool:python>`` block, then on the next turn emit a ``<final>`` answer.
    Pass ``scripted=[...]`` to return fixed responses in order.
    """

    def __init__(self, scripted: list[str] | None = None) -> None:
        self.scripted = scripted
        self._i = 0

    def complete(self, messages, max_tokens=None, temperature=None, **kwargs):
        if self.scripted is not None:
            text = self.scripted[min(self._i, len(self.scripted) - 1)]
            self._i += 1
            return LLMResponse(content=text, usage={}, raw=None)

        last = messages[-1].content if messages else ""
        if "TOOL RESULTS" in last:
            text = (
                "<final>\n"
                "We compute directly: $2 + 2 = 4$.\n\n"
                "Answer: $\\boxed{4}$\n"
                "</final>"
            )
        else:
            text = (
                "I will verify the arithmetic with Python.\n"
                "<tool:python>\nprint(2 + 2)\n</tool>"
            )
        return LLMResponse(content=text, usage={}, raw=None)

    def complete_with_tools(self, messages, tools, max_tokens=None, **kwargs):
        return self.complete(messages, max_tokens=max_tokens)


def build_provider(kind: str = "litellm", **kwargs: Any) -> LLMProvider:
    """Factory: ``kind`` in {"litellm", "mock"}."""
    if kind == "mock":
        return MockProvider(scripted=kwargs.get("scripted"))
    # drop provider-irrelevant keys (e.g. a None model passthrough)
    kwargs.pop("scripted", None)
    if kwargs.get("model") is None:
        kwargs.pop("model", None)
    return LiteLLMProvider(**kwargs)
