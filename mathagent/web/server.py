"""FastAPI server — the same MathAgent core, exposed online.

Local CLI and this web service share one agent/tools/workspace, so behaviour is
identical. Run with ``mathagent serve`` (see cli.py) or::

    uvicorn mathagent.web.server:app --reload

Verification tools (Python, Lean) are OFF by default — the plain model solve is the
out-of-the-box experience. Enable them per request or with ``--python`` / ``--lean``.
Security note: the Python tool executes model-written code in a subprocess (runs as
*you*); only enable it for trusted use, or sandbox the runner.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent_evolve import Task

from ..agent import MathAgent
from ..benchmarks import extract_boxed
from ..llm import build_provider
from ..tools import default_registry, export

_STATIC = Path(__file__).resolve().parent / "static"

# A few popular families only. Static fallback (used if the live OpenRouter fetch fails).
_CURATED_MODELS = [
    {"id": "openrouter/anthropic/claude-opus-4.8", "label": "Claude Opus 4.8", "price": "$5 / $25"},
    {"id": "openrouter/anthropic/claude-sonnet-4.6", "label": "Claude Sonnet 4.6", "price": "$3 / $15"},
    {"id": "openrouter/openai/gpt-5.5", "label": "GPT-5.5", "price": "$5 / $30"},
    {"id": "openrouter/deepseek/deepseek-v4-pro", "label": "DeepSeek V4 Pro", "price": "$0.43 / $1"},
]

# For each family: prefix, substrings to exclude (variants), optional special filter.
_FAMILIES = [
    ("Claude Opus", "anthropic/claude-opus", ("fast",), None),
    ("Claude Sonnet", "anthropic/claude-sonnet", ("thinking",), None),
    ("GPT", "openai/gpt-5", ("mini", "nano", "codex", "image", "search", "audio", "-chat", "-pro"), None),
    ("DeepSeek", "deepseek/deepseek", ("distill", "flash"), None),
]
_models_cache: dict = {"ts": 0.0, "data": None}


def _price(m: dict) -> str:
    p = m.get("pricing") or {}
    try:
        return f"${float(p.get('prompt', 0)) * 1e6:.2f} / ${float(p.get('completion', 0)) * 1e6:.0f}"
    except Exception:
        return ""


def _curate(allm: list) -> list:
    out = []
    for label, pref, excl, special in _FAMILIES:
        c = [m for m in allm if m["id"].startswith(pref)
             and not any(x in m["id"] for x in excl) and ":free" not in m["id"]]
        if special == "qwen":
            c = [m for m in c if "max" in m["id"] or "thinking" in m["id"]] or c
        if special == "gemini":
            c = [m for m in c if "pro" in m["id"]] or c
        if not c:
            continue
        m = max(c, key=lambda x: x.get("created", 0))
        out.append({"id": "openrouter/" + m["id"],
                    "label": f"{label} — {m['id'].split('/')[-1]}", "price": _price(m)})
    return out


def _live_models() -> list | None:
    """Newest flagship per family, fetched live from OpenRouter (cached 10 min)."""
    now = time.time()
    if _models_cache["data"] and now - _models_cache["ts"] < 600:
        return _models_cache["data"]
    key = os.environ.get("OPENROUTER_API_KEY")
    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {key}"} if key else {},
        )
        with urllib.request.urlopen(req, timeout=8) as r:
            allm = json.load(r)["data"]
        data = _curate(allm)
        if data:
            _models_cache.update(ts=now, data=data)
            return data
    except Exception:
        pass
    return None


class SolveRequest(BaseModel):
    problem: str
    provider: str | None = None
    model: str | None = None
    enable_python: bool | None = None
    enable_lean: bool | None = None
    format: str = "pdf"  # none | latex | pdf | docx
    max_steps: int = 8
    max_cost_usd: float | None = None


def create_app(
    workspace: str | None = None,
    provider_kind: str = "litellm",
    model: str | None = None,
    enable_python: bool = False,
    enable_lean: bool = False,
    out_dir: str = "out",
) -> FastAPI:
    from .._env import load_dotenv
    from .._workspace import resolve_workspace

    load_dotenv()
    workspace = resolve_workspace(workspace)
    app = FastAPI(title="mathagent", version="0.1.0")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    _env_cap = os.environ.get("MATHAGENT_MAX_COST")
    app.state.cfg = dict(
        workspace=workspace,
        provider_kind=provider_kind,
        model=model,
        enable_python=enable_python,
        enable_lean=enable_lean,
        out_dir=out_dir,
        max_cost=float(_env_cap) if _env_cap else None,
    )

    app.mount("/files", StaticFiles(directory=out_dir), name="files")
    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

    _NOCACHE = {"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"}

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(
            (_STATIC / "index.html").read_text(encoding="utf-8"), headers=_NOCACHE
        )

    def _resolved_model() -> str:
        cfg = app.state.cfg
        if cfg["provider_kind"] == "mock":
            return "mock"
        return cfg["model"] or os.environ.get("MATHAGENT_MODEL", "(unset)")

    @app.get("/api/health")
    def health() -> JSONResponse:
        cfg = app.state.cfg
        return JSONResponse(
            {
                "status": "ok",
                "ui": "studio-v12",
                "provider": cfg["provider_kind"],
                "model": _resolved_model(),
                "python": cfg["enable_python"],
                "lean": cfg["enable_lean"],
            },
            headers=_NOCACHE,
        )

    @app.get("/api/models")
    def models() -> JSONResponse:
        live = _live_models() if app.state.cfg["provider_kind"] != "mock" else None
        return JSONResponse(
            {"default": _resolved_model(), "models": live or _CURATED_MODELS, "live": bool(live)},
            headers=_NOCACHE,
        )

    @app.post("/api/solve")
    def api_solve(req: SolveRequest) -> JSONResponse:
        cfg = app.state.cfg
        prov = build_provider(
            req.provider or cfg["provider_kind"], model=req.model or cfg["model"]
        )
        py = cfg["enable_python"] if req.enable_python is None else req.enable_python
        ln = cfg["enable_lean"] if req.enable_lean is None else req.enable_lean
        tools = default_registry(python=py, lean=ln)
        agent = MathAgent(
            cfg["workspace"], prov, tools=tools, max_steps=req.max_steps,
            max_cost_usd=(req.max_cost_usd if req.max_cost_usd is not None else cfg["max_cost"]),
        )

        t0 = time.time()
        try:
            traj = agent.solve(Task(id=uuid.uuid4().hex[:8], input=req.problem))
        except Exception as e:  # surface provider/credential errors to the UI
            return JSONResponse(status_code=502, content={"error": f"{type(e).__name__}: {e}"})
        elapsed = round(time.time() - t0, 2)

        files: dict[str, str] = {}
        if req.format != "none":
            name = f"sol-{uuid.uuid4().hex[:8]}"
            produced = export(traj.output, cfg["out_dir"], name=name, fmt=req.format)
            for kind, p in produced.items():
                if kind in ("tex", "pdf", "docx"):
                    files[kind] = f"/files/{Path(p).name}"

        return JSONResponse(
            {
                "solution": traj.output,
                "answer": extract_boxed(traj.output),
                "files": files,
                "model": req.model or _resolved_model(),
                "elapsed_s": elapsed,
                "usage": getattr(agent, "last_usage", {}),
                "steps": traj.steps,
            }
        )

    @app.post("/api/solve_stream")
    async def solve_stream(req: SolveRequest) -> StreamingResponse:
        """Live streaming: emits one NDJSON line per agent step (with running
        token/cost usage), then a final ``done`` line with the full result."""
        import asyncio
        import queue
        import threading

        cfg = app.state.cfg
        prov = build_provider(req.provider or cfg["provider_kind"], model=req.model or cfg["model"])
        py = cfg["enable_python"] if req.enable_python is None else req.enable_python
        ln = cfg["enable_lean"] if req.enable_lean is None else req.enable_lean
        tools = default_registry(python=py, lean=ln)
        agent = MathAgent(
            cfg["workspace"], prov, tools=tools, max_steps=req.max_steps,
            max_cost_usd=(req.max_cost_usd if req.max_cost_usd is not None else cfg["max_cost"]),
        )
        model = req.model or _resolved_model()

        q: queue.Queue = queue.Queue()

        def run() -> None:
            t0 = time.time()
            try:
                traj = agent.solve(
                    Task(id=uuid.uuid4().hex[:8], input=req.problem), on_event=q.put
                )
                files: dict[str, str] = {}
                if req.format != "none":
                    name = f"sol-{uuid.uuid4().hex[:8]}"
                    produced = export(traj.output, cfg["out_dir"], name=name, fmt=req.format)
                    for kind, p in produced.items():
                        if kind in ("tex", "pdf", "docx"):
                            files[kind] = f"/files/{Path(p).name}"
                q.put({"ev": "done", "result": {
                    "solution": traj.output, "answer": extract_boxed(traj.output),
                    "files": files, "model": model, "elapsed_s": round(time.time() - t0, 2),
                    "usage": getattr(agent, "last_usage", {}), "steps": traj.steps,
                }})
            except Exception as e:
                q.put({"ev": "error", "error": f"{type(e).__name__}: {e}"})
            finally:
                q.put(None)

        threading.Thread(target=run, daemon=True).start()

        async def gen():
            loop = asyncio.get_event_loop()
            while True:
                ev = await loop.run_in_executor(None, q.get)
                if ev is None:
                    break
                yield json.dumps(ev) + "\n"

        return StreamingResponse(
            gen(), media_type="application/x-ndjson",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    return app


# default app for `uvicorn mathagent.web.server:app`
app = create_app()
