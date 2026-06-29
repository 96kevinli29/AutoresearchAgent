"""FastAPI server — the same MathAgent core, exposed online.

Local CLI and this web service share one agent/tools/workspace, so behaviour is
identical. Run with ``mathagent serve`` (see cli.py) or::

    uvicorn mathagent.web.server:app --reload

Security note: the Python tool executes model-written code in a subprocess (runs as
*you*). For a multi-user/public deployment, start with ``--no-python`` or sandbox the
runner (M5). For personal/trusted use it is fine.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent_evolve import Task

from ..agent import MathAgent
from ..benchmarks import extract_boxed
from ..llm import build_provider
from ..tools import default_registry, export

_STATIC = Path(__file__).resolve().parent / "static"


class SolveRequest(BaseModel):
    problem: str
    provider: str | None = None
    model: str | None = None
    enable_python: bool | None = None
    enable_lean: bool | None = None
    format: str = "pdf"  # none | latex | pdf | docx
    max_steps: int = 8


def create_app(
    workspace: str | None = None,
    provider_kind: str = "litellm",
    model: str | None = None,
    enable_python: bool = True,
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

    app.state.cfg = dict(
        workspace=workspace,
        provider_kind=provider_kind,
        model=model,
        enable_python=enable_python,
        enable_lean=enable_lean,
        out_dir=out_dir,
    )

    app.mount("/files", StaticFiles(directory=out_dir), name="files")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse((_STATIC / "index.html").read_text(encoding="utf-8"))

    @app.get("/api/health")
    def health() -> dict:
        cfg = app.state.cfg
        return {
            "status": "ok",
            "provider": cfg["provider_kind"],
            "model": cfg["model"],
            "python": cfg["enable_python"],
            "lean": cfg["enable_lean"],
        }

    @app.post("/api/solve")
    def api_solve(req: SolveRequest) -> JSONResponse:
        cfg = app.state.cfg
        prov = build_provider(
            req.provider or cfg["provider_kind"], model=req.model or cfg["model"]
        )
        py = cfg["enable_python"] if req.enable_python is None else req.enable_python
        ln = cfg["enable_lean"] if req.enable_lean is None else req.enable_lean
        tools = default_registry(python=py, lean=ln)
        agent = MathAgent(cfg["workspace"], prov, tools=tools, max_steps=req.max_steps)

        try:
            traj = agent.solve(Task(id=uuid.uuid4().hex[:8], input=req.problem))
        except Exception as e:  # surface provider/credential errors to the UI
            return JSONResponse(status_code=502, content={"error": f"{type(e).__name__}: {e}"})

        files: dict[str, str] = {}
        if req.format != "none":
            name = f"sol-{uuid.uuid4().hex[:8]}"
            produced = export(traj.output, cfg["out_dir"], name=name, fmt=req.format)
            for kind, p in produced.items():
                if kind in ("tex", "pdf", "docx"):
                    files[kind] = f"/files/{Path(p).name}"

        tool_calls = [s for s in traj.steps if s.get("type") == "tool"]
        return JSONResponse(
            {
                "solution": traj.output,
                "answer": extract_boxed(traj.output),
                "files": files,
                "tool_calls": tool_calls,
            }
        )

    return app


# default app for `uvicorn mathagent.web.server:app`
app = create_app()
