"""mathagent CLI — solve a problem end-to-end, or evaluate a task set.

Examples
--------
    # smoke test, no API key needed:
    mathagent solve "Compute 2+2." --provider mock --format pdf

    # real run (set ANTHROPIC_API_KEY / MATHAGENT_MODEL first):
    mathagent solve "Find all real roots of x^2-5x+6=0." --format pdf --lean

    # evaluate the seed task set:
    mathagent eval data/seed_problems.jsonl --provider mock
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from agent_evolve import EvolveConfig, Evolver, Task

from .agent import MathAgent
from .benchmarks import MathBenchmark
from .engine import MathEvolutionEngine
from .llm import build_provider
from .tools import default_registry, export

_DEFAULT_WS = str(Path(__file__).resolve().parent.parent / "workspace")


def _make_agent(args) -> MathAgent:
    provider = build_provider(
        args.provider, model=getattr(args, "model", None)
    )
    tools = default_registry(python=not args.no_python, lean=args.lean)
    return MathAgent(args.workspace, provider, tools=tools, max_steps=args.max_steps)


def cmd_solve(args) -> int:
    problem = args.problem
    if args.file:
        problem = Path(args.file).read_text(encoding="utf-8")
    if not problem:
        print("error: provide a problem (positional or --file)", file=sys.stderr)
        return 2

    agent = _make_agent(args)
    traj = agent.solve(Task(id="cli", input=problem))

    print("\n===== SOLUTION =====\n")
    print(traj.output)

    if args.format != "none":
        produced = export(
            traj.output,
            out_dir=args.out_dir,
            name=args.name,
            title=args.title,
            fmt=args.format,
        )
        print("\n===== OUTPUT FILES =====")
        for k, v in produced.items():
            print(f"  {k}: {v}")
    return 0


def cmd_eval(args) -> int:
    bench = MathBenchmark(args.taskfile)
    agent = _make_agent(args)
    tasks = bench.get_tasks(split=args.split, limit=args.limit)

    n = 0
    correct = 0
    for t in tasks:
        traj = agent.solve(t)
        fb = bench.evaluate(t, traj)
        n += 1
        correct += int(fb.success)
        mark = "ok " if fb.success else "FAIL"
        print(f"[{mark}] {t.id}: {fb.detail}")
    acc = correct / n if n else 0.0
    print(f"\nAccuracy: {correct}/{n} = {acc:.3f}")
    return 0


def cmd_evolve(args) -> int:
    # work on a copy so the pristine seed workspace stays clean
    work_root = Path(args.work_dir)
    work_ws = work_root / "workspace"
    if work_ws.exists() and args.fresh:
        shutil.rmtree(work_ws)
    if not work_ws.exists():
        shutil.copytree(args.workspace, work_ws)

    provider = build_provider(args.provider, model=args.model)
    tools = default_registry(python=not args.no_python, lean=args.lean)
    agent = MathAgent(work_ws, provider, tools=tools, max_steps=args.max_steps)
    bench = MathBenchmark(args.taskfile)

    evolver_provider = None
    if not args.rule:
        evolver_provider = build_provider(
            args.provider, model=(args.evolver_model or args.model)
        )
    engine = MathEvolutionEngine(
        provider=evolver_provider, holdout_limit=args.holdout_limit
    )

    config = EvolveConfig(
        batch_size=args.batch_size,
        max_cycles=args.cycles,
        egl_window=args.egl_window,
    )
    evolver = Evolver(
        agent=agent, benchmark=bench, config=config, engine=engine, work_dir=str(work_root)
    )
    result = evolver.run(cycles=args.cycles)

    print("\n===== EVOLUTION RESULT =====")
    print(f"cycles: {result.cycles_completed}  converged: {result.converged}")
    print(f"score history: {[round(s, 3) for s in result.score_history]}")
    print(f"final score: {result.final_score:.3f}")
    print("\nskills after evolution:")
    for s in agent.workspace.list_skills():
        print(f"  - {s.name}: {s.description[:70]}")
    git_log = agent.workspace.read_evolution_history() if hasattr(
        agent.workspace, "read_evolution_history"
    ) else ""
    if git_log:
        print("\nevolution history (git):")
        print(git_log)
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    from .web import create_app

    app = create_app(
        workspace=args.workspace,
        provider_kind=args.provider,
        model=args.model,
        enable_python=not args.no_python,
        enable_lean=args.lean,
        out_dir=args.out_dir,
    )
    print(f"mathagent web → http://{args.host}:{args.port}  (provider={args.provider})")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mathagent")

    # shared options, available on every subcommand (after the subcommand name)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--workspace", default=_DEFAULT_WS, help="evolvable workspace dir")
    common.add_argument("--provider", default="litellm", choices=["litellm", "mock"])
    common.add_argument("--model", default=None, help="LiteLLM model id, e.g. anthropic/claude-opus-4-6")
    common.add_argument("--no-python", action="store_true", help="disable the Python tool")
    common.add_argument("--lean", action="store_true", help="enable Lean verification (M3)")
    common.add_argument("--max-steps", type=int, default=8)

    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("solve", help="solve one problem", parents=[common])
    s.add_argument("problem", nargs="?", default="")
    s.add_argument("--file", help="read the problem from a file")
    s.add_argument("--format", default="latex", choices=["none", "latex", "pdf", "docx"])
    s.add_argument("--out-dir", default="out")
    s.add_argument("--name", default="solution")
    s.add_argument("--title", default="Solution")
    s.set_defaults(func=cmd_solve)

    e = sub.add_parser("eval", help="evaluate a task set", parents=[common])
    e.add_argument("taskfile")
    e.add_argument("--split", default="train", choices=["train", "test", "all"])
    e.add_argument("--limit", type=int, default=None)
    e.set_defaults(func=cmd_eval)

    v = sub.add_parser("evolve", help="self-improve the workspace on a task set", parents=[common])
    v.add_argument("taskfile")
    v.add_argument("--cycles", type=int, default=5)
    v.add_argument("--batch-size", type=int, default=6)
    v.add_argument("--holdout-limit", type=int, default=4)
    v.add_argument("--egl-window", type=int, default=2)
    v.add_argument("--rule", action="store_true", help="rule-based mutations (no evolver LLM)")
    v.add_argument("--evolver-model", default=None, help="model for the mutation LLM (defaults to --model)")
    v.add_argument("--work-dir", default="evolution_workdir")
    v.add_argument("--fresh", action="store_true", help="reset the working workspace copy first")
    v.set_defaults(func=cmd_evolve)

    sv = sub.add_parser("serve", help="run the web UI/API", parents=[common])
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--out-dir", default="out")
    sv.set_defaults(func=cmd_serve)

    return p


def main(argv=None) -> int:
    from ._env import load_dotenv

    load_dotenv()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
