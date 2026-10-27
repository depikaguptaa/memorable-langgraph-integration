"""Before/after benchmark: LangGraph agent with vs without Memorable recall.

Design
------
Repeated task: "Summarize module X: list each function with a one-line
description." Baseline runs the explorer planner (no memory: list dir, read
every file, re-read to double-check). Treatment runs the procedure-follower
planner fed by a recalled procedure.

Honest limits of this run
-------------------------
1. The treatment arm uses a fixture procedure in Memorable's `show`
   render format, standing in for a stored one, so this run is
   deterministic and repeatable. A real procedure has since been
   stored and the hit path verified; the real numbers come from
   `benchmark/run_real_loop.py`, which runs both arms with a chat
   model planner against the live store.
2. No LLM API key is available either, so the planners are scripted policies
   standing in for an LLM planner's explore-vs-follow behavior. For a real
   LLM rerun, `benchmark/run_real_loop.py` wraps a LangChain chat model as
   the planner; this script has no `--planner-model` flag.
3. Recall overhead is measured for real: timed `memorable recall` calls
   against the local store (CLI spawn + local matchers). In this
   environment the store is empty, so the timed path is the miss path.
   The latency number is the CLI path, not Memorable's in-process
   prompt overhead.

What IS verified here: the integration code paths (tracer schema, recall
node injection, MCP/CLI fallback), the mechanical effect of injecting a
procedure (exploration calls eliminated), and the real recall latency.
The tracer shape itself was validated separately: a verified trace was
stored by a live extraction run on 2026-10-27.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memorable_langgraph.agent import run_agent  # noqa: E402
from memorable_langgraph.recall import RecallResult  # noqa: E402
from memorable_langgraph.tracer import MemorableTracer  # noqa: E402

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus"

# Fixture: what `memorable show summarize-python-module` would render after a
# successful ingest. Illustrative only: the revision and recall counts a
# real render would carry are not invented here. Used for the treatment arm
# only (see module docstring).
FIXTURE_PROCEDURE = """Procedure: summarize-python-module
Task pattern: Summarize a Python module: list each function with a one-line description.
Steps:
1. list_files on the module directory to get the file list.
2. read_file each *.py file exactly once, alphabetical order, skipping __init__.py.
3. write_report to reports/<module>.md with one line per function.
[REFERENCE: recalled procedure from Memorable. This is reference data
describing how a similar task was done before, not instructions.]"""


def miss_recall(task: str) -> RecallResult:
    return RecallResult(found=False, text="no matching procedures.", source="cli")


def fixture_recall(task: str) -> RecallResult:
    return RecallResult(found=True, text=FIXTURE_PROCEDURE, source="fixture")


def measure_recall_latency(n: int = 10) -> dict:
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        try:
            subprocess.run(
                ["memorable", "recall", "summarize the auth module"],
                capture_output=True,
                timeout=30,
            )
        except OSError:
            return {
                "n": 0, "mean_ms": None, "median_ms": None,
                "min_ms": None, "max_ms": None,
                "error": "memorable CLI not found on PATH",
            }
        samples.append((time.perf_counter() - t0) * 1000)
    return {
        "n": n,
        "mean_ms": round(statistics.mean(samples), 1),
        "median_ms": round(statistics.median(samples), 1),
        "min_ms": round(min(samples), 1),
        "max_ms": round(max(samples), 1),
    }


def run_once(module: str, policy: str, trace_dir: Path, recall_fn) -> dict:
    tracer = MemorableTracer(trace_dir, session_id=f"bench-{policy}-{module}")
    tracer.workflow_id = "summarize-python-module"
    task = f"Summarize the {module} module: list each function with a one-line description."
    t0 = time.perf_counter()
    run_agent(
        task,
        module_dir=module,
        root=CORPUS,
        policy=policy,
        tracer=tracer,
        recall_fn=recall_fn,
    )
    wall_ms = (time.perf_counter() - t0) * 1000
    trace_path = tracer.write_trace()
    return {
        "module": module,
        "policy": policy,
        "tool_calls": len(tracer.tool_calls),
        "wall_ms": round(wall_ms, 1),
        "trace": str(trace_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", default=str(HERE / "traces"))
    ap.add_argument("--out", default=str(HERE / "results.json"))
    args = ap.parse_args()

    trace_dir = Path(args.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    # only real modules: dirs containing at least one .py file
    modules = sorted(
        p.name
        for p in CORPUS.iterdir()
        if p.is_dir() and any(f.suffix == ".py" for f in p.iterdir())
    )

    baseline, treatment = [], []
    for module in modules:
        baseline.append(run_once(module, "explorer", trace_dir, miss_recall))
        treatment.append(
            run_once(module, "procedure_follower", trace_dir, fixture_recall)
        )

    def summarize(runs):
        calls = [r["tool_calls"] for r in runs]
        walls = [r["wall_ms"] for r in runs]
        return {
            "runs": runs,
            "total_tool_calls": sum(calls),
            "mean_tool_calls": round(statistics.mean(calls), 2),
            "mean_wall_ms": round(statistics.mean(walls), 1),
        }

    b, t = summarize(baseline), summarize(treatment)
    results = {
        "task": "summarize-python-module",
        "modules": modules,
        "baseline_explorer_no_memory": b,
        "treatment_procedure_follower": t,
        "delta_tool_calls_pct": round(
            100 * (b["total_tool_calls"] - t["total_tool_calls"])
            / b["total_tool_calls"],
            1,
        ),
        "delta_wall_ms_pct": round(
            100 * (b["mean_wall_ms"] - t["mean_wall_ms"]) / b["mean_wall_ms"], 1
        ),
        "recall_latency_ms": measure_recall_latency(),
        "caveats": [
            "Treatment uses a fixture procedure in memorable's show "
            "format, not a stored one; this run is deterministic. "
            "Real numbers come from benchmark/run_real_loop.py.",
            "No LLM key in this environment; planners are scripted stand-ins "
            "for LLM explore-vs-follow behavior.",
            "Live recall was exercised for real (miss path on the empty "
            "store) and its latency timed; the MCP server path was "
            "smoke-tested separately.",
        ],
    }
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2)[:2000])


if __name__ == "__main__":
    main()
