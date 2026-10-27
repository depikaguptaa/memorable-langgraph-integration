"""Run one summarize-module task and write a trace Memorable can admit.

Memorable refuses a trace that ends without a clean execute-class
command: the last such command is kept as the proof the run worked,
and a trace with only reads and writes stores nothing. The plain
benchmark traces end in a write, so `memorable ingest` rejects them
with no_postcondition. This script runs the same agent with a verify
command as its last tool call, so the trace ends with an execute step
whose result starts with its exit code.

Usage (from the repo root, in the venv):

    python benchmark/run_verified_trace.py
    npx memorable-cli ingest benchmark/traces/bench-verified-auth.json
    npx memorable-cli list

The ingest step needs a logged-in CLI on the machine that runs it;
this script only writes the trace, it never calls the CLI itself.
Admission is still unproven until a live ingest succeeds: the
script prints the last step so the caller can check it before
ingesting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memorable_langgraph.agent import run_agent  # noqa: E402
from memorable_langgraph.tracer import MemorableTracer  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
CORPUS = HERE / "corpus"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="auth")
    ap.add_argument("--trace-dir", default=str(HERE / "traces"))
    ap.add_argument(
        "--verify-command",
        default="python -m pytest tests/test_integration.py -q",
        help="Command run as the last tool call. Runs from --verify-cwd.",
    )
    ap.add_argument("--verify-cwd", default=str(REPO_ROOT))
    args = ap.parse_args()

    trace_dir = Path(args.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    tracer = MemorableTracer(trace_dir, session_id=f"verified-{args.module}")
    tracer.workflow_id = "summarize-python-module"
    task = (
        f"Summarize the {args.module} module: "
        "list each function with a one-line description."
    )
    state = run_agent(
        task,
        module_dir=args.module,
        root=CORPUS,
        policy="explorer",
        tracer=tracer,
        verify_command=args.verify_command,
        verify_cwd=args.verify_cwd,
    )
    trace_path = tracer.write_trace(f"bench-verified-{args.module}.json")
    trace = json.loads(trace_path.read_text())
    last = trace["tool_calls"][-1] if trace["tool_calls"] else None
    print(json.dumps({
        "trace": str(trace_path),
        "tool_calls": len(trace["tool_calls"]),
        "report_path": state.get("report_path"),
        "functions_found": state.get("functions_found"),
        "last_step": last["name"] if last else None,
        "last_result_head": (
            last["result"][:120] if last else None
        ),
        "verify_result_head": (
            str(state.get("verify_result", ""))[:120]
            if state.get("verify_result") is not None
            else None
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
