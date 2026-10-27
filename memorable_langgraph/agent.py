"""Demo LangGraph agent wired with Memorable recall + trace capture.

Task: summarize a Python module (list each function with a one-line
description) and write the report. The graph is:

    memorable_recall -> execute -> END

`memorable_recall` injects any stored procedure for the task. `execute`
runs the planner's tool calls through real LangChain tools so the
MemorableTracer sees every call. The planner is swappable:

- "explorer": scripted stand-in for an LLM planner working without memory.
  Lists the directory, reads every file, re-reads to double-check, then
  writes the report. This is the baseline.
- "procedure_follower": follows an injected procedure's file list and order
  exactly, no re-reads. This is the treatment arm.
- a planner callable, for example the wrapper `benchmark/run_real_loop.py`
  builds around a LangChain chat model (needs an API key; not exercised
  in this environment).
"""

from __future__ import annotations

import ast
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from .recall import RecallResult, build_recall_node


class AgentState(TypedDict, total=False):
    task: str
    memorable_procedure: str
    report_path: str
    functions_found: int
    verify_result: str


def _split_command(command: str) -> list[str]:
    """Split a command string without a shell.

    `shlex.split` breaks an executable path that contains a space, for
    example a venv under a folder named "side quest": the first token
    becomes a path that does not exist and the run reports 127. When
    the first token resolves to neither a file nor a PATH lookup, glue
    leading tokens back together while the joined path names a real
    file. Quoted paths never reach this step: shlex already keeps them
    as one token, which resolves on the first check.
    """
    parts = shlex.split(command)
    if not parts:
        return parts
    if os.path.isfile(parts[0]) or shutil.which(parts[0]) is not None:
        return parts
    for end in range(2, len(parts) + 1):
        candidate = " ".join(parts[:end])
        if os.path.isfile(candidate):
            return [candidate] + parts[end:]
    return parts


def make_tools(root: str | Path):
    root = Path(root).resolve()

    def _safe(rel: str) -> Path:
        p = (root / rel).resolve()
        if not p.is_relative_to(root):
            raise ValueError("path escapes module dir")
        return p

    @tool
    def list_files(directory: str) -> str:
        """List files in a directory relative to the module root."""
        p = _safe(directory)
        return "\n".join(sorted(x.name for x in p.iterdir() if x.is_file()))

    @tool
    def read_file(path: str) -> str:
        """Read a file relative to the module root."""
        return _safe(path).read_text()

    @tool
    def write_report(path: str, content: str) -> str:
        """Write the summary report."""
        target = _safe(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return f"wrote {target}"

    @tool
    def run_command(command: str, cwd: str = "") -> str:
        """Run a command without a shell and report its exit code.

        Local verify commands only. No shell is used, so pipes and
        redirects are literal arguments, never interpreted. The result
        always starts with the exit code, so a trace reader can tell a
        clean run from a failed one without parsing output text.
        """
        try:
            args = _split_command(command)
        except ValueError as exc:
            return f"exit_code: 2\ncould not parse command: {exc}"
        if not args:
            return "exit_code: 2\nempty command"
        try:
            proc = subprocess.run(
                args,
                cwd=cwd or None,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except FileNotFoundError as exc:
            return f"exit_code: 127\ncommand not found: {exc}"
        except subprocess.TimeoutExpired:
            return "exit_code: 124\ncommand timed out after 120s"
        out = (proc.stdout + proc.stderr).strip()
        if out:
            return f"exit_code: {proc.returncode}\n{out}"
        return f"exit_code: {proc.returncode}"

    return {
        "list_files": list_files,
        "read_file": read_file,
        "write_report": write_report,
        "run_command": run_command,
    }


def _module_files(root: str | Path, module_dir: str, skip_init: bool) -> list[str]:
    files = sorted(
        f
        for f in Path(root, module_dir).iterdir()
        if f.is_file() and f.suffix == ".py"
    )
    if skip_init:
        files = [f for f in files if f.name != "__init__.py"]
    return [f"{module_dir}/{f.name}" for f in files]


def explorer_plan(
    task: str, module_dir: str, procedure: str, root: str | Path
) -> list[tuple[str, dict]]:
    """Baseline: no memory, so explore broadly and double-check."""
    calls: list[tuple[str, dict]] = [("list_files", {"directory": module_dir})]
    for rel in _module_files(root, module_dir, skip_init=False):
        calls.append(("read_file", {"path": rel}))
    first = _module_files(root, module_dir, skip_init=False)[:1]
    for rel in first:
        # the classic LLM dither: re-read the first file "to be sure"
        calls.append(("read_file", {"path": rel}))
    calls.append(("write_report", {"path": f"reports/{module_dir}.md", "content": ""}))
    return calls


def procedure_follower_plan(
    task: str, module_dir: str, procedure: str, root: str | Path
) -> list[tuple[str, dict]]:
    """Treatment: trust the recalled procedure, read each file once."""
    calls: list[tuple[str, dict]] = [("list_files", {"directory": module_dir})]
    for rel in _module_files(root, module_dir, skip_init=True):
        calls.append(("read_file", {"path": rel}))
    calls.append(("write_report", {"path": f"reports/{module_dir}.md", "content": ""}))
    return calls


def _summarize_source(name: str, source: str) -> list[str]:
    lines = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [f"- {name}: (unparseable)"]
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node)
            first = doc.strip().splitlines()[0] if doc else "no docstring"
            lines.append(f"- {node.name}: {first}")
    return lines or [f"- {name}: (no functions)"]


def run_agent(
    task: str,
    module_dir: str,
    root: str | Path,
    policy: str | Callable = "explorer",
    tracer: Any = None,  # noqa: ANN401
    recall_fn: Callable[[str], RecallResult] | None = None,
    procedure_text: str = "",
    verify_command: str | None = None,
    verify_cwd: str | None = None,
) -> dict[str, Any]:
    """Run the summarize-module agent once. Returns the final graph state.

    When verify_command is given, it runs as the last tool call, after
    the report is written, so the trace ends with a clean execute step
    that proves the run worked. Memorable refuses traces that end
    without such a step, so a plain summarize run can never be stored.
    """
    tools = make_tools(root)
    callbacks = {"callbacks": [tracer]} if tracer is not None else {}
    recall_node = build_recall_node(recall_fn=recall_fn)

    if isinstance(policy, str):
        planner = {"explorer": explorer_plan,
                   "procedure_follower": procedure_follower_plan}[policy]
    else:
        planner = policy

    def execute(state: dict[str, Any]) -> dict[str, Any]:
        procedure = state.get("memorable_procedure", "") or procedure_text
        planned = planner(task, module_dir, procedure, root)
        read_results: dict[str, str] = {}
        # pass 1: reads (the report needs their contents)
        for name, kwargs in planned:
            if name == "write_report":
                continue
            result = tools[name].invoke(kwargs, config=callbacks)
            if name == "read_file":
                read_results[kwargs["path"]] = result
        # pass 2: build and write the report from what was read
        lines = ["# Module summary", ""]
        for rel in sorted(read_results):
            lines.append(f"## {rel}")
            lines.extend(_summarize_source(rel, read_results[rel]))
            lines.append("")
        report_rel = f"reports/{module_dir}.md"
        tools["write_report"].invoke(
            {"path": report_rel, "content": "\n".join(lines)}, config=callbacks
        )
        state: dict[str, Any] = {
            "report_path": report_rel,
            "functions_found": len(read_results),
        }
        if verify_command is not None:
            # Last tool call on purpose: Memorable only stores a trace
            # that ends with a clean execute-class command, so the
            # verify command is what proves this run worked.
            state["verify_result"] = tools["run_command"].invoke(
                {"command": verify_command, "cwd": verify_cwd or ""},
                config=callbacks,
            )
        return state

    graph = StateGraph(AgentState)
    graph.add_node("memorable_recall", recall_node)
    graph.add_node("execute", execute)
    graph.set_entry_point("memorable_recall")
    graph.add_edge("memorable_recall", "execute")
    graph.add_edge("execute", END)
    app = graph.compile()
    if tracer is not None and tracer.prompt is None:
        tracer.prompt = task
    config = {"callbacks": [tracer]} if tracer is not None else {}
    return app.invoke({"task": task}, config=config)
