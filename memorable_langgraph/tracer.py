"""Capture a LangGraph agent's tool calls as a Memorable trace.

Memorable's universal entry point is `memorable ingest <trace.json>`, where the
trace is plain JSON shaped like::

    {
        "session_id": "run-001",
        "workflow_id": "summarize-module",   # optional
        "prompt": "Summarize the utils module.",  # optional
        "tool_calls": [
            {"name": "read_file", "input": {"path": "x.py"}, "result": "..."},
        ],
    }

This tracer records that shape, as Memorable's docs describe it for
`memorable ingest`. The shape was not validated against a live
extraction run in this repo, so the hand-off is unproven until an
ingest succeeds.
"""

from __future__ import annotations

import ast
import json
import subprocess
import time
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.tracers import BaseTracer


def _parse_tool_input(input_str: str) -> dict[str, Any]:
    """Tool inputs arrive as JSON or as a Python repr; parse either."""
    if not input_str:
        return {}
    try:
        parsed = json.loads(input_str)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(input_str)
        except (ValueError, SyntaxError):
            return {"_raw": input_str}
    return parsed if isinstance(parsed, dict) else {"_raw": input_str}


class MemorableTracer(BaseTracer):
    """A LangChain tracer that writes Memorable-shaped trace JSON files."""

    def __init__(self, trace_dir: str | Path, session_id: str | None = None) -> None:
        super().__init__()
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or f"langgraph-{int(time.time() * 1000)}"
        self.workflow_id: str | None = None
        self.prompt: str | None = None
        self.tool_calls: list[dict[str, Any]] = []
        self._pending: dict[str, dict[str, Any]] = {}

    # -- BaseTracer hooks -------------------------------------------------
    def _persist_run(self, run: Any) -> None:  # noqa: ANN401
        return None

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        name = (serialized or {}).get("name", "unknown_tool")
        tool_input = _parse_tool_input(input_str)
        self._pending[str(run_id)] = {
            "name": name,
            "input": tool_input,
            "started_at": time.time(),
        }

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        pending = self._pending.pop(str(run_id), None)
        if pending is None:
            return
        result = output if isinstance(output, str) else json.dumps(output, default=str)
        self.tool_calls.append(
            {
                "name": pending["name"],
                "input": pending["input"],
                "result": result[:4000],
            }
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        # Without this hook a failed tool call vanishes from the trace:
        # on_tool_start recorded it, on_tool_end never fires, and the
        # pending entry leaks. Record the failure instead, so the trace
        # shows the procedure as it actually ran.
        pending = self._pending.pop(str(run_id), None)
        if pending is None:
            return
        self.tool_calls.append(
            {
                "name": pending["name"],
                "input": pending["input"],
                "result": f"error: {str(error)[:4000]}",
            }
        )

    # -- trace file --------------------------------------------------------
    def to_trace_dict(self) -> dict[str, Any]:
        trace: dict[str, Any] = {
            "session_id": self.session_id,
            "tool_calls": self.tool_calls,
        }
        if self.workflow_id:
            trace["workflow_id"] = self.workflow_id
        if self.prompt:
            trace["prompt"] = self.prompt
        return trace

    def write_trace(self, filename: str | None = None) -> Path:
        path = self.trace_dir / (filename or f"{self.session_id}.json")
        path.write_text(json.dumps(self.to_trace_dict(), indent=2))
        return path

    # -- hand-off to the memorable CLI -------------------------------------
    def ingest(self, trace_path: str | Path) -> tuple[bool, str]:
        """Pipe a trace file into `memorable ingest`.

        Returns (ok, message). When the extraction API has no credentials
        (needs `memorable login`), ok is False and the message says so plainly
        instead of pretending the procedure was stored.
        """
        try:
            proc = subprocess.run(
                ["memorable", "ingest", str(trace_path)],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except FileNotFoundError:
            return False, "memorable CLI not found on PATH"
        out = (proc.stdout + proc.stderr).strip()
        if proc.returncode == 0:
            return True, out or "ingested"
        if "no API credentials" in out or "memorable login" in out:
            return (
                False,
                "blocked: extraction API needs credentials; "
                "run `memorable login` (browser approval)",
            )
        return False, out or f"ingest failed (exit {proc.returncode})"
