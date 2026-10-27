import json
import os
import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# the memorable CLI lives in ~/.local/bin in this environment
os.environ["PATH"] = os.path.expanduser("~/.local/bin") + os.pathsep + os.environ["PATH"]

from memorable_langgraph.agent import run_agent
from memorable_langgraph.recall import (
    GUARDRAIL,
    RecallResult,
    build_recall_node,
    format_for_injection,
)
from memorable_langgraph.tracer import MemorableTracer

CORPUS = Path(__file__).resolve().parent.parent / "benchmark" / "corpus"


def _fake_run(name, tool_input):
    class R:
        pass

    r = R()
    r.id = uuid4()
    return r


def test_tracer_writes_memorable_schema(tmp_path):
    tracer = MemorableTracer(tmp_path, session_id="test-1")
    tracer.workflow_id = "wf"
    tracer.prompt = "do the thing"
    run = _fake_run("read_file", {"path": "x.py"})
    tracer.on_tool_start({"name": "read_file"}, json.dumps({"path": "x.py"}),
                         run_id=run.id)
    tracer.on_tool_end("file contents", run_id=run.id)
    path = tracer.write_trace()
    trace = json.loads(path.read_text())
    assert trace["session_id"] == "test-1"
    assert trace["workflow_id"] == "wf"
    assert trace["prompt"] == "do the thing"
    assert trace["tool_calls"] == [
        {"name": "read_file", "input": {"path": "x.py"}, "result": "file contents"}
    ]


def test_tracer_handles_unmatched_tool_end(tmp_path):
    tracer = MemorableTracer(tmp_path)
    tracer.on_tool_end("orphan", run_id=uuid4())  # must not raise
    assert tracer.tool_calls == []


def test_ingest_reports_login_block(tmp_path):
    tracer = MemorableTracer(tmp_path)
    path = tracer.write_trace("empty.json")
    ok, msg = tracer.ingest(path)
    # In this environment there are no extraction-API credentials, so ingest
    # must fail loudly, never silently.
    assert ok is False
    assert "login" in msg.lower()


def test_recall_node_injects_guarded_procedure():
    node = build_recall_node(
        recall_fn=lambda t: RecallResult(True, "1. read a\n2. read b", "fixture")
    )
    state = node({"task": "summarize auth"})
    assert GUARDRAIL in state["memorable_procedure"]
    assert "1. read a" in state["memorable_procedure"]


def test_recall_node_passes_through_on_miss():
    node = build_recall_node(
        recall_fn=lambda t: RecallResult(False, "no matching procedures.", "cli")
    )
    state = node({"task": "summarize auth"})
    assert state["memorable_procedure"] == ""


def test_format_for_injection_empty_on_miss():
    assert format_for_injection(RecallResult(False, "nothing", "cli")) == ""


def test_agent_runs_end_to_end(tmp_path):
    tracer = MemorableTracer(tmp_path, session_id="e2e")
    tracer.workflow_id = "summarize-python-module"
    state = run_agent(
        "Summarize the auth module.",
        module_dir="auth",
        root=CORPUS,
        policy="procedure_follower",
        tracer=tracer,
        recall_fn=lambda t: RecallResult(True, "fixture procedure", "fixture"),
        procedure_text="fixture procedure",
    )
    assert state["report_path"] == "reports/auth.md"
    assert state["memorable_procedure"].startswith("[REFERENCE")
    names = [c["name"] for c in tracer.tool_calls]
    assert names[0] == "list_files"
    assert "write_report" in names
    # follower reads each non-init file exactly once
    reads = [c for c in tracer.tool_calls if c["name"] == "read_file"]
    assert len(reads) == 2
    assert len({c["input"]["path"] for c in reads}) == 2
    report = CORPUS / "reports" / "auth.md"
    assert report.exists()
    assert "hash_password" in report.read_text()


def test_explorer_uses_more_calls_than_follower(tmp_path):
    counts = {}
    for policy in ("explorer", "procedure_follower"):
        tracer = MemorableTracer(tmp_path, session_id=f"cmp-{policy}")
        run_agent(
            "Summarize the queue module.",
            module_dir="queue",
            root=CORPUS,
            policy=policy,
            tracer=tracer,
            recall_fn=lambda t: RecallResult(False, "", "none"),
        )
        counts[policy] = len(tracer.tool_calls)
    assert counts["explorer"] > counts["procedure_follower"]
