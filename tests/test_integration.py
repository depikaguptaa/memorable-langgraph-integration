import json
import os
import shutil
import sys
from pathlib import Path
from uuid import uuid4

import pytest

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
    # This test only means something where the CLI is installed but no
    # login exists (the VM). On a Mac where the binary is missing, or
    # where login is already approved, the block path is not exercised.
    if shutil.which("memorable") is None:
        pytest.skip("memorable CLI not on PATH")
    tracer = MemorableTracer(tmp_path)
    path = tracer.write_trace("empty.json")
    ok, msg = tracer.ingest(path)
    if ok:
        pytest.skip("memorable login is configured here; login-block path not exercised")
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


def test_tracer_records_tool_error(tmp_path):
    tracer = MemorableTracer(tmp_path, session_id="err-1")
    run = _fake_run("read_file", {"path": "missing.py"})
    tracer.on_tool_start({"name": "read_file"}, json.dumps({"path": "missing.py"}),
                         run_id=run.id)
    tracer.on_tool_error(FileNotFoundError("missing.py"), run_id=run.id)
    assert len(tracer.tool_calls) == 1
    assert tracer.tool_calls[0]["name"] == "read_file"
    assert "error" in tracer.tool_calls[0]["result"]


def test_agent_runs_end_to_end(tmp_path):
    # Copy the corpus: the agent writes its report into it, and a test
    # must not rewrite the repo's own benchmark corpus.
    corpus = tmp_path / "corpus"
    shutil.copytree(CORPUS, corpus)
    tracer = MemorableTracer(tmp_path, session_id="e2e")
    tracer.workflow_id = "summarize-python-module"
    state = run_agent(
        "Summarize the auth module.",
        module_dir="auth",
        root=corpus,
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
    report = corpus / "reports" / "auth.md"
    assert report.exists()
    assert "hash_password" in report.read_text()


def test_planner_accepts_fenced_json_and_rejects_escape(tmp_path):
    from benchmark.run_real_loop import _planner_from_model

    class FakeReply:
        def __init__(self, content):
            self.content = content

    class FakeModel:
        def __init__(self, content):
            self.content = content

        def invoke(self, prompt):
            return FakeReply(self.content)

    fenced = FakeModel('```json\n[["list_files", {"directory": "auth"}], ["write_report", {"path": "reports/auth.md", "content": ""}]]\n```')
    planner = _planner_from_model(fenced)
    calls = planner("Summarize the auth module.", "auth", "", CORPUS)
    assert planner.fallbacks["count"] == 0
    assert calls[0][0] == "list_files"

    escaping = FakeModel('[["read_file", {"path": "../secret.txt"}], ["write_report", {"path": "reports/auth.md", "content": ""}]]')
    planner2 = _planner_from_model(escaping)
    calls2 = planner2("Summarize the auth module.", "auth", "", CORPUS)
    assert planner2.fallbacks["count"] == 1
    assert all(c[0] != "read_file" or not c[1]["path"].startswith("..") for c in calls2)


def test_explorer_uses_more_calls_than_follower(tmp_path):
    corpus = tmp_path / "corpus"
    shutil.copytree(CORPUS, corpus)
    counts = {}
    for policy in ("explorer", "procedure_follower"):
        tracer = MemorableTracer(tmp_path, session_id=f"cmp-{policy}")
        run_agent(
            "Summarize the queue module.",
            module_dir="queue",
            root=corpus,
            policy=policy,
            tracer=tracer,
            recall_fn=lambda t: RecallResult(False, "", "none"),
        )
        counts[policy] = len(tracer.tool_calls)
    assert counts["explorer"] > counts["procedure_follower"]
