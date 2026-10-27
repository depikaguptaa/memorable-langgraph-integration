"""The real loop must end every trace with a verify step.

Without a clean execute step at the end, the extraction service
refuses the trace, and the run's ingest counts say nothing. The
verify step goes to both arms equally, so the comparison stays fair.
The agent, tracer, and recall surfaces are stubbed; the corpus
directories on disk decide the module list.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import benchmark.run_real_loop as real_loop


class _FakeTracer:
    def __init__(self, trace_dir, session_id):
        self.tool_calls = [{"name": "list_files"}, {"name": "run_command"}]
        self.workflow_id = ""

    def write_trace(self):
        return Path("/tmp/fake-trace.json")

    def ingest(self, path):
        return True, "ok"


def _run_main(monkeypatch, tmp_path, extra_argv):
    calls = []

    def fake_run_agent(*args, **kwargs):
        calls.append(kwargs)
        return {"report_path": "reports/x.md"}

    monkeypatch.setattr(real_loop, "_stored_count", lambda: 1)
    monkeypatch.setattr(real_loop, "run_agent", fake_run_agent)
    monkeypatch.setattr(real_loop, "MemorableTracer", _FakeTracer)
    monkeypatch.setattr(
        real_loop, "recall_procedure",
        lambda t: real_loop.RecallResult(False, "", "none"),
    )
    out = tmp_path / "out.json"
    argv = [
        "run_real_loop.py", "--allow-scripted",
        "--trace-dir", str(tmp_path / "traces"),
        "--out", str(out), *extra_argv,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert real_loop.main() == 0
    return calls, json.loads(out.read_text())


def test_both_arms_get_default_verify_step(monkeypatch, tmp_path):
    calls, results = _run_main(monkeypatch, tmp_path, [])
    assert len(calls) == 12
    repo_root = str(Path(real_loop.__file__).resolve().parent.parent)
    for kwargs in calls:
        assert kwargs["verify_command"] == (
            "python -m pytest tests/test_integration.py -q"
        )
        assert kwargs["verify_cwd"] == repo_root
    assert results["ingest_ok_count"] == 12
    assert results["fixture_used"] is False


def test_verify_command_override_reaches_both_arms(monkeypatch, tmp_path):
    calls, _ = _run_main(
        monkeypatch, tmp_path,
        ["--verify-command", "echo ok", "--verify-cwd", "/tmp"],
    )
    assert len(calls) == 12
    for kwargs in calls:
        assert kwargs["verify_command"] == "echo ok"
        assert kwargs["verify_cwd"] == "/tmp"
