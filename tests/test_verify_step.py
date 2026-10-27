import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memorable_langgraph.agent import make_tools, run_agent
from memorable_langgraph.tracer import MemorableTracer

CORPUS = Path(__file__).resolve().parent.parent / "benchmark" / "corpus"


def _copy_corpus(tmp_path):
    dest = tmp_path / "corpus"
    shutil.copytree(CORPUS, dest)
    return dest


def test_run_command_reports_clean_exit(tmp_path):
    tools = make_tools(tmp_path)
    result = tools["run_command"].invoke(
        {"command": f'{sys.executable} -c "print(42)"', "cwd": str(tmp_path)}
    )
    assert result.startswith("exit_code: 0")
    assert "42" in result


def test_run_command_reports_failed_exit_honestly(tmp_path):
    tools = make_tools(tmp_path)
    result = tools["run_command"].invoke(
        {"command": f'{sys.executable} -c "import sys; sys.exit(3)"',
         "cwd": str(tmp_path)}
    )
    assert result.startswith("exit_code: 3")


def test_run_command_does_not_use_a_shell(tmp_path):
    tools = make_tools(tmp_path)
    result = tools["run_command"].invoke(
        {"command": "echo hi | cat", "cwd": str(tmp_path)}
    )
    # A shell would pipe and print only "hi". Without one, the pipe
    # character is a literal argument to echo.
    assert result.startswith("exit_code: 0")
    assert "hi | cat" in result


def test_verify_step_is_last_tracer_step(tmp_path):
    root = _copy_corpus(tmp_path)
    tracer = MemorableTracer(tmp_path, session_id="verify-test")
    state = run_agent(
        "Summarize the auth module.",
        module_dir="auth",
        root=root,
        policy="explorer",
        tracer=tracer,
        verify_command=f'{sys.executable} -c "print(1)"',
        verify_cwd=str(tmp_path),
    )
    assert state["functions_found"] > 0
    assert state["verify_result"].startswith("exit_code: 0")
    last = tracer.tool_calls[-1]
    assert last["name"] == "run_command"
    assert last["result"].startswith("exit_code: 0")
    path = tracer.write_trace()
    import json

    trace = json.loads(path.read_text())
    assert trace["tool_calls"][-1]["name"] == "run_command"


def test_no_verify_command_means_no_run_command_step(tmp_path):
    root = _copy_corpus(tmp_path)
    tracer = MemorableTracer(tmp_path, session_id="no-verify-test")
    run_agent(
        "Summarize the auth module.",
        module_dir="auth",
        root=root,
        policy="explorer",
        tracer=tracer,
    )
    names = [c["name"] for c in tracer.tool_calls]
    assert "run_command" not in names
    assert names[-1] == "write_report"
