"""Tests for recall-then-show, chain, miss wordings, and silent failure.

The MCP and CLI surfaces are stubbed here, because no stored procedure
exists in this environment. The stubbed outputs use the real wordings
observed from `memorable` on an empty store.
"""

import json
import subprocess

import pytest

from memorable_langgraph.recall import (
    RecallResult,
    _is_miss,
    build_recall_node,
    recall_chain,
    recall_procedure,
)


class _Proc:
    def __init__(self, stdout=""):
        self.stdout = stdout
        self.stderr = ""
        self.returncode = 0


def _mcp_response(text):
    return json.dumps(
        {"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": text}]}}
    )


def _stub_mcp(monkeypatch, recall_text, show_text):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        request = json.loads(kwargs.get("input") or "{}")
        name = request.get("params", {}).get("name")
        if name == "memorable_recall":
            return _Proc(_mcp_response(recall_text))
        if name == "memorable_show":
            return _Proc(_mcp_response(show_text))
        return _Proc(_mcp_response(""))

    monkeypatch.setattr("memorable_langgraph.recall.shutil.which", lambda _: "/bin/memorable")
    monkeypatch.setattr("memorable_langgraph.recall.subprocess.run", fake_run)
    return calls


def test_mcp_miss_text_is_a_miss_not_a_hit(monkeypatch):
    # Regression: the first version read this text as a hit, because it
    # only checked for "no matching".
    assert _is_miss("No stored procedure matches that task.")
    calls = _stub_mcp(
        monkeypatch,
        "No stored procedure matches that task.",
        "1. do the thing",
    )
    result = recall_procedure("summarize the auth module", via="mcp")
    assert result.found is False
    # show must never be called on a miss
    assert len(calls) == 1


def test_mcp_recall_then_show_injects_steps_not_slugs(monkeypatch):
    calls = _stub_mcp(
        monkeypatch,
        "summarize-python-module\nSummarize a Python module",
        "1. list_files on the module directory.\n2. read_file each file once.",
    )
    result = recall_procedure("summarize the auth module", via="mcp")
    assert result.found is True
    assert result.source == "mcp-show"
    assert result.slug == "summarize-python-module"
    assert "read_file each file once" in result.text
    assert len(calls) == 2


def test_cli_recall_then_show(monkeypatch):
    def fake_run(args, **kwargs):
        if args == ["memorable", "recall", "summarize the auth module"]:
            return _Proc("summarize-python-module")
        if args == ["memorable", "show", "summarize-python-module"]:
            return _Proc("1. list_files.\n2. read_file each file once.")
        raise AssertionError(args)

    monkeypatch.setattr("memorable_langgraph.recall.shutil.which", lambda _: "/bin/memorable")
    monkeypatch.setattr("memorable_langgraph.recall.subprocess.run", fake_run)
    result = recall_procedure("summarize the auth module", via="cli")
    assert result.found is True
    assert result.source == "cli-show"
    assert "read_file each file once" in result.text


def test_chain_assembles_shown_procedures_in_order(monkeypatch):
    def fake_run(args, **kwargs):
        if args[:2] == ["memorable", "chain"]:
            if "--json" in args:
                return _Proc(json.dumps({
                    "segments": ["summarize the auth module"],
                    "items": [{"slug": "summarize-python-module"}],
                    "coverage": 80, "score": 1,
                }))
            return _Proc("no chain.")
        if args == ["memorable", "show", "summarize-python-module"]:
            return _Proc("1. list_files.\n2. read_file each file once.")
        raise AssertionError(args)

    monkeypatch.setattr("memorable_langgraph.recall.shutil.which", lambda _: "/bin/memorable")
    monkeypatch.setattr("memorable_langgraph.recall.subprocess.run", fake_run)
    result = recall_chain("summarize the auth module")
    assert result.found is True
    assert result.source == "cli-chain"
    assert "read_file each file once" in result.text


def test_chain_empty_store_is_a_miss(monkeypatch):
    def fake_run(args, **kwargs):
        if "--json" in args:
            return _Proc(json.dumps({"segments": ["x"], "items": [], "coverage": 0, "score": 0}))
        return _Proc("no chain.")

    monkeypatch.setattr("memorable_langgraph.recall.shutil.which", lambda _: "/bin/memorable")
    monkeypatch.setattr("memorable_langgraph.recall.subprocess.run", fake_run)
    result = recall_chain("summarize the auth module")
    assert result.found is False


def test_node_silent_failure_writes_empty(monkeypatch):
    def boom(task):
        raise RuntimeError("recall exploded")

    node = build_recall_node(recall_fn=boom, silent=True)
    assert node({"task": "summarize the auth module"}) == {"memorable_procedure": ""}
    with pytest.raises(RuntimeError):
        build_recall_node(recall_fn=boom, silent=False)({"task": "x"})


def test_node_still_injects_stubbed_procedure():
    node = build_recall_node(
        recall_fn=lambda t: RecallResult(True, "1. read a", "fixture")
    )
    state = node({"task": "summarize auth"})
    assert "1. read a" in state["memorable_procedure"]
