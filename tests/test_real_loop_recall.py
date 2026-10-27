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


def test_miss_check_ignores_phrase_inside_procedure_text():
    # A procedure that mentions 404 handling is not a miss.
    assert not _is_miss("Steps: handle 404 not found errors by retrying.")
    assert _is_miss("No stored procedure matches that task.")
    assert _is_miss("no matching procedures.")
    assert _is_miss("")


def test_node_handles_none_result_and_missing_show_text(monkeypatch):
    # A recall function returning None must not crash the node, even
    # with silent=False, and format_for_injection(None) is empty.
    from memorable_langgraph.recall import format_for_injection

    assert format_for_injection(None) == ""
    node = build_recall_node(recall_fn=lambda t: None, silent=False)
    assert node({"task": "summarize the auth module"}) == {"memorable_procedure": ""}


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


def test_mcp_recall_uses_full_procedures_slug(monkeypatch):
    full = "procedures/92ef4af3-add-auth-module-structure-to-documentation"
    _stub_mcp(
        monkeypatch,
        f"{full}\nAdd auth module structure to documentation",
        "1. list_files.\n2. read_file each file once.",
    )
    result = recall_procedure("summarize the auth module", via="mcp")
    assert result.found is True
    assert result.source == "mcp-show"
    assert result.slug == full
    assert "read_file each file once" in result.text


def test_cli_recall_uses_full_procedures_slug(monkeypatch):
    full = "procedures/92ef4af3-add-auth-module-structure-to-documentation"

    def fake_run(args, **kwargs):
        if args == ["memorable", "recall", "summarize the auth module"]:
            return _Proc(f"0.728  {full}  [lexical,semantic]")
        if args == ["memorable", "show", full]:
            return _Proc("1. list_files.\n2. read_file each file once.")
        raise AssertionError(args)

    monkeypatch.setattr("memorable_langgraph.recall.shutil.which", lambda _: "/bin/memorable")
    monkeypatch.setattr("memorable_langgraph.recall.subprocess.run", fake_run)
    result = recall_procedure("summarize the auth module", via="cli")
    assert result.found is True
    assert result.source == "cli-show"
    assert result.slug == full


def test_node_still_injects_stubbed_procedure():
    node = build_recall_node(
        recall_fn=lambda t: RecallResult(True, "1. read a", "fixture")
    )
    state = node({"task": "summarize auth"})
    assert "1. read a" in state["memorable_procedure"]


def test_extract_slugs_prefers_full_procedures_slug():
    from memorable_langgraph.recall import _extract_slugs

    cli_line = (
        "0.728  procedures/92ef4af3-add-auth-module-structure-to-documentation"
        "  [lexical,semantic]"
    )
    assert _extract_slugs(cli_line) == [
        "procedures/92ef4af3-add-auth-module-structure-to-documentation"
    ]
    mcp_text = (
        "procedures/92ef4af3-add-auth-module-structure-to-documentation\n"
        "Add auth module structure to documentation"
    )
    assert _extract_slugs(mcp_text)[0] == (
        "procedures/92ef4af3-add-auth-module-structure-to-documentation"
    )
    # The old bare pattern still works for plain slugs.
    assert _extract_slugs("summarize-python-module") == ["summarize-python-module"]


def test_recall_then_show_uses_full_slug():
    from memorable_langgraph.recall import _recall_then_show

    asked = []

    def show_fn(slug):
        asked.append(slug)
        return "1. list_files.\n2. read_file each file once."

    result = _recall_then_show(
        "0.728  procedures/92ef4af3-add-auth-module-structure-to-documentation",
        show_fn,
        "cli-show",
        "cli-recall-only",
    )
    assert asked == [
        "procedures/92ef4af3-add-auth-module-structure-to-documentation"
    ]
    assert result.found is True
    assert result.source == "cli-show"
    assert result.slug == (
        "procedures/92ef4af3-add-auth-module-structure-to-documentation"
    )
