"""LangGraph integration for Memorable (memorable.sh) procedural memory.

A custom harness integration: recall a stored procedure at task start via
Memorable's read-only MCP server (with CLI fallback), inject it into the
agent's context, and capture the agent's tool calls as a trace in the JSON
shape described in Memorable's docs for `memorable ingest` (validated
against a live extraction run on 2026-10-27).
"""

from .recall import RecallResult, build_recall_node, recall_chain, recall_procedure
from .tracer import MemorableTracer

__all__ = ["MemorableTracer", "RecallResult", "build_recall_node", "recall_chain", "recall_procedure"]
