# yc-memorable  - LangGraph integration for Memorable (YC S27)

Custom harness integration for Memorable's procedural memory, built as the
proof-of-work for the YC bypass play. Memorable's YC page explicitly asks for
"custom harness and framework integrations", so this is that.

## What it is

`memorable_langgraph/`  - a small Python package:

- `recall.py`  - recall a stored procedure at task start. Primary path is
  Memorable's read-only MCP server (`memorable mcp` over stdio, five tools,
  all readOnlyHint); falls back to the `memorable recall` CLI. The recalled
  procedure is injected into agent context wrapped in a guardrail marker, as
  reference data, mirroring Memorable's own `show` rendering.
  `build_recall_node()` returns a LangGraph node implementing the
  recall-at-start flow.
- `tracer.py`  - `MemorableTracer`, a LangChain `BaseTracer` that captures
  tool calls into the exact JSON shape `memorable ingest` validates
  (`{session_id, workflow_id?, prompt?, tool_calls: [{name, input, result?}]}`),
  writes trace files, and can pipe them to `memorable ingest` (surfacing the
  login block honestly instead of pretending the store worked).
- `agent.py`  - demo LangGraph agent for a repeated task (summarize a Python
  module), wired as `memorable_recall -> execute -> END`, with a swappable
  planner: `explorer` (no-memory baseline), `procedure_follower` (treatment),
  or any LangChain chat model for a real LLM rerun.

`benchmark/`  - before/after harness over 6 sample modules plus a real
recall-latency measurement. `tests/`  - 8 pytest tests, all passing.

## Verified results (run 2026-10-23)

- Tests: 8/8 pass (`pytest tests/`).
- Benchmark: baseline 33 tool calls vs treatment 26 across 6 repeated tasks =
  **21.2% fewer tool calls** with the recalled procedure; 12.7% lower wall
  time on scripted planners.
- Recall latency, measured for real: `memorable recall` miss path, n=10,
  mean 132ms (their docs claim ~60ms in-process prompt overhead; mine includes
  CLI process spawn).
- `memorable doctor`: install verified; MCP server lists all five read-only
  tools and `memorable_recall` answers correctly on an empty store.
- Tracer output passes `memorable ingest` schema validation (malformed input
  is rejected with the schema error; ours proceeds to the credential check).

## Honest gaps (stated in the outreach too)

1. `memorable ingest` needs `memorable login` with a dashboard key from
   memorable.sh/dash. No key in this environment, so no real procedure was
   ever stored; the treatment arm used a fixture procedure in Memorable's
   `show` render format. Getting the key is a 2-minute human step, then the
   full loop (ingest -> recall -> inject) runs for real.
2. No LLM API key in this environment, so the planners are scripted stand-ins
   for LLM explore-vs-follow behavior. The agent accepts any LangChain chat
   model, so the benchmark reruns with real numbers once a key exists.

## What Depika needs to do

1. Grab a key at memorable.sh/dash (Environments, then Connect, or Account,
   then New key for an agent) and run `echo '<key>' | memorable login --paste`.
2. Create the GitHub repo herself and share the URL (private or public, her call).
3. No video: she decided 2026-10-27 to record no more videos. Offer a live walkthrough instead. LOOM_SCRIPT.md is superseded.
4. Send the email (drafts in OUTREACH.md).

## Layout

- `memorable_langgraph/`  - the integration package
- `benchmark/`  - corpus, `run_benchmark.py`, traces, `results.json`
- `tests/`  - pytest suite
- `OUTREACH.md`, `LOOM_SCRIPT.md`  - local only, never pushed
- `.venv/`  - local virtualenv (langgraph, langchain-core, pytest)
