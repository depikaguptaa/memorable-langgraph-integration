# LangGraph integration for Memorable

Memorable asks for custom harness and framework integrations on its YC page. This is one, built and measured: a LangGraph node that recalls a stored procedure at task start, and a tracer that captures an agent's tool calls as ingest-ready JSON.

## What you get

- `memorable_langgraph/recall.py`: `build_recall_node()` returns a LangGraph node. It recalls at task start through Memorable's read-only MCP server (`memorable mcp`, five tools, all read-only), with the `memorable recall` CLI as fallback. The MCP `memorable_recall` tool returns titles and slugs, not steps, so a hit is followed by `memorable_show` for the full procedure before injection, wrapped in a guardrail marker, as reference data, not instructions. `recall_chain()` uses `memorable chain --json` plus `show` for multi-step tasks that one procedure cannot cover. A miss returns an empty string, so the agent just proceeds without memory. Recall failures never raise out of the node by default (silent mode, like Memorable's own hook), so a recall failure can never break a prompt.
- `memorable_langgraph/tracer.py`: `MemorableTracer`, a LangChain tracer that records every tool call as `{name, input, result}` in the JSON shape Memorable's docs describe for `memorable ingest`. The shape was validated on 2026-10-27: a trace ending in a clean execute step was stored as a procedure, and recall, show, and chain all hit it. The key detail: ingest forwards a step's result only when it is an object with a numeric exit_code, so execute-step results are stored as exit_code objects; a plain string result is dropped by ingest and the service refuses the trace. `tracer.write_trace()` writes the file. `tracer.ingest(path)` pipes it to `memorable ingest` and fails loudly if the extraction API has no credentials, instead of pretending the procedure was stored.
- `memorable_langgraph/agent.py`: a demo LangGraph agent for a repeated task, "summarize a Python module". Graph: `memorable_recall -> execute -> END`. The planner is swappable: a scripted explorer (baseline), a scripted procedure follower (treatment), or any LangChain chat model for a real LLM rerun. It also has a `run_command` tool (no shell, result always starts with the exit code) and an optional verify step: `run_agent(..., verify_command="...")` runs the command as the last tool call, so the trace ends with proof the run worked.

## Measured results

Deterministic fixture numbers: scripted planners, fixture procedure, no LLM key. Repeated task, 6 modules in `benchmark/corpus`: summarize the module, list each function with a one-line description, write the report.

| | Tool calls |
|---|---|
| Baseline, no memory (explorer) | 33 |
| With recalled procedure (follower) | 26 |
| Difference | 21.2% fewer |

Recall latency, measured for real: 10 timed `memorable recall` calls against the empty local store, mean about 130 ms. That includes the CLI process spawn. Memorable's docs claim about 60 ms in-process prompt overhead; this number is not that, it is the CLI path.

The first real run with a chat-model planner happened 2026-10-27 (see Honest limits). Its totals are not in this repo; they live in the run's results file on the logged-in machine.

Tests: 35/35 passing (`pytest tests/`), including recall-then-show, chain, MCP miss wording, silent-failure, verify-step, exit_code-object, slug-format and full-slug recall, planner-validation, spaced executable paths, and real-loop verify-wiring tests. Full numbers and per-module traces are in `benchmark/results.json` and `benchmark/traces/`.

## Try it

```bash
pip install -r requirements.txt
curl -fsSL https://memorable.sh/install.sh | sh   # Memorable's own CLI installer, from memorable.sh
python -m pytest tests/
python benchmark/run_benchmark.py
python benchmark/run_verified_trace.py
npx memorable-cli ingest benchmark/traces/bench-verified-auth.json
```

## Use it in your own agent

```python
from memorable_langgraph import MemorableTracer, build_recall_node

graph.add_node("memorable_recall", build_recall_node())
graph.set_entry_point("memorable_recall")

tracer = MemorableTracer("traces", session_id="run-001")
# run your graph with callbacks=[tracer]
path = tracer.write_trace()
ok, msg = tracer.ingest(path)  # needs `memorable login` first
```

## Honest limits

Read these before trusting the numbers above.

1. A real procedure is now stored, and the hit path has been exercised against it on a logged-in machine: recall scored 0.728, chain matched with coverage 1, and `show` renders the verify command and the execute step. The caveat is that the rendered procedure shows only the execute step; the read and write steps from the trace do not appear, and chain reads/writes are empty. What recall returns today is the verified command, not the full read/write plan.
2. The first real run happened 2026-10-27 with a LangChain planner (openai:gpt-4o-mini, zero fallbacks to the scripted explorer). Its ingests were all refused because the traces ended in a write with no execute step; this version ends every run with a verify step so the traces can be admitted. Both arms use the same planner; only the recalled procedure differs. Wall time is not claimed: it is mostly model latency. Tool calls are the measured effect.
3. The demo task is deliberately narrow: one repeated task, six small modules. It shows the integration works and that injecting a procedure removes exploration calls. It is not a general claim about Memorable on arbitrary tasks.

## The real loop

`benchmark/run_real_loop.py` runs the loop for real, and refuses to fake it. It checks that procedures are actually stored, runs the agent once without memory and once with live recall, ingests each trace, and writes results with provenance: planner type, stored procedure count, and whether each ingest succeeded. With no stored procedures or no chat model key it stops and says so, instead of writing benchmark numbers. Run it where you are logged in:

```bash
memorable login
python benchmark/run_real_loop.py --planner-model openai:gpt-4o-mini
# or, to smoke-test the wiring with scripted planners, labeled as scripted:
python benchmark/run_real_loop.py --allow-scripted
```

On a Mac with Codex, `record` does not work: it needs gbrain session capture, and Codex has none, so it reports no session receipt. The path that works is the verified trace: run `benchmark/run_verified_trace.py`, ingest the trace it writes, then check `memorable list`. That path stored the first procedure on 2026-10-27, and recall, show, and chain all hit it. `benchmark/run_real_loop.py` now ends every trace with the same verify step.

Built by Depika Gupta, AI and backend engineer. I build LangChain multi-agent systems, RAG, and LLM observability at a remote US startup. github.com/depikaguptaa
