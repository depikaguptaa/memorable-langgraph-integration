# LangGraph integration for Memorable

Memorable asks for custom harness and framework integrations on its YC page. This is one, built and measured: a LangGraph node that recalls a stored procedure at task start, and a tracer that captures an agent's tool calls as ingest-ready JSON.

## What you get

- `memorable_langgraph/recall.py`: `build_recall_node()` returns a LangGraph node. It recalls at task start through Memorable's read-only MCP server (`memorable mcp`, five tools, all read-only), with the `memorable recall` CLI as fallback. The MCP `memorable_recall` tool returns titles and slugs, not steps, so a hit is followed by `memorable_show` for the full procedure before injection, wrapped in a guardrail marker, as reference data, not instructions. `recall_chain()` uses `memorable chain --json` plus `show` for multi-step tasks that one procedure cannot cover. A miss returns an empty string, so the agent just proceeds without memory. Recall failures never raise out of the node by default (silent mode, like Memorable's own hook), so a recall failure can never break a prompt.
- `memorable_langgraph/tracer.py`: `MemorableTracer`, a LangChain tracer that records every tool call as `{name, input, result}` in the JSON shape Memorable's docs describe for `memorable ingest`. That shape was never validated against a live extraction run in this repo: every ingest attempt here failed for lack of credentials. `tracer.write_trace()` writes the file. `tracer.ingest(path)` pipes it to `memorable ingest` and fails loudly if the extraction API has no credentials, instead of pretending the procedure was stored.
- `memorable_langgraph/agent.py`: a demo LangGraph agent for a repeated task, "summarize a Python module". Graph: `memorable_recall -> execute -> END`. The planner is swappable: a scripted explorer (baseline), a scripted procedure follower (treatment), or any LangChain chat model for a real LLM rerun.

## Measured results

Repeated task, 6 modules in `benchmark/corpus`: summarize the module, list each function with a one-line description, write the report.

| | Tool calls |
|---|---|
| Baseline, no memory (explorer) | 33 |
| With recalled procedure (follower) | 26 |
| Difference | 21.2% fewer |

Recall latency, measured for real: 10 timed `memorable recall` calls against the empty local store, mean about 130 ms. That includes the CLI process spawn. Memorable's docs claim about 60 ms in-process prompt overhead; this number is not that, it is the CLI path.

Tests: 15/15 passing (`pytest tests/`), including recall-then-show, chain, MCP miss wording, and silent-failure tests. Full numbers and per-module traces are in `benchmark/results.json` and `benchmark/traces/`.

## Try it

```bash
pip install -r requirements.txt
curl -fsSL https://memorable.sh/install.sh | sh   # memorable CLI, see install.sh
python -m pytest tests/
python benchmark/run_benchmark.py
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

1. No real procedure was stored in this run. `memorable ingest` needs `memorable login` with a dashboard key, and no key was available here. The treatment arm uses a fixture procedure in Memorable's `show` render format, standing in for what the extraction API would return. Only the miss path was ever exercised, on an empty store, through both MCP and CLI. The hit path, recall then show, was never exercised against a real stored procedure, and the tracer JSON was never accepted by a live extraction run. The store was empty. The first version of this code also had a real bug on that path: it injected MCP recall text directly, and it read the MCP miss text ("No stored procedure matches that task.") as a hit, because it only checked for the words "no matching". The recall code now does recall then show, checks all three miss wordings, and is covered by a regression test for the MCP miss text.
2. No LLM key was available, so the planners are scripted stand-ins for an LLM planner's explore vs follow behavior. The agent accepts any LangChain chat model, so the benchmark reruns with real numbers once a key exists. Wall time is not claimed: with scripted planners it is mostly noise. Tool calls are the measured effect.
3. The demo task is deliberately narrow: one repeated task, six small modules. It shows the integration works and that injecting a procedure removes exploration calls. It is not a general claim about Memorable on arbitrary tasks.

## The real loop

`benchmark/run_real_loop.py` runs the loop for real, and refuses to fake it. It checks that procedures are actually stored, runs the agent once without memory and once with live recall, ingests each trace, and writes results with provenance: planner type, stored procedure count, and whether each ingest succeeded. With no stored procedures or no chat model key it stops and says so, instead of writing benchmark numbers. Run it where you are logged in:

```bash
memorable login
python benchmark/run_real_loop.py --planner-model openai:gpt-4o-mini
# or, to smoke-test the wiring with scripted planners, labeled as scripted:
python benchmark/run_real_loop.py --allow-scripted
```

On a Mac with Codex, the human part is: clone this repo, run the tracer test task in Codex in that folder, then in the same terminal run `npx memorable-cli record` and `npx memorable-cli list`, then repeat a similar task and watch whether recall fires. That first stored procedure is the one this repo has never had.

Built by Depika Gupta, AI and backend engineer. I build LangChain multi-agent systems, RAG, and LLM observability at a remote US startup. github.com/depikaguptaa
