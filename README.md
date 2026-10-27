# LangGraph integration for Memorable

Memorable asks for custom harness and framework integrations on its YC page. This is one, built and measured: a LangGraph node that recalls a stored procedure at task start, and a tracer that captures an agent's tool calls as ingest-ready JSON.

## What you get

- `memorable_langgraph/recall.py`: `build_recall_node()` returns a LangGraph node. It recalls at task start through Memorable's read-only MCP server (`memorable mcp`, five tools, all read-only), with the `memorable recall` CLI as fallback. A hit is injected into agent context wrapped in a guardrail marker, as reference data, not instructions. A miss returns an empty string, so the agent just proceeds without memory.
- `memorable_langgraph/tracer.py`: `MemorableTracer`, a LangChain tracer that records every tool call as `{name, input, result}` in the exact JSON shape `memorable ingest` validates. `tracer.write_trace()` writes the file. `tracer.ingest(path)` pipes it to `memorable ingest` and fails loudly if the extraction API has no credentials, instead of pretending the procedure was stored.
- `memorable_langgraph/agent.py`: a demo LangGraph agent for a repeated task, "summarize a Python module". Graph: `memorable_recall -> execute -> END`. The planner is swappable: a scripted explorer (baseline), a scripted procedure follower (treatment), or any LangChain chat model for a real LLM rerun.

## Measured results

Repeated task, 6 modules in `benchmark/corpus`: summarize the module, list each function with a one-line description, write the report.

| | Tool calls |
|---|---|
| Baseline, no memory (explorer) | 33 |
| With recalled procedure (follower) | 26 |
| Difference | 21.2% fewer |

Recall latency, measured for real: 10 timed `memorable recall` calls against the empty local store, mean about 130 ms. That includes the CLI process spawn. Memorable's docs claim about 60 ms in-process prompt overhead; this number is not that, it is the CLI path.

Tests: 8/8 passing (`pytest tests/`). Full numbers and per-module traces are in `benchmark/results.json` and `benchmark/traces/`.

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

1. No real procedure was stored in this run. `memorable ingest` needs `memorable login` with a dashboard key, and no key was available here. The treatment arm uses a fixture procedure in Memorable's `show` render format, standing in for what the extraction API would return. The recall path is real (MCP server and CLI were exercised, including the miss path on an empty store). The store was empty.
2. No LLM key was available, so the planners are scripted stand-ins for an LLM planner's explore vs follow behavior. The agent accepts any LangChain chat model, so the benchmark reruns with real numbers once a key exists. Wall time is not claimed: with scripted planners it is mostly noise. Tool calls are the measured effect.
3. The demo task is deliberately narrow: one repeated task, six small modules. It shows the integration works and that injecting a procedure removes exploration calls. It is not a general claim about Memorable on arbitrary tasks.

## What is next

The useful next step is the real loop, not more fixtures: log in, ingest a real trace, recall it on a repeat task, and rerun the benchmark with a real LLM planner. That needs a dashboard key and a chat model key, both 2-minute human steps. I am happy to run it end to end on a real workflow as a trial, or walk through the code live.

Built by Depika Gupta, AI and backend engineer. I build LangChain multi-agent systems, RAG, and LLM observability at a remote US startup. github.com/depikaguptaa
