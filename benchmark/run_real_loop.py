"""Run the Memorable loop for real, and refuse to fake it.

Steps, per module in benchmark/corpus:
  1. Run the agent with no recall (miss). Ingest the trace.
  2. Run the agent again with live recall (recall then show, via
     recall_procedure). Ingest that trace too.
  3. Compare tool calls, and record whether each ingest succeeded.

Each run ends with a verify step (the integration tests), so the
trace ends in a clean execute step and the extraction service can
admit it. Without that step every trace is refused, as the first
real run showed.

Provenance is written into the output file: planner type, how many
procedures were stored before the run, and whether each ingest succeeded.
With no stored procedures, or with no chat model key and no
--allow-scripted, the script stops and says so instead of writing
benchmark numbers. Scripted planners are only allowed with
--allow-scripted, and the output is labeled scripted in that case.

Usage, where you are logged in:
  memorable login
  python benchmark/run_real_loop.py --planner-model openai:gpt-4o-mini
  python benchmark/run_real_loop.py --allow-scripted   # smoke test only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memorable_langgraph.agent import run_agent  # noqa: E402
from memorable_langgraph.recall import RecallResult, recall_procedure  # noqa: E402
from memorable_langgraph.tracer import MemorableTracer  # noqa: E402

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus"


def _stored_count() -> int:
    try:
        proc = subprocess.run(
            ["memorable", "list", "--json"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return -1
    try:
        data = json.loads(proc.stdout.strip() or "[]")
    except json.JSONDecodeError:
        return -1
    return len(data) if isinstance(data, list) else -1


def _load_planner(spec: str):
    """Load a LangChain chat model from a 'provider:model' spec.

    Supported providers are the ones LangChain ships as separate packages:
    openai, anthropic, google-genai. The model needs its API key in the
    environment, as usual for that provider.
    """
    provider, _, model = spec.partition(":")
    if not model:
        raise ValueError("--planner-model needs 'provider:model', e.g. openai:gpt-4o-mini")
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model)
    if provider == "google-genai":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model)
    raise ValueError(f"unsupported planner provider: {provider!r}")


def _extract_json_array(text: str) -> str:
    """Pull the JSON array out of a planner reply, fences and prose aside.

    Models commonly wrap the array in a ```json fence or add a sentence
    around it. Rejecting those replies as unusable would silently turn
    the treatment arm into the scripted explorer, so strip the fence
    and take the text from the first "[" to the last "]".
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:])
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    start = stripped.find("[")
    end = stripped.rfind("]")
    if start < 0 or end < start:
        raise ValueError("no JSON array in planner reply")
    return stripped[start : end + 1]


def _planner_from_model(model) -> object:
    """Wrap a chat model as the agent's planner callable.

    The planner asks the model for a JSON list of tool calls, given the
    task, the recalled procedure (possibly empty), and the module's file
    list. Every call is validated: only list_files, read_file, and
    write_report, only paths inside the module directory. An unusable
    reply falls back to the scripted explorer plan, and the fallback is
    counted on the planner so the output can say how often it happened.
    The model decides both arms; the procedure is the only difference
    between them, which is the point of the comparison.
    """
    import json as _json
    from pathlib import Path as _Path

    from memorable_langgraph.agent import explorer_plan

    fallbacks = {"count": 0}

    def planner(task, module_dir, procedure, root):
        files = sorted(
            f"{module_dir}/{f.name}"
            for f in _Path(root, module_dir).iterdir()
            if f.is_file() and f.suffix == ".py"
        )
        prompt = (
            "Plan tool calls to finish this task. Reply with a JSON array "
            "only, each item [tool_name, {arguments}]. Allowed tools: "
            "list_files {directory}, read_file {path}, write_report "
            "{path, content}. The report path is reports/<module>.md. "
            "Read each file you need exactly once unless you have a reason "
            "to re-read it.\n\n"
            f"Task: {task}\nFiles: {', '.join(files)}\n"
            f"Recalled procedure (empty means no memory):\n{procedure or '(none)'}"
        )
        try:
            reply = model.invoke(prompt)
            content = reply.content if hasattr(reply, "content") else str(reply)
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") for part in content
                    if isinstance(part, dict)
                )
            parsed = _json.loads(_extract_json_array(str(content)))
            calls: list[tuple[str, dict]] = []
            for item in parsed:
                name, kwargs = item[0], dict(item[1])
                if name not in ("list_files", "read_file", "write_report"):
                    raise ValueError(f"unknown tool {name!r}")
                if name == "list_files":
                    if str(kwargs.get("directory", "")) != module_dir:
                        raise ValueError(f"unexpected directory: {kwargs.get('directory')!r}")
                if name == "read_file":
                    path = str(kwargs.get("path", ""))
                    if path not in files:
                        # A hallucinated path would crash the run in
                        # read_file; fall back instead of failing.
                        raise ValueError(f"unknown file: {path!r}")
                if name == "write_report":
                    path = str(kwargs.get("path", ""))
                    if path != f"reports/{module_dir}.md":
                        raise ValueError(f"unexpected report path: {path!r}")
                calls.append((name, kwargs))
            if not any(n == "write_report" for n, _ in calls):
                calls.append(("write_report", {"path": f"reports/{module_dir}.md", "content": ""}))
            return calls
        except Exception:
            fallbacks["count"] += 1
            return explorer_plan(task, module_dir, procedure, root)

    planner.fallbacks = fallbacks  # type: ignore[attr-defined]
    return planner


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner-model", default="")
    ap.add_argument("--allow-scripted", action="store_true")
    ap.add_argument("--verify-command",
                    default="python -m pytest tests/test_integration.py -q")
    ap.add_argument("--verify-cwd", default=str(HERE.parent))
    ap.add_argument("--trace-dir", default=str(HERE / "traces"))
    ap.add_argument("--out", default=str(HERE / "real_loop_results.json"))
    args = ap.parse_args()

    stored = _stored_count()
    if stored == 0:
        print("BLOCKED: no procedures are stored. Run `memorable login`, "
              "do a task, run `memorable record`, then rerun this script.")
        return 2
    if stored < 0:
        print("BLOCKED: could not read the stored procedure list. "
              "Is `memorable` on PATH and logged in?")
        return 2

    if args.planner_model:
        try:
            model = _load_planner(args.planner_model)
        except Exception as exc:  # noqa: BLE001 - report, do not fake a run
            print(f"BLOCKED: could not load planner model: {exc}")
            return 2
        planner_kind = f"llm:{args.planner_model}"
        planner = _planner_from_model(model)
    elif args.allow_scripted:
        planner_kind = "scripted"
        planner = None  # agent defaults: explorer / procedure_follower by name
    else:
        print("BLOCKED: no chat model key path given. Pass --planner-model "
              "provider:model, or --allow-scripted for a labeled smoke test.")
        return 2

    trace_dir = Path(args.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    modules = sorted(
        p.name for p in CORPUS.iterdir()
        if p.is_dir() and any(f.suffix == ".py" for f in p.iterdir())
    )

    runs: list[dict] = []
    for module in modules:
        for arm in ("baseline", "treatment"):
            tracer = MemorableTracer(trace_dir, session_id=f"real-{arm}-{module}")
            tracer.workflow_id = "summarize-python-module"
            task = (f"Summarize the {module} module: list each function "
                    "with a one-line description.")
            if arm == "baseline":
                # Same planner as the treatment arm, with recall forced to
                # a miss. Only the recalled procedure differs between arms.
                arm_recall_fn = lambda t: RecallResult(False, "", "none")  # noqa: E731
                arm_policy = planner if planner is not None else "explorer"
            else:
                arm_recall_fn = lambda t: recall_procedure(t)  # noqa: E731
                arm_policy = planner if planner is not None else "procedure_follower"
            t0 = time.perf_counter()
            state = run_agent(task, module_dir=module, root=CORPUS,
                              policy=arm_policy, tracer=tracer,
                              recall_fn=arm_recall_fn,
                              verify_command=args.verify_command,
                              verify_cwd=args.verify_cwd)
            wall_ms = round((time.perf_counter() - t0) * 1000, 1)
            trace_path = tracer.write_trace()
            ok, msg = tracer.ingest(trace_path)
            runs.append({
                "module": module, "arm": arm,
                "tool_calls": len(tracer.tool_calls),
                "wall_ms": wall_ms, "trace": str(trace_path),
                "ingest_ok": ok, "ingest_msg": msg,
                "report": state.get("report_path"),
            })

    fallbacks = getattr(planner, "fallbacks", {}).get("count", 0) if planner is not None else 0
    baseline = sum(r["tool_calls"] for r in runs if r["arm"] == "baseline")
    treatment = sum(r["tool_calls"] for r in runs if r["arm"] == "treatment")
    results = {
        "planner": planner_kind,
        "planner_fallbacks_to_scripted_explorer": fallbacks,
        "stored_procedures_before_run": stored,
        "fixture_used": False,
        "modules": modules,
        "runs": runs,
        "baseline_total_tool_calls": baseline,
        "treatment_total_tool_calls": treatment,
        "delta_tool_calls_pct": round(100 * (baseline - treatment) / baseline, 1) if baseline else None,
        "ingest_ok_count": sum(1 for r in runs if r["ingest_ok"]),
        "ingest_total": len(runs),
    }
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2)[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
