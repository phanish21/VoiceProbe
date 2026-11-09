import json
import time
from dataclasses import asdict
from pathlib import Path

from .metrics import aggregate_runs, summarize_run
from .scenario import Scenario
from .metrics import aggregate_by_turn, aggregate_runs, summarize_run


def _agent_turn(reply) -> dict:
    """Keep every field the adapter filled in, drop the empty ones."""
    fields = {k: v for k, v in asdict(reply).items() if v is not None}
    fields["llm_ms"] = round(reply.llm_ms, 1)
    return {"role": "agent", **fields}


def _save(result: dict, out_dir: str, name: str) -> None:
    Path(out_dir).mkdir(exist_ok=True)
    Path(out_dir, f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

def _caller_lines(scenario: Scenario, caller, turns: list):
    """Yield (text, behavior) for each caller line.

    With a simulator the next line is generated lazily from `turns`, which the runner
    keeps appending to, so each line reacts to the agent's latest reply.
    """
    if caller is None:
        for t in scenario.turns:
            yield t["say"], t.get("behavior")
    else:
        for _ in range(caller.max_turns):
            yield caller.next_line(turns), "simulated"


def run_scenario(scenario: Scenario, agent, out_dir: str | None = "runs", caller=None) -> dict:
    turns = []
    try:
        turns.append(_agent_turn(agent.start()))
        for say, behavior in _caller_lines(scenario, caller, turns):
            turns.append({"role": "caller", "text": say, "behavior": behavior})
            turns.append(_agent_turn(agent.respond(say)))
    finally:
        agent.close()

    agent_turns = [t for t in turns if t["role"] == "agent"]
    result = {
        "scenario": scenario.name,
        "timestamp": time.time(),
        "model": next((t["model"] for t in agent_turns if "model" in t), None),
        "turns": turns,
        "metrics": summarize_run(agent_turns),
    }
    if out_dir is not None:
        _save(result, out_dir, scenario.name)
    return result

def run_repeated(
    scenario: Scenario, make_agent, n: int, out_dir: str | None = "runs", warmup: int = 0, caller=None
) -> dict:
    """Run the scenario n times, each with a brand-new agent.

    After every finished run a partial file is written, so a crash does not lose the
    runs that already completed. The partial file is removed when all runs finish.
    """
    for _ in range(warmup):
        run_scenario(scenario, make_agent(), out_dir=None, caller=caller)
    runs = []
    for _ in range(n):
        runs.append(run_scenario(scenario, make_agent(), out_dir=None))
        if out_dir is not None:
            _save(
                {"scenario": scenario.name, "complete": False,
                 "n_done": len(runs), "n_requested": n, "runs": runs},
                out_dir, f"{scenario.name}.partial",
            )
    result = {
        "scenario": scenario.name,
        "timestamp": time.time(),
        "model": runs[0]["model"],
        "n_runs": n,
        "warmup_runs": warmup,
        "runs": runs,
        "summary": aggregate_runs(runs),
        "summary_by_turn": aggregate_by_turn(runs),
    }
    if out_dir is not None:
        _save(result, out_dir, scenario.name)
        Path(out_dir, f"{scenario.name}.partial.json").unlink(missing_ok=True)
    return result