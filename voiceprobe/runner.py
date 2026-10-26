import json
import time
from dataclasses import asdict
from pathlib import Path

from .metrics import aggregate_runs, summarize_run
from .scenario import Scenario


def _agent_turn(reply) -> dict:
    """Keep every field the adapter filled in, drop the empty ones."""
    fields = {k: v for k, v in asdict(reply).items() if v is not None}
    fields["llm_ms"] = round(reply.llm_ms, 1)
    return {"role": "agent", **fields}


def _save(result: dict, out_dir: str, name: str) -> None:
    Path(out_dir).mkdir(exist_ok=True)
    Path(out_dir, f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


def run_scenario(scenario: Scenario, agent, out_dir: str | None = "runs") -> dict:
    turns = []
    try:
        turns.append(_agent_turn(agent.start()))
        for turn in scenario.turns:
            turns.append({"role": "caller", "text": turn["say"], "behavior": turn.get("behavior")})
            turns.append(_agent_turn(agent.respond(turn["say"])))
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


def run_repeated(scenario: Scenario, make_agent, n: int, out_dir: str | None = "runs") -> dict:
    """Run the scenario n times, each with a brand-new agent."""
    runs = [run_scenario(scenario, make_agent(), out_dir=None) for _ in range(n)]
    result = {
        "scenario": scenario.name,
        "timestamp": time.time(),
        "model": runs[0]["model"],
        "n_runs": n,
        "runs": runs,
        "summary": aggregate_runs(runs),
    }
    if out_dir is not None:
        _save(result, out_dir, scenario.name)
    return result