import json
import time
from dataclasses import asdict
from pathlib import Path

from .scenario import Scenario


def _agent_turn(reply) -> dict:
    """Keep every field the adapter filled in, drop the empty ones."""
    fields = {k: v for k, v in asdict(reply).items() if v is not None}
    fields["llm_ms"] = round(reply.llm_ms, 1)
    return {"role": "agent", **fields}


def _summarize(agent_turns: list[dict]) -> dict:
    latencies = [t["llm_ms"] for t in agent_turns]
    metrics = {
        "llm_ms_avg": round(sum(latencies) / len(latencies), 1),
        "llm_ms_max": max(latencies),
    }
    ttfat = [t["llm_ttfat_ms"] for t in agent_turns if "llm_ttfat_ms" in t]
    if ttfat:
        metrics["ttfat_ms_avg"] = round(sum(ttfat) / len(ttfat), 1)
        metrics["ttfat_ms_max"] = max(ttfat)
    tokens = sum(
        t.get("prompt_tokens", 0) + t.get("completion_tokens", 0) for t in agent_turns
    )
    if tokens:
        metrics["total_tokens"] = tokens
    return metrics


def run_scenario(scenario: Scenario, agent, out_dir: str = "runs") -> dict:
    turns = [_agent_turn(agent.start())]

    for turn in scenario.turns:
        turns.append({"role": "caller", "text": turn["say"], "behavior": turn.get("behavior")})
        turns.append(_agent_turn(agent.respond(turn["say"])))

    agent_turns = [t for t in turns if t["role"] == "agent"]
    result = {
        "scenario": scenario.name,
        "timestamp": time.time(),
        "model": next((t["model"] for t in agent_turns if "model" in t), None),
        "turns": turns,
        "metrics": _summarize(agent_turns),
    }

    Path(out_dir).mkdir(exist_ok=True)
    Path(out_dir, f"{scenario.name}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result