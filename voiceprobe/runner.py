import json
import time
from pathlib import Path

from .scenario import Scenario

def run_scenario(scenario: Scenario, agent, out_dir: str = "runs") -> dict:
    turns = []

    reply = agent.start()
    turns.append({"role": "agent", "text": reply.text, "llm_ms": round(reply.llm_ms, 1)})

    for turn in scenario.turns:
        turns.append({"role": "caller", "text": turn["say"], "behavior": turn.get("behavior")})
        reply = agent.respond(turn["say"])
        turns.append({"role": "agent", "text": reply.text, "llm_ms": round(reply.llm_ms, 1)})

    latencies = [t["llm_ms"] for t in turns if t["role"] == "agent"]
    result = {
        "scenario": scenario.name,
        "timestamp": time.time(),
        "turns": turns,
        "metrics": {
            "llm_ms_avg": round(sum(latencies) / len(latencies), 1),
            "llm_ms_max": max(latencies),
        },
    }

    Path(out_dir).mkdir(exist_ok=True)
    Path(out_dir, f"{scenario.name}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result