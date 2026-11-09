from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Scenario:
    name: str
    description: str
    turns: list[dict] = field(default_factory=list)
    expect: dict = field(default_factory=dict)
    caller: dict = field(default_factory=dict)  # persona/goal/max_turns for an LLM-played caller


def load_scenario(path: str) -> Scenario:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    scenario = Scenario(
        name=data["name"],
        description=data.get("description", ""),
        turns=data.get("turns", []),
        expect=data.get("expect", {}),
        caller=data.get("caller", {}),
    )
    if not scenario.turns and not scenario.caller:
        raise ValueError(f"{path}: a scenario needs either 'turns' or a 'caller' block")
    return scenario