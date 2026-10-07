from dataclasses import dataclass, field
from pathlib import Path

import yaml

@dataclass
class Scenario:
    name: str
    description: str
    turns: list[dict]
    expect: dict = field(default_factory=dict)

def load_scenario(path: str) -> Scenario:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Scenario(
        name=data["name"],
        description=data.get("description", ""),
        turns=data["turns"],
        expect=data.get("expect", {}),
    )