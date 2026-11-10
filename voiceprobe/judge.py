"""LLM-as-judge: scores a finished conversation against the scenario's expectations."""
import json
import sys
import re
from pathlib import Path

from .llm import GroqLLM
from .scenario import load_scenario

JUDGE_MODEL = "qwen/qwen3.8-27b"
DEFAULT_MIN_SCORE = 7

JUDGE_SYSTEM = """You are a strict evaluator of a voice AI agent's phone conversation.
You get the agent's instructions, what the agent was expected to do, and the transcript.
Caller lines come with what the agent's speech recognition heard, which can differ from
what the caller said. Do not blame the agent for replying sensibly to a mis-heard line.
Judge only what is in the transcript.

Score the agent from 1 (fails the expectations) to 10 (fully meets them and follows its
instructions). For each expectation, say whether it was met.

Use the full 1-10 range. 10 means flawless: every expectation met and every instruction
in the agent's prompt followed. A conversation with a minor slip (for example a reply
longer than the prompt allows) is at most 8. Missing any expected behavior is at most 5.

Reply with ONLY one JSON object, no other text:
{"score": <integer 1-10>,
 "reasons": "<two or three sentences>",
 "checks": [{"criterion": "<expectation, copied>", "passed": <true|false>, "note": "<short>"}]}"""


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"  # models sometimes answer "false" as a string


def parse_verdict(raw: str) -> dict:
    """Pull the JSON object out of the judge's reply and validate it."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)  # drop reasoning blocks
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in the judge's reply")
    data = json.loads(raw[start : end + 1])
    score = int(data["score"])
    if not 1 <= score <= 10:
        raise ValueError(f"score out of range: {score}")
    checks = [
        {"criterion": str(c["criterion"]), "passed": _as_bool(c["passed"]), "note": str(c.get("note", ""))}
        for c in data.get("checks", [])
    ]
    return {"score": score, "reasons": str(data.get("reasons", "")), "checks": checks}


def format_transcript(turns: list[dict]) -> str:
    lines = []
    for i, t in enumerate(turns):
        if t["role"] == "agent":
            lines.append(f"AGENT: {t['text']}")
        else:
            lines.append(f"CALLER: {t['text']}")
            heard = turns[i + 1].get("heard") if i + 1 < len(turns) else None
            if heard is not None:
                lines.append(f"  (speech recognition heard: {heard.strip()})")
    return "\n".join(lines)


def build_prompt(scenario, turns: list[dict], agent_prompt: str) -> str:
    expected = "\n".join(f"- {x}" for x in scenario.expect.get("agent_should", [])) or "- (none given)"
    return (
        f"AGENT INSTRUCTIONS:\n{agent_prompt}\n\n"
        f"SCENARIO: {scenario.description}\n\n"
        f"THE AGENT SHOULD:\n{expected}\n\n"
        f"TRANSCRIPT:\n{format_transcript(turns)}"
    )


def judge_run(scenario, run: dict, agent_prompt: str, llm, max_attempts: int = 2) -> dict:
    prompt = build_prompt(scenario, run["turns"], agent_prompt)
    reminder, last_error = "", None
    for _ in range(max_attempts):
        raw = llm.chat(JUDGE_SYSTEM, [{"role": "user", "content": prompt + reminder}])
        try:
            verdict = parse_verdict(raw)
            break
        except (ValueError, KeyError, TypeError) as e:
            last_error = e
            reminder = "\n\nYour last reply was not valid JSON in the required shape. Reply with only the JSON object."
    else:
        raise RuntimeError(f"judge returned unusable output: {last_error}")
    min_score = scenario.expect.get("min_judge_score", DEFAULT_MIN_SCORE)
    verdict["min_score"] = min_score
    verdict["passed"] = verdict["score"] >= min_score and all(c["passed"] for c in verdict["checks"])
    return verdict


if __name__ == "__main__":
    # python -m voiceprobe.judge scenarios/x.yaml runs/x.json
    scenario = load_scenario(sys.argv[1])
    data = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    runs = data["runs"] if "runs" in data else [data]
    prompt = Path("examples/sample_agent/prompt.txt").read_text(encoding="utf-8")
    llm = GroqLLM(model=JUDGE_MODEL)
    for i, run in enumerate(runs, 1):
        v = judge_run(scenario, run, prompt, llm)
        print(f"run {i}: score {v['score']}/{v['min_score']} {'PASS' if v['passed'] else 'FAIL'}")
        print(f"  {v['reasons']}")
        for c in v["checks"]:
            print(f"  [{'x' if c['passed'] else ' '}] {c['criterion']}: {c['note']}")