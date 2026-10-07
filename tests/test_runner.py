from pathlib import Path

from voiceprobe.adapters.text_adapter import TextAdapter
from voiceprobe.llm import MockLLM
from voiceprobe.runner import run_scenario
from voiceprobe.scenario import load_scenario

ROOT = Path(__file__).parent.parent


def run(name, tmp_path):
    scenario = load_scenario(str(ROOT / "scenarios" / f"{name}.yaml"))
    agent = TextAdapter(MockLLM(), "test prompt")
    return run_scenario(scenario, agent, out_dir=str(tmp_path))


def test_wrong_answer_gets_probed(tmp_path):
    result = run("wrong_answer", tmp_path)
    last_reply = result["turns"][-1]["text"].lower()
    assert "sure" in last_reply


def test_correct_answer_is_acknowledged(tmp_path):
    result = run("correct_answer", tmp_path)
    last_reply = result["turns"][-1]["text"].lower()
    assert "correct" in last_reply


def test_transcript_alternates_and_ends_with_agent(tmp_path):
    result = run("wrong_answer", tmp_path)
    roles = [t["role"] for t in result["turns"]]
    assert roles == ["agent", "caller", "agent"]


def test_metrics_are_recorded_and_saved(tmp_path):
    result = run("wrong_answer", tmp_path)
    assert result["metrics"]["llm_ms_max"] > 0
    assert (tmp_path / "wrong_answer.json").exists()