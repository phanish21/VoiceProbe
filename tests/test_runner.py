from pathlib import Path

from voiceprobe.adapters.text_adapter import TextAdapter
from voiceprobe.llm import MockLLM
from voiceprobe.runner import run_scenario
from voiceprobe.scenario import load_scenario

from voiceprobe.adapters.base import AgentAdapter, AgentReply

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

class FakeAdapter(AgentAdapter):
    """Pretends to be an agent that reports rich metrics. No network, no Pipecat."""

    def start(self):
        return AgentReply(text="hi", llm_ms=100.0, llm_ttfat_ms=80.0,
                          prompt_tokens=10, completion_tokens=5, model="fake")

    def respond(self, caller_text):
        return AgentReply(text="ok", llm_ms=200.0, llm_ttfat_ms=120.0,
                          prompt_tokens=20, completion_tokens=5, model="fake")


def test_rich_metrics_are_recorded(tmp_path):
    scenario = load_scenario(str(ROOT / "scenarios" / "wrong_answer.yaml"))
    result = run_scenario(scenario, FakeAdapter(), out_dir=str(tmp_path))
    assert result["metrics"]["ttfat_ms_avg"] == 100.0
    assert result["metrics"]["ttfat_ms_max"] == 120.0
    assert result["metrics"]["total_tokens"] == 40
    assert result["model"] == "fake"