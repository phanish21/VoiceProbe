import pytest

from voiceprobe.judge import judge_run, parse_verdict
from voiceprobe.scenario import Scenario


def test_parse_handles_text_around_the_json_and_string_booleans():
    raw = 'Sure! {"score": 8, "reasons": "ok", "checks": [{"criterion": "a", "passed": "false"}]} Done.'
    v = parse_verdict(raw)
    assert v["score"] == 8 and v["checks"][0]["passed"] is False


def test_parse_rejects_bad_output():
    with pytest.raises(ValueError):
        parse_verdict("no json here")
    with pytest.raises(ValueError):
        parse_verdict('{"score": 11, "reasons": "x", "checks": []}')


class _Flaky:
    """First answer is garbage, second is valid."""
    def __init__(self):
        self.calls = 0

    def chat(self, system, messages):
        self.calls += 1
        if self.calls == 1:
            return "I think it went well."
        return '{"score": 6, "reasons": "r", "checks": [{"criterion": "c", "passed": true, "note": ""}]}'


def test_retry_then_pass_rule():
    scenario = Scenario(name="t", description="d", turns=[], caller={"x": 1},
                        expect={"agent_should": ["c"], "min_judge_score": 7})
    run = {"turns": [{"role": "agent", "text": "hi"}]}
    v = judge_run(scenario, run, "prompt", _Flaky())
    assert v["score"] == 6
    assert v["passed"] is False  # all checks pass, but 6 is below the minimum of 7

def test_parse_ignores_a_think_block():
    raw = '<think>maybe {"score": 1}</think>{"score": 7, "reasons": "r", "checks": []}'
    assert parse_verdict(raw)["score"] == 7