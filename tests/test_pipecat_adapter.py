import pytest

pytest.importorskip("pipecat")  # skipped where Pipecat isn't installed, like CI

from pipecat.metrics.metrics import (
    LLMTokenUsage,
    LLMUsageMetricsData,
    ProcessingMetricsData,
    TTFATMetricsData,
    TTFBMetricsData,
)

from voiceprobe.adapters.pipecat_adapter import PipecatAdapter


def make_metrics(model="m"):
    return [
        TTFBMetricsData(processor="llm", model=None, value=0.0),  # placeholder, must be ignored
        TTFBMetricsData(processor="llm", model=model, value=0.5),
        TTFATMetricsData(processor="llm", model=model, ttfat=0.75, ttfb=0.5, thinking_time=0.25),
        ProcessingMetricsData(processor="llm", model=model, value=1.25),
        LLMUsageMetricsData(
            processor="llm",
            model=model,
            value=LLMTokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        ),
    ]


def test_seconds_become_milliseconds_and_placeholders_are_skipped():
    reply = PipecatAdapter("prompt", model="m")._to_reply("hello", make_metrics())
    assert reply.llm_ttfb_ms == 500.0
    assert reply.llm_ttfat_ms == 750.0
    assert reply.llm_ms == 1250.0
    assert (reply.prompt_tokens, reply.completion_tokens) == (10, 5)
    assert reply.model == "m"


def test_model_mismatch_prints_a_warning(capsys):
    PipecatAdapter("prompt", model="wanted")._to_reply("hello", make_metrics(model="other"))
    assert "WARNING" in capsys.readouterr().err