import pytest

pytest.importorskip("pipecat")  # skipped where Pipecat isn't installed, like CI

from pipecat.metrics.metrics import (
    LLMTokenUsage,
    LLMUsageMetricsData,
    ProcessingMetricsData,
    TTFAMetricsData,
    TTFATMetricsData,
    TTFBMetricsData,
)

from voiceprobe.adapters.pipecat_voice_adapter import build_reply, route_metrics, _TurnState


def make_metrics(llm_model="m"):
    stt, llm, tts = "WhisperSTTService#0", "GroqLLMService#0", "PiperTTSService#0"
    return [
        TTFBMetricsData(processor=stt, model=None, value=0.0),  # placeholder, ignored
        ProcessingMetricsData(processor=stt, model="base", value=0.02),
        TTFBMetricsData(processor=llm, model=llm_model, value=0.4),
        TTFATMetricsData(processor=llm, model=llm_model, ttfat=0.5, ttfb=0.4, thinking_time=0.1),
        ProcessingMetricsData(processor=llm, model=llm_model, value=0.6),
        LLMUsageMetricsData(
            processor=llm, model=llm_model,
            value=LLMTokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        ),
        TTFBMetricsData(processor=tts, model="", value=0.1),
        TTFAMetricsData(processor=tts, model="", ttfa=0.18, ttfb=0.1, leading_silence=0.08),
        ProcessingMetricsData(processor=tts, model="", value=0.3),
    ]


def test_metrics_are_routed_to_the_right_stage():
    fields = route_metrics(make_metrics())
    assert fields["stt_pipecat_ms"] == pytest.approx(20.0)
    assert fields["llm_ttfb_ms"] == pytest.approx(400.0)
    assert fields["llm_ttfat_ms"] == pytest.approx(500.0)
    assert fields["llm_ms"] == pytest.approx(600.0)
    assert fields["tts_ttfb_ms"] == pytest.approx(100.0)
    assert fields["tts_ttfa_ms"] == pytest.approx(180.0)
    assert fields["tts_ms"] == pytest.approx(300.0)
    assert (fields["prompt_tokens"], fields["completion_tokens"]) == (10, 5)


def test_model_name_comes_from_the_llm_not_the_empty_tts_name():
    assert route_metrics(make_metrics())["model"] == "m"


def test_stopwatch_fields_and_missing_audio():
    reply = build_reply("hi", make_metrics(), 10.0, 11.2, 12.5, "m")
    assert reply.stt_ms == pytest.approx(1200.0)
    assert reply.e2e_ms == pytest.approx(2500.0)
    assert reply.llm_ms == pytest.approx(600.0)
    greeting = build_reply("hi", make_metrics(), 10.0, None, 10.7, "m")
    assert greeting.stt_ms is None  # the agent's opening line has no STT stage
    assert greeting.e2e_ms == pytest.approx(700.0)


def test_model_mismatch_prints_a_warning(capsys):
    build_reply("hi", make_metrics(llm_model="other"), 0.0, None, 1.0, "wanted")
    assert "WARNING" in capsys.readouterr().err


def test_renamed_stt_subclass_is_still_routed_to_stt():
    # Regression: a subclass name must not leak STT metrics into the LLM fields.
    metrics = [
        ProcessingMetricsData(processor="_SharedWhisperSTTService#3", model="base", value=0.01),
        ProcessingMetricsData(processor="GroqLLMService#3", model="m", value=0.6),
    ]
    fields = route_metrics(metrics)
    assert fields["stt_pipecat_ms"] == pytest.approx(10.0)
    assert fields["llm_ms"] == pytest.approx(600.0)


def test_unrecognized_processor_is_skipped_with_a_warning(capsys):
    metrics = [
        ProcessingMetricsData(processor="MysteryThing#0", model="x", value=9.0),
        ProcessingMetricsData(processor="GroqLLMService#0", model="m", value=0.6),
    ]
    fields = route_metrics(metrics)
    assert fields["llm_ms"] == pytest.approx(600.0)  # not overwritten by the unknown one
    assert "MysteryThing" in capsys.readouterr().err


def test_turn_error_wakes_the_waiter_and_raises():
    state = _TurnState()
    assert not state.done.is_set()
    state.fail("transcription failed")
    assert state.done.is_set()
    with pytest.raises(RuntimeError, match="transcription failed"):
        state.raise_if_failed()
    state.reset()
    state.raise_if_failed()  # a new turn starts clean

def test_heard_text_is_kept_on_the_reply():
    reply = build_reply("hi", make_metrics(), 10.0, 11.0, 12.0, "m", heard="the stack is last in")
    assert reply.heard == "the stack is last in"
    assert build_reply("hi", make_metrics(), 10.0, None, 11.0, "m").heard is None