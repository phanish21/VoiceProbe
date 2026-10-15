import pytest

from voiceprobe.metrics import aggregate_runs, percentile, summarize_run


def test_percentile_basics():
    assert percentile([5], 95) == 5
    assert percentile([10, 20], 50) == 15
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([1, 2, 3, 4, 5], 95) == pytest.approx(4.8)


def test_percentile_ignores_input_order():
    assert percentile([5, 1, 3, 2, 4], 50) == 3


def test_percentile_empty_raises():
    with pytest.raises(ValueError):
        percentile([], 50)


def fake_result(llm_values):
    turns = [{"role": "agent", "text": "x", "llm_ms": v} for v in llm_values]
    turns.insert(1, {"role": "caller", "text": "y"})  # caller turns carry no metrics
    return {"turns": turns, "metrics": {}}


def test_aggregate_pools_agent_turns_across_runs():
    summary = aggregate_runs([fake_result([100, 200]), fake_result([300, 400])])
    assert summary["llm_ms"]["n"] == 4
    assert summary["llm_ms"]["median"] == 250.0
    assert summary["llm_ms"]["max"] == 400.0
    assert "llm_ttfat_ms" not in summary  # never measured, so not reported


def test_summarize_run_skips_missing_metrics():
    assert summarize_run([{"llm_ms": 100.0}, {"llm_ms": 300.0}]) == {
        "llm_ms_avg": 200.0,
        "llm_ms_max": 300.0,
    }