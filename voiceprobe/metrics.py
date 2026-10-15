import math

# Per-turn fields we summarize across runs. Missing ones are skipped.
TURN_METRICS = ["llm_ms", "llm_ttfb_ms", "llm_ttfat_ms", "stt_ms", "tts_ms"]


def percentile(values: list[float], p: float) -> float:
    """p-th percentile (0-100), interpolating linearly between neighbours."""
    if not values:
        raise ValueError("percentile() needs at least one value")
    ordered = sorted(values)
    k = (len(ordered) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def describe(values: list[float]) -> dict:
    return {
        "n": len(values),
        "median": round(percentile(values, 50), 1),
        "p95": round(percentile(values, 95), 1),
        "min": round(min(values), 1),
        "max": round(max(values), 1),
    }


def summarize_run(agent_turns: list[dict]) -> dict:
    """Summary of ONE run (moved here from runner.py)."""
    latencies = [t["llm_ms"] for t in agent_turns]
    metrics = {
        "llm_ms_avg": round(sum(latencies) / len(latencies), 1),
        "llm_ms_max": max(latencies),
    }
    ttfat = [t["llm_ttfat_ms"] for t in agent_turns if "llm_ttfat_ms" in t]
    if ttfat:
        metrics["ttfat_ms_avg"] = round(sum(ttfat) / len(ttfat), 1)
        metrics["ttfat_ms_max"] = max(ttfat)
    tokens = sum(
        t.get("prompt_tokens", 0) + t.get("completion_tokens", 0) for t in agent_turns
    )
    if tokens:
        metrics["total_tokens"] = tokens
    return metrics


def aggregate_runs(results: list[dict]) -> dict:
    """Pool every agent turn from every run, then describe each metric."""
    summary = {}
    for name in TURN_METRICS:
        values = [
            t[name]
            for r in results
            for t in r["turns"]
            if t["role"] == "agent" and name in t
        ]
        if values:
            summary[name] = describe(values)
    tokens = [r["metrics"]["total_tokens"] for r in results if "total_tokens" in r["metrics"]]
    if tokens:
        summary["total_tokens_per_run"] = describe(tokens)
    return summary