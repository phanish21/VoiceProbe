"""Latency waterfall for one scenario: where does a reply's time go?"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to a file, no window
import matplotlib.pyplot as plt

STAGES = [
    ("STT (Whisper)", "stt_ms", "#4C78A8"),
    ("LLM time to first answer token", "llm_ttfat_ms", "#F58518"),
    ("TTS time to first audio (Piper)", "tts_ttfa_ms", "#54A24B"),
]

SETUP = "Whisper base (CPU, int8), Groq LLM, Piper TTS, one Windows machine"
MIN_PUBLISH_RUNS = 30  # below this the chart is labeled as development numbers


def make_waterfall(run_file: str, out_file: str) -> None:
    data = json.loads(Path(run_file).read_text(encoding="utf-8"))
    reply = data["summary_by_turn"]["reply"]
    n = reply["e2e_ms"]["n"]
    warmup = data.get("warmup_runs", 0)

    fig, ax = plt.subplots(figsize=(9, 4.0))
    start = 0.0
    labels = []
    for i, (label, key, color) in enumerate(STAGES):
        width = reply[key]["median"]
        ax.barh(i, width, left=start, color=color)
        ax.text(start + width + 15, i, f"{width:.0f} ms", va="center", fontsize=9)
        labels.append(label)
        start += width

    total = reply["e2e_ms"]["median"]
    i = len(STAGES)
    ax.barh(i, total, color="#555555")
    ax.text(total + 15, i, f"{total:.0f} ms", va="center", fontsize=9, fontweight="bold")
    labels.append("Measured end to end (stopwatch)")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("milliseconds after the caller stops speaking (median)")
    ax.set_xlim(0, max(start, total) * 1.15)
    ax.set_title(f"Reply latency, {data['scenario']}, {data['model']} (n={n} replies)")
    ax.spines[["top", "right"]].set_visible(False)

    lines = [
        SETUP,
        "Stage medians are measured separately, so they do not sum exactly to the "
        "end-to-end median."
        + (f" {warmup} warm-up run excluded." if warmup else ""),
        "Synthetic caller audio, VAD events injected (real VAD delay not included).",
    ]
    if n < MIN_PUBLISH_RUNS:
        lines.append(f"Development numbers: n={n}, fewer than {MIN_PUBLISH_RUNS} runs.")
    fig.text(0.01, 0.01, "\n".join(lines), fontsize=8, color="#555555", va="bottom")
    fig.tight_layout(rect=(0, 0.05 + 0.035 * len(lines), 1, 1))
    fig.savefig(out_file, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    run_file = sys.argv[1] if len(sys.argv) > 1 else "runs/correct_answer.json"
    out_file = sys.argv[2] if len(sys.argv) > 2 else "docs/latency_waterfall.png"
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    make_waterfall(run_file, out_file)
    print(f"saved {out_file}")