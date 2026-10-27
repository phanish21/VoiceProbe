import argparse
import sys
from pathlib import Path

from .adapters.text_adapter import TextAdapter
from .llm import get_llm
from .runner import run_repeated, run_scenario
from .scenario import load_scenario


def build_agent(args, prompt: str):
    if args.agent in ("pipecat", "pipecat-voice"):
        if args.mock:
            raise SystemExit("--mock only works with --agent text")
        try:
            # imported here so the CLI works without Pipecat installed
            if args.agent == "pipecat-voice":
                from .adapters.pipecat_voice_adapter import PipecatVoiceAdapter as Adapter
            else:
                from .adapters.pipecat_adapter import PipecatAdapter as Adapter
        except ImportError:
            raise SystemExit('Pipecat is not installed. Run: pip install -e ".[pipecat]"')
        kwargs = {"model": args.model} if args.model else {}
        return Adapter(prompt, **kwargs)
    return TextAdapter(get_llm(args.mock, use_cache=not args.no_cache), prompt)


def print_single(result: dict) -> None:
    for turn in result["turns"]:
        print(f"[{turn['role']:6}] {turn['text']}")
    print(f"\nmetrics: {result['metrics']}  ->  runs/{result['scenario']}.json")

def _print_table(title: str, summary: dict) -> None:
    print(title)
    print(f"{'metric':<22}{'n':>4}{'median':>10}{'p95':>10}{'min':>10}{'max':>10}")
    for name, s in summary.items():
        if name == "total_tokens_per_run":
            continue
        print(f"{name:<22}{s['n']:>4}{s['median']:>10}{s['p95']:>10}{s['min']:>10}{s['max']:>10}")
    print()

def print_summary(result: dict) -> None:
    print(
        f"{result['scenario']}: {result['n_runs']} runs "
        f"(+{result['warmup_runs']} warm-up, discarded), model={result['model']}\n"
    )
    _print_table("GREETING turns (no STT stage)", result["summary_by_turn"]["greeting"])
    _print_table("REPLY turns (caller audio in, reply audio out)", result["summary_by_turn"]["reply"])
    tokens = result["summary"].get("total_tokens_per_run")
    if tokens:
        print(f"tokens per run: median {tokens['median']}, p95 {tokens['p95']}")
    print(f"\n->  runs/{result['scenario']}.json")


def main():
    parser = argparse.ArgumentParser(prog="voiceprobe")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run one scenario against the sample agent")
    run.add_argument("scenario", help="path to a scenario YAML file")
    run.add_argument("--mock", action="store_true", help="use the offline mock LLM")
    run.add_argument(
        "--no-cache",
        action="store_true",
        help="skip the response cache so latency numbers are real",
    )
    run.add_argument(
        "--prompt",
        default="examples/sample_agent/prompt.txt",
        help="path to the agent's system prompt",
    )
    run.add_argument(
        "--agent",
        choices=["text", "pipecat", "pipecat-voice"],
        default="text",
        help="which agent adapter to test (pipecat needs: pip install -e '.[pipecat]')",
    )
    run.add_argument("--model", default=None, help="model for the pipecat agent")
    run.add_argument("--runs", type=int, default=1, help="repeat N times, report median and p95")
    run.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="extra runs to discard first (default: 1 for pipecat-voice, else 0)",
    )

    args = parser.parse_args()

    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")
    if args.runs > 1 and args.agent == "text" and not args.mock and not args.no_cache:
        raise SystemExit("--runs > 1 needs --no-cache: cached answers repeat the same latency")
    if args.warmup is not None and args.warmup < 0:
        raise SystemExit("--warmup cannot be negative")
    
    scenario = load_scenario(args.scenario)
    prompt = Path(args.prompt).read_text(encoding="utf-8")

    if args.runs == 1:
        print_single(run_scenario(scenario, build_agent(args, prompt)))
        return

    warmup = args.warmup if args.warmup is not None else (1 if args.agent == "pipecat-voice" else 0)
    done = {"count": 0}

    def make_agent():
        done["count"] += 1
        if done["count"] <= warmup:
            label = "warm-up"
        else:
            label = f"run {done['count'] - warmup}/{args.runs}"
        print(f"{label} ...", file=sys.stderr)
        return build_agent(args, prompt)

    print_summary(run_repeated(scenario, make_agent, args.runs, warmup=warmup))


if __name__ == "__main__":
    main()