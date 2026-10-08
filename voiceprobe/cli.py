import argparse
from pathlib import Path

from .adapters.text_adapter import TextAdapter
from .llm import get_llm
from .runner import run_scenario
from .scenario import load_scenario


def build_agent(args, prompt: str):
    if args.agent == "pipecat":
        if args.mock:
            raise SystemExit("--mock only works with --agent text")
        try:
            # imported here, not at the top, so the CLI works without Pipecat installed
            from .adapters.pipecat_adapter import PipecatAdapter
        except ImportError:
            raise SystemExit('Pipecat is not installed. Run: pip install -e ".[pipecat]"')
        kwargs = {"model": args.model} if args.model else {}
        return PipecatAdapter(prompt, **kwargs)
    return TextAdapter(get_llm(args.mock, use_cache=not args.no_cache), prompt)

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
        choices=["text", "pipecat"],
        default="text",
        help="which agent adapter to test (pipecat needs: pip install -e '.[pipecat]')",
    )
    run.add_argument("--model", default=None, help="model for the pipecat agent")

    args = parser.parse_args()

    scenario = load_scenario(args.scenario)
    prompt = Path(args.prompt).read_text(encoding="utf-8")
    agent = build_agent(args, prompt)
    result = run_scenario(scenario, agent)

    for turn in result["turns"]:
        print(f"[{turn['role']:6}] {turn['text']}")
    print(f"\nmetrics: {result['metrics']}  ->  runs/{scenario.name}.json")


if __name__ == "__main__":
    main()