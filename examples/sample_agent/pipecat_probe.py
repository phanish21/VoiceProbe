import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from pipecat.frames.frames import (
    EndFrame,
    LLMFullResponseEndFrame,
    LLMRunFrame,
    LLMTextFrame,
    MetricsFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.groq.llm import GroqLLMService


class MetricsCollector(FrameProcessor):
    """Watches frames go by, records what we care about, forwards everything."""

    def __init__(self):
        super().__init__()
        self.text_parts: list[str] = []
        self.metrics = [] 
        self.done = asyncio.Event()

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, MetricsFrame):
            self.metrics.extend(frame.data)
        elif isinstance(frame, LLMTextFrame):
            self.text_parts.append(frame.text)
        elif isinstance(frame, LLMFullResponseEndFrame):
            self.done.set()

        await self.push_frame(frame, direction)  # never swallow frames


async def main():
    load_dotenv()
    prompt = Path("examples/sample_agent/prompt.txt").read_text(encoding="utf-8")

    llm = GroqLLMService(
        api_key=os.environ["GROQ_API_KEY"],
        settings=GroqLLMService.Settings(model="openai/gpt-oss-120b"),
    )
    context = LLMContext(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Sure, a stack is first in first out."},
        ]
    )
    pair = LLMContextAggregatorPair(context)
    collector = MetricsCollector()

    pipeline = Pipeline([pair.user(), llm, collector, pair.assistant()])
    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )

    async def stop_when_done():
        await collector.done.wait()
        await task.queue_frames([EndFrame()])

    await task.queue_frames([LLMRunFrame()])
    runner = PipelineRunner(handle_sigint=False)
    await asyncio.gather(runner.run(task), stop_when_done())

    print("\n--- reply ---")
    print("".join(collector.text_parts))
    print("\n--- metrics ---")
    for m in collector.metrics:
        print(type(m).__name__, m.model_dump())


if __name__ == "__main__":
    asyncio.run(main())