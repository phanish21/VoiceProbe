import asyncio
import os
import sys

from dotenv import load_dotenv
from pipecat.frames.frames import (
    EndFrame,
    LLMFullResponseEndFrame,
    LLMRunFrame,
    LLMTextFrame,
    MetricsFrame,
)
from pipecat.metrics.metrics import (
    LLMUsageMetricsData,
    ProcessingMetricsData,
    TTFATMetricsData,
    TTFBMetricsData,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.groq.llm import GroqLLMService

from .base import AgentAdapter, AgentReply


class _Collector(FrameProcessor):
    """Records reply text and metrics, forwards every frame untouched."""

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
        await self.push_frame(frame, direction)


class PipecatAdapter(AgentAdapter):
    def __init__(self, system_prompt: str, model: str = "openai/gpt-oss-120b"):
        load_dotenv()
        self.system = system_prompt
        self.model = model
        self.history: list[dict] = []

    def start(self) -> AgentReply:
        self.history.append({"role": "user", "content": "(call connected)"})
        return self._turn()

    def respond(self, caller_text: str) -> AgentReply:
        self.history.append({"role": "user", "content": caller_text})
        return self._turn()

    def _turn(self) -> AgentReply:
        text, metrics = asyncio.run(self._run_pipeline())
        self.history.append({"role": "assistant", "content": text})
        return self._to_reply(text, metrics)

    async def _run_pipeline(self):
        llm = GroqLLMService(
            api_key=os.environ["GROQ_API_KEY"],
            settings=GroqLLMService.Settings(model=self.model),
        )
        context = LLMContext(
            messages=[{"role": "system", "content": self.system}, *self.history]
        )
        pair = LLMContextAggregatorPair(context)
        collector = _Collector()
        task = PipelineTask(
            Pipeline([pair.user(), llm, collector, pair.assistant()]),
            params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        )

        async def stop_when_done():
            await collector.done.wait()
            await task.queue_frames([EndFrame()])

        await task.queue_frames([LLMRunFrame()])
        runner = PipelineRunner(handle_sigint=False)
        await asyncio.gather(runner.run(task), stop_when_done())
        return "".join(collector.text_parts), collector.metrics

    def _to_reply(self, text, metrics) -> AgentReply:
        reply = AgentReply(text=text, llm_ms=0.0)
        for m in metrics:
            if m.model is None:  # skip the placeholder entries
                continue
            reply.model = m.model
            if isinstance(m, TTFBMetricsData):
                reply.llm_ttfb_ms = round(m.value * 1000, 1)
            elif isinstance(m, TTFATMetricsData):
                reply.llm_ttfat_ms = round(m.ttfat * 1000, 1)
            elif isinstance(m, ProcessingMetricsData):
                reply.llm_ms = round(m.value * 1000, 1)
            elif isinstance(m, LLMUsageMetricsData):
                reply.prompt_tokens = m.value.prompt_tokens
                reply.completion_tokens = m.value.completion_tokens
        if reply.model and reply.model != self.model:
            print(f"WARNING: asked for {self.model}, but {reply.model} ran", file=sys.stderr)
        return reply