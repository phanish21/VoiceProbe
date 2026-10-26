import asyncio
import os
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from pipecat.frames.frames import (
    InputAudioRawFrame,
    LLMRunFrame,
    LLMTextFrame,
    MetricsFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import (
    LLMUsageMetricsData,
    ProcessingMetricsData,
    TTFAMetricsData,
    TTFATMetricsData,
    TTFBMetricsData,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.piper.tts import PiperTTSService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.workers.runner import WorkerRunner

from ..audio import CallerVoice, chunk_pcm, to_16k_mono
from .base import AgentAdapter, AgentReply

RATE = 16000
GRACE_S = 0.3        # wait after the reply ends for late metrics to arrive
TURN_TIMEOUT_S = 90  # give up on a turn that never produces speech


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 1)


def _stage(processor: str) -> str:
    """'WhisperSTTService#0' -> 'stt', 'PiperTTSService#0' -> 'tts', anything else -> 'llm'."""
    name = processor.split("#")[0]
    if name.endswith("STTService"):
        return "stt"
    if name.endswith("TTSService"):
        return "tts"
    return "llm"


def route_metrics(metrics) -> dict:
    """Sort Pipecat's metric objects into AgentReply fields. The first value of each kind wins."""
    out: dict = {}
    for m in metrics:
        if m.model is None:  # placeholder entries emitted before a request starts
            continue
        stage = _stage(m.processor)
        if isinstance(m, TTFBMetricsData):
            if stage == "llm":
                out.setdefault("llm_ttfb_ms", _ms(m.value))
            elif stage == "tts":
                out.setdefault("tts_ttfb_ms", _ms(m.value))
        elif isinstance(m, TTFATMetricsData):
            out.setdefault("llm_ttfat_ms", _ms(m.ttfat))
        elif isinstance(m, TTFAMetricsData):
            out.setdefault("tts_ttfa_ms", _ms(m.ttfa))
        elif isinstance(m, ProcessingMetricsData):
            key = {"llm": "llm_ms", "tts": "tts_ms", "stt": "stt_pipecat_ms"}[stage]
            out.setdefault(key, _ms(m.value))
        elif isinstance(m, LLMUsageMetricsData) and stage == "llm":
            out.setdefault("prompt_tokens", m.value.prompt_tokens)
            out.setdefault("completion_tokens", m.value.completion_tokens)
        if stage == "llm" and m.model:
            out.setdefault("model", m.model)  # TTS reports '' here, so only the LLM names the model
    return out


def build_reply(text, metrics, t_ref, t_transcript, t_first_audio, requested_model) -> AgentReply:
    fields = route_metrics(metrics)
    reply = AgentReply(text=text, llm_ms=fields.pop("llm_ms", 0.0), **fields)
    if t_transcript is not None:
        reply.stt_ms = _ms(t_transcript - t_ref)
    if t_first_audio is not None:
        reply.e2e_ms = _ms(t_first_audio - t_ref)
    if reply.model and reply.model != requested_model:
        print(f"WARNING: asked for {requested_model}, but {reply.model} ran", file=sys.stderr)
    return reply


class _TurnState:
    """What the taps record during one turn."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = []
        self.text_parts = []
        self.transcript = None
        self.t_transcript = None
        self.t_first_audio = None
        self.done = asyncio.Event()


class _Tap(FrameProcessor):
    """Pass-through processor that notes what it sees. It always forwards every frame."""

    def __init__(self, state: _TurnState, kind: str):
        super().__init__()
        self.state = state
        self.kind = kind  # 'stt' (after Whisper), 'llm' (after the LLM) or 'tts' (after Piper)

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        s = self.state
        if self.kind == "stt" and isinstance(frame, TranscriptionFrame):
            s.transcript = frame.text
            if s.t_transcript is None:
                s.t_transcript = time.perf_counter()
        elif self.kind == "llm" and isinstance(frame, LLMTextFrame):
            s.text_parts.append(frame.text)
        elif self.kind == "tts":
            if isinstance(frame, TTSAudioRawFrame):
                if s.t_first_audio is None:
                    s.t_first_audio = time.perf_counter()
            elif isinstance(frame, MetricsFrame):
                s.metrics.extend(frame.data)  # only here: metrics pass every tap, count them once
            elif isinstance(frame, TTSStoppedFrame):
                s.done.set()
        await self.push_frame(frame, direction)


class _LoopThread:
    """An asyncio event loop on a background thread, so synchronous code can call async code."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def run(self, coro, timeout=300):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def stop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)


class PipecatVoiceAdapter(AgentAdapter):
    """The agent as a real voice pipeline: caller audio -> Whisper -> LLM -> Piper."""

    def __init__(self, system_prompt, model="openai/gpt-oss-120b",
                 stt_model="base", voice="en_US-lessac-medium"):
        load_dotenv()
        logger.remove()
        logger.add(sys.stderr, level="WARNING")
        self.system, self.model = system_prompt, model
        self.stt_model, self.voice = stt_model, voice
        self.caller_voice = CallerVoice(model_path=f"voices/{voice}.onnx")
        self._loop = _LoopThread()
        self._state = None
        self._worker = None
        self._run_task = None

    # --- the public interface the runner uses (synchronous) ---

    def start(self) -> AgentReply:
        return self._loop.run(self._open_and_greet())

    def respond(self, caller_text: str) -> AgentReply:
        return self._loop.run(self._hear_and_reply(caller_text))

    def close(self) -> None:
        try:
            self._loop.run(self._shutdown(), timeout=30)
        finally:
            self._loop.stop()

    # --- async internals, all running on the background loop ---

    async def _open(self):
        stt = WhisperSTTService(
            device="cpu", compute_type="int8",
            settings=WhisperSTTService.Settings(model=self.stt_model),
        )
        llm = GroqLLMService(
            api_key=os.environ["GROQ_API_KEY"],
            settings=GroqLLMService.Settings(model=self.model, system_instruction=self.system),
        )
        tts = PiperTTSService(
            download_dir=Path("voices"),
            settings=PiperTTSService.Settings(voice=self.voice),
        )
        self._state = _TurnState()
        context = LLMContext(messages=[{"role": "user", "content": "(call connected)"}])
        pair = LLMContextAggregatorPair(context)
        taps = {k: _Tap(self._state, k) for k in ("stt", "llm", "tts")}
        pipeline = Pipeline(
            [stt, taps["stt"], pair.user(), llm, taps["llm"], tts, taps["tts"], pair.assistant()]
        )
        self._worker = PipelineWorker(
            pipeline,
            params=PipelineParams(
                audio_in_sample_rate=RATE, enable_metrics=True, enable_usage_metrics=True
            ),
        )
        self._run_task = asyncio.create_task(WorkerRunner(handle_sigint=False).run(self._worker))

    async def _open_and_greet(self) -> AgentReply:
        await self._open()
        self._state.reset()
        t_ref = time.perf_counter()
        await self._worker.queue_frames([LLMRunFrame()])
        return await self._finish(t_ref)

    async def _hear_and_reply(self, caller_text: str) -> AgentReply:
        pcm = to_16k_mono(self.caller_voice.synthesize(caller_text))  # cached; before the clock starts
        chunks = chunk_pcm(pcm, pad_s=0.3, rate=RATE)
        frames = [VADUserStartedSpeakingFrame()] + [
            InputAudioRawFrame(audio=c, sample_rate=RATE, num_channels=1) for c in chunks
        ]
        self._state.reset()
        await self._worker.queue_frames(frames)
        t_ref = time.perf_counter()  # the caller has finished speaking
        await self._worker.queue_frames([VADUserStoppedSpeakingFrame()])
        return await self._finish(t_ref)

    async def _finish(self, t_ref: float) -> AgentReply:
        s = self._state
        try:
            await asyncio.wait_for(s.done.wait(), timeout=TURN_TIMEOUT_S)
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"No spoken reply within {TURN_TIMEOUT_S}s (transcript so far: {s.transcript!r})"
            ) from None
        await asyncio.sleep(GRACE_S)
        return build_reply(
            "".join(s.text_parts), s.metrics, t_ref, s.t_transcript, s.t_first_audio, self.model
        )

    async def _shutdown(self):
        if self._worker is None:
            return
        await self._worker.cancel()
        try:
            await self._run_task
        except asyncio.CancelledError:
            pass