from dataclasses import dataclass

@dataclass
class AgentReply:
    text: str
    llm_ms: float                        # total LLM stage time for this reply
    stt_ms: float | None = None          # filled in when we add audio
    tts_ms: float | None = None
    llm_ttfb_ms: float | None = None     # time to first byte
    llm_ttfat_ms: float | None = None    # time to first answer token
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    model: str | None = None             # the model that actually ran
    stt_pipecat_ms: float | None = None   # Pipecat's own STT metric, kept for comparison
    tts_ttfb_ms: float | None = None
    tts_ttfa_ms: float | None = None      # first audible sound = TTFB + leading silence
    e2e_ms: float | None = None           # caller audio ended -> first reply audio

class AgentAdapter:
    """Every agent under test implements these two methods."""

    def start(self) -> AgentReply:
        raise NotImplementedError

    def respond(self,caller_text: str) -> AgentReply:
        raise NotImplementedError

    def close(self) -> None:
        """Release resources (pipelines, threads). Default: nothing to do."""