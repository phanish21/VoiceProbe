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

class AgentAdapter:
    """Every agent under test implements these two methods."""

    def start(self) -> AgentReply:
        raise NotImplementedError

    def respond(self,caller_text: str) -> AgentReply:
        raise NotImplementedError