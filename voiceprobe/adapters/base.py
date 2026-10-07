from dataclasses import dataclass

@dataclass
class AgentReply:
    text: str
    llm_ms: float
    stt_ms: float | None = None
    tts_ms: float | None = None

class AgentAdapter:
    """Every agent under test implements these two methods."""

    def start(self) -> AgentReply:
        raise NotImplementedError

    def respond(self,caller_text: str) -> AgentReply:
        raise NotImplementedError