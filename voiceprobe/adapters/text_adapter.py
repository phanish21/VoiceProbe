import time

from .base import AgentAdapter, AgentReply

class TextAdapter(AgentAdapter):
    def __init__(self, llm, system_prompt: str):
        self.llm = llm
        self.system = system_prompt
        self.history: list[dict] = []

    def _call(self) -> AgentReply:
        t0 = time.perf_counter()
        text = self.llm.chat(self.system, self.history)
        ms = (time.perf_counter() - t0) * 1000
        self.history.append({"role": "assistant", "content": text})
        return AgentReply(text=text, llm_ms=ms)
    
    def start(self) -> AgentReply:
        self.history.append({"role": "user", "content": "(call connected)"})
        return self._call()

    def respond(self, caller_text: str) -> AgentReply:
        self.history.append({"role": "user", "content": caller_text})
        return self._call()