"""An LLM that plays the caller, so the caller can react to what the agent says."""


class CallerSimulator:
    def __init__(self, llm, persona: str, goal: str, max_turns: int = 3):
        self.llm = llm
        self.max_turns = max_turns
        self.system = (
            "You are role-playing a job candidate on a phone call with an interviewer. "
            f"{persona}\n"
            f"Your goal: {goal}\n"
            "Speak the way a person talks out loud: one or two short sentences. "
            "Reply with only the words you say. No quotation marks, stage directions, "
            "markdown or emoji, because your words are turned into speech."
            "Stay in character. Never give a more complete or more correct answer than "
            "your goal describes. "
        )

    def next_line(self, transcript: list[dict]) -> str:
        """The caller's next line, given the call so far.

        From the caller's point of view the agent is the other party ('user') and the
        caller's own earlier lines are its own words ('assistant').
        """
        messages = [
            {"role": "user" if t["role"] == "agent" else "assistant", "content": t["text"]}
            for t in transcript
        ]
        line = self.llm.chat(self.system, messages).strip().strip('"')
        if not line:
            raise RuntimeError("The caller simulator returned an empty line")
        return line