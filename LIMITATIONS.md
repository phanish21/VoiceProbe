# Limitations

- **Warm-up run excluded.** The first run in a process pays one-time model loading
  (Whisper, Piper, first Groq connection), so `--runs N` discards one warm-up run by
  default for the voice agent. Use `--warmup 0` to include it.
- **VAD events are injected, not detected.** A real voice activity detector adds a
  delay before the agent hears that the caller stopped. VoiceProbe does not measure it.
- **Clean synthetic caller audio.** Callers are synthesized with Piper. Real phone
  audio is noisier, so STT will be slower and less accurate than reported here.
- **Pipecat's STT processing metric is not used.** It reports about 10 ms where a
  stopwatch shows about 850 ms. VoiceProbe reports the stopwatch as `stt_ms`.
- **Small samples.** Numbers from 10 runs are for development. Publish 30 or more.
- **TTS time depends on the reply text.** Piper emits one chunk per sentence, so time to
  first audio follows the length of the agent's first sentence (a one-word "Correct."
  takes about 35 ms, a full sentence about 200 ms). Leading silence also varies by text
  (up to about 180 ms observed). The agent's wording differs between runs.
- **LLM latency includes network jitter.** One greeting out of 30 waited about 15 s for
  the first byte from Groq. Medians are robust to this; means are not. Report medians and p95.