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