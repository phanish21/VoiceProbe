# VoiceProbe Lite
![tests](https://github.com/phanish21/VoiceProbe/actions/workflows/tests.yml/badge.svg)

**Unit tests for voice agents: catch latency and quality regressions before your users do.**

Work in progress. An open-source, CI-first test harness for voice and LLM agents.

## Quickstart

```bash
git clone https://github.com/phanish21/VoiceProbe
cd VoiceProbe
pip install -e ".[dev]"
pytest -v

# Offline, no API key
voiceprobe run scenarios/wrong_answer.yaml --mock
```

To test a real LLM, put `GROQ_API_KEY=...` in a `.env` file (git-ignored), then:

```bash
voiceprobe run scenarios/wrong_answer.yaml --no-cache
```

### Voice agent (optional)

```bash
pip install -e ".[pipecat]"
mkdir voices
python -m piper.download_voices en_US-lessac-medium --download-dir voices
voiceprobe run scenarios/correct_answer.yaml --agent pipecat-voice --runs 30
```

Runs 30 times plus one discarded warm-up run, and reports median and p95 for
greeting turns and reply turns separately.