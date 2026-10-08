# VoiceProbe Lite
![tests](https://github.com/phanish21/VoiceProbe/actions/workflows/tests.yml/badge.svg)

**Unit tests for voice agents: catch latency and quality regressions before your users do.**

Work in progress. An open-source, CI-first test harness for voice and LLM agents.

## Quickstart

    pip install -e ".[dev]"
    voiceprobe run scenarios/wrong_answer.yaml --mock
    pytest