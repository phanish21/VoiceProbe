import wave

import pytest

from voiceprobe.audio import CallerVoice, duration_s, read_wav, chunk_pcm

def make_wav(path, seconds=0.5, rate=16000, width=2):
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(width)
        f.setframerate(rate)
        f.writeframes(b"\x00" * int(seconds * rate) * width)


def test_cache_key_is_stable_and_text_specific(tmp_path):
    voice = CallerVoice(cache_dir=str(tmp_path))
    assert voice.wav_path("hello") == voice.wav_path("hello")
    assert voice.wav_path("hello") != voice.wav_path("goodbye")


def test_cache_key_depends_on_the_voice(tmp_path):
    a = CallerVoice(model_path="voices/a.onnx", cache_dir=str(tmp_path))
    b = CallerVoice(model_path="voices/b.onnx", cache_dir=str(tmp_path))
    assert a.wav_path("hello") != b.wav_path("hello")


def test_cached_line_is_returned_without_loading_piper(tmp_path):
    voice = CallerVoice(model_path="voices/does_not_exist.onnx", cache_dir=str(tmp_path))
    make_wav(voice.wav_path("hello"))
    assert voice.synthesize("hello") == voice.wav_path("hello")  # would crash if it tried Piper


def test_read_wav_and_duration(tmp_path):
    path = tmp_path / "a.wav"
    make_wav(path, seconds=0.5, rate=16000)
    pcm, rate, channels = read_wav(path)
    assert (rate, channels) == (16000, 1)
    assert len(pcm) == 16000  # 8000 frames * 2 bytes each
    assert duration_s(path) == pytest.approx(0.5)


def test_read_wav_rejects_non_16_bit(tmp_path):
    path = tmp_path / "eight_bit.wav"
    make_wav(path, width=1)
    with pytest.raises(ValueError):
        read_wav(path)

def test_chunk_pcm_pads_and_splits_evenly():
    pcm = b"\x01\x00" * 1000                       # 1000 samples = 2000 bytes
    chunks = chunk_pcm(pcm, pad_s=0.1)              # + 1600 samples of silence = 5200 bytes
    assert len(chunks) == 9                         # padded up to 5760 bytes = 9 chunks of 640
    assert all(len(c) == 640 for c in chunks)
    joined = b"".join(chunks)
    assert joined[:2000] == pcm                     # original audio untouched
    assert set(joined[2000:]) == {0}                # everything after it is silence