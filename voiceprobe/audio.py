import hashlib
import wave
from pathlib import Path


class CallerVoice:
    """Turns caller text into cached WAV files using Piper (local, $0)."""

    def __init__(self, model_path="voices/en_US-lessac-medium.onnx", cache_dir="audio_cache"):
        self.model_path = Path(model_path)
        self.cache = Path(cache_dir)
        self.cache.mkdir(exist_ok=True)
        self._voice = None  # loaded on first use, then reused

    def _load(self):
        if self._voice is None:
            from piper import PiperVoice  # lazy: only needed if we must synthesize

            self._voice = PiperVoice.load(str(self.model_path))
        return self._voice

    def wav_path(self, text: str) -> Path:
        key = hashlib.sha256(f"{self.model_path.name}|{text}".encode()).hexdigest()[:16]
        return self.cache / f"{key}.wav"

    def synthesize(self, text: str) -> Path:
        path = self.wav_path(text)
        if path.exists():
            return path
        voice = self._load()
        tmp = path.with_suffix(".tmp")
        with wave.open(str(tmp), "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)
        tmp.replace(path)  # only a finished file ever gets the real name
        return path


def read_wav(path) -> tuple[bytes, int, int]:
    """Returns (raw 16-bit PCM bytes, sample rate in Hz, channel count)."""
    with wave.open(str(path), "rb") as f:
        if f.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit audio, got {8 * f.getsampwidth()}-bit")
        return f.readframes(f.getnframes()), f.getframerate(), f.getnchannels()


def duration_s(path) -> float:
    with wave.open(str(path), "rb") as f:
        return f.getnframes() / f.getframerate()