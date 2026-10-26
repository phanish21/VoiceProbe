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

def to_16k_mono(path, target_rate: int = 16000) -> bytes:
    """WAV file -> raw 16-bit mono PCM at target_rate (needs numpy and soxr, installed with Pipecat)."""
    import numpy as np
    import soxr

    pcm, rate, channels = read_wav(path)
    if channels != 1:
        raise ValueError(f"{path}: expected mono audio, got {channels} channels")
    samples = np.frombuffer(pcm, dtype=np.int16)
    if rate != target_rate:
        samples = soxr.resample(samples, rate, target_rate)
    return samples.astype(np.int16).tobytes()


def chunk_pcm(pcm: bytes, pad_s: float = 0.3, rate: int = 16000, chunk_ms: int = 20) -> list[bytes]:
    """Append pad_s seconds of silence, then split into equal chunks (20 ms = 640 bytes at 16 kHz)."""
    size = rate * chunk_ms // 1000 * 2
    pcm = pcm + b"\x00" * (int(pad_s * rate) * 2)
    pcm += b"\x00" * (-len(pcm) % size)  # pad the last chunk so every chunk is full size
    return [pcm[i : i + size] for i in range(0, len(pcm), size)]