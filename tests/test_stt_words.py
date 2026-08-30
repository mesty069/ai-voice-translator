import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.stt import SpeechToText, Word  # noqa: E402


class _FakeModel:
    def __init__(self, delay=0.0):
        self.calls = []
        self.delay = delay

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        time.sleep(self.delay)
        seg1 = SimpleNamespace(text=" Hello world.", words=[
            SimpleNamespace(word=" Hello", start=0.1, end=0.4),
            SimpleNamespace(word=" world.", start=0.5, end=0.9),
        ])
        seg2 = SimpleNamespace(text=" Next", words=[
            SimpleNamespace(word=" Next", start=1.2, end=1.5),
            SimpleNamespace(word="  ", start=1.5, end=1.6),   # 空字要略過
        ])
        return iter([seg1, seg2]), None


def _stt_with(model):
    stt = SpeechToText()
    stt._model = model
    return stt


def test_transcribe_words_returns_stripped_words_with_times():
    stt = _stt_with(_FakeModel())
    words = stt.transcribe_words(np.zeros(16000, dtype=np.float32), "en")
    assert words == [Word("Hello", 0.1, 0.4), Word("world.", 0.5, 0.9),
                     Word("Next", 1.2, 1.5)]


def test_transcribe_words_requests_word_timestamps_and_greedy():
    model = _FakeModel()
    stt = _stt_with(model)
    stt.transcribe_words(np.zeros(16000, dtype=np.float32), "en")
    kwargs = model.calls[0]
    assert kwargs["word_timestamps"] is True
    assert kwargs["beam_size"] == 1
    assert kwargs["language"] == "en"


def test_transcribe_words_without_model_raises():
    stt = SpeechToText()
    with pytest.raises(RuntimeError, match="尚未載入"):
        stt.transcribe_words(np.zeros(16000, dtype=np.float32), "en")


def test_transcribe_words_shares_lock_with_transcribe():
    """麥克風的 transcribe 與字幕的 transcribe_words 不得同時進 GPU。"""
    model = _FakeModel(delay=0.05)
    stt = _stt_with(model)
    active, max_active = [0], [0]
    counter_lock = threading.Lock()
    original = model.transcribe

    def counted(audio, **kwargs):
        with counter_lock:
            active[0] += 1
            max_active[0] = max(max_active[0], active[0])
        try:
            return original(audio, **kwargs)
        finally:
            with counter_lock:
                active[0] -= 1

    model.transcribe = counted
    audio = np.zeros(16000, dtype=np.float32)
    threads = [threading.Thread(target=lambda: stt.transcribe(audio, "en")),
               threading.Thread(target=lambda: stt.transcribe_words(audio, "en"))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(2.0)
    assert max_active[0] == 1
