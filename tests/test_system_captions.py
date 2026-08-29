import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Config
from app.core.system_captions import SystemCaptionsController


class _FakeStt:
    def __init__(self):
        self.languages = []

    def transcribe(self, audio, language="zh"):
        self.languages.append(language)
        return "hello world"


class _FakeTranslator:
    def __init__(self):
        self.calls = []

    def translate(self, text, src, tgt):
        self.calls.append((text, src, tgt))
        return "你好世界"

    def set_engine(self, engine):
        pass

    def set_compute_device(self, device):
        pass


def _controller(tmp_path, mic_busy=lambda: False):
    cfg = Config(tmp_path / "config.json")
    cfg.set("language", "source", "zh")
    cfg.set("language", "target", "en")
    ctrl = SystemCaptionsController(cfg, _FakeStt(), mic_busy)
    ctrl._translator = _FakeTranslator()
    return cfg, ctrl


def test_segment_produces_bilingual_caption(tmp_path):
    cfg, ctrl = _controller(tmp_path)
    captions = []
    ctrl.caption_ready.connect(lambda a, b: captions.append((a, b)))
    ctrl._process(np.zeros(16000, dtype=np.float32))
    assert captions == [("hello world", "你好世界")]


def test_uses_target_language_for_recognition_and_native_for_translation(tmp_path):
    cfg, ctrl = _controller(tmp_path)
    ctrl._process(np.zeros(16000, dtype=np.float32))
    assert ctrl.stt.languages == ["en"]          # 系統聲音＝目標語言
    assert ctrl._translator.calls[0][1:] == ("en", "zh")  # 翻成母語


def test_explicit_language_overrides_target(tmp_path):
    cfg, ctrl = _controller(tmp_path)
    cfg.set("system_captions", "language", "ja")
    ctrl._process(np.zeros(16000, dtype=np.float32))
    assert ctrl.stt.languages == ["ja"]
    assert ctrl._translator.calls[0][1:] == ("ja", "zh")


def test_empty_transcription_emits_nothing(tmp_path):
    cfg, ctrl = _controller(tmp_path)
    ctrl.stt.transcribe = lambda audio, language="zh": "   "
    captions = []
    ctrl.caption_ready.connect(lambda a, b: captions.append((a, b)))
    ctrl._process(np.zeros(16000, dtype=np.float32))
    assert captions == []


def test_waits_while_microphone_is_busy(tmp_path):
    busy = {"value": True}
    cfg, ctrl = _controller(tmp_path, mic_busy=lambda: busy["value"])
    ctrl._mic_wait_timeout = 1.0
    started = time.monotonic()

    def release():
        time.sleep(0.3)
        busy["value"] = False

    import threading
    threading.Thread(target=release, daemon=True).start()
    ctrl._process(np.zeros(16000, dtype=np.float32))
    assert time.monotonic() - started >= 0.25


def test_queue_drops_backlog_to_stay_realtime(tmp_path):
    cfg, ctrl = _controller(tmp_path)
    for _ in range(10):
        ctrl._on_segment(np.zeros(1600, dtype=np.float32))
    assert ctrl._queue.qsize() <= ctrl.MAX_QUEUE
