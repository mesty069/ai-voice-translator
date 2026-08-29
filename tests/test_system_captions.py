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
    ctrl._running = True
    captions = []
    ctrl.caption_ready.connect(lambda a, b: captions.append((a, b)))
    ctrl._process(np.zeros(16000, dtype=np.float32), ctrl._generation)
    assert captions == [("hello world", "你好世界")]


def test_uses_target_language_for_recognition_and_native_for_translation(tmp_path):
    cfg, ctrl = _controller(tmp_path)
    ctrl._running = True
    ctrl._process(np.zeros(16000, dtype=np.float32), ctrl._generation)
    assert ctrl.stt.languages == ["en"]          # 系統聲音＝目標語言
    assert ctrl._translator.calls[0][1:] == ("en", "zh")  # 翻成母語


def test_explicit_language_overrides_target(tmp_path):
    cfg, ctrl = _controller(tmp_path)
    cfg.set("system_captions", "language", "ja")
    ctrl._running = True
    ctrl._process(np.zeros(16000, dtype=np.float32), ctrl._generation)
    assert ctrl.stt.languages == ["ja"]
    assert ctrl._translator.calls[0][1:] == ("ja", "zh")


def test_empty_transcription_emits_nothing(tmp_path):
    cfg, ctrl = _controller(tmp_path)
    ctrl._running = True
    ctrl.stt.transcribe = lambda audio, language="zh": "   "
    captions = []
    ctrl.caption_ready.connect(lambda a, b: captions.append((a, b)))
    ctrl._process(np.zeros(16000, dtype=np.float32), ctrl._generation)
    assert captions == []


def test_waits_while_microphone_is_busy(tmp_path):
    busy = {"value": True}
    cfg, ctrl = _controller(tmp_path, mic_busy=lambda: busy["value"])
    ctrl._running = True
    ctrl._mic_wait_timeout = 1.0
    started = time.monotonic()

    def release():
        time.sleep(0.3)
        busy["value"] = False

    import threading
    threading.Thread(target=release, daemon=True).start()
    ctrl._process(np.zeros(16000, dtype=np.float32), ctrl._generation)
    assert time.monotonic() - started >= 0.25


def test_queue_drops_backlog_to_stay_realtime(tmp_path):
    cfg, ctrl = _controller(tmp_path)
    ctrl._running = True
    for _ in range(10):
        ctrl._on_segment(np.zeros(1600, dtype=np.float32), ctrl._generation)
    assert ctrl._queue.qsize() <= ctrl.MAX_QUEUE


class _FakeCapture:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.is_running = False

    def start(self):
        self.is_running = True

    def stop(self):
        self.is_running = False


def test_stale_worker_does_not_emit_after_stop(tmp_path):
    """stop() 後才完成的舊 worker 不得再發字幕。"""
    cfg, ctrl = _controller(tmp_path)
    captions = []
    ctrl.caption_ready.connect(lambda a, b: captions.append((a, b)))
    ctrl._running = True
    gen = ctrl._generation
    ctrl._generation += 1          # 模擬 stop()/restart 已發生
    ctrl._process(np.zeros(16000, dtype=np.float32), gen)
    assert captions == []


def test_restart_replaces_queue_and_generation(tmp_path):
    """重新 start() 必須換新佇列並遞增世代，舊 worker 無法再消費。"""
    cfg, ctrl = _controller(tmp_path)
    ctrl._capture_factory = lambda **kw: _FakeCapture()
    ctrl.start()
    first_queue, first_gen = ctrl._queue, ctrl._generation
    ctrl.stop()
    ctrl.start()
    assert ctrl._queue is not first_queue
    assert ctrl._generation > first_gen
    ctrl.stop()


def test_capture_error_wakes_worker(tmp_path):
    """擷取失敗後 worker 不得永遠卡在 queue.get()。"""
    cfg, ctrl = _controller(tmp_path)
    ctrl._capture_factory = lambda **kw: _FakeCapture()
    ctrl.start()
    worker = ctrl._worker
    ctrl._on_capture_error(RuntimeError("裝置消失"), ctrl._generation)
    worker.join(timeout=2.0)
    assert not worker.is_alive()


def test_stale_capture_error_is_ignored(tmp_path):
    """重啟後，舊擷取執行緒遲來的錯誤不得影響新一代 worker。"""
    cfg, ctrl = _controller(tmp_path)
    ctrl._capture_factory = lambda **kw: _FakeCapture()
    errors = []
    ctrl.error_occurred.connect(errors.append)
    ctrl.start()
    old_gen = ctrl._generation
    ctrl.stop()
    ctrl.start()
    worker = ctrl._worker
    ctrl._on_capture_error(RuntimeError("舊裝置消失"), old_gen)
    assert errors == []
    assert ctrl._running is True
    assert worker.is_alive()
    ctrl.stop()


def test_stale_capture_segment_is_dropped(tmp_path):
    """舊擷取執行緒的音訊段不得進入新一代的佇列。"""
    cfg, ctrl = _controller(tmp_path)
    ctrl._capture_factory = lambda **kw: _FakeCapture()
    ctrl.start()
    old_gen = ctrl._generation
    ctrl.stop()
    ctrl.start()
    ctrl._on_segment(np.zeros(1600, dtype=np.float32), old_gen)
    assert ctrl._queue.qsize() == 0
    ctrl.stop()
