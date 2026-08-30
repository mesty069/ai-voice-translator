import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import Qt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Config
from app.core.local_translate import ModelLoadError
from app.core.system_captions import SystemCaptionsController


class _FakeStt:
    is_ready = True

    def __init__(self):
        self.languages = []
        self.beam_sizes = []

    def transcribe(self, audio, language="zh", beam_size=5):
        self.languages.append(language)
        self.beam_sizes.append(beam_size)
        return "hello world"


class _FakeTranslator:
    def __init__(self):
        self.calls = []

    def translate(self, text, src, tgt, progress_cb=None):
        self.calls.append((text, src, tgt))
        if progress_cb is not None:
            self.progress_cb = progress_cb
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
    ctrl.stt.transcribe = lambda audio, language="zh", beam_size=5: "   "
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
    ctrl.fatal_error.connect(errors.append)
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


def test_queue_trim_keeps_newest(tmp_path):
    """佇列滿了要丟最舊的，留下最新的三段（即時性優先）。"""
    cfg, ctrl = _controller(tmp_path)
    ctrl._running = True
    for i in range(5):
        ctrl._on_segment(np.full(4, float(i), dtype=np.float32),
                         ctrl._generation)
    kept = []
    while not ctrl._queue.empty():
        kept.append(float(ctrl._queue.get_nowait()[0]))
    assert kept == [2.0, 3.0, 4.0]


def test_transient_error_does_not_fatal(tmp_path):
    """單段處理失敗只是暫時狀況（例如某段音訊有問題），
    不能因此把整個系統字幕功能關掉。"""
    cfg, ctrl = _controller(tmp_path)
    ctrl._capture_factory = lambda **kw: _FakeCapture()

    def boom(audio, language="zh"):
        raise ValueError("boom")

    ctrl.stt.transcribe = boom
    fatals, states = [], []
    # worker 是另一條執行緒，測試裡沒有事件迴圈可以派送佇列式連線，
    # 所以明確用 DirectConnection（正式環境是 GUI 執行緒的事件迴圈在收）
    direct = Qt.ConnectionType.DirectConnection
    ctrl.fatal_error.connect(fatals.append, direct)
    ctrl.state_changed.connect(lambda s, m: states.append((s, m)), direct)

    ctrl.start()
    ctrl._queue.put(np.zeros(16000, dtype=np.float32))
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if any(state == "error" for state, _ in states):
            break
        time.sleep(0.02)
    ctrl.stop()

    assert fatals == [], "暫時性錯誤不該觸發致命錯誤（會自動關閉功能）"
    assert any(state == "error" and "boom" in msg for state, msg in states)
    assert ctrl._worker is None


def test_model_load_error_is_fatal_and_stops_worker(tmp_path):
    """翻譯模型下載/載入失敗要用型別（ModelLoadError）判斷致命，不是訊息
    字串前綴：否則之後每一段都會重新嘗試下載，永遠不會真的停止。"""
    cfg, ctrl = _controller(tmp_path)
    ctrl._capture_factory = lambda **kw: _FakeCapture()

    def boom(text, src, tgt, progress_cb=None):
        raise ModelLoadError("翻譯模型載入失敗：連線逾時")

    ctrl._translator.translate = boom
    fatals, states = [], []
    direct = Qt.ConnectionType.DirectConnection
    ctrl.fatal_error.connect(fatals.append, direct)
    ctrl.state_changed.connect(lambda s, m: states.append((s, m)), direct)

    ctrl.start()
    ctrl._queue.put(np.zeros(16000, dtype=np.float32))
    worker = ctrl._worker
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not fatals:
        time.sleep(0.02)
    worker.join(timeout=2.0)
    ctrl.stop()

    assert len(fatals) == 1
    assert not worker.is_alive(), "模型載不起來要停止 worker，不能每段都重試下載"


def test_transient_translate_error_does_not_fatal(tmp_path):
    """單段翻譯失敗（非 ModelLoadError）只是暫時狀況，不能關掉整個功能；
    走 state_changed("error", ...)，不是已移除的 error_occurred。"""
    cfg, ctrl = _controller(tmp_path)
    ctrl._capture_factory = lambda **kw: _FakeCapture()

    def boom(text, src, tgt, progress_cb=None):
        raise RuntimeError("x")

    ctrl._translator.translate = boom
    fatals, states = [], []
    direct = Qt.ConnectionType.DirectConnection
    ctrl.fatal_error.connect(fatals.append, direct)
    ctrl.state_changed.connect(lambda s, m: states.append((s, m)), direct)

    ctrl.start()
    ctrl._queue.put(np.zeros(16000, dtype=np.float32))
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if any(state == "error" for state, _ in states):
            break
        time.sleep(0.02)
    worker = ctrl._worker
    assert worker is not None and worker.is_alive(), "暫時性錯誤不能讓 worker 停掉"
    ctrl.stop()

    assert fatals == []
    assert any(state == "error" and "x" in msg for state, msg in states)


class _LateReadyStt:
    """模擬語音模型還在載入：0.3 秒後才就緒。"""

    def __init__(self, delay=0.3):
        self._ready_at = time.monotonic() + delay

    @property
    def is_ready(self):
        return time.monotonic() >= self._ready_at

    def transcribe(self, audio, language="zh", beam_size=5):
        assert self.is_ready, "模型還沒載入完就去辨識了"
        return "hello world"


def test_waits_for_stt_ready(tmp_path):
    """開機就按熱鍵時語音模型還在載入：要等它好，不能直接報錯關掉功能。"""
    cfg, ctrl = _controller(tmp_path)
    ctrl.stt = _LateReadyStt(0.3)
    ctrl._running = True
    states, captions, fatals = [], [], []
    ctrl.state_changed.connect(lambda s, m: states.append((s, m)))
    ctrl.caption_ready.connect(lambda a, b: captions.append((a, b)))
    ctrl.fatal_error.connect(fatals.append)

    started = time.monotonic()
    ctrl._process(np.zeros(16000, dtype=np.float32), ctrl._generation)

    assert time.monotonic() - started >= 0.25
    assert states and states[0][0] == "loading"
    assert captions == [("hello world", "你好世界")]
    assert fatals == []


class _NeverReadyStt:
    """模擬語音模型永遠不會就緒（例如載入卡死）。"""

    is_ready = False

    def transcribe(self, audio, language="zh", beam_size=5):
        raise AssertionError("不該辨識：STT 從未就緒")


def test_stt_wait_timeout_is_fatal(tmp_path):
    """語音模型一直沒就緒：等到逾時要當成致命錯誤，不能無限等下去，
    也不能悄悄跳過該段。STT_WAIT_TIMEOUT 是類別屬性，測試可以覆寫成很短。"""
    cfg, ctrl = _controller(tmp_path)
    ctrl.STT_WAIT_TIMEOUT = 0.05
    ctrl.stt = _NeverReadyStt()
    ctrl._running = True
    fatals, captions = [], []
    ctrl.fatal_error.connect(fatals.append)
    ctrl.caption_ready.connect(lambda a, b: captions.append((a, b)))

    ctrl._process(np.zeros(16000, dtype=np.float32), ctrl._generation)

    assert fatals == ["語音模型載入逾時，系統字幕已停止"]
    assert captions == []


def test_source_text_is_emitted_before_translation(tmp_path):
    """原文先上字幕、翻譯完成再補：體感延遲減半。"""
    cfg, ctrl = _controller(tmp_path)
    ctrl._running = True
    events = []
    ctrl.source_ready.connect(lambda a: events.append(("source", a)))
    ctrl.caption_ready.connect(lambda a, b: events.append(("caption", a, b)))
    ctrl._process(np.zeros(16000, dtype=np.float32), ctrl._generation)
    assert events == [("source", "hello world"),
                      ("caption", "hello world", "你好世界")]


def test_live_captions_use_greedy_decoding(tmp_path):
    """即時字幕用 beam_size=1（麥克風翻譯維持 5），速度優先。"""
    cfg, ctrl = _controller(tmp_path)
    ctrl._running = True
    ctrl._process(np.zeros(16000, dtype=np.float32), ctrl._generation)
    assert ctrl.stt.beam_sizes == [1]
