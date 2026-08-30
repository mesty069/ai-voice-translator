import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Config  # noqa: E402
from app.core.streaming_captions import Row  # noqa: E402
from app.core.system_captions import SystemCaptionsController  # noqa: E402


class _FakeStt:
    is_ready = True

    def transcribe_words(self, audio, language="en", beam_size=1):
        return []


class _FakeCapture:
    instances = []

    def __init__(self, device_name="default", on_frames=None, on_error=None):
        self.device_name = device_name
        self.on_frames = on_frames
        self.on_error = on_error
        self.started = False
        self.stopped = False
        _FakeCapture.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


class _FakeEngine:
    instances = []

    def __init__(self, buffer, stt, translator, languages, mic_busy,
                 on_rows, on_state, on_fatal, on_final, display_rows=3):
        self.buffer = buffer
        self.languages = languages
        self.on_rows, self.on_state = on_rows, on_state
        self.on_fatal, self.on_final = on_fatal, on_final
        self.display_rows = display_rows
        self.started = self.stopped = False
        _FakeEngine.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def set_display_rows(self, n):
        self.display_rows = n


def _controller(tmp_path):
    _FakeCapture.instances.clear()
    _FakeEngine.instances.clear()
    cfg = Config(tmp_path / "config.json")
    cfg.set("language", "source", "zh")
    cfg.set("language", "target", "en")
    cfg.set("system_captions", "display_rows", 4)
    ctrl = SystemCaptionsController(cfg, _FakeStt(), lambda: False)
    ctrl._capture_factory = _FakeCapture
    ctrl._engine_factory = _FakeEngine
    return cfg, ctrl


def test_start_wires_capture_into_buffer_and_engine(tmp_path):
    cfg, ctrl = _controller(tmp_path)
    ctrl.start()
    cap, eng = _FakeCapture.instances[0], _FakeEngine.instances[0]
    assert cap.started and eng.started
    assert eng.display_rows == 4
    assert eng.languages() == ("en", "zh")     # 辨識目標語言、翻成母語
    cap.on_frames(np.zeros(1600, dtype=np.float32))
    assert abs(eng.buffer.total_seconds - 0.1) < 1e-6
    ctrl.stop()
    assert cap.stopped and eng.stopped


def test_explicit_spoken_language_overrides_target(tmp_path):
    cfg, ctrl = _controller(tmp_path)
    cfg.set("system_captions", "language", "ja")
    ctrl.start()
    assert _FakeEngine.instances[0].languages() == ("ja", "zh")
    ctrl.stop()


def test_rows_and_finals_are_forwarded_as_signals(tmp_path):
    cfg, ctrl = _controller(tmp_path)
    got_rows, got_finals = [], []
    ctrl.rows_changed.connect(got_rows.append)
    ctrl.sentence_finalized.connect(lambda o, t: got_finals.append((o, t)))
    ctrl.start()
    eng = _FakeEngine.instances[0]
    eng.on_rows([Row("a", "甲", True)])
    eng.on_final("a", "甲")
    assert got_rows == [[Row("a", "甲", True)]]
    assert got_finals == [("a", "甲")]
    ctrl.stop()


def test_callbacks_from_old_generation_are_ignored(tmp_path):
    cfg, ctrl = _controller(tmp_path)
    got = []
    ctrl.rows_changed.connect(got.append)
    ctrl.start()
    old_eng, old_cap = _FakeEngine.instances[0], _FakeCapture.instances[0]
    ctrl.stop()
    old_eng.on_rows([Row("stale")])
    old_cap.on_frames(np.zeros(1600, dtype=np.float32))
    assert got == []
    ctrl.start()
    new_eng = _FakeEngine.instances[1]
    assert new_eng.buffer.total_seconds == 0.0     # 舊音框沒漏進新緩衝
    old_eng.on_rows([Row("stale again")])
    assert got == []
    ctrl.stop()


def test_capture_error_is_fatal(tmp_path):
    cfg, ctrl = _controller(tmp_path)
    fatals = []
    ctrl.fatal_error.connect(fatals.append)
    ctrl.start()
    _FakeCapture.instances[0].on_error(RuntimeError("dead"))
    assert fatals and "dead" in fatals[0]
    assert ctrl.is_running is False


def test_engine_fatal_is_forwarded(tmp_path):
    cfg, ctrl = _controller(tmp_path)
    fatals = []
    ctrl.fatal_error.connect(fatals.append)
    ctrl.start()
    _FakeEngine.instances[0].on_fatal("no model")
    assert fatals == ["no model"]
    assert ctrl.is_running is False


def test_set_display_rows_reaches_running_engine(tmp_path):
    cfg, ctrl = _controller(tmp_path)
    ctrl.start()
    ctrl.set_display_rows(2)
    assert _FakeEngine.instances[0].display_rows == 2
    ctrl.stop()
    ctrl.set_display_rows(5)   # 沒在跑也不能炸


def test_stop_is_idempotent_and_emits_idle_once(tmp_path):
    cfg, ctrl = _controller(tmp_path)
    states = []
    ctrl.state_changed.connect(lambda s, m: states.append(s))
    ctrl.start()
    ctrl.stop()
    ctrl.stop()
    assert states.count("idle") == 1


def test_real_engine_thread_starts_and_stops_quickly(tmp_path):
    """不用假引擎：確認 start/stop 真的起執行緒且 stop 不會卡 GUI。"""
    cfg = Config(tmp_path / "config.json")
    ctrl = SystemCaptionsController(cfg, _FakeStt(), lambda: False)
    ctrl._capture_factory = _FakeCapture
    ctrl.start()
    t0 = time.monotonic()
    ctrl.stop()
    assert time.monotonic() - t0 < 0.5
