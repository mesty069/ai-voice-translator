import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.system_audio import (  # noqa: E402
    SAMPLE_RATE,
    RollingAudioBuffer,
    SystemAudioCapture,
)


def _sec(n, value=0.5):
    return np.full(int(SAMPLE_RATE * n), value, dtype=np.float32)


def test_append_advances_absolute_time():
    buf = RollingAudioBuffer()
    buf.append(_sec(1.0))
    buf.append(_sec(0.5))
    assert abs(buf.total_seconds - 1.5) < 1e-6
    assert buf.start_seconds == 0.0


def test_since_returns_audio_from_absolute_time():
    buf = RollingAudioBuffer()
    buf.append(_sec(1.0, 0.1))
    buf.append(_sec(1.0, 0.2))
    out = buf.since(1.0)
    assert len(out) == SAMPLE_RATE
    assert np.allclose(out, 0.2)


def test_trim_keeps_absolute_time_axis():
    buf = RollingAudioBuffer()
    buf.append(_sec(2.0, 0.1))
    buf.trim_before(1.5)
    buf.append(_sec(1.0, 0.3))
    assert abs(buf.start_seconds - 1.5) < 1e-6
    assert abs(buf.total_seconds - 3.0) < 1e-6
    out = buf.since(2.0)          # 絕對秒 2.0 起 = 剛加進去的那 1 秒
    assert len(out) == SAMPLE_RATE
    assert np.allclose(out, 0.3)


def test_since_before_start_clamps_to_start():
    buf = RollingAudioBuffer()
    buf.append(_sec(2.0, 0.1))
    buf.trim_before(1.0)
    assert len(buf.since(0.0)) == SAMPLE_RATE


def test_max_seconds_auto_trims():
    buf = RollingAudioBuffer(max_seconds=2.0)
    buf.append(_sec(3.0))
    assert buf.total_seconds - buf.start_seconds <= 2.0 + 1e-6
    assert abs(buf.total_seconds - 3.0) < 1e-6


def test_tail_peak():
    buf = RollingAudioBuffer()
    buf.append(_sec(1.0, 0.8))
    buf.append(_sec(1.0, 0.0))
    assert buf.tail_peak(0.5) == 0.0
    assert buf.tail_peak(1.5) > 0.7
    assert RollingAudioBuffer().tail_peak(1.0) == 0.0


def test_since_with_empty_buffer_is_empty():
    assert len(RollingAudioBuffer().since(0.0)) == 0


# ---- SystemAudioCapture：假 soundcard，確認每塊音框都原樣送出 ----

class _FakeRecorder:
    def __init__(self, blocks):
        self._blocks = list(blocks)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def record(self, numframes):
        if self._blocks:
            return self._blocks.pop(0)
        import time
        time.sleep(0.01)
        return np.zeros((numframes, 1), dtype=np.float32)


class _FakeMic:
    def __init__(self, blocks):
        self._blocks = blocks

    def recorder(self, samplerate, channels):
        return _FakeRecorder(self._blocks)


class _FakeSpeaker:
    name = "Fake Speaker"


class _FakeSoundcard:
    def __init__(self, blocks):
        self._blocks = blocks

    def default_speaker(self):
        return _FakeSpeaker()

    def all_speakers(self):
        return [_FakeSpeaker()]

    def get_microphone(self, name, include_loopback=False):
        return _FakeMic(self._blocks)


def test_capture_forwards_every_block(monkeypatch):
    import time
    blocks = [np.full((1600, 1), 0.1, dtype=np.float32),
              np.full((1600, 1), 0.2, dtype=np.float32)]
    monkeypatch.setitem(sys.modules, "soundcard", _FakeSoundcard(blocks))
    got = []
    cap = SystemAudioCapture(on_frames=lambda f: got.append(f.copy()))
    cap.start()
    deadline = time.monotonic() + 2.0
    while len(got) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    cap.stop()
    assert len(got) >= 2
    assert got[0].ndim == 1 and len(got[0]) == 1600
    assert np.allclose(got[0], 0.1) and np.allclose(got[1], 0.2)


def test_capture_error_is_reported(monkeypatch):
    import time

    class _Broken(_FakeSoundcard):
        def get_microphone(self, name, include_loopback=False):
            raise RuntimeError("no loopback")

    monkeypatch.setitem(sys.modules, "soundcard", _Broken([]))
    errors = []
    cap = SystemAudioCapture(on_error=errors.append)
    cap.start()
    deadline = time.monotonic() + 2.0
    while not errors and time.monotonic() < deadline:
        time.sleep(0.01)
    assert errors and "no loopback" in str(errors[0])
    assert cap.is_running is False
