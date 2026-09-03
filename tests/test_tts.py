import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import tts  # noqa: E402

SAMPLERATE = 16000


class _FakeStream:
    def __init__(self):
        self.active = False


class _FakeSd:
    """假的 sounddevice：記下每次 play 的資料，stop 就把串流關掉。"""

    def __init__(self):
        self.plays = []      # [(data, samplerate, device)]
        self.stops = 0
        self._stream = _FakeStream()

    def play(self, data, samplerate, device=None):
        self.plays.append((np.asarray(data), samplerate, device))
        self._stream.active = True

    def stop(self):
        self.stops += 1
        self._stream.active = False

    def get_stream(self):
        return self._stream

    def finish(self):
        """模擬播放自然結束。"""
        self._stream.active = False

    def query_devices(self, idx=None):
        return []

    def query_hostapis(self, idx=None):
        return None


class _FakeSf:
    def __init__(self, data):
        self.data = data

    def read(self, path):
        return self.data, SAMPLERATE


def _wait_for(pred, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def audio():
    # 3 秒的音訊，暫停時 offset 一定落在中間
    return np.linspace(0.0, 1.0, SAMPLERATE * 3, dtype="float32")


@pytest.fixture
def fake_sd(monkeypatch, audio):
    sd = _FakeSd()
    monkeypatch.setattr(tts, "sd", sd)
    monkeypatch.setattr(tts, "sf", _FakeSf(audio))
    return sd


def _worker(monkeypatch):
    worker = tts._TTSWorker()
    monkeypatch.setattr(worker, "_synthesize",
                        lambda text, rate, lang="en": "fake.wav")
    return worker


class _Done:
    def __init__(self):
        self.calls = []

    def __call__(self, completed, error):
        self.calls.append((completed, error))


def test_volume_scales_played_data(monkeypatch, fake_sd, audio):
    worker = _worker(monkeypatch)
    worker.submit("hello", "default", 200, "en", volume=0.5)
    assert _wait_for(lambda: fake_sd.plays)
    played = fake_sd.plays[0][0]
    assert np.allclose(played, audio * 0.5, atol=1e-6)


def test_volume_clips_to_unit_range(monkeypatch, audio):
    sd = _FakeSd()
    monkeypatch.setattr(tts, "sd", sd)
    monkeypatch.setattr(tts, "sf", _FakeSf(np.array([2.0, -2.0, 0.5],
                                                    dtype="float32")))
    worker = _worker(monkeypatch)
    worker.submit("hello", "default", 200, "en", volume=0.9)
    assert _wait_for(lambda: sd.plays)
    played = sd.plays[0][0]
    assert np.allclose(played, [1.0, -1.0, 0.45], atol=1e-6)


def test_volume_one_plays_untouched_data(monkeypatch, fake_sd, audio):
    worker = _worker(monkeypatch)
    worker.submit("hello", "default", 200, "en")
    assert _wait_for(lambda: fake_sd.plays)
    assert np.allclose(fake_sd.plays[0][0], audio)


def test_pause_stops_playback_without_calling_on_done(monkeypatch, fake_sd):
    worker = _worker(monkeypatch)
    done = _Done()
    worker.submit("hello", "default", 200, "en", on_done=done)
    assert _wait_for(lambda: fake_sd.plays)
    time.sleep(0.15)  # 讓已播樣本數大於 0
    stops_before = fake_sd.stops
    assert worker.pause() is True
    assert _wait_for(lambda: worker.is_paused)
    assert fake_sd.stops > stops_before
    assert worker.is_playing is True      # 暫停仍算「朗讀中」
    time.sleep(0.15)
    assert done.calls == []               # 暫停不觸發 on_done
    assert worker.played_offset > 0


def test_resume_plays_from_recorded_offset(monkeypatch, fake_sd, audio):
    worker = _worker(monkeypatch)
    done = _Done()
    worker.submit("hello", "default", 200, "en", on_done=done)
    assert _wait_for(lambda: fake_sd.plays)
    time.sleep(0.15)
    worker.pause()
    assert _wait_for(lambda: worker.is_paused)
    offset = worker.played_offset
    assert worker.resume() is True
    assert _wait_for(lambda: len(fake_sd.plays) == 2)
    resumed = fake_sd.plays[1][0]
    assert len(resumed) == len(audio) - offset
    assert np.allclose(resumed, audio[offset:], atol=1e-6)
    assert worker.is_paused is False
    # 續播完照樣回報 completed=True
    fake_sd.finish()
    assert _wait_for(lambda: done.calls == [(True, None)])
    assert worker.is_playing is False


def test_resume_applies_new_volume(monkeypatch, fake_sd, audio):
    worker = _worker(monkeypatch)
    worker.submit("hello", "default", 200, "en")
    assert _wait_for(lambda: fake_sd.plays)
    time.sleep(0.15)
    worker.pause()
    assert _wait_for(lambda: worker.is_paused)
    offset = worker.played_offset
    worker.resume(volume=0.25)
    assert _wait_for(lambda: len(fake_sd.plays) == 2)
    assert np.allclose(fake_sd.plays[1][0], audio[offset:] * 0.25, atol=1e-6)


def test_new_submit_clears_paused_state(monkeypatch, fake_sd, audio):
    worker = _worker(monkeypatch)
    done = _Done()
    worker.submit("hello", "default", 200, "en", on_done=done)
    assert _wait_for(lambda: fake_sd.plays)
    time.sleep(0.15)
    worker.pause()
    assert _wait_for(lambda: worker.is_paused)
    worker.submit("world", "default", 200, "en")
    assert _wait_for(lambda: len(fake_sd.plays) == 2)
    assert worker.is_paused is False
    assert len(fake_sd.plays[1][0]) == len(audio)   # 新的一句從頭播
    # 舊請求被取代 → completed=False
    assert _wait_for(lambda: done.calls and done.calls[0] == (False, None))


def test_pause_when_idle_does_nothing(monkeypatch, fake_sd):
    worker = _worker(monkeypatch)
    assert worker.pause() is False
    assert worker.resume() is False
    assert worker.is_playing is False
    assert worker.is_paused is False


def test_module_level_playback_state(monkeypatch, fake_sd):
    monkeypatch.setattr(tts, "_worker", _worker(monkeypatch))
    assert tts.playback_state() == {"playing": False, "paused": False}
    tts.submit("hello", "default", 200, "en")
    assert _wait_for(lambda: fake_sd.plays)
    assert tts.playback_state()["playing"] is True
    time.sleep(0.15)
    assert tts.pause() is True
    assert _wait_for(lambda: tts.playback_state()["paused"])
    assert tts.resume() is True
    assert _wait_for(lambda: len(fake_sd.plays) == 2)
    assert tts.playback_state() == {"playing": True, "paused": False}
