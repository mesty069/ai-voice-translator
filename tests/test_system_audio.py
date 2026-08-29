import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.system_audio import SAMPLE_RATE, SegmentAccumulator


def _speech(seconds):
    n = int(SAMPLE_RATE * seconds)
    t = np.arange(n) / SAMPLE_RATE
    return (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _silence(seconds):
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)


def _feed(acc, audio, block=1600):
    out = []
    for i in range(0, len(audio), block):
        out.extend(acc.push(audio[i:i + block]))
    return out


def test_silence_produces_nothing():
    acc = SegmentAccumulator()
    assert _feed(acc, _silence(3.0)) == []


def test_speech_then_silence_emits_one_segment():
    acc = SegmentAccumulator(silence_ms=600, min_seconds=0.8)
    segments = _feed(acc, np.concatenate([_speech(2.0), _silence(1.0)]))
    assert len(segments) == 1
    assert len(segments[0]) >= int(SAMPLE_RATE * 2.0)


def test_short_blip_is_discarded():
    acc = SegmentAccumulator(silence_ms=600, min_seconds=0.8)
    assert _feed(acc, np.concatenate([_speech(0.3), _silence(1.0)])) == []


def test_long_speech_is_force_cut():
    acc = SegmentAccumulator(silence_ms=600, max_seconds=4.0)
    segments = _feed(acc, _speech(9.0))
    assert len(segments) >= 2
    for seg in segments:
        assert len(seg) <= int(SAMPLE_RATE * 4.0) + 1600


def test_speech_after_force_cut_still_accumulates():
    acc = SegmentAccumulator(silence_ms=600, max_seconds=2.0, min_seconds=0.5)
    first = _feed(acc, _speech(5.0))
    rest = _feed(acc, _silence(1.0))
    assert len(first) >= 2
    assert len(rest) == 1


def test_drain_returns_pending_audio():
    acc = SegmentAccumulator(min_seconds=0.5)
    acc.push(_speech(1.0))
    tail = acc.drain()
    assert tail is not None
    assert len(tail) >= int(SAMPLE_RATE * 1.0)


def _background(seconds, amplitude=0.03):
    """模擬桌面持續背景音（實測本機為 0.03 以上）。"""
    n = int(SAMPLE_RATE * seconds)
    t = np.arange(n) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * 60 * t)).astype(np.float32)


def test_constant_background_does_not_block_segmentation():
    """回歸測試：持續背景音高於任何固定門檻時，仍要切得出段落。"""
    acc = SegmentAccumulator(silence_ms=600, min_seconds=0.8)
    audio = np.concatenate([_background(1.0), _speech(2.0), _background(1.5)])
    segments = _feed(acc, audio)
    assert len(segments) == 1


def test_pure_silence_segment_is_discarded_on_force_cut():
    """整段都是數位靜音時，即使觸發強制切段也不該送出。"""
    acc = SegmentAccumulator(silence_ms=600, max_seconds=2.0, min_seconds=0.5)
    assert _feed(acc, _silence(6.0)) == []


class _FakeRecorder:
    """假的 loopback 錄音器：固定吐出語音，吐滿指定塊數後讓擷取停止。"""

    def __init__(self, capture, blocks):
        self._capture = capture
        self._blocks = blocks
        self._sent = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def record(self, numframes):
        self._sent += 1
        if self._sent >= self._blocks:
            self._capture._running = False   # 模擬使用者按下停止
        return _speech(numframes / SAMPLE_RATE)


class _FakeSpeaker:
    name = "假喇叭"


class _FakeSoundcard:
    def __init__(self, capture, blocks):
        self._capture = capture
        self._blocks = blocks

    def default_speaker(self):
        return _FakeSpeaker()

    def all_speakers(self):
        return [_FakeSpeaker()]

    def get_microphone(self, name, include_loopback=False):
        recorder = _FakeRecorder(self._capture, self._blocks)
        return type("Mic", (), {"recorder": lambda _s, **kw: recorder})()


def test_stop_flushes_pending_tail(monkeypatch):
    """停止擷取時，最後一句還沒遇到停頓的尾段也要送出，不能整句吞掉。"""
    from app.core import system_audio

    segments = []
    capture = system_audio.SystemAudioCapture(on_segment=segments.append)
    # 10 塊 × 0.1 秒 ＝ 1 秒語音：不到 max_seconds、也沒有停頓，
    # 所以整段只會停在緩衝裡，唯有停止時的 drain 才會送出
    monkeypatch.setitem(sys.modules, "soundcard",
                        _FakeSoundcard(capture, blocks=10))
    capture.start()
    capture._thread.join(timeout=5.0)

    assert len(segments) == 1, "停止時未送出尾段"
    assert len(segments[0]) >= int(SAMPLE_RATE * 0.8)
