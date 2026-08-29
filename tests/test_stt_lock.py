import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.stt import SpeechToText


class _SlowFakeModel:
    """記錄是否有兩個 transcribe 同時進行。"""

    def __init__(self):
        self.active = 0
        self.max_active = 0
        self._guard = threading.Lock()

    def transcribe(self, audio, **kwargs):
        with self._guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.15)
        with self._guard:
            self.active -= 1
        return ([], None)


def test_transcribe_is_serialized():
    stt = SpeechToText()
    fake = _SlowFakeModel()
    stt._model = fake
    audio = np.zeros(1600, dtype=np.float32)

    threads = [threading.Thread(target=stt.transcribe, args=(audio,))
               for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert fake.max_active == 1, "transcribe 沒有序列化，兩條管線會同時擠 GPU"


class _FastFakeModel:
    def __init__(self, name):
        self.name = name
        self.used = 0

    def transcribe(self, audio, **kwargs):
        self.used += 1
        return ([], None)


def test_reload_does_not_race_transcribe():
    """transcribe 在鎖外檢查 self._model，reload 在這個空隙把它清成 None，
    進了鎖再取用就會 AttributeError。取用必須在鎖內、且取本地變數。"""
    stt = SpeechToText()
    first = _FastFakeModel("first")
    second = _FastFakeModel("second")
    stt._model = first
    audio = np.zeros(160, dtype=np.float32)

    errors = []
    stop = threading.Event()
    # 把 GIL 切換間隔縮到最小，讓「檢查後、取鎖前」這個空隙容易被插隊
    original_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)

    def slow_load(model):
        # 真實的 load() 要好幾秒，_model 為 None 的空窗期很長
        def _load():
            time.sleep(0.003)
            stt._model = model
        return _load

    def transcriber():
        while not stop.is_set():
            try:
                stt.transcribe(audio)
            except RuntimeError:
                pass          # 「模型尚未載入完成」是合理結果
            except Exception as e:  # AttributeError 等等＝踩到競態
                errors.append(e)

    def reloader():
        while not stop.is_set():
            stt.load = slow_load(second)
            stt.reload("small")
            stt.load = slow_load(first)
            stt.reload("small")

    threads = [threading.Thread(target=transcriber) for _ in range(3)]
    threads.append(threading.Thread(target=reloader))
    for t in threads:
        t.start()
    time.sleep(1.0)
    stop.set()
    for t in threads:
        t.join(timeout=5.0)
    sys.setswitchinterval(original_interval)

    assert not errors, f"reload 與 transcribe 有競態：{errors[:3]}"

    # reload 之後的呼叫要用到新模型
    stt.load = slow_load(second)
    stt.reload("small")
    before = second.used
    stt.transcribe(audio)
    assert second.used == before + 1
