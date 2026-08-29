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
