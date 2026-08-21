import threading

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000


class Recorder:
    """按住期間從預設麥克風錄音，16kHz mono float32，可直接餵 faster-whisper。"""

    def __init__(self):
        self._chunks = []
        self._stream = None
        self._lock = threading.Lock()

    def start(self):
        with self._lock:
            if self._stream is not None:
                return
            self._chunks = []
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()

    def _callback(self, indata, frames, time_info, status):
        self._chunks.append(indata.copy())

    def stop(self) -> np.ndarray:
        with self._lock:
            if self._stream is None:
                return np.zeros(0, dtype=np.float32)
            self._stream.stop()
            self._stream.close()
            self._stream = None
            if not self._chunks:
                return np.zeros(0, dtype=np.float32)
            audio = np.concatenate(self._chunks, axis=0).flatten()
            self._chunks = []
            return audio

    @property
    def is_recording(self) -> bool:
        return self._stream is not None
