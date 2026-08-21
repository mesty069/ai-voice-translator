import threading

import numpy as np


class SpeechToText:
    """faster-whisper 中文語音辨識。模型載入很慢，需在背景執行緒呼叫 load()。"""

    def __init__(self, model_size: str = "small", device: str = "cpu",
                 compute_type: str = "int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None
        self._lock = threading.Lock()

    def load(self):
        from faster_whisper import WhisperModel
        with self._lock:
            self._model = WhisperModel(
                self.model_size, device=self.device, compute_type=self.compute_type)

    def reload(self, model_size: str):
        with self._lock:
            self._model = None
        self.model_size = model_size
        self.load()

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def transcribe(self, audio: np.ndarray) -> str:
        if self._model is None:
            raise RuntimeError("語音模型尚未載入完成")
        segments, _info = self._model.transcribe(
            audio,
            language="zh",
            beam_size=5,
            vad_filter=True,
            initial_prompt="以下是繁體中文的句子。",
        )
        return "".join(seg.text for seg in segments).strip()
